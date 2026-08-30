"""Route readiness probing (startup /health diagnostics).

Extracted from Gateway: one prober owns the per-route probe snapshot, its
TTL + concurrency semantics, and the readiness projection built from the
snapshot.  The gateway keeps only thin delegations (``_probe_routes``,
``readiness``) so existing CLI/server/test callers are unchanged.

Coupling contract: the prober holds no request-scoped state.  The three
gateway collaborators probe needs — profile resolution, upstream-auth
resolution, and the transport — are resolved lazily through the
``gateway`` back-reference on every call (tests replace ``Gateway.
_probe_routes`` / ``_probe_one_route`` after construction).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from agent_interop.transport.http import PreparedUpstreamRequest
from agent_interop.upstreams.registry import get_codec

logger = logging.getLogger("agent_interop.readiness")

__all__ = ["ReadinessProber"]


class ReadinessProber:
    """Bounded-concurrency, TTL-cached per-route readiness probing."""

    PROBE_CONCURRENCY = 8

    def __init__(self, *, gateway: Any) -> None:
        self._gateway = gateway
        # Keyed by route_id (not append-only) and cleared at the start of
        # every probe pass, so a route that stops failing doesn't leave its
        # earlier failure entries lingering alongside the new success.
        self.results: dict[str, dict[str, Any]] = {}
        # Timestamped so /health/ready and /health can serve a cached
        # snapshot instead of re-probing every configured backend (with a
        # 10s-per-route timeout) on every single health check request —
        # see probe_routes()'s ttl handling.
        self.last_run: float = 0.0
        self._lock = asyncio.Lock()

    async def probe_routes(self, *, force: bool = False, ttl: float = 5.0) -> None:
        """Refresh readiness state for every configured route, from a
        cached snapshot when possible.

        Previously this cleared and fully re-probed every route (each with
        a 10s timeout) on EVERY call — and both /health/ready and /health
        called it on every single request, so a slow or unreachable
        backend meant every health check blocked for up to
        ``10 * len(routes)`` seconds, sequentially. Now: a snapshot younger
        than ``ttl`` seconds is served as-is; only a stale (or ``force``d)
        snapshot triggers a real re-probe, and that re-probe runs all
        routes CONCURRENTLY (bounded by PROBE_CONCURRENCY) instead of one
        at a time. An asyncio.Lock prevents two concurrent callers (e.g.
        two health-check requests arriving while a probe is already in
        flight) from each starting their own redundant full probe pass.
        """
        routes = self._gateway.config.routes
        if not routes:
            return

        now = time.monotonic()
        if not force and self.results and (now - self.last_run) < ttl:
            return

        async with self._lock:
            # Re-check inside the lock: another caller may have just
            # finished refreshing while we were waiting for the lock.
            now = time.monotonic()
            if not force and self.results and (now - self.last_run) < ttl:
                return

            semaphore = asyncio.Semaphore(self.PROBE_CONCURRENCY)

            async def _bounded_probe(route_id: str, route: Any) -> tuple[str, dict[str, Any]]:
                async with semaphore:
                    return route_id, await self.probe_one_route(route_id, route)

            results = await asyncio.gather(
                *(_bounded_probe(rid, r) for rid, r in routes.items())
            )
            self.results = dict(results)
            self.last_run = time.monotonic()

    async def probe_one_route(self, route_id: str, route: Any) -> dict[str, Any]:
        """Probe a single route's backend reachability, auth, model
        presence, and profile resolution. Factored out of probe_routes()
        so routes can be probed concurrently via asyncio.gather."""
        result: dict[str, Any] = {
            "reachable": False,
            "authenticated": False,
            "model_present": None,  # None = backend exposes no inventory to check against
            "codec_ready": False,
            "profile_resolved": False,
            "profile_id": None,
            "profile_source": None,
            "reason": "",
        }

        try:
            codec = get_codec(route.upstream.wire_protocol)
            result["codec_ready"] = True
        except Exception as exc:
            result["reason"] = f"codec resolution failed: {exc}"
            return result

        # Actually resolve a profile (cheap — no I/O) rather than
        # hardcoding profile_resolved=True unconditionally: "resolved"
        # now means "matched a real builtin/explicit profile", not merely
        # "resolution didn't raise" — every model, even one nobody has
        # ever seen, successfully resolves to the conservative fallback
        # tier, so treating that as equivalent to a real match made the
        # field report true for literally every route.
        try:
            resolved_profile = self._gateway._resolve_profile(route)
            result["profile_id"] = getattr(resolved_profile, "profile_id", None)
            result["profile_source"] = getattr(resolved_profile, "source", None)
            result["profile_resolved"] = result["profile_source"] not in (None, "fallback")
        except Exception as exc:
            result["reason"] = f"profile resolution failed: {exc}"

        try:
            base_url = route.upstream.base_url.rstrip("/")
            url = f"{base_url}{codec.probe_endpoint()}"
            # Resolve upstream auth the SAME way real requests do (via the
            # typed UpstreamAuthConfig mechanism) so probing, inference,
            # streaming, and count_tokens all resolve auth identically —
            # including the legacy api_key_env field.
            auth_config = self._gateway._build_upstream_auth_config(route)
            from agent_interop.auth import build_upstream_headers
            headers = build_upstream_headers(
                {}, auth_config, route.upstream.static_headers,
            )
            # Codec-required headers (Content-Type, etc.)
            headers.update(codec.required_headers())

            probe_request = PreparedUpstreamRequest(
                method="GET",
                url=url,
                headers=headers,
                stream=False,
                timeout_seconds=10.0,
            )
            r = await self._gateway.transport.send(probe_request)
            if r.transport_failed:
                # The backend never actually answered — send() returns a
                # synthetic status_code=503 after exhausting retries on a
                # connect/timeout failure. That is NOT "reachable"; a
                # real 503 response from a reachable backend also lands
                # here as a normal status check below, distinguished by
                # this flag rather than by status code alone.
                result["reason"] = "unreachable (connection failed)"
                logger.warning("route '%s' probe failed: unreachable", route_id)
            elif r.status_code == 200:
                result["reachable"] = True
                result["authenticated"] = True
                # Verify the configured model is present when the
                # backend exposes model inventory (Ollama /api/tags
                # style "models", OpenAI-compatible /v1/models "data").
                models: list[str] = []
                try:
                    data = r.json()
                    if "models" in data:
                        models = [m.get("name", "") for m in data["models"]]
                    elif "data" in data:
                        models = [m.get("id", "") for m in data["data"]]
                except Exception:
                    pass
                if models:
                    from agent_interop.model_names import model_names_match
                    # Tag-aware, not exact string match — the same
                    # normalizer the managed launcher uses (model_names.py)
                    # so "qwen3-coder" configured against a backend that
                    # reports "qwen3-coder:latest" isn't reported missing.
                    result["model_present"] = any(
                        model_names_match(route.upstream_model, m) for m in models
                    )
                    if not result["model_present"]:
                        result["reason"] = (
                            f"model '{route.upstream_model}' not found in "
                            f"backend inventory ({len(models)} available)"
                        )
                logger.info(
                    "route '%s' probe OK — %d models", route_id, len(models),
                )
            elif r.status_code == 401:
                result["reachable"] = True
                result["reason"] = "unauthenticated"
                logger.warning("route '%s' probe: unauthenticated", route_id)
            else:
                result["reachable"] = True
                result["reason"] = f"probe returned status {r.status_code}"
                logger.warning("route '%s' probe returned %d", route_id, r.status_code)
        except Exception as exc:
            result["reason"] = str(exc)
            logger.warning("route '%s' probe failed: %s", route_id, exc)

        return result

    def readiness(self) -> dict[str, Any]:
        """Return structured per-route readiness from the most recent probe.

        Distinct from liveness: a process can be alive (accepting
        connections) while every route is unreachable, unauthenticated, or
        missing its configured model. Call ``probe_routes()`` first (or
        rely on startup probing) for this to reflect live backend state —
        a route with no probe on record reports not-ready with
        reason="not probed", never a false "ok".
        """
        config = self._gateway.config
        routes_status: dict[str, Any] = {}
        for route_id in config.routes:
            probed = self.results.get(route_id)
            if probed is None:
                entry: dict[str, Any] = {
                    "reachable": False,
                    "authenticated": False,
                    "model_present": None,
                    "codec_ready": False,
                    "profile_resolved": False,
                    "profile_id": None,
                    "profile_source": None,
                    "reason": "not probed",
                }
            else:
                entry = dict(probed)
            entry["ready"] = bool(
                entry["reachable"]
                and entry["authenticated"]
                and entry["codec_ready"]
                and entry["model_present"] is not False
            )
            routes_status[route_id] = entry

        if config.default_route_id:
            default_entry = routes_status.get(config.default_route_id)
            overall_ready = bool(default_entry and default_entry["ready"])
        else:
            overall_ready = bool(routes_status) and all(
                r["ready"] for r in routes_status.values()
            )

        return {
            "ready": overall_ready,
            "default_route": config.default_route_id,
            "routes": routes_status,
        }
