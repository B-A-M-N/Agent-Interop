"""The core gateway engine — orchestrates protocol translation, model calls,
and response conversion.

The Gateway is the central object that ties together:
1. Client protocol adapters (inbound protocol parsing)
2. Model profile registry (capability-aware resolution)
3. Upstream codecs (protocol-native rendering/decoding)
4. Tool-call parsers (extraction from model output)
5. Transaction service (validation and repair)
6. Response encoding (back to client protocol)

Production path:
    HTTP request → client protocol adapter → canonical request → request context
    → route resolution → history reconciliation → model/backend profile resolution
    → repair policy → invocation plan → upstream codec rendering
    → authenticated transport → upstream codec decoding
    → model-dialect extraction → universal tool transaction
    → canonical response/events → client protocol adapter encoding
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from agent_interop import __version__
from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalError,
    CanonicalEvent,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalUsage,
    RawToolCallCandidate,
    RepairStatus,
    ToolChoiceMode,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairPolicy,
    ToolMode,
)
from agent_interop.errors import InteropErrorCode, classify_http_status
from agent_interop.evidence.store import EvidenceStore
from agent_interop.execution import InteropRequestExecution
from agent_interop.extraction import get_default_registry
from agent_interop.history.reconcile import reconcile_history
from agent_interop.model.registry import ModelProfileRegistry
from agent_interop.model.registry import get_default_registry as get_default_profile_registry
from agent_interop.repair.invocation import StreamExtractionMode, build_invocation_plan
from agent_interop.replay.types import CompatibilityResult
from agent_interop.streaming.coordinator import (
    PendingToolCall,
    StreamCoordinator,
    StreamLimits,
    ToolCallLimitExceeded,
    ToolStreamKey,
)
from agent_interop.transaction import ToolBatchPolicy, ToolTransactionContext, process_tool_batch
from agent_interop.transport.http import (
    PreparedUpstreamRequest,
    UpstreamResponseTooLargeError,
    UpstreamTransport,
)
from agent_interop.transport.ndjson import MalformedNDJSONLine
from agent_interop.types import ServerInfo
from agent_interop.upstreams.codec import (
    DecodedModelResponse,
    DecodedStreamComplete,
    DecodedStreamError,
    DecodedStreamEvent,
    DecodedTextDelta,
    DecodedToolBatchComplete,
    DecodedToolCallComplete,
    DecodedToolFragment,
    DecodedUsageUpdate,
)
from agent_interop.upstreams.registry import get_codec

logger = logging.getLogger("agent_interop.gateway")


# Minimum sample base before a piece of evidence is trusted to gate
# compatibility-pack activation. Below this, a record is too thin to act on.
MIN_EVIDENCE_SAMPLE_COUNT = 5


# ─── ResolvedInvocation (P0.1 contract) ────────────────────────────────────


@dataclass(frozen=True)
class ResolvedInvocation:
    """Request-scoped preparation result (P0.1).

    Created once per request through :meth:`Gateway._prepare_invocation`.
    Carries every resolved component needed by both streaming and
    non-streaming request paths.
    """

    request_context: Any  # RequestContext
    original_request: CanonicalRequest
    reconciled_request: CanonicalRequest
    route: ModelRoute
    backend_metadata: Any  # BackendMetadata
    model_profile: Any  # ResolvedModelProfile
    repair_policy: RepairPolicy
    invocation_plan: Any  # InvocationPlan
    codec: Any  # ModelCodec
    compatibility_key: Any  # CompatibilityKey
    evidence_record: Any | None  # EvidenceRecord when one exists
    repair_budget: Any  # RepairBudget
    execution_record: Any  # InteropRequestExecution


def _canonicalize_json_ish(value: Any) -> str:
    """Canonicalize a JSON-ish value to a stable string representation.

    JSON *strings* are parsed and re-serialized with sorted keys so that
    semantically-identical payloads differing only in key order or whitespace
    collapse to the same output. This is what makes argument-based
    de-duplication and loop detection robust to a model re-encoding the same
    logical arguments differently.

    A string that fails to parse falls back to ``str(value)`` rather than
    raising, since this helper is used in non-fatal bookkeeping paths.
    """
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return str(value)
    return json.dumps(parsed, sort_keys=True, default=str)


class Gateway:
    """Core agent compatibility gateway.

    Accepts route-based InteropServerConfig. Each request resolves a route
    by model name, and the route determines upstream model, wire protocol,
    tool mode, and repair settings.

    Dependency injection via constructor arguments enables tests to prove
    which codec, transport, profile, and evidence records were selected.
    """

    def __init__(
        self,
        config: InteropServerConfig,
        *,
        transport: UpstreamTransport | None = None,
        profile_registry: ModelProfileRegistry | None = None,
        extractor_registry: Any | None = None,
        session_manager: Any | None = None,
        telemetry: Any | None = None,
        evidence_store: EvidenceStore | None = None,
        allow_invalid_config: bool = False,
    ) -> None:
        """Construct a Gateway directly from a config.

        Validates ``config`` with ``validate_config`` and raises
        ``ValueError`` on any issue — this is the lowest construction
        boundary; CLI-level validation (``deploy``/``check``) and
        ``server.app.create_app`` both happen ABOVE this, but a caller that
        constructs ``Gateway(config)`` directly (bypassing both) previously
        reached startup with no validation at all. ``allow_invalid_config``
        exists strictly for tests that intentionally probe invalid-config
        behavior; production call sites must never pass it.
        """
        if not allow_invalid_config:
            from agent_interop.config import validate_config
            issues = validate_config(config)
            if issues:
                raise ValueError(
                    "Invalid InteropServerConfig:\n" + "\n".join(f"  - {i}" for i in issues)
                )
        self.config = config
        self._transport = transport
        self._extractor_registry = extractor_registry or get_default_registry()
        self._profile_registry = profile_registry or get_default_profile_registry()
        self._session_manager = session_manager
        self._telemetry = telemetry
        # Opt-in only: defaults to None (disabled). Do NOT default to
        # get_default_store() here — that would make every unconfigured
        # Gateway silently read/write a real on-disk store (~/.local/state/...),
        # polluting tests and any deployment that doesn't explicitly opt in.
        self._evidence_store = evidence_store
        # Keyed by route_id (not append-only) and cleared at the start of
        # every probe pass, so a route that stops failing doesn't leave its
        # earlier failure entries lingering alongside the new success.
        self._probe_results: dict[str, dict[str, Any]] = {}
        # Timestamped so /health/ready and /health can serve a cached
        # snapshot instead of re-probing every configured backend (with a
        # 10s-per-route timeout) on every single health check request —
        # see _probe_routes()'s ttl handling.
        self._probe_last_run: float = 0.0
        self._probe_lock = asyncio.Lock()

    @property
    def transport(self) -> UpstreamTransport:
        """Lazily build a default ``UpstreamTransport`` from config when none
        was injected. Transport settings (P0.6) are mapped from the
        ``InteropServerConfig`` fields."""
        if self._transport is None:
            cfg = self.config
            max_conn = getattr(cfg, "max_connections", 100) or 100
            max_keepalive = getattr(cfg, "max_keepalive_connections", 20) or 20
            max_retries = getattr(cfg, "max_retries", 2) or 2
            read_timeout = getattr(cfg, "read_timeout", cfg.backend_timeout) or cfg.backend_timeout or 120.0
            connect_timeout = getattr(cfg, "connect_timeout", None)
            write_timeout = getattr(cfg, "write_timeout", None)
            pool_timeout = getattr(cfg, "pool_timeout", None)
            max_stream_frame = getattr(cfg, "max_stream_frame_bytes", 1 * 1024 * 1024)
            max_response = getattr(cfg, "max_response_bytes", 256 * 1024 * 1024)
            retryable_statuses = getattr(cfg, "retryable_statuses", (429, 500, 502, 503, 504))
            tls_verify = getattr(cfg, "tls_verify", True)

            self._transport = UpstreamTransport(
                max_connections=max_conn,
                max_keepalive=max_keepalive,
                max_retries=max_retries,
                retryable_statuses=retryable_statuses,
                timeout_seconds=read_timeout,
                connect_timeout=connect_timeout,
                write_timeout=write_timeout,
                pool_timeout=pool_timeout,
                max_sse_data_bytes=max_stream_frame,
                max_ndjson_frame_bytes=max_stream_frame,
                max_total_stream_bytes=max_response,
                max_response_bytes=max_response,
                tls_verify=tls_verify,
            )
        return self._transport

    async def close(self) -> None:
        if self._transport is not None:
            await self._transport.close()
            self._transport = None

    # ─── Startup / Probe ──────────────────────────────────────────────────

    async def startup(self) -> None:
        """Initialize the gateway.

        Validates config, then probes every configured route when
        ``probe_on_startup`` is True.
        """
        from agent_interop.config import validate_config

        issues = validate_config(self.config)
        if issues:
            raise RuntimeError(
                f"Invalid gateway configuration: {'; '.join(issues)}"
            )

        if not self.config.routes:
            logger.warning("interop starting — no routes configured")
            return

        logger.info(
            "interop starting — routes=%d default=%s",
            len(self.config.routes),
            self.config.default_route_id,
        )

        if self.config.probe_on_startup:
            await self._probe_routes()

    _PROBE_CONCURRENCY = 8

    async def _probe_routes(self, *, force: bool = False, ttl: float = 5.0) -> None:
        """Refresh readiness state for every configured route, from a
        cached snapshot when possible.

        Previously this cleared and fully re-probed every route (each with
        a 10s timeout) on EVERY call — and both /health/ready and /health
        called it on every single request, so a slow or unreachable
        backend meant every health check blocked for up to
        ``10 * len(routes)`` seconds, sequentially. Now: a snapshot younger
        than ``ttl`` seconds is served as-is; only a stale (or ``force``d)
        snapshot triggers a real re-probe, and that re-probe runs all
        routes CONCURRENTLY (bounded by _PROBE_CONCURRENCY) instead of one
        at a time. An asyncio.Lock prevents two concurrent callers (e.g.
        two health-check requests arriving while a probe is already in
        flight) from each starting their own redundant full probe pass.
        """
        if not self.config.routes:
            return

        now = time.monotonic()
        if not force and self._probe_results and (now - self._probe_last_run) < ttl:
            return

        async with self._probe_lock:
            # Re-check inside the lock: another caller may have just
            # finished refreshing while we were waiting for the lock.
            now = time.monotonic()
            if not force and self._probe_results and (now - self._probe_last_run) < ttl:
                return

            semaphore = asyncio.Semaphore(self._PROBE_CONCURRENCY)

            async def _bounded_probe(route_id: str, route: ModelRoute) -> tuple[str, dict[str, Any]]:
                async with semaphore:
                    return route_id, await self._probe_one_route(route_id, route)

            results = await asyncio.gather(
                *(_bounded_probe(rid, r) for rid, r in self.config.routes.items())
            )
            self._probe_results = dict(results)
            self._probe_last_run = time.monotonic()

    async def _probe_one_route(self, route_id: str, route: ModelRoute) -> dict[str, Any]:
        """Probe a single route's backend reachability, auth, model
        presence, and profile resolution. Factored out of _probe_routes()
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
            resolved_profile = self._resolve_profile(route)
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
            auth_config = self._build_upstream_auth_config(route)
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
            r = await self.transport.send(probe_request)
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
        missing its configured model. Call ``_probe_routes()`` first (or
        rely on startup probing) for this to reflect live backend state —
        a route with no probe on record reports not-ready with
        reason="not probed", never a false "ok".
        """
        routes_status: dict[str, Any] = {}
        for route_id in self.config.routes:
            probed = self._probe_results.get(route_id)
            if probed is None:
                entry = {
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

        if self.config.default_route_id:
            default_entry = routes_status.get(self.config.default_route_id)
            overall_ready = bool(default_entry and default_entry["ready"])
        else:
            overall_ready = bool(routes_status) and all(
                r["ready"] for r in routes_status.values()
            )

        return {
            "ready": overall_ready,
            "default_route": self.config.default_route_id,
            "routes": routes_status,
        }

    # ─── Server info ──────────────────────────────────────────────────────

    def server_info(self) -> ServerInfo:
        """Return aggregate service information and route summaries."""
        route_summaries = []
        for route_id, route in self.config.routes.items():
            route_summaries.append({
                "route_id": route_id,
                "upstream_model": route.upstream_model,
                "upstream_kind": route.upstream.kind.value,
                "wire_protocol": route.upstream.wire_protocol.value,
                "tool_mode": route.tool_mode.value,
                "profile": route.profile,
                "default": route_id == self.config.default_route_id,
            })

        return ServerInfo(
            version=__version__,
            model=",".join(
                r.upstream_model for r in self.config.routes.values()
            ),
            routes=route_summaries,
        )

    def get_route_for_model(self, model_name: str) -> ModelRoute | None:
        """Resolve a model name to a route."""
        return self.config.get_route_for_model(model_name)

    def _resolve_route(self, canonical: CanonicalRequest) -> ModelRoute:
        """Resolve the route for a request, raising on unknown model."""
        requested = canonical.model.requested_name
        route = self.config.get_route_for_model(requested)
        if route is None:
            if requested:
                raise ValueError(
                    f"Unknown model: '{requested}'. "
                    f"Available: {self.config.all_model_aliases()}",
                )
            raise ValueError(
                "No model specified and no default route configured",
            )
        return route

    def _get_session_state(self, context: Any) -> Any:
        """Resolve session state for loop detection — the single touch
        point for a request (increments request_count exactly once, via
        SessionManager.begin_request). Every other lookup of the same
        session within this request must use ``.get()`` instead, or
        request_count ends up counting internal lookups rather than
        requests.

        Returns None (no session tracking) when the client supplied no
        session ID — a synthetic per-request ID would create one
        one-shot session-store entry per stateless request, both
        polluting the bounded store and being useless for loop detection
        (which needs repeated requests in the SAME session to say anything).
        """
        if self._session_manager is None:
            return None
        session_id = getattr(context, 'session_id', None) if context else None
        if not session_id:
            return None
        route_id = getattr(context, 'route_id', '') if context else ''
        return self._session_manager.begin_request(session_id, route_id=route_id)

    def _record_repairs_to_session(
        self,
        batch_decision: Any,
        context: Any,
    ) -> None:
        """Record repair outcomes from a batch decision into session state.

        This feeds the session manager's loop detection with data about
        which tools were repaired, rejected, or succeeded. Without this
        call, the loop detector is starved of input data.

        Does NOT touch the session (no begin_request call) — by the time
        this runs, `_get_session_state` has already been called once for
        this request and created/touched the session if a session_id was
        present. Calling begin_request again here double-counted
        request_count for every batch of repairs recorded in one request.
        """
        if self._session_manager is None:
            return
        session_id = getattr(context, 'session_id', None) if context else None
        if not session_id:
            return
        route_id = getattr(context, 'route_id', '') if context else ''

        for decision in batch_decision.decisions:
            outcome = decision.outcome
            tool_name = outcome.call_name or decision.candidate.name
            # Digest the raw candidate arguments so the loop detector can
            # distinguish repairs of the same tool with DIFFERENT arguments
            # (legitimate) from repeated identical-argument repairs (a loop).
            # Prefer the raw candidate arguments: a rejected call may not have
            # a populated outcome.accepted, but the candidate always carries
            # the original arguments the model produced.
            argument_digest = self._compute_argument_digest(
                decision.candidate.raw_arguments,
            )
            if outcome.was_repaired:
                self._session_manager.record_repair(
                    session_id,
                    route_id,
                    tool_name,
                    len(outcome.initial_issues),
                    "repaired",
                    argument_digest=argument_digest,
                )
            elif not outcome.is_accepted:
                self._session_manager.record_repair(
                    session_id,
                    route_id,
                    tool_name,
                    len(outcome.initial_issues),
                    "rejected",
                    argument_digest=argument_digest,
                )

    def _record_tool_decisions(
        self,
        batch_decision: Any,
        execution: InteropRequestExecution,
    ) -> None:
        """Record per-call repair outcomes onto the shared execution record.

        The in-memory ``execution.tool_decisions`` record is always populated
        so downstream consumers (``finalize_response`` outcome classification,
        summary logging, replay/evidence) can observe what happened on this
        request regardless of whether an evidence store is configured.

        Persisting to the evidence store itself remains opt-in: that write-back
        happens in :meth:`_record_evidence_observation`, which is separately
        gated on ``self._evidence_store``.
        """
        for decision in batch_decision.decisions:
            execution.record_tool_decision(
                tool_name=decision.outcome.call_name or decision.candidate.name or "",
                candidate_id=decision.candidate.id or "",
                outcome=decision.outcome,
                accepted=decision.is_accepted,
            )

    def _record_evidence_observation(
        self,
        invocation: ResolvedInvocation,
        execution: InteropRequestExecution,
    ) -> None:
        """Persist a single-request compatibility observation.

        Read-modify-write: merges this request's outcome into any existing
        record for the exact compatibility tuple. Per-tool-call counters are
        accumulated exactly (each decision contributes one unit, so a request
        with 10 decisions moves the aggregate 10x as far as one with 1) and
        the rate fields are re-derived from the merged counters — rates are
        never averaged in per-request space. Verification state
        (``manually_verified`` / ``last_verified_at``) and revocation state
        are never touched by live traffic, so a live request can never
        auto-verify, refresh the certification clock, or un-revoke a record.
        Pre-v4 records that stored rates without counters are seeded from
        their stored rates before merging, so historical data is preserved.
        ``task_completion_rate`` is not derivable from a single live
        request, so it is left at the existing value (or 0.0 for a
        brand-new record) rather than being guessed.

        A persistence failure must never break the client request, so the
        whole operation is wrapped to log and swallow.
        """
        store = self._evidence_store
        if store is None:
            return
        try:
            self._record_evidence_observation_inner(invocation, execution, store)
        except Exception:
            logger.warning("failed to record compatibility evidence", exc_info=True)

    def _record_evidence_observation_inner(
        self,
        invocation: ResolvedInvocation,
        execution: InteropRequestExecution,
        store: EvidenceStore,
    ) -> None:
        key = invocation.compatibility_key
        if key is None:
            return

        decisions = execution.tool_decisions
        n = len(decisions)

        # Per-request OBSERVATION captured as COUNTERS (one unit per tool-call
        # decision), not as rates. This is what lets the merge weight each
        # call equally: a request with 10 decisions moves the aggregate 10x
        # as far as a request with 1. Rates are re-derived from the merged
        # counters below, never averaged in per-request space.
        #
        # task_completion_rate is deliberately excluded — it cannot be observed
        # from a single live request, so folding a guessed 0.0 into the
        # aggregate would only degrade an existing value.
        if n == 0:
            # Tools were offered but the model produced no tool-call decision
            # (e.g. it replied with text). No candidate calls to count — only
            # the no-selection signal is recorded.
            observation: dict[str, int | bool] = {
                "candidate_count": 0,
                "valid_unchanged_count": 0,
                "repaired_count": 0,
                "regenerated_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "no_selection": True,
            }
        else:
            valid_unchanged = sum(
                1 for d in decisions
                if d.outcome_status == RepairStatus.VALID_UNCHANGED.value
            )
            repaired = sum(
                1 for d in decisions
                if d.outcome_status == RepairStatus.REPAIRED.value
            )
            regenerated = sum(
                1 for d in decisions
                if d.outcome_status == RepairStatus.REGENERATED.value
            )
            accepted = sum(1 for d in decisions if d.accepted)
            observation = {
                "candidate_count": n,
                "valid_unchanged_count": valid_unchanged,
                "repaired_count": repaired,
                "regenerated_count": regenerated,
                "accepted_count": accepted,
                "rejected_count": (n - accepted),
                "no_selection": False,
            }

        existing = store.get_result(key)
        now = datetime.now(UTC).isoformat()

        if existing is not None:
            new_n = existing.sample_count + 1

            # ── Migration / seeding for pre-v4 records ─────────────────────
            # A record written before this fix (or seeded directly by a
            # pre-existing test) stores rates but has candidate_count == 0.
            # Seed the counters from the stored rates BEFORE adding this
            # request's contribution, so historical data isn't silently
            # discarded. The sample_count guard ensures we only seed when
            # there is something to seed from.
            if existing.candidate_count == 0 and existing.no_selection_request_count == 0 and existing.sample_count > 0:
                seed_n = existing.sample_count
                seed_candidate_count = seed_n
                seed_valid_unchanged_count = round(
                    existing.valid_call_rate_before_repair * seed_n
                )
                seed_repaired_count = round(
                    existing.deterministic_repair_rate * seed_n
                )
                seed_regenerated_count = round(
                    existing.regeneration_rate * seed_n
                )
                seed_accepted_count = round(
                    existing.valid_call_rate_after_repair * seed_n
                )
                seed_rejected_count = round(existing.rejection_rate * seed_n)
                seed_no_selection = round(
                    (1.0 - existing.tool_selection_rate) * seed_n
                )
            else:
                # Genuine v4 record: pass existing counters through unchanged.
                seed_candidate_count = existing.candidate_count
                seed_valid_unchanged_count = existing.valid_unchanged_count
                seed_repaired_count = existing.repaired_count
                seed_regenerated_count = existing.regenerated_count
                seed_accepted_count = existing.accepted_count
                seed_rejected_count = existing.rejected_count
                seed_no_selection = existing.no_selection_request_count

            # Accumulate this request's counters on top of the seed.
            c_candidate = seed_candidate_count + observation["candidate_count"]
            c_valid_unchanged = (
                seed_valid_unchanged_count + observation["valid_unchanged_count"]
            )
            c_repaired = seed_repaired_count + observation["repaired_count"]
            c_regenerated = (
                seed_regenerated_count + observation["regenerated_count"]
            )
            c_accepted = seed_accepted_count + observation["accepted_count"]
            c_rejected = seed_rejected_count + observation["rejected_count"]
            c_no_selection = seed_no_selection + (
                1 if observation["no_selection"] else 0
            )

            # Derive rates fresh from the merged counters. Guard
            # divide-by-zero: a record with zero candidates has 0.0 rates.
            # Rates are passed as explicit kwargs (not a ``**dict`` spread)
            # so mypy can verify each field's type — a ``dict[str, float]``
            # spread is not assignable to the dataclass's per-field types.
            result = replace(
                existing,
                sample_count=new_n,
                last_observed_at=now,
                candidate_count=c_candidate,
                valid_unchanged_count=c_valid_unchanged,
                repaired_count=c_repaired,
                regenerated_count=c_regenerated,
                accepted_count=c_accepted,
                rejected_count=c_rejected,
                no_selection_request_count=c_no_selection,
                valid_call_rate_before_repair=(
                    c_valid_unchanged / c_candidate if c_candidate else 0.0
                ),
                valid_call_rate_after_repair=(
                    c_accepted / c_candidate if c_candidate else 0.0
                ),
                deterministic_repair_rate=(
                    c_repaired / c_candidate if c_candidate else 0.0
                ),
                regeneration_rate=(
                    c_regenerated / c_candidate if c_candidate else 0.0
                ),
                rejection_rate=(
                    c_rejected / c_candidate if c_candidate else 0.0
                ),
                tool_selection_rate=(
                    (new_n - c_no_selection) / new_n if new_n else 0.0
                ),
            )
            # replace() only overrides the fields passed in, so
            # manually_verified / revoked / revocation_reason / created_at /
            # tested_at / last_verified_at are preserved from the existing
            # record unchanged. CRUCIALLY we do NOT set tested_at or
            # last_verified_at here — live traffic must never refresh the
            # certification clock. last_observed_at tracks only that we saw
            # this tuple, for informational purposes.
        else:
            # Brand-new record from live traffic alone. tested_at /
            # last_verified_at are intentionally left at their defaults ("")
            # — a live-only record was never certified. The trust gate
            # (gateway.py _prepare_invocation) already requires
            # manually_verified=True before a record is trusted, so an
            # uncertified live-only record can never be activated.
            c_candidate = observation["candidate_count"]
            result = CompatibilityResult(
                sample_count=1,
                created_at=now,
                last_observed_at=now,
                manually_verified=False,
                revoked=False,
                candidate_count=c_candidate,
                valid_unchanged_count=observation["valid_unchanged_count"],
                repaired_count=observation["repaired_count"],
                regenerated_count=observation["regenerated_count"],
                accepted_count=observation["accepted_count"],
                rejected_count=observation["rejected_count"],
                no_selection_request_count=(
                    1 if observation["no_selection"] else 0
                ),
                valid_call_rate_before_repair=(
                    observation["valid_unchanged_count"] / c_candidate
                    if c_candidate else 0.0
                ),
                valid_call_rate_after_repair=(
                    observation["accepted_count"] / c_candidate
                    if c_candidate else 0.0
                ),
                deterministic_repair_rate=(
                    observation["repaired_count"] / c_candidate
                    if c_candidate else 0.0
                ),
                regeneration_rate=(
                    observation["regenerated_count"] / c_candidate
                    if c_candidate else 0.0
                ),
                rejection_rate=(
                    observation["rejected_count"] / c_candidate
                    if c_candidate else 0.0
                ),
                tool_selection_rate=(
                    0.0 if observation["no_selection"] else 1.0
                ),
            )

        store.store_result(key, result)

    # ─── Request preparation (P0.1 contract) ───────────────────────────

    def _resolve_invocation_plan_and_key(
        self,
        route: ModelRoute,
        request: CanonicalRequest,
        context: Any,
        streaming: bool,
    ) -> tuple[Any, Any, RepairPolicy, Any, Any]:
        """Resolve backend metadata, model profile, repair policy, invocation
        plan, and the authoritative compatibility key for an already-routed
        request — the exact computation ``_prepare_invocation`` performs,
        factored out so other callers (certify/conformance tooling) can
        obtain a key that will byte-for-byte match what live traffic
        produces for the same route+request+context, instead of hand-rolling
        a sparse key that can never be found by the live gate.

        The ``route`` must already be resolved; ``request`` should be the
        history-reconciled request that will actually be sent upstream.

        Returns:
            (backend_metadata, model_profile, repair_policy, invocation_plan,
            compat_key)
        """
        from agent_interop.evidence.key import CompatibilityKeyInputs, build_compatibility_key

        # Resolve the codec up front — it only depends on the route (already in
        # hand) and is needed for pre-upstream tool-contract validation below.
        codec = get_codec(route.upstream.wire_protocol)

        # Resolve backend metadata and model profile. These only depend on the
        # route (already in hand), so resolve them BEFORE tool-contract validation
        # and plan construction — both need the profile and the resolved mode.
        backend_metadata = self._get_backend_metadata(route)
        model_profile = self._resolve_profile(route, backend_metadata)

        # A profile that declares streaming_supported=False is a real,
        # executable constraint (unlike declared_tokens/safe_tokens, which
        # remain informational-only pending real token counting) — silently
        # downgrading to non-streaming or silently proceeding would give the
        # client output framed as a stream when the model can't actually
        # produce one incrementally. Reject before contacting the backend,
        # the same way an invalid tool contract is rejected above.
        if (
            streaming
            and model_profile is not None
            and not getattr(model_profile, "streaming_supported", True)
        ):
            raise ValueError(
                f"Model profile '{getattr(model_profile, 'profile_id', '')}' does not "
                "support streaming, but the request asked for stream=true"
            )

        # Resolve the effective tool mode ONCE, from route config × profile × codec,
        # BEFORE anything downstream computes plan fields. This is the single source
        # of truth for tool-mode negotiation — so the plan is built exactly once, already
        # correctly negotiated, and never needs post-hoc mutation.
        from agent_interop.config import ToolMode
        from agent_interop.repair.invocation import resolve_effective_tool_mode
        codec_caps = codec.capabilities()
        effective_tool_mode = resolve_effective_tool_mode(route.tool_mode, model_profile, codec_caps)

        # Validate tool contract (pre-upstream). This can raise ValueError for an
        # invalid contract, mirroring exactly what _prepare_invocation does —
        # callers that cannot compute a key for this request (e.g. a conformance
        # test whose tools violate the backend contract) must catch locally.
        from agent_interop.request_validation import validate_tool_contract
        backend_constraints = codec.backend_constraints()
        if effective_tool_mode != ToolMode.NATIVE:
            # max_tools models a native tool-array limit. PROMPTED/TEXTUAL/DISABLED
            # never send a native tools array (tools are embedded as text or absent),
            # so that limit does not apply — enforcing it anyway produces false
            # rejections of valid requests that would never hit the native array cap.
            from dataclasses import replace as _replace
            backend_constraints = _replace(backend_constraints, max_tools=0)
        is_valid, validation_issues = validate_tool_contract(
            tools=request.tools,
            tool_choice=request.tool_choice,
            tool_mode=route.tool_mode,
            backend_constraints=backend_constraints,
        )
        if not is_valid:
            issue_messages = "; ".join(i.message for i in validation_issues)
            raise ValueError(f"Invalid tool contract: {issue_messages}")

        # Construct repair policy with confidence gating.
        repair_policy = RepairPolicy.from_config(route.repair)
        profile_confidence = getattr(model_profile, 'source_confidence', 0.5) if model_profile else 0.5
        repair_policy = self._apply_confidence_gate(repair_policy, profile_confidence)

        # Build invocation plan exactly once. The mode is already fully resolved
        # (route × profile × codec), so the plan is correct as built — no post-hoc
        # codec validation or mutation needed.
        plan = build_invocation_plan(
            tools=request.tools,
            tool_choice=request.tool_choice,
            route_mode=effective_tool_mode,
            model_profile=model_profile,
            repair_policy=repair_policy,
            codec_capabilities=codec_caps,
        )

        # Compute compatibility key.
        tool_schema_fingerprint = self._compute_tool_schema_fingerprint(request.tools)
        compat_key = build_compatibility_key(CompatibilityKeyInputs(
            # Pass the context OBJECT, not context.client_id — the builder reads
            # .client_id/.client_version/.client_protocol off it. Passing the
            # string client_id crashes with AttributeError whenever it is non-empty.
            request_context=context,
            route=route,
            request=request,
            backend_metadata=backend_metadata,
            model_profile=model_profile,
            invocation_plan=plan,
            tool_schema_fingerprint=tool_schema_fingerprint,
            streaming=streaming,
        ))

        return backend_metadata, model_profile, repair_policy, plan, compat_key

    def _prepare_invocation(
        self,
        request: CanonicalRequest,
        context: Any,
        streaming: bool,
        execution: InteropRequestExecution,
    ) -> ResolvedInvocation:
        """Prepare a resolved invocation for one request (P0.1).

        Creates the request-scoped structure containing every resolved
        component needed by both streaming and non-streaming paths.

        Preparation order:
            1. Resolve route from request.model.requested_name
            2. Reject explicit unknown model (no silent fallback)
            3. Use default_route_id only when model omitted
            4. Reconcile conversation history
            5. Reject unsafe history before contacting backend
            6. Resolve backend metadata and model profile
            7. Construct repair policy
            8. Build invocation plan exactly once
        """
        from agent_interop.repair.pipeline import RepairBudget

        # 1-3. Resolve the route
        route = self._resolve_route(request)
        # Attach to the caller-supplied execution record as soon as it is
        # available so it is populated even on the early-exit branches below.
        execution.route = route

        # 3.5 Check for session loop (before expensive preparation)
        session_state = self._get_session_state(context)
        if session_state is not None and session_state.flagged:
            execution.finalize_error(CanonicalError(
                code="GENERATION_LOOP_DETECTED",
                message="Session flagged for generation loop — refusing new requests",
            ))
            raise ValueError(
                f"Session '{getattr(context, 'session_id', '?')}' flagged for generation loop"
            )

        # 4. Reconcile conversation history
        history_result = reconcile_history(
            request.messages,
            session_id=getattr(context, "session_id", "") or "",
            request_id=getattr(context, "request_id", "") or request.request_id or "",
        )
        # Attach history diagnostics to the shared execution record as soon as
        # they are computed so they survive the unsafe-history early exit below.
        execution.history_diagnostics.extend(history_result.diagnostics)

        # 5. Reject unsafe history
        if not history_result.is_safe:
            from dataclasses import replace
            reconciled = replace(request, messages=history_result.messages)
            unsafe_backend_metadata = self._get_backend_metadata(route)
            return ResolvedInvocation(
                request_context=context,
                original_request=request,
                reconciled_request=reconciled,
                route=route,
                backend_metadata=unsafe_backend_metadata,
                model_profile=self._resolve_profile(route, unsafe_backend_metadata),
                repair_policy=RepairPolicy.from_config(route.repair),
                invocation_plan=None,
                codec=None,
                compatibility_key=None,
                evidence_record=None,
                repair_budget=None,
                execution_record=execution,
            )

        from dataclasses import replace
        reconciled_request = replace(request, messages=history_result.messages)

        # Resolve the codec up front — it only depends on the route (already in
        # hand) and is needed for the ResolvedInvocation returned below.
        codec = get_codec(route.upstream.wire_protocol)

        # Resolve backend metadata, model profile, repair policy, invocation
        # plan, and the authoritative compatibility key — the exact computation
        # live traffic performs, factored into _resolve_invocation_plan_and_key
        # so certify/conformance tooling can obtain a key that byte-for-byte
        # matches what the live gate looks up. This includes the tool-contract
        # validation step that can raise ValueError (propagates unchanged).
        backend_metadata, model_profile, repair_policy, plan, compat_key = (
            self._resolve_invocation_plan_and_key(
                route, reconciled_request, context, streaming,
            )
        )
        execution.invocation_plan = plan
        execution.compatibility_key = compat_key

        # Look up verified evidence for this exact tuple (opt-in only). A record
        # only qualifies when it is present, manually verified, not revoked, not
        # stale, and backed by a sufficient sample base. Absent that, packs must
        # NOT be activated on a merely well-formed key.
        evidence_record = None
        if self._evidence_store is not None:
            candidate = self._evidence_store.get_result(compat_key)
            if (
                candidate is not None
                and candidate.manually_verified
                and not candidate.revoked
                and not self._evidence_store.is_stale(compat_key)
                and candidate.sample_count >= MIN_EVIDENCE_SAMPLE_COUNT
            ):
                evidence_record = candidate

        # Create request-scoped repair budget
        repair_budget = RepairBudget()
        execution.repair_budget = repair_budget

        return ResolvedInvocation(
            request_context=context,
            original_request=request,
            reconciled_request=reconciled_request,
            route=route,
            backend_metadata=backend_metadata,
            model_profile=model_profile,
            repair_policy=repair_policy,
            invocation_plan=plan,
            codec=codec,
            compatibility_key=compat_key,
            evidence_record=evidence_record,
            repair_budget=repair_budget,
            execution_record=execution,
        )

    # ─── Non-streaming request ────────────────────────────────────────────

    async def handle_request(
        self,
        canonical: CanonicalRequest,
        context: Any,
    ) -> CanonicalResponse:
        """Handle a non-streaming request end-to-end.

        Production path:
            _prepare_invocation → codec render → transport → decode
            → extraction → transaction → canonical assembly
        """
        exec_record = InteropRequestExecution(context=context)
        try:
            # Prepare the resolved invocation (passes in the shared record so
            # diagnostics/route/plan all land on the object that gets finalized)
            invocation = self._prepare_invocation(
                canonical, context, streaming=False, execution=exec_record,
            )
            result = await self._handle_request_send(invocation, exec_record)
            # Live evidence write-back: only on the success path, only when tools
            # were offered, only when an evidence store was injected. Backend
            # errors carry no tool-calling signal, so they are skipped.
            if (
                self._evidence_store is not None
                and canonical.tools
                and result.error is None
            ):
                self._record_evidence_observation(invocation, exec_record)
            # finalize_response() logs the summary itself (see execution.py) —
            # relying on a caller to do it separately after the fact is what
            # let the streaming path silently skip logging entirely (the ASGI
            # consumer never resumes the generator far enough to reach a
            # trailing call).
            exec_record.finalize_response(result)
            return result
        except asyncio.CancelledError:
            # Mid-request cancellation: finalize the record as CANCELLED so it
            # is not left permanently ACTIVE, then re-raise. Do NOT swallow —
            # the caller must see the cancellation.
            exec_record.finalize_cancelled()
            raise
        except Exception as exc:
            exc_err = CanonicalError(code="HANDLE_ERROR", message=str(exc)) if not isinstance(exc, CanonicalError) else exc
            exec_record.finalize_error(exc_err)
            raise

    def _build_transaction_context(
        self,
        invocation: ResolvedInvocation,
        canonical: CanonicalRequest,
    ) -> ToolTransactionContext:
        """Build the ``ToolTransactionContext`` for a tool-call batch.

        Shared by the streaming and non-streaming paths so both use the exact
        same repair policy (confidence-gated), the request-scoped repair budget
        (shared across every batch in the request), the correct request_id, and
        the compatibility key. Duplicating this logic in each path caused the
        streaming path to skip the confidence gate, reset the budget per batch,
        and drop telemetry/compatibility_key.
        """
        request_context = invocation.request_context
        return ToolTransactionContext(
            request_id=request_context.request_id if request_context else "",
            session_id=getattr(request_context, 'session_id', '') if request_context else '',
            tool_choice=canonical.tool_choice,
            repair_policy=invocation.repair_policy,
            client_id=request_context.client_id if request_context else None,
            telemetry=self._telemetry,
            budget=invocation.repair_budget,
            compatibility_key=invocation.compatibility_key,
            # evidence_record is only ever non-null when it passed all the
            # verification gates in _prepare_invocation, so its presence is the
            # verified signal. A merely well-formed key is not sufficient.
            compatibility_verified=invocation.evidence_record is not None,
        )

    async def _handle_request_send(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
    ) -> CanonicalResponse:
        """Send a prepared invocation to the upstream and decode the response.

        Steps:
            1. Handle unsafe history (return error response)
            2. Render the reconciled request through the codec
            3. Apply the invocation plan (tool mode adjustments)
            4. Build typed PreparedUpstreamRequest
            5. Send with bounded retries
            6. Decode the upstream response
            7. Extract tool calls from model-dialect output
            8. Run the tool transaction pipeline
            9. Assemble the canonical response
        """
        # Unsafe history — return error response
        if invocation.invocation_plan is None or invocation.codec is None:
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=invocation.reconciled_request.model.requested_name,
                    resolved_name=invocation.route.upstream_model,
                ),
                error=CanonicalError(
                    code=InteropErrorCode.HISTORY_UNSAFE,
                    message="History reconciliation detected unsafe history",
                ),
            )

        route = invocation.route
        plan = invocation.invocation_plan
        codec = invocation.codec
        canonical = invocation.reconciled_request

        choice_conflict = self._disabled_tool_choice_conflict(plan)
        if choice_conflict is not None:
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=choice_conflict,
            )

        # 1. Render through codec
        request_local = copy.deepcopy(canonical)
        rendered = codec.render_request(request_local, route.upstream_model, stream=False)

        # 2. Apply invocation plan
        rendered = self._apply_invocation_plan_to_request(rendered, plan, route)

        # 3. Build typed upstream request
        upstream_request = PreparedUpstreamRequest(
            method="POST",
            url=f"{route.upstream.base_url}{codec.endpoint_path()}",
            headers=self._build_upstream_headers(
                route,
                client_headers=dict(invocation.request_context.forwardable_transport_headers),
                codec_headers=codec.required_headers(),
            ),
            body=rendered,
            stream=False,
            timeout_seconds=route.upstream.timeout_seconds,
        )

        # 4. Send with bounded retries
        try:
            response = await self.transport.send(upstream_request)
        except UpstreamResponseTooLargeError as exc:
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.INVALID_OUTPUT,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=CanonicalError(
                    code=InteropErrorCode.STREAM_SIZE_LIMIT,
                    message=str(exc),
                ),
            )

        if response.is_error():
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=CanonicalError(
                    code=classify_http_status(response.status_code),
                    message=f"Upstream returned {response.status_code}: {response.body[:500].decode('utf-8', errors='replace')}",
                ),
            )

        # 5. Decode upstream response
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=CanonicalError(
                    code="INVALID_UPSTREAM_OUTPUT",
                    message=f"Upstream returned non-JSON response (status={response.status_code})",
                ),
            )

        decoded = codec.decode_response(data)

        # 6. Extract tool calls from model-dialect output
        candidates = self._extract_tool_candidates(decoded, invocation)

        # 7. Run tool transaction pipeline
        from agent_interop.transaction import ToolBatchPolicy, process_tool_batch

        transaction_context = self._build_transaction_context(invocation, canonical)
        batch_decision = await process_tool_batch(
            candidates,
            canonical.tools,
            context=transaction_context,
            policy=ToolBatchPolicy(invocation.repair_policy.batch_policy),
        )

        # 7.5 Record repairs into session state for loop detection
        self._record_repairs_to_session(batch_decision, invocation.request_context)

        # 7.6 Record per-call decisions onto the shared execution record. The
        # in-memory record is always populated (so finalize_response's outcome
        # classification sees the decisions); evidence-store write-back is a
        # separate, opt-in step in _record_evidence_observation.
        self._record_tool_decisions(batch_decision, exec_record)

        # 8. Assemble canonical response
        return self._assemble_response(decoded, batch_decision, canonical, route)

    def _disabled_tool_choice_conflict(self, plan: Any) -> CanonicalError | None:
        """A DISABLED route combined with a required/named tool choice is a
        contradiction: the client demands a tool call the route guarantees
        will never be produced. Reject before contacting the backend rather
        than silently ignoring the choice."""
        if plan is None or plan.effective_tool_mode != ToolMode.DISABLED:
            return None
        if plan.original_tool_choice is None:
            return None
        mode = plan.original_tool_choice.mode
        if mode == ToolChoiceMode.REQUIRED or mode == ToolChoiceMode.NAMED:
            return CanonicalError(
                code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                message=(
                    f"tool_choice={mode.value!r} requires a tool call, but this route's "
                    "tool_mode is disabled"
                ),
            )
        return None

    def _extract_tool_candidates(
        self,
        decoded: DecodedModelResponse,
        invocation: ResolvedInvocation,
    ) -> list[RawToolCallCandidate]:
        """Extract raw tool call candidates from a decoded model response.

        Merges codec-native candidates (``decoded.tool_candidates``), any
        pre-structured ``CanonicalToolCallBlock`` entries in content, and —
        when the invocation plan specifies a textual parser — candidates
        recovered by the ``ExtractorRegistry`` from the model's raw text
        output. Mutates ``decoded.content`` in place to remove consumed
        envelope text so it never leaks into the assembled response.
        """
        route = invocation.route
        plan = invocation.invocation_plan

        # Fail-closed boundary: a DISABLED route must never surface a tool
        # call regardless of what the model or backend emits. This check is
        # deliberately redundant with build_invocation_plan() clearing
        # parser_id/fallback_strategies for DISABLED plans — a future
        # plan-construction regression must not silently reactivate tools.
        if plan is None or plan.effective_tool_mode == ToolMode.DISABLED:
            return []

        native: list[RawToolCallCandidate] = list(decoded.tool_candidates)

        for block in decoded.content:
            if isinstance(block, CanonicalToolCallBlock):
                native.append(RawToolCallCandidate(
                    id=block.id,
                    name=block.name,
                    raw_arguments=json.dumps(block.arguments) if isinstance(block.arguments, dict) else str(block.arguments),
                    source_protocol=route.upstream.wire_protocol.value,
                    source_index=0,
                    choice_index=0,
                    tool_index=0,
                ))

        textual: list[RawToolCallCandidate] = []
        if plan is not None and plan.parser_id:
            # Textual extraction always runs when a parser is configured, even
            # when native candidates are already present. A hybrid response (one
            # that carries a native tool call AND a distinct textual <tool_call>
            # envelope) must contribute both: the native call and the textual
            # one. Shadow duplicates — a textual echo that exactly matches a
            # native candidate — are removed by _dedup_tool_candidates below,
            # which merges on (choice_index, tool_index, name, normalized_args),
            # so a well-formed native response is never masked by its own echo
            # while a genuinely distinct hybrid call survives.
            result = self._extractor_registry.extract(
                decoded.content,
                extractor_id=plan.parser_id,
                tools=plan.validation_tools,
                envelope=plan.output_envelope,
                fallback_strategies=plan.fallback_strategies,
                tool_choice=plan.original_tool_choice,
                native_candidates_present=bool(native),
                expected_execution_nonce=plan.execution_nonce,
            )
            textual = list(result.candidates)
            decoded.content = list(result.remaining_content)
            if invocation.execution_record is not None:
                for diag in result.diagnostics:
                    invocation.execution_record.record_parser_diagnostic(
                        f"[{diag.level}] {diag.envelope}: {diag.message}"
                    )

        return self._dedup_tool_candidates(native, textual)

    def _dedup_tool_candidates(
        self,
        native: list[RawToolCallCandidate],
        textual: list[RawToolCallCandidate],
    ) -> list[RawToolCallCandidate]:
        """Merge native and textually-extracted candidates, dropping a
        textual echo of a call the backend already reported natively.

        Every native candidate always survives unchanged — this function
        only ever decides whether a TEXTUAL candidate is a duplicate of
        some native one, never native-vs-native or textual-vs-textual, so
        genuinely parallel identical calls within either list are never
        touched here.

        Two matching strategies, applied in order:

        1. Provider call ID, when both sides have a real (non-empty) one.
           An exact ID match is the strongest possible duplicate signal —
           unrelated to name/arguments/index — so it settles the question
           on its own. Two candidates with DIFFERENT non-empty IDs are
           never merged by content alone; that would collapse genuinely
           distinct parallel calls that happen to share identical
           name+arguments (audit finding: "collapse distinct identical
           calls with separate IDs").
        2. Content signature (name + normalized arguments) — used ONLY as
           a fallback when the textual candidate has no ID to compare
           (the common case: text-dialect extractors rarely have access
           to the backend's native call ID). Deliberately excludes
           choice_index/tool_index: pre-structured candidates (e.g.
           whole-message JSON) are constructed with those indexes forced
           to 0 regardless of their real position, so index equality is
           neither necessary (a genuine echo can land at a different
           index) nor sufficient (two unrelated zero-indexed candidates
           would falsely look identical) for this decision.
        """
        if not textual:
            return native
        if not native:
            return textual

        def _sig(c: RawToolCallCandidate) -> tuple:
            return (c.name, _canonicalize_json_ish(c.raw_arguments))

        native_ids = {c.id for c in native if c.id}
        native_sigs = {_sig(c) for c in native}

        merged = list(native)
        for c in textual:
            if c.id:
                if c.id in native_ids:
                    continue  # confirmed duplicate — exact provider ID match
            elif _sig(c) in native_sigs:
                continue  # no ID to compare — fall back to content echo suppression
            merged.append(c)
        return merged

    def _apply_invocation_plan_to_request(
        self,
        rendered: dict[str, Any],
        plan: Any,
        route: ModelRoute,
    ) -> dict[str, Any]:
        """Apply the invocation plan to the rendered request.

        - NATIVE: send plan.upstream_tools (already in rendered)
        - PROMPTED: remove native tools, inject prompt_contract
        - DISABLED: remove tools, reject required/named choice
        - TEXTUAL: remove native tools
        """
        from agent_interop.config import ToolMode

        if plan.effective_tool_mode == ToolMode.NATIVE:
            # Tools already rendered
            pass
        elif plan.effective_tool_mode == ToolMode.PROMPTED:
            # Remove native tools and inject the contract
            rendered.pop("tools", None)
            rendered.pop("tool_choice", None)
            # Inject prompt contract into system message
            if plan.prompt_contract:
                rendered = self._inject_prompt_contract(rendered, plan.prompt_contract)
        elif plan.effective_tool_mode == ToolMode.DISABLED:
            # Remove all tools
            rendered.pop("tools", None)
            rendered.pop("tool_choice", None)
        elif plan.effective_tool_mode == ToolMode.TEXTUAL:
            # Remove native tools
            rendered.pop("tools", None)
            rendered.pop("tool_choice", None)

        return rendered

    def _inject_prompt_contract(self, rendered: dict[str, Any], contract: str) -> dict[str, Any]:
        """Inject the prompt contract into the system message."""
        messages = rendered.get("messages", [])
        if messages and messages[0].get("role") == "system":
            # Append to existing system message
            messages[0]["content"] = (messages[0].get("content", "") + "\n\n" + contract).strip()
        else:
            # Prepend a new system message
            messages.insert(0, {"role": "system", "content": contract})
        rendered["messages"] = messages
        return rendered

    def _resolve_profile(self, route: ModelRoute, backend_metadata: Any = None) -> Any:
        """Resolve the model profile for a route using ModelProfileRegistry.

        ``backend_metadata`` was previously computed by the caller (for the
        compatibility key) but never actually passed into ``resolve()`` —
        the registry's documented priority-5 tier ("backend metadata") was
        unreachable dead code from the live request path; only explicit
        profile ID, built-in pattern match, and the conservative fallback
        could ever fire.
        """
        return self._profile_registry.resolve(
            model_name=route.upstream_model,
            backend=route.upstream.kind,
            backend_metadata=backend_metadata,
            explicit_profile_id=route.profile if route.profile != "auto" else None,
        )

    def _get_backend_metadata(self, route: ModelRoute) -> Any:
        """Build BackendMetadata from route config (item 83).

        Populates what's available; empty strings for unknown dimensions.
        The evidence key still works — it just has fewer discriminating fields.
        """
        from agent_interop.model.registry import BackendMetadata

        return BackendMetadata(
            backend_kind=route.upstream.kind,
            model_name=route.upstream_model,
        )

    def _compute_tool_schema_fingerprint(self, tools: list[Any]) -> str:
        """Compute a fingerprint of the tool schema set for evidence lookup (item 83).

        Uses a hash of tool names + schema structure so that evidence is
        invalidated when tools change.
        """
        import hashlib
        import json

        if not tools:
            return ""
        # Canonical representation: sorted by name, with schema
        canonical = sorted(
            [{"name": t.name, "schema": t.input_schema} for t in tools],
            key=lambda x: x["name"],
        )
        raw = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _compute_argument_digest(self, arguments: Any) -> str:
        """Compute a stable digest of a tool call's arguments for loop detection.

        Matches the style of ``_compute_tool_schema_fingerprint``: a SHA-256
        prefix of a canonical JSON representation. Returns "" when no
        arguments are present so that argument-less calls collapse to a
        single digest (and thus still flag genuine repeat calls).
        """
        import hashlib

        if not arguments:
            return ""
        raw = _canonicalize_json_ish(arguments)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _apply_confidence_gate(self, policy: Any, confidence: float) -> Any:
        """Gate risky repair tiers when profile confidence is low (item 86 integration).

        Low confidence (fallback, <0.5): only SYNTAX_ONLY and SAFE_SHAPE tiers.
        Medium confidence (builtin, ~0.8): add COERCIVE.
        High confidence (override/explicit, ≥0.9): all tiers including REGENERATION.
        """
        from dataclasses import replace

        if not hasattr(policy, 'enabled_tiers'):
            return policy

        from agent_interop.config import RepairTier

        current = set(policy.enabled_tiers)

        if confidence < 0.5:
            # Low confidence — disable coercive and regeneration
            current.discard(RepairTier.COERCIVE)
            current.discard(RepairTier.REGENERATION)
        elif confidence < 0.9:
            # Medium confidence — allow coercive but not regeneration
            current.discard(RepairTier.REGENERATION)
        # High confidence — keep all enabled tiers

        if current == set(policy.enabled_tiers):
            return policy
        return replace(policy, enabled_tiers=frozenset(current))

    def _assemble_response(
        self,
        decoded: DecodedModelResponse,
        batch_decision: Any,
        canonical: CanonicalRequest,
        route: ModelRoute,
    ) -> CanonicalResponse:
        """Assemble the canonical response from decoded + transaction output."""

        content: list[CanonicalContentBlock] = []

        # Add non-tool content
        for block in decoded.content:
            if not isinstance(block, CanonicalToolCallBlock):
                content.append(block)

        # Add accepted tool blocks from the transaction decision
        content.extend(batch_decision.accepted_blocks)

        # Determine stop reason
        stop_reason = decoded.stop_reason
        if batch_decision.accepted_blocks and stop_reason == CanonicalStopReason.END_TURN:
            stop_reason = CanonicalStopReason.TOOL_CALL

        # Check for batch-level errors
        error = None
        if not batch_decision.is_accepted and not batch_decision.accepted_blocks:
            # Complete batch rejection. Never report a TOOL_CALL stop reason
            # when the whole batch was rejected — the client must see
            # INVALID_OUTPUT so it knows none of the calls were executed.
            stop_reason = CanonicalStopReason.INVALID_OUTPUT
            error = self._build_batch_rejection_error(batch_decision, canonical.request_id)

        return CanonicalResponse(
            content=content,
            stop_reason=stop_reason,
            usage=decoded.usage,
            model=CanonicalModelReference(
                requested_name=canonical.model.requested_name,
                resolved_name=route.upstream_model,
            ),
            request_id=canonical.request_id,
            response_id=decoded.extra.get("response_id", ""),
            error=error,
        )

    def _build_batch_rejection_error(
        self,
        batch_decision: Any,
        request_id: str,
    ) -> CanonicalError:
        """Build a structured ``CanonicalError`` for a fully-rejected tool batch.

        Two distinct scenarios:
        - **Tool-choice policy violation** (``choice_error`` is set): the batch
          was rejected wholesale because the calls violated the
          REQUIRED/NAMED/NONE choice contract. This gets its own code.
        - **Per-call rejections**: individual calls failed validation/repair.
          The batch-level error is synthesized from the per-decision data so
          the client gets structured, actionable feedback.
        """
        # Tool-choice policy violation — whole batch rejected before per-call
        # processing. This is the canonical "the model disobeyed the choice
        # contract" signal.
        if batch_decision.choice_error:
            return CanonicalError(
                code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                message=batch_decision.choice_error,
                retryable=True,
                request_id=request_id,
            )

        # Per-call rejections: synthesize a batch-level error from the
        # per-decision data.
        rejected = [d for d in batch_decision.decisions if d.is_rejected]
        messages = [d.outcome.error for d in rejected if d.outcome.error]
        if not messages:
            messages = [d.outcome.final_issues[0].message for d in rejected
                        if d.outcome.final_issues]
        message = "; ".join(messages) if messages else "All tool calls in the batch were rejected"

        # A repair attempt that still failed is retryable (a regeneration may
        # succeed next time); a call that was never valid to begin with is not.
        repair_attempted_failed = any(
            d.outcome.status == RepairStatus.REJECTED and d.outcome.steps
            for d in rejected
        )
        code = (
            InteropErrorCode.TOOL_CALL_REPAIR_FAILED
            if repair_attempted_failed
            else InteropErrorCode.TOOL_CALL_INVALID
        )

        # Structured per-call correction info so the client can see exactly
        # which calls failed and why.
        rejected_calls: list[dict[str, Any]] = []
        for d in rejected:
            correction = d.correction
            if correction is not None:
                rejected_calls.append({
                    "tool_name": correction.tool_name,
                    "candidate_id": correction.candidate_id,
                    "issue_path": correction.issue_path,
                    "schema_keyword": correction.schema_keyword,
                    "observed_type": correction.observed_type,
                    "expected_type": correction.expected_type,
                    "allowed_values": correction.allowed_values,
                    "message": correction.message,
                    "retryable": correction.retryable,
                })
            else:
                rejected_calls.append({
                    "tool_name": d.outcome.call_name or d.candidate.name or "",
                    "candidate_id": d.candidate.id or "",
                    "message": d.outcome.error,
                })

        return CanonicalError(
            code=code,
            message=message,
            retryable=repair_attempted_failed,
            request_id=request_id,
            details={"rejected_calls": rejected_calls},
        )

    # ─── Streaming request ────────────────────────────────────────────────

    async def handle_stream(
        self,
        canonical: CanonicalRequest,
        context: Any,
    ) -> AsyncGenerator[CanonicalEvent, None]:
        """Handle a streaming request, yielding canonical events.

        The streaming path uses the same preparation pipeline as non-streaming:
            _prepare_invocation → codec render → transport → decode
            → extraction → transaction → canonical events

        For NATIVE_FRAGMENTS mode: text streams through immediately, tool fragments
        are accumulated and validated through the transaction service before emission.

        For BUFFER_TEXTUAL_RESPONSE mode: buffer model text until complete, then
        extract and validate.
        """
        exec_record = InteropRequestExecution(context=context)
        try:
            # Prepare the resolved invocation (same as non-streaming; passes in
            # the shared record so diagnostics/route/plan land on the record
            # that gets finalized)
            invocation = self._prepare_invocation(
                canonical, context, streaming=True, execution=exec_record,
            )
            # The sub-generator finalizes the record (success or internal
            # terminal error) — and logs the summary — BEFORE yielding its
            # terminal event, not after. The ASGI server stops consuming this
            # generator as soon as it sees message_stop, so nothing here can
            # rely on running after the last yield to be reached.
            async for event in self._handle_stream_send(invocation, exec_record):
                yield event
        except asyncio.CancelledError:
            # Mid-request cancellation: finalize the record as CANCELLED so it is
            # not left permanently ACTIVE, then re-raise. Do NOT yield
            # any further frames — the client connection is already gone.
            exec_record.finalize_cancelled()
            raise
        except Exception as exc:
            exc_err = CanonicalError(code="STREAM_ERROR", message=str(exc)) if not isinstance(exc, CanonicalError) else exc
            exec_record.finalize_error(exc_err)
            yield CanonicalEvent(type="error", error=exc_err)
            yield CanonicalEvent(type="message_stop")
            return

    async def _handle_stream_send(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
    ) -> AsyncIterator[CanonicalEvent]:
        """Send a prepared invocation to the upstream and stream decoded events."""
        # Unsafe history — finalize BEFORE yielding the terminal event. The
        # server stops consuming this generator as soon as it sees
        # message_stop, so any bookkeeping placed after that yield may never
        # run through the real ASGI path.
        if invocation.invocation_plan is None or invocation.codec is None:
            err = CanonicalError(
                code=InteropErrorCode.HISTORY_UNSAFE,
                message="History reconciliation detected unsafe history",
            )
            exec_record.finalize_error(err)
            yield CanonicalEvent(type="error", error=err)
            yield CanonicalEvent(type="message_stop")
            return

        route = invocation.route
        plan = invocation.invocation_plan
        codec = invocation.codec
        canonical = invocation.reconciled_request

        choice_conflict = self._disabled_tool_choice_conflict(plan)
        if choice_conflict is not None:
            exec_record.finalize_error(choice_conflict)
            yield CanonicalEvent(type="error", error=choice_conflict)
            yield CanonicalEvent(type="message_stop")
            return

        # Render through codec
        request_local = copy.deepcopy(canonical)
        rendered = codec.render_request(request_local, route.upstream_model, stream=True)
        rendered = self._apply_invocation_plan_to_request(rendered, plan, route)

        # Build typed upstream request
        upstream_request = PreparedUpstreamRequest(
            method="POST",
            url=f"{route.upstream.base_url}{codec.endpoint_path()}",
            headers=self._build_upstream_headers(
                route,
                client_headers=dict(invocation.request_context.forwardable_transport_headers),
                codec_headers=codec.required_headers(),
            ),
            body=rendered,
            stream=True,
            timeout_seconds=route.upstream.timeout_seconds,
        )

        coordinator = StreamCoordinator(
            route.upstream.wire_protocol,
            limits=StreamLimits(
                max_accumulated_arg_bytes=self.config.max_tool_argument_bytes,
                max_simultaneous_tool_calls=self.config.max_simultaneous_tool_calls,
            ),
        )

        async with self.transport.stream(upstream_request) as stream:
            if stream.status_code >= 400:
                # Read a short excerpt of the error body for the message
                excerpt = ""
                try:
                    parts: list[str] = []
                    total = 0
                    async for raw in stream.raw_lines():
                        parts.append(raw)
                        total += len(raw)
                        if total >= 200 or len(parts) >= 5:
                            break
                    excerpt = "".join(parts)[:200]
                except Exception:
                    excerpt = ""
                error_msg = f"Upstream returned {stream.status_code}"
                if excerpt:
                    error_msg += f": {excerpt}"
                canonical_error = CanonicalError(
                    code=classify_http_status(stream.status_code), message=error_msg
                )
                exec_record.finalize_error(canonical_error)
                yield CanonicalEvent(type="error", error=canonical_error)
                yield CanonicalEvent(type="message_stop")
                return

            # The invocation plan decides how tool calls are extracted. For
            # PROMPTED-mode local models stream_extraction_mode is
            # BUFFER_TEXTUAL_RESPONSE: raw <tool_call>...</tool_call> envelopes
            # must be buffered until the response is complete, then run through
            # the SAME textual-extraction machinery as the non-streaming path —
            # never streamed straight through as text. Any other mode uses the
            # native-fragments path (text streams through immediately).
            mode = plan.stream_extraction_mode
            buffered_text_parts: list[str] = []
            final_stop_reason: CanonicalStopReason | None = None
            final_usage: CanonicalUsage | None = None
            malformed_frame_count = 0

            # Decode stream frames through codec
            async for frame_data, raw_frame_text in self._iter_frame_data(stream, codec.stream_framing):
                if frame_data is None:
                    # Malformed frame. Record a bounded, sanitized diagnostic
                    # regardless of outcome so evidence/replay can see how
                    # often a backend emits unparseable frames.
                    malformed_frame_count += 1
                    exec_record.record_malformed_frame(
                        malformed_frame_count, "unparseable_frame", raw_frame_text,
                    )
                    if coordinator.has_pending_tool_calls:
                        # Fail the tool batch and terminate
                        coordinator.tool_accumulator.fail_all_pending("malformed_frame")
                        err = CanonicalError(
                            code="MALFORMED_FRAME",
                            message="Malformed frame with open tool state",
                        )
                        # GAP 5 FIX — a malformed frame with open tool state is a
                        # terminal error; the record must be finalized as FAILED
                        # rather than left permanently ACTIVE. Finalize BEFORE
                        # yielding message_stop — the server stops consuming
                        # this generator as soon as it sees that event.
                        exec_record.finalize_error(err)
                        yield CanonicalEvent(type="error", error=err)
                        yield CanonicalEvent(type="message_stop")
                        return
                    if malformed_frame_count > self.config.max_malformed_stream_frames:
                        # No open tool state, but the backend has now sent
                        # more malformed frames than the configured threshold.
                        # Without a bound, a backend that never resends valid
                        # frames could be silently `continue`d forever.
                        err = CanonicalError(
                            code="MALFORMED_FRAME",
                            message=(
                                f"Too many malformed stream frames "
                                f"({malformed_frame_count} > "
                                f"{self.config.max_malformed_stream_frames})"
                            ),
                        )
                        exec_record.finalize_error(err)
                        yield CanonicalEvent(type="error", error=err)
                        yield CanonicalEvent(type="message_stop")
                        return
                    continue

                # GAP 2 FIX — always decode the frame FIRST, even when it is
                # the terminal frame. The terminal frame often carries the final
                # tool fragments, usage, and/or stop reason; checking
                # is_stream_complete first and returning early silently dropped
                # all of that. Decode now, then consult is_stream_complete below.
                is_complete = codec.is_stream_complete(frame_data)
                decoded_events: list[DecodedStreamEvent] = codec.decode_stream_chunk(frame_data)

                stream_error: CanonicalError | None = None
                for decoded_event in decoded_events:
                    # Handle text deltas. In BUFFER_TEXTUAL_RESPONSE mode the
                    # model emits raw tool envelopes as text, so we must buffer
                    # and extract later — never yield the envelope literally.
                    if isinstance(decoded_event, DecodedTextDelta):
                        if mode == StreamExtractionMode.BUFFER_TEXTUAL_RESPONSE:
                            buffered_text_parts.append(decoded_event.text)
                        else:
                            yield CanonicalEvent(
                                type="text_delta",
                                index=0,
                                partial=decoded_event.text,
                            )

                    # Handle tool batch completion — complete all pending calls
                    elif isinstance(decoded_event, DecodedToolBatchComplete):
                        coordinator.tool_accumulator.complete_all_pending()

                    # Handle tool fragments - accumulate via the accumulator's
                    # feed methods so size limits are actually enforced.
                    elif isinstance(decoded_event, DecodedToolFragment):
                        key = ToolStreamKey(
                            choice_index=decoded_event.choice_index,
                            tool_index=decoded_event.tool_index,
                        )
                        coordinator.tool_accumulator.start_call(key, decoded_event.call_id_fragment or None)
                        if decoded_event.name_fragment:
                            coordinator.tool_accumulator.feed_name(key, decoded_event.name_fragment)
                        if decoded_event.argument_fragment:
                            # GAP 3 FIX — route argument fragments through
                            # feed_arguments so max_accumulated_arg_bytes is
                            # enforced. The old direct-append path bypassed the
                            # limit entirely. A limit breach is a terminal error.
                            try:
                                coordinator.tool_accumulator.feed_arguments(key, decoded_event.argument_fragment)
                            except ToolCallLimitExceeded as exc:
                                err = CanonicalError(
                                    code="TOOL_CALL_LIMIT_EXCEEDED",
                                    message=str(exc),
                                )
                                exec_record.finalize_error(err)
                                yield CanonicalEvent(type="error", error=err)
                                yield CanonicalEvent(type="message_stop")
                                return

                    # Handle per-call completion
                    elif isinstance(decoded_event, DecodedToolCallComplete):
                        key = ToolStreamKey(choice_index=decoded_event.choice_index, tool_index=decoded_event.tool_index)
                        coordinator.tool_accumulator.complete_call(key)

                    # GAP 2 — capture usage updates from terminal frames instead
                    # of silently dropping them.
                    elif isinstance(decoded_event, DecodedUsageUpdate):
                        final_usage = decoded_event.usage

                    # Capture the stream's stop reason and any trailing usage.
                    elif isinstance(decoded_event, DecodedStreamComplete):
                        final_stop_reason = decoded_event.stop_reason
                        if decoded_event.usage is not None:
                            final_usage = decoded_event.usage

                    # Surface a stream-level error as a terminal error event.
                    elif isinstance(decoded_event, DecodedStreamError):
                        stream_error = CanonicalError(
                            code="BACKEND_STREAM_ERROR",
                            message=decoded_event.error,
                        )

                if stream_error is not None:
                    exec_record.finalize_error(stream_error)
                    yield CanonicalEvent(type="error", error=stream_error)
                    yield CanonicalEvent(type="message_stop")
                    return

                # GAP 4 FIX — the mid-loop drain_completed() + immediate
                # _process_completed_stream_tools(...) call that used to live
                # here is DELETED. Completed tool calls now accumulate in the
                # coordinator until end-of-turn (below), so the whole turn's
                # tool calls are validated as ONE atomic batch instead of being
                # split across per-drain decisions.

                if is_complete:
                    break

            # ── End-of-turn (shared by terminal-frame break AND natural loop-end) ──
            # Native fragments are accumulated during the loop regardless of the
            # plan's extraction mode (the codec emits them from the wire stream,
            # which does not depend on the plan). So always finish any still-
            # pending calls and drain the WHOLE turn's completed calls. In BUFFER
            # mode a prompted model emits its tool envelopes as text deltas rather
            # than native fragments — but a model can also hallucinate native-style
            # tool_calls deltas even when PROMPTED mode stripped tool schemas from
            # the request, so the two sources are NOT guaranteed mutually exclusive.
            # To honour the turn-level atomicity guarantee, BOTH candidate sources
            # are merged into ONE deduped list (via _dedup_tool_candidates) and
            # decided as a single atomic batch. In non-BUFFER mode textual
            # candidates are empty, so this collapses to the single native batch
            # exactly as before.
            coordinator.tool_accumulator.complete_all_pending()
            remaining = coordinator.tool_accumulator.drain_completed()

            if mode == StreamExtractionMode.BUFFER_TEXTUAL_RESPONSE:
                # GAP 1 FIX — run the buffered raw text through the SAME
                # textual-extraction machinery as the non-streaming path. The
                # envelopes are consumed here so the literal <tool_call> text
                # never leaks to the client as a text delta. Native fragments (if
                # any) are merged with the textual candidates below so the WHOLE
                # turn is decided as ONE atomic batch instead of two.
                native_candidates = self._pending_to_candidates(remaining, invocation)
                textual_candidates: list[RawToolCallCandidate] = []
                remaining_content: list[CanonicalContentBlock] = []
                buffered_text = "".join(buffered_text_parts)
                if buffered_text:
                    decoded = DecodedModelResponse(
                        content=[CanonicalTextBlock(text=buffered_text)],
                    )
                    textual_candidates = self._extract_tool_candidates(decoded, invocation)
                    remaining_content = list(decoded.content)

                merged = self._dedup_tool_candidates(native_candidates, textual_candidates)
                if merged:
                    transaction_context = self._build_transaction_context(invocation, canonical)
                    batch_decision = await process_tool_batch(
                        merged,
                        canonical.tools,
                        context=transaction_context,
                        policy=ToolBatchPolicy(invocation.repair_policy.batch_policy),
                    )
                    self._record_repairs_to_session(batch_decision, invocation.request_context)
                    # Record per-call decisions onto the shared execution record
                    # unconditionally — the in-memory record is always populated
                    # (so finalize_response's outcome classification sees the
                    # decisions), matching the non-streaming path (~line 938) and
                    # the non-BUFFER streaming path (_process_completed_stream_
                    # tools). Evidence-store write-back remains a separate, opt-in
                    # step in _record_evidence_observation.
                    self._record_tool_decisions(batch_decision, invocation.execution_record)
                    # Emit the remaining (non-envelope) text, then the accepted
                    # tool calls, exactly as the non-streaming path assembles them.
                    # Plain text is emitted even on rejection — _assemble_response
                    # keeps content text blocks alongside a set .error.
                    for block in remaining_content:
                        if isinstance(block, CanonicalTextBlock) and block.text:
                            yield CanonicalEvent(type="text_delta", index=0, partial=block.text)
                    # Emit the decided batch: either the accepted tool_use blocks,
                    # or — on a fully-rejected batch — a structured error +
                    # INVALID_OUTPUT message_stop (mirrors _assemble_response's
                    # non-streaming handling). The shared helper also finalizes the
                    # record as failed and marks the coordinator's turn rejected so
                    # the caller skips its generic end-of-turn tail.
                    async for event in self._emit_batch_decision_events(
                        batch_decision, canonical.request_id, coordinator, invocation.execution_record,
                    ):
                        yield event
                else:
                    # No candidates extracted but there is plain text — yield it.
                    if buffered_text:
                        yield CanonicalEvent(type="text_delta", index=0, partial=buffered_text)
            else:
                # Non-BUFFER mode: native fragments only (text already streamed as
                # it arrived). Decide them as one atomic batch via the single-source
                # path — byte-for-byte identical to the pre-fix behaviour.
                if remaining:
                    async for event in self._process_completed_stream_tools(
                        remaining, invocation, coordinator,
                    ):
                        yield event

            # A fully-rejected batch already emitted its terminal error +
            # message_stop(INVALID_OUTPUT) and finalized the record as failed via
            # the shared helper. Skip the generic end-of-turn tail (stop-reason
            # computation, a second message_stop, evidence write-back, and
            # finalize_response) — mirroring the non-streaming path's
            # `result.error is None` gate on evidence write-back.
            if coordinator.turn_rejected:
                return

            # GAP 6 FIX — read the public property instead of the private attr.
            stop_reason = final_stop_reason or (
                CanonicalStopReason.TOOL_CALL
                if coordinator.has_emitted_tool_calls
                else CanonicalStopReason.END_TURN
            )

            # Live evidence write-back at clean stream end: only when tools were
            # offered and an evidence store was injected. This shared end-of-turn
            # path is reached by BOTH the terminal-frame break and the natural
            # loop-end, so write-back fires on both success exits.
            #
            # IMPORTANT: this tail runs BEFORE the terminal message_stop is
            # yielded. The server stops consuming this generator as soon as it
            # sees message_stop, so bookkeeping placed after that yield may
            # never execute through the real ASGI path (only direct-generator
            # tests would see it run). If finalization itself fails, surface
            # it as a genuine terminal error instead of silently completing.
            try:
                if self._evidence_store is not None and canonical.tools:
                    self._record_evidence_observation(invocation, exec_record)

                # GAP 5 FIX — the terminal-frame completion path (the most common
                # one for OpenAI/Ollama streams) used to return WITHOUT finalizing
                # the record. Unifying both exits into this shared path means a
                # normal completion now finalizes as SUCCEEDED.
                exec_record.finalize_response(CanonicalResponse(usage=final_usage or CanonicalUsage()))
            except Exception as exc:
                logger.warning(
                    "stream finalization failed before the terminal event",
                    exc_info=True,
                )
                err = CanonicalError(code="STREAM_ERROR", message=str(exc))
                exec_record.finalize_error(err)
                yield CanonicalEvent(type="error", error=err)
                yield CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT)
                return

            if final_usage is not None:
                yield CanonicalEvent(
                    type="usage_update",
                    input_tokens=final_usage.input_tokens,
                    output_tokens=final_usage.output_tokens,
                )

            yield CanonicalEvent(type="message_stop", stop_reason=stop_reason)

    async def _emit_batch_decision_events(
        self,
        batch_decision: Any,
        request_id: str,
        coordinator: StreamCoordinator,
        exec_record: InteropRequestExecution,
    ) -> AsyncIterator[CanonicalEvent]:
        """Emit canonical events for a decided tool batch.

        Fully rejected atomic batch (no calls accepted): emits an ``error``
        event with the structured rejection (mirrors ``_assemble_response``'s
        non-streaming handling), then ``message_stop(INVALID_OUTPUT)``,
        finalizes the execution record as failed, and marks the coordinator's
        turn as rejected so the caller skips its own generic end-of-turn
        handling (stop-reason computation, second message_stop, and evidence
        write-back). Accepted / partially-accepted batches emit each accepted
        tool_use block, unchanged from prior behavior.
        """
        if not batch_decision.is_accepted and not batch_decision.accepted_blocks:
            rejection_error = self._build_batch_rejection_error(batch_decision, request_id)
            exec_record.finalize_error(rejection_error)
            coordinator.mark_turn_rejected()
            yield CanonicalEvent(type="error", error=rejection_error)
            yield CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT)
            return

        for accepted_block in batch_decision.accepted_blocks:
            coordinator.mark_tool_calls_emitted()
            yield CanonicalEvent(type="tool_use", index=0, content_block=accepted_block)

    def _pending_to_candidates(
        self,
        completed: list[PendingToolCall],
        invocation: ResolvedInvocation,
    ) -> list[RawToolCallCandidate]:
        """Convert drained ``PendingToolCall``s to ``RawToolCallCandidate``s.

        Shared by the native-only streaming path (``_process_completed_stream_
        tools``) and the combined native+textual BUFFER path so both build
        candidates from the same wire fragments.
        """
        route = invocation.route
        candidates: list[RawToolCallCandidate] = []
        for c in completed:
            if c.completed and c.name_fragments:
                name = "".join(c.name_fragments)
                args_str = "".join(c.argument_fragments)
                candidates.append(RawToolCallCandidate(
                    id=c.call_id,
                    name=name,
                    raw_arguments=args_str,
                    source_protocol=route.upstream.wire_protocol.value,
                    source_index=c.tool_index,
                    choice_index=c.choice_index,
                    tool_index=c.tool_index,
                ))
        return candidates

    async def _process_completed_stream_tools(
        self,
        completed: list[PendingToolCall],
        invocation: ResolvedInvocation,
        coordinator: StreamCoordinator,
    ) -> AsyncIterator[CanonicalEvent]:
        """Process completed native tool calls through the transaction service.

        Used by the non-BUFFER streaming path where native fragments are the
        only candidate source. The combined native+textual BUFFER path builds
        its candidate list separately (via ``_pending_to_candidates``) but runs
        the same single atomic batch below the end-of-turn merge.
        """
        candidates = self._pending_to_candidates(completed, invocation)
        if not candidates:
            return

        canonical = invocation.reconciled_request
        request_context = invocation.request_context

        # Run through the transaction service. Uses the shared helper so the
        # streaming path gets the confidence-gated repair policy, the
        # request-scoped budget (shared across batches), telemetry, and the
        # compatibility key — identical to the non-streaming path.
        transaction_context = self._build_transaction_context(invocation, canonical)
        batch_decision = await process_tool_batch(
            candidates,
            canonical.tools,
            context=transaction_context,
            policy=ToolBatchPolicy(invocation.repair_policy.batch_policy),
        )

        # Record repairs into session state for loop detection
        self._record_repairs_to_session(batch_decision, request_context)

        # Record per-call decisions onto the shared execution record. The
        # in-memory record is always populated; evidence-store write-back is a
        # separate, opt-in step in _record_evidence_observation.
        self._record_tool_decisions(batch_decision, invocation.execution_record)

        # Emit the decided batch: either the accepted tool_use blocks, or — on a
        # fully-rejected batch — a structured error + INVALID_OUTPUT message_stop
        # (mirrors _assemble_response's non-streaming handling). The shared
        # helper also finalizes the record as failed and marks the coordinator's
        # turn rejected so the caller skips its generic end-of-turn tail.
        async for event in self._emit_batch_decision_events(
            batch_decision, canonical.request_id, coordinator, invocation.execution_record,
        ):
            yield event

    async def _iter_frame_data(
        self,
        stream: Any,  # UpstreamStream
        framing: Any,  # StreamFraming
    ) -> AsyncIterator[tuple[dict[str, Any] | None, str]]:
        """Normalize SSE or NDJSON frames into the shape the downstream
        streaming loop expects.

        Yields ``(frame, raw_text)`` per wire frame. ``frame`` is a ``dict``
        for a successfully parsed frame, or ``None`` for a malformed/
        unparseable one — matching exactly what ``_parse_stream_frame`` used
        to return. ``raw_text`` is the frame's original text (bounded by the
        caller before use) so malformed frames can be recorded as bounded
        diagnostics instead of being dropped with no trace.
        """
        from agent_interop.upstreams.codec import StreamFraming

        if framing == StreamFraming.NDJSON:
            async for item in stream.ndjson_events():
                if isinstance(item, MalformedNDJSONLine):
                    yield None, item.line
                else:
                    # NDJSON yields already-parsed frames (dicts); yield them
                    # directly. Non-dict items are treated as unparseable.
                    yield (item if isinstance(item, dict) else None), str(item)
            return

        # Default: SSE framing
        async for frame in stream.sse_events():
            data = frame.data
            if not data or not data.strip():
                # Skip empty/whitespace-only frames (e.g. keep-alives)
                continue
            stripped = data.strip()
            if stripped == "[DONE]":
                yield {"done": True}, stripped
                continue
            try:
                yield json.loads(stripped), stripped
            except json.JSONDecodeError:
                yield None, stripped

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _build_upstream_auth_config(self, route: ModelRoute) -> Any:
        """Build a typed UpstreamAuthConfig from the route's loose auth dict.

        The legacy ``route.upstream.api_key_env`` field (used by the probe
        path) is translated into the equivalent typed ``API_KEY`` config here
        when no explicit ``auth`` dict is present, so probing, inference,
        streaming, and count_tokens all resolve upstream auth identically.
        """
        from agent_interop.auth import UpstreamAuthConfig, UpstreamAuthMode

        auth = route.upstream.auth
        if not auth:
            # Translate the legacy api_key_env field into the typed
            # UpstreamAuthConfig mechanism so real requests honor it too.
            if route.upstream.api_key_env:
                return UpstreamAuthConfig(
                    mode=UpstreamAuthMode.API_KEY,
                    env_key=route.upstream.api_key_env,
                )
            return UpstreamAuthConfig(mode=UpstreamAuthMode.NONE)

        mode_str = auth.get("mode", "none")
        try:
            mode = UpstreamAuthMode(mode_str)
        except ValueError:
            # An invalid mode string is a config error, not a silent NONE
            # fallback — but this method can't raise per its contract, so the
            # invalid value is caught and rejected by validate_config at load
            # time (which see). Treat as NONE here as a last-resort default.
            mode = UpstreamAuthMode.NONE

        # The "command" field may be a list or a stringified list; normalize to list
        raw_command: Any = auth.get("command", [])
        if isinstance(raw_command, str):
            import shlex
            command = shlex.split(raw_command)
        elif isinstance(raw_command, list):
            command = raw_command
        else:
            command = []

        return UpstreamAuthConfig(
            mode=mode,
            api_key=auth.get("token") or auth.get("api_key"),
            api_key_header=auth.get("api_key_header", "Authorization"),
            env_key=auth.get("env_key"),
            command=command,
        )

    def _build_upstream_headers(
        self,
        route: ModelRoute,
        client_headers: dict[str, str] | None = None,
        codec_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build upstream headers using the auth module (item 92).

        Merges codec-required headers with route auth headers.
        """
        from agent_interop.auth import build_upstream_headers

        auth_config = self._build_upstream_auth_config(route)
        headers = build_upstream_headers(
            client_headers or {},
            auth_config,
            route.upstream.static_headers,
        )
        # Codec-required headers (Content-Type, anthropic-version, etc.)
        if codec_headers:
            headers.update(codec_headers)
        return headers