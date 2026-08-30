"""Ollama runtime inspection through Interop's shared transport."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from agent_interop.backends.base import ModelRuntimeCapabilities
from agent_interop.capabilities import CapabilityState
from agent_interop.config import ModelRoute
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamTransport


def _digest(value: Any) -> str:
    if not value:
        return ""
    raw = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _context_from_options(*sources: dict[str, Any]) -> int:
    for source in sources:
        # Check flat keys first
        for key in ("num_ctx", "context_length", "context_window", "num_ctx_train"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
        # Check architecture-qualified keys (e.g. 'llama.context_length')
        for key, value in source.items():
            if key.endswith(".context_length") or key.endswith(".num_ctx"):
                if isinstance(value, int) and value > 0:
                    return value
                if isinstance(value, str) and value.isdigit():
                    return int(value)
    return 0


def _headers(route: ModelRoute) -> dict[str, str]:
    """Build the same static/API-key header shape as normal upstream calls."""
    from agent_interop.auth import UpstreamAuthConfig, UpstreamAuthMode, build_upstream_headers

    raw = route.upstream.auth
    if raw:
        try:
            mode = UpstreamAuthMode(raw.get("mode", "none"))
        except ValueError:
            mode = UpstreamAuthMode.NONE
        auth = UpstreamAuthConfig(
            mode=mode,
            api_key=raw.get("token") or raw.get("api_key"),
            api_key_header=raw.get("api_key_header", "Authorization"),
            env_key=raw.get("env_key"),
        )
    elif route.upstream.api_key_env:
        auth = UpstreamAuthConfig(mode=UpstreamAuthMode.API_KEY, env_key=route.upstream.api_key_env)
    else:
        auth = UpstreamAuthConfig(mode=UpstreamAuthMode.NONE)
    return build_upstream_headers({}, auth, route.upstream.static_headers)


class OllamaInspector:
    """Read Ollama's model/runtime metadata without creating an httpx client."""

    def __init__(self) -> None:
        self._qualification_cache: dict[str, ModelRuntimeCapabilities] = {}

    async def _request(
        self, transport: UpstreamTransport, route: ModelRoute, method: str, path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await transport.send(PreparedUpstreamRequest(
            method=method,
            url=f"{route.upstream.base_url.rstrip('/')}{path}",
            headers=_headers(route),
            body=body or {},
            stream=False,
            timeout_seconds=min(route.upstream.timeout_seconds, 15.0),
        ))
        if response.transport_failed or response.status_code >= 400:
            return {}
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    async def _probe(
        self, transport: UpstreamTransport, route: ModelRoute, body: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Run one bounded, side-effect-free /api/chat feature probe."""
        response = await transport.send(PreparedUpstreamRequest(
            method="POST",
            url=f"{route.upstream.base_url.rstrip('/')}/api/chat",
            headers=_headers(route),
            body=body,
            stream=False,
            timeout_seconds=min(route.upstream.timeout_seconds, 15.0),
        ))
        if response.transport_failed or response.status_code >= 400:
            return False, {}
        try:
            payload = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}
        return True, payload if isinstance(payload, dict) else {}

    async def inspect_runtime_metadata(
        self, route: ModelRoute, transport: UpstreamTransport,
    ) -> ModelRuntimeCapabilities:
        """Cheaply determine model/runtime metadata WITHOUT generations.

        P1.3: Split from behavioral qualification. Metadata reads
        (digest, architecture, quantization, context, template,
        advertised capabilities) are cheap and should not require
        running /api/chat probes.
        """
        version, tags, shown, running = await __import__("asyncio").gather(
            self._request(transport, route, "GET", "/api/version"),
            self._request(transport, route, "GET", "/api/tags"),
            self._request(transport, route, "POST", "/api/show", {"model": route.upstream_model}),
            self._request(transport, route, "GET", "/api/ps"),
        )
        tag = next((item for item in tags.get("models", []) if item.get("name") == route.upstream_model), {})
        loaded = next((item for item in running.get("models", []) if item.get("name") == route.upstream_model), {})
        details = shown.get("details") if isinstance(shown.get("details"), dict) else {}
        model_info = shown.get("model_info") if isinstance(shown.get("model_info"), dict) else {}
        capabilities = {str(item).lower() for item in shown.get("capabilities", [])}
        template = str(shown.get("template", ""))
        architecture_limit = _context_from_options(model_info, details)
        # P0.21: configured_limit = serving allocation (from /api/ps), only
        # valid when the model is actually loaded with a num_ctx. When the
        # model isn't loaded, this is 0 (unknown), NOT architecture_limit.
        configured_limit = _context_from_options(loaded, loaded.get("details", {}) if isinstance(loaded.get("details"), dict) else {})
        # P0.21: effective = the actual serving allocation. Falls back to 0
        # (unknown) when the model isn't loaded and no serving config exists.
        # Architecture limit is a ceiling, not a serving guarantee.
        if configured_limit > 0:
            effective = configured_limit
        elif architecture_limit > 0:
            # Model not loaded; architecture_limit is the model family maximum,
            # not what Ollama is serving. Report 0 as unknown serving capacity.
            effective = 0
        else:
            effective = 0
        declared_tools = "tools" in capabilities
        declared_images = "vision" in capabilities or "images" in capabilities
        family = details.get("family")
        families = details.get("families") or []
        architecture = str(family or (families[0] if families else ""))
        return ModelRuntimeCapabilities(
            backend_kind=route.upstream.kind,
            backend_version=str(version.get("version", "")),
            model_name=route.upstream_model,
            model_digest=str(tag.get("digest") or shown.get("digest") or loaded.get("digest") or ""),
            architecture=architecture,
            quantization=str(details.get("quantization_level", "")),
            parameter_count=str(details.get("parameter_size", "")),
            architecture_context_tokens=architecture_limit,
            configured_context_tokens=configured_limit,
            effective_context_tokens=effective,
            chat_template=template,
            chat_template_digest=_digest(template),
            accepts_native_tools=CapabilityState.DECLARED if declared_tools else CapabilityState.UNSUPPORTED,
            returns_native_tool_calls=CapabilityState.DECLARED if declared_tools else CapabilityState.UNSUPPORTED,
            accepts_named_tool_choice=CapabilityState.UNSUPPORTED,
            accepts_required_tool_choice=CapabilityState.UNSUPPORTED,
            accepts_parallel_tool_flag=CapabilityState.UNSUPPORTED,
            supports_json_schema=CapabilityState.DECLARED if "structured_output" in capabilities else CapabilityState.UNSUPPORTED,
            supports_json_mode=CapabilityState.DECLARED if "structured_output" in capabilities else CapabilityState.UNSUPPORTED,
            supports_grammar=CapabilityState.UNSUPPORTED,
            supports_streaming=CapabilityState.PROBED,
            supports_images=CapabilityState.DECLARED if declared_images else CapabilityState.UNSUPPORTED,
            serving_config_digest=_digest({"loaded": loaded, "template": template}),
            probed_at=datetime.now(UTC).isoformat(),
        )

    async def qualify_behavior(
        self, route: ModelRoute, transport: UpstreamTransport,
        metadata: ModelRuntimeCapabilities | None = None,
    ) -> ModelRuntimeCapabilities:
        """Run behavioral qualification probes (P1.3).

        Actual generations that probe tool/JSON/schema behavior. Only run
        when behavioral qualification is needed (not on every request).
        Digest-cached so repeated inspections don't re-run probes.
        """
        # Check cache first
        if metadata is not None:
            cache_key = f"{metadata.model_digest}:{metadata.chat_template_digest}"
            cached = self._qualification_cache.get(cache_key)
            if cached is not None:
                return cached
        else:
            cache_key = None

        probe_options: dict[str, Any] = {"temperature": 0}
        if route.upstream.ollama_num_ctx:
            probe_options["num_ctx"] = route.upstream.ollama_num_ctx
        probe_base = {
            "model": route.upstream_model,
            "stream": False,
            "options": probe_options,
        }
        synthetic_tool = {
            "type": "function",
            "function": {
                "name": "interop_probe",
                "description": "Return marker only; no side effects.",
                "parameters": {
                    "type": "object",
                    "properties": {"marker": {"type": "string"}},
                    "required": ["marker"],
                },
            },
        }
        tools_ok, tool_payload = await self._probe(transport, route, {
            **probe_base,
            "messages": [{"role": "user", "content": "Call interop_probe with marker runtime."}],
            "tools": [synthetic_tool],
        })
        json_ok, _ = await self._probe(transport, route, {
            **probe_base,
            "messages": [{"role": "user", "content": "Return exactly an empty JSON object."}],
            "format": "json",
        })
        schema_ok, _ = await self._probe(transport, route, {
            **probe_base,
            "messages": [{"role": "user", "content": "Return a JSON object with marker runtime."}],
            "format": {
                "type": "object",
                "properties": {"marker": {"type": "string"}},
                "required": ["marker"],
            },
        })
        tool_calls = tool_payload.get("message", {}).get("tool_calls", [])
        result = ModelRuntimeCapabilities(
            backend_kind=route.upstream.kind,
            backend_version="",
            model_name=route.upstream_model,
            model_digest="",
            architecture="",
            quantization="",
            parameter_count="",
            architecture_context_tokens=0,
            configured_context_tokens=0,
            effective_context_tokens=0,
            chat_template="",
            chat_template_digest="",
            accepts_native_tools=(CapabilityState.PROBED if tools_ok else CapabilityState.UNSUPPORTED),
            returns_native_tool_calls=(CapabilityState.PROBED if isinstance(tool_calls, list) and tool_calls else CapabilityState.UNSUPPORTED),
            accepts_named_tool_choice=CapabilityState.UNSUPPORTED,
            accepts_required_tool_choice=CapabilityState.UNSUPPORTED,
            accepts_parallel_tool_flag=CapabilityState.UNSUPPORTED,
            supports_json_schema=(CapabilityState.PROBED if schema_ok else CapabilityState.UNSUPPORTED),
            supports_json_mode=(CapabilityState.PROBED if json_ok else CapabilityState.UNSUPPORTED),
            supports_grammar=CapabilityState.UNSUPPORTED,
            supports_streaming=CapabilityState.PROBED,
            supports_images=CapabilityState.UNSUPPORTED,
            serving_config_digest="",
            probed_at=datetime.now(UTC).isoformat(),
        )
        if cache_key is not None:
            self._qualification_cache[cache_key] = result
        return result

    async def inspect(
        self, route: ModelRoute, transport: UpstreamTransport,
    ) -> ModelRuntimeCapabilities:
        """P0.36: DEPRECATED — use inspect_runtime_metadata instead.

        This legacy method runs metadata + behavioral qualification
        (3 model generations). It is only called as a fallback for
        inspectors that don't implement inspect_runtime_metadata.
        """
        import warnings
        warnings.warn(
            "OllamaInspector.inspect() is deprecated. "
            "Use inspect_runtime_metadata() for metadata-only inspection.",
            DeprecationWarning,
            stacklevel=2,
        )
        metadata = await self.inspect_runtime_metadata(route, transport)
        behavior = await self.qualify_behavior(route, transport, metadata=metadata)
        # Merge: metadata provides identity, behavior provides probed capabilities
        return replace(metadata,
            accepts_native_tools=behavior.accepts_native_tools,
            returns_native_tool_calls=behavior.returns_native_tool_calls,
            supports_json_schema=behavior.supports_json_schema,
            supports_json_mode=behavior.supports_json_mode,
            supports_streaming=behavior.supports_streaming,
        )
