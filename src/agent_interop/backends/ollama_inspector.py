"""Ollama runtime inspection through Interop's shared transport."""

from __future__ import annotations

import hashlib
import json
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
        for key in ("num_ctx", "context_length", "context_window", "num_ctx_train"):
            value = source.get(key)
            if isinstance(value, int) and value > 0:
                return value
            if isinstance(value, str) and value.isdigit():
                return int(value)
    return 0


class OllamaInspector:
    """Read Ollama's model/runtime metadata without creating an httpx client."""

    async def _request(
        self, transport: UpstreamTransport, route: ModelRoute, method: str, path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = await transport.send(PreparedUpstreamRequest(
            method=method,
            url=f"{route.upstream.base_url.rstrip('/')}{path}",
            headers=dict(route.upstream.static_headers),
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

    async def inspect(
        self, route: ModelRoute, transport: UpstreamTransport,
    ) -> ModelRuntimeCapabilities:
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
        configured_limit = _context_from_options(loaded, loaded.get("details", {}) if isinstance(loaded.get("details"), dict) else {})
        effective = min((limit for limit in (architecture_limit, configured_limit) if limit > 0), default=architecture_limit or configured_limit)
        declared_tools = "tools" in capabilities
        declared_images = "vision" in capabilities or "images" in capabilities
        return ModelRuntimeCapabilities(
            backend_kind=route.upstream.kind,
            backend_version=str(version.get("version", "")),
            model_name=route.upstream_model,
            model_digest=str(tag.get("digest") or shown.get("digest") or loaded.get("digest") or ""),
            architecture=str(details.get("family") or details.get("families", [""])[0] if details.get("families") else ""),
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
