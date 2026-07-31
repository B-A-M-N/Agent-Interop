"""Conservative runtime inspection for OpenAI-compatible backends."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

from agent_interop.backends.base import ModelRuntimeCapabilities
from agent_interop.capabilities import CapabilityState
from agent_interop.config import ModelRoute
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamTransport


class OpenAICompatibleInspector:
    """Inspect ``/v1/models`` without inferring model behaviour from HTTP API.

    OpenAI-compatible servers expose wildly different extensions.  The model
    list is useful identity metadata; all feature flags intentionally stay
    unknown until targeted qualification/evidence proves them.
    """

    async def inspect(self, route: ModelRoute, transport: UpstreamTransport) -> ModelRuntimeCapabilities:
        response = await transport.send(PreparedUpstreamRequest(
            method="GET",
            url=f"{route.upstream.base_url.rstrip('/')}/v1/models",
            headers=dict(route.upstream.static_headers),
            stream=False,
            timeout_seconds=min(route.upstream.timeout_seconds, 15.0),
        ))
        body: dict = {}
        if not response.transport_failed and response.status_code < 400:
            try:
                candidate = response.json()
                body = candidate if isinstance(candidate, dict) else {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        model = next((item for item in body.get("data", []) if item.get("id") == route.upstream_model), {})
        digest = str(model.get("digest") or model.get("id") or "")
        serving = hashlib.sha256(json.dumps(model, sort_keys=True, default=str).encode()).hexdigest()[:16] if model else ""
        return ModelRuntimeCapabilities(
            backend_kind=route.upstream.kind,
            model_name=route.upstream_model,
            model_digest=digest,
            supports_streaming=CapabilityState.DECLARED if model else CapabilityState.UNSUPPORTED,
            serving_config_digest=serving,
            probed_at=datetime.now(UTC).isoformat(),
        )
