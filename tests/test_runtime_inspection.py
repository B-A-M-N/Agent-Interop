"""Runtime-inspection tests for metadata and bounded feature probes."""

from __future__ import annotations

import asyncio
import json

from agent_interop.backends.ollama_inspector import OllamaInspector
from agent_interop.capabilities import CapabilityState
from agent_interop.config import ModelRoute, UpstreamConfig, UpstreamKind, UpstreamProtocol
from agent_interop.gateway import Gateway
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamResponse


class _Transport:
    def __init__(self) -> None:
        self.requests: list[PreparedUpstreamRequest] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        self.requests.append(request)
        if request.url.endswith("/api/version"):
            body = {"version": "0.12.0"}
        elif request.url.endswith("/api/tags"):
            body = {"models": [{"name": "model:latest", "digest": "sha256:model"}]}
        elif request.url.endswith("/api/show"):
            body = {
                "digest": "sha256:model",
                "capabilities": ["tools", "structured_output"],
                "template": "{{ .Messages }}",
                "details": {"family": "qwen", "quantization_level": "Q4", "parameter_size": "7B"},
                "model_info": {"context_length": 8192},
            }
        elif request.url.endswith("/api/ps"):
            body = {"models": [{"name": "model:latest", "details": {"num_ctx": 4096}}]}
        elif request.url.endswith("/api/chat"):
            body = {"message": {"tool_calls": [{"function": {"name": "interop_probe"}}]}}
        else:  # pragma: no cover - makes unexpected inspector calls obvious
            raise AssertionError(request.url)
        return UpstreamResponse(status_code=200, body=json.dumps(body).encode())


def test_ollama_inspection_uses_shared_transport_for_feature_probes() -> None:
    route = ModelRoute(
        id="local",
        client_model_aliases=["local"],
        upstream_model="model:latest",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://ollama.test",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
            static_headers={"X-Interop-Test": "enabled"},
        ),
    )
    transport = _Transport()
    runtime = asyncio.run(OllamaInspector().inspect(route, transport))

    assert runtime.model_digest == "sha256:model"
    assert runtime.effective_context_tokens == 4096
    assert runtime.accepts_native_tools is CapabilityState.PROBED
    assert runtime.returns_native_tool_calls is CapabilityState.PROBED
    assert runtime.supports_json_mode is CapabilityState.PROBED
    assert runtime.supports_json_schema is CapabilityState.PROBED
    chat_probes = [request for request in transport.requests if request.url.endswith("/api/chat")]
    assert len(chat_probes) == 3
    assert all(request.headers["x-interop-test"] == "enabled" for request in transport.requests)


def test_ollama_context_setting_is_sent_to_probes_and_runtime_requests() -> None:
    route = ModelRoute(
        id="local",
        client_model_aliases=["local"],
        upstream_model="model:latest",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://ollama.test",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
            ollama_num_ctx=16384,
        ),
    )
    transport = _Transport()
    asyncio.run(OllamaInspector().inspect(route, transport))
    chat_probes = [request for request in transport.requests if request.url.endswith("/api/chat")]
    assert all(request.body["options"]["num_ctx"] == 16384 for request in chat_probes)

    rendered = {"options": {"temperature": 0}}
    Gateway._apply_route_runtime_options(rendered, route)
    assert rendered["options"]["num_ctx"] == 16384
