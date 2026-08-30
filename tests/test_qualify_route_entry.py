"""Gateway.qualify_route — the public route-qualification entry point.

Regression: the dee1a97 god-object extraction moved probe execution into
the qualification coordinator but left ``interop qualify`` and
``InteropRuntime.qualify`` calling ``gateway.qualify_route(...)``, a
method that no longer existed anywhere. Both raised AttributeError at
runtime. These tests pin the restored contract:

1. qualify_route runs the synthetic battery against a scripted backend
   and records the resulting evidence under the runtime's digest key.
2. An unknown model fails honestly with MODEL_NOT_FOUND instead of
   silently qualifying nothing.
"""

from __future__ import annotations

import asyncio
import json

from agent_interop.backends.base import ModelRuntimeCapabilities
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.gateway import Gateway
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamResponse


def _route() -> ModelRoute:
    return ModelRoute(
        id="qual",
        client_model_aliases=["qual-model"],
        upstream_model="qual-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        context=ContextConfig(context_limit_tokens=4096, output_reserve_tokens=256),
    )


def _completion(content: str, tool_calls: list[dict] | None = None) -> dict:
    message: dict = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "resp",
        "object": "chat.completion",
        "model": "qual-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


class _ProbeTransport:
    """Answers each battery probe in the way the probe contract expects."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        # Inspector traffic (runtime metadata) has no body; generation
        # probes carry the serialized rendered request.
        if request.serialized_body is None:
            if request.url.endswith("/v1/models"):
                payload = {"data": [{"id": "qual-model", "digest": "sha256:qual"}]}
            else:
                raise AssertionError(f"unexpected bodyless request: {request.url}")
            return UpstreamResponse(status_code=200, body=json.dumps(payload).encode())
        body = json.loads(request.serialized_body)
        self.bodies.append(body)
        prompt = body["messages"][-1]["content"]
        if "INTEROP_PROBE_OK" in prompt:
            payload = _completion("INTEROP_PROBE_OK")
        elif "marker native" in prompt:
            payload = _completion(None, tool_calls=[{
                "id": "call_n", "type": "function",
                "function": {"name": "interop_probe", "arguments": '{"marker": "native"}'},
            }])
        elif "marker prompted" in prompt:
            payload = _completion(None, tool_calls=[{
                "id": "call_p", "type": "function",
                "function": {"name": "interop_probe", "arguments": '{"marker": "prompted"}'},
            }])
        elif "no tool needed" in prompt:
            payload = _completion("no tool needed")
        elif "marker=done" in prompt:
            payload = _completion("continued")
        else:
            payload = _completion("INTEROP_PROBE_OK")
        return UpstreamResponse(status_code=200, body=json.dumps(payload).encode())


def _runtime() -> ModelRuntimeCapabilities:
    return ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OPENAI_COMPATIBLE,
        model_name="qual-model",
        model_digest="sha256:qual",
        configured_context_tokens=4096,
    )


def test_qualify_route_runs_battery_and_records_evidence():
    transport = _ProbeTransport()
    gw = Gateway(
        InteropServerConfig(probe_on_startup=False, log_level="error", routes={"qual": _route()}),
        transport=transport,
    )
    runtime = _runtime()

    record = asyncio.run(gw.qualify_route(runtime.model_digest, runtime))

    # Staged battery: native forced tool first (the model complied), then
    # continuation evidence because qualify_route demands it. Native PASS
    # ends staging — the prompted probe is only exercised after a native
    # failure. The contract under test is that probes actually ran through
    # the real machinery and evidence was recorded, not a probe count.
    assert len(transport.bodies) >= 2
    prompts = [b["messages"][-1]["content"] for b in transport.bodies]
    assert any("marker native" in p for p in prompts)
    assert gw._qualification_record_for_runtime(runtime) is record
    assert record.model_digest == "sha256:qual"


def test_qualify_route_unknown_model_fails_honestly():
    gw = Gateway(
        InteropServerConfig(probe_on_startup=False, log_level="error", routes={}),
        allow_invalid_config=True,
    )
    runtime = ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OPENAI_COMPATIBLE, model_name="ghost",
    )
    from agent_interop.errors import InteropError

    try:
        asyncio.run(gw.qualify_route("", runtime))
    except InteropError as exc:
        assert "MODEL_NOT_FOUND" in str(exc)
    else:
        raise AssertionError("qualify_route silently qualified a model with no route")
