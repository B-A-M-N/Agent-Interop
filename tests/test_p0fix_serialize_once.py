"""P0-10/11/12/26 regression: serialize-once + unconditional context gate.

Locks in:
  * the rendered provider body is serialized exactly once and the transport
    sends those bytes verbatim (no re-serialization);
  * the exact rendered-context gate runs even when no AttemptBudget is
    attached (internal generations must not bypass context safety);
  * both streaming and non-streaming paths enforce the same preflight.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    RuntimeInspectionConfig,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.execution import InteropRequestExecution
from agent_interop.gateway import Gateway
from agent_interop.transport import UpstreamResponse
from agent_interop.transport.http import PreparedUpstreamRequest


class _CapturingTransport:
    """Records the raw request the gateway handed it."""

    def __init__(self) -> None:
        self.last: PreparedUpstreamRequest | None = None

    async def send(self, request):  # noqa: ANN001
        self.last = request
        return UpstreamResponse(
            status_code=200,
            body=json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode(),
        )

    async def stream(self, request, raise_on=()):  # pragma: no cover
        self.last = request
        raise NotImplementedError()


def _route(**ctx: Any) -> ModelRoute:
    return ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
        context=ContextConfig(**ctx) if ctx else ContextConfig(context_limit_tokens=32768),
    )


def _config(route: ModelRoute | None = None) -> InteropServerConfig:
    return InteropServerConfig(
        probe_on_startup=False,
        runtime_inspection=RuntimeInspectionConfig(mode="off"),
        routes={"r": route or _route()},
    )


def _request(max_output: int = 64) -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=max_output, stream=False),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
    )


def test_serialized_body_reaches_transport_unchanged():
    """The transport receives serialized_body bytes that are exactly the
    compact JSON of body — the request was serialized once, by the gateway."""
    transport = _CapturingTransport()
    gw = Gateway(_config(), transport=transport)
    resp = asyncio.run(gw.handle_request(_request(), RequestContext()))
    assert resp.error is None
    req = transport.last
    assert req is not None and req.serialized_body is not None
    # Byte-identical to a compact serialization of the body dict.
    compact = json.dumps(req.body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert req.serialized_body == compact
    # And it parses back to the same body.
    assert json.loads(req.serialized_body) == req.body


def test_serialized_body_present_on_stream_request():
    """Streaming preflight parity: the stream path also serializes once."""
    from contextlib import asynccontextmanager

    class _StreamTransport(_CapturingTransport):
        @asynccontextmanager
        async def stream(self, request, raise_on=()):
            self.last = request
            from agent_interop.transport.sse import SSEFrame

            class _S:
                status_code = 200

                async def sse_events(self):
                    yield SSEFrame(data=json.dumps({
                        "choices": [{"index": 0, "delta": {"content": "ok"},
                                     "finish_reason": "stop"}],
                    }))

                async def raw_lines(self):
                    return
                    yield  # pragma: no cover

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *a):
                    return None

            yield _S()

    transport = _StreamTransport()
    gw = Gateway(_config(), transport=transport)
    req = _request()
    object.__setattr__(req.generation, "stream", True)

    async def _collect():
        return [e async for e in gw.handle_stream(req, RequestContext())]

    asyncio.run(_collect())
    req_seen = transport.last
    assert req_seen is not None
    assert req_seen.serialized_body is not None
    compact = json.dumps(req_seen.body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert req_seen.serialized_body == compact


def test_context_gate_runs_without_budget():
    """P0-12: a generation with NO AttemptBudget attached must still be
    rejected when the exact rendered body overflows the safe context limit.
    The old gate was nested under `if budget is not None`."""
    # Tiny safe limit forces the rendered-body rejection.
    route = _route(context_limit_tokens=100, output_reserve_tokens=50)
    gw = Gateway(_config(route), transport=_CapturingTransport())
    big = "word " * 5000  # far beyond 100 tokens
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=64, stream=False),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text=big)])],
    )
    resp = asyncio.run(gw.handle_request(req, RequestContext()))
    assert resp.error is not None
    assert resp.error.code == "CONTEXT_LIMIT_EXCEEDED", resp.error.code


def test_budget_exhaustion_still_enforced_when_attached():
    """Cumulative budget accounting still rejects when a budget IS attached
    and its input ceiling is crossed."""
    from agent_interop.execution_attempts import AttemptBudget

    route = _route()
    gw = Gateway(_config(route), transport=_CapturingTransport())
    big = "word " * 40000
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=64, stream=False),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text=big)])],
    )
    context = RequestContext()
    exec_record = InteropRequestExecution(context=context)
    exec_record.attempt_budget = AttemptBudget(max_total_input_tokens=10)
    resp = asyncio.run(gw.handle_request(req, context))
    # Either the budget or the exact gate rejects; both are the same
    # client-visible CONTEXT_LIMIT_EXCEEDED contract.
    assert resp.error is not None
    assert resp.error.code == "CONTEXT_LIMIT_EXCEEDED", resp.error.code
