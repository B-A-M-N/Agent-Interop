"""P0-24/25/27 regression: selective stream buffering + ref pinning.

Buffering is a TTFT tax, so the policy must be selective:
  private capabilities active  -> ALWAYS buffer (already-streamed text
                                  cannot be withdrawn);
  no tools                     -> never buffer;
  tool_choice=none, no caps    -> never buffer;
  unverified tool-bearing      -> buffer until validated;
  verified native + evidence   -> stream immediately.

And streaming pins request refs across the whole generator lifecycle,
unpinning even on cancellation.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.context_store.store import ContextStore
from agent_interop.gateway import Gateway


def _invocation(
    *,
    tools: bool = True,
    tool_choice_none: bool = False,
    private_caps: bool = False,
    buffered_flag: bool = True,
    path: str = "adapted",
    evidence: Any = None,
    effective_mode: Any = None,
) -> Any:
    from agent_interop.config import ToolMode as TM

    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        tools=[CanonicalTool(name="t", description="d", input_schema={"type": "object"})] if tools else [],
        tool_choice=CanonicalToolChoice.none() if tool_choice_none else CanonicalToolChoice.auto(),
    )
    caps = None
    if private_caps:
        from agent_interop.projection.types import PrivateCapabilityPlan

        caps = PrivateCapabilityPlan(read_result=True)
    return SimpleNamespace(
        authoritative_request=request,
        reconciled_request=request,
        route=SimpleNamespace(compatibility=SimpleNamespace(buffer_unverified_streaming=buffered_flag)),
        private_capabilities=caps,
        invocation_plan=SimpleNamespace(effective_tool_mode=effective_mode or TM.NATIVE),
        compatibility_plan=SimpleNamespace(path=SimpleNamespace(value=path)),
        evidence_record=evidence,
    )


def test_private_capabilities_always_buffer():
    """P0-25: even a verified direct native stream buffers when private
    retrieval is active — streamed text cannot be un-sent before a private
    continuation consumes it."""
    invocation = _invocation(
        private_caps=True, path="direct",
        evidence=object(), effective_mode=ToolMode.NATIVE,
    )
    assert Gateway._requires_buffered_stream_validation(invocation) is True


def test_tool_free_stream_never_buffers():
    invocation = _invocation(tools=False, path="adapted", evidence=None)
    assert Gateway._requires_buffered_stream_validation(invocation) is False


def test_tool_choice_none_never_buffers_without_caps():
    invocation = _invocation(tool_choice_none=True, path="adapted", evidence=None)
    assert Gateway._requires_buffered_stream_validation(invocation) is False


def test_unverified_tool_stream_buffers():
    invocation = _invocation()
    assert Gateway._requires_buffered_stream_validation(invocation) is True


def test_verified_native_direct_streams_immediately():
    invocation = _invocation(
        path="direct", evidence=object(), effective_mode=ToolMode.NATIVE,
    )
    assert Gateway._requires_buffered_stream_validation(invocation) is False


def test_flag_disabled_short_circuits_everything():
    invocation = _invocation(private_caps=True, buffered_flag=False)
    assert Gateway._requires_buffered_stream_validation(invocation) is False


# ─── P0-27: streaming ref pinning lifecycle ─────────────────────────────────


class _UnpinTrackingStore(ContextStore):
    def __init__(self) -> None:
        super().__init__()
        self.pins: dict[str, int] = {}
        self.unpins: dict[str, int] = {}
        self.batch_pin_calls: int = 0

    def pin_ref(self, ref: str, request_id: str) -> None:
        self.pins[ref] = self.pins.get(ref, 0) + 1
        super().pin_ref(ref, request_id)

    def unpin_ref(self, ref: str, request_id: str) -> None:
        self.unpins[ref] = self.unpins.get(ref, 0) + 1
        super().unpin_ref(ref, request_id)

    # P1-G: the gateway pins the whole set in ONE batched call.
    def pin_refs(self, refs, request_id: str, session_id: str | None = None) -> int:
        self.batch_pin_calls += 1
        for ref in refs:
            self.pins[ref] = self.pins.get(ref, 0) + 1
        return super().pin_refs(refs, request_id, session_id=session_id)

    def unpin_refs(self, refs, request_id: str, session_id: str | None = None) -> int:
        for ref in refs:
            self.unpins[ref] = self.unpins.get(ref, 0) + 1
        return super().unpin_refs(refs, request_id, session_id=session_id)


def test_streaming_pins_and_unpins_refs():
    """A streaming request whose projection virtualized refs must pin them
    for the whole generator and unpin when the stream completes."""
    from agent_interop.abi import (
        CanonicalGenerationOptions,
        CanonicalToolCallBlock,
        CanonicalToolResultBlock,
    )

    route = ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        context=ContextConfig(context_limit_tokens=2000, output_reserve_tokens=500),
    )
    store = _UnpinTrackingStore()
    gw = Gateway(InteropServerConfig(
        probe_on_startup=False, log_level="error", routes={"r": route},
    ))
    gw._context_store = store
    gw._internal_executor = __import__(
        "agent_interop.context_store.executor", fromlist=["InternalToolExecutor"],
    ).InternalToolExecutor(store)

    big = "\n".join(f"line {i}" for i in range(500))
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=256, stream=True),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="old")]),
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(
                id="old_call", name="tool_0", arguments={"x": "v"},
            )]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(
                tool_call_id="old_call", content=big,
            )]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="do work")]),
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(
                id="c1", name="tool_0", arguments={"x": "v"},
            )]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(
                tool_call_id="c1", content="current",
            )]),
        ],
        tools=[CanonicalTool(name="tool_0", description="d", input_schema={"type": "object"})],
        tool_choice=CanonicalToolChoice.auto(),
    )

    class _T:
        async def send(self, request):  # pragma: no cover
            raise NotImplementedError()

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def stream(self, request, raise_on=()):
            import json as _json

            from agent_interop.transport.sse import SSEFrame

            class _S:
                status_code = 200

                async def sse_events(self):
                    yield SSEFrame(data=_json.dumps({
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

    gw._transport = _T()

    async def run():
        return [e async for e in gw.handle_stream(request, RequestContext(session_id="s-pin"))]

    events = asyncio.run(run())
    assert any(e.type == "message_stop" for e in events), [e.type for e in events]
    assert store.pins, "streaming must pin request refs"
    assert store.pins.keys() == store.unpins.keys(), (
        "every pinned ref must be unpinned on stream completion"
    )
