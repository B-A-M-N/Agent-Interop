"""P0.1/P0.2/P0.4 regression: the private __interop_* continuation loop.

These exercise the *production* gateway path (handle_request → private loop),
not just the isolated InternalToolExecutor, proving:

  * model emits __interop_read_result → Interop executes it privately →
    model emits a public tool call → client sees ONLY the public call;
  * a model that keeps requesting internal retrieval past the loop cap yields
    INTERNAL_TOOL_LOOP_EXHAUSTED and never leaks internal content;
  * a mixed step (internal + public in one model turn) suppresses the public
    call until the internal retrieval completes (P0.4).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    ContextConfig,
)
from agent_interop.context import RequestContext
from agent_interop.execution import InteropRequestExecution
from agent_interop.gateway import Gateway
from agent_interop.private_loop import MAX_INTERNAL_TOOL_LOOP_DEPTH
from agent_interop.transport import UpstreamResponse

# Context sizing that triggers deterministic virtualization of an old large
# tool result (mirrors test_private_tools_conditional_with_virtualization):
# private execution authority is request-scoped (P0-21), so a test that
# scripts a __interop_read_result call MUST make its own request virtualize
# state — read_result capability is granted only by this request's projection.
_VirtualizingContextSize = {"context_limit_tokens": 2000, "output_reserve_tokens": 500}


def _openai_chat_completion(tool_calls: list[dict[str, Any]], finish: str = "tool_calls") -> bytes:
    return json.dumps({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            },
            "finish_reason": finish,
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }).encode()


class _ScriptedTransport:
    """Returns scripted OpenAI-chat completions, counting real upstream calls."""

    def __init__(self, responses: list[bytes]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def send(self, request):  # noqa: ANN001 - matches TransportProtocol
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return UpstreamResponse(status_code=200, body=self._responses[idx])

    async def stream(self, request, raise_on=()):  # pragma: no cover - unused here
        raise NotImplementedError()


def _make_gateway(tool_mode: ToolMode = ToolMode.NATIVE):
    route = ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=tool_mode,
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
        # Tight-but-sufficient capacity: virtualizes the old large tool result
        # in _virtualizing_request() so the request grants its own read_result
        # capability (P0-21), while these tests exercise the private
        # continuation loop, not the unknown-capacity reject policy.
        context=ContextConfig(**_VirtualizingContextSize),
    )
    config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
    return Gateway(config)


def _session() -> RequestContext:
    # Internal tools execute against the caller's own ContextStore session
    # (the store fails closed across sessions), so scripted private calls
    # need the same session_id the pre-stored ref was stored under.
    return RequestContext(session_id="sess-private-loop")


_BIG_OLD_RESULT = "\n".join(f"line {i}" for i in range(500))  # ~3000 bytes


def _virtualizing_request(tools: int, session_seed: str) -> CanonicalRequest:
    """A request whose history virtualizes deterministically.

    Layout mirrors test_private_tools_conditional_with_virtualization: an OLD
    oversized tool result that compaction stores into the ContextStore, and a
    recent small exchange that stays protected. The returned request carries
    read_result capability on its own projection.
    """
    from agent_interop.abi import CanonicalGenerationOptions, CanonicalToolResultBlock

    big = _BIG_OLD_RESULT
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        # Small explicit output budget: the rendered-body preflight reserves
        # min(1024, max(64, max_output_tokens)) against the safe limit, and the
        # dataclass default (4096 → clamped 1024) would overflow it.
        generation=CanonicalGenerationOptions(max_output_tokens=256, stream=False),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text=f"old task {session_seed}")]),
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
        tools=_client_tools(tools),
        tool_choice=CanonicalToolChoice.auto(),
    )


def _client_tools(n: int) -> list[CanonicalTool]:
    return [
        CanonicalTool(
            name=f"tool_{i}",
            description=f"tool {i}",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        )
        for i in range(n)
    ]


def test_private_non_stream_continuation_client_sees_only_public_call():
    gw = _make_gateway()
    # Pre-store a blob the model will privately retrieve, in the SAME session
    # the request runs under (the store fails closed across sessions).
    ref = gw._context_store.store(
        "sess-private-loop", "secret line one\nsecret line two\n", kind="tool_result", tool_call_id="R1",
    ).ref
    internal_tc = [{
        "id": "call_int",
        "type": "function",
        "function": {"name": "__interop_read_result", "arguments": json.dumps({"ref": ref, "start_line": 1})},
    }]
    public_tc = [{
        "id": "call_pub",
        "type": "function",
        "function": {"name": "tool_0", "arguments": json.dumps({"x": "hello"})},
    }]
    transport = _ScriptedTransport([
        _openai_chat_completion(internal_tc),
        _openai_chat_completion(public_tc),
    ])
    gw._transport = transport

    # The request itself must virtualize state — read_result capability is
    # granted by THIS request's projection (P0-21), never globally.
    req = _virtualizing_request(4, "seed-1")
    resp = asyncio.run(gw.handle_request(req, _session()))

    # A second upstream call occurred (private continuation happened).
    assert transport.calls == 2, transport.calls
    # Client sees ONLY the public tool call — no internal call, no internal result.
    names = [b.name for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
    assert "__interop_read_result" not in names, names
    assert "tool_0" in names, names
    # No internal ref leaked into any text block.
    for b in resp.content:
        if isinstance(b, CanonicalTextBlock) and b.text:
            assert "[Interop result ref" not in b.text


def test_private_loop_exhaustion_returns_exhausted_and_no_internal_leak():
    gw = _make_gateway()
    ref = gw._context_store.store(
        "sess-private-loop", "data\n", kind="tool_result", tool_call_id="R2",
    ).ref
    # Every scripted call keeps requesting the internal tool.
    internal_tc = [{
        "id": "call_int",
        "type": "function",
        "function": {"name": "__interop_read_result", "arguments": json.dumps({"ref": ref})},
    }]
    many = [_openai_chat_completion(internal_tc)] * (MAX_INTERNAL_TOOL_LOOP_DEPTH + 2)
    transport = _ScriptedTransport(many)
    gw._transport = transport

    req = _virtualizing_request(2, "seed-2")
    resp = asyncio.run(gw.handle_request(req, _session()))
    assert resp.error is not None
    # The loop cap (or the generation budget guarding it) stops the model —
    # either way the client gets INTERNAL_TOOL_LOOP_EXHAUSTED, never a
    # fall-through to the public transaction layer.
    assert resp.error.code == "INTERNAL_TOOL_LOOP_EXHAUSTED", resp.error.code
    # No internal block reached the client.
    names = [b.name for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
    assert "__interop_read_result" not in names


def test_mixed_private_public_suppresses_public_call_until_retrieval():
    gw = _make_gateway()
    ref = gw._context_store.store(
        "sess-private-loop", "private data\n", kind="tool_result", tool_call_id="R3",
    ).ref
    # One model turn emits BOTH an internal read and a public client call.
    mixed = [
        {"id": "call_int", "type": "function",
         "function": {"name": "__interop_read_result", "arguments": json.dumps({"ref": ref})}},
        {"id": "call_pub", "type": "function",
         "function": {"name": "tool_1", "arguments": json.dumps({"x": "now"})}},
    ]
    # Second turn: only the public call (after the model has seen retrieved ctx).
    public_tc = [{
        "id": "call_pub2", "type": "function",
        "function": {"name": "tool_1", "arguments": json.dumps({"x": "later"})},
    }]
    transport = _ScriptedTransport([
        _openai_chat_completion(mixed),
        _openai_chat_completion(public_tc),
    ])
    gw._transport = transport

    req = _virtualizing_request(4, "seed-3")
    resp = asyncio.run(gw.handle_request(req, _session()))
    # The first public call was suppressed; the second (post-retrieval) call is what surfaces.
    names = [b.name for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
    assert "__interop_read_result" not in names
    assert "tool_1" in names
    assert transport.calls == 2


# ─── Streaming (P0.2) ───────────────────────────────────────────────────────


class _ScriptedStreamTransport:
    """Serves the first (streamed) turn via ``stream`` and the private-loop
    continuation via ``send`` (which the non-streaming loop uses)."""

    def __init__(self, first_sse: list[bytes], continuation_responses: list[bytes]) -> None:
        self._first_sse = first_sse
        self._continuation = list(continuation_responses)
        self.send_calls = 0

    @asynccontextmanager
    async def stream(self, request):  # noqa: ANN001
        from agent_interop.transport.sse import SSEFrame

        class _S:
            status_code = 200

            async def sse_events(self):
                for line in self._lines:
                    yield SSEFrame(data=line.decode())

            async def raw_lines(self):
                return
                yield  # pragma: no cover

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

        s = _S()
        s._lines = self._first_sse
        yield s

    async def send(self, request):  # noqa: ANN001
        idx = min(self.send_calls, len(self._continuation) - 1)
        self.send_calls += 1
        return UpstreamResponse(status_code=200, body=self._continuation[idx])


def _sse_tool_call(tool_calls: list[dict[str, Any]]) -> bytes:
    return json.dumps({
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }],
    }).encode()


def test_stream_private_continuation_no_internal_leak():
    from agent_interop.abi import CanonicalEvent

    gw = _make_gateway()
    ref = gw._context_store.store(
        "sess-private-loop", "streamed secret\n", kind="tool_result", tool_call_id="RS",
    ).ref
    # Streamed first turn emits an internal __interop_read_result call.
    first_sse = [_sse_tool_call([{
        "index": 0, "id": "call_int", "type": "function",
        "function": {"name": "__interop_read_result", "arguments": json.dumps({"ref": ref})},
    }])]
    # The private continuation (via _handle_request_send) emits a public call.
    public_tc = [{
        "id": "call_pub", "type": "function",
        "function": {"name": "tool_0", "arguments": json.dumps({"x": "done"})},
    }]
    transport = _ScriptedStreamTransport(first_sse, [_openai_chat_completion(public_tc)])
    gw._transport = transport

    req = _virtualizing_request(4, "seed-s1")

    async def _collect():
        events: list[CanonicalEvent] = []
        async for ev in gw.handle_stream(req, _session()):
            events.append(ev)
        return events

    events = asyncio.run(_collect())

    # No internal tool_use event reaches the client.
    tool_use_names = [e.content_block.name for e in events if e.type == "tool_use"]
    assert "__interop_read_result" not in tool_use_names, tool_use_names
    # The public call does surface.
    assert "tool_0" in tool_use_names, tool_use_names
    # No internal ref text leaked into text deltas.
    text = "".join(getattr(e, "partial", "") for e in events if e.type == "text_delta")
    assert "[Interop result ref" not in text
    # The continuation used the send() path (private loop executed).
    assert transport.send_calls >= 1


# ─── P1.1: TTL lifecycle wired into request path ─────────────────────────────


def test_ttl_eviction_runs_on_request_lifecycle():
    """A stored entry older than ttl_seconds must be evicted when a new request
    triggers the lifecycle cleanup - proving evict_expired() is actually wired,
    not dead code."""
    from agent_interop.config import ResourceConfig

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
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
        context=ContextConfig(context_limit_tokens=32768),
    )
    # Tiny TTL so a short sleep expires stored entries.
    config = InteropServerConfig(
        routes={"r": route}, probe_on_startup=False, log_level="error",
        resources=ResourceConfig(ttl_seconds=1),
    )
    gw = Gateway(config)
    ref = gw._context_store.store(
        "sess-ttl", "expiring\n", kind="tool_result", tool_call_id="RT",
    ).ref
    # Entry is present immediately.
    assert gw._context_store.get(ref, "sess-ttl") is not None

    # Let it expire, then run any request — the lifecycle hook must evict it.
    import time
    time.sleep(1.1)
    transport = _ScriptedTransport([
        _openai_chat_completion([{
            "id": "c1", "type": "function",
            "function": {"name": "tool_0", "arguments": json.dumps({"x": "ping"})},
        }]),
    ])
    gw._transport = transport
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        tools=_client_tools(4),
        tool_choice=CanonicalToolChoice.auto(),
    )
    asyncio.run(gw.handle_request(req, RequestContext()))

    # Entry is gone after the request-driven eviction.
    assert gw._context_store.get(ref, "sess-ttl") is None
    # Store is safe to keep using.
    assert gw._context_store.evict_expired() == 0


# ─── P0.40/P1.6: ModelView reflects the actual final request ────────────────


def test_model_view_propagates_virtualization_flags():
    """P0.40/P1.6: ModelProjector.build_model_view() must propagate the
    actual transformation flags (results_virtualized, virtualized_refs_count)
    into the returned ModelView — the old inline construction always
    hard-coded these to False/0."""
    from agent_interop.projection.planner import ModelProjector

    view = ModelProjector.build_model_view(
        authorized_tool_count=20,
        visible_client_tool_names=("tool_0", "tool_1"),
        visible_private_tool_names=("__interop_read_result",),
        results_virtualized=True,
        virtualized_refs_count=3,
    )
    # The view describes what was actually done — not the stale defaults.
    assert view.results_virtualized is True
    assert view.virtualized_refs_count == 3
    # Authoritative registry is the full client set used for validation.
    assert view.authorized_tool_count == 20
    # Visible surface is the narrowed client tools (private tracked separately).
    assert view.visible_tool_count == 2
    assert view.visible_tool_names == ("tool_0", "tool_1")
    assert view.visible_client_tool_names == ("tool_0", "tool_1")
    assert view.visible_private_tool_names == ("__interop_read_result",)


# ─── P1.3: structured tool-result content preserved ─────────────────────────


def test_policy_preserves_structured_content_deterministically():
    """P1.3: a tool result with structured (non-string) content must NOT be
    flattened via str(); it must be serialized deterministically so
    search/paging stay lossless."""
    from agent_interop.abi import CanonicalToolResultBlock, CanonicalTextBlock
    from agent_interop.context_store.policy import VirtualizationPolicy

    p = VirtualizationPolicy(max_inline_lines=2, max_inline_bytes=10)
    structured = CanonicalToolResultBlock(
        tool_call_id="c1",
        content=[CanonicalTextBlock(text="line1"), CanonicalTextBlock(text="line2")],
    )
    text = p._extract_text(structured)
    # Deterministic: text blocks joined, not str([...]).
    assert text == "line1\nline2"
    # Decision based on real content, not a str() representation.
    d = p.decide(structured)
    assert d.should_virtualize is True
    assert "2 lines" in d.reason


# ─── P0.7: Conditional private tools ────────────────────────────────────────


def test_private_tools_conditional_no_state():
    """P0.7: When no state was virtualized and no tools were withheld,
    the model surface should include NO private __interop_* retrieval schemas."""
    from agent_interop.abi import CanonicalToolResultBlock

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
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
        context=ContextConfig(context_limit_tokens=32768),
    )
    config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
    gw = Gateway(config)

    # Small result — no virtualization.
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="do work")]),
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(
                id="c1", name="tool_0", arguments={"x": "v"},
            )]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(
                tool_call_id="c1", content="small",
            )]),
        ],
        tools=_client_tools(2),
        tool_choice=CanonicalToolChoice.auto(),
    )
    ctx = RequestContext()
    inv = asyncio.run(gw._prepare_invocation_async(req, ctx, streaming=False, execution=InteropRequestExecution(context=ctx)))

    private_names = [t.name for t in inv.model_request.tools if t.name.startswith("__interop_")]
    # No virtualized state → no retrieval schemas.
    assert "__interop_read_result" not in private_names, private_names
    assert "__interop_recall_history" not in private_names, private_names
    assert "__interop_search_history" not in private_names, private_names


def test_private_tools_conditional_with_virtualization():
    """P0.7: When a large result is virtualized, __interop_read_result should
    appear in the model surface."""
    from agent_interop.abi import CanonicalToolResultBlock
    from agent_interop.config import ContextConfig

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
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
        # Context limit where the un-virtualized request (~2587 tokens)
        # doesn't fit (safe_limit=1800), so compaction triggers and the
        # large tool result gets virtualized.
        context=ContextConfig(context_limit_tokens=2000, output_reserve_tokens=500),
    )
    config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
    gw = Gateway(config)

    # Large multi-line result → virtualized (exceeds max_inline_bytes=8000).
    # The OLD tool result is compacted; the latest is protected.
    big = "\n".join(f"line {i}" for i in range(500))  # 500 lines, ~3000 bytes
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="old task")]),
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
        tools=_client_tools(2),
        tool_choice=CanonicalToolChoice.auto(),
    )
    ctx = RequestContext(session_id="test-session")
    inv = asyncio.run(gw._prepare_invocation_async(req, ctx, streaming=False, execution=InteropRequestExecution(context=ctx)))

    private_names = [t.name for t in inv.model_request.tools if t.name.startswith("__interop_")]
    # Virtualized state → read_result schema present.
    assert "__interop_read_result" in private_names, private_names
