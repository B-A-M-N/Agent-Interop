"""Tests proving each of the six streaming tool-call correctness gaps is fixed.

These drive the live Gateway streaming path with fake SSE transports and assert
the specific behavior each gap's fix guarantees. They are intentionally narrow
and deterministic — each test targets exactly one gap.

Gaps under test (see gateway.py / streaming/coordinator.py):
  1. BUFFER_TEXTUAL_RESPONSE envelopes are buffered, never leaked as text.
  2. Terminal-frame decode — usage/stop-reason on the terminal frame is kept.
  3. Accumulator byte-limit enforcement via feed_arguments.
  4. Atomicity is per-turn (one process_tool_batch call per turn).
  5. Finalization on BOTH success exits (terminal-frame and natural loop-end).
  6. Malformed-frame-with-pending-tools finalizes the record as FAILED.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Self
from unittest.mock import Mock

import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.errors import InteropErrorCode
from agent_interop.execution import ExecutionState, InteropRequestExecution
from agent_interop.gateway import Gateway
from agent_interop.transport.http import PreparedUpstreamRequest

READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)

TEST_TOOL = CanonicalTool(
    name="test_tool",
    description="A test tool",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
)


# ─── Fake SSE transport (mirrors test_evidence_store_wiring.py) ─────────────


class _FakeSseStream:
    """Minimal ``UpstreamStream`` stand-in yielding pre-seeded SSE data lines."""

    def __init__(self, data_lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._data_lines = data_lines

    async def sse_events(self):
        from agent_interop.transport.sse import SSEFrame

        for line in self._data_lines:
            yield SSEFrame(data=line)

    async def raw_lines(self):
        return
        yield  # pragma: no cover

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeSseTransport:
    """Transport that yields a pre-seeded SSE stream for every request."""

    def __init__(self, data_lines: list[str], status_code: int = 200) -> None:
        self._data_lines = data_lines
        self._status_code = status_code

    @asynccontextmanager
    async def stream(
        self, request: PreparedUpstreamRequest,
    ) -> AsyncIterator[_FakeSseStream]:
        yield _FakeSseStream(self._data_lines, self._status_code)

    async def close(self) -> None:
        return None


# ─── Helpers ────────────────────────────────────────────────────────────────


def _make_gateway(
    *,
    tool_mode: ToolMode = ToolMode.AUTO,
    max_tool_argument_bytes: int = 16 * 1024 * 1024,
) -> Gateway:
    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        max_tool_argument_bytes=max_tool_argument_bytes,
        routes={
            "test-route": ModelRoute(
                id="test-route",
                client_model_aliases=["test-model"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=tool_mode,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    return Gateway(config=config)


def _make_request(*, tools: list[CanonicalTool] | None = None) -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="test-model"),
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalTextBlock(text="Do the thing")],
            )
        ],
        tools=tools or [READ_FILE_TOOL],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )


async def _collect(
    gateway: Gateway, request: CanonicalRequest,
) -> list[CanonicalRequest.__class__]:
    """Collect canonical events from handle_stream."""
    from agent_interop.abi import CanonicalEvent  # local to avoid heavy import at module load

    events: list[CanonicalEvent] = []
    async for event in gateway.handle_stream(request, RequestContext()):
        events.append(event)
    return events  # type: ignore[return-value]


def _text_deltas(events: list[Any]) -> str:
    return "".join(getattr(e, "partial", "") for e in events if e.type == "text_delta")


def _tool_use(events: list[Any]) -> list[Any]:
    return [e for e in events if e.type == "tool_use"]


# ═══════════════════════════════════════════════════════════════════════
# Gap 1: BUFFER_TEXTUAL_RESPONSE envelopes are buffered, never leaked
# ═══════════════════════════════════════════════════════════════════════


class TestGap1BufferTextualResponse:
    """A PROMPTED-mode local model emits raw <tool_call> envelopes as text.
    In BUFFER_TEXTUAL_RESPONSE mode those envelopes must be buffered until the
    response is complete and run through textual extraction — the client must
    NEVER see a text_delta containing the literal <tool_call> substring, and
    MUST receive a tool_use event for the extracted call."""

    @pytest.mark.asyncio
    async def test_envelope_buffered_not_leaked(self):
        # Explicit PROMPTED mode forces stream_extraction_mode ==
        # BUFFER_TEXTUAL_RESPONSE regardless of profile resolution.
        gw = _make_gateway(tool_mode=ToolMode.PROMPTED)

        # Split the envelope across multiple text-delta frames.
        envelope = '<tool_call>{"name":"test_tool","arguments":{"key":"value"}}</tool_call>'
        part1 = envelope[: len(envelope) // 2]
        part2 = envelope[len(envelope) // 2 :]
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": part1},
                "index": 0,
                "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {"content": part2},
                "index": 0,
                "finish_reason": None,
            }]}),
            # Terminal frame.
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)

        request = _make_request(tools=[TEST_TOOL])
        events = []
        async for event in gw.handle_stream(request, RequestContext()):
            events.append(event)

        text = _text_deltas(events)
        # The literal envelope must NEVER leak to the client.
        assert "<tool_call>" not in text, f"envelope leaked as text: {text!r}"
        # Surrounding text (if any) is preserved — here there is none, so no
        # text_delta is expected at all.
        uses = _tool_use(events)
        assert len(uses) == 1, f"expected exactly one tool_use, got {len(uses)}: {events}"
        block = uses[0].content_block
        assert block.name == "test_tool"
        assert block.arguments == {"key": "value"}

    @pytest.mark.asyncio
    async def test_plain_text_still_streams_in_buffer_mode(self):
        """In BUFFER mode, plain text (no envelope) is still delivered — it is
        just buffered until end-of-turn instead of streamed per-frame."""
        gw = _make_gateway(tool_mode=ToolMode.PROMPTED)
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": "Hello "},
                "index": 0,
                "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {"content": "world"},
                "index": 0,
                "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request(tools=[TEST_TOOL])
        events = []
        async for event in gw.handle_stream(request, RequestContext()):
            events.append(event)

        text = _text_deltas(events)
        assert "Hello world" in text
        assert "<tool_call>" not in text


# ═══════════════════════════════════════════════════════════════════════
# Gap 2: Terminal-frame decode — usage is captured, not dropped
# ═══════════════════════════════════════════════════════════════════════


class TestGap2TerminalFrameUsage:
    """A terminal frame carrying a `usage` field must have that usage captured
    into the CanonicalResponse passed to finalize_response — it must not be
    dropped by the early-return-on-terminal-frame path."""

    @pytest.mark.asyncio
    async def test_usage_from_terminal_frame_reaches_response(self):
        gw = _make_gateway()
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": "Hi"},
                "index": 0,
                "finish_reason": None,
            }]}),
            # Terminal frame WITH usage at top level (OpenAI-style).
            json.dumps({
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        # Drive _handle_stream_send directly with a custom exec_record so we
        # can observe the CanonicalResponse handed to finalize_response.
        from agent_interop.abi import CanonicalResponse

        observed: dict[str, Any] = {}

        class _RecordingExec(InteropRequestExecution):
            def finalize_response(self, response: CanonicalResponse) -> None:
                observed["usage"] = response.usage
                super().finalize_response(response)

        exec_record = _RecordingExec()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )

        async for _ in gw._handle_stream_send(invocation, exec_record):
            pass

        usage = observed.get("usage")
        assert usage is not None, "finalize_response was not called with a response"
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5
        assert usage.total_tokens == 15


# ═══════════════════════════════════════════════════════════════════════
# Gap 3: Accumulator byte-limit enforcement
# ═══════════════════════════════════════════════════════════════════════


class TestGap3AccumulatorLimit:
    """A tool call whose argument fragments exceed max_accumulated_arg_bytes
    must be rejected as a clean error (error + message_stop) with the exec
    record finalized as FAILED — NOT an unhandled ToolCallLimitExceeded."""

    @pytest.mark.asyncio
    async def test_argument_over_limit_yields_error_and_fails(self):
        # Tiny limit so a single fragment trips it.
        gw = _make_gateway(max_tool_argument_bytes=10)
        big_args = json.dumps({"path": "x" * 100})  # way over 10 bytes
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "tc_001",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": big_args},
                }]},
                "index": 0,
                "finish_reason": None,
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )

        events = []
        async for event in gw._handle_stream_send(invocation, exec_record):
            events.append(event)

        # Clean error terminal, not an unhandled exception.
        assert len(events) == 2, f"expected error + message_stop, got {events}"
        assert events[0].type == "error"
        assert events[0].error.code == "TOOL_CALL_LIMIT_EXCEEDED"
        assert events[1].type == "message_stop"
        assert exec_record.state == ExecutionState.FAILED


# ═══════════════════════════════════════════════════════════════════════
# Gap 4: Atomicity is per-turn (one batch decision per turn)
# ═══════════════════════════════════════════════════════════════════════


class TestGap4AtomicPerTurn:
    """Two distinct tool calls completing on different frames within one turn
    must be validated as ONE atomic batch — process_tool_batch is invoked
    exactly once for the whole turn, not once per drain."""

    @pytest.mark.asyncio
    async def test_two_calls_one_batch(self):
        """Gap 4: two tool calls completing on different frames within one turn
        must be validated as ONE atomic batch.

        The OpenAI Chat codec only emits batch-level completion on the terminal
        frame, which cannot distinguish the old mid-loop-drain behavior from the
        fixed end-of-turn drain. So we patch the codec to emit per-call
        ``DecodedToolCallComplete`` events on separate (non-terminal) frames —
        exactly the shape that made the old code drain+batch call A on frame 2
        (its own batch) and call B on frame 4 (a second batch). The fix defers
        all draining to end-of-turn, so both calls land in a single batch.
        """
        gw = _make_gateway()
        # Frames 1-2 carry the two tool fragments; frames 3-4 are empty
        # non-terminal frames we use to inject per-call completions on
        # SEPARATE frames; frame 5 is the terminal. Staggering the completions
        # across frames is what triggers the old per-drain batching bug.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 0, "id": "tc_a", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"/a"}'},
                }]},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 1, "id": "tc_b", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"/b"}'},
                }]},
                "index": 0, "finish_reason": None,
            }]}),
            # Empty non-terminal frame → inject complete(call A) here.
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": None,
            }]}),
            # Empty non-terminal frame → inject complete(call B) here.
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": None,
            }]}),
            # Terminal frame.
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        # Patch the codec so call A completes on frame 3 and call B on frame 4
        # — each on its own frame, before the terminal frame.
        from agent_interop.upstreams.codec import DecodedToolCallComplete

        codec = gw._prepare_invocation(
            request, RequestContext(), streaming=True,
            execution=InteropRequestExecution(),
        ).codec
        original_decode = codec.decode_stream_chunk
        frame_index = [0]

        def patched_decode(chunk):
            events = original_decode(chunk)
            frame_index[0] += 1
            if frame_index[0] == 3:
                events.append(DecodedToolCallComplete(choice_index=0, tool_index=0))
            elif frame_index[0] == 4:
                events.append(DecodedToolCallComplete(choice_index=0, tool_index=1))
            return events

        codec.decode_stream_chunk = patched_decode

        # Spy on process_tool_batch via the binding gateway.py actually uses
        # (it imported the name into its own namespace, so we must patch there).
        from agent_interop import gateway as gateway_mod

        call_count = 0
        candidate_counts: list[int] = []
        original_ptb = gateway_mod.process_tool_batch

        async def fake_process_tool_batch(candidates, tools, **kwargs):
            nonlocal call_count
            call_count += 1
            candidate_counts.append(len(candidates))
            # Return an accepted decision with an accepted_block per candidate so
            # tool_use events are actually emitted to the client.
            from agent_interop.abi import CanonicalToolCallBlock
            from agent_interop.transaction import ToolBatchDecision
            accepted = []
            for c in candidates:
                raw = c.raw_arguments
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (json.JSONDecodeError, TypeError):
                    args = {}
                accepted.append(CanonicalToolCallBlock(
                    id=c.id or "", name=c.name or "", arguments=args,
                ))
            return ToolBatchDecision(is_accepted=True, accepted_blocks=accepted)

        gateway_mod.process_tool_batch = fake_process_tool_batch
        try:
            events = []
            async for event in gw.handle_stream(request, RequestContext()):
                events.append(event)
        finally:
            gateway_mod.process_tool_batch = original_ptb
            codec.decode_stream_chunk = original_decode

        # THE assertion: exactly one batch decision for the whole turn.
        assert call_count == 1, (
            f"process_tool_batch called {call_count} times — expected 1 "
            f"(atomic per-turn). candidate_counts={candidate_counts}"
        )
        # And that single batch contained BOTH calls.
        assert candidate_counts == [2], (
            f"expected one batch of 2 candidates, got {candidate_counts}"
        )
        # Both calls reach the client.
        uses = [e for e in events if e.type == "tool_use"]
        assert len(uses) == 2, f"expected 2 tool_use events, got {len(uses)}: {events}"


# ═══════════════════════════════════════════════════════════════════════
# Gap 5: Finalization on BOTH success exits
# ═══════════════════════════════════════════════════════════════════════


class TestGap5FinalizationBothExits:
    """A normal completion must finalize the exec record as SUCCEEDED on BOTH
    success exits: the terminal-frame path AND the natural-loop-end path."""

    @pytest.mark.asyncio
    async def test_terminal_frame_completes_succeeded(self):
        gw = _make_gateway()
        # Terminal frame present (finish_reason) — the common OpenAI path.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": "Hello"},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )
        async for _ in gw._handle_stream_send(invocation, exec_record):
            pass

        assert exec_record.state == ExecutionState.SUCCEEDED, (
            f"terminal-frame completion left state={exec_record.state}"
        )

    @pytest.mark.asyncio
    async def test_natural_loop_end_completes_succeeded(self):
        gw = _make_gateway()
        # NO terminal frame — the upstream just closes after a text delta.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": "Hello"},
                "index": 0, "finish_reason": None,
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )
        async for _ in gw._handle_stream_send(invocation, exec_record):
            pass

        assert exec_record.state == ExecutionState.SUCCEEDED, (
            f"natural-loop-end completion left state={exec_record.state}"
        )


# ═══════════════════════════════════════════════════════════════════════
# Gap 6: Malformed frame with pending tools finalizes FAILED
# ═══════════════════════════════════════════════════════════════════════


class TestGap6MalformedFrameFinalizesFailed:
    """A malformed frame arriving while a tool call is pending must abort with
    an error AND finalize the exec record as FAILED (not leave it ACTIVE)."""

    @pytest.mark.asyncio
    async def test_malformed_frame_with_pending_tools_fails(self):
        gw = _make_gateway()
        # Frame 1: open a pending tool call. Frame 2: non-JSON -> malformed.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 0, "id": "tc_001", "type": "function",
                    "function": {"name": "read_file", "arguments": ""},
                }]},
                "index": 0, "finish_reason": None,
            }]}),
            "this is not valid json {{{",
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )

        events = []
        async for event in gw._handle_stream_send(invocation, exec_record):
            events.append(event)

        assert len(events) == 2, f"expected error + message_stop, got {events}"
        assert events[0].type == "error"
        assert events[0].error.code == "MALFORMED_FRAME"
        assert events[1].type == "message_stop"
        # THE assertion for gap 6: record finalized as FAILED, not left ACTIVE.
        assert exec_record.state == ExecutionState.FAILED, (
            f"malformed-frame completion left state={exec_record.state}"
        )


# ═══════════════════════════════════════════════════════════════════════
# MVP-11: malformed stream frames must be bounded, not skipped forever
# ═══════════════════════════════════════════════════════════════════════


class TestMalformedFrameBound:
    """A malformed frame with no open tool state used to be `continue`d
    silently and unconditionally — a backend that never sends another valid
    frame would leave the stream open forever. It must terminate once
    max_malformed_stream_frames is exceeded, and each malformed frame must
    be recorded as a diagnostic."""

    @pytest.mark.asyncio
    async def test_terminates_after_threshold_with_no_open_tool_state(self):
        gw = _make_gateway()
        gw.config.max_malformed_stream_frames = 2
        data_lines = [
            "not valid json {{{",
            "still not valid json {{{",
            "definitely not valid json {{{",
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )
        events = [event async for event in gw._handle_stream_send(invocation, exec_record)]

        errors = [e for e in events if e.type == "error"]
        assert len(errors) == 1
        assert errors[0].error.code == "MALFORMED_FRAME"
        stops = [e for e in events if e.type == "message_stop"]
        assert len(stops) == 1
        assert exec_record.state == ExecutionState.FAILED
        assert len(exec_record.raw_frame_evidence) == 3

    @pytest.mark.asyncio
    async def test_stays_open_below_threshold(self):
        gw = _make_gateway()
        gw.config.max_malformed_stream_frames = 5
        data_lines = [
            "not valid json {{{",
            json.dumps({"choices": [{
                "delta": {"content": "Hello"},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        events = await _collect(gw, request)

        errors = [e for e in events if e.type == "error"]
        assert not errors
        text_deltas = [e for e in events if e.type == "text_delta"]
        assert any(e.partial == "Hello" for e in text_deltas)


# ═══════════════════════════════════════════════════════════════════════
# MVP-10: streaming usage must reach the client before message_stop
# ═══════════════════════════════════════════════════════════════════════


class TestStreamingUsageEmitted:
    """final_usage is captured from the upstream stream but was never
    yielded as a usage_update event — coding clients that use token usage
    for context-pressure decisions never saw it in streaming mode."""

    @pytest.mark.asyncio
    async def test_usage_update_yielded_before_message_stop(self):
        gw = _make_gateway()
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": "Hello"},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({
                "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
            }),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        events = await _collect(gw, request)

        usage_events = [e for e in events if e.type == "usage_update"]
        assert len(usage_events) == 1, f"expected exactly one usage_update: {events}"
        assert usage_events[0].input_tokens == 12
        assert usage_events[0].output_tokens == 7

        stop_index = next(i for i, e in enumerate(events) if e.type == "message_stop")
        usage_index = events.index(usage_events[0])
        assert usage_index < stop_index, "usage_update must be yielded before message_stop"


# ═══════════════════════════════════════════════════════════════════════
# Regression: post-message_stop bookkeeping must not double message_stop
# ═══════════════════════════════════════════════════════════════════════


class TestPostMessageStopBookkeeping:
    """End-of-turn bookkeeping (evidence write-back and finalize_response)
    runs BEFORE the terminal message_stop is yielded, not after.

    This matters because the real ASGI server (server/app.py) stops
    consuming the Gateway's stream generator as soon as it sees
    message_stop — anything scheduled after that yield is not guaranteed to
    run through the real request path. If finalization itself fails, the
    client must see a genuine terminal error rather than a message_stop
    that silently lied about success.
    """

    @pytest.mark.asyncio
    async def test_finalize_response_failure_surfaces_as_terminal_error(self):
        """A finalize_response() failure must produce error + message_stop
        (INVALID_OUTPUT) — exactly one message_stop, and the execution
        record must end up FAILED, not SUCCEEDED nor left ACTIVE."""
        gw = _make_gateway()
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": "Hello"},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request()

        from agent_interop.execution import ExecutionState, InteropRequestExecution

        original_finalize = InteropRequestExecution.finalize_response

        def failing_finalize(self, response):
            raise RuntimeError("simulated finalize_response failure")

        InteropRequestExecution.finalize_response = failing_finalize  # type: ignore[assignment]
        try:
            exec_record = InteropRequestExecution()
            invocation = gw._prepare_invocation(
                request, RequestContext(), streaming=True, execution=exec_record,
            )
            events = [event async for event in gw._handle_stream_send(invocation, exec_record)]
        finally:
            InteropRequestExecution.finalize_response = original_finalize  # type: ignore[assignment]

        stops = [e for e in events if e.type == "message_stop"]
        assert len(stops) == 1, (
            f"expected exactly one message_stop, got {len(stops)}: {events}"
        )
        assert stops[0].stop_reason == CanonicalStopReason.INVALID_OUTPUT

        errors = [e for e in events if e.type == "error"]
        assert len(errors) == 1, (
            f"expected the finalize failure to surface as a terminal error: {events}"
        )
        assert exec_record.state == ExecutionState.FAILED


# ═══════════════════════════════════════════════════════════════════════
# P0.2 — Fully-rejected tool batch must surface as INVALID_OUTPUT, not
#         silently collapse to END_TURN, in BOTH streaming modes.
# ═══════════════════════════════════════════════════════════════════════


class TestP02FullRejectionInvalidOutput:
    """A fully-rejected tool batch (no calls accepted) in streaming mode must
    emit a structured ``error`` event + ``message_stop(INVALID_OUTPUT)`` and
    finalize the record as FAILED — exactly mirroring the non-streaming
    ``_assemble_response`` path. The stop reason must NOT silently collapse
    to END_TURN, and evidence write-back must be skipped."""

    @staticmethod
    async def _drive(gw: Gateway, request: CanonicalRequest) -> tuple:
        """Drive ``_handle_stream_send`` directly and return (events, exec_record)."""
        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )
        events = []
        async for event in gw._handle_stream_send(invocation, exec_record):
            events.append(event)
        return events, exec_record

    @pytest.mark.asyncio
    async def test_full_rejection_buffer_mode(self):
        """P0.2 / BUFFER (prompted) mode: a textual tool call that fails
        validation is fully rejected → error + INVALID_OUTPUT, no tool_use,
        record FAILED, no evidence write-back."""
        gw = _make_gateway(tool_mode=ToolMode.PROMPTED)
        # Envelope calls a tool NOT in the registered list → guaranteed
        # rejection (canonicalize_tool_name fails → REJECTED, atomic batch).
        envelope = '<tool_call>{"name":"no_such_tool","arguments":{}}</tool_call>'
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": envelope},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        # Inject an evidence store so the tail WOULD attempt write-back if it
        # ran — proving the rejection path genuinely skips it.
        evidence_spy = Mock()
        gw._evidence_store = Mock()
        original_record_evidence = gw._record_evidence_observation
        gw._record_evidence_observation = lambda inv, exe: evidence_spy(inv, exe)

        try:
            request = _make_request(tools=[TEST_TOOL])
            events, exec_record = await self._drive(gw, request)
        finally:
            gw._record_evidence_observation = original_record_evidence

        errors = [e for e in events if e.type == "error"]
        stops = [e for e in events if e.type == "message_stop"]
        uses = [e for e in events if e.type == "tool_use"]

        # A structured rejection error is surfaced.
        assert len(errors) == 1, f"expected exactly one error event, got {events}"
        assert errors[0].error.code == InteropErrorCode.TOOL_CALL_INVALID, (
            f"expected TOOL_CALL_INVALID, got {errors[0].error.code}"
        )
        # Terminal stop reason is INVALID_OUTPUT, not a silent END_TURN.
        assert len(stops) == 1, f"expected exactly one message_stop, got {events}"
        assert stops[0].stop_reason == CanonicalStopReason.INVALID_OUTPUT, (
            f"expected INVALID_OUTPUT, got {stops[0].stop_reason}"
        )
        # No tool_use events — nothing was accepted.
        assert uses == [], f"expected no tool_use events, got {uses}"
        # Record finalized as FAILED.
        assert exec_record.state == ExecutionState.FAILED, (
            f"expected FAILED state, got {exec_record.state}"
        )
        # Evidence write-back was NOT attempted (tail skipped on rejection).
        evidence_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_full_rejection_native_mode(self):
        """P0.2 / non-BUFFER (native) mode: a native tool call that fails
        validation is fully rejected → error + INVALID_OUTPUT, no tool_use,
        record FAILED, no evidence write-back."""
        gw = _make_gateway()
        # Native fragment for a tool NOT in the registered list → guaranteed
        # rejection. Fragment on frame 1, terminal frame 2.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 0, "id": "tc_001", "type": "function",
                    "function": {"name": "no_such_tool", "arguments": "{}"},
                }]},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        evidence_spy = Mock()
        gw._evidence_store = Mock()
        original_record_evidence = gw._record_evidence_observation
        gw._record_evidence_observation = lambda inv, exe: evidence_spy(inv, exe)

        try:
            request = _make_request(tools=[READ_FILE_TOOL])
            events, exec_record = await self._drive(gw, request)
        finally:
            gw._record_evidence_observation = original_record_evidence

        errors = [e for e in events if e.type == "error"]
        stops = [e for e in events if e.type == "message_stop"]
        uses = [e for e in events if e.type == "tool_use"]

        assert len(errors) == 1, f"expected exactly one error event, got {events}"
        assert errors[0].error.code == InteropErrorCode.TOOL_CALL_INVALID, (
            f"expected TOOL_CALL_INVALID, got {errors[0].error.code}"
        )
        assert len(stops) == 1, f"expected exactly one message_stop, got {events}"
        assert stops[0].stop_reason == CanonicalStopReason.INVALID_OUTPUT, (
            f"expected INVALID_OUTPUT, got {stops[0].stop_reason}"
        )
        assert uses == [], f"expected no tool_use events, got {uses}"
        assert exec_record.state == ExecutionState.FAILED, (
            f"expected FAILED state, got {exec_record.state}"
        )
        evidence_spy.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════
# P0.5 — Hybrid native + textual extraction must not drop the textual call.
# ═══════════════════════════════════════════════════════════════════════


class TestP05HybridNativeTextual:
    """A response carrying BOTH a native tool-call fragment AND a distinct
    textual <tool_call> envelope must contribute BOTH calls (the textual one
    must not be dropped, nor its raw text leaked into visible content)."""

    @pytest.mark.asyncio
    async def test_hybrid_native_and_textual_both_emitted(self):
        """P0.5: one native fragment (read_file) + one textual envelope for a
        DIFFERENT tool (test_tool) → both survive as separate tool_use events
        in BUFFER mode."""
        gw = _make_gateway(tool_mode=ToolMode.PROMPTED)
        # Frame 1: native fragment for read_file. Frame 2: textual envelope for
        # test_tool (a DIFFERENT tool). Frame 3: terminal.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 0, "id": "tc_native", "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"/x"}'},
                }]},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {"content": '<tool_call>{"name":"test_tool","arguments":{"key":"v"}}</tool_call>'},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        # Register BOTH tools so each call validates.
        request = _make_request(tools=[READ_FILE_TOOL, TEST_TOOL])

        events = []
        async for event in gw.handle_stream(request, RequestContext()):
            events.append(event)

        uses = [e for e in events if e.type == "tool_use"]
        assert len(uses) == 2, f"expected 2 tool_use events, got {len(uses)}: {events}"
        names = sorted(u.content_block.name for u in uses)
        assert names == ["read_file", "test_tool"], f"got tool names {names}"
        # The textual envelope text must NOT leak as a visible text delta.
        text = _text_deltas(events)
        assert "<tool_call>" not in text, f"envelope leaked as text: {text!r}"


# ═══════════════════════════════════════════════════════════════════════
# P0.6 — Prompted-only streaming tool calls must record decisions.
# ═══════════════════════════════════════════════════════════════════════


class TestP06PromptedToolDecisionsRecorded:
    """A purely-textual (prompted, no native fragments) successful streaming
    tool call must populate ``execution.tool_decisions`` — previously the
    BUFFER path gated recording behind ``if native_candidates:``."""

    @pytest.mark.asyncio
    async def test_prompted_textual_call_records_tool_decisions(self):
        """P0.6: a prompted-mode streaming tool call with no native fragments
        must leave execution.tool_decisions non-empty after the stream."""
        gw = _make_gateway(tool_mode=ToolMode.PROMPTED)
        envelope = '<tool_call>{"name":"test_tool","arguments":{"key":"value"}}</tool_call>'
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"content": envelope},
                "index": 0, "finish_reason": None,
            }]}),
            json.dumps({"choices": [{
                "delta": {}, "index": 0, "finish_reason": "stop",
            }]}),
        ]
        gw._transport = _FakeSseTransport(data_lines)
        request = _make_request(tools=[TEST_TOOL])

        exec_record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=True, execution=exec_record,
        )
        async for _ in gw._handle_stream_send(invocation, exec_record):
            pass

        assert len(exec_record.tool_decisions) == 1, (
            f"expected 1 tool decision, got {len(exec_record.tool_decisions)}: "
            f"{exec_record.tool_decisions}"
        )
        assert exec_record.tool_decisions[0].tool_name == "test_tool"
        assert exec_record.tool_decisions[0].accepted is True
