"""Regression test: a streamed turn that emits a tool_use block must report
stop_reason=tool_use, even when the backend's own terminal frame disagrees.

Found via a real live-client acceptance run (Claude Code 2.1.220 against
Ollama's gpt-oss:20b-cloud through a real Interop gateway): Ollama streams
the tool_calls fragment in a NON-terminal chunk, then closes with a
`done:true` frame whose own `done_reason` is `"stop"` (mapped to END_TURN)
and no tool_calls of its own. ``decode_stream_chunk`` has no cross-chunk
state — it can only see that one terminal frame — so it reported
DecodedStreamComplete(stop_reason=END_TURN), which silently overrode the
correct TOOL_CALL signal ``StreamCoordinator`` had already recorded from
the earlier chunk. The client (Claude Code) received a response with a
real tool_use content block but stop_reason=end_turn — a self-contradictory
message the Anthropic Messages protocol never produces — and treated the
turn as finished, never executing the tool or printing anything.

A response containing an emitted tool_use block reporting anything other
than stop_reason=tool_use is a protocol invariant violation regardless of
which upstream backend or wire format produced it — this mirrors the
equivalent guard already in place on the non-streaming path (~gateway.py
line 1710: ``if batch_decision.accepted_blocks and stop_reason == END_TURN:
stop_reason = TOOL_CALL``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Self

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


class _FakeNdjsonStream:
    """Minimal ``UpstreamStream`` stand-in yielding pre-seeded, already-
    parsed NDJSON frames — mirrors Ollama's real /api/chat streaming shape,
    where each line is a complete JSON object."""

    def __init__(self, frames: list[dict[str, Any]], status_code: int = 200) -> None:
        self.status_code = status_code
        self._frames = frames

    async def ndjson_events(self):
        for frame in self._frames:
            yield frame

    async def raw_lines(self):
        return
        yield  # pragma: no cover

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeNdjsonTransport:
    def __init__(self, frames: list[dict[str, Any]], status_code: int = 200) -> None:
        self._frames = frames
        self._status_code = status_code

    @asynccontextmanager
    async def stream(
        self, request: PreparedUpstreamRequest,
    ) -> AsyncIterator[_FakeNdjsonStream]:
        yield _FakeNdjsonStream(self._frames, self._status_code)

    async def close(self) -> None:
        return None


def _make_ollama_gateway(*, tool_mode: ToolMode = ToolMode.AUTO) -> Gateway:
    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "test-route": ModelRoute(
                id="test-route",
                client_model_aliases=["test-model"],
                upstream_model="gpt-oss:20b-cloud",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=tool_mode,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    return Gateway(config=config)


def _make_request() -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="test-model"),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="Read /tmp/x.txt")]),
        ],
        tools=[READ_FILE_TOOL],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )


# Reproduces gpt-oss:20b-cloud's real observed frame sequence: several
# content-free "thinking" deltas, then the tool_calls fragment in a
# NON-terminal frame, then a terminal frame with done_reason="stop" and no
# tool_calls of its own — the exact shape that broke stop_reason.
_GPT_OSS_STYLE_FRAMES = [
    {"message": {"role": "assistant", "content": "", "thinking": "Let me"}, "done": False},
    {"message": {"role": "assistant", "content": "", "thinking": " check the file."}, "done": False},
    {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_abc123",
                "function": {"index": 0, "name": "read_file", "arguments": {"path": "/tmp/x.txt"}},
            }],
        },
        "done": False,
    },
    {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
]


class TestStreamedToolCallReportsToolUseStopReason:
    @pytest.mark.asyncio
    async def test_stop_reason_is_tool_use_not_end_turn(self):
        gw = _make_ollama_gateway()
        gw._transport = _FakeNdjsonTransport(_GPT_OSS_STYLE_FRAMES)

        events = [e async for e in gw.handle_stream(_make_request(), RequestContext())]

        uses = [e for e in events if e.type == "tool_use"]
        assert len(uses) == 1, f"expected exactly one tool_use event, got {events}"
        assert uses[0].content_block.name == "read_file"

        stops = [e for e in events if e.type == "message_stop"]
        assert len(stops) == 1
        assert stops[0].stop_reason == CanonicalStopReason.TOOL_CALL, (
            f"a message with an emitted tool_use block must report "
            f"stop_reason=tool_use, got {stops[0].stop_reason!r}"
        )

    @pytest.mark.asyncio
    async def test_plain_text_turn_still_reports_end_turn(self):
        """The fix must not blanket-force TOOL_CALL for every response —
        only when a tool_use block was actually emitted this turn."""
        gw = _make_ollama_gateway()
        frames = [
            {"message": {"role": "assistant", "content": "", "thinking": "thinking..."}, "done": False},
            {"message": {"role": "assistant", "content": "banana"}, "done": False},
            {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"},
        ]
        gw._transport = _FakeNdjsonTransport(frames)

        events = [e async for e in gw.handle_stream(_make_request(), RequestContext())]

        assert [e for e in events if e.type == "tool_use"] == []
        stops = [e for e in events if e.type == "message_stop"]
        assert len(stops) == 1
        assert stops[0].stop_reason == CanonicalStopReason.END_TURN
