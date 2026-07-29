"""Tests for malformed streaming frame handling.

The streaming path must:

* yield typed ``MalformedFrame`` instead of a private dict sentinel,
* fail any open tool-call accumulator when a malformed frame occurs,
* emit a canonical stream error and terminate with INVALID_OUTPUT,
* enforce per-frame and per-tool-call size limits,
* flush SSE/NDJSON decoders at end-of-stream.
"""

from __future__ import annotations

import pytest

from agent_interop.streaming.coordinator import (
    MalformedFrame,
    PendingToolCallAccumulator,
    StreamLimits,
    ToolCallLimitExceeded,
    ToolStreamKey,
)
from agent_interop.transport.ndjson import MalformedNDJSONLine, NDJSONDecoder
from agent_interop.transport.sse import SSEDecoder


class TestMalformedFrameDataclass:
    def test_required_fields(self):
        f = MalformedFrame(
            raw_frame="{bad",
            stream_ordinal=3,
            event_name="message",
            parse_error="Expecting value",
            tool_call_accumulator_open=True,
            framing="sse",
        )
        assert f.raw_frame == "{bad"
        assert f.stream_ordinal == 3
        assert f.tool_call_accumulator_open is True
        assert f.framing == "sse"


class TestAccumulatorSizeLimits:
    def test_max_simultaneous_tool_calls(self):
        limits = StreamLimits(max_simultaneous_tool_calls=2)
        acc = PendingToolCallAccumulator(limits=limits)
        acc.start_call(ToolStreamKey(0, 0), "tc_1")
        acc.start_call(ToolStreamKey(0, 1), "tc_2")
        with pytest.raises(ToolCallLimitExceeded):
            acc.start_call(ToolStreamKey(0, 2), "tc_3")

    def test_max_accumulated_argument_bytes(self):
        limits = StreamLimits(max_accumulated_arg_bytes=10)
        acc = PendingToolCallAccumulator(limits=limits)
        key = ToolStreamKey(0, 0)
        acc.start_call(key, "tc_1")
        acc.feed_arguments(key, "x" * 6)
        with pytest.raises(ToolCallLimitExceeded):
            acc.feed_arguments(key, "x" * 6)

    def test_feed_arguments_passes_under_limit(self):
        limits = StreamLimits(max_accumulated_arg_bytes=100)
        acc = PendingToolCallAccumulator(limits=limits)
        key = ToolStreamKey(0, 0)
        acc.start_call(key, "tc_1")
        acc.feed_arguments(key, "abc")
        acc.feed_arguments(key, "def")
        assert acc._pending[key].assembled_arguments == "abcdef"


class TestSSEDecoderFlush:
    def test_flush_emits_partial_record(self):
        decoder = SSEDecoder()
        decoder.feed("event: message\n")
        decoder.feed("data: {\"x\": 1}\n")
        # No trailing blank line
        result = decoder.flush()
        assert result is not None
        assert result.event == "message"
        assert result.data == '{"x": 1}'

    def test_flush_returns_none_for_clean_state(self):
        decoder = SSEDecoder()
        assert decoder.flush() is None


class TestNDJSONDecoderMalformed:
    def test_yields_malformed_marker(self):
        decoder = NDJSONDecoder()
        out = list(decoder.feed("{\"a\": 1}\nnot_json\n{\"b\": 2}\n"))
        assert len(out) == 3
        assert out[0] == {"a": 1}
        # The middle element should be a MalformedNDJSONLine
        assert isinstance(out[1], MalformedNDJSONLine)
        assert out[1].parse_error  # has some parse error
        assert out[1].ordinal == 2
        assert out[2] == {"b": 2}

    def test_oversized_line_marked_malformed(self):
        decoder = NDJSONDecoder(max_frame_bytes=10)
        out = list(decoder.feed("{\"a\": \"" + ("x" * 100) + "\"}\n"))
        assert len(out) == 1
        assert isinstance(out[0], MalformedNDJSONLine)
        assert "max_frame_bytes" in out[0].parse_error

    def test_flush_handles_trailing_garbage(self):
        decoder = NDJSONDecoder()
        out = list(decoder.feed("{\"a\": 1}\n"))
        assert out == [{"a": 1}]
        # Now feed some trailing unparseable text via buffer state
        decoder._buffer = "garbage"
        flush_out = decoder.flush()
        assert len(flush_out) == 1
        assert isinstance(flush_out[0], MalformedNDJSONLine)
