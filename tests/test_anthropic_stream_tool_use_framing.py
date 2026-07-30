"""Regression test: a streamed tool_use content block must be protocol-
correct — content_block_start with an EMPTY input, the real arguments
delivered via a content_block_delta(input_json_delta), and an explicit
content_block_stop closing the block.

Found via a real live-client acceptance run (Claude Code 2.1.220 against
gpt-oss:20b-cloud through a real Interop gateway) — even after fixing the
dropped-first-text-delta bug and the wrong stop_reason bug, the round trip
still failed with Claude Code's own client-side error "The model's tool
call could not be parsed (retry also failed)", despite the tool call
itself being perfect (correct name, correct file_path argument). The
encoder was putting the FULLY-POPULATED arguments dict directly into
content_block_start's "input" field and never emitting any
input_json_delta or content_block_stop for the block at all —
gateway.py's only tool-call streaming path (_emit_batch_decision_events)
hands the encoder one already-fully-decided CanonicalToolCallBlock per
call and never emits a matching close event, so the encoder must
synthesize the complete, protocol-correct start+delta+stop sequence
itself in one encode() call. Claude Code's own SDK rebuilds "input"
purely from accumulated input_json_delta chunks and finalizes it on
content_block_stop — with neither ever sent, it ended up trying to parse
an empty string, not the fully-formed input Interop had already put in
content_block_start.
"""

from __future__ import annotations

import json

from agent_interop.abi import CanonicalEvent, CanonicalToolCallBlock
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter


def _encoder():
    adapter = AnthropicMessagesAdapter()
    return adapter.create_stream_encoder({"response_id": "msg_1", "model": "test-model"})


def _tool_use_event() -> CanonicalEvent:
    return CanonicalEvent(
        type="tool_use",
        index=0,
        content_block=CanonicalToolCallBlock(
            id="toolu_abc123",
            name="Read",
            arguments={"file_path": "/tmp/secret.txt"},
        ),
    )


class TestToolUseStreamFraming:
    def test_content_block_start_has_empty_input(self):
        encoder = _encoder()
        frame = encoder.encode(_tool_use_event())
        assert frame is not None
        start_line = next(
            line for line in frame.splitlines()
            if line.startswith("data:") and '"content_block_start"' in line
        )
        payload = json.loads(start_line[len("data: "):])
        assert payload["content_block"]["type"] == "tool_use"
        assert payload["content_block"]["input"] == {}
        assert payload["content_block"]["name"] == "Read"
        assert payload["content_block"]["id"] == "toolu_abc123"

    def test_real_arguments_delivered_via_input_json_delta(self):
        encoder = _encoder()
        frame = encoder.encode(_tool_use_event())
        assert "input_json_delta" in frame
        delta_line = next(
            line for line in frame.splitlines()
            if line.startswith("data:") and '"content_block_delta"' in line
        )
        payload = json.loads(delta_line[len("data: "):])
        assert payload["delta"]["type"] == "input_json_delta"
        # The client concatenates partial_json chunks and parses the
        # result once the block closes — it must be exactly the real
        # arguments, round-tripping through json.loads cleanly.
        assert json.loads(payload["delta"]["partial_json"]) == {"file_path": "/tmp/secret.txt"}

    def test_block_is_explicitly_closed(self):
        """Without this, gateway.py never sends a separate closing event
        for tool_use blocks (its only caller hands the encoder one fully-
        decided call and moves straight to message_stop) — the block must
        be self-closing within this one encode() call."""
        encoder = _encoder()
        frame = encoder.encode(_tool_use_event())
        assert "content_block_stop" in frame

    def test_frame_order_is_start_then_delta_then_stop(self):
        encoder = _encoder()
        frame = encoder.encode(_tool_use_event())
        start_pos = frame.index("event: content_block_start")
        delta_pos = frame.index("event: content_block_delta")
        stop_pos = frame.index("event: content_block_stop")
        assert start_pos < delta_pos < stop_pos

    def test_index_is_consistent_across_all_three_frames(self):
        encoder = _encoder()
        frame = encoder.encode(_tool_use_event())
        indices = set()
        for line in frame.splitlines():
            if line.startswith("data:"):
                payload = json.loads(line[len("data: "):])
                indices.add(payload["index"])
        assert indices == {0}

    def test_empty_arguments_still_produce_a_valid_delta(self):
        encoder = _encoder()
        event = CanonicalEvent(
            type="tool_use",
            index=0,
            content_block=CanonicalToolCallBlock(id="toolu_x", name="TaskList", arguments={}),
        )
        frame = encoder.encode(event)
        assert frame is not None
        delta_line = next(
            line for line in frame.splitlines()
            if line.startswith("data:") and '"content_block_delta"' in line
        )
        payload = json.loads(delta_line[len("data: "):])
        assert json.loads(payload["delta"]["partial_json"]) == {}
