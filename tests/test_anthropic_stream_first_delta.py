"""Regression test: AnthropicStreamEncoder must not drop the first
text_delta's content.

Found via a real live-client acceptance run (Claude Code 2.1.220 against
`gpt-oss:20b-cloud` through a real Interop gateway): the encoder's
first-delta branch emitted content_block_start with a hardcoded empty
``"text": ""`` and returned WITHOUT ever emitting a content_block_delta
carrying the actual text. Any response whose entire text arrives in a
single text_delta CanonicalEvent — which is exactly what the
BUFFER_TEXTUAL_RESPONSE path does (it buffers a whole prompted-mode turn
and yields it as one event once the stream ends) — lost its content
entirely: the client received content_block_start -> content_block_stop
with nothing in between and silently printed nothing.

Existing coverage (test_protocols.py's test_stream_event_encoding) did not
catch this because it exercises AnthropicMessagesAdapter.encode_event, a
separate stateless method never reached by the real server code path —
create_stream_encoder() is overridden to return the stateful
AnthropicStreamEncoder these tests exercise directly, matching what
server/app.py actually calls.
"""

from __future__ import annotations

import json

from agent_interop.abi import CanonicalEvent
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter


def _encoder():
    adapter = AnthropicMessagesAdapter()
    return adapter.create_stream_encoder({"response_id": "msg_1", "model": "test-model"})


class TestFirstTextDeltaNotDropped:
    def test_single_delta_carries_full_text(self):
        """The exact bug scenario: one text_delta event, e.g. from
        BUFFER_TEXTUAL_RESPONSE flushing an entire buffered turn at once."""
        encoder = _encoder()
        frame = encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="banana"))
        assert frame is not None
        assert "content_block_start" in frame
        assert "content_block_delta" in frame
        assert '"text": "banana"' in frame

    def test_first_frame_is_start_then_delta_in_order(self):
        encoder = _encoder()
        frame = encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="hi"))
        start_pos = frame.index("event: content_block_start")
        delta_pos = frame.index("event: content_block_delta")
        assert start_pos < delta_pos

    def test_multi_chunk_stream_preserves_every_chunk(self):
        """The bug was most visible when the whole reply arrives in one
        shot, but a normal multi-chunk stream must also lose nothing —
        each chunk after the first already worked; this locks that in."""
        encoder = _encoder()
        frames = [
            encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="Hel")),
            encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="lo")),
            encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="!")),
        ]
        joined = "".join(f for f in frames if f)
        assert '"text": "Hel"' in joined
        assert '"text": "lo"' in joined
        assert '"text": "!"' in joined

    def test_empty_first_partial_does_not_emit_spurious_delta(self):
        """An empty-string first delta (e.g. a decoder that fires on a
        zero-length chunk) must still only emit content_block_start, not a
        content_block_delta with nothing in it."""
        encoder = _encoder()
        frame = encoder.encode(CanonicalEvent(type="text_delta", index=0, partial=""))
        assert frame is not None
        assert "content_block_start" in frame
        assert "content_block_delta" not in frame

    def test_content_block_delta_json_is_well_formed(self):
        encoder = _encoder()
        frame = encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="banana"))
        delta_data_line = next(
            line for line in frame.splitlines()
            if line.startswith("data:") and '"content_block_delta"' in line
        )
        payload = json.loads(delta_data_line[len("data: "):])
        assert payload["delta"]["text"] == "banana"
        assert payload["delta"]["type"] == "text_delta"
        assert payload["index"] == 0

    def test_index_matches_between_start_and_delta(self):
        """A second, independent text block (index 1) must keep its own
        start/delta pair consistent — not accidentally reuse index 0."""
        encoder = _encoder()
        encoder.encode(CanonicalEvent(type="text_delta", index=0, partial="first"))
        encoder.encode(CanonicalEvent(type="content_block_stop"))
        frame = encoder.encode(CanonicalEvent(type="text_delta", index=1, partial="second"))
        start_line = next(
            line for line in frame.splitlines()
            if line.startswith("data:") and '"content_block_start"' in line
        )
        delta_line = next(
            line for line in frame.splitlines()
            if line.startswith("data:") and '"content_block_delta"' in line
        )
        start_payload = json.loads(start_line[len("data: "):])
        delta_payload = json.loads(delta_line[len("data: "):])
        assert start_payload["index"] == delta_payload["index"] == 1
