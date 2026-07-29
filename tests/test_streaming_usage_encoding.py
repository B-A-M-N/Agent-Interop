"""MVP-10: usage_update must be encoded correctly by all three client protocols.

The Gateway now yields a usage_update CanonicalEvent before the terminal
message_stop in streaming mode. Each protocol's stream encoder must surface
that usage to the client rather than silently dropping it.
"""

from __future__ import annotations

import json

from agent_interop.abi import CanonicalEvent, CanonicalStopReason
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from agent_interop.protocols.openai_chat import OpenAIChatAdapter
from agent_interop.protocols.openai_responses import OpenAIResponsesAdapter


def _usage_event() -> CanonicalEvent:
    return CanonicalEvent(type="usage_update", input_tokens=12, output_tokens=7)


def _stop_event() -> CanonicalEvent:
    return CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.END_TURN)


class TestAnthropicUsageEncoding:
    def test_usage_reaches_message_delta(self):
        adapter = AnthropicMessagesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "msg_1", "model": "test-model"})

        assert encoder.encode(_usage_event()) is None  # tracked, not emitted standalone

        frame = encoder.encode(_stop_event())
        assert frame is not None
        assert "input_tokens" in frame
        assert '"input_tokens": 12' in frame
        assert '"output_tokens": 7' in frame


class TestOpenAIChatUsageEncoding:
    def test_usage_reaches_trailing_chunk(self):
        adapter = OpenAIChatAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "chatcmpl_1", "model": "test-model"})

        assert encoder.encode(_usage_event()) is None
        assert encoder.encode(_stop_event()) is not None

        trailer = encoder.finish()
        assert trailer is not None
        assert "prompt_tokens" in trailer
        data_lines = [line[len("data: "):] for line in trailer.split("\n\n") if line.startswith("data: ")]
        usage_payloads = [json.loads(d) for d in data_lines if d.strip() != "[DONE]"]
        assert any(p.get("usage", {}).get("prompt_tokens") == 12 for p in usage_payloads)
        assert any(p.get("usage", {}).get("completion_tokens") == 7 for p in usage_payloads)


class TestOpenAIResponsesUsageEncoding:
    def test_usage_reaches_response_completed(self):
        adapter = OpenAIResponsesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "resp_1", "model": "test-model"})

        assert encoder.encode(_usage_event()) is None

        frame = encoder.encode(_stop_event())
        assert frame is not None
        assert '"input_tokens": 12' in frame
        assert '"output_tokens": 7' in frame
