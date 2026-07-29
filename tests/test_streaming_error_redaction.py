"""MVP-08: streaming error frames must redact secrets too.

serialize_client_error() (used by the non-streaming path) already redacts
CanonicalError.message and sanitizes .details. The streaming encoders build
their error frames inline in encode() rather than through that function —
each one needs its own redaction wiring, verified here directly.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalError, CanonicalEvent, CanonicalStopReason
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from agent_interop.protocols.openai_chat import OpenAIChatAdapter
from agent_interop.protocols.openai_responses import OpenAIResponsesAdapter

SECRET_MESSAGE = "Upstream returned 401: Authorization: Bearer sk-super-secret-token-value"
SECRET_DETAILS = {"api_key": "should-not-leak", "hint": "safe to keep"}


def _error_event() -> CanonicalEvent:
    return CanonicalEvent(
        type="error",
        error=CanonicalError(code="BACKEND_ERROR", message=SECRET_MESSAGE, details=SECRET_DETAILS),
    )


def _stop_event() -> CanonicalEvent:
    return CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT)


class TestAnthropicStreamingErrorRedaction:
    def test_error_frame_redacted(self):
        adapter = AnthropicMessagesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "msg_1", "model": "test-model"})

        frame = encoder.encode(_error_event())
        assert frame is not None
        assert "sk-super-secret-token-value" not in frame
        assert "should-not-leak" not in frame
        assert "safe to keep" in frame

    def test_message_delta_terminal_redacted(self):
        adapter = AnthropicMessagesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "msg_1", "model": "test-model"})

        encoder.encode(_error_event())
        frame = encoder.encode(_stop_event())
        assert frame is not None
        assert "sk-super-secret-token-value" not in frame
        assert "should-not-leak" not in frame


class TestOpenAIChatStreamingErrorRedaction:
    def test_error_frame_redacted(self):
        adapter = OpenAIChatAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "chatcmpl_1", "model": "test-model"})

        frame = encoder.encode(_error_event())
        assert frame is not None
        assert "sk-super-secret-token-value" not in frame
        assert "should-not-leak" not in frame
        assert "safe to keep" in frame


class TestOpenAIResponsesStreamingErrorRedaction:
    def test_error_frame_redacted(self):
        adapter = OpenAIResponsesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "resp_1", "model": "test-model"})

        frame = encoder.encode(_error_event())
        assert frame is not None
        assert "sk-super-secret-token-value" not in frame
        assert "should-not-leak" not in frame
        assert "safe to keep" in frame

    def test_response_failed_terminal_redacted(self):
        adapter = OpenAIResponsesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "resp_1", "model": "test-model"})

        encoder.encode(_error_event())
        frame = encoder.encode(_stop_event())
        assert frame is not None
        assert "sk-super-secret-token-value" not in frame
