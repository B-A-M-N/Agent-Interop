"""Tests for Phase 3 codec correctness: capabilities, unknown content, error handling."""

from __future__ import annotations

from agent_interop.abi import CanonicalUnknownBlock
from agent_interop.upstreams.anthropic import AnthropicCodec
from agent_interop.upstreams.codec import CodecCapabilities, StreamFraming
from agent_interop.upstreams.ollama_chat import OllamaChatCodec
from agent_interop.upstreams.openai_chat import OpenAIChatCodec
from agent_interop.upstreams.openai_responses import OpenAIResponsesCodec

# ─── Codec capabilities (item 74) ──────────────────────────────────────────


class TestCodecCapabilities:
    def test_openai_chat_capabilities(self):
        caps = OpenAIChatCodec().capabilities()
        assert isinstance(caps, CodecCapabilities)
        assert caps.supports_native_tools is True
        assert caps.supports_parallel_tool_calls is True
        assert caps.streaming_framing == StreamFraming.SSE

    def test_ollama_capabilities(self):
        caps = OllamaChatCodec().capabilities()
        assert caps.supports_native_tools is True
        assert caps.supports_parallel_tool_calls is False
        assert caps.streaming_framing == StreamFraming.NDJSON
        assert caps.max_tools == 64

    def test_anthropic_capabilities(self):
        caps = AnthropicCodec().capabilities()
        assert caps.supports_native_tools is True
        assert caps.supports_vision is True
        assert caps.max_tools == 200

    def test_openai_responses_capabilities(self):
        caps = OpenAIResponsesCodec().capabilities()
        assert caps.supports_native_tools is True
        assert caps.supports_system_messages is False  # uses instructions

    def test_capabilities_default_streaming_framing(self):
        caps = OpenAIChatCodec().capabilities()
        assert caps.streaming_framing == StreamFraming.SSE
        caps = OllamaChatCodec().capabilities()
        assert caps.streaming_framing == StreamFraming.NDJSON


# ─── Unknown content preservation (item 68) ─────────────────────────────────


class TestUnknownContentPreservation:
    def test_anthropic_preserves_unknown_block_type(self):
        codec = AnthropicCodec()
        body = {
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "future_block_type", "data": {"key": "value"}},
            ],
            "stop_reason": "end_turn",
        }
        resp = codec.decode_response(body)
        unknown_blocks = [b for b in resp.content if isinstance(b, CanonicalUnknownBlock)]
        assert len(unknown_blocks) == 1
        assert unknown_blocks[0].source_type == "future_block_type"
        assert unknown_blocks[0].raw.get("data") == {"key": "value"}

    def test_openai_chat_preserves_refusal(self):
        codec = OpenAIChatCodec()
        body = {
            "choices": [{
                "message": {
                    "content": None,
                    "refusal": "I cannot help with that",
                },
                "finish_reason": "stop",
            }],
        }
        resp = codec.decode_response(body)
        unknown_blocks = [b for b in resp.content if isinstance(b, CanonicalUnknownBlock)]
        refusal_blocks = [b for b in unknown_blocks if b.source_type == "openai_refusal"]
        assert len(refusal_blocks) == 1
        assert refusal_blocks[0].raw == "I cannot help with that"

    def test_openai_responses_preserves_unknown_item_type(self):
        codec = OpenAIResponsesCodec()
        body = {
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]},
                {"type": "future_output_type", "data": [1, 2, 3]},
            ],
            "status": "completed",
        }
        resp = codec.decode_response(body)
        unknown_blocks = [b for b in resp.content if isinstance(b, CanonicalUnknownBlock)]
        assert len(unknown_blocks) == 1
        assert unknown_blocks[0].source_type == "responses_future_output_type"


# ─── Response ID extraction (item 67) ───────────────────────────────────────


class TestResponseIdExtraction:
    def test_openai_chat_response_id(self):
        codec = OpenAIChatCodec()
        body = {
            "id": "chatcmpl-abc123",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        }
        resp = codec.decode_response(body)
        assert resp.extra.get("response_id") == "chatcmpl-abc123"

    def test_anthropic_response_id(self):
        codec = AnthropicCodec()
        body = {
            "id": "msg_abc123",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        }
        resp = codec.decode_response(body)
        assert resp.extra.get("response_id") == "msg_abc123"

    def test_openai_responses_response_id(self):
        codec = OpenAIResponsesCodec()
        body = {
            "id": "resp_abc123",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}],
            "status": "completed",
        }
        resp = codec.decode_response(body)
        assert resp.extra.get("response_id") == "resp_abc123"

    def test_ollama_no_response_id(self):
        codec = OllamaChatCodec()
        body = {"message": {"content": "hi"}, "done": True}
        resp = codec.decode_response(body)
        # Ollama doesn't have a response ID
        assert resp.extra.get("response_id", "") == ""


# ─── Protocol identifiers typed (item 72) ────────────────────────────────────


class TestTypedProtocolIdentifiers:
    def test_anthropic_uses_protocol_kind_enum(self):
        from agent_interop.abi import ProtocolKind
        codec = AnthropicCodec()
        body = {
            "content": [{"type": "tool_use", "id": "tc_1", "name": "read", "input": {"p": "/x"}}],
            "stop_reason": "tool_use",
        }
        resp = codec.decode_response(body)
        assert len(resp.tool_candidates) == 1
        assert resp.tool_candidates[0].source_protocol == ProtocolKind.ANTHROPIC_MESSAGES

    def test_openai_uses_protocol_kind_enum(self):
        from agent_interop.abi import ProtocolKind
        codec = OpenAIChatCodec()
        body = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {"name": "read", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
        resp = codec.decode_response(body)
        assert len(resp.tool_candidates) == 1
        assert resp.tool_candidates[0].source_protocol == ProtocolKind.OPENAI_CHAT


# ─── Codec-specific stop reasons (item 73) ─────────────────────────────────


class TestCodecStopReasons:
    def test_anthropic_tool_use_stop(self):
        codec = AnthropicCodec()
        body = {
            "content": [{"type": "tool_use", "id": "tc_1", "name": "read", "input": {}}],
            "stop_reason": "tool_use",
        }
        from agent_interop.abi import CanonicalStopReason
        resp = codec.decode_response(body)
        assert resp.stop_reason == CanonicalStopReason.TOOL_CALL

    def test_ollama_tool_calls_stop(self):
        codec = OllamaChatCodec()
        body = {
            "message": {
                "tool_calls": [{"function": {"name": "read"}}],
            },
            "done": True,
            "done_reason": "tool_calls",
        }
        from agent_interop.abi import CanonicalStopReason
        resp = codec.decode_response(body)
        assert resp.stop_reason == CanonicalStopReason.TOOL_CALL

    def test_openai_responses_incomplete(self):
        codec = OpenAIResponsesCodec()
        body = {"output": [], "status": "incomplete"}
        from agent_interop.abi import CanonicalStopReason
        resp = codec.decode_response(body)
        assert resp.stop_reason == CanonicalStopReason.MAX_TOKENS
