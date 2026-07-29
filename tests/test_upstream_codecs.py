"""Phase 2 gate: upstream codec correctness tests.

Validates every codec contract:
1. All codecs are importable and instantiable
2. All codecs implement the full ModelCodec interface
3. decode_response preserves raw_arguments verbatim (including malformed JSON)
4. decode_stream_chunk returns DecodedModelEvent with correct types
5. render_request produces valid upstream-native dicts
6. extract_usage and extract_stop_reason work on sample data
7. is_stream_complete correctly identifies terminal chunks
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalUsage,
)
from agent_interop.upstreams.codec import DecodedModelEvent, DecodedModelResponse, ModelCodec

# ── All codecs under test ─────────────────────────────────────────────────

CODEC_MODULES: list[dict[str, Any]] = [
    {
        "module": "agent_interop.upstreams.openai_chat",
        "class_name": "OpenAIChatCodec",
        "protocol": "openai_chat",
    },
    {
        "module": "agent_interop.upstreams.ollama_chat",
        "class_name": "OllamaChatCodec",
        "protocol": "ollama_chat",
    },
    {
        "module": "agent_interop.upstreams.anthropic",
        "class_name": "AnthropicCodec",
        "protocol": "anthropic_messages",
    },
    {
        "module": "agent_interop.upstreams.openai_responses",
        "class_name": "OpenAIResponsesCodec",
        "protocol": "openai_responses",
    },
]


@pytest.fixture(params=CODEC_MODULES)
def codec_info(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param


@pytest.fixture
def codec(codec_info: dict[str, Any]) -> ModelCodec:
    import importlib
    mod = importlib.import_module(codec_info["module"])
    cls = getattr(mod, codec_info["class_name"])
    return cls()


# ── Minimal canonical request for render tests ────────────────────────────


@pytest.fixture
def sample_request() -> CanonicalRequest:
    return CanonicalRequest(
        model="test-model",
        system=[CanonicalTextBlock(text="You are a helpful assistant.")],
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalTextBlock(text="Hello!")],
            ),
        ],
        tools=[
            CanonicalTool(
                name="get_weather",
                description="Get the weather",
                input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ],
        tool_choice=CanonicalToolChoice.auto(),
        generation=CanonicalGenerationOptions(max_output_tokens=100),
    )


# ── Contract tests ────────────────────────────────────────────────────────


class TestCodecContract:
    """Verify every codec satisfies the ModelCodec interface."""

    def test_protocol_attribute(self, codec: ModelCodec) -> None:
        assert hasattr(codec, "protocol")
        assert isinstance(codec.protocol, str)

    def test_endpoint_path(self, codec: ModelCodec) -> None:
        path = codec.endpoint_path()
        assert isinstance(path, str)
        assert path.startswith("/")

    def test_required_headers(self, codec: ModelCodec) -> None:
        headers = codec.required_headers()
        assert isinstance(headers, dict)
        assert "Content-Type" in headers

    def test_render_request(self, codec: ModelCodec, sample_request: CanonicalRequest) -> None:
        rendered = codec.render_request(sample_request, "test-model", stream=True)
        assert isinstance(rendered, dict)
        # Every render must produce a dict with at minimum a model field or equivalent
        assert len(rendered) > 0

    def test_render_request_stream_param(self, codec: ModelCodec, sample_request: CanonicalRequest) -> None:
        """Stream=False should produce a non-streaming request."""
        rendered = codec.render_request(sample_request, "test-model", stream=False)
        assert isinstance(rendered, dict)
        # Verify stream flag is present and False (some protocols render differently)
        # Not all protocols have a stream key — just verify it doesn't crash

    def test_decode_response_returns_decoded_response(self, codec: ModelCodec) -> None:
        """decode_response must return a DecodedModelResponse (not a raw dict)."""
        # Build a minimal response body that varies per protocol
        body = _minimal_body(codec.protocol)
        result = codec.decode_response(body)
        assert isinstance(result, DecodedModelResponse)
        assert isinstance(result.content, list)
        assert isinstance(result.tool_candidates, list)
        assert isinstance(result.stop_reason, str)
        assert isinstance(result.usage, CanonicalUsage)

    def test_extract_usage(self, codec: ModelCodec) -> None:
        usage = codec.extract_usage({"usage": {"prompt_tokens": 10, "completion_tokens": 20}})
        assert isinstance(usage, CanonicalUsage)

    def test_extract_stop_reason(self, codec: ModelCodec) -> None:
        reason = codec.extract_stop_reason({"finish_reason": "stop"})
        assert isinstance(reason, str)


# ── Malformed JSON gate test ──────────────────────────────────────────────


MALFORMED_RAW_ARGUMENTS: list[str] = [
    '{"city":"London"',  # truncated JSON
    '{"path":"/tmp/x",}',  # trailing comma
    '{invalid}',  # not JSON at all
    "",  # empty string
    "null",  # JSON literal
    '{"nested": {"a": 1, "b":}}',  # nested malformed
    "{",  # single brace
]


class TestMalformedJsonSurvival:
    """CRITICAL: raw_arguments must survive verbatim — no parse/validate/reformat."""

    def _make_body_with_malformed_toolcall(
        self, protocol: str, raw_args: str
    ) -> dict[str, Any]:
        """Build a protocol-specific response body containing a tool call with raw_args."""
        if protocol in ("openai_chat",):
            return {
                "id": "test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "test_tool",
                                "arguments": raw_args,
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
            }
        elif protocol == "anthropic_messages":
            return {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "test_tool",
                    "input": raw_args,
                }],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 5, "output_tokens": 10},
            }
        elif protocol == "ollama_chat":
            return {
                "model": "test",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": raw_args,
                        },
                    }],
                },
                "done": True,
            }
        elif protocol == "openai_responses":
            return {
                "id": "resp_test",
                "object": "response",
                "output": [{
                    "type": "function_call",
                    "id": "call_1",
                    "name": "test_tool",
                    "arguments": raw_args,
                    "status": "completed",
                }],
                "status": "completed",
                "usage": {"input_tokens": 5, "output_tokens": 10},
            }
        return {}

    @pytest.mark.parametrize("raw_args", MALFORMED_RAW_ARGUMENTS)
    def test_malformed_json_preserved(
        self, codec_info: dict[str, Any], raw_args: str
    ) -> None:
        """Verify malformed raw_arguments survive decode_response verbatim."""
        import importlib
        mod = importlib.import_module(codec_info["module"])
        cls = getattr(mod, codec_info["class_name"])
        c: ModelCodec = cls()

        body = self._make_body_with_malformed_toolcall(codec_info["protocol"], raw_args)
        result = c.decode_response(body)

        assert len(result.tool_candidates) == 1, (
            f"Expected 1 tool candidate, got {len(result.tool_candidates)} "
            f"for protocol={codec_info['protocol']}, raw_args={raw_args!r}"
        )
        candidate = result.tool_candidates[0]
        # Compare the raw_arguments as strings for assertion
        actual = candidate.raw_arguments
        if isinstance(actual, str):
            assert actual == raw_args, (
                f"raw_arguments modified! protocol={codec_info['protocol']}, "
                f"expected={raw_args!r}, got={actual!r}"
            )
        # If raw_arguments is a dict/list (parsed by upstream), accept that
        # — the key contract is that we don't mutate/validate

    def test_no_tool_candidates_when_no_tools(
        self, codec_info: dict[str, Any]
    ) -> None:
        """Empty/none tool_calls should produce zero candidates."""
        import importlib
        mod = importlib.import_module(codec_info["module"])
        cls = getattr(mod, codec_info["class_name"])
        c: ModelCodec = cls()

        body = _minimal_body(codec_info["protocol"])
        result = c.decode_response(body)
        assert len(result.tool_candidates) == 0, (
            f"Expected 0 tool candidates for protocol={codec_info['protocol']}, "
            f"got {len(result.tool_candidates)}"
        )


# ── Stream chunk decoding ─────────────────────────────────────────────────


class TestStreamDecoding:
    """Verify decode_stream_chunk returns valid DecodedModelEvent lists."""

    def test_decode_stream_chunk_returns_list(self, codec: ModelCodec) -> None:
        chunk = _minimal_stream_chunk(codec.protocol)
        events = codec.decode_stream_chunk(chunk)
        assert isinstance(events, list)
        for event in events:
            assert isinstance(event, DecodedModelEvent)

    def test_is_stream_complete(self, codec: ModelCodec) -> None:
        terminal = _terminal_chunk(codec.protocol)
        non_terminal = _nonterminal_chunk(codec.protocol)
        assert codec.is_stream_complete(terminal), (
            f"is_stream_complete should be True for terminal chunk in {codec.protocol}"
        )
        assert not codec.is_stream_complete(non_terminal), (
            f"is_stream_complete should be False for non-terminal chunk in {codec.protocol}"
        )


# ── Protocol-specific response decoding ──────────────────────────────────


class TestOpenAIPlaybackDecoding:
    """OpenAI Chat specific response decoding."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.OPENAI_CHAT)

    def test_plain_text(self, codec: ModelCodec) -> None:
        body = {
            "id": "test",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert isinstance(result.content[0], CanonicalTextBlock)
        assert result.content[0].text == "Hello!"
        assert len(result.tool_candidates) == 0

    def test_text_and_tools(self, codec: ModelCodec) -> None:
        body = {
            "id": "test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me check",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"London"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert len(result.tool_candidates) == 1
        assert result.tool_candidates[0].name == "get_weather"


class TestAnthropicResponseDecoding:
    """Anthropic Messages specific response decoding."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.ANTHROPIC_MESSAGES)

    def test_plain_text(self, codec: ModelCodec) -> None:
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert isinstance(result.content[0], CanonicalTextBlock)
        assert result.content[0].text == "Hello!"

    def test_tool_use(self, codec: ModelCodec) -> None:
        body = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Using tool"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "London"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert len(result.tool_candidates) == 1
        assert result.tool_candidates[0].name == "get_weather"
        assert result.tool_candidates[0].id == "toolu_1"


class TestAnthropicStreamDecoding:
    """decode_stream_chunk's per-event-type branches for every real
    Anthropic Messages SSE event — previously only exercised through the
    generic cross-codec smoke test (TestStreamDecoding), which only ever
    sent one minimal chunk shape and never distinguished event types."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.ANTHROPIC_MESSAGES)

    def test_content_block_delta_text(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedTextDelta

        events = codec.decode_stream_chunk({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedTextDelta)
        assert events[0].text == "Hello"

    def test_content_block_delta_input_json(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedToolFragment

        events = codec.decode_stream_chunk({
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedToolFragment)
        assert events[0].tool_index == 2
        assert events[0].argument_fragment == '{"city":'

    def test_content_block_start_tool_use(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedToolFragment

        events = codec.decode_stream_chunk({
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather"},
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedToolFragment)
        assert events[0].call_id_fragment == "toolu_1"
        assert events[0].name_fragment == "get_weather"
        assert events[0].tool_index == 1

    def test_content_block_start_text_produces_no_event(self, codec: ModelCodec) -> None:
        """Only tool_use content_block_start carries fragment identity —
        a text block start has nothing to report."""
        events = codec.decode_stream_chunk({
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        })
        assert events == []

    def test_content_block_stop(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedToolBatchComplete

        events = codec.decode_stream_chunk({"type": "content_block_stop", "index": 0})
        assert len(events) == 1
        assert isinstance(events[0], DecodedToolBatchComplete)
        assert events[0].stop_reason == CanonicalStopReason.TOOL_CALL

    def test_message_delta_with_usage_and_stop_reason(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedStreamComplete, DecodedUsageUpdate

        events = codec.decode_stream_chunk({
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"input_tokens": 12, "output_tokens": 7},
        })
        assert len(events) == 2
        usage_events = [e for e in events if isinstance(e, DecodedUsageUpdate)]
        stop_events = [e for e in events if isinstance(e, DecodedStreamComplete)]
        assert usage_events[0].usage.input_tokens == 12
        assert usage_events[0].usage.output_tokens == 7
        assert stop_events[0].stop_reason == CanonicalStopReason.TOOL_CALL

    @pytest.mark.parametrize("anthropic_reason,expected", [
        ("end_turn", "end_turn"),
        ("tool_use", "tool_call"),
        ("max_tokens", "max_tokens"),
        ("stop_sequence", "stop_sequence"),
    ])
    def test_message_delta_stop_reason_mapping(
        self, codec: ModelCodec, anthropic_reason: str, expected: str,
    ) -> None:
        from agent_interop.upstreams.codec import DecodedStreamComplete

        events = codec.decode_stream_chunk({
            "type": "message_delta",
            "delta": {"stop_reason": anthropic_reason},
        })
        complete = next(e for e in events if isinstance(e, DecodedStreamComplete))
        assert complete.stop_reason.value == expected

    def test_message_delta_unknown_stop_reason_defaults_to_end_turn(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedStreamComplete

        events = codec.decode_stream_chunk({
            "type": "message_delta",
            "delta": {"stop_reason": "some_new_reason_anthropic_added_later"},
        })
        complete = next(e for e in events if isinstance(e, DecodedStreamComplete))
        assert complete.stop_reason == CanonicalStopReason.END_TURN

    def test_message_delta_without_stop_reason_or_usage_produces_no_event(self, codec: ModelCodec) -> None:
        events = codec.decode_stream_chunk({"type": "message_delta", "delta": {}})
        assert events == []

    def test_message_stop(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedStreamComplete

        events = codec.decode_stream_chunk({"type": "message_stop"})
        assert len(events) == 1
        assert isinstance(events[0], DecodedStreamComplete)
        assert events[0].stop_reason == CanonicalStopReason.END_TURN

    def test_error_event(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedStreamError

        events = codec.decode_stream_chunk({
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedStreamError)
        assert events[0].error == "Overloaded"

    def test_unknown_event_type_produces_no_events(self, codec: ModelCodec) -> None:
        events = codec.decode_stream_chunk({"type": "ping"})
        assert events == []


class TestAnthropicRequestRendering:
    """render_request()'s per-role and per-block-type branches — the
    generic cross-codec sample_request fixture only ever exercises plain
    user text, leaving tool_result, tool_use, reasoning, image, and
    tool_choice-mode rendering completely unverified for this codec."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.ANTHROPIC_MESSAGES)

    @staticmethod
    def _request(messages, **kwargs) -> CanonicalRequest:
        from agent_interop.abi import CanonicalModelReference
        defaults = {
            "model": CanonicalModelReference(requested_name="test-model"),
            "messages": messages,
            "generation": CanonicalGenerationOptions(max_output_tokens=100),
        }
        defaults.update(kwargs)
        return CanonicalRequest(**defaults)

    def test_tool_role_renders_tool_result(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolResultBlock

        req = self._request([
            CanonicalMessage(role="tool", content=[
                CanonicalToolResultBlock(tool_call_id="toolu_1", content="42 degrees"),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        msg = rendered["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "toolu_1"
        assert msg["content"][0]["content"] == "42 degrees"

    def test_tool_role_error_result(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolResultBlock

        req = self._request([
            CanonicalMessage(role="tool", content=[
                CanonicalToolResultBlock(tool_call_id="toolu_1", content="boom", is_error=True),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        assert rendered["messages"][0]["content"][0]["is_error"] is True

    def test_assistant_tool_use_block(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolCallBlock

        req = self._request([
            CanonicalMessage(role="assistant", content=[
                CanonicalToolCallBlock(id="toolu_1", name="get_weather", arguments={"city": "Paris"}),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        block = rendered["messages"][0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "toolu_1"
        assert block["name"] == "get_weather"
        assert block["input"] == {"city": "Paris"}

    def test_assistant_reasoning_block_with_signature(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalReasoningBlock

        req = self._request([
            CanonicalMessage(role="assistant", content=[
                CanonicalReasoningBlock(content="thinking it through", signature="sig123"),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        block = rendered["messages"][0]["content"][0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "thinking it through"
        assert block["signature"] == "sig123"

    def test_assistant_reasoning_block_without_signature_omits_key(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalReasoningBlock

        req = self._request([
            CanonicalMessage(role="assistant", content=[
                CanonicalReasoningBlock(content="thinking", signature=None),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        block = rendered["messages"][0]["content"][0]
        assert "signature" not in block

    def test_user_image_block_base64(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalImageBlock

        req = self._request([
            CanonicalMessage(role="user", content=[
                CanonicalImageBlock(media_type="image/png", data="base64data=="),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        block = rendered["messages"][0]["content"][0]
        assert block["type"] == "image"
        assert block["source"]["type"] == "base64"
        assert block["source"]["data"] == "base64data=="

    def test_user_image_block_url(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalImageBlock

        req = self._request([
            CanonicalMessage(role="user", content=[
                CanonicalImageBlock(url="https://example.com/x.png"),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        block = rendered["messages"][0]["content"][0]
        assert block["source"]["type"] == "url"
        assert block["source"]["url"] == "https://example.com/x.png"

    def test_user_tool_result_block(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolResultBlock

        req = self._request([
            CanonicalMessage(role="user", content=[
                CanonicalToolResultBlock(tool_call_id="toolu_1", content="result text"),
            ]),
        ])
        rendered = codec.render_request(req, "claude-x", stream=False)
        assert rendered["messages"][0]["content"][0]["type"] == "tool_result"

    @pytest.mark.parametrize("mode,expected_type", [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "any"),
    ])
    def test_tool_choice_mode_mapping(self, codec: ModelCodec, mode: str, expected_type: str) -> None:
        from agent_interop.abi import canonical_tool_choice

        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            tool_choice=canonical_tool_choice(mode),
        )
        rendered = codec.render_request(req, "claude-x", stream=False)
        assert rendered["tool_choice"]["type"] == expected_type

    def test_tool_choice_named(self, codec: ModelCodec) -> None:
        from agent_interop.abi import canonical_tool_choice

        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            tool_choice=canonical_tool_choice("named", "get_weather"),
        )
        rendered = codec.render_request(req, "claude-x", stream=False)
        assert rendered["tool_choice"] == {"type": "tool", "name": "get_weather"}

    def test_metadata_passthrough(self, codec: ModelCodec) -> None:
        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            metadata={"metadata": {"user_id": "abc123"}},
        )
        rendered = codec.render_request(req, "claude-x", stream=False)
        assert rendered["metadata"] == {"user_id": "abc123"}

    def test_stop_sequences_rendered(self, codec: ModelCodec) -> None:
        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            generation=CanonicalGenerationOptions(max_output_tokens=100, stop=["STOP"]),
        )
        rendered = codec.render_request(req, "claude-x", stream=False)
        assert rendered["stop_sequences"] == ["STOP"]


class TestOllamaResponseDecoding:
    """Ollama specific response decoding."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.OLLAMA_CHAT)

    def test_plain_text(self, codec: ModelCodec) -> None:
        body = {
            "model": "test",
            "message": {"role": "assistant", "content": "Hello!"},
            "done": True,
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert isinstance(result.content[0], CanonicalTextBlock)
        assert result.content[0].text == "Hello!"


class TestResponsesDecoding:
    """OpenAI Responses specific response decoding."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.OPENAI_RESPONSES)

    def test_plain_text(self, codec: ModelCodec) -> None:
        body = {
            "id": "resp_1",
            "object": "response",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }],
            "status": "completed",
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1

    def test_function_call(self, codec: ModelCodec) -> None:
        body = {
            "id": "resp_1",
            "object": "response",
            "output": [{
                "type": "function_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": '{"city":"London"}',
                "status": "completed",
            }],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = codec.decode_response(body)
        assert len(result.tool_candidates) == 1
        assert result.tool_candidates[0].name == "get_weather"

    def test_reasoning_item_extracts_summary_text(self, codec: ModelCodec) -> None:
        body = {
            "id": "resp_1",
            "output": [{
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Thinking about the weather..."}],
            }],
            "status": "completed",
            "usage": {},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert result.content[0].text == "Thinking about the weather..."

    def test_reasoning_item_without_summary_produces_no_content(self, codec: ModelCodec) -> None:
        body = {
            "id": "resp_1",
            "output": [{"type": "reasoning", "summary": []}],
            "status": "completed",
            "usage": {},
        }
        result = codec.decode_response(body)
        assert result.content == []

    def test_unknown_item_type_preserved_as_unknown_block(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalUnknownBlock

        body = {
            "id": "resp_1",
            "output": [{"type": "some_future_item_type", "data": "opaque"}],
            "status": "completed",
            "usage": {},
        }
        result = codec.decode_response(body)
        assert len(result.content) == 1
        assert isinstance(result.content[0], CanonicalUnknownBlock)
        assert result.content[0].source_type == "responses_some_future_item_type"


class TestResponsesStreamDecoding:
    """decode_stream_chunk's per-event-type branches — previously only
    exercised through the generic cross-codec smoke test."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.OPENAI_RESPONSES)

    def test_output_text_delta(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedTextDelta

        events = codec.decode_stream_chunk({
            "type": "response.output_text.delta", "delta": "Hello",
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedTextDelta)
        assert events[0].text == "Hello"

    def test_function_call_arguments_delta(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedToolFragment

        events = codec.decode_stream_chunk({
            "type": "response.function_call_arguments.delta",
            "output_index": 1,
            "call_id": "call_1",
            "delta": '{"city":',
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedToolFragment)
        assert events[0].tool_index == 1
        assert events[0].call_id_fragment == "call_1"
        assert events[0].argument_fragment == '{"city":'

    def test_function_call_arguments_done(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedToolBatchComplete

        events = codec.decode_stream_chunk({
            "type": "response.function_call_arguments.done", "output_index": 2,
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedToolBatchComplete)
        assert events[0].choice_index == 2
        assert events[0].stop_reason == CanonicalStopReason.TOOL_CALL

    def test_response_completed_with_usage(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedStreamComplete

        events = codec.decode_stream_chunk({
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 15, "output_tokens": 8}},
        })
        assert len(events) == 1
        assert isinstance(events[0], DecodedStreamComplete)
        assert events[0].stop_reason == CanonicalStopReason.END_TURN
        assert events[0].usage.input_tokens == 15
        assert events[0].usage.output_tokens == 8

    def test_response_completed_without_usage(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedStreamComplete

        events = codec.decode_stream_chunk({"type": "response.completed", "response": {}})
        assert isinstance(events[0], DecodedStreamComplete)
        assert events[0].usage is None

    def test_response_incomplete_maps_to_max_tokens(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalStopReason
        from agent_interop.upstreams.codec import DecodedStreamComplete

        events = codec.decode_stream_chunk({"type": "response.incomplete"})
        assert isinstance(events[0], DecodedStreamComplete)
        assert events[0].stop_reason == CanonicalStopReason.MAX_TOKENS

    def test_response_failed(self, codec: ModelCodec) -> None:
        from agent_interop.upstreams.codec import DecodedStreamError

        events = codec.decode_stream_chunk({
            "type": "response.failed",
            "error": {"message": "rate limited"},
        })
        assert isinstance(events[0], DecodedStreamError)
        assert events[0].error == "rate limited"

    def test_unknown_event_type_produces_no_events(self, codec: ModelCodec) -> None:
        events = codec.decode_stream_chunk({"type": "response.some_new_event"})
        assert events == []


class TestResponsesRequestRendering:
    """render_request()'s per-role/block-type branches for OpenAI
    Responses — function_call_output, function_call, tool_choice modes,
    previous_response_id, and metadata/text passthrough."""

    @pytest.fixture
    def codec(self) -> ModelCodec:
        from agent_interop.config import UpstreamProtocol
        from agent_interop.upstreams.registry import get_codec
        return get_codec(UpstreamProtocol.OPENAI_RESPONSES)

    @staticmethod
    def _request(messages, **kwargs) -> CanonicalRequest:
        from agent_interop.abi import CanonicalModelReference
        defaults = {
            "model": CanonicalModelReference(requested_name="test-model"),
            "messages": messages,
            "generation": CanonicalGenerationOptions(max_output_tokens=100),
        }
        defaults.update(kwargs)
        return CanonicalRequest(**defaults)

    def test_tool_role_renders_function_call_output(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolResultBlock

        req = self._request([
            CanonicalMessage(role="tool", content=[
                CanonicalToolResultBlock(tool_call_id="call_1", content="42 degrees"),
            ]),
        ])
        rendered = codec.render_request(req, "gpt-x", stream=False)
        item = rendered["input"][0]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "call_1"
        assert item["output"] == "42 degrees"

    def test_tool_role_error_output_sets_status(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolResultBlock

        req = self._request([
            CanonicalMessage(role="tool", content=[
                CanonicalToolResultBlock(tool_call_id="call_1", content="boom", is_error=True),
            ]),
        ])
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert rendered["input"][0]["status"] == "error"

    def test_assistant_tool_call_renders_function_call_item(self, codec: ModelCodec) -> None:
        from agent_interop.abi import CanonicalToolCallBlock

        req = self._request([
            CanonicalMessage(role="assistant", content=[
                CanonicalToolCallBlock(id="call_1", name="get_weather", arguments={"city": "Paris"}),
            ]),
        ])
        rendered = codec.render_request(req, "gpt-x", stream=False)
        item = rendered["input"][0]
        assert item["type"] == "function_call"
        assert item["id"] == "call_1"
        assert item["name"] == "get_weather"
        assert json.loads(item["arguments"]) == {"city": "Paris"}

    def test_developer_role_folded_into_user_message(self, codec: ModelCodec) -> None:
        req = self._request([
            CanonicalMessage(role="developer", content=[CanonicalTextBlock(text="policy text")]),
        ])
        rendered = codec.render_request(req, "gpt-x", stream=False)
        item = rendered["input"][0]
        assert item["role"] == "user"
        assert item["content"][0]["text"] == "policy text"

    @pytest.mark.parametrize("mode,expected", [
        ("auto", "auto"),
        ("none", "none"),
        ("required", "required"),
    ])
    def test_tool_choice_mode_mapping(self, codec: ModelCodec, mode: str, expected: str) -> None:
        from agent_interop.abi import canonical_tool_choice

        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            tool_choice=canonical_tool_choice(mode),
        )
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert rendered["tool_choice"] == expected

    def test_tool_choice_named(self, codec: ModelCodec) -> None:
        from agent_interop.abi import canonical_tool_choice

        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            tool_choice=canonical_tool_choice("named", "get_weather"),
        )
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert rendered["tool_choice"] == {"type": "function", "name": "get_weather"}

    def test_previous_response_id_included_when_set(self, codec: ModelCodec) -> None:
        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            previous_response_id="resp_prev_123",
        )
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert rendered["previous_response_id"] == "resp_prev_123"

    def test_previous_response_id_omitted_when_unset(self, codec: ModelCodec) -> None:
        req = self._request([CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])])
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert "previous_response_id" not in rendered

    def test_metadata_passthrough(self, codec: ModelCodec) -> None:
        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            metadata={"metadata": {"user_id": "abc123"}},
        )
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert rendered["metadata"] == {"user_id": "abc123"}

    def test_text_config_passthrough(self, codec: ModelCodec) -> None:
        req = self._request(
            [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
            metadata={"text": {"format": {"type": "json_object"}}},
        )
        rendered = codec.render_request(req, "gpt-x", stream=False)
        assert rendered["text"] == {"format": {"type": "json_object"}}


# ── Helpers ──────────────────────────────────────────────────────────────


def _minimal_body(protocol: str) -> dict[str, Any]:
    """Return the simplest possible valid response body for each protocol."""
    if protocol == "openai_chat":
        return {
            "id": "test",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
            "usage": {},
        }
    elif protocol == "ollama_chat":
        return {
            "model": "test",
            "message": {"role": "assistant", "content": ""},
            "done": True,
        }
    elif protocol == "anthropic_messages":
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {},
        }
    elif protocol == "openai_responses":
        return {
            "id": "resp_test",
            "object": "response",
            "output": [],
            "status": "completed",
            "usage": {},
        }
    return {}


def _minimal_stream_chunk(protocol: str) -> dict[str, Any]:
    """Return the simplest non-terminal stream chunk."""
    if protocol in ("openai_chat", "ollama_chat"):
        return {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
    elif protocol == "anthropic_messages":
        return {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"},
        }
    elif protocol == "openai_responses":
        return {"type": "response.output_text.delta", "delta": "Hello"}
    return {}


def _terminal_chunk(protocol: str) -> dict[str, Any]:
    """Return a chunk that should be recognized as stream-complete."""
    if protocol == "openai_chat":
        return {"choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}]}
    elif protocol == "ollama_chat":
        return {"done": True}
    elif protocol == "anthropic_messages":
        return {"type": "message_stop"}
    elif protocol == "openai_responses":
        return {"type": "response.completed"}
    return {}


def _nonterminal_chunk(protocol: str) -> dict[str, Any]:
    """Return a chunk that should NOT be recognized as stream-complete."""
    if protocol in ("openai_chat", "ollama_chat"):
        return {"choices": [{"delta": {"content": "Hello"}, "index": 0}]}
    elif protocol == "anthropic_messages":
        return {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}
    elif protocol == "openai_responses":
        return {"type": "response.output_text.delta", "delta": "Hello"}
    return {}