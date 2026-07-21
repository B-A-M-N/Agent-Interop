"""Tests for protocol adapters — request/response translation."""

from __future__ import annotations

import json

from interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from interop.protocols.openai_chat import OpenAIChatAdapter
from interop.protocols.openai_responses import OpenAIResponsesAdapter
from interop.protocols.registry import detect_protocol, get_adapter
from interop.types import (
    CanonicalEvent,
    CanonicalTool,
    ContentBlock,
    ProtocolKind,
    ToolCall,
)


class TestDetectProtocol:
    def test_anthropic_by_path(self):
        kind = detect_protocol("/v1/messages", {}, {})
        assert kind == ProtocolKind.ANTHROPIC_MESSAGES

    def test_openai_chat_by_path(self):
        kind = detect_protocol("/v1/chat/completions", {}, {})
        assert kind == ProtocolKind.OPENAI_CHAT

    def test_openai_responses_by_path(self):
        kind = detect_protocol("/v1/responses", {}, {})
        assert kind == ProtocolKind.OPENAI_RESPONSES

    def test_anthropic_by_header(self):
        kind = detect_protocol("/unknown/path", {"anthropic-version": "2023-06-01"}, {})
        assert kind == ProtocolKind.ANTHROPIC_MESSAGES

    def test_responses_by_body(self):
        kind = detect_protocol("/unknown", {}, {
            "input": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "test"}],
        })
        assert kind == ProtocolKind.OPENAI_RESPONSES

    def test_chat_by_body_default(self):
        kind = detect_protocol("/unknown", {}, {"messages": [{"role": "user", "content": "hi"}]})
        assert kind == ProtocolKind.OPENAI_CHAT

    def test_get_adapter(self):
        adapter = get_adapter(ProtocolKind.ANTHROPIC_MESSAGES)
        assert adapter.protocol == ProtocolKind.ANTHROPIC_MESSAGES

        adapter2 = get_adapter(ProtocolKind.ANTHROPIC_MESSAGES)
        assert adapter is adapter2  # cached


class TestAnthropicMessages:
    adapter = AnthropicMessagesAdapter()

    def test_decode_simple(self):
        body = {
            "model": "claude-sonnet-4",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "max_tokens": 100,
            "stream": False,
        }
        req = self.adapter.decode_request(body, {})
        assert len(req.messages) == 2
        assert req.messages[0].role == "user"
        assert req.messages[0].content == "Hello"
        assert req.messages[1].role == "assistant"
        assert req.messages[1].content == "Hi there"

    def test_decode_with_system(self):
        body = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        req = self.adapter.decode_request(body, {})
        assert req.system == "You are helpful."

    def test_decode_tool_use_content_blocks(self):
        body = {
            "system": "Be helpful.",
            "messages": [
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll read it."},
                        {
                            "type": "tool_use",
                            "id": "toolu_abc123",
                            "name": "read_file",
                            "input": {"path": "/tmp/test.txt"},
                        },
                    ],
                },
            ],
            "tools": [{"name": "read_file", "description": "Read a file",
                       "input_schema": {
                           "type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"],
                       }}],
            "tool_choice": {"type": "auto"},
        }
        req = self.adapter.decode_request(body, {})
        assert len(req.messages) == 2
        assert len(req.tools) == 1
        assert req.tools[0].name == "read_file"

        assistant = req.messages[1]
        assert assistant.role == "assistant"
        if isinstance(assistant.content, list):
            assert len(assistant.content) == 2
            assert assistant.content[0].type == "text"
            assert assistant.content[1].type == "tool_use"
            assert assistant.content[1].tool_call is not None
            assert assistant.content[1].tool_call.id == "toolu_abc123"
            assert assistant.content[1].tool_call.name == "read_file"

    def test_decode_tool_result(self):
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": "File contents here",
                        }
                    ],
                }
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert len(req.messages) == 1
        assert req.messages[0].role == "tool"
        assert req.messages[0].tool_call_id == "toolu_abc123"

    def test_stream_event_encoding(self):
        event = CanonicalEvent(type="text_delta", index=0, partial="Hello")
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "event: content_block_delta" in sse
        assert "text_delta" in sse
        assert "Hello" in sse

        event2 = CanonicalEvent(type="message_stop")
        sse2 = self.adapter.encode_stream_event(event2)
        assert sse2 is not None
        assert "event: message_stop" in sse2

    def test_stream_text_start(self):
        event = CanonicalEvent(type="text", index=0, partial="Hello")
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "event: content_block_start" in sse
        assert "Hello" in sse

    def test_stream_tool_use(self):
        tc = ToolCall(id="toolu_abc", name="read_file", arguments={"path": "/tmp/x"})
        event = CanonicalEvent(
            type="tool_use",
            index=1,
            content_block=ContentBlock(type="tool_use", tool_call=tc),
        )
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "event: content_block_start" in sse
        assert "tool_use" in sse
        assert "toolu_abc" in sse

    def test_stream_thinking(self):
        event = CanonicalEvent(type="thinking_delta", index=0, partial="thinking text")
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "thinking_delta" in sse

    def test_parse_tool_result_text(self):
        result = self.adapter.parse_tool_result({"content": "file content"})
        assert result == "file content"

    def test_parse_tool_result_list(self):
        result = self.adapter.parse_tool_result({
            "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]
        })
        assert result == "hello world"


class TestOpenAIChat:
    adapter = OpenAIChatAdapter()

    def test_decode_simple(self):
        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "max_tokens": 100,
        }
        req = self.adapter.decode_request(body, {})
        assert req.system == "Be helpful."
        assert len(req.messages) == 2
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"

    def test_decode_with_tool_calls(self):
        body = {
            "messages": [
                {"role": "user", "content": "Read the file"},
                {
                    "role": "assistant",
                    "content": "I'll read it.",
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "/tmp/test.txt"}',
                            },
                        }
                    ],
                },
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }],
        }
        req = self.adapter.decode_request(body, {})
        assert len(req.tools) == 1
        assert req.tools[0].name == "read_file"

        assistant = req.messages[1]
        assert assistant.role == "assistant"
        assert assistant.tool_calls is not None
        assert assistant.tool_calls[0].id == "call_abc123"

    def test_decode_tool_message(self):
        body = {
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": "{}"}},
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "file content"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        tool_msg = req.messages[1]
        assert tool_msg.role == "tool"
        assert tool_msg.tool_call_id == "call_1"
        assert tool_msg.content == "file content"

    def test_decode_strings_args(self):
        """Test that string arguments are parsed as JSON."""
        body = {
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "c1", "type": "function",
                     "function": {"name": "read_file",
                                  "arguments": '{"path": "/tmp/x"}'}},
                ]},
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert req.messages[0].tool_calls[0].arguments["path"] == "/tmp/x"

    def test_stream_event(self):
        event = CanonicalEvent(type="text_delta", index=0, partial="Hello")
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "data: " in sse
        assert "Hello" in sse

    def test_chat_no_tool_call_event(self):
        """Chat protocol doesn't stream tool call partials."""
        event = CanonicalEvent(type="tool_use_delta", index=0, partial='{"path')
        sse = self.adapter.encode_stream_event(event)
        assert sse is None

    def test_parse_tool_result(self):
        result = self.adapter.parse_tool_result({"content": "test"})
        assert result == "test"

    def test_count_tokens(self):
        req = self.adapter.count_tokens_request({"messages": [{"role": "user", "content": "hi"}]})
        assert "messages" in req


class TestOpenAIResponses:
    adapter = OpenAIResponsesAdapter()

    def test_decode_simple(self):
        body = {
            "model": "gpt-4o",
            "input": [
                {"role": "user", "content": "Hello"},
            ],
            "instructions": "Be helpful.",
        }
        req = self.adapter.decode_request(body, {})
        assert req.system == "Be helpful."
        assert len(req.messages) == 1

    def test_decode_system_instructions(self):
        body = {
            "input": [
                {"role": "developer", "content": "You are a coder."},
                {"role": "user", "content": "Help me"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert "coder" in req.system

    def test_decode_with_tools(self):
        body = {
            "input": [
                {"role": "user", "content": "Read the file"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
                },
            }],
        }
        req = self.adapter.decode_request(body, {})
        assert len(req.tools) == 1

    def test_stream_event(self):
        event = CanonicalEvent(type="text_delta", index=0, partial="Hello")
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "response.output_text.delta" in sse

    def test_stream_tool_use(self):
        tc = ToolCall(id="call_1", name="read_file", arguments={"path": "/tmp/x"})
        event = CanonicalEvent(
            type="tool_use",
            index=1,
            content_block=ContentBlock(type="tool_use", tool_call=tc),
        )
        sse = self.adapter.encode_stream_event(event)
        assert sse is not None
        assert "function_call_arguments.done" in sse

    def test_parse_tool_result(self):
        body = {"output": [{"type": "function_call_output", "output": "result text"}]}
        result = self.adapter.parse_tool_result(body)
        assert result == "result text"

    def test_parse_tool_result_fallback(self):
        result = self.adapter.parse_tool_result({"content": "fallback"})
        assert result == "fallback"