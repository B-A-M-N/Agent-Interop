"""Tests for protocol adapters — request/response translation."""

from __future__ import annotations

import json

import pytest

from agent_interop.abi import (
    CanonicalEvent,
    CanonicalToolCallBlock,
    ProtocolKind,
)
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from agent_interop.protocols.openai_chat import OpenAIChatAdapter
from agent_interop.protocols.openai_responses import OpenAIResponsesAdapter
from agent_interop.protocols.registry import detect_protocol, get_adapter


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
        assert req.messages[0].content[0].text == "Hello"
        assert req.messages[1].role == "assistant"
        assert req.messages[1].content[0].text == "Hi there"

    def test_decode_with_system(self):
        body = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        req = self.adapter.decode_request(body, {})
        assert req.system[0].text == "You are helpful."

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
        assert len(assistant.content) == 2
        assert assistant.content[0].type == "text"
        assert assistant.content[1].type == "tool_call"
        assert assistant.content[1].id == "toolu_abc123"
        assert assistant.content[1].name == "read_file"

    def test_decode_tool_result(self):
        # Anthropic's wire format has no dedicated "tool" role — a real
        # Claude Code turn puts the tool_result inside a role:"user"
        # message (confirmed via a live captured request). Interop
        # normalizes a PURE tool-result message to canonical role="tool"
        # to match its own internal convention (same as the OpenAI Chat
        # decoder's dedicated "tool" role, and what
        # history/reconcile.py's safety check and upstreams/anthropic.py's
        # outbound encoder both already expect) — a live acceptance run
        # against Claude Code 2.1.220 found this mismatch caused Interop's
        # OWN history-safety check to reject every real multi-turn tool
        # call as "unsafe history".
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
        assert req.messages[0].content[0].tool_call_id == "toolu_abc123"
        assert req.messages[0].content[0].content == "File contents here"

    def test_decode_tool_result_mixed_with_text_keeps_user_role(self):
        """A user message that mixes a tool_result with the user's own
        new text is a real user turn with a tool_result attached, not a
        pure tool-result message — it must keep role="user", not be
        force-normalized to "tool" alongside content that isn't one."""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_abc123",
                            "content": "File contents here",
                        },
                        {"type": "text", "text": "Also, what's 2+2?"},
                    ],
                }
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"
        assert req.messages[0].content[0].tool_call_id == "toolu_abc123"
        assert req.messages[0].content[1].text == "Also, what's 2+2?"

    def test_stream_event_encoding(self):
        event = CanonicalEvent(type="text_delta", index=0, partial="Hello")
        sse = self.adapter.encode_event(event)
        assert sse is not None
        assert "event: content_block_delta" in sse
        assert "text_delta" in sse
        assert "Hello" in sse

        event2 = CanonicalEvent(type="message_stop")
        sse2 = self.adapter.encode_event(event2)
        assert sse2 is not None
        assert "event: message_stop" in sse2

    def test_stream_text_start(self):
        event = CanonicalEvent(type="text", index=0, partial="Hello")
        sse = self.adapter.encode_event(event)
        assert sse is not None
        assert "event: content_block_start" in sse
        assert "Hello" in sse

    def test_stream_tool_use(self):
        tc = CanonicalToolCallBlock(id="toolu_abc", name="read_file", arguments={"path": "/tmp/x"})
        event = CanonicalEvent(
            type="tool_use",
            index=1,
            content_block=tc,
        )
        sse = self.adapter.encode_event(event)
        assert sse is not None
        assert "event: content_block_start" in sse
        assert "tool_use" in sse
        assert "toolu_abc" in sse

    def test_stream_thinking(self):
        event = CanonicalEvent(type="thinking_delta", index=0, partial="thinking text")
        sse = self.adapter.encode_event(event)
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
        assert req.system[0].text == "Be helpful."
        assert len(req.messages) == 2
        assert req.messages[0].role == "user"
        assert req.messages[1].role == "assistant"

    def test_decode_developer_message_preserved(self):
        """A developer-role message must not be silently dropped (re-audit
        P0#3) — it round-trips as an ordered role='developer' message,
        not merged into the top-level system field."""
        body = {
            "messages": [
                {"role": "developer", "content": "Never modify files outside the workspace."},
                {"role": "user", "content": "Hello"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert req.system == []
        assert len(req.messages) == 2
        assert req.messages[0].role == "developer"
        assert req.messages[0].content[0].text == "Never modify files outside the workspace."
        assert req.messages[1].role == "user"

    def test_decode_developer_and_system_ordering_preserved(self):
        """developer + system + user: system is hoisted (as today), but
        developer keeps its position relative to the other messages."""
        body = {
            "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "developer", "content": "Follow workspace policy."},
                {"role": "user", "content": "Hello"},
                {"role": "developer", "content": "Second reminder."},
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert req.system[0].text == "Be helpful."
        assert [m.role for m in req.messages] == ["developer", "user", "developer"]
        assert req.messages[0].content[0].text == "Follow workspace policy."
        assert req.messages[2].content[0].text == "Second reminder."

    def test_decode_developer_message_structured_blocks(self):
        """Developer content given as a block array (not a bare string)
        must parse through the same content-block path as user messages."""
        body = {
            "messages": [
                {"role": "developer", "content": [
                    {"type": "text", "text": "Rule one."},
                    {"type": "text", "text": "Rule two."},
                ]},
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert req.messages[0].role == "developer"
        assert [b.text for b in req.messages[0].content] == ["Rule one.", "Rule two."]

    def test_decode_developer_message_renders_as_system_on_egress(self):
        """The OpenAI Chat egress renderer maps role='developer' -> 'system'
        per-message (upstreams/openai_chat.py._render_message) — verify the
        ingress-decoded message actually round-trips through it."""
        from agent_interop.upstreams.openai_chat import OpenAIChatCodec

        body = {
            "model": "gpt-4o",
            "messages": [
                {"role": "developer", "content": "Workspace policy."},
                {"role": "user", "content": "Hello"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        rendered = OpenAIChatCodec().render_request(req, "gpt-4o", stream=False)
        wire_messages = rendered["messages"]
        assert wire_messages[0]["role"] == "system"
        assert wire_messages[0]["content"] == "Workspace policy."

    def test_decode_developer_message_renders_through_anthropic_egress(self):
        """Cross-protocol: a developer message ingested from OpenAI Chat must
        still reach an Anthropic-protocol backend, not vanish at the seam."""
        from agent_interop.upstreams.anthropic import AnthropicCodec

        body = {
            "messages": [
                {"role": "developer", "content": "Workspace policy."},
                {"role": "user", "content": "Hello"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        rendered = AnthropicCodec().render_request(req, "claude-x", stream=False)
        wire_messages = rendered["messages"]
        assert any("Workspace policy." in json.dumps(m) for m in wire_messages)

    def test_decode_developer_message_renders_through_responses_egress(self):
        """Cross-protocol: a developer message ingested from OpenAI Chat must
        still reach an OpenAI Responses-protocol backend."""
        from agent_interop.upstreams.openai_responses import OpenAIResponsesCodec

        body = {
            "messages": [
                {"role": "developer", "content": "Workspace policy."},
                {"role": "user", "content": "Hello"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        rendered = OpenAIResponsesCodec().render_request(req, "gpt-x", stream=False)
        assert "Workspace policy." in json.dumps(rendered["input"])

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
        assert len(assistant.content) == 2
        assert assistant.content[0].text == "I'll read it."
        assert assistant.content[1].type == "tool_call"
        assert assistant.content[1].id == "call_abc123"
        assert assistant.content[1].name == "read_file"

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
        assert tool_msg.content[0].tool_call_id == "call_1"
        assert tool_msg.content[0].content == "file content"

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
        assert req.messages[0].content[0].arguments["path"] == "/tmp/x"

    def test_stream_event(self):
        event = CanonicalEvent(type="text_delta", index=0, partial="Hello")
        sse = self.adapter.encode_event(event)
        assert sse is not None
        assert "data: " in sse
        assert "Hello" in sse

    def test_chat_no_tool_call_event(self):
        """Chat protocol doesn't stream tool call partials."""
        event = CanonicalEvent(type="tool_use_delta", index=0, partial='{"path')
        sse = self.adapter.encode_event(event)
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
        assert req.system[0].text == "Be helpful."
        assert len(req.messages) == 1

    def test_decode_system_instructions(self):
        body = {
            "input": [
                {"role": "developer", "content": "You are a coder."},
                {"role": "user", "content": "Help me"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        assert "coder" in req.system[0].text

    def test_decode_native_function_call_preserves_raw_arguments(self):
        """The native Responses-API "function_call" item shape (what Codex,
        the real client for this protocol, actually sends) must carry
        raw_arguments/arguments_validated exactly like the backward-compat
        "message role=assistant with tool_calls" shape does — previously
        it silently dropped raw_arguments (None) and defaulted
        arguments_validated to True even when the JSON failed to parse."""
        body = {
            "input": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/x"}',
                },
            ],
        }
        req = self.adapter.decode_request(body, {})
        tc = req.messages[0].content[0]
        assert tc.arguments == {"path": "/tmp/x"}
        assert tc.raw_arguments == '{"path": "/tmp/x"}'
        assert tc.arguments_validated is True

    def test_decode_native_function_call_malformed_json_not_marked_validated(self):
        body = {
            "input": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "name": "read_file",
                    "arguments": '{"path": ',  # truncated / invalid JSON
                },
            ],
        }
        req = self.adapter.decode_request(body, {})
        tc = req.messages[0].content[0]
        assert tc.arguments_validated is False
        assert tc.raw_arguments == '{"path": '

    def test_decode_function_call_uses_call_id_not_item_id(self):
        """Re-audit P0#4: a function_call item's own "id" (item_123) is
        distinct from its "call_id" (call_456) — a later
        function_call_output only ever carries call_id, so canonical
        pairing must key off call_id, not id, or history reconciliation
        classifies a valid pair as unmatched."""
        body = {
            "input": [
                {
                    "type": "function_call",
                    "id": "item_123",
                    "call_id": "call_456",
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/x"}',
                },
            ],
        }
        req = self.adapter.decode_request(body, {})
        tc = req.messages[0].content[0]
        assert tc.id == "call_456"
        # The item id is not discarded — it round-trips via provider metadata.
        assert tc.provider_metadata is not None
        assert tc.provider_metadata.metadata_kind == "responses_item_id"
        assert tc.provider_metadata.opaque_value == "item_123"

    def test_decode_function_call_output_pairs_with_decoded_call_id(self):
        """The function_call and its later function_call_output must
        resolve to the SAME canonical id when id != call_id, so a
        tool-loop reconciler can actually match them."""
        body = {
            "input": [
                {
                    "type": "function_call",
                    "id": "item_123",
                    "call_id": "call_456",
                    "name": "read_file",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_456",
                    "output": "file contents",
                },
            ],
        }
        req = self.adapter.decode_request(body, {})
        call_block = req.messages[0].content[0]
        result_block = req.messages[1].content[0]
        assert call_block.id == result_block.tool_call_id == "call_456"

    def test_decode_function_call_without_distinct_call_id_falls_back_to_item_id(self):
        """When the client only sends "id" (no separate call_id — the
        common/simple case), that id is still usable as the canonical id
        and no synthetic provider_metadata is attached."""
        body = {
            "input": [
                {"type": "function_call", "id": "fc_1", "name": "read_file", "arguments": "{}"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        tc = req.messages[0].content[0]
        assert tc.id == "fc_1"
        assert tc.provider_metadata is None

    def test_encode_response_emits_both_id_and_call_id(self):
        """Non-streaming egress must emit the real Responses wire shape:
        a function_call item has both "id" (item id) and "call_id" (the
        id used for tool-loop pairing) as separate keys."""
        from agent_interop.abi import CanonicalResponse, ProviderMetadata

        resp = CanonicalResponse(content=[
            CanonicalToolCallBlock(
                id="call_456",
                name="read_file",
                arguments={"path": "/tmp/x"},
                provider_metadata=ProviderMetadata(
                    metadata_kind="responses_item_id",
                    opaque_value="item_123",
                ),
            ),
        ])
        encoded = self.adapter.encode_response(resp)
        fc_item = next(o for o in encoded["output"] if o["type"] == "function_call")
        assert fc_item["id"] == "item_123"
        assert fc_item["call_id"] == "call_456"

    def test_encode_response_without_preserved_item_id_reuses_call_id(self):
        """A tool call synthesized internally (no preserved Responses item
        id) still emits a self-consistent "id"/"call_id" pair rather than
        omitting one of the two required keys."""
        from agent_interop.abi import CanonicalResponse

        resp = CanonicalResponse(content=[
            CanonicalToolCallBlock(id="call_999", name="read_file", arguments={}),
        ])
        encoded = self.adapter.encode_response(resp)
        fc_item = next(o for o in encoded["output"] if o["type"] == "function_call")
        assert fc_item["id"] == "call_999"
        assert fc_item["call_id"] == "call_999"

    def test_streaming_tool_use_emits_both_id_and_call_id(self):
        """Streaming output_item.added must also carry the item/call id
        pair — not just the non-streaming path — since a real Codex client
        pairs the eventual function_call_output against whichever id the
        stream told it was the call_id."""
        from agent_interop.abi import ProviderMetadata

        adapter = OpenAIResponsesAdapter()
        encoder = adapter.create_stream_encoder({"response_id": "resp_1", "model": "test-model"})
        cb = CanonicalToolCallBlock(
            id="call_456",
            name="read_file",
            arguments={},
            provider_metadata=ProviderMetadata(
                metadata_kind="responses_item_id", opaque_value="item_123",
            ),
        )
        event = CanonicalEvent(type="tool_use", index=0, content_block=cb)
        frame = encoder.encode(event)
        assert frame is not None
        payload = json.loads(frame.split("data: ", 1)[1].split("\n\n")[0])
        assert payload["item"]["id"] == "item_123"
        assert payload["item"]["call_id"] == "call_456"

    def test_decoded_call_id_reconciles_cleanly_through_history(self):
        """End-to-end: history reconciliation pairs a function_call/
        function_call_output by canonical id — proving the call_id fix
        actually fixes reconciliation, not just decode_request in isolation.
        Before the fix, tc.id was "item_123" and the result's tool_call_id
        was "call_456": two different ids, so reconcile_history would see
        an orphaned call and an orphaned result instead of one exchange."""
        from agent_interop.history.reconcile import reconcile_history

        body = {
            "input": [
                {"role": "user", "content": "read the file"},
                {
                    "type": "function_call",
                    "id": "item_123",
                    "call_id": "call_456",
                    "name": "read_file",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_456", "output": "contents"},
            ],
        }
        req = self.adapter.decode_request(body, {})
        result = reconcile_history(req.messages, session_id="s1", request_id="r1")
        assert result.is_safe, result.diagnostics
        assert not any("orphan" in d for d in result.diagnostics)

    def test_decode_user_content_preserves_non_text_blocks(self):
        """A non-text block (e.g. input_image) in a user message's content
        list must be preserved as an unknown block, not silently dropped —
        previously only "input_text"/"text" items survived."""
        body = {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "look at this"},
                        {"type": "input_image", "image_url": "https://example.com/x.png"},
                    ],
                },
            ],
        }
        req = self.adapter.decode_request(body, {})
        blocks = req.messages[0].content
        assert len(blocks) == 2
        assert blocks[0].type == "text"
        assert blocks[0].text == "look at this"
        assert blocks[1].type == "unknown"
        assert blocks[1].source_type == "input_image"

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
        sse = self.adapter.encode_event(event)
        assert sse is not None
        assert "response.output_text.delta" in sse

    def test_stream_tool_use(self):
        tc = CanonicalToolCallBlock(id="call_1", name="read_file", arguments={"path": "/tmp/x"})
        event = CanonicalEvent(
            type="tool_use",
            index=1,
            content_block=tc,
        )
        sse = self.adapter.encode_event(event)
        assert sse is not None
        assert "function_call_arguments.done" in sse

    def test_parse_tool_result(self):
        body = {"output": [{"type": "function_call_output", "output": "result text"}]}
        result = self.adapter.parse_tool_result(body)
        assert result == "result text"

    def test_parse_tool_result_fallback(self):
        result = self.adapter.parse_tool_result({"content": "fallback"})
        assert result == "fallback"


class TestRequestValidationRejectsSilentSemanticChanges:
    """An unrecognized tool_choice or a nonsensical generation param used
    to be silently coerced (tool_choice -> "auto", garbage max_tokens /
    temperature passed straight through to the backend) instead of
    rejected — a real request the client asked for would silently become
    a different one. decode_request() must raise ValueError (the
    server-boundary layer already converts that into a clean 4xx) rather
    than swallow it into a default.
    """

    def _base_anthropic_body(self, **overrides):
        body = {
            "model": "claude-sonnet-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        body.update(overrides)
        return body

    def _base_openai_chat_body(self, **overrides):
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
        }
        body.update(overrides)
        return body

    def _base_openai_responses_body(self, **overrides):
        body = {
            "model": "test-model",
            "input": [{"role": "user", "content": "hi"}],
            "max_output_tokens": 100,
        }
        body.update(overrides)
        return body

    # ─── tool_choice ────────────────────────────────────────────────────

    def test_anthropic_unrecognized_tool_choice_string_rejected(self):
        adapter = AnthropicMessagesAdapter()
        body = self._base_anthropic_body(tool_choice="definitely_required")
        with pytest.raises(ValueError, match="tool_choice"):
            adapter.decode_request(body, {})

    def test_anthropic_unrecognized_tool_choice_type_rejected(self):
        adapter = AnthropicMessagesAdapter()
        body = self._base_anthropic_body(tool_choice={"type": "bogus"})
        with pytest.raises(ValueError, match="tool_choice"):
            adapter.decode_request(body, {})

    def test_openai_chat_unrecognized_tool_choice_rejected(self):
        adapter = OpenAIChatAdapter()
        body = self._base_openai_chat_body(tool_choice="definitely_required")
        with pytest.raises(ValueError, match="tool_choice"):
            adapter.decode_request(body, {})

    def test_openai_responses_unrecognized_tool_choice_rejected(self):
        adapter = OpenAIResponsesAdapter()
        body = self._base_openai_responses_body(tool_choice="definitely_required")
        with pytest.raises(ValueError, match="tool_choice"):
            adapter.decode_request(body, {})

    # ─── generation options ─────────────────────────────────────────────

    def test_anthropic_negative_max_tokens_rejected(self):
        adapter = AnthropicMessagesAdapter()
        body = self._base_anthropic_body(max_tokens=-1)
        with pytest.raises(ValueError, match="max_tokens"):
            adapter.decode_request(body, {})

    def test_openai_chat_non_numeric_temperature_rejected(self):
        adapter = OpenAIChatAdapter()
        body = self._base_openai_chat_body(temperature="hot")
        with pytest.raises(ValueError, match="temperature"):
            adapter.decode_request(body, {})

    def test_openai_responses_zero_max_tokens_rejected(self):
        adapter = OpenAIResponsesAdapter()
        body = self._base_openai_responses_body(max_output_tokens=0)
        with pytest.raises(ValueError, match="max_"):
            adapter.decode_request(body, {})

    def test_valid_requests_still_decode_cleanly(self):
        """The validators must not reject well-formed requests — this is
        the counterpart to the rejection tests above."""
        AnthropicMessagesAdapter().decode_request(self._base_anthropic_body(), {})
        OpenAIChatAdapter().decode_request(self._base_openai_chat_body(), {})
        OpenAIResponsesAdapter().decode_request(self._base_openai_responses_body(), {})


class TestTopPAndStopSurviveIngressAndEgress:
    """Re-audit P0#5: top_p and stop were canonicalized nowhere — every
    client ingress adapter silently dropped them even though the ABI
    (CanonicalGenerationOptions) already had fields for both and the
    Anthropic egress codec already knew how to render them. A client that
    asked for top_p=0.3 or a stop sequence had that request silently
    changed to "whatever the backend's default is" with no error and no
    signal. These tests prove the value survives ingress AND reaches the
    rendered upstream wire body for every (client protocol, backend
    protocol) pair Interop actually supports today (OpenAI Chat, Anthropic,
    Ollama; Responses upstream has no wire-level stop equivalent, so only
    top_p is checked there)."""

    def test_openai_chat_ingress_decodes_top_p_and_stop(self):
        req = OpenAIChatAdapter().decode_request({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.3,
            "stop": ["STOP"],
        }, {})
        assert req.generation.top_p == 0.3
        assert req.generation.stop == ["STOP"]

    def test_openai_chat_ingress_decodes_bare_string_stop(self):
        req = OpenAIChatAdapter().decode_request({
            "messages": [{"role": "user", "content": "hi"}],
            "stop": "STOP",
        }, {})
        assert req.generation.stop == ["STOP"]

    def test_anthropic_ingress_decodes_top_p_and_stop_sequences(self):
        req = AnthropicMessagesAdapter().decode_request({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "top_p": 0.5,
            "stop_sequences": ["END"],
        }, {})
        assert req.generation.top_p == 0.5
        assert req.generation.stop == ["END"]

    def test_openai_responses_ingress_decodes_top_p(self):
        req = OpenAIResponsesAdapter().decode_request({
            "model": "m",
            "input": [{"role": "user", "content": "hi"}],
            "top_p": 0.7,
        }, {})
        assert req.generation.top_p == 0.7

    def test_top_p_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="top_p"):
            OpenAIChatAdapter().decode_request({
                "messages": [{"role": "user", "content": "hi"}],
                "top_p": 1.5,
            }, {})

    def test_openai_chat_egress_renders_top_p_and_stop(self):
        from agent_interop.upstreams.openai_chat import OpenAIChatCodec

        req = OpenAIChatAdapter().decode_request({
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.3,
            "stop": ["STOP"],
        }, {})
        rendered = OpenAIChatCodec().render_request(req, "m", stream=False)
        assert rendered["top_p"] == 0.3
        assert rendered["stop"] == ["STOP"]

    def test_anthropic_egress_renders_top_p_and_stop_sequences(self):
        from agent_interop.upstreams.anthropic import AnthropicCodec

        req = AnthropicMessagesAdapter().decode_request({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "top_p": 0.5,
            "stop_sequences": ["END"],
        }, {})
        rendered = AnthropicCodec().render_request(req, "m", stream=False)
        assert rendered["top_p"] == 0.5
        assert rendered["stop_sequences"] == ["END"]

    def test_openai_responses_egress_renders_top_p(self):
        from agent_interop.upstreams.openai_responses import OpenAIResponsesCodec

        req = OpenAIResponsesAdapter().decode_request({
            "input": [{"role": "user", "content": "hi"}],
            "top_p": 0.7,
        }, {})
        rendered = OpenAIResponsesCodec().render_request(req, "m", stream=False)
        assert rendered["top_p"] == 0.7

    def test_ollama_egress_renders_top_p_and_stop(self):
        from agent_interop.upstreams.ollama_chat import OllamaChatCodec

        req = OpenAIChatAdapter().decode_request({
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.4,
            "stop": ["STOP"],
        }, {})
        rendered = OllamaChatCodec().render_request(req, "m", stream=False)
        assert rendered["options"]["top_p"] == 0.4
        assert rendered["options"]["stop"] == ["STOP"]

    def test_cross_protocol_openai_chat_ingress_to_anthropic_egress(self):
        """A client speaking OpenAI Chat routed to an Anthropic backend
        must still have top_p/stop reach the rendered request — proving
        the fix isn't just same-protocol coincidence."""
        from agent_interop.upstreams.anthropic import AnthropicCodec

        req = OpenAIChatAdapter().decode_request({
            "messages": [{"role": "user", "content": "hi"}],
            "top_p": 0.6,
            "stop": ["DONE"],
        }, {})
        rendered = AnthropicCodec().render_request(req, "claude-x", stream=False)
        assert rendered["top_p"] == 0.6
        assert rendered["stop_sequences"] == ["DONE"]