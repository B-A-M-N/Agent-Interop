"""Tests for interop core types and data structures."""

from __future__ import annotations

import json

from interop.types import (
    CanonicalTool,
    CapabilityLevel,
    ContentBlock,
    ProtocolKind,
    ToolCall,
    ToolCallDialect,
    ToolResult,
    tool_from_anthropic,
    tool_from_openai,
    tool_to_anthropic,
    tool_to_openai,
)


class TestToolCall:
    def test_from_openai(self):
        spec = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        tool = tool_from_openai(spec)
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert "path" in tool.parameters["properties"]

    def test_from_anthropic(self):
        spec = {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        }
        tool = tool_from_anthropic(spec)
        assert tool.name == "read_file"
        assert "path" in tool.parameters["properties"]

    def test_to_openai(self):
        tool = CanonicalTool(
            name="search",
            description="Search code",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
        result = tool_to_openai(tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "search"

    def test_to_anthropic(self):
        tool = CanonicalTool(
            name="search",
            description="Search code",
            parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
        )
        result = tool_to_anthropic(tool)
        assert result["name"] == "search"
        assert "input_schema" in result


class TestCapabilityLevel:
    def test_ordering(self):
        assert CapabilityLevel.L0.value < CapabilityLevel.L4.value
        assert CapabilityLevel.L1 < CapabilityLevel.L3

    def test_all_levels_exist(self):
        assert len(CapabilityLevel) == 5


class TestContentBlock:
    def test_text_block(self):
        b = ContentBlock(type="text", text="hello")
        assert b.type == "text"
        assert b.text == "hello"

    def test_tool_use_block(self):
        tc = ToolCall(id="tc1", name="read", arguments={"path": "/tmp"})
        b = ContentBlock(type="tool_use", tool_call=tc)
        assert b.type == "tool_use"
        assert b.tool_call.name == "read"

    def test_tool_result_block(self):
        tr = ToolResult(call_id="tc1", tool_name="read", content="file content")
        b = ContentBlock(type="tool_result", tool_result=tr)
        assert b.type == "tool_result"
        assert b.tool_result.content == "file content"

    def test_thinking_block(self):
        b = ContentBlock(type="thinking", text="thinking...", signature="sig123")
        assert b.type == "thinking"
        assert b.signature == "sig123"


class TestToolCallObject:
    def test_roundtrip_dict(self):
        tc = ToolCall(id="tc1", name="read_file", arguments={"path": "/tmp/test.txt"})
        d = tc.to_dict()
        assert d["id"] == "tc1"
        assert d["name"] == "read_file"
        assert d["arguments"]["path"] == "/tmp/test.txt"

    def test_default_dialect(self):
        tc = ToolCall(id="t1", name="test", arguments={})
        assert tc.dialect == ToolCallDialect.GENERIC_JSON
        assert tc.repair.value == "none"


class TestProtocolKind:
    def test_values(self):
        assert ProtocolKind.ANTHROPIC_MESSAGES.value == "anthropic_messages"
        assert ProtocolKind.OPENAI_CHAT.value == "openai_chat"
        assert ProtocolKind.OPENAI_RESPONSES.value == "openai_responses"


class TestToolRelation:
    def test_canonical_tool_json_schema(self):
        tool = CanonicalTool(
            name="test",
            description="test tool",
            parameters={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        s = tool.to_json_schema()
        assert s["type"] == "function"
        assert s["function"]["parameters"]["properties"]["x"]["type"] == "string"