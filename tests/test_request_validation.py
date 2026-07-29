"""Tests for request validation — client-side tool contract checks."""

from __future__ import annotations

from agent_interop.abi import CanonicalTool, CanonicalToolChoice, ToolChoiceMode
from agent_interop.config import ToolMode
from agent_interop.request_validation import ValidationIssue, validate_tool_contract


class TestValidateToolContract:
    def test_empty_tools_is_valid(self):
        is_valid, issues = validate_tool_contract(tools=[], tool_choice=None)
        assert is_valid
        assert len(issues) == 0

    def test_duplicate_tool_names(self):
        tools = [
            CanonicalTool(name="read_file", description="", input_schema={"type": "object"}),
            CanonicalTool(name="read_file", description="", input_schema={"type": "object"}),
        ]
        is_valid, issues = validate_tool_contract(tools=tools, tool_choice=None)
        assert not is_valid
        codes = [i.code for i in issues]
        assert "TOOL_CALL_INVALID" in codes

    def test_empty_tool_name(self):
        tools = [
            CanonicalTool(name="", description="", input_schema={"type": "object"}),
        ]
        is_valid, issues = validate_tool_contract(tools=tools, tool_choice=None)
        assert not is_valid
        assert issues[0].code == "TOOL_CALL_INVALID"

    def test_required_without_tools(self):
        is_valid, issues = validate_tool_contract(
            tools=[],
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.REQUIRED),
        )
        assert not is_valid
        assert issues[0].code == "TOOL_CHOICE_VIOLATION"

    def test_named_tool_not_found(self):
        tools = [CanonicalTool(name="read_file", description="", input_schema={"type": "object"})]
        is_valid, issues = validate_tool_contract(
            tools=tools,
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.NAMED, name="nonexistent_tool"),
        )
        assert not is_valid
        assert issues[0].code == "TOOL_CHOICE_VIOLATION"

    def test_named_without_name(self):
        tools = [CanonicalTool(name="read_file", description="", input_schema={"type": "object"})]
        is_valid, issues = validate_tool_contract(
            tools=tools,
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.NAMED),
        )
        assert not is_valid
        assert issues[0].code == "TOOL_CHOICE_VIOLATION"

    def test_valid_named_tool(self):
        tools = [CanonicalTool(name="read_file", description="", input_schema={"type": "object"})]
        is_valid, _issues = validate_tool_contract(
            tools=tools,
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.NAMED, name="read_file"),
        )
        assert is_valid

    def test_disabled_with_tools(self):
        tools = [CanonicalTool(name="read_file", description="", input_schema={"type": "object"})]
        is_valid, issues = validate_tool_contract(
            tools=tools,
            tool_choice=None,
            tool_mode=ToolMode.DISABLED,
        )
        assert not is_valid
        assert issues[0].code == "TOOL_CHOICE_VIOLATION"

    def test_disabled_conflicts_with_required(self):
        is_valid, issues = validate_tool_contract(
            tools=[],
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.REQUIRED),
            tool_mode=ToolMode.DISABLED,
        )
        assert not is_valid
        codes = {i.code for i in issues}
        assert "TOOL_CHOICE_VIOLATION" in codes

    def test_none_choice_without_tools(self):
        is_valid, _issues = validate_tool_contract(
            tools=[],
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.NONE),
        )
        assert is_valid

    def test_auto_skips_validation(self):
        tools = [CanonicalTool(name="read_file", description="", input_schema={"type": "object"})]
        is_valid, _issues = validate_tool_contract(
            tools=tools,
            tool_choice=CanonicalToolChoice(mode=ToolChoiceMode.AUTO),
        )
        assert is_valid


class TestValidationIssue:
    def test_repr(self):
        issue = ValidationIssue("something bad", code="BAD_STUFF")
        assert "BAD_STUFF" in repr(issue)
        assert "something bad" in repr(issue)