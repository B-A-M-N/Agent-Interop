"""Tests for the conformance test suite."""

from __future__ import annotations

from interop.testing.conformance import (
    ALL_TOOLS,
    CONFORMANCE_TESTS,
    ConformanceRunner,
    verify_arguments,
    verify_correct_tool,
    verify_id_preserved,
    verify_no_tool,
    verify_sequential,
)
from interop.types import CapabilityLevel, ToolCall, ToolCallDialect


class TestVerifiers:
    def test_verify_correct_tool_pass(self):
        calls = [ToolCall(id="t1", name="read_file", arguments={"path": "/tmp"})]
        result = verify_correct_tool(calls, ["read_file"])
        assert result.passed
        assert "read_file" in result.detail

    def test_verify_correct_tool_fail_no_calls(self):
        result = verify_correct_tool([], ["read_file"])
        assert not result.passed

    def test_verify_correct_tool_fail_wrong_tool(self):
        calls = [ToolCall(id="t1", name="wrong_tool", arguments={})]
        result = verify_correct_tool(calls, ["read_file"])
        assert not result.passed

    def test_verify_no_tool_pass(self):
        result = verify_no_tool([])
        assert result.passed

    def test_verify_no_tool_fail(self):
        calls = [ToolCall(id="t1", name="any_tool", arguments={})]
        result = verify_no_tool(calls)
        assert not result.passed

    def test_verify_arguments_all_present(self):
        call = ToolCall(id="t1", name="read_file", arguments={"path": "/tmp/x", "pattern": "test"})
        result = verify_arguments(call, ["path", "pattern"])
        assert result.passed

    def test_verify_arguments_missing(self):
        call = ToolCall(id="t1", name="read_file", arguments={"pattern": "test"})
        result = verify_arguments(call, ["path", "pattern"])
        assert not result.passed
        assert "path" in result.detail

    def test_verify_id_preserved_pass(self):
        call = ToolCall(id="call_abc123", name="test", arguments={})
        result = verify_id_preserved(call)
        assert result.passed

    def test_verify_id_preserved_fail(self):
        call = ToolCall(id="", name="test", arguments={})
        result = verify_id_preserved(call)
        assert not result.passed

    def test_verify_sequential_pass(self):
        calls = [
            ToolCall(id="t1", name="list_files", arguments={}),
            ToolCall(id="t2", name="read_file", arguments={}),
            ToolCall(id="t3", name="edit_file", arguments={}),
        ]
        result = verify_sequential(calls, ["list_files", "read_file"])
        assert result.passed

    def test_verify_sequential_fail(self):
        calls = [
            ToolCall(id="t1", name="edit_file", arguments={}),
            ToolCall(id="t2", name="list_files", arguments={}),
        ]
        result = verify_sequential(calls, ["list_files", "read_file"])
        assert not result.passed


class TestConformanceRunner:
    def test_all_tools_defined(self):
        assert len(ALL_TOOLS) >= 10

    def test_conformance_tests_structure(self):
        for level, tests in CONFORMANCE_TESTS.items():
            assert isinstance(level, str)
            assert level.startswith("L")
            for name, prompt, tools_for_test, validator in tests:
                assert name
                assert prompt
                assert isinstance(tools_for_test, list)
                assert callable(validator)

    def test_all_levels_present(self):
        # L0 has no tool tests (chat only), so L1-L4 should be present
        for level in ["L1", "L2", "L3", "L4"]:
            assert level in CONFORMANCE_TESTS

    def test_runner_empty(self):
        runner = ConformanceRunner()
        suite = runner.evaluate([])
        assert suite.level == CapabilityLevel.L0
        assert suite.passed == 0
        assert suite.total == 0

    def test_runner_single_pass(self):
        runner = ConformanceRunner()
        call = ToolCall(id="t1", name="read_file", arguments={"path": "/tmp/x"})
        # verify_arguments just checks the call's arguments
        suite = runner.evaluate([("test1", [call], lambda calls: verify_arguments(calls[0], ["path"]))])
        assert suite.passed >= 1
        assert suite.total == 1


class TestToolDefinitions:
    def test_tool_names(self):
        names = [t.name for t in ALL_TOOLS]
        assert "read_file" in names
        assert "edit_file" in names
        assert "search_code" in names
        assert "run_command" in names
        assert "list_files" in names
        assert "save_file" in names

    def test_tool_parameters(self):
        for tool in ALL_TOOLS:
            assert tool.parameters.get("type") == "object"
            assert "properties" in tool.parameters