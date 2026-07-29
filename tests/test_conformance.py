"""Tests for the conformance test suite."""

from __future__ import annotations

from agent_interop.abi import CanonicalToolCallBlock
from agent_interop.testing.conformance import (
    ALL_TOOLS,
    CONFORMANCE_TESTS,
    ConformanceRunner,
    verify_arguments,
    verify_correct_tool,
    verify_id_preserved,
    verify_no_tool,
    verify_sequential,
)
from agent_interop.types import CapabilityLevel


class TestVerifiers:
    def test_verify_correct_tool_pass(self):
        calls = [CanonicalToolCallBlock(id="t1", name="read_file", arguments={"path": "/tmp"})]
        result = verify_correct_tool(calls, ["read_file"])
        assert result.passed
        assert "read_file" in result.detail

    def test_verify_correct_tool_fail_no_calls(self):
        result = verify_correct_tool([], ["read_file"])
        assert not result.passed

    def test_verify_correct_tool_fail_wrong_tool(self):
        calls = [CanonicalToolCallBlock(id="t1", name="wrong_tool", arguments={})]
        result = verify_correct_tool(calls, ["read_file"])
        assert not result.passed

    def test_verify_no_tool_pass(self):
        result = verify_no_tool([])
        assert result.passed

    def test_verify_no_tool_fail(self):
        calls = [CanonicalToolCallBlock(id="t1", name="any_tool", arguments={})]
        result = verify_no_tool(calls)
        assert not result.passed

    def test_verify_arguments_all_present(self):
        call = CanonicalToolCallBlock(id="t1", name="read_file", arguments={"path": "/tmp/x", "pattern": "test"})
        result = verify_arguments(call, ["path", "pattern"])
        assert result.passed

    def test_verify_arguments_missing(self):
        call = CanonicalToolCallBlock(id="t1", name="read_file", arguments={"pattern": "test"})
        result = verify_arguments(call, ["path", "pattern"])
        assert not result.passed
        assert "path" in result.detail

    def test_verify_id_preserved_pass(self):
        call = CanonicalToolCallBlock(id="call_abc123", name="test", arguments={})
        result = verify_id_preserved(call)
        assert result.passed

    def test_verify_id_preserved_fail(self):
        call = CanonicalToolCallBlock(id="", name="test", arguments={})
        result = verify_id_preserved(call)
        assert not result.passed

    def test_verify_sequential_pass(self):
        calls = [
            CanonicalToolCallBlock(id="t1", name="list_files", arguments={}),
            CanonicalToolCallBlock(id="t2", name="read_file", arguments={}),
            CanonicalToolCallBlock(id="t3", name="edit_file", arguments={}),
        ]
        result = verify_sequential(calls, ["list_files", "read_file"])
        assert result.passed

    def test_verify_sequential_fail(self):
        calls = [
            CanonicalToolCallBlock(id="t1", name="edit_file", arguments={}),
            CanonicalToolCallBlock(id="t2", name="list_files", arguments={}),
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
        call = CanonicalToolCallBlock(id="t1", name="read_file", arguments={"path": "/tmp/x"})
        # verify_arguments just checks the call's arguments
        suite = runner.evaluate([("test1", [call], lambda calls: verify_arguments(calls[0], ["path"]))])
        assert suite.passed >= 1
        assert suite.total == 1

    def test_level_is_cumulative_not_per_level_isolated(self):
        """A model that fails every L1 test must NOT reach L4 merely by
        passing the three L4-named tests — level computation must require
        ALL lower levels' criteria too, not just the target level's own
        (isolated) requirement set."""
        from agent_interop.testing.conformance import ConformanceResult, ConformanceSuite

        runner = ConformanceRunner()
        suite = ConformanceSuite()
        # Only the L4 test names pass; every L1/L2/L3 name is absent
        # (never even attempted) or explicitly failing.
        suite.tests = [
            ConformanceResult(name="parallel_calls", passed=True),
            ConformanceResult(name="edit_and_verify", passed=True),
            ConformanceResult(name="distinct_ids", passed=True),
            ConformanceResult(name="explicit_tool", passed=False),
            ConformanceResult(name="explicit_tool2", passed=False),
            ConformanceResult(name="arguments_check", passed=False),
        ]
        level = runner._compute_level(suite)
        assert level != CapabilityLevel.L4
        assert level == CapabilityLevel.L0

    def test_level_reaches_l2_when_l1_and_l2_both_satisfied(self):
        from agent_interop.testing.conformance import ConformanceResult, ConformanceSuite

        runner = ConformanceRunner()
        suite = ConformanceSuite()
        suite.tests = [
            ConformanceResult(name="explicit_tool", passed=True),
            ConformanceResult(name="explicit_tool2", passed=True),
            ConformanceResult(name="arguments_check", passed=True),
            ConformanceResult(name="implicit_tool", passed=True),
            ConformanceResult(name="no_tool_when_unnecessary", passed=True),
        ]
        level = runner._compute_level(suite)
        assert level == CapabilityLevel.L2


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
            assert tool.input_schema.get("type") == "object"
            assert "properties" in tool.input_schema