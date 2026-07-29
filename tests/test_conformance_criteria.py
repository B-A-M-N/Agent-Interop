"""Unit tests for RealConformanceRunner._verify_criteria's new criteria:
expected_tool_order, requires_distinct_call_ids, requires_same_turn_parallel,
and expected_arguments.

These close the "named tests not testing their names" gap in Finding #14 of
the external production-readiness review: several standard conformance
tests (edit_and_verify, distinct_ids, parallel_calls, sequential_calls,
tool_error_recovery, malformed_call_repair, nested_arguments) previously
passed regardless of whether the behavior their name claims actually
happened. Each test here proves the NEW criterion actually rejects the
specific bad behavior its predecessor test would have silently accepted.
"""

from __future__ import annotations

from agent_interop.config import InteropServerConfig
from agent_interop.evidence.store import EvidenceStore
from agent_interop.testing.runner import (
    ConformanceRunResult,
    ConformanceTest,
    RealConformanceRunner,
    ToolCallOutcome,
)


def _runner() -> RealConformanceRunner:
    return RealConformanceRunner(
        InteropServerConfig(probe_on_startup=False),
        evidence_store=EvidenceStore(db_path=":memory:"),
    )


def _result(*calls: ToolCallOutcome) -> ConformanceRunResult:
    r = ConformanceRunResult(test_name="t")
    r.tool_calls = list(calls)
    return r


class TestExpectedToolOrder:
    """sequential_calls / edit_and_verify / tool_error_recovery: order,
    not just membership, must be checked."""

    def test_correct_order_passes(self):
        test = ConformanceTest(name="t", prompt="p", expected_tool_order=["list_files", "read_file"])
        result = _result(
            ToolCallOutcome(tool_name="list_files", arguments={}),
            ToolCallOutcome(tool_name="read_file", arguments={}),
        )
        assert _runner()._verify_criteria(test, result) is None

    def test_reversed_order_fails(self):
        """The exact bug: both tools called, but in the wrong order — a
        test named for SEQUENCE must reject this, not just check membership."""
        test = ConformanceTest(name="t", prompt="p", expected_tool_order=["list_files", "read_file"])
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}),
            ToolCallOutcome(tool_name="list_files", arguments={}),
        )
        failure = _runner()._verify_criteria(test, result)
        assert failure is not None
        assert "order" in failure.lower()

    def test_repeated_failing_call_does_not_satisfy_recovery_order(self):
        """tool_error_recovery: calling the SAME failing tool twice must
        not satisfy an order requiring the recovery tool afterward."""
        test = ConformanceTest(name="t", prompt="p", expected_tool_order=["read_file", "list_files"])
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}, is_error=True),
            ToolCallOutcome(tool_name="read_file", arguments={}, is_error=True),
        )
        assert _runner()._verify_criteria(test, result) is not None


class TestDistinctCallIds:
    """distinct_ids: must actually check IDs, not merely call count."""

    def test_distinct_ids_pass(self):
        test = ConformanceTest(name="t", prompt="p", requires_distinct_call_ids=True)
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}, call_id="a"),
            ToolCallOutcome(tool_name="read_file", arguments={}, call_id="b"),
        )
        assert _runner()._verify_criteria(test, result) is None

    def test_duplicate_ids_fail(self):
        test = ConformanceTest(name="t", prompt="p", requires_distinct_call_ids=True)
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}, call_id="a"),
            ToolCallOutcome(tool_name="read_file", arguments={}, call_id="a"),
        )
        failure = _runner()._verify_criteria(test, result)
        assert failure is not None
        assert "duplicate" in failure.lower()

    def test_missing_ids_fail(self):
        """The pre-fix state: ToolCallOutcome had no id field at all, so
        this could never have been checked in the first place."""
        test = ConformanceTest(name="t", prompt="p", requires_distinct_call_ids=True)
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}, call_id=""),
            ToolCallOutcome(tool_name="read_file", arguments={}, call_id=""),
        )
        assert _runner()._verify_criteria(test, result) is not None


class TestSameTurnParallel:
    """parallel_calls: multiple calls spread across sequential turns must
    NOT satisfy a same-turn (single-response) parallel requirement."""

    def test_two_calls_same_turn_pass(self):
        test = ConformanceTest(name="t", prompt="p", requires_same_turn_parallel=2)
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}, turn=1),
            ToolCallOutcome(tool_name="read_file", arguments={}, turn=1),
        )
        assert _runner()._verify_criteria(test, result) is None

    def test_two_calls_across_two_turns_fail(self):
        """The exact bug: min_tool_calls=2 alone was satisfied equally by
        two calls in ONE turn or spread across TWO turns — only the
        former is genuinely parallel."""
        test = ConformanceTest(name="t", prompt="p", requires_same_turn_parallel=2)
        result = _result(
            ToolCallOutcome(tool_name="read_file", arguments={}, turn=1),
            ToolCallOutcome(tool_name="read_file", arguments={}, turn=2),
        )
        failure = _runner()._verify_criteria(test, result)
        assert failure is not None
        assert "turn" in failure.lower()


class TestExpectedArguments:
    """nested_arguments / malformed_call_repair: argument VALUES must be
    checked, not just that the right tool was called with something."""

    def test_matching_arguments_pass(self):
        test = ConformanceTest(
            name="t", prompt="p",
            expected_arguments={"search_code": {"path": "/src", "pattern": "TODO"}},
        )
        result = _result(
            ToolCallOutcome(tool_name="search_code", arguments={"path": "/src", "pattern": "TODO"}),
        )
        assert _runner()._verify_criteria(test, result) is None

    def test_wrong_argument_value_fails(self):
        """The exact bug this closes: a call to the right tool with the
        WRONG argument value previously passed regardless."""
        test = ConformanceTest(
            name="t", prompt="p",
            expected_arguments={"search_code": {"path": "/src", "pattern": "TODO"}},
        )
        result = _result(
            ToolCallOutcome(tool_name="search_code", arguments={"path": "/other", "pattern": "TODO"}),
        )
        failure = _runner()._verify_criteria(test, result)
        assert failure is not None
        assert "search_code" in failure

    def test_mangled_escaped_quote_fails(self):
        """malformed_call_repair's actual scenario: an embedded double
        quote that repair/extraction mangled must fail the check."""
        test = ConformanceTest(
            name="t", prompt="p",
            expected_arguments={"edit_file": {"old_string": 'he said "hi"'}},
        )
        result = _result(
            ToolCallOutcome(
                tool_name="edit_file",
                arguments={"old_string": "he said hi"},  # quotes silently dropped
            ),
        )
        assert _runner()._verify_criteria(test, result) is not None

    def test_tool_never_called_fails(self):
        test = ConformanceTest(
            name="t", prompt="p",
            expected_arguments={"edit_file": {"old_string": "x"}},
        )
        result = _result(ToolCallOutcome(tool_name="read_file", arguments={}))
        assert _runner()._verify_criteria(test, result) is not None


class TestSandboxedFileExecutor:
    """Re-audit P1#10: 'edit_and_verify' operated entirely on canned
    strings — any edit_file call at all produced "Edit applied" and any
    read_file call returned the hardcoded "new_value", regardless of what
    the model actually sent or whether an edit happened at all. These
    tests exercise make_sandboxed_file_executor directly: it must perform
    REAL file I/O so a no-op/wrong-argument edit is visibly different from
    a correct one, and the workspace confinement must reject a path that
    tries to escape it.
    """

    def test_edit_then_read_round_trips_through_real_disk(self, tmp_path):
        from agent_interop.testing.runner import make_sandboxed_file_executor

        (tmp_path / "test.txt").write_text("The value is old_value here.")
        executor = make_sandboxed_file_executor(tmp_path)

        edit_outcome = executor("edit_file", {
            "path": "test.txt", "old_string": "old_value", "new_string": "new_value",
        })
        assert edit_outcome.is_error is False
        assert (tmp_path / "test.txt").read_text() == "The value is new_value here."

        read_outcome = executor("read_file", {"path": "test.txt"})
        assert read_outcome.result == "The value is new_value here."

    def test_edit_with_wrong_old_string_fails_visibly(self, tmp_path):
        """A model (or a broken repair path) that sends the wrong
        old_string must get a real error, not a canned "Edit applied" —
        this is exactly the gap that let a no-op edit "pass" before."""
        from agent_interop.testing.runner import make_sandboxed_file_executor

        (tmp_path / "test.txt").write_text("The value is old_value here.")
        executor = make_sandboxed_file_executor(tmp_path)

        outcome = executor("edit_file", {
            "path": "test.txt", "old_string": "totally_wrong", "new_string": "new_value",
        })
        assert outcome.is_error is True
        # The file must be untouched.
        assert (tmp_path / "test.txt").read_text() == "The value is old_value here."

    def test_read_reflects_real_content_not_a_canned_string(self, tmp_path):
        from agent_interop.testing.runner import make_sandboxed_file_executor

        (tmp_path / "test.txt").write_text("whatever was actually written")
        executor = make_sandboxed_file_executor(tmp_path)
        outcome = executor("read_file", {"path": "test.txt"})
        assert outcome.result == "whatever was actually written"

    def test_path_cannot_escape_workspace(self, tmp_path):
        """A model-supplied absolute path (or one with '../' components)
        must never reach a real file outside the workspace — only the
        basename is honored, so "/etc/passwd" is confined to
        workspace_dir/passwd (which doesn't exist here), not the real
        /etc/passwd."""
        from agent_interop.testing.runner import make_sandboxed_file_executor

        executor = make_sandboxed_file_executor(tmp_path)
        outcome = executor("read_file", {"path": "/etc/passwd"})
        assert outcome.is_error is True
        # Proves containment: the error path is inside tmp_path, not /etc.
        assert str(tmp_path) in outcome.result
        assert "/etc/passwd" not in outcome.result

    def test_traversal_path_cannot_escape_workspace(self, tmp_path):
        from agent_interop.testing.runner import make_sandboxed_file_executor

        secret = tmp_path.parent / "secret.txt"
        secret.write_text("outside the sandbox")
        executor = make_sandboxed_file_executor(tmp_path)
        outcome = executor("read_file", {"path": "../secret.txt"})
        assert outcome.is_error is True
        assert outcome.result != "outside the sandbox"

    def test_list_files_reflects_real_directory(self, tmp_path):
        from agent_interop.testing.runner import make_sandboxed_file_executor

        (tmp_path / "a.txt").write_text("")
        (tmp_path / "b.txt").write_text("")
        executor = make_sandboxed_file_executor(tmp_path)
        outcome = executor("list_files", {"path": str(tmp_path)})
        assert "a.txt" in outcome.result
        assert "b.txt" in outcome.result


class TestEditAndVerifyPostRunCheck:
    """Re-audit P1#10: independent post-run verification for
    edit_and_verify — re-reads the file itself rather than trusting the
    model's own read_file call to have reported real content."""

    def test_get_standard_tests_without_workspace_uses_canned_executor(self):
        """Backward compatible default: omitting workspace_dir keeps the
        original fixed-string behavior (e.g. for isolated criteria tests
        that don't need real I/O)."""
        from agent_interop.testing.runner import get_standard_tests

        tests = get_standard_tests()
        edit_test = next(t for t in tests if t.name == "edit_and_verify")
        assert edit_test.post_run_verify is None
        outcome = edit_test.tool_executor("edit_file", {})
        assert outcome.result == "Edit applied"

    def test_get_standard_tests_with_workspace_seeds_real_file(self, tmp_path):
        from agent_interop.testing.runner import get_standard_tests

        tests = get_standard_tests(workspace_dir=tmp_path)
        assert (tmp_path / "test.txt").exists()
        assert "old_value" in (tmp_path / "test.txt").read_text()
        edit_test = next(t for t in tests if t.name == "edit_and_verify")
        assert edit_test.post_run_verify is not None
        assert str(tmp_path) in edit_test.prompt

    def test_post_run_check_passes_after_real_edit(self, tmp_path):
        from agent_interop.testing.runner import get_standard_tests

        tests = get_standard_tests(workspace_dir=tmp_path)
        edit_test = next(t for t in tests if t.name == "edit_and_verify")
        edit_test.tool_executor("edit_file", {
            "path": "test.txt", "old_string": "old_value", "new_string": "new_value",
        })
        assert edit_test.post_run_verify() is None

    def test_post_run_check_fails_if_edit_never_actually_happened(self, tmp_path):
        """The scenario this whole fix targets: a model 'calls' edit_file
        and read_file, criteria (tool names/order) are satisfied, but the
        real file was never actually changed (e.g. wrong path, or the
        executor being a no-op canned stub). post_run_verify is the check
        that catches this when the criteria checks above cannot."""
        from agent_interop.testing.runner import get_standard_tests

        tests = get_standard_tests(workspace_dir=tmp_path)
        edit_test = next(t for t in tests if t.name == "edit_and_verify")
        # No real edit performed — file still has its seeded content.
        failure = edit_test.post_run_verify()
        assert failure is not None
        assert "old_value" in failure

    def test_criteria_passing_does_not_itself_run_post_run_verify(self):
        """post_run_verify is deliberately NOT part of _verify_criteria —
        it's composed on top of it in run_test (see runner.py) so that a
        criteria-only caller (like the tests above) can check name/order/
        argument criteria without needing real disk state to back a
        post_run_verify callback. This documents that boundary."""
        test = ConformanceTest(
            name="t", prompt="p",
            expected_tools=["edit_file"],
            post_run_verify=lambda: "disk state does not match",
        )
        result = _result(ToolCallOutcome(tool_name="edit_file", arguments={}))
        assert _runner()._verify_criteria(test, result) is None
        # Confirms the callback itself still reports the failure it's
        # configured to — run_test's own composition (invoked only when
        # _verify_criteria returns None) is exercised end-to-end by the
        # certify CLI tests in test_cli_command_fixes.py.
        assert test.post_run_verify() == "disk state does not match"
