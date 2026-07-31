"""Tests for path-specific conformance suite orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agent_interop.testing.runner import ConformanceRunResult, ConformanceTest
from agent_interop.testing.suites import (
    ConformancePath,
    PathSuite,
    SuiteRunStatus,
    get_path_suite,
    run_path_suite,
)


class _Runner:
    def __init__(self, *, delay: float = 0.0, error_code: str | None = None) -> None:
        self.delay = delay
        self.error_code = error_code

    async def run_test(self, test, **_kwargs):
        if self.delay:
            await asyncio.sleep(self.delay)
        return ConformanceRunResult(
            test_name=test.name,
            passed=self.error_code is None,
            error_code=self.error_code,
            error="backend unavailable" if self.error_code else "",
        )


def test_direct_suite_retains_the_full_standard_battery() -> None:
    suite = get_path_suite(ConformancePath.DIRECT)
    assert len(suite.tests) == 12


def test_adapted_suite_is_limited_to_adaptation_sensitive_cases() -> None:
    suite = get_path_suite(ConformancePath.ADAPTED)
    assert "malformed_call_repair" in {test.name for test in suite.tests}
    assert len(suite.tests) < 12


def test_suite_writes_completed_result_atomically(tmp_path: Path) -> None:
    suite = PathSuite(ConformancePath.PRIMARY_WORKER, (ConformanceTest("chat", "hello"),), "test")
    output = tmp_path / "result.json"
    result = asyncio.run(run_path_suite(
        _Runner(), suite, model_name="model", route=object(), result_json=output,
    ))
    assert result.status == SuiteRunStatus.COMPLETED
    assert json.loads(output.read_text())["status"] == "completed"


def test_suite_timeout_is_aborted_not_certified(tmp_path: Path) -> None:
    suite = PathSuite(ConformancePath.PRIMARY_WORKER, (ConformanceTest("slow", "hello"),), "test")
    output = tmp_path / "result.json"
    result = asyncio.run(run_path_suite(
        _Runner(delay=0.05), suite, model_name="model", route=object(),
        test_timeout=0.001, result_json=output,
    ))
    assert result.status == SuiteRunStatus.ABORTED
    assert not result.passed
    assert json.loads(output.read_text())["status"] == "aborted"
