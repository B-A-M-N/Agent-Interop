"""Path-specific conformance suites with bounded, atomic results."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from agent_interop.testing.runner import (
    ConformanceRunResult,
    ConformanceTest,
    RealConformanceRunner,
    get_standard_tests,
)


class ConformancePath(str, Enum):
    DIRECT = "direct"
    ADAPTED = "adapted"
    CONTROLLER = "controller"
    PRIMARY_WORKER = "primary_worker"
    CLIENT_CONTRACT = "client_contract"


class SuiteRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    INFRASTRUCTURE_INCONCLUSIVE = "infrastructure_inconclusive"


@dataclass(frozen=True)
class PathSuite:
    path: ConformancePath
    tests: tuple[ConformanceTest, ...]
    description: str


@dataclass
class PathSuiteResult:
    path: ConformancePath
    status: SuiteRunStatus = SuiteRunStatus.RUNNING
    results: list[ConformanceRunResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str = ""

    @property
    def passed(self) -> bool:
        return self.status == SuiteRunStatus.COMPLETED and all(item.passed for item in self.results)


def get_path_suite(path: ConformancePath, workspace_dir: Path | None = None) -> PathSuite:
    """Return tests appropriate to a compatibility path, not a model label."""
    standard = {test.name: test for test in get_standard_tests(workspace_dir)}
    selections: dict[ConformancePath, tuple[str, ...]] = {
        # Direct is the existing full battery and the only suite eligible for
        # a model-level L0-L4 conformance claim.
        ConformancePath.DIRECT: tuple(standard),
        # Adapted proves the transformation-sensitive behavior, including
        # prompt contracts, bounded repair, filtering, and continuation.
        ConformancePath.ADAPTED: (
            "explicit_forced_tool", "nested_arguments", "malformed_call_repair",
            "no_tool_request", "tool_result_continuation", "sequential_calls",
        ),
        # Controller calls remain client-executed. These cases exercise
        # selection, forbidden-tool restraint, continuation, and recovery.
        ConformancePath.CONTROLLER: (
            "explicit_forced_tool", "no_tool_request", "tool_result_continuation",
            "tool_error_recovery", "sequential_calls",
        ),
        # A primary worker has no tool authority: only its compact work
        # product and honest no-tool behavior are evaluated here.
        ConformancePath.PRIMARY_WORKER: ("no_tool_request",),
        ConformancePath.CLIENT_CONTRACT: ("explicit_forced_tool", "tool_result_continuation", "distinct_ids"),
    }
    descriptions = {
        ConformancePath.DIRECT: "Full direct-model conformance battery",
        ConformancePath.ADAPTED: "Prompted/filtered/constrained/repair adaptation battery",
        ConformancePath.CONTROLLER: "Controller selection and continuation battery",
        ConformancePath.PRIMARY_WORKER: "Tool-free primary worker battery",
        ConformancePath.CLIENT_CONTRACT: "Client protocol contract battery",
    }
    names = selections[path]
    return PathSuite(path, tuple(standard[name] for name in names), descriptions[path])


def write_result_atomic(path: Path, result: PathSuiteResult) -> None:
    """Atomically expose status so incomplete runs never look certified."""
    payload = asdict(result)
    payload["path"] = result.path.value
    payload["status"] = result.status.value
    destination = path.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, default=str, sort_keys=True, indent=2))
    os.replace(temporary, destination)


async def run_path_suite(
    runner: RealConformanceRunner,
    suite: PathSuite,
    *,
    model_name: str,
    route: Any,
    test_timeout: float = 120.0,
    suite_timeout: float = 900.0,
    max_turns: int | None = None,
    result_json: Path | None = None,
) -> PathSuiteResult:
    """Run a suite with explicit timeout/infrastructure result states."""
    result = PathSuiteResult(path=suite.path)
    if result_json is not None:
        write_result_atomic(result_json, result)

    async def run() -> None:
        for test in suite.tests:
            effective_test = test
            if max_turns is not None:
                effective_test = ConformanceTest(**{**test.__dict__, "max_turns": min(test.max_turns, max_turns)})
            try:
                test_result = await asyncio.wait_for(
                    runner.run_test(effective_test, model_name=model_name, route=route),
                    timeout=test_timeout,
                )
            except TimeoutError:
                result.status = SuiteRunStatus.ABORTED
                result.error = f"test timeout: {test.name}"
                return
            result.results.append(test_result)

    try:
        await asyncio.wait_for(run(), timeout=suite_timeout)
    except TimeoutError:
        result.status = SuiteRunStatus.ABORTED
        result.error = "suite timeout"
    else:
        if result.status == SuiteRunStatus.RUNNING:
            failures = [item for item in result.results if not item.passed]
            infra = bool(failures) and all(item.error_code for item in failures)
            result.status = SuiteRunStatus.INFRASTRUCTURE_INCONCLUSIVE if infra else SuiteRunStatus.COMPLETED
    finally:
        result.completed_at = time.time()
        if result_json is not None:
            write_result_atomic(result_json, result)
    return result
