"""Compare — measure actual repair benefit across policies.

Compares replay results across different repair policies to determine
whether repair actually helped or introduced unintended executions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_interop.replay.types import ReplayCase, ReplayResult


@dataclass(frozen=True)
class PolicyComparison:
    """Comparison of outcomes across repair policies for one case."""

    results: Mapping[str, ReplayResult]
    case: ReplayCase | None = None

    @property
    def baseline_executable(self) -> bool:
        """Was the baseline (repair disabled) executable?"""
        baseline = self.results.get("repair_disabled")
        return baseline.executable if baseline else False

    @property
    def best_policy(self) -> str:
        """Name of the best policy for this case."""
        best = ""
        best_score = -1
        for name, result in self.results.items():
            score = self._score_result(result)
            if score > best_score:
                best_score = score
                best = name
        return best

    @property
    def repair_helped(self) -> bool:
        """Did repair actually help compared to baseline?"""
        baseline = self.results.get("repair_disabled")
        if not baseline or baseline.executable:
            return False  # Baseline already worked

        # Check if any repair policy made it executable
        for name, result in self.results.items():
            if name == "repair_disabled":
                continue
            if result.executable and result.arguments_valid:
                return True
        return False

    @property
    def introduced_unintended(self) -> bool:
        """Did repair create an execution baseline wouldn't have performed?"""
        baseline = self.results.get("repair_disabled")
        if not baseline:
            return False

        baseline_tools = {baseline.output_tool_name} if baseline.output_tool_name else set()

        for name, result in self.results.items():
            if name == "repair_disabled":
                continue
            if result.executable and result.output_tool_name:
                if result.output_tool_name not in baseline_tools:
                    return True
        return False

    @staticmethod
    def _score_result(result: ReplayResult) -> int:
        """Score a result — higher is better."""
        score = 0
        if result.executable:
            score += 10
        if result.arguments_valid:
            score += 5
        if result.tool_identity_preserved:
            score += 3
        if result.retry_avoided:
            score += 2
        return score


def compare_policies(results: Mapping[str, ReplayResult]) -> PolicyComparison:
    """Compare results across policies."""
    return PolicyComparison(
        case=None,
        results=results,
    )


def summarize_comparisons(
    comparisons: list[PolicyComparison],
) -> dict[str, int]:
    """Summarize multiple case comparisons."""
    summary = {
        "total_cases": len(comparisons),
        "repair_helped_count": 0,
        "repair_harmed_count": 0,
        "baseline_ok_count": 0,
        "introduced_unintended_count": 0,
    }

    for comp in comparisons:
        if comp.baseline_executable:
            summary["baseline_ok_count"] += 1
        elif comp.repair_helped:
            summary["repair_helped_count"] += 1

        if comp.introduced_unintended:
            summary["introduced_unintended_count"] += 1

    return summary
