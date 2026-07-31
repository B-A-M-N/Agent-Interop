"""Bounded compatibility attempt ladder executor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_interop.abi import (
    CanonicalError,
    CanonicalResponse,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    ToolChoiceMode,
)
from agent_interop.errors import InteropErrorCode
from agent_interop.execution_attempts.budget import AttemptBudget
from agent_interop.execution_attempts.results import AttemptResult
from agent_interop.planning.types import CompatibilityAttempt

ExecuteAttempt = Callable[[Any], Awaitable[CanonicalResponse]]
BuildAttemptInvocation = Callable[[Any, CompatibilityAttempt], Any]
ReplanWithheldTool = Callable[[Any, str], Any]


class CompatibilityAttemptExecutor:
    """Select the first validated result from a compatibility plan.

    An automatic request that returns no call is not a fallback failure. A
    named/required request with no valid call advances to the next bounded
    attempt. The gateway owns actual transport and controller dispatch.
    """

    def __init__(self, budget: AttemptBudget | None = None) -> None:
        self.budget = budget or AttemptBudget()
        self.results: list[AttemptResult] = []
        self._withheld_replanned = False

    @staticmethod
    def _satisfies_tool_requirement(response: CanonicalResponse, invocation: Any) -> bool:
        choice = invocation.reconciled_request.tool_choice
        if choice.mode == ToolChoiceMode.AUTO or choice.mode == ToolChoiceMode.NONE:
            return True
        calls = [block for block in response.content if isinstance(block, CanonicalToolCallBlock)]
        if choice.mode == ToolChoiceMode.REQUIRED:
            return bool(calls)
        return any(call.name == choice.name for call in calls)

    @staticmethod
    def _output_tokens(response: CanonicalResponse) -> int:
        usage = getattr(response, "usage", None)
        if usage is not None and getattr(usage, "output_tokens", 0):
            return int(usage.output_tokens)
        text = "".join(
            block.text if isinstance(block, CanonicalTextBlock) else str(getattr(block, "arguments", ""))
            for block in response.content
        )
        return (len(text) + 3) // 4

    async def execute(
        self,
        invocation: Any,
        *,
        build_invocation: BuildAttemptInvocation,
        execute_attempt: ExecuteAttempt,
        replan_withheld_tool: ReplanWithheldTool | None = None,
    ) -> CanonicalResponse:
        plan = invocation.compatibility_plan
        attempts = plan.attempts if plan is not None else ()
        if not attempts:
            return CanonicalResponse(error=getattr(invocation, "unavailable_error", None))
        latest: CanonicalResponse | None = None
        for attempt in attempts:
            if not self.budget.allow(attempt.use_controller):
                break
            candidate = build_invocation(invocation, attempt)
            execution = getattr(candidate, "execution_record", None)
            if execution is not None:
                execution.record_attempt()
            response = await execute_attempt(candidate)
            self.budget.record_generated_tokens(self._output_tokens(response))
            details = response.error.details if response.error is not None else {}
            requested_tool = details.get("withheld_tool_requested", "") if isinstance(details, dict) else ""
            if (
                requested_tool
                and not self._withheld_replanned
                and replan_withheld_tool is not None
                and self.budget.allow(False)
            ):
                self._withheld_replanned = True
                candidate = replan_withheld_tool(candidate, requested_tool)
                if execution is not None:
                    execution.record_attempt()
                response = await execute_attempt(candidate)
                self.budget.record_generated_tokens(self._output_tokens(response))
            latest = response
            accepted = response.error is None and self._satisfies_tool_requirement(response, candidate)
            self.results.append(AttemptResult(attempt, accepted, "accepted" if accepted else "tool_contract_unsatisfied"))
            if accepted:
                return response
            # No-tool automatic selection remains a legitimate model decision.
            if candidate.reconciled_request.tool_choice.mode == ToolChoiceMode.AUTO:
                return response
        if self.budget.exhausted_by:
            if getattr(invocation, "execution_record", None) is not None:
                invocation.execution_record.record_compatibility_event(
                    f"attempt_budget_exhausted:{self.budget.exhausted_by}"
                )
            return CanonicalResponse(error=CanonicalError(
                code=InteropErrorCode.ATTEMPT_BUDGET_EXHAUSTED,
                message=f"Compatibility attempt budget exhausted: {self.budget.exhausted_by}",
                details={"limit": self.budget.exhausted_by, "path": "compatibility_attempts"},
            ))
        return latest or CanonicalResponse()
