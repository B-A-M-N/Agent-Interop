"""Bounded compatibility attempt ladder executor."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_interop.abi import (
    CanonicalError,
    CanonicalResponse,
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
HintKeyFn = Callable[[Any], str]
HintGetFn = Callable[[str], Any | None]
HintRecordFn = Callable[[str, Any], None]


def _max_selection_rounds(invocation: Any) -> int:
    """P0.38: Get max_selection_rounds from invocation's route, defaulting to 1."""
    try:
        return getattr(invocation.route.tool_surface, "max_selection_rounds", 1)
    except AttributeError:
        return 1


class CompatibilityAttemptExecutor:
    """Select the first validated result from a compatibility plan.

    An automatic request that returns no call is not a fallback failure. A
    named/required request with no valid call advances to the next bounded
    attempt. The gateway owns actual transport and controller dispatch.
    """

    def __init__(self, budget: AttemptBudget | None = None) -> None:
        self.budget = budget or AttemptBudget()
        self.results: list[AttemptResult] = []
        self._selection_replans = 0  # P0.38: track selection replans

    @staticmethod
    def _satisfies_tool_requirement(response: CanonicalResponse, invocation: Any) -> bool:
        choice = invocation.reconciled_request.tool_choice
        if choice.mode == ToolChoiceMode.AUTO or choice.mode == ToolChoiceMode.NONE:
            return True
        calls = [block for block in response.content if isinstance(block, CanonicalToolCallBlock)]
        if choice.mode == ToolChoiceMode.REQUIRED:
            return bool(calls)
        return any(call.name == choice.name for call in calls)

    async def execute(
        self,
        invocation: Any,
        *,
        build_invocation: BuildAttemptInvocation,
        execute_attempt: ExecuteAttempt,
        replan_withheld_tool: ReplanWithheldTool | None = None,
        hint_key: HintKeyFn | None = None,
        hint_get: HintGetFn | None = None,
        hint_record: HintRecordFn | None = None,
    ) -> CanonicalResponse:
        plan = invocation.compatibility_plan
        attempts = plan.attempts if plan is not None else ()
        # P0-45: a recorded operational hint reorders the ladder — moving the
        # previously-accepted kind to the front — but never adds, removes, or
        # authorizes an attempt.  (hints are keyed by serving tuple; only
        # kinds the planner already deemed permissible can be promoted.)
        if attempts and hint_key is not None and hint_get is not None:
            from agent_interop.planning.hints import reorder_attempts_by_hint

            preferred = hint_get(hint_key(invocation))
            attempts = reorder_attempts_by_hint(attempts, preferred)
        if not attempts:
            # Preparation can intentionally return a planless invocation for
            # unsafe history.  It still needs the gateway's ordinary
            # structured history error, never an empty successful response.
            if getattr(invocation, "invocation_plan", None) is None:
                return await execute_attempt(invocation)
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
            # P0-15: no token accounting here — every model token is recorded
            # by the generation seam (reserve/commit in the send path). This
            # executor counts only attempt-ladder rungs and selection replans;
            # two accounting sources would double-count every generation.
            details = response.error.details if response.error is not None else {}
            requested_tool = details.get("withheld_tool_requested", "") if isinstance(details, dict) else ""
            if (
                requested_tool
                and self._selection_replans < _max_selection_rounds(invocation)
                and replan_withheld_tool is not None
                and self.budget.allow(False)
            ):
                self._selection_replans += 1
                candidate = replan_withheld_tool(candidate, requested_tool)
                if execution is not None:
                    execution.record_attempt()
                response = await execute_attempt(candidate)
            latest = response
            accepted = response.error is None and self._satisfies_tool_requirement(response, candidate)
            self.results.append(AttemptResult(attempt, accepted, "accepted" if accepted else "tool_contract_unsatisfied"))
            if accepted:
                # P0-45: record what actually worked for this serving tuple —
                # an observed preference only, never trusted evidence.
                if hint_key is not None and hint_record is not None:
                    try:
                        key = hint_key(candidate)
                        if key:
                            hint_record(key, attempt.kind)
                    except Exception:  # hints must never fail a request
                        pass
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
