"""Context capacity enforcement planner.

This planner only chooses a safe transformation order.  Mutation of message
payloads remains explicit in the gateway/controller, so no history can be
silently discarded during planning.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalRequest, CanonicalToolCallBlock, CanonicalToolResultBlock
from agent_interop.context_budget.estimator import estimate_request_context
from agent_interop.context_budget.types import ContextPlan


class ContextLimitExceededError(ValueError):
    """Structured preflight failure after safe adaptation is exhausted."""

    def __init__(self, plan: ContextPlan, attempted_strategies: tuple[str, ...]) -> None:
        self.plan = plan
        self.attempted_strategies = attempted_strategies
        super().__init__(
            f"context requires {plan.after.total_required_tokens} tokens; "
            f"safe limit is {plan.safe_limit_tokens}"
        )

    def details(self) -> dict[str, int | list[str]]:
        breakdown = self.plan.after
        return {
            "runtime_limit": self.plan.runtime_limit_tokens,
            "safe_limit": self.plan.safe_limit_tokens,
            "required": breakdown.total_required_tokens,
            "system": breakdown.system_tokens,
            "tools": breakdown.tool_schema_tokens,
            "history": breakdown.message_tokens,
            "output_reserve": breakdown.output_reserve_tokens,
            "attempted_strategies": list(self.attempted_strategies),
        }


def effective_context_limit(
    architecture_limit: int,
    configured_limit: int,
    route_override: int = 0,
    observed_effective_limit: int = 0,
) -> int:
    limits = [value for value in (
        architecture_limit, configured_limit, route_override, observed_effective_limit,
    ) if value > 0]
    return min(limits) if limits else 0


class ContextBudgetPlanner:
    """Build a plan that never silently removes required current context."""

    def plan(
        self,
        request: CanonicalRequest,
        *,
        runtime_limit_tokens: int,
        output_reserve_tokens: int | None = None,
        visible_tools=None,
        original_tools=None,
        prompted_contract: str = "",
    ) -> ContextPlan:
        before = estimate_request_context(
            request,
            visible_tools=original_tools if original_tools is not None else request.tools,
            prompted_contract=prompted_contract,
            output_reserve_tokens=output_reserve_tokens,
        )
        after = estimate_request_context(
            request,
            visible_tools=visible_tools if visible_tools is not None else request.tools,
            prompted_contract=prompted_contract,
            output_reserve_tokens=output_reserve_tokens,
        )
        safe_limit = int(runtime_limit_tokens * 0.90) if runtime_limit_tokens else 0
        fits = not safe_limit or after.total_required_tokens <= safe_limit
        all_indices = tuple(range(len(request.messages)))
        if fits:
            return ContextPlan(
                runtime_limit_tokens=runtime_limit_tokens,
                safe_limit_tokens=safe_limit,
                before=before,
                after=after,
                fits_directly=True,
                preserved_message_indices=all_indices,
                transformations=("reduce_tool_surface",) if after.tool_schema_tokens < before.tool_schema_tokens else (),
            )

        # Preserve required constraints, the latest user turn, and the most
        # recent complete tool exchange.  Older pageable tool output may be
        # reduced by the explicit compactor, but unknown/error output never is.
        protected: set[int] = set()
        latest_user = max((index for index, message in enumerate(request.messages)
                           if message.role == "user"), default=-1)
        if latest_user >= 0:
            protected.add(latest_user)
        latest_result = max((index for index, message in enumerate(request.messages)
                             if any(isinstance(block, CanonicalToolResultBlock) for block in message.content)), default=-1)
        if latest_result >= 0:
            protected.add(latest_result)
            # A matching assistant call is action-critical to its result.
            result_ids = {block.tool_call_id for block in request.messages[latest_result].content
                          if isinstance(block, CanonicalToolResultBlock)}
            for index in range(latest_result - 1, -1, -1):
                if any(isinstance(block, CanonicalToolCallBlock) and block.id in result_ids
                       for block in request.messages[index].content):
                    protected.add(index)
                    break
        for index, message in enumerate(request.messages):
            if message.role in {"system", "developer"}:
                protected.add(index)
        if request.messages:
            protected.add(len(request.messages) - 1)
        compacted = tuple(index for index in all_indices if index not in protected)
        return ContextPlan(
            runtime_limit_tokens=runtime_limit_tokens,
            safe_limit_tokens=safe_limit,
            before=before,
            after=after,
            fits_directly=False,
            compaction_required=True,
            selected_strategy="reduce_tools_then_compact_tool_results_then_summarize_history",
            preserved_message_indices=tuple(sorted(protected)),
            compacted_message_indices=compacted,
            transformations=(
                "reduce_tool_surface",
                "remove_duplicate_provider_decorations",
                "compact_old_tool_results",
                "summarize_old_history_in_controlled_mode",
                "delegate_through_controller",
            ),
        )
