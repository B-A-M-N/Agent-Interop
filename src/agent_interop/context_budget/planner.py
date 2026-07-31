"""Context capacity enforcement planner.

This planner only chooses a safe transformation order.  Mutation of message
payloads remains explicit in the gateway/controller, so no history can be
silently discarded during planning.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalRequest
from agent_interop.context_budget.estimator import estimate_request_context
from agent_interop.context_budget.types import ContextPlan


def effective_context_limit(
    architecture_limit: int, configured_limit: int, route_override: int = 0,
) -> int:
    limits = [value for value in (architecture_limit, configured_limit, route_override) if value > 0]
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

        # Preserve the latest user message and every current tool result. The
        # concrete compactor may reduce older tool result payloads first.
        protected: set[int] = set()
        for index, message in enumerate(request.messages):
            if message.role == "user" or any(getattr(block, "type", "") == "tool_result" for block in message.content):
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
