"""Context capacity enforcement planner.

This planner only chooses a safe transformation order.  Mutation of message
payloads remains explicit in the gateway/controller, so no history can be
silently discarded during planning.
"""

from __future__ import annotations

from typing import Any

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


class ContextCapacityUnknownError(ValueError):
    """Context capacity is unknown — cannot confirm request fits.

    This is NOT the same as infinite. The caller must either:
    - Fail with CONTEXT_CAPACITY_UNKNOWN
    - Apply a conservative operator fallback (e.g., a small default limit)
    """

    def __init__(self, plan: ContextPlan | None = None, message: str | None = None) -> None:
        self.plan = plan
        super().__init__(
            message
            or (
                "Context capacity is unknown — cannot confirm request fits. "
                "Configure context_limit_tokens or set ollama_num_ctx."
            )
        )

    @classmethod
    def unresolvable(cls, model_name: str = "", policy_hint: str = "") -> ContextCapacityUnknownError:
        """Raise-form for a request rejected by the unknown-capacity policy
        before a ContextPlan exists (review #13/#14)."""
        hint = f" {policy_hint}" if policy_hint else ""
        return cls(
            message=(
                "CONTEXT_CAPACITY_UNKNOWN: context capacity for model "
                f"'{model_name}' could not be established (no observed runtime "
                "limit, route context_limit_tokens, ollama num_ctx, or trusted "
                f"model profile).{hint}"
            ),
        )

    def details(self) -> dict[str, Any]:
        if self.plan is None:
            return {"capacity_unknown": True}
        return {
            "required_tokens": self.plan.after.total_required_tokens,
            "message_tokens": self.plan.after.message_tokens,
        }


def effective_context_limit(
    architecture_limit: int = 0,
    configured_limit: int = 0,
    route_override: int = 0,
    observed_effective_limit: int = 0,
    *,
    architecture_ceiling: int | None = None,
) -> int:
    """Return the usable serving-context limit.

    P0.20: thin wrapper over resolve_context_capacity.  New code should use
    resolve_context_capacity directly for the full ContextCapacity object.

    P0-49 source semantics:

    * Usable capacity is chosen from OBSERVED serving context, then the
      operator route cap, then the configured serving context — actual or
      operator-asserted allocations only.
    * ``architecture_ceiling`` (or the legacy positional
      ``architecture_limit``) NEVER supplies capacity by itself.  It is a
      clamp: any resolved limit above the architecture's hard maximum is
      cut down to it.  The architecture maximum says nothing about what a
      backend actually allocated (num_ctx) — treating it as capacity let
      a 32K-architecture model with an 8K num_ctx plan against 32K.
    """
    from agent_interop.admission import resolve_context_capacity

    ceiling = (
        architecture_ceiling
        if architecture_ceiling is not None
        else architecture_limit
    )
    capacity = resolve_context_capacity(
        observed_runtime=observed_effective_limit,
        route_context_limit=route_override,
        ollama_num_ctx=configured_limit,
        profile_max_context=0,  # P0-49: architecture is a clamp, not a source
    )
    tokens = capacity.tokens or 0
    if ceiling > 0 and tokens > ceiling:
        return int(ceiling)
    return tokens


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
        cost_snapshot: Any | None = None,
    ) -> ContextPlan:
        before = estimate_request_context(
            request,
            visible_tools=original_tools if original_tools is not None else request.tools,
            prompted_contract=prompted_contract,
            output_reserve_tokens=output_reserve_tokens,
            snapshot=cost_snapshot,
        )
        after = estimate_request_context(
            request,
            visible_tools=visible_tools if visible_tools is not None else request.tools,
            prompted_contract=prompted_contract,
            output_reserve_tokens=output_reserve_tokens,
            snapshot=cost_snapshot,
        )
        safe_limit = int(runtime_limit_tokens * 0.90) if runtime_limit_tokens else 0
        # P0.6: Unknown context capacity does NOT mean infinite.
        # Use a conservative default (8K tokens) for unknown capacity.
        # The capacity_unknown flag is set so the gateway can warn or apply
        # a stricter policy for beta.
        if safe_limit == 0:
            safe_limit = 8192  # conservative default for unknown capacity
            capacity_unknown = True
        else:
            capacity_unknown = False
        fits = after.total_required_tokens <= safe_limit
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
                capacity_unknown=capacity_unknown,
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
            capacity_unknown=capacity_unknown,
        )
