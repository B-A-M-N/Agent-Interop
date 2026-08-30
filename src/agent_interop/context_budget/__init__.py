"""Context budgeting and capacity planning."""

from agent_interop.context_budget.compaction import (
    ContextAdaptationResult,
    compact_safe_tool_results,
)
from agent_interop.context_budget.estimator import (
    build_request_cost_snapshot,
    estimate_request_context,
    estimate_tool_schema_tokens,
    price_tool_subset,
)
from agent_interop.context_budget.model_view import ModelView
from agent_interop.context_budget.planner import (
    ContextBudgetPlanner,
    ContextCapacityUnknownError,
    ContextLimitExceededError,
    effective_context_limit,
)
from agent_interop.context_budget.tool_results import ToolResultPolicy, default_tool_result_policy
from agent_interop.context_budget.types import (
    ContextBreakdown,
    ContextPlan,
    RequestCostSnapshot,
    TokenEstimate,
)

__all__ = [
    "ContextAdaptationResult",
    "ContextBreakdown",
    "ContextBudgetPlanner",
    "ContextCapacityUnknownError",
    "ContextLimitExceededError",
    "ContextPlan",
    "ModelView",
    "RequestCostSnapshot",
    "TokenEstimate",
    "ToolResultPolicy",
    "build_request_cost_snapshot",
    "compact_safe_tool_results",
    "default_tool_result_policy",
    "effective_context_limit",
    "estimate_request_context",
    "estimate_tool_schema_tokens",
    "price_tool_subset",
]
