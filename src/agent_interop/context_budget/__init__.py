"""Context budgeting and capacity planning."""

from agent_interop.context_budget.compaction import (
    ContextAdaptationResult,
    compact_safe_tool_results,
)
from agent_interop.context_budget.estimator import (
    estimate_request_context,
    estimate_tool_schema_tokens,
)
from agent_interop.context_budget.planner import (
    ContextBudgetPlanner,
    ContextLimitExceededError,
    effective_context_limit,
)
from agent_interop.context_budget.tool_results import ToolResultPolicy, default_tool_result_policy
from agent_interop.context_budget.types import ContextBreakdown, ContextPlan, TokenEstimate

__all__ = [
    "ContextAdaptationResult",
    "ContextBreakdown",
    "ContextBudgetPlanner",
    "ContextLimitExceededError",
    "ContextPlan",
    "TokenEstimate",
    "ToolResultPolicy",
    "compact_safe_tool_results",
    "default_tool_result_policy",
    "effective_context_limit",
    "estimate_request_context",
    "estimate_tool_schema_tokens",
]
