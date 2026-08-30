"""Context virtualization store (P0.3).

Retains the full client state (tool results, history fragments) that Interop
holds on behalf of the local model, exposing bounded handles and private
retrieval tools so the model can pull slices on demand.
"""

from agent_interop.context_store.executor import (
    InternalExecutionContext,
    InternalToolExecutor,
    InternalToolResult,
)
from agent_interop.context_store.handles import is_valid_ref
from agent_interop.context_store.policy import VirtualizationPolicy, default_virtualization_policy
from agent_interop.context_store.registry import RequestRefRegistry
from agent_interop.context_store.schema_tools import all_schema_tools
from agent_interop.context_store.store import ContextStore, StoredEntry
from agent_interop.context_store.tools import (
    all_internal_tools,
    internal_tool_names,
    read_result_tool,
    recall_history_tool,
    search_history_tool,
)

__all__ = [
    "ContextStore",
    "InternalExecutionContext",
    "InternalToolExecutor",
    "InternalToolResult",
    "RequestRefRegistry",
    "StoredEntry",
    "VirtualizationPolicy",
    "all_internal_tools",
    "all_schema_tools",
    "default_virtualization_policy",
    "internal_tool_names",
    "is_valid_ref",
    "read_result_tool",
    "recall_history_tool",
    "search_history_tool",
]
