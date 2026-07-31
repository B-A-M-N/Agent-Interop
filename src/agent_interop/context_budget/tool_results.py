"""Safe defaults for model-visible tool-result budgeting.

Interop does not own every client tool.  Unknown tool output is consequently
verbatim by default; integrations may opt into a narrower representation only
where they can preserve action-critical evidence.
"""

from __future__ import annotations

from enum import Enum


class ToolResultPolicy(str, Enum):
    VERBATIM = "verbatim"
    BOUNDED_LINES = "bounded_lines"
    STRUCTURED_REDUCTION = "structured_reduction"
    RETRY_NARROWER = "retry_narrower"
    CONTROLLER_SUMMARY = "controller_summary"


def default_tool_result_policy(tool_name: str) -> ToolResultPolicy:
    """Classify only well-known, safely pageable tool shapes.

    Patch and diagnostic tools retain exact output because line anchors,
    hashes, and failure details can be execution-critical.
    """
    normalized = tool_name.lower()
    if normalized in {"read_file", "read", "grep", "search", "list_directory", "list_dir"}:
        return ToolResultPolicy.BOUNDED_LINES
    if normalized in {"directory_list", "list_files"}:
        return ToolResultPolicy.RETRY_NARROWER
    if normalized in {"json_query", "query_json"}:
        return ToolResultPolicy.STRUCTURED_REDUCTION
    return ToolResultPolicy.VERBATIM
