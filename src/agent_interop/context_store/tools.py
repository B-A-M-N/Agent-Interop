"""Private Interop retrieval tools.

These tools are Interop-internal: they are exposed to the local model so it
can retrieve slices of the full client state that Interop retains, but they
are NEVER forwarded to the coding client. They generalize the
``controller_delegate_tool`` precedent.

Each schema is intentionally tiny — the model-visible surface stays small
even though the retained state behind a ref can be enormous.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalTool


def read_result_tool() -> CanonicalTool:
    """Read a slice of a stored tool result by ref."""
    return CanonicalTool(
        name="__interop_read_result",
        description=(
            "Read a stored tool result. Returns a bounded slice of the full "
            "content retained by Interop. Use start_line/line_count to page "
            "through large results. Structured metadata (total_lines, has_more, "
            "next_start_line) is returned alongside the content."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Opaque handle returned when the result was stored.",
                },
                "start_line": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "1-indexed line to start from.",
                },
                "line_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 500,
                    "description": (
                        "Number of lines to return. Omit for a bounded default "
                        "(200 lines). Maximum 500."
                    ),
                },
            },
            "required": ["ref"],
            "additionalProperties": False,
        },
    )


def recall_history_tool() -> CanonicalTool:
    """Recall a stored history fragment by ref."""
    return CanonicalTool(
        name="__interop_recall_history",
        description=(
            "Recall a stored fragment of earlier conversation history retained "
            "by Interop."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Opaque handle returned when the fragment was stored.",
                },
            },
            "required": ["ref"],
            "additionalProperties": False,
        },
    )


def search_history_tool() -> CanonicalTool:
    """Search stored history/tool results for a query."""
    return CanonicalTool(
        name="__interop_search_history",
        description=(
            "Search stored conversation history and tool results for a query. "
            "Returns matching refs with a short snippet."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Substring to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "description": "Maximum number of matches to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )


def all_internal_tools() -> tuple[CanonicalTool, ...]:
    """Return all private Interop tools."""
    return (read_result_tool(), recall_history_tool(), search_history_tool())


def internal_tool_names() -> frozenset[str]:
    return frozenset(t.name for t in all_internal_tools())
