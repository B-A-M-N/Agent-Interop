"""Schema-on-demand internal tool (P0.16).

Provides __interop_get_tool_schema: a private capability that lets the model
request the full schema for an already client-authorized tool. This enables
a compact model view: core schemas + task-relevant schemas in context, and
a name+one-line-description index for the rest.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalTool


def get_tool_schema_tool() -> CanonicalTool:
    """Return the schema-on-demand tool definition."""
    return CanonicalTool(
        name="__interop_get_tool_schema",
        description=(
            "Get the full input schema for a client-authorized tool. "
            "Use this when you need the exact parameters for a tool that is "
            "not in your current tool list. The tool must be authorized by "
            "the client."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the authorized tool to get the schema for.",
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    )


def all_schema_tools() -> tuple[CanonicalTool, ...]:
    """Return all schema-on-demand tools."""
    return (get_tool_schema_tool(),)
