"""MCP schema conversion — MCP tool definitions ↔ CanonicalTool.

MCP tools use the format:
{
    "name": "server__tool_name",
    "description": "...",
    "inputSchema": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}

Namespace stripping (mcp__server__tool → tool) is NOT universally safe.
It requires collision checking.
"""

from __future__ import annotations

from typing import Any

from agent_interop.abi import CanonicalTool


def mcp_to_canonical_tool(
    mcp_tool: dict[str, Any],
    *,
    strip_namespace: bool = False,
) -> CanonicalTool:
    """Convert an MCP tool definition to CanonicalTool.

    Args:
        mcp_tool: MCP tool definition with name, description, inputSchema.
        strip_namespace: If True, strip mcp__server__tool → tool.
            Only safe when no collisions exist (checked separately).

    Returns:
        CanonicalTool with the MCP schema preserved losslessly.
    """
    name = mcp_tool.get("name", "")
    description = mcp_tool.get("description", "")
    input_schema = mcp_tool.get("inputSchema", mcp_tool.get("input_schema", {
        "type": "object",
        "properties": {},
    }))

    if strip_namespace:
        name = strip_mcp_namespace(name)

    return CanonicalTool(
        name=name,
        description=description,
        input_schema=input_schema,
    )


def canonical_tool_to_mcp(tool: CanonicalTool) -> dict[str, Any]:
    """Convert a CanonicalTool to MCP format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


def strip_mcp_namespace(name: str) -> str:
    """Strip MCP namespace prefix (mcp__server__tool → tool).

    WARNING: Only safe when no naming collisions exist after stripping.
    Use check_namespace_collisions() before bulk stripping.
    """
    if "__" in name:
        # MCP convention: mcp__server__tool → tool
        return name.rsplit("__", 1)[-1]
    return name


def check_namespace_collisions(
    mcp_tools: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    """Check for naming collisions after namespace stripping.

    Returns list of (name1, name2) pairs that would collide.
    """
    stripped_names: dict[str, str] = {}
    collisions: list[tuple[str, str]] = []

    for tool in mcp_tools:
        original = tool.get("name", "")
        stripped = strip_mcp_namespace(original)
        if stripped in stripped_names:
            collisions.append((stripped_names[stripped], original))
        else:
            stripped_names[stripped] = original

    return collisions


def ingest_mcp_tools(
    mcp_tools: list[dict[str, Any]],
    *,
    strip_namespaces: bool = False,
) -> tuple[list[CanonicalTool], list[str]]:
    """Ingest a list of MCP tools, converting to CanonicalTool.

    Args:
        mcp_tools: List of MCP tool definitions.
        strip_namespaces: If True, strip namespaces (only if no collisions).

    Returns:
        (canonical_tools, warnings) — warnings describe any issues.
    """
    warnings: list[str] = []

    if strip_namespaces:
        collisions = check_namespace_collisions(mcp_tools)
        if collisions:
            collision_desc = "; ".join(
                f"{a} and {b} → {a.rsplit('__', 1)[-1]}"
                for a, b in collisions
            )
            warnings.append(
                f"Namespace stripping would cause collisions: {collision_desc}. "
                f"Namespaces preserved."
            )
            strip_namespaces = False

    canonical_tools = [
        mcp_to_canonical_tool(tool, strip_namespace=strip_namespaces)
        for tool in mcp_tools
    ]

    return canonical_tools, warnings
