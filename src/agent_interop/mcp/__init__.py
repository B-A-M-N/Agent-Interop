"""MCP integration boundary.

MCP (Model Context Protocol) is a supported integration boundary,
NOT the internal canonical transport. This module provides:

- MCP tool schema ingestion (MCP → CanonicalTool)
- MCP call argument repair (via universal transaction service)
- MCP tool-result preservation
- MCP server/client naming and namespace mappings
- Replay-case submission and evaluation

The canonical ABI remains the internal center. MCP is one of several
supported boundaries (alongside Anthropic Messages, OpenAI Chat, OpenAI Responses).

``diagnostics.MCPDiagnostics`` / ``run_mcp_diagnostics`` are intentionally
NOT exported here: no production CLI command or gateway path currently
calls them. Import directly from ``interop.mcp.diagnostics`` if you are
building on them, but they are not a supported public surface until a
tested path actually wires them in.
"""

from agent_interop.mcp.schemas import (
    canonical_tool_to_mcp,
    check_namespace_collisions,
    ingest_mcp_tools,
    mcp_to_canonical_tool,
    strip_mcp_namespace,
)

__all__ = [
    "canonical_tool_to_mcp",
    "check_namespace_collisions",
    "ingest_mcp_tools",
    "mcp_to_canonical_tool",
    "strip_mcp_namespace",
]
