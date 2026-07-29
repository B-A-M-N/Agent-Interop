"""MCP diagnostics — compatibility testing and replay-case submission.

Provides tools to verify MCP tool compatibility, run diagnostics,
and submit replay cases for evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_interop.abi import CanonicalTool, CanonicalToolResultBlock
from agent_interop.replay import capture_case
from agent_interop.replay.types import CompatibilityKey, ReplayCase, ReplayInvariant

logger = logging.getLogger("agent_interop.mcp")


class MCPDiagnostics:
    """Run compatibility diagnostics for MCP tools."""

    def __init__(self, tools: list[dict[str, Any]] | None = None) -> None:
        self._mcp_tools = tools or []
        self._canonical_tools: list[CanonicalTool] = []

    def ingest(self, mcp_tools: list[dict[str, Any]]) -> list[str]:
        """Ingest MCP tools and return warnings."""
        from agent_interop.mcp.schemas import ingest_mcp_tools

        self._mcp_tools = mcp_tools
        self._canonical_tools, warnings = ingest_mcp_tools(mcp_tools)
        return warnings

    def check_schema_compatibility(self) -> list[dict[str, Any]]:
        """Check each MCP tool schema for compatibility issues."""
        issues = []
        for mcp_tool, canonical in zip(self._mcp_tools, self._canonical_tools):
            schema = canonical.input_schema
            if not schema:
                issues.append({
                    "tool": canonical.name,
                    "issue": "empty_schema",
                    "severity": "warning",
                })
            elif schema.get("type") != "object":
                issues.append({
                    "tool": canonical.name,
                    "issue": "non_object_schema",
                    "severity": "warning",
                    "detail": f"Schema type is '{schema.get('type')}', expected 'object'",
                })
        return issues

    def create_replay_case(
        self,
        tool_name: str,
        test_arguments: dict[str, Any],
        *,
        client_protocol: str = "mcp",
        upstream_protocol: str = "ollama_chat",
    ) -> ReplayCase:
        """Create a replay case for testing an MCP tool call."""
        tool = next(
            (t for t in self._canonical_tools if t.name == tool_name),
            None,
        )
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Build a simulated MCP request
        mcp_request = {
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": test_arguments,
            },
        }

        # Build the upstream response (simulating model output)
        upstream_response = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "tc_mcp_001",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": test_arguments,
                        },
                    }],
                },
            }],
        }

        return capture_case(
            client_protocol=client_protocol,
            upstream_protocol=upstream_protocol,
            inbound_request=mcp_request,
            upstream_request=mcp_request,
            raw_upstream_response=upstream_response,
            tool_registry=self._canonical_tools,
            expected_invariants=[
                ReplayInvariant(
                    type="tool_name",
                    expected=tool_name,
                    description=f"MCP tool {tool_name} should be called correctly",
                ),
            ],
            compatibility_key=CompatibilityKey(
                client_id="mcp",
                client_protocol=client_protocol,
                model_id="mcp-model",
                upstream_protocol=upstream_protocol,
                profile_id="mcp-default",
            ),
        )

    def encode_result_for_mcp(
        self,
        result: CanonicalToolResultBlock,
    ) -> dict[str, Any]:
        """Encode a canonical tool result back to MCP format."""
        return {
            "type": "tool_result",
            "tool_use_id": result.tool_call_id,
            "content": result.content,
            "is_error": result.is_error,
        }


async def run_mcp_diagnostics(
    mcp_tools: list[dict[str, Any]],
    *,
    upstream_protocol: str = "ollama_chat",
) -> dict[str, Any]:
    """Run full MCP compatibility diagnostics.

    Returns a report with compatibility issues and replay results.
    """
    diagnostics = MCPDiagnostics(mcp_tools)
    warnings = diagnostics.ingest(mcp_tools)
    schema_issues = diagnostics.check_schema_compatibility()

    return {
        "tools_count": len(mcp_tools),
        "warnings": warnings,
        "schema_issues": schema_issues,
        "canonical_tools": [t.name for t in diagnostics._canonical_tools],
    }
