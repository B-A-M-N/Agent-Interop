"""Projection type definitions (P0.7 / P1.6).

Moved here from gateway.py to break the backwards dependency where
projection.planner imported PrivateCapabilityPlan and Gateway from
agent_interop.gateway. The gateway will later re-export from this module.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrivateCapabilityPlan:
    """Request-scoped set of private tools that may be used (P0.7).

    Fields are intentionally small — the planner derives which
    __interop_* tools are active from the compatibility plan rather
    than the gateway guessing at invocation time.
    """

    read_result: bool = False
    recall_history: bool = False
    search_history: bool = False
    get_tool_schema: bool = False

    @property
    def has_any(self) -> bool:
        """True when at least one private capability is active."""
        return bool(
            self.read_result
            or self.recall_history
            or self.search_history
            or self.get_tool_schema
        )
