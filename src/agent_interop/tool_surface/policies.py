"""Withheld-tool escalation policy."""

from __future__ import annotations


def should_replan_with_withheld_tool(tool_name: str, withheld_names: tuple[str, ...], retry_count: int) -> bool:
    """Only a single declared-but-hidden tool request earns a replan."""
    return retry_count == 0 and tool_name in withheld_names
