"""Controller safety invariants."""

from __future__ import annotations

from dataclasses import replace

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
    ToolCallProvenance,
)


def mark_controller_provenance(call: CanonicalToolCallBlock) -> CanonicalToolCallBlock:
    """Ensure evidence cannot attribute controller calls to the primary model."""
    return replace(
        call,
        provenance=ToolCallProvenance(source="compatibility_controller", dialect="canonical"),
    )


def missing_controller_result_ids(
    messages: list[CanonicalMessage], pending_tool_call_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return pending controller calls absent from the current client result."""
    returned = {
        block.tool_call_id
        for message in messages
        for block in message.content
        if isinstance(block, CanonicalToolResultBlock)
    }
    return tuple(call_id for call_id in pending_tool_call_ids if call_id not in returned)
