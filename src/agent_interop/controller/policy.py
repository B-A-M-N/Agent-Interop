"""Controller safety invariants."""

from __future__ import annotations

from dataclasses import replace

from agent_interop.abi import CanonicalToolCallBlock, ToolCallProvenance


def mark_controller_provenance(call: CanonicalToolCallBlock) -> CanonicalToolCallBlock:
    """Ensure evidence cannot attribute controller calls to the primary model."""
    return replace(
        call,
        provenance=ToolCallProvenance(source="compatibility_controller", dialect="canonical"),
    )
