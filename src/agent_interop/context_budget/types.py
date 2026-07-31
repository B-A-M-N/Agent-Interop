"""Types for deterministic, conservative request context budgeting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenEstimate:
    input_tokens: int = 0
    confidence: str = "estimated"


@dataclass(frozen=True)
class ContextBreakdown:
    system_tokens: int = 0
    message_tokens: int = 0
    tool_schema_tokens: int = 0
    prompted_contract_tokens: int = 0
    provider_overhead_tokens: int = 0
    output_reserve_tokens: int = 0
    total_required_tokens: int = 0
    confidence: str = "estimated"


@dataclass(frozen=True)
class ContextPlan:
    runtime_limit_tokens: int = 0
    safe_limit_tokens: int = 0
    before: ContextBreakdown = ContextBreakdown()
    after: ContextBreakdown = ContextBreakdown()
    fits_directly: bool = True
    compaction_required: bool = False
    selected_strategy: str = "direct"
    preserved_message_indices: tuple[int, ...] = ()
    compacted_message_indices: tuple[int, ...] = ()
    transformations: tuple[str, ...] = ()

