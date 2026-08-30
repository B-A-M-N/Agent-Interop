"""Types for deterministic, conservative request context budgeting."""

from __future__ import annotations

from dataclasses import dataclass, field


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
class RequestCostSnapshot:
    """Request-scoped token/byte cost, computed ONCE per planning pass.

    Every downstream consumer (requirements derivation, tool-surface
    planning, context budgeting, evidence-key fingerprinting) reads this
    snapshot instead of re-serializing the same system prompt, message
    history, and tool schemas. For a large conversation each redundant
    serialization is a multi-megabyte json.dumps; the happy path must pay
    that exactly once.

    All values are derived from the request the planning pass actually
    sees — the caller rebuilds the snapshot when the projected request
    diverges from the authoritative one, so consumers never read a stale
    history size.
    """

    system_tokens: int = 0
    message_tokens: int = 0
    # Full declared tool surface.
    tool_schema_tokens: int = 0
    tool_schema_bytes: int = 0
    # Canonical full-surface fingerprint (same form the evidence key uses).
    tool_schema_fingerprint: str = ""
    # Per-tool cost, keyed by tool name. ``tool_item_bytes`` holds the
    # serialized per-item JSON byte length (list brackets excluded) so any
    # subset — in ANY order — can be priced exactly without re-serializing.
    per_tool_tokens: dict[str, int] = field(default_factory=dict)
    tool_item_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def estimated_input_tokens(self) -> int:
        return self.system_tokens + self.message_tokens + self.tool_schema_tokens


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
    # P0.6: True when context capacity is unknown (no source provided a limit).
    # Unknown does NOT mean infinite — the gateway must either fail with
    # CONTEXT_CAPACITY_UNKNOWN or apply a conservative operator fallback.
    capacity_unknown: bool = False
    # P0.23: Executable policy flags derived from strategy × route config
    allow_tool_reduction: bool = True
    allow_result_virtualization: bool = True
    allow_history_paging: bool = True
    allow_semantic_summary: bool = True
    allow_controller_decomposition: bool = True

