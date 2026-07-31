"""Replay types — capture and replay contracts for deterministic evaluation.

A ReplayCase captures a complete request/response cycle so it can be
replayed with different repair policies to measure the actual benefit
of repair transformations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTool
from agent_interop.config import RepairPolicy, RepairTier


@dataclass(frozen=True)
class CompatibilityKey:
    """Exact client/model/backend/profile tuple for empirical compatibility.

    NOTE: evidence_verified is deliberately NOT part of the key identity.
    Verification is record state, not tuple identity. The evidence store
    tracks verification state separately.
    """

    client_id: str = ""
    client_version: str = ""
    client_protocol: str = ""

    model_id: str = ""
    model_digest: str = ""
    quantization: str = ""

    backend_kind: str = ""
    backend_version: str = ""
    upstream_protocol: str = ""
    chat_template_digest: str = ""

    profile_id: str = ""
    profile_revision: str = ""

    # Additional dimensions for exact tuple verification
    tool_schema_fingerprint: str = ""
    streaming: bool = False
    effective_tool_mode: str = ""
    parser_id: str = ""
    template_revision: str = ""
    backend_serving_config: str = ""

    # Compatibility-planning identity. Evidence from one path/tool surface
    # must never be reused for another merely because model/profile match.
    interop_build_commit: str = ""
    interop_build_dirty: bool | None = None
    planner_revision: str = ""
    runtime_context_tokens: int = 0
    runtime_capability_digest: str = ""
    compatibility_path: str = ""
    attempt_kind: str = ""
    controller_model_id: str = ""
    controller_model_digest: str = ""
    controller_profile_revision: str = ""
    tool_surface_mode: str = ""
    visible_tool_fingerprint: str = ""
    tool_selector_revision: str = ""
    context_strategy: str = ""
    context_plan_revision: str = ""
    streaming_policy: str = ""


@dataclass(frozen=True)
class CompatibilityQuirk:
    """A known quirk for a specific compatibility tuple."""

    description: str = ""
    severity: str = "info"  # info | warning | error
    workaround: str = ""


@dataclass(frozen=True)
class EvidencePassFailBreakdown:
    """Detailed pass/fail counters for a compatibility test run."""

    total_samples: int = 0
    tool_selection_pass: int = 0
    tool_selection_fail: int = 0
    valid_call_pass: int = 0
    valid_call_fail: int = 0
    task_completion_pass: int = 0
    task_completion_fail: int = 0
    streaming_equivalent_pass: int = 0
    streaming_equivalent_fail: int = 0
    history_round_trip_pass: int = 0
    history_round_trip_fail: int = 0


@dataclass(frozen=True)
class CompatibilityResult:
    """Empirical compatibility result for an exact client/model/backend/profile tuple."""

    tested_at: str = ""  # ISO datetime of last automated test
    sample_count: int = 0

    # Rates (0.0 - 1.0)
    tool_selection_rate: float = 0.0
    valid_call_rate_before_repair: float = 0.0
    valid_call_rate_after_repair: float = 0.0
    task_completion_rate: float = 0.0

    deterministic_repair_rate: float = 0.0
    regeneration_rate: float = 0.0
    rejection_rate: float = 0.0

    # Booleans
    streaming_equivalent: bool = False
    history_round_trip_valid: bool = False
    verified_capabilities: frozenset[str] = field(default_factory=frozenset)
    known_quirks: tuple[CompatibilityQuirk, ...] = ()

    # ── Evidence lifecycle ────────────────────────────────────────────
    created_at: str = ""  # ISO datetime when first recorded
    last_verified_at: str = ""  # ISO datetime of last manual/automated verification
    passes_expiry_hours: int = 720  # Default 30-day staleness window
    manually_verified: bool = False  # Has a human reviewed this evidence
    revoked: bool = False  # Explicitly revoked (no longer trusted)
    revocation_reason: str = ""  # Why revoked (empty if not revoked)
    attestation: str = ""  # Reviewer's note recorded at approval time (empty if never approved)

    # ── Counter-based rate aggregation (schema v4) ───────────────────
    # Rates are DERIVED from these counters; counters are accumulated per
    # tool-call decision (NOT per request) so a request with 10 decisions
    # moves the aggregate exactly as much as a request with 1.
    last_observed_at: str = ""  # ISO datetime of last live observation (NOT certification)
    no_selection_request_count: int = 0  # Requests where model made no tool-call decision
    candidate_count: int = 0  # Total tool-call decisions observed
    valid_unchanged_count: int = 0  # Decisions that were VALID_UNCHANGED
    repaired_count: int = 0  # Decisions that were REPAIRED
    regenerated_count: int = 0  # Decisions that were REGENERATED
    accepted_count: int = 0  # Decisions ultimately accepted
    rejected_count: int = 0  # Decisions ultimately rejected

    pass_fail_breakdown: EvidencePassFailBreakdown | None = None

    # ── Conformance-level provenance (testing/levels.py) ──────────────
    battery_version: str = ""
    """testing.levels.BATTERY_VERSION at the time this result's L0-L4
    level (if any) was computed. Empty for evidence that never went
    through level computation (e.g. hand-stored records). A stored
    result whose battery_version doesn't match the CURRENT
    testing.levels.BATTERY_VERSION is stale — the test-name-to-level
    mapping or battery composition has changed since, so the level it
    implies should not be presented as current (see
    evidence.store.capability_source)."""


@dataclass(frozen=True)
class ReplayInvariant:
    """An expected invariant that should hold after repair."""

    type: str = ""  # tool_name | arguments | tool_count | stop_reason | no_undeclared_tools
    expected: Any = None
    description: str = ""


@dataclass(frozen=True)
class ReplayCase:
    """A captured request/response cycle for deterministic replay."""

    format_version: str = "interop.replay.v1"

    case_id: str = ""
    client_protocol: str = ""
    upstream_protocol: str = ""
    compatibility_key: CompatibilityKey = field(default_factory=CompatibilityKey)

    inbound_request: Mapping[str, Any] = field(default_factory=dict)
    canonical_request: CanonicalRequest | None = None
    upstream_request: Mapping[str, Any] = field(default_factory=dict)

    raw_upstream_response: Mapping[str, Any] | None = None
    raw_upstream_frames: tuple[str, ...] = ()

    tool_registry: tuple[CanonicalTool, ...] = ()
    expected_invariants: tuple[ReplayInvariant, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying a case with a specific repair policy."""

    case_id: str = ""
    policy_name: str = ""
    repair_policy: RepairPolicy = field(default_factory=RepairPolicy)

    # Outcomes
    executable: bool = False
    arguments_valid: bool = False
    tool_identity_preserved: bool = False
    retry_avoided: bool = False
    task_progress_improved: bool = False
    introduced_unintended_execution: bool = False

    # Details
    output_tool_name: str | None = None
    output_arguments: Mapping[str, Any] | None = None
    repair_steps: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReplayComparison:
    """Comparison of the same case across multiple repair policies."""

    case: ReplayCase = field(default_factory=ReplayCase)
    results: Mapping[str, ReplayResult] = field(default_factory=dict)

    # Summary
    best_policy: str = ""
    repair_benefit: bool = False  # Did repair actually help?


# ── Standard repair policies for comparison ─────────────────────────────────

REPAIR_POLICIES: Mapping[str, RepairPolicy] = {
    "repair_disabled": RepairPolicy(
        enabled_tiers=frozenset(),
        max_regenerations=0,
    ),
    "syntax_only": RepairPolicy(
        enabled_tiers=frozenset({RepairTier.SYNTAX_ONLY}),
        max_regenerations=0,
    ),
    "safe_shape": RepairPolicy(
        enabled_tiers=frozenset({RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE}),
        max_regenerations=0,
    ),
    "coercive": RepairPolicy(
        enabled_tiers=frozenset({RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE, RepairTier.COERCIVE}),
        max_regenerations=0,
    ),
    "regeneration": RepairPolicy(
        enabled_tiers=frozenset({RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE}),
        max_regenerations=1,
    ),
}
