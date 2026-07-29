"""Universal Tool Transaction Service — single authority for all tool-call processing.

Every source of tool calls (native structured output, prompted envelopes, streaming
accumulations, regeneration output) produces ``RawToolCallCandidate``. This service
performs the entire pipeline:

    candidate
      → identity/canonicalize tool name
      → recover argument JSON (syntax recovery)
      → validate against JSON Schema
      → bounded deterministic repair
      → revalidate
      → optional regeneration
      → accept or reject

Returns ``ToolCallDecision`` — the single output contract. Callers MUST use
``accepted_block`` when present, never the original raw candidate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from agent_interop.abi import (
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    RawToolCallCandidate,
    RepairOutcome,
    RepairStatus,
    ToolCallCorrection,
    ToolCallDecision,
    ToolChoiceMode,
    new_tool_call_id,
)
from agent_interop.config import RepairPolicy
from agent_interop.repair.pipeline import RepairBudget, canonicalize_tool_name, repair_one
from agent_interop.replay.types import CompatibilityKey

logger = logging.getLogger("agent_interop.transaction")


class ToolBatchPolicy(str, Enum):
    """How to handle a batch of parallel tool calls."""
    ATOMIC = "atomic"       # all-or-nothing
    BEST_EFFORT = "best_effort"  # emit accepted calls even if some fail


# ─── Typed contracts ─────────────────────────────────────────────────────

class RegenerationFn(Protocol):
    """Async callback for hidden constrained regeneration.

    Takes a structured correction request and returns the model's
    regenerated output as a string (protocol-native format).
    """

    async def __call__(self, correction: dict[str, Any]) -> str: ...


class RepairTelemetry(Protocol):
    """Interface for recording repair telemetry events.

    Implementations can emit to logs, metrics, evidence store, etc.
    """

    def record_repair(
        self,
        tool_name: str,
        outcome: str,
        steps: Sequence[str],
        accepted: bool,
    ) -> None: ...

    def record_regeneration(
        self,
        tool_name: str,
        attempt: int,
        success: bool,
    ) -> None: ...

    def emit_repaired(
        self,
        request_id: str,
        tool_name: str,
        rules: Sequence[str],
        paths: Sequence[str],
    ) -> None: ...


@dataclass(frozen=True)
class ToolTransactionContext:
    """Context passed through the transaction pipeline.

    Carries request identity, tool choice, repair policy, and budget.
    """

    request_id: str = ""
    session_id: str = ""
    tool_choice: CanonicalToolChoice = field(default_factory=CanonicalToolChoice.auto)
    repair_policy: RepairPolicy = field(default_factory=RepairPolicy)
    client_id: str | None = None
    regenerate_fn: RegenerationFn | None = None
    telemetry: RepairTelemetry | None = None
    budget: RepairBudget | None = None  # Shared across all candidates in a batch
    compatibility_key: CompatibilityKey | None = None
    compatibility_verified: bool = False


@dataclass
class ToolBatchDecision:
    """Result of processing a batch of tool calls."""

    decisions: list[ToolCallDecision] = field(default_factory=list)
    accepted_blocks: list[CanonicalToolCallBlock] = field(default_factory=list)
    rejected_count: int = 0
    is_accepted: bool = True
    choice_error: str = ""  # Set when tool-choice policy is violated


def validate_batch_choice(
    candidates: Sequence[RawToolCallCandidate],
    choice: CanonicalToolChoice,
    tools: Sequence[CanonicalTool],
) -> str:
    """Validate a batch of candidates against the tool choice mode.

    Returns an empty string if valid, or an error message if the batch
    violates the tool-choice contract.

    Candidate names are canonicalized before comparison so that safely
    normalizable names (case, hyphens, namespaces) are not rejected
    before reaching the transaction service.

    Rules:
    - NONE: Any candidate is invalid.
    - NAMED: The named tool must exist. At least one candidate must be
      present. All candidates must resolve to the named tool.
    - REQUIRED: At least one candidate must be present. All candidates
      must resolve to a declared tool.
    - AUTO: Zero candidates is valid.
    """
    from agent_interop.repair.pipeline import canonicalize_tool_name

    tool_names = frozenset(t.name for t in tools)
    tools_list = list(tools)
    mode = choice.mode

    if mode == ToolChoiceMode.NONE:
        if candidates:
            names = [c.name for c in candidates if c.name]
            return f"Tool calls not permitted (choice=none): {', '.join(names)}"
        return ""

    if mode == ToolChoiceMode.NAMED:
        if not choice.name:
            return "Named tool choice has no tool name"
        if choice.name not in tool_names:
            return f"Named tool '{choice.name}' is not in the declared tools"
        if not candidates:
            return f"Required tool '{choice.name}' not called (no candidates)"
        for c in candidates:
            if not c.name:
                continue
            canonical = canonicalize_tool_name(c.name, tools_list)
            effective_name = canonical or c.name
            if effective_name != choice.name:
                return f"Candidate '{c.name}' does not match named tool '{choice.name}'"
        return ""

    if mode == ToolChoiceMode.REQUIRED:
        if not candidates:
            return "Required tool call missing (no candidates)"
        for c in candidates:
            if not c.name:
                continue
            canonical = canonicalize_tool_name(c.name, tools_list)
            if canonical is None:
                return f"Candidate '{c.name}' is not a declared tool"
        return ""

    # AUTO: always valid
    return ""


def _build_correction(
    candidate: RawToolCallCandidate,
    outcome: RepairOutcome,
) -> ToolCallCorrection | None:
    """Build a structured correction from repair outcome issues.

    Extracts the first final issue (if any) to populate the correction
    with failure location, observed/expected types, and retryability.
    """
    # Prefer final_issues (post-repair), fall back to initial_issues
    issues = outcome.final_issues or outcome.initial_issues
    if not issues:
        return None

    primary = issues[0]
    # Determine retryable status: retryable if the issue keyword suggests
    # a potentially fixable problem (type, format, enum, required) and
    # regeneration is available
    retryable_keywords = {"type", "format", "enum", "required", "additionalProperties", "minimum", "maximum"}
    retryable = primary.keyword in retryable_keywords

    # Extract allowed values from the issue if available (e.g., enums)
    allowed: list[str] = []
    if primary.keyword == "enum":
        # SchemaIssue.expected may contain the allowed set as stringified JSON
        import json
        try:
            parsed = json.loads(primary.expected)
            if isinstance(parsed, list):
                allowed = [str(v) for v in parsed]
        except (json.JSONDecodeError, TypeError):
            pass

    return ToolCallCorrection(
        tool_name=outcome.call_name or candidate.name or "",
        candidate_id=candidate.id or "",
        issue_path=".".join(str(p) for p in primary.path) if primary.path else "root",
        schema_keyword=primary.keyword,
        observed_type=primary.actual,
        expected_type=primary.expected,
        allowed_values=allowed,
        message=primary.message,
        retryable=retryable,
    )


async def tool_transaction_service(
    candidate: RawToolCallCandidate,
    tools: list[CanonicalTool],
    *,
    context: ToolTransactionContext | None = None,
    client_id: str | None = None,
    allow_regeneration: bool | None = None,
) -> ToolCallDecision:
    """Process a single RawToolCallCandidate through the full pipeline.

    Args:
        candidate: The raw model output that might be a tool call.
        tools: The list of registered canonical tool definitions.
        context: Transaction context carrying request ID, policy, tool choice,
            and the regeneration callback (``context.regenerate_fn``).
        allow_regeneration: If True and deterministic repair fails, attempt
            hidden constrained regeneration. Defaults to policy setting.

    Returns:
        A ``ToolCallDecision``. Check ``is_accepted`` before using.
    """
    # Derive effective settings from context/policy
    if context is None:
        context = ToolTransactionContext()
    regenerate_fn = context.regenerate_fn
    if allow_regeneration is None:
        allow_regeneration = context.repair_policy.max_regenerations > 0
    telemetry = context.telemetry
    request_id = context.request_id
    repair_policy = context.repair_policy
    # ``client_id`` may be passed as a kwarg (legacy) or carried on
    # the context.  Prefer the kwarg when set; fall back to the
    # context for the standard call path used by
    # ``process_tool_batch``.
    if client_id is None:
        client_id = context.client_id
    if not candidate.name:
        return ToolCallDecision(
            candidate=candidate,
            outcome=RepairOutcome(
                status=RepairStatus.REJECTED,
                call_name=candidate.name or "",
                accepted=None,
                error="Tool call has no name",
            ),
            accepted_block=None,
        )

    # ── Step 1: Tool-name canonicalization ────────────────────────────
    canonical_name = canonicalize_tool_name(candidate.name, tools, _policy=repair_policy)
    if canonical_name is None:
        return ToolCallDecision(
            candidate=candidate,
            outcome=RepairOutcome(
                status=RepairStatus.REJECTED,
                call_name=candidate.name,
                accepted=None,
                error=f"Tool '{candidate.name}' not found in registered tools",
            ),
            accepted_block=None,
        )

    tool_map = {t.name: t for t in tools}
    tool = tool_map[canonical_name]

    # ── Step 2-5: Run the validate-then-repair pipeline ───────────────
    # repair_one handles: arg recovery → validation → repair → revalidation
    raw_args = candidate.raw_arguments

    # Top-level list arguments are not supported. The canonical ABI types
    # ``CanonicalToolCallBlock.arguments`` as ``dict[str, Any]``, so a bare
    # list cannot be represented without inventing a wrapping field. Coding-
    # agent interoperability consistently requires object-root tool schemas,
    # so any top-level list is rejected uniformly regardless of schema.
    if isinstance(raw_args, list):
        return ToolCallDecision(
            candidate=candidate,
            outcome=RepairOutcome(
                status=RepairStatus.REJECTED,
                call_name=canonical_name,
                accepted=None,
                error="Top-level list arguments are not supported — tool must declare an object-root schema",
            ),
            accepted_block=None,
        )

    outcome = repair_one(
        call_name=canonical_name,
        call_arguments=raw_args,
        tools=tools,
        policy=repair_policy,
        client_id=client_id,
        telemetry=telemetry,
        request_id=request_id,
        budget=context.budget,
        compatibility_key=context.compatibility_key,
        compatibility_verified=context.compatibility_verified,
    )

    # ── Step 6: Optional regeneration ─────────────────────────────────
    # Only regenerate when:
    # - Deterministic repair failed
    # - Tool name was recoverable (we know which tool)
    # - Original choice is REQUIRED/NAMED, or AUTO correction is enabled
    # - Attempt budget remains
    if (
        not outcome.is_accepted
        and canonical_name is not None
        and allow_regeneration
        and regenerate_fn is not None
    ):
        # Check tool-choice eligibility — only regenerate when the
        # tool-choice mode makes the call mandatory (REQUIRED/NAMED).
        # AUTO mode alone must not trigger hidden regeneration.
        choice_mode = context.tool_choice.mode if context.tool_choice else ToolChoiceMode.AUTO
        if choice_mode in (ToolChoiceMode.REQUIRED, ToolChoiceMode.NAMED):
            # Enforce regeneration budget before calling the model
            budget = context.budget
            max_regen = getattr(context.repair_policy, 'max_regenerations', 1) if hasattr(context, 'repair_policy') else 1
            if budget is not None and not budget.can_regenerate(max_regen):
                logger.info("regeneration skipped for %s: budget exhausted", canonical_name)
            else:
                if budget is not None:
                    budget.regeneration_attempts += 1
                outcome = await _attempt_regeneration(
                    candidate=candidate,
                    canonical_name=canonical_name,
                    tool=tool,
                    outcome=outcome,
                    regenerate_fn=regenerate_fn,
                    telemetry=telemetry,
                    request_id=request_id,
                    tools=tools,
                    policy=repair_policy,
                    budget=budget,
                    client_id=client_id,
                    compatibility_key=context.compatibility_key,
                    compatibility_verified=context.compatibility_verified,
                )

    # ── Step 7: Build result ──────────────────────────────────────────
    if outcome.is_accepted and outcome.accepted is not None:
        accepted_block = CanonicalToolCallBlock(
            id=candidate.id or new_tool_call_id(),
            name=outcome.call_name,
            arguments=outcome.accepted,
        )
        return ToolCallDecision(
            candidate=candidate,
            outcome=outcome,
            accepted_block=accepted_block,
        )

    # Build structured correction data from the first final issue
    correction = _build_correction(candidate, outcome)
    return ToolCallDecision(
        candidate=candidate,
        outcome=outcome,
        accepted_block=None,
        correction=correction,
    )


async def _attempt_regeneration(
    candidate: RawToolCallCandidate,
    canonical_name: str,
    tool: CanonicalTool,
    outcome: RepairOutcome,
    regenerate_fn: RegenerationFn,
    telemetry: RepairTelemetry | None = None,
    request_id: str = "",
    tools: list[CanonicalTool] | None = None,
    policy: RepairPolicy | None = None,
    budget: RepairBudget | None = None,
    client_id: str | None = None,
    compatibility_key: CompatibilityKey | None = None,
    compatibility_verified: bool = False,
) -> RepairOutcome:
    """Attempt hidden constrained regeneration when deterministic repair fails.

    Regenerated output is routed through the full repair pipeline (extraction →
    parsing → deterministic rules → schema validation) rather than being
    privileged with direct acceptance.

    Only performed when the caller has explicitly enabled regeneration
    and the model emitted a declared but malformed tool call.
    """
    from agent_interop.repair.regenerate import RegenerationOrchestrator

    max_attempts = getattr(policy, 'max_regenerations', 1) if policy else 1
    try:
        orchestrator = RegenerationOrchestrator(max_attempts=max_attempts)
        corrected = await orchestrator.attempt(
            tool_name=canonical_name,
            raw_arguments=candidate.raw_arguments,
            issues=outcome.initial_issues or [],
            tool=tool,
            regenerate_fn=regenerate_fn,
        )
    except Exception as exc:
        logger.warning("regeneration failed for %s: %s", canonical_name, exc)
        return outcome

    if corrected is None:
        return outcome

    # Route the regenerated output through the full repair pipeline
    # instead of accepting it directly. This ensures exact parsing,
    # deterministic repair rules, and schema validation are all applied.
    regenerated_args = corrected.get("arguments", {})
    # Serialize back to JSON string so repair_one can re-parse it
    import json
    raw_json = json.dumps(regenerated_args)

    tools_list = tools or [tool]
    pipeline_outcome = repair_one(
        call_name=canonical_name,
        call_arguments=raw_json,
        tools=tools_list,
        policy=policy,
        client_id=client_id,
        telemetry=telemetry,
        request_id=request_id,
        budget=budget,
        compatibility_key=compatibility_key,
        compatibility_verified=compatibility_verified,
    )

    if pipeline_outcome.is_accepted and pipeline_outcome.accepted is not None:
        if telemetry and request_id:
            telemetry.emit_repaired(
                request_id, canonical_name,
                rules=["regeneration"] + [s.rule for s in pipeline_outcome.steps],
                paths=["root"] + [str(s.path) for s in pipeline_outcome.steps],
            )
        return RepairOutcome(
            status=RepairStatus.REGENERATED,
            call_name=canonical_name,
            accepted=pipeline_outcome.accepted,
            initial_issues=outcome.initial_issues,
            final_issues=[],
            steps=pipeline_outcome.steps,
        )

    return outcome


async def process_tool_batch(
    candidates: Sequence[RawToolCallCandidate],
    tools: list[CanonicalTool],
    *,
    context: ToolTransactionContext | None = None,
    policy: ToolBatchPolicy = ToolBatchPolicy.ATOMIC,
) -> ToolBatchDecision:
    """Process a batch of tool-call candidates with atomic semantics.

    Default policy is ATOMIC: all calls must be accepted, or none are emitted.
    This ensures streaming and non-streaming produce identical decisions.

    Enforces tool-choice policy before any candidate processing:
    - NONE rejects all candidates immediately.
    - REQUIRED rejects an empty batch immediately.
    - NAMED rejects empty or mismatched candidates immediately.

    Creates a shared RepairBudget that tracks latency, input bytes, and
    regeneration attempts across all candidates in the batch.

    The regeneration callback is read exclusively from
    ``context.regenerate_fn`` — callers must not pass it separately.
    """
    if context is None:
        context = ToolTransactionContext()

    # Reuse the request-scoped budget when supplied; create one only when absent.
    # This ensures all candidates and regenerations in a request share one budget.
    if getattr(context, 'budget', None) is None:
        from dataclasses import replace as _replace
        budget = RepairBudget()
        context = _replace(context, budget=budget)  # type: ignore[arg-type]

    # Resolve the effective batch policy. Gateway now always passes an explicit
    # value (read from the route's repair config), so it wins; fall back to the
    # context's repair policy only for non-Gateway callers.
    batch_policy = policy if policy else ToolBatchPolicy(
        getattr(context.repair_policy, 'batch_policy', 'atomic') if hasattr(context, 'repair_policy') else 'atomic'
    )

    # Enforce tool-choice policy at batch level before processing any candidate
    choice = context.tool_choice
    choice_error = validate_batch_choice(candidates, choice, tools)
    if choice_error:
        return ToolBatchDecision(
            is_accepted=False,
            rejected_count=len(candidates),
            choice_error=choice_error,
        )

    decisions = []
    for candidate in candidates:
        decision = await tool_transaction_service(
            candidate,
            tools,
            context=context,
            client_id=context.client_id if context else None,
        )
        decisions.append(decision)

    all_accepted = all(d.is_accepted for d in decisions)

    batch = ToolBatchDecision(
        decisions=decisions,
        rejected_count=sum(1 for d in decisions if not d.is_accepted),
        is_accepted=all_accepted,
    )

    if all_accepted:
        batch.accepted_blocks = [
            d.accepted_block for d in decisions if d.accepted_block is not None
        ]
    elif batch_policy == ToolBatchPolicy.BEST_EFFORT:
        # Best-effort policy: emit accepted calls even if some were rejected
        batch.accepted_blocks = [
            d.accepted_block for d in decisions if d.is_accepted and d.accepted_block is not None
        ]
        batch.is_accepted = len(batch.accepted_blocks) > 0

    return batch