"""Runner — replay captured cases with different repair policies.

Replays a ReplayCase with each standard repair policy and measures
whether the repair actually helped.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from agent_interop.abi import CanonicalTool, CanonicalToolChoice
from agent_interop.config import RepairPolicy
from agent_interop.repair.schema import validate_against_schema
from agent_interop.replay.types import (
    REPAIR_POLICIES,
    ReplayCase,
    ReplayInvariant,
    ReplayResult,
)

logger = logging.getLogger("agent_interop.replay")


async def replay_case(
    case: ReplayCase,
    policy_name: str,
    repair_policy: RepairPolicy,
    *,
    regenerate_fn: Any = None,
) -> ReplayResult:
    """Replay a single case with the given repair policy.

    Processes the upstream response through the repair pipeline
    and measures the outcome.
    """
    if not case.canonical_request:
        return ReplayResult(
            case_id=case.case_id,
            policy_name=policy_name,
            repair_policy=repair_policy,
            diagnostics=("No canonical request in case",),
        )

    # Get tool candidates from the raw response
    raw_response = case.raw_upstream_response
    if not raw_response:
        return ReplayResult(
            case_id=case.case_id,
            policy_name=policy_name,
            repair_policy=repair_policy,
            diagnostics=("No raw upstream response",),
        )

    # Extract tool calls from raw response
    candidates = _extract_candidates_from_raw(raw_response, case.tool_registry)

    if not candidates:
        # No tool calls to repair — check if that's expected
        has_tool_invariant = any(
            inv.type in ("tool_name", "tool_count")
            for inv in case.expected_invariants
        )
        return ReplayResult(
            case_id=case.case_id,
            policy_name=policy_name,
            repair_policy=repair_policy,
            executable=not has_tool_invariant,
            diagnostics=("No tool candidates found",),
        )

    # Process each candidate through repair
    from agent_interop.transaction import ToolTransactionContext, tool_transaction_service

    context = ToolTransactionContext(
        tool_choice=case.canonical_request.tool_choice if case.canonical_request else CanonicalToolChoice.auto(),
        repair_policy=repair_policy,
        regenerate_fn=regenerate_fn,
    )

    results = []
    for candidate in candidates:
        decision = await tool_transaction_service(
            candidate,
            list(case.tool_registry),
            context=context,
        )
        results.append(decision)

    # Measure outcomes
    accepted = [r for r in results if r.is_accepted]
    rejected = [r for r in results if not r.is_accepted]

    # Check invariants
    tool_identity_preserved = _check_tool_identity(accepted, case.expected_invariants)
    arguments_valid = all(
        _check_arguments_valid(r, case.tool_registry) for r in accepted
    )

    return ReplayResult(
        case_id=case.case_id,
        policy_name=policy_name,
        repair_policy=repair_policy,
        executable=len(accepted) > 0 and len(rejected) == 0,
        arguments_valid=arguments_valid,
        tool_identity_preserved=tool_identity_preserved,
        retry_avoided=len(accepted) > 0,
        output_tool_name=accepted[0].accepted_block.name if accepted and accepted[0].accepted_block else None,
        output_arguments=accepted[0].accepted_block.arguments if accepted and accepted[0].accepted_block else None,
        repair_steps=tuple(
            str(step)
            for r in results
            for step in (r.outcome.steps or [])
        ),
        diagnostics=tuple(
            f"{r.candidate.name}: {r.outcome.error}"
            for r in rejected
        ),
    )


async def replay_all_policies(
    case: ReplayCase,
    *,
    regenerate_fn: Any = None,
) -> Mapping[str, ReplayResult]:
    """Replay a case with all standard repair policies."""
    results = {}
    for policy_name, repair_policy in REPAIR_POLICIES.items():
        result = await replay_case(
            case, policy_name, repair_policy, regenerate_fn=regenerate_fn
        )
        results[policy_name] = result
    return results


def _extract_candidates_from_raw(
    raw_response: Mapping[str, Any],
    tools: Sequence[CanonicalTool],
) -> list[Any]:
    """Extract raw tool call candidates from an upstream response."""
    from agent_interop.abi import RawToolCallCandidate

    candidates = []

    # OpenAI format
    choices = raw_response.get("choices", [])
    for choice in choices:
        msg = choice.get("message", {})
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            candidates.append(RawToolCallCandidate(
                id=tc.get("id"),
                name=fn.get("name"),
                raw_arguments=fn.get("arguments", "{}"),
                source_protocol="openai_chat",
            ))

    # Anthropic format
    for block in raw_response.get("content", []):
        if block.get("type") == "tool_use":
            candidates.append(RawToolCallCandidate(
                id=block.get("id"),
                name=block.get("name"),
                raw_arguments=str(block.get("input", {})),
                source_protocol="anthropic_messages",
            ))

    return candidates


def _check_tool_identity(
    decisions: list[Any],
    invariants: tuple[ReplayInvariant, ...],
) -> bool:
    """Check if tool identity is preserved."""
    expected_name_inv = [
        inv for inv in invariants if inv.type == "tool_name"
    ]
    if not expected_name_inv:
        return True

    expected_name = expected_name_inv[0].expected
    return all(
        d.accepted_block.name == expected_name
        for d in decisions
        if d.accepted_block
    )


def _check_arguments_valid(
    decision: Any,
    tools: Sequence[CanonicalTool],
) -> bool:
    """Check if the repaired arguments validate against the tool schema."""
    if not decision.accepted_block or not decision.accepted_block.arguments:
        return False

    tool_name = decision.accepted_block.name
    tool = next((t for t in tools if t.name == tool_name), None)
    if not tool:
        return False

    issues = validate_against_schema(decision.accepted_block.arguments, tool.input_schema)
    return len(issues) == 0
