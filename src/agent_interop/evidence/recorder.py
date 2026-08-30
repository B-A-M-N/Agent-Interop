"""Live compatibility-evidence recording (extracted from Gateway).

One module owns the read-modify-write merge of a live request's outcome
into the evidence store: per-decision counter accumulation, pre-v4 record
seeding, rate re-derivation from merged counters, and the never-touch-
certification-state rule.

Coupling contract: pure mechanism over (invocation, execution, store) —
no Gateway state, no policy decisions about WHEN to record (callers gate
on "evidence store configured" and "tools were offered").
"""

from __future__ import annotations

import logging
from dataclasses import replace as _dc_replace
from datetime import UTC, datetime
from typing import Any

from agent_interop.execution import InteropRequestExecution
from agent_interop.replay.types import CompatibilityResult

logger = logging.getLogger("agent_interop.evidence.recorder")

__all__ = ["record_evidence_observation", "selected_evidence_key"]


def selected_evidence_key(
    invocation: Any,
    execution: InteropRequestExecution,
) -> Any:
    """Use an enriched key only when a controller actually selected it."""
    original = invocation.compatibility_key
    selected = execution.compatibility_key
    if selected is None or original is None:
        return selected or original
    controller_dimensions = (
        "controller_model_id",
        "controller_model_digest",
        "controller_profile_revision",
    )
    if any(
        getattr(selected, field, "") != getattr(original, field, "")
        for field in controller_dimensions
    ):
        return selected
    return original


def record_evidence_observation_inner(
    invocation: Any,
    execution: InteropRequestExecution,
    store: Any,
) -> None:
    """Merge this request's outcome into the store's record for its tuple.

    Per-tool-call counters are accumulated exactly (each decision
    contributes one unit) and the rate fields are re-derived from the
    merged counters — rates are never averaged in per-request space.
    Verification state (``manually_verified`` / ``last_verified_at``) and
    revocation state are never touched by live traffic. Pre-v4 records
    that stored rates without counters are seeded from their stored rates
    before merging. ``task_completion_rate`` is not derivable from a
    single live request, so it is left at the existing value.
    """
    key = selected_evidence_key(invocation, execution)
    if key is None:
        return

    decisions = execution.tool_decisions
    n = len(decisions)

    # One row per tool-call decision, for `interop repair stats`.
    route_id = invocation.route.id if invocation.route is not None else ""
    for d in decisions:
        store.record_repair_event_async(
            route_id=route_id,
            model_id=key.model_id,
            client_id=key.client_id,
            tool_name=d.tool_name,
            outcome=d.outcome_status,
            repair_rules=d.repair_steps,
        )

    # Per-request OBSERVATION captured as COUNTERS (one unit per tool-call
    # decision), not as rates. task_completion_rate is deliberately
    # excluded — it cannot be observed from a single live request.
    if n == 0:
        # Tools were offered but the model produced no tool-call decision
        # (e.g. it replied with text). Only the no-selection signal.
        observation: dict[str, int | bool] = {
            "candidate_count": 0,
            "valid_unchanged_count": 0,
            "repaired_count": 0,
            "regenerated_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "no_selection": True,
        }
    else:
        from agent_interop.abi import RepairStatus

        valid_unchanged = sum(
            1 for d in decisions
            if d.outcome_status == RepairStatus.VALID_UNCHANGED.value
        )
        repaired = sum(
            1 for d in decisions if d.outcome_status == RepairStatus.REPAIRED.value
        )
        regenerated = sum(
            1 for d in decisions if d.outcome_status == RepairStatus.REGENERATED.value
        )
        accepted = sum(1 for d in decisions if d.accepted)
        observation = {
            "candidate_count": n,
            "valid_unchanged_count": valid_unchanged,
            "repaired_count": repaired,
            "regenerated_count": regenerated,
            "accepted_count": accepted,
            "rejected_count": (n - accepted),
            "no_selection": False,
        }

    existing = store.get_result(key)
    now = datetime.now(UTC).isoformat()

    if existing is not None:
        new_n = existing.sample_count + 1

        # ── Migration / seeding for pre-v4 records ─────────────────────
        # A record written before this fix stores rates but has
        # candidate_count == 0. Seed the counters from the stored rates
        # BEFORE adding this request's contribution. The sample_count
        # guard ensures we only seed when there is something to seed from.
        if existing.candidate_count == 0 and existing.no_selection_request_count == 0 and existing.sample_count > 0:
            seed_n = existing.sample_count
            seed_candidate_count = seed_n
            seed_valid_unchanged_count = round(
                existing.valid_call_rate_before_repair * seed_n
            )
            seed_repaired_count = round(
                existing.deterministic_repair_rate * seed_n
            )
            seed_regenerated_count = round(
                existing.regeneration_rate * seed_n
            )
            seed_accepted_count = round(
                existing.valid_call_rate_after_repair * seed_n
            )
            seed_rejected_count = round(existing.rejection_rate * seed_n)
            seed_no_selection = round(
                (1.0 - existing.tool_selection_rate) * seed_n
            )
        else:
            # Genuine v4 record: pass existing counters through unchanged.
            seed_candidate_count = existing.candidate_count
            seed_valid_unchanged_count = existing.valid_unchanged_count
            seed_repaired_count = existing.repaired_count
            seed_regenerated_count = existing.regenerated_count
            seed_accepted_count = existing.accepted_count
            seed_rejected_count = existing.rejected_count
            seed_no_selection = existing.no_selection_request_count

        # Accumulate this request's counters on top of the seed.
        c_candidate = seed_candidate_count + observation["candidate_count"]
        c_valid_unchanged = (
            seed_valid_unchanged_count + observation["valid_unchanged_count"]
        )
        c_repaired = seed_repaired_count + observation["repaired_count"]
        c_regenerated = (
            seed_regenerated_count + observation["regenerated_count"]
        )
        c_accepted = seed_accepted_count + observation["accepted_count"]
        c_rejected = seed_rejected_count + observation["rejected_count"]
        c_no_selection = seed_no_selection + (
            1 if observation["no_selection"] else 0
        )

        # Derive rates fresh from the merged counters. Rates are passed as
        # explicit kwargs (not a ``**dict`` spread) so mypy can verify each
        # field's type.
        result = _dc_replace(
            existing,
            sample_count=new_n,
            last_observed_at=now,
            candidate_count=c_candidate,
            valid_unchanged_count=c_valid_unchanged,
            repaired_count=c_repaired,
            regenerated_count=c_regenerated,
            accepted_count=c_accepted,
            rejected_count=c_rejected,
            no_selection_request_count=c_no_selection,
            valid_call_rate_before_repair=(
                c_valid_unchanged / c_candidate if c_candidate else 0.0
            ),
            valid_call_rate_after_repair=(
                c_accepted / c_candidate if c_candidate else 0.0
            ),
            deterministic_repair_rate=(
                c_repaired / c_candidate if c_candidate else 0.0
            ),
            regeneration_rate=(
                c_regenerated / c_candidate if c_candidate else 0.0
            ),
            rejection_rate=(
                c_rejected / c_candidate if c_candidate else 0.0
            ),
            tool_selection_rate=(
                (new_n - c_no_selection) / new_n if new_n else 0.0
            ),
        )
        # _dc_replace only overrides the fields passed in, so
        # manually_verified / revoked / revocation_reason / created_at /
        # tested_at / last_verified_at are preserved from the existing
        # record unchanged. CRUCIALLY we do NOT set tested_at or
        # last_verified_at here — live traffic must never refresh the
        # certification clock. last_observed_at tracks only that we saw
        # this tuple, for informational purposes.
    else:
        # Brand-new record from live traffic alone. tested_at /
        # last_verified_at are intentionally left at their defaults ("")
        # — a live-only record was never certified.
        c_candidate = observation["candidate_count"]
        result = CompatibilityResult(
            sample_count=1,
            created_at=now,
            last_observed_at=now,
            manually_verified=False,
            revoked=False,
            candidate_count=c_candidate,
            valid_unchanged_count=observation["valid_unchanged_count"],
            repaired_count=observation["repaired_count"],
            regenerated_count=observation["regenerated_count"],
            accepted_count=observation["accepted_count"],
            rejected_count=observation["rejected_count"],
            no_selection_request_count=(
                1 if observation["no_selection"] else 0
            ),
            valid_call_rate_before_repair=(
                observation["valid_unchanged_count"] / c_candidate
                if c_candidate else 0.0
            ),
            valid_call_rate_after_repair=(
                observation["accepted_count"] / c_candidate
                if c_candidate else 0.0
            ),
            deterministic_repair_rate=(
                observation["repaired_count"] / c_candidate
                if c_candidate else 0.0
            ),
            regeneration_rate=(
                observation["regenerated_count"] / c_candidate
                if c_candidate else 0.0
            ),
            rejection_rate=(
                observation["rejected_count"] / c_candidate
                if c_candidate else 0.0
            ),
            tool_selection_rate=(
                0.0 if observation["no_selection"] else 1.0
            ),
        )

    store.store_result(key, result)


def record_evidence_observation(
    invocation: Any,
    execution: InteropRequestExecution,
    store: Any,
) -> None:
    """Swallow-and-log wrapper: a persistence failure must never break the
    client request."""
    try:
        record_evidence_observation_inner(invocation, execution, store)
    except Exception:
        logger.warning("failed to record compatibility evidence", exc_info=True)
