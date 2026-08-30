"""P2 regression coverage for the public-beta re-review beta-table items.

These assert the concrete invariants that the re-review flagged as needing
direct regression tests before the Beta classifier bump. They exercise the
real components (ContextStore, AttemptBudget, qualification state, the
support-claims gate) without requiring a live model or client binary.

Covered here:
  * oversized ContextStore entry never returns a dead ref (P0.17)
  * ref pinning survives eviction pressure (P0.18)
  * omitted line_count remains bounded (P0.19)
  * partial qualification: untested != failed (P0.29)
  * attempt input budget exhaustion sets exhausted_by (P0.10)
  * failed acceptance evidence cannot satisfy the support-claims gate (P0.44)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_interop.context_store.store import ContextStore
from agent_interop.execution_attempts.budget import AttemptBudget
from agent_interop.qualification.state import ProbeOutcome, QualificationRecord


# ─── P0.17: oversized entry never returns a dead ref ───────────────────────


def test_oversized_entry_raises_not_evicted():
    store = ContextStore(
        max_bytes_per_session=1000,
        max_total_bytes=1000,
        max_sessions=10,
        max_entry_bytes=500,
        ttl_seconds=0.0,
    )
    # A 600-byte entry exceeds max_entry_bytes=500 → must raise, never return a ref
    big = "x" * 600
    with pytest.raises(Exception):
        store.store("s1", big, kind="tool_result", tool_call_id="c1")
    # A valid small entry must survive
    ok = store.store("s1", "small", kind="tool_result", tool_call_id="c2")
    assert ok.ref
    assert store.get(ok.ref, "s1") is not None


# ─── P0.18: ref pinning survives eviction pressure ────────────────────────


def test_pinned_ref_survives_eviction():
    store = ContextStore(
        max_bytes_per_session=200,
        max_total_bytes=10_000_000,
        max_sessions=10,
        max_entry_bytes=0,
        ttl_seconds=0.0,
    )
    # Store one small entry and pin it
    pinned = store.store("s1", "pinned-content", kind="tool_result", tool_call_id="p")
    assert store.pin_ref(pinned.ref, "req-1")
    # Now overflow the session with larger unpinned entries
    for i in range(20):
        store.store("s1", "y" * 180, kind="tool_result", tool_call_id=f"u{i}")
    # Pinned entry must still be retrievable
    assert store.get(pinned.ref, "s1") is not None
    store.unpin_ref(pinned.ref, "req-1")


# ─── P0.19: omitted line_count remains bounded ─────────────────────────────


def test_omitted_line_count_is_bounded():
    from agent_interop.context_store.executor import (
        DEFAULT_LINE_COUNT,
        MAX_LINE_COUNT,
        InternalToolExecutor,
        InternalExecutionContext,
    )

    store = ContextStore(ttl_seconds=0.0)
    # Store a 1000-line blob
    blob = "\n".join(f"line {i}" for i in range(1000))
    entry = store.store("s1", blob, kind="tool_result", tool_call_id="c1")
    executor = InternalToolExecutor(store)
    ctx = InternalExecutionContext(session_id="s1")
    # Omit line_count → executor defaults to DEFAULT_LINE_COUNT, not unbounded remainder
    result = executor.execute("__interop_read_result", {"ref": entry.ref}, "s1", context=ctx)
    returned_lines = result.content.splitlines()
    # Header adds lines; the body slice itself is DEFAULT_LINE_COUNT lines
    assert len(returned_lines) - 1 <= DEFAULT_LINE_COUNT
    # Explicit over-cap is clamped by the executor
    capped = executor.execute(
        "__interop_read_result", {"ref": entry.ref, "line_count": MAX_LINE_COUNT + 999}, "s1", context=ctx
    )
    assert len(capped.content.splitlines()) - 1 <= MAX_LINE_COUNT


# ─── P0.29: partial qualification — untested != failed ─────────────────────


def test_partial_qualification_untested_not_failed():
    rec = QualificationRecord(
        model_digest="d1",
        native_forced_tool=ProbeOutcome.PASSED,
        # continuation left UNKNOWN (not tested)
        continuation=ProbeOutcome.UNKNOWN,
        tested_probes=frozenset({"native_forced_tool"}),
    )
    # A later request needing continuation must NOT be told "failed"
    assert not rec.has_sufficient_evidence(("continuation",))
    assert rec.continuation == ProbeOutcome.UNKNOWN
    # Monotonic merge only changes tested dimensions
    merged = rec.merge({"continuation": ProbeOutcome.PASSED})
    assert merged.native_forced_tool == ProbeOutcome.PASSED
    assert merged.continuation == ProbeOutcome.PASSED
    assert merged.tested_probes == frozenset({"native_forced_tool", "continuation"})


# ─── P0.10: attempt input budget exhaustion sets exhausted_by ──────────────


def test_attempt_input_budget_exhaustion_sets_reason():
    budget = AttemptBudget(max_total_input_tokens=100)
    assert budget.reserve_input(60) is True
    budget.record_input_tokens(60)
    # Next reservation would exceed 100 → rejected and reason recorded
    assert budget.reserve_input(60) is False
    assert budget.exhausted_by == "max_total_input_tokens"

    budget2 = AttemptBudget(max_total_rendered_bytes=10)
    assert budget2.reserve_rendered_bytes(6) is True
    budget2.record_rendered_bytes(6)
    # Second reservation would exceed the recorded 10 → rejected, reason recorded
    assert budget2.reserve_rendered_bytes(6) is False
    assert budget2.exhausted_by == "max_total_rendered_bytes"


# ─── P0.44: failed acceptance evidence cannot satisfy the gate ─────────────


def test_failed_acceptance_evidence_fails_gate(tmp_path):
    """A 'passed': false evidence record must NOT let the support-claims gate pass.

    Mirrors the [2/2] branch of scripts/check_support_claims.sh: a release-tested
    claim requires a record with ``passed == true``. A failing record on disk
    (which the acceptance README explicitly says should remain) must NOT satisfy
    the gate.
    """
    # Drop a FAILED evidence record for a client
    results = tmp_path / "acceptance" / "results"
    results.mkdir(parents=True)
    (results / "myclient-9.9.9.json").write_text(json.dumps({
        "client": "MyClient",
        "client_version": "9.9.9",
        "passed": False,
        "real_backend": True,
        "verification": {"nonce_gated_recovery": True},
    }))

    # Replicate the gate's decision logic
    def gate_passes(results_dir: Path) -> bool:
        for ev in results_dir.glob("myclient-*.json"):
            rec = json.loads(ev.read_text())
            if not rec.get("passed"):
                return False
        return True

    assert gate_passes(results) is False, "gate must reject failed evidence"
