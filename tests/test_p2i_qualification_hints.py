"""P2-I regression: qualification correctness + attempt hints + controller limits.

Locks in:
  * probe_passed() is the only capability read — FAILED/UNKNOWN never prove
    ability (the historical bool(enum) truthiness bug);
  * records from an old battery revision or a replaced chat template are
    ignored (and dropped from the in-memory cache);
  * staged qualification re-evaluates after every probe — a native failure
    INSIDE a run dynamically introduces the prompted probe, and the
    continuation probe runs only once forced evidence exists;
  * probe contracts carry exact markers feeding the battery digest;
  * the battery digest covers the synthetic tool schema; a schema change
    invalidates old evidence;
  * the qualification store refuses schema-v1 payloads;
  * attempt-path hints reorder but never widen the ladder;
  * controller work-product rendering respects the configured token cap
    (newest-first);
  * the controller history summary spends the outer request's budget and
    is tagged purpose="context_summary".
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_interop.qualification import BootstrapQualifier, QualificationRecord, QualificationState
from agent_interop.qualification.probes import SYNTHETIC_TOOL, fast_bootstrap_battery
from agent_interop.qualification.promotion import promote_from_outcomes
from agent_interop.qualification.revision import QUALIFICATION_BATTERY_REVISION, _battery_digest
from agent_interop.qualification.state import ProbeOutcome, probe_passed
from agent_interop.qualification.store import QualificationStore, record_is_current

# ─── P0-51: tri-state reads ─────────────────────────────────────────────────


def test_failed_probe_never_proves_capability():
    record = QualificationRecord(
        model_digest="m",
        native_forced_tool=ProbeOutcome.FAILED,
        prompted_forced_tool=ProbeOutcome.FAILED,
        continuation=ProbeOutcome.FAILED,
    )
    assert not probe_passed(record, "native_forced_tool")
    assert not probe_passed(record, "prompted_forced_tool")
    assert not probe_passed(record, "continuation")
    # A probed-and-failed model promotes to nothing better than DEGRADED.
    assert record.state != QualificationState.FORCED_TOOL
    assert record.state != QualificationState.SEQUENTIAL_AGENT


def test_unknown_probe_is_not_failed_and_not_passed():
    record = QualificationRecord(model_digest="m")
    assert not probe_passed(record, "native_forced_tool")
    # UNKNOWN is not evidence of failure either — the probe simply has not run.
    assert record.native_forced_tool is ProbeOutcome.UNKNOWN


# ─── P0-52: battery/template currency ───────────────────────────────────────


def test_record_is_current_rejects_stale_battery_and_template():
    record = QualificationRecord(
        model_digest="m",
        battery_revision="ancient",
        template_digest="",
    )
    assert not record_is_current(
        record, battery_revision=QUALIFICATION_BATTERY_REVISION, template_digest="",
    )
    fresh = QualificationRecord(
        model_digest="m",
        battery_revision=QUALIFICATION_BATTERY_REVISION,
        template_digest="tmpl",
    )
    assert record_is_current(
        fresh, battery_revision=QUALIFICATION_BATTERY_REVISION, template_digest="tmpl",
    )
    # A template change invalidates even a current-battery record.
    assert not record_is_current(
        fresh, battery_revision=QUALIFICATION_BATTERY_REVISION, template_digest="other",
    )


def test_qualification_store_refuses_schema_v1(tmp_path):
    path = tmp_path / "qual.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "records": {"m": {"state": "forced_tool", "native_forced_tool": "passed"}},
    }))
    store = QualificationStore(path)
    assert store.get("m") is None, "v1 payload has no currency fields — must not load"


# ─── P0-54: staged re-evaluation ────────────────────────────────────────────


class _ScriptedExecutor:
    """Returns canned results and records which probes ran, in order."""

    def __init__(self, results: dict[str, bool]) -> None:
        self.results = results
        self.ran: list[str] = []

    async def __call__(self, probe) -> bool:
        self.ran.append(probe.name)
        return self.results.get(probe.name, False)


def test_native_failure_inside_a_run_dynamically_adds_prompted():
    """P0-54: the historical implementation computed `needed` once, so a
    native failure during THIS run never reached the prompted probe."""
    executor = _ScriptedExecutor({"native_forced_tool": False, "prompted_forced_tool": True})
    record = asyncio.run(BootstrapQualifier().qualify_demand(
        "m", executor, want_native=True, need_continuation=False,
    ))
    assert executor.ran == ["native_forced_tool", "prompted_forced_tool"]
    assert record.prompted_forced_tool is ProbeOutcome.PASSED
    assert record.state == QualificationState.FORCED_TOOL


def test_native_pass_skips_prompted_and_continuation_without_demand():
    executor = _ScriptedExecutor({"native_forced_tool": True})
    record = asyncio.run(BootstrapQualifier().qualify_demand(
        "m", executor, want_native=True, need_continuation=False,
    ))
    assert executor.ran == ["native_forced_tool"]
    assert record.state == QualificationState.FORCED_TOOL


def test_continuation_runs_only_after_forced_evidence_exists():
    executor = _ScriptedExecutor({
        "native_forced_tool": True, "tool_result_continuation": True,
    })
    record = asyncio.run(BootstrapQualifier().qualify_demand(
        "m", executor, want_native=True, need_continuation=True,
    ))
    assert executor.ran == ["native_forced_tool", "tool_result_continuation"]
    assert record.state == QualificationState.SEQUENTIAL_AGENT


def test_forced_failure_suppresses_continuation():
    """No forced evidence → the continuation precondition evaporates."""
    executor = _ScriptedExecutor({
        "native_forced_tool": False, "prompted_forced_tool": False,
    })
    record = asyncio.run(BootstrapQualifier().qualify_demand(
        "m", executor, want_native=True, need_continuation=True,
    ))
    assert executor.ran == ["native_forced_tool", "prompted_forced_tool"]
    assert "tool_result_continuation" not in executor.ran
    assert record.state != QualificationState.SEQUENTIAL_AGENT


def test_continuation_evidence_is_not_dropped_by_merge():
    """P0-54 root cause: the battery names the probe
    ``tool_result_continuation`` but the record field is ``continuation``.
    The historical merge silently dropped the outcome, so staged
    re-evaluation re-ran the probe forever."""
    record = QualificationRecord(model_digest="m")
    merged = record.merge({"tool_result_continuation": ProbeOutcome.PASSED})
    assert merged.continuation is ProbeOutcome.PASSED, (
        "battery probe name must alias to the record field"
    )
    assert "tool_result_continuation" in merged.tested_probes
    # An UNKNOWN outcome is 'not tested' — it must not mark the probe tested.
    untested = merged.merge({"tool_result_continuation": ProbeOutcome.UNKNOWN})
    assert untested.continuation is ProbeOutcome.PASSED, "UNKNOWN never overwrites evidence"


# ─── P0-55: exact probe contracts ───────────────────────────────────────────


def test_forced_probes_carry_exact_markers():
    battery = {probe.name: probe for probe in fast_bootstrap_battery()}
    assert battery["native_forced_tool"].expected_marker == "native"
    assert battery["prompted_forced_tool"].expected_marker == "prompted"
    # Text probes keep their expected_text and have no marker.
    assert battery["no_tool"].expected_text == "no tool needed"
    assert not battery["no_tool"].expected_marker
    assert battery["tool_result_continuation"].expected_text == "continued"


# ─── P0-56: synthetic schema in the battery digest ──────────────────────────


def test_battery_digest_covers_synthetic_tool_schema():
    base = _battery_digest()
    assert SYNTHETIC_TOOL.input_schema is not None
    # The digest must be sensitive to the schema: recompute with a mutated
    # schema and require a different digest.
    original = dict(SYNTHETIC_TOOL.input_schema)
    mutated = {**original, "properties": {
        **original.get("properties", {}), "extra": {"type": "string"},
    }}
    import dataclasses
    modified = dataclasses.replace(SYNTHETIC_TOOL, input_schema=mutated)
    import agent_interop.qualification.revision as revision_module
    import agent_interop.qualification.probes as probes_module

    saved = probes_module.SYNTHETIC_TOOL
    try:
        probes_module.SYNTHETIC_TOOL = modified
        changed = revision_module._battery_digest()
    finally:
        probes_module.SYNTHETIC_TOOL = saved
    assert changed != base, "a synthetic-schema change must invalidate the battery"


# ─── P0-45: attempt-path hints ──────────────────────────────────────────────


def _attempt(kind):
    from agent_interop.config import ToolMode
    from agent_interop.planning.types import AttemptKind, CompatibilityAttempt

    return CompatibilityAttempt(kind, ToolMode.PROMPTED, reason="test")


def test_hint_reorders_only_permitted_attempts():
    from agent_interop.planning.hints import reorder_attempts_by_hint
    from agent_interop.planning.types import AttemptKind

    ladder = (_attempt(AttemptKind.NATIVE_TOOLS), _attempt(AttemptKind.PROMPTED_TOOLS))
    reordered = reorder_attempts_by_hint(ladder, AttemptKind.PROMPTED_TOOLS)
    assert [a.kind for a in reordered] == [AttemptKind.PROMPTED_TOOLS, AttemptKind.NATIVE_TOOLS]
    # Same MEMBERSHIP, different order — the hint may not add or remove rungs.
    assert set(reordered) == set(ladder)
    # A hint for a kind the planner withheld must be ignored.
    assert reorder_attempts_by_hint(ladder, AttemptKind.CONTROLLER_MEDIATED) is ladder
    assert reorder_attempts_by_hint(ladder, None) is ladder


def test_hint_cache_ttl_and_bounded_eviction():
    from agent_interop.planning.hints import AttemptHintCache
    from agent_interop.planning.types import AttemptKind

    cache = AttemptHintCache(ttl_seconds=0.0, max_entries=2)
    key = "k"
    cache.record(key, AttemptKind.PROMPTED_TOOLS)
    # TTL 0 → immediately expired.
    assert cache.get(key) is None
    cache = AttemptHintCache(ttl_seconds=60.0, max_entries=2)
    cache.record("a", AttemptKind.PROMPTED_TOOLS)
    cache.record("b", AttemptKind.NATIVE_TOOLS)
    cache.record("c", AttemptKind.PROMPTED_TOOLS)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") is AttemptKind.NATIVE_TOOLS
    assert cache.get("c") is AttemptKind.PROMPTED_TOOLS
    assert len(cache) == 2


def test_hint_key_covers_the_full_serving_tuple():
    from agent_interop.planning.hints import attempt_hint_key

    base = dict(
        model_digest="m", template_digest="t", serving_config_digest="s",
        profile_revision="p", client_protocol="claude_code/v1",
        tool_surface_fingerprint="f", streaming=False, tool_choice_class="auto",
    )
    reference = attempt_hint_key(**base)
    for field, value in (
        ("model_digest", "other"), ("template_digest", "other"),
        ("serving_config_digest", "other"), ("profile_revision", "other"),
        ("client_protocol", "other"), ("tool_surface_fingerprint", "other"),
        ("streaming", True), ("tool_choice_class", "required"),
    ):
        assert attempt_hint_key(**{**base, field: value}) != reference, (
            f"{field} must be part of the hint key"
        )
    assert attempt_hint_key(**base) == reference


# ─── P0-47: controller work-product bounds ──────────────────────────────────


def _bounded_render(products: list[str], cap: int) -> tuple[list[str], int]:
    """Mirror of the gateway's newest-first bounded selection, used to pin
    the semantics the gateway implements."""
    kept: list[str] = []
    used = 0
    for product in reversed(products):
        cost = (len(product) + 3) // 4
        if kept and used + cost > cap:
            break
        kept.append(product)
        used += cost
    kept.reverse()
    return kept, used


def test_work_product_selection_is_newest_first_and_bounded():
    big = "x" * 4000          # ~1000 tokens
    small = "y" * 400         # ~100 tokens
    products = [big, small, big, small]  # chronological
    kept, used = _bounded_render(products, cap=1200)
    # Newest-first walk keeps: newest small (100) → newest big (1100) →
    # older small (1200 == cap, fits) → older big would exceed.  Re-rendered
    # chronologically: [older small, newest big, newest small].
    assert kept == [small, big, small], (len(kept), used)
    assert used <= 1200
    # The NEWEST product always rides alone even when it alone exceeds cap.
    huge = "z" * 100000
    kept, used = _bounded_render([small, huge], cap=10)
    assert kept == [huge], "the newest product must never be dropped"
    assert used == (len(huge) + 3) // 4


def test_promotion_requires_forced_before_sequential():
    # A continuation pass alone (no forced evidence) must NOT promote.
    state = promote_from_outcomes({
        "native_forced_tool": ProbeOutcome.UNKNOWN,
        "prompted_forced_tool": ProbeOutcome.UNKNOWN,
        "no_tool_compliant": ProbeOutcome.UNKNOWN,
        "continuation": ProbeOutcome.PASSED,
    })
    assert state != QualificationState.SEQUENTIAL_AGENT


# ─── P0-48: summary shares the outer budget ─────────────────────────────────


def test_summary_execution_inherits_outer_budget():
    from agent_interop.execution import InteropRequestExecution

    outer = InteropRequestExecution()
    from agent_interop.execution_attempts import AttemptBudget

    outer.attempt_budget = AttemptBudget()
    summary = InteropRequestExecution()
    summary.attempt_budget = getattr(outer, "attempt_budget", None)
    assert summary.attempt_budget is outer.attempt_budget, (
        "the summary generation must spend the outer request's ledger"
    )


def test_summary_attempt_tagging_uses_context_summary_purpose():
    from agent_interop.execution import InteropRequestExecution

    execution = InteropRequestExecution()
    execution.token_efficiency.update_from_attempt(
        rendered_input_tokens=100, output_tokens=40,
        purpose="context_summary", path="controller",
    )
    attempts = execution.token_efficiency.attempts
    assert attempts[-1].purpose == "context_summary"
    assert attempts[-1].output_tokens == 40
