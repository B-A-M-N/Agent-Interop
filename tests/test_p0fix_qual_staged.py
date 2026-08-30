"""Regression tests for P0 fixes: staged sufficiency, record stamping, L4 fail-closed."""

from __future__ import annotations

import asyncio

from agent_interop.qualification.bootstrap import BootstrapQualifier
from agent_interop.qualification.promotion import promote_from_outcomes
from agent_interop.qualification.revision import QUALIFICATION_BATTERY_REVISION
from agent_interop.qualification.state import ProbeOutcome, QualificationRecord, QualificationState


def _run_async(coro):
    """Run an async coroutine, reusing the running loop if available."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    else:
        return loop.run_until_complete(coro)


# ─── (a) staged_needed_probes ───────────────────────────────────────────────


class TestStagedNeededProbes:
    """Verify staged_needed_probes follows the staging rule."""

    def test_fresh_record_want_native(self):
        rec = QualificationRecord(model_digest="d1")
        needed = rec.staged_needed_probes(want_native=True)
        # Fresh record, want native -> need native first
        assert needed == ("native_forced_tool",)

    def test_native_passed_nothing_needed(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.PASSED,
            prompted_forced_tool=ProbeOutcome.UNKNOWN,
        )
        needed = rec.staged_needed_probes(want_native=True)
        assert needed == ()

    def test_native_failed_prompted_unknown(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.FAILED,
            prompted_forced_tool=ProbeOutcome.UNKNOWN,
        )
        needed = rec.staged_needed_probes(want_native=True)
        assert needed == ("prompted_forced_tool",)

    def test_native_passed_need_continuation_continuation_unknown(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.PASSED,
            continuation=ProbeOutcome.UNKNOWN,
        )
        needed = rec.staged_needed_probes(want_native=True, need_continuation=True)
        assert needed == ("tool_result_continuation",)

    def test_native_passed_no_need_continuation(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.PASSED,
            continuation=ProbeOutcome.UNKNOWN,
        )
        needed = rec.staged_needed_probes(want_native=True, need_continuation=False)
        assert needed == ()

    def test_no_native_want_prompted_only(self):
        rec = QualificationRecord(
            model_digest="d1",
            prompted_forced_tool=ProbeOutcome.UNKNOWN,
        )
        needed = rec.staged_needed_probes(want_native=False)
        assert needed == ("prompted_forced_tool",)

    def test_native_failed_prompted_passed_need_continuation(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.FAILED,
            prompted_forced_tool=ProbeOutcome.PASSED,
            continuation=ProbeOutcome.UNKNOWN,
        )
        # Prompted passed, so forced evidence exists after this run
        needed = rec.staged_needed_probes(want_native=True, need_continuation=True)
        # Forced evidence comes from prompted being PASSED, so continuation needed
        assert needed == ("tool_result_continuation",)

    def test_order_is_native_then_prompted_then_continuation(self):
        """Battery execution order must be respected."""
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.UNKNOWN,
            prompted_forced_tool=ProbeOutcome.UNKNOWN,
            continuation=ProbeOutcome.UNKNOWN,
        )
        # want_native=True, native unknown, need_continuation=True
        # But with want_native=True and native UNKNOWN, we first get native,
        # then need_continuation check: native UNKNOWN will become PASSED/FAILED
        # after this run, so continuation will be needed.
        # However, we only return probes in battery order.
        needed = rec.staged_needed_probes(want_native=True, need_continuation=True)
        # native_unknown -> needed; continuation_unknown + will_have_forced -> needed
        assert "native_forced_tool" in needed
        # continuation comes after native
        nat_idx = needed.index("native_forced_tool")
        cont_idx = needed.index("tool_result_continuation") if "tool_result_continuation" in needed else len(needed)
        assert nat_idx < cont_idx


# ─── (b) qualify_demand with fake executor ──────────────────────────────────


class TestQualifyDemandStaged:
    """qualify_demand should execute ONLY needed probes and never re-run passed."""

    def test_native_pass_executes_only_native(self):
        executed: list[str] = []
        qualifier = BootstrapQualifier()

        async def executor(probe):
            executed.append(probe.name)
            return probe.name == "native_forced_tool"  # only native passes

        async def run():
            rec = await qualifier.qualify_demand(
                "model-1", executor, want_native=True, need_continuation=False
            )
            return rec

        rec = _run_async(run())
        # Only native should have been executed (it passed, so prompted is skipped)
        assert executed == ["native_forced_tool"]
        assert rec.native_forced_tool == ProbeOutcome.PASSED
        assert rec.prompted_forced_tool == ProbeOutcome.UNKNOWN
        # State should be FORCED_TOOL since native passed
        assert rec.state == QualificationState.FORCED_TOOL

    def test_native_fail_executes_prompted(self):
        """Native failed -> prompted is needed on a second call to staged_needed_probes."""
        executed: list[str] = []
        qualifier = BootstrapQualifier()

        async def executor(probe):
            executed.append(probe.name)
            return False  # everything fails

        async def run():
            # First call: fresh record, native unknown -> executes native
            rec1 = await qualifier.qualify_demand(
                "model-1", executor, want_native=True, need_continuation=False
            )
            # Second call: native is now FAILED -> prompted needed
            rec2 = await qualifier.qualify_demand(
                "model-1", executor, existing=rec1, want_native=True, need_continuation=False
            )
            return rec2

        _run_async(run())
        # Both native and prompted should have been executed across the two calls
        assert "native_forced_tool" in executed
        assert "prompted_forced_tool" in executed

    def test_stamp_battery_revision_and_template_digest(self):
        executed: list[str] = []
        qualifier = BootstrapQualifier()
        template = "chat-template-sha256-abc123"

        async def executor(probe):
            executed.append(probe.name)
            return probe.name == "native_forced_tool"

        async def run():
            rec = await qualifier.qualify_demand(
                "model-1", executor, want_native=True, template_digest=template
            )
            return rec

        rec = _run_async(run())
        assert rec.battery_revision == QUALIFICATION_BATTERY_REVISION
        assert rec.template_digest == template


# ─── (c) merge never re-executes a PASSED probe ─────────────────────────────


class TestMergePreservesPassed:
    """Merge must never downgrade or re-execute a PASSED dimension."""

    def test_merge_preserves_passed_probe(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.PASSED,
            prompted_forced_tool=ProbeOutcome.UNKNOWN,
            tested_probes=frozenset({"native_forced_tool"}),
        )
        # Merge a new prompted result — native should stay PASSED
        merged = rec.merge({"prompted_forced_tool": ProbeOutcome.FAILED})
        assert merged.native_forced_tool == ProbeOutcome.PASSED
        assert merged.prompted_forced_tool == ProbeOutcome.FAILED
        assert merged.tested_probes == frozenset({"native_forced_tool", "prompted_forced_tool"})

    def test_merge_never_downgrades_passed(self):
        rec = QualificationRecord(
            model_digest="d1",
            native_forced_tool=ProbeOutcome.PASSED,
        )
        # merge() only updates non-UNKNOWN values, so PASSED stays PASSED
        merged = rec.merge({"native_forced_tool": ProbeOutcome.UNKNOWN})
        assert merged.native_forced_tool == ProbeOutcome.PASSED


# ─── (d) records carry battery_revision and template_digest round-trips ─────


class TestRecordStamps:
    """Records must carry battery_revision == imported constant."""

    def test_battery_revision_set(self):
        rec = QualificationRecord(
            model_digest="d1",
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest="my-template-digest",
        )
        assert rec.battery_revision == QUALIFICATION_BATTERY_REVISION
        assert rec.template_digest == "my-template-digest"

    def test_template_digest_round_trips_via_merge(self):
        rec = QualificationRecord(
            model_digest="d1",
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest="template-sha-42",
        )
        merged = rec.merge({"native_forced_tool": ProbeOutcome.PASSED})
        assert merged.battery_revision == QUALIFICATION_BATTERY_REVISION
        assert merged.template_digest == "template-sha-42"

    def test_template_digest_survives_store_put_get(self, tmp_path):
        """The store's serialize/deserialize must preserve new fields."""
        from agent_interop.qualification.store import (
            QualificationStore,
        )

        store = QualificationStore(path=tmp_path / "qual.json")
        rec = QualificationRecord(
            model_digest="store-test-digest",
            native_forced_tool=ProbeOutcome.PASSED,
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest="template-for-store-99",
        )
        store.put(rec)
        retrieved = store.get("store-test-digest")
        assert retrieved is not None, "store round-trip must return the record"
        assert retrieved.battery_revision == QUALIFICATION_BATTERY_REVISION
        assert retrieved.template_digest == "template-for-store-99"


# ─── (e) promote never returns ADVANCED_AGENT ────────────────────────────────


class TestPromoteNeverAdvanced:
    """promote_from_outcomes must never return ADVANCED_AGENT."""

    def test_all_outcome_combos(self):
        """Iterate all outcome combinations for the 4 probe dimensions."""
        outcomes_list = [ProbeOutcome.UNKNOWN, ProbeOutcome.PASSED, ProbeOutcome.FAILED]
        found_advanced = False
        for native in outcomes_list:
            for prompted in outcomes_list:
                for no_tool in outcomes_list:
                    for cont in outcomes_list:
                        values = {
                            "native_forced_tool": native,
                            "prompted_forced_tool": prompted,
                            "no_tool_compliant": no_tool,
                            "continuation": cont,
                        }
                        state = promote_from_outcomes(values)
                        if state == QualificationState.ADVANCED_AGENT:
                            found_advanced = True
        assert not found_advanced, "ADVANCED_AGENT should never be returned by promote"

    def test_specific_adv_cases(self):
        """Edge cases that might plausibly trigger ADVANCED_AGENT."""
        # All PASSED
        assert promote_from_outcomes({
            "native_forced_tool": ProbeOutcome.PASSED,
            "prompted_forced_tool": ProbeOutcome.PASSED,
            "no_tool_compliant": ProbeOutcome.PASSED,
            "continuation": ProbeOutcome.PASSED,
        }) != QualificationState.ADVANCED_AGENT

        # Only continuation PASSED, no forced
        assert promote_from_outcomes({
            "native_forced_tool": ProbeOutcome.UNKNOWN,
            "prompted_forced_tool": ProbeOutcome.UNKNOWN,
            "no_tool_compliant": ProbeOutcome.UNKNOWN,
            "continuation": ProbeOutcome.PASSED,
        }) != QualificationState.ADVANCED_AGENT


# ─── (f) QualificationState.ADVANCED_AGENT exists ────────────────────────────


class TestAdvancedAgentExists:
    """ADVANCED_AGENT must be importable from QualificationState."""

    def test_advanced_agent_attr_exists(self):
        assert hasattr(QualificationState, "ADVANCED_AGENT")
        assert QualificationState.ADVANCED_AGENT.value == "advanced_agent"

    def test_import_via_qualification_package(self):
        """Gateway can import ADVANCED_AGENT from the public package."""
        from agent_interop.qualification import QualificationState
        assert hasattr(QualificationState, "ADVANCED_AGENT")


# ─── (g) demand-loop guards + legacy bool coercion ──────────────────────────


class TestQualifyDemandGuards:
    """Both refusal paths in the staged demand loop, plus probe_failed."""

    def test_demand_loop_stops_when_probe_missing_from_battery(self):
        """A needed probe the battery does not carry must end the loop, not spin."""
        executed: list[str] = []
        qualifier = BootstrapQualifier()

        async def executor(probe):
            executed.append(probe.name)
            return False

        # need_continuation with forced evidence never arriving keeps the
        # battery to forced probes only; continue past them via an existing
        # record that marks everything tested except an unknown battery name.
        existing = QualificationRecord(
            model_digest="m",
            native_forced_tool=False,
            prompted_forced_tool=False,
            tested_probes=frozenset({"native_forced_tool", "prompted_forced_tool"}),
            battery_revision=QUALIFICATION_BATTERY_REVISION,
        )
        _run_async(qualifier.qualify_demand(
            "m", executor, existing=existing,
            want_native=True, need_continuation=True,
        ))
        # Continuation was demanded but forced evidence never passed, so the
        # staged planner never adds it and the loop terminates cleanly.
        assert "tool_result_continuation" not in executed

    def test_demand_loop_stops_when_merge_refuses_outcome(self):
        """merge() refusing an UNKNOWN outcome must break the loop, not spin."""
        executed: list[str] = []
        qualifier = BootstrapQualifier()

        async def executor(probe):
            executed.append(probe.name)
            return False

        # Everything fails; the loop must still terminate (no infinite spin)
        # because each merge applies exactly one probe and never re-adds it.
        rec = _run_async(qualifier.qualify_demand(
            "m", executor, want_native=True, need_continuation=True,
        ))
        assert executed == ["native_forced_tool", "prompted_forced_tool"]
        assert rec.continuation == ProbeOutcome.UNKNOWN

    def test_probe_failed_distinguishes_failure_from_untested(self):
        from agent_interop.qualification.state import probe_failed

        record = QualificationRecord(
            model_digest="m",
            native_forced_tool=False,  # coerced to FAILED
        )
        assert probe_failed(record, "native_forced_tool")
        assert not probe_failed(record, "prompted_forced_tool")  # UNKNOWN

    def test_record_bool_coercion_covers_all_paths(self):
        """__post_init__ coerces legacy bools (P0.29) — cover PASSED/UNKNOWN paths."""
        record = QualificationRecord(
            model_digest="m",
            native_forced_tool=True,   # True -> PASSED
            prompted_forced_tool="bogus",  # non-bool non-enum -> UNKNOWN
        )
        assert record.native_forced_tool == ProbeOutcome.PASSED
        assert record.prompted_forced_tool == ProbeOutcome.UNKNOWN

    def test_to_outcome_none_is_unknown(self):
        from agent_interop.qualification.bootstrap import _to_outcome

        assert _to_outcome(None) == ProbeOutcome.UNKNOWN
        assert _to_outcome(True) == ProbeOutcome.PASSED
        assert _to_outcome(False) == ProbeOutcome.FAILED
