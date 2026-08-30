"""Qualification coordinator contract.

Execution is injected so the gateway can use its normal authenticated
transport and never grants probes filesystem or shell tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent_interop.qualification.probes import BootstrapProbe, fast_bootstrap_battery
from agent_interop.qualification.promotion import promote
from agent_interop.qualification.revision import QUALIFICATION_BATTERY_REVISION
from agent_interop.qualification.state import ProbeOutcome, QualificationRecord

ProbeExecutor = Callable[[BootstrapProbe], Awaitable[bool]]


class BootstrapQualifier:
    async def qualify(
        self,
        model_digest: str,
        execute: ProbeExecutor,
        *,
        template_digest: str = "",
    ) -> QualificationRecord:
        """Run the full bootstrap battery (legacy)."""
        # P0.31: Use ordered battery instead of set
        outcomes: dict[str, bool] = {}
        for probe in fast_bootstrap_battery():
            outcomes[probe.name] = await execute(probe)

        record = QualificationRecord(
            model_digest=model_digest,
            native_forced_tool=_to_outcome(outcomes.get("native_forced_tool", False)),
            prompted_forced_tool=_to_outcome(outcomes.get("prompted_forced_tool", False)),
            no_tool_compliant=_to_outcome(outcomes.get("no_tool", False)),
            continuation=_to_outcome(outcomes.get("tool_result_continuation", False)),
            tested_probes=frozenset(name for name, result in outcomes.items() if result is not None),
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest=template_digest,
        )
        return QualificationRecord(**{**record.__dict__, "state": promote(record)})

    async def qualify_demand(
        self,
        model_digest: str,
        execute: ProbeExecutor,
        *,
        existing: QualificationRecord | None = None,
        want_native: bool = True,
        need_continuation: bool = False,
        template_digest: str = "",
    ) -> QualificationRecord:
        """P0.9/P0.30: Run only the probes demanded by the current request.

        P0.30: Staged — try native forced first, only try prompted if native
        fails.  No unnecessary exact_text probe.

        P0-54: the needed set is RE-EVALUATED after every probe rather than
        computed once up front.  A native failure inside this run must
        dynamically introduce the prompted probe (and the continuation probe
        must wait until forced evidence actually exists); a precomputed list
        can neither add the follow-on probe nor skip one whose precondition
        evaporated.

        Uses an existing record when provided so that already-PASSED probes
        are never re-executed (merge preserves them).
        """
        all_probes = {probe.name: probe for probe in fast_bootstrap_battery()}

        # Start from existing record or a fresh UNKNOWN record.
        record = existing if existing is not None else QualificationRecord(
            model_digest=model_digest,
        )

        # Sequential staged execution: after each probe, merge the outcome and
        # ask the record what is still needed.  Each loop iteration runs at
        # most one probe; the loop ends when staged_needed_probes returns
        # nothing (or a needed probe is not part of the battery).
        while True:
            needed = record.staged_needed_probes(
                want_native=want_native, need_continuation=need_continuation
            )
            if not needed:
                break
            probe_name = needed[0]
            probe = all_probes.get(probe_name)
            if probe is None:
                break
            result = await execute(probe)
            outcome = ProbeOutcome.PASSED if result else ProbeOutcome.FAILED
            merged = record.merge({probe_name: outcome})
            if merged is record:
                # Merge refused the outcome — refuse to spin.
                break
            record = merged

        # Stamp battery revision and template digest.
        record = QualificationRecord(
            model_digest=record.model_digest,
            state=record.state,
            native_forced_tool=record.native_forced_tool,
            prompted_forced_tool=record.prompted_forced_tool,
            no_tool_compliant=record.no_tool_compliant,
            continuation=record.continuation,
            tested_probes=record.tested_probes,
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest=template_digest,
        )

        # Recompute promotion state.
        return QualificationRecord(
            model_digest=record.model_digest,
            state=promote(record),
            native_forced_tool=record.native_forced_tool,
            prompted_forced_tool=record.prompted_forced_tool,
            no_tool_compliant=record.no_tool_compliant,
            continuation=record.continuation,
            tested_probes=record.tested_probes,
            battery_revision=record.battery_revision,
            template_digest=record.template_digest,
        )


def _to_outcome(value: bool | None) -> ProbeOutcome:
    """Convert legacy bool to ProbeOutcome."""
    if value is None:
        return ProbeOutcome.UNKNOWN
    return ProbeOutcome.PASSED if value else ProbeOutcome.FAILED
