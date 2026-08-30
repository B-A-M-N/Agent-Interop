"""Qualification state model keyed by immutable model digest."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class QualificationState(str, Enum):
    UNKNOWN = "unknown"
    PROBING = "probing"
    CHAT_ONLY = "chat_only"
    FORCED_TOOL = "forced_tool"
    SEQUENTIAL_AGENT = "sequential_agent"
    ADVANCED_AGENT = "advanced_agent"  # Reserved for real L4 conformance evidence — parallel
    # calls, edit/verify, distinct IDs — which no current probe produces.
    # Promotion never emits it, so L4 selection fails closed.
    DEGRADED = "degraded"


class ProbeOutcome(str, Enum):
    """P0.29: Tri-state evidence for a single probe dimension.

    UNKNOWN = probe not yet run
    PASSED = probe ran and model complied
    FAILED = probe ran and model did not comply
    """
    UNKNOWN = "unknown"
    PASSED = "passed"
    FAILED = "failed"


@dataclass(frozen=True)
class QualificationRecord:
    model_digest: str
    state: QualificationState = QualificationState.UNKNOWN
    # P0.29: Use ProbeOutcome instead of bool to distinguish unknown from failed
    # Backwards-compatible: accepts bool (True=PASSED, False=FAILED/UNKNOWN)
    native_forced_tool: ProbeOutcome = ProbeOutcome.UNKNOWN
    prompted_forced_tool: ProbeOutcome = ProbeOutcome.UNKNOWN
    no_tool_compliant: ProbeOutcome = ProbeOutcome.UNKNOWN
    continuation: ProbeOutcome = ProbeOutcome.UNKNOWN
    # P0.29: Track which probes have been tested and when
    tested_probes: frozenset[str] = field(default_factory=frozenset)
    battery_revision: str = ""
    template_digest: str = ""

    def __post_init__(self) -> None:
        # P0.29: Coerce bool values to ProbeOutcome for backwards compatibility
        object.__setattr__(self, "native_forced_tool", self._coerce(self.native_forced_tool))
        object.__setattr__(self, "prompted_forced_tool", self._coerce(self.prompted_forced_tool))
        object.__setattr__(self, "no_tool_compliant", self._coerce(self.no_tool_compliant))
        object.__setattr__(self, "continuation", self._coerce(self.continuation))

    @staticmethod
    def _coerce(value: object) -> "ProbeOutcome":
        if isinstance(value, ProbeOutcome):
            return value
        if value is True:
            return ProbeOutcome.PASSED
        if value is False:
            return ProbeOutcome.FAILED
        return ProbeOutcome.UNKNOWN

    def merge(self, new_results: dict[str, ProbeOutcome]) -> "QualificationRecord":
        """P0.29: Monotonic merge — only tested dimensions change.

        P0-54: battery probe names are translated to record field names
        (the continuation probe is ``tool_result_continuation`` in the
        battery but ``continuation`` on the record).  Without the alias the
        outcome was silently dropped: the record kept claiming UNKNOWN, so
        staged re-evaluation would re-run the probe forever (and the
        historical compute-once loop simply never noticed the lost
        evidence).
        """
        updates = {}
        tested = set(self.tested_probes)
        for key, value in new_results.items():
            field_name = _PROBE_FIELD_ALIASES.get(key, key)
            if value != ProbeOutcome.UNKNOWN:
                updates[field_name] = value
                tested.add(key)
            else:
                # An UNKNOWN outcome means "not tested" — never add a field
                # alias that has no evidence behind it.
                tested.discard(field_name)
        if not updates:
            return self
        return QualificationRecord(
            model_digest=self.model_digest,
            state=_recompute_state({**self._as_dict(), **updates}),
            native_forced_tool=updates.get("native_forced_tool", self.native_forced_tool),
            prompted_forced_tool=updates.get("prompted_forced_tool", self.prompted_forced_tool),
            no_tool_compliant=updates.get("no_tool_compliant", self.no_tool_compliant),
            continuation=updates.get("continuation", self.continuation),
            tested_probes=frozenset(tested),
            battery_revision=self.battery_revision,
            template_digest=self.template_digest,
        )

    def _as_dict(self) -> dict[str, ProbeOutcome]:
        return {
            "native_forced_tool": self.native_forced_tool,
            "prompted_forced_tool": self.prompted_forced_tool,
            "no_tool_compliant": self.no_tool_compliant,
            "continuation": self.continuation,
        }

    def is_tested(self, probe_name: str) -> bool:
        """P0.29: Check if a probe has been tested (not just failed)."""
        return probe_name in self.tested_probes

    def has_sufficient_evidence(self, required_probes: tuple[str, ...]) -> bool:
        """P0.29: Check if all required probes have been tested."""
        return all(self.is_tested(p) for p in required_probes)

    def staged_needed_probes(
        self, *, want_native: bool, need_continuation: bool = False
    ) -> tuple[str, ...]:
        """Compute which probes must still be executed under staged qualification.

        Staging rationale: when the preferred (native) path already passed, do NOT
        re-run the prompted path — that would cause infinite re-probe loops for
        demand-driven qualification.  The prompted probe is only exercised after a
        native failure, because staged qualification assumes the preferred path
        ends the process once it succeeds.

        Returns probe names in battery execution order
        ("native_forced_tool", "prompted_forced_tool", "tool_result_continuation").
        """
        needed: list[str] = []

        # Stage 1 — native forced tool
        if want_native:
            if self.native_forced_tool == ProbeOutcome.PASSED:
                # Native passed: preferred path complete, no more probes needed.
                pass
            elif self.native_forced_tool == ProbeOutcome.UNKNOWN:
                needed.append("native_forced_tool")
            elif self.native_forced_tool == ProbeOutcome.FAILED:
                # Native failed: fall through to prompted path.
                pass

            # Stage 2 — prompted forced tool (only after native failure)
            if not want_native or self.native_forced_tool == ProbeOutcome.FAILED:
                if self.prompted_forced_tool == ProbeOutcome.UNKNOWN:
                    needed.append("prompted_forced_tool")

        else:
            # Direct-to-prompted path (no native).
            if self.prompted_forced_tool == ProbeOutcome.UNKNOWN:
                needed.append("prompted_forced_tool")

        # Continuation — only needed if we have or will have forced evidence.
        if need_continuation and self.continuation == ProbeOutcome.UNKNOWN:
            # Check whether forced evidence already exists or will be produced.
            has_forced = (
                self.native_forced_tool == ProbeOutcome.PASSED
                or self.prompted_forced_tool == ProbeOutcome.PASSED
            )
            will_have_forced = (
                want_native and self.native_forced_tool == ProbeOutcome.UNKNOWN
            )
            if has_forced or will_have_forced:
                needed.append("tool_result_continuation")

        return tuple(needed)


def _recompute_state(values: dict[str, ProbeOutcome]) -> QualificationState:
    """Recompute state from probe outcomes."""
    from agent_interop.qualification.promotion import promote_from_outcomes
    return promote_from_outcomes(values)


# P0-54: battery probe name → record field name.  Only the continuation
# probe differs; the forced probes already match.
_PROBE_FIELD_ALIASES: dict[str, str] = {
    "tool_result_continuation": "continuation",
}


def probe_passed(record: QualificationRecord, probe_name: str) -> bool:
    """Check if a probe passed."""
    outcome = getattr(record, probe_name, ProbeOutcome.UNKNOWN)
    return outcome == ProbeOutcome.PASSED


def probe_failed(record: QualificationRecord, probe_name: str) -> bool:
    """Check if a probe failed (not just untested)."""
    outcome = getattr(record, probe_name, ProbeOutcome.UNKNOWN)
    return outcome == ProbeOutcome.FAILED
