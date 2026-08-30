"""Safe promotion from bounded qualification evidence."""

from __future__ import annotations

from agent_interop.qualification.state import ProbeOutcome, QualificationState


def promote(record: object) -> QualificationState:
    """Legacy promote for backwards compatibility."""
    values = {
        "native_forced_tool": getattr(record, "native_forced_tool", ProbeOutcome.UNKNOWN),
        "prompted_forced_tool": getattr(record, "prompted_forced_tool", ProbeOutcome.UNKNOWN),
        "no_tool_compliant": getattr(record, "no_tool_compliant", ProbeOutcome.UNKNOWN),
        "continuation": getattr(record, "continuation", ProbeOutcome.UNKNOWN),
    }
    return promote_from_outcomes(values)


def promote_from_outcomes(values: dict[str, ProbeOutcome]) -> QualificationState:
    """P0.29: Promote based on probe outcomes, treating UNKNOWN as not-yet-tested.

    ADVANCED_AGENT exists in the state enum as a reserved slot for real L4
    conformance evidence (parallel calls, edit/verify, distinct IDs).  No
    current probe produces it and no code path returns it here, so L4
    selection fails closed — the gateway never promotes to it.
    """
    native = values.get("native_forced_tool", ProbeOutcome.UNKNOWN)
    prompted = values.get("prompted_forced_tool", ProbeOutcome.UNKNOWN)
    no_tool = values.get("no_tool_compliant", ProbeOutcome.UNKNOWN)
    cont = values.get("continuation", ProbeOutcome.UNKNOWN)

    # Only consider PASSED probes
    forced = native == ProbeOutcome.PASSED or prompted == ProbeOutcome.PASSED
    sequential = cont == ProbeOutcome.PASSED and forced

    if sequential:
        return QualificationState.SEQUENTIAL_AGENT
    if forced:
        return QualificationState.FORCED_TOOL
    if no_tool == ProbeOutcome.PASSED:
        return QualificationState.CHAT_ONLY
    return QualificationState.DEGRADED
