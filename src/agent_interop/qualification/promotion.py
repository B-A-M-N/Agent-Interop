"""Safe promotion from bounded qualification evidence."""

from __future__ import annotations

from agent_interop.qualification.state import QualificationRecord, QualificationState


def promote(record: QualificationRecord) -> QualificationState:
    if record.continuation and (record.native_forced_tool or record.prompted_forced_tool):
        return QualificationState.SEQUENTIAL_AGENT
    if record.native_forced_tool or record.prompted_forced_tool:
        return QualificationState.FORCED_TOOL
    if record.no_tool_compliant:
        return QualificationState.CHAT_ONLY
    return QualificationState.DEGRADED
