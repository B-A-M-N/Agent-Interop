"""Qualification state model keyed by immutable model digest."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QualificationState(str, Enum):
    UNKNOWN = "unknown"
    PROBING = "probing"
    CHAT_ONLY = "chat_only"
    FORCED_TOOL = "forced_tool"
    AUTOMATIC_TOOL = "automatic_tool"
    SEQUENTIAL_AGENT = "sequential_agent"
    ADVANCED_AGENT = "advanced_agent"
    DEGRADED = "degraded"


@dataclass(frozen=True)
class QualificationRecord:
    model_digest: str
    state: QualificationState = QualificationState.UNKNOWN
    native_forced_tool: bool = False
    prompted_forced_tool: bool = False
    no_tool_compliant: bool = False
    continuation: bool = False
