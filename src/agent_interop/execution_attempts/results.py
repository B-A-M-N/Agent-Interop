"""Results recorded for every compatibility attempt."""

from __future__ import annotations

from dataclasses import dataclass

from agent_interop.planning.types import CompatibilityAttempt


@dataclass(frozen=True)
class AttemptResult:
    attempt: CompatibilityAttempt
    accepted: bool
    reason: str = ""
