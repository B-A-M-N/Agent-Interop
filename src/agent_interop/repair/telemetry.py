"""Repair telemetry — observable events from the tool-call repair pipeline.

Events are lightweight dataclasses emitted through a simple emitter.
In production they feed metrics/observability; in development they
enable the ``interop repair stats`` command.

No sensitive data is logged: no source code content, file contents,
shell commands, credentials, or complete tool arguments — unless an
explicit debug-sensitive mode is enabled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RepairEvent:
    """Base telemetry event for the repair pipeline."""

    event_type: str
    request_id: str = ""
    session_id_hash: str = ""  # hashed, never raw session ID
    route_id: str = ""
    model: str = ""
    backend: str = ""
    tool_name: str = ""
    latency_ms: float = 0.0
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RequestStartedEvent(RepairEvent):
    event_type: str = "request_started"


@dataclass
class RouteSelectedEvent(RepairEvent):
    event_type: str = "route_selected"


@dataclass
class ProfileResolvedEvent(RepairEvent):
    event_type: str = "profile_resolved"
    profile_id: str = ""
    tool_mode: str = ""


@dataclass
class ToolCandidateDetectedEvent(RepairEvent):
    event_type: str = "tool_candidate_detected"
    parser: str = ""
    raw_name: str = ""


@dataclass
class ToolInputValidEvent(RepairEvent):
    event_type: str = "tool_input_valid"


@dataclass
class ToolInputRepairedEvent(RepairEvent):
    event_type: str = "tool_input_repaired"
    repair_rules: list[str] = field(default_factory=list)
    issue_paths: list[str] = field(default_factory=list)
    attempt_count: int = 1


@dataclass
class ToolInputRejectedEvent(RepairEvent):
    event_type: str = "tool_input_rejected"
    issue_paths: list[str] = field(default_factory=list)
    attempt_count: int = 1


@dataclass
class RegenerationStartedEvent(RepairEvent):
    event_type: str = "regeneration_started"


@dataclass
class RegenerationSucceededEvent(RepairEvent):
    event_type: str = "regeneration_succeeded"
    attempt_count: int = 1


@dataclass
class RegenerationFailedEvent(RepairEvent):
    event_type: str = "regeneration_failed"
    attempt_count: int = 1


@dataclass
class ResponseCompletedEvent(RepairEvent):
    event_type: str = "response_completed"
    tool_call_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    regenerated_count: int = 0


# ─── Event Emitter ────────────────────────────────────────────────────────────


class RepairTelemetry:
    """Lightweight event emitter for repair pipeline observability.

    Collects events in memory. Can be extended to push to external
    observability systems.
    """

    def __init__(self, debug_sensitive: bool = False) -> None:
        self._debug_sensitive = debug_sensitive
        self._events: list[RepairEvent] = []
        self._summary: dict[str, int] = {}

    def emit(self, event: RepairEvent) -> None:
        """Emit a telemetry event."""
        self._events.append(event)
        self._summary[event.event_type] = self._summary.get(event.event_type, 0) + 1

    def emit_request_started(
        self,
        request_id: str,
        session_id: str,
        route_id: str = "",
        model: str = "",
    ) -> RequestStartedEvent:
        event = RequestStartedEvent(
            request_id=request_id,
            session_id_hash=self._hash(session_id),
            route_id=route_id,
            model=model,
        )
        self.emit(event)
        return event

    def emit_route_selected(
        self,
        request_id: str,
        route_id: str,
        model: str = "",
        backend: str = "",
    ) -> RouteSelectedEvent:
        event = RouteSelectedEvent(
            request_id=request_id,
            route_id=route_id,
            model=model,
            backend=backend,
        )
        self.emit(event)
        return event

    def emit_tool_candidate(
        self,
        request_id: str,
        tool_name: str,
        parser: str = "",
        raw_name: str = "",
    ) -> ToolCandidateDetectedEvent:
        event = ToolCandidateDetectedEvent(
            request_id=request_id,
            tool_name=tool_name,
            parser=parser,
            raw_name=raw_name,
        )
        self.emit(event)
        return event

    def emit_repaired(
        self,
        request_id: str,
        tool_name: str,
        rules: list[str],
        paths: list[str],
        attempt: int = 1,
    ) -> ToolInputRepairedEvent:
        event = ToolInputRepairedEvent(
            request_id=request_id,
            tool_name=tool_name,
            repair_rules=rules,
            issue_paths=paths,
            attempt_count=attempt,
        )
        self.emit(event)
        return event

    def emit_rejected(
        self,
        request_id: str,
        tool_name: str,
        paths: list[str],
        attempt: int = 1,
    ) -> ToolInputRejectedEvent:
        event = ToolInputRejectedEvent(
            request_id=request_id,
            tool_name=tool_name,
            issue_paths=paths,
            attempt_count=attempt,
        )
        self.emit(event)
        return event

    def emit_response_completed(
        self,
        request_id: str,
        tool_call_count: int = 0,
        accepted_count: int = 0,
        rejected_count: int = 0,
        regenerated_count: int = 0,
    ) -> ResponseCompletedEvent:
        event = ResponseCompletedEvent(
            request_id=request_id,
            tool_call_count=tool_call_count,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            regenerated_count=regenerated_count,
        )
        self.emit(event)
        return event

    def get_events(self, event_type: str | None = None) -> list[RepairEvent]:
        if event_type:
            return [e for e in self._events if e.event_type == event_type]
        return list(self._events)

    @property
    def summary(self) -> dict[str, int]:
        return dict(self._summary)

    def clear(self) -> None:
        self._events.clear()
        self._summary.clear()

    def _hash(self, value: str) -> str:
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:16]