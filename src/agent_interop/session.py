"""Bounded session state — per-session tracking with TTL, LRU eviction, and loop detection.

Uses Claude Code's session ID header when available. Derives or generates
a gateway session identity for other clients.

State tracked:
- Recent repairs (by tool name, for loop detection)
- Loop state
- Route ID
- Last seen timestamp

Bounds:
- TTL per session
- Maximum total sessions
- Maximum repairs per session
- LRU eviction
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class RepairMemory:
    """Memory of a single repair event."""

    tool_name: str
    issue_count: int
    status: str  # repaired / rejected / regenerated
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass(frozen=True)
class ProgressSignature:
    """A signature for detecting repetitive non-progressing loops.

    Tracks tool name, argument digest, result class, and repair status.
    Repeated identical signatures with equivalent results indicate a loop.
    """

    tool_name: str
    argument_digest: str
    result_class: str  # "success" | "error" | "repair" | "rejection"
    repair_status: str = ""  # "" | "repaired" | "regenerated"


class LoopState:
    """Tracks loop detection for a session using progress-aware signatures.

    A loop is repeated identical or near-identical input with equivalent
    results and no progress — not merely repeated use of the same tool.
    """

    def __init__(self, window_size: int = 10) -> None:
        self._window_size = window_size
        self._signatures: list[ProgressSignature] = []
        self._consecutive_repairs: int = 0
        self._flagged: bool = False

    @property
    def flagged(self) -> bool:
        return self._flagged

    def record_tool(
        self,
        tool_name: str,
        argument_digest: str = "",
        result_class: str = "success",
        was_repaired: bool = False,
        is_correction: bool = False,
    ) -> None:
        """Record a tool call with its progress signature and check for loops.

        is_correction: True for regeneration/correction passes — these don't
        count as independent agent loop iterations.
        """
        if is_correction:
            # Correction passes don't advance the loop window
            return

        repair_status = "repaired" if was_repaired else ""
        sig = ProgressSignature(
            tool_name=tool_name,
            argument_digest=argument_digest,
            result_class=result_class,
            repair_status=repair_status,
        )

        if was_repaired:
            self._consecutive_repairs += 1
        else:
            self._consecutive_repairs = 0

        self._signatures.append(sig)

        # Keep only the recent window
        if len(self._signatures) > self._window_size:
            self._signatures = self._signatures[-self._window_size:]

        # Detect loop: repeated identical signature with equivalent results
        self._detect_loop()

    def _detect_loop(self) -> None:
        """Check for repetitive non-progressing patterns."""
        # Flag if >3 consecutive repairs
        if self._consecutive_repairs > 3:
            self._flagged = True
            return

        # Flag if same tool+args pattern repeats >3 times with same result class
        if len(self._signatures) < 3:
            return

        # Count recent identical signatures
        recent = self._signatures[-5:]
        if len(recent) < 3:
            return

        # Check for repeating pattern
        from collections import Counter

        sig_counts = Counter(
            (s.tool_name, s.argument_digest, s.result_class)
            for s in recent
        )
        most_common_count = sig_counts.most_common(1)[0][1] if sig_counts else 0

        # If the same (tool, args, result) appears >=3 times in window, it's a loop
        if most_common_count >= 3:
            self._flagged = True

    def reset(self) -> None:
        self._signatures.clear()
        self._consecutive_repairs = 0
        self._flagged = False


@dataclass
class SessionState:
    """Per-session state for repair tracking and loop detection."""

    session_id: str
    route_id: str = ""
    last_seen: float = 0.0
    recent_repairs: dict[str, RepairMemory] = field(default_factory=dict)
    loop_state: LoopState = field(default_factory=LoopState)
    request_count: int = 0
    repair_count: int = 0

    def __post_init__(self) -> None:
        if self.last_seen == 0.0:
            self.last_seen = time.time()

    def touch(self) -> None:
        self.last_seen = time.time()
        self.request_count += 1

    def record_repair(
        self,
        tool_name: str,
        issue_count: int,
        status: str,
        argument_digest: str = "",
    ) -> None:
        self.repair_count += 1
        self.recent_repairs[tool_name] = RepairMemory(
            tool_name=tool_name,
            issue_count=issue_count,
            status=status,
        )
        # Map the repair status to the ProgressSignature result_class values
        # declared in ProgressSignature ("success" | "error" | "repair" |
        # "rejection"). Previously this was hardcoded to "success", which
        # conflated rejections with successes in the signature window.
        result_class = {
            "repaired": "repair",
            "regenerated": "repair",
            "rejected": "rejection",
        }.get(status, "success")
        self.loop_state.record_tool(
            tool_name,
            argument_digest=argument_digest,
            result_class=result_class,
            was_repaired=status in ("repaired", "regenerated"),
        )

    @property
    def flagged(self) -> bool:
        return self.loop_state.flagged


# ─── Session Manager ──────────────────────────────────────────────────────────


DEFAULT_TTL_SECONDS = 3600  # 1 hour
DEFAULT_MAX_SESSIONS = 100
DEFAULT_MAX_REPAIRS_PER_SESSION = 50


class SessionManager:
    """Manages bounded session state with LRU eviction and TTL expiration.

    Sessions are keyed by ``(session_id, route_id)``, not ``session_id``
    alone — the same client session ID talking to two different routes must
    not share (and silently corrupt) loop-detection state between them.

    ``begin_request`` is the single touch-once entry point for the start of
    a request: it is the only method that increments ``request_count``.
    ``get`` and ``record_repair`` never touch — call them as many times as
    needed within one request without inflating the counter.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_repairs_per_session: int = DEFAULT_MAX_REPAIRS_PER_SESSION,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_sessions = max_sessions
        self._max_repairs = max_repairs_per_session
        self._sessions: dict[tuple[str, str], SessionState] = {}
        self._last_access: dict[tuple[str, str], float] = {}

    def begin_request(self, session_id: str, route_id: str = "") -> SessionState:
        """Mark the start of one request against this session — creates the
        session if needed and touches it EXACTLY once (request_count += 1).

        Call this once per request, at the point the session is first
        resolved. Everything else that looks up the same session within
        that request should use ``get()`` instead, or request_count ends up
        counting "how many times something looked at this session" rather
        than "how many requests this session made".
        """
        self._evict_expired()

        key = (session_id, route_id)
        if key not in self._sessions:
            if len(self._sessions) >= self._max_sessions:
                self._evict_lru()
            self._sessions[key] = SessionState(
                session_id=session_id,
                route_id=route_id,
            )

        session = self._sessions[key]
        session.touch()
        self._last_access[key] = time.time()
        return session

    def get(self, session_id: str, route_id: str = "") -> SessionState | None:
        """Look up existing session state without touching it (no
        request_count increment, no creation)."""
        return self._sessions.get((session_id, route_id))

    def record_repair(
        self,
        session_id: str,
        route_id: str,
        tool_name: str,
        issue_count: int,
        status: str,
        argument_digest: str = "",
    ) -> None:
        session = self.get(session_id, route_id)
        if session and session.repair_count < self._max_repairs:
            session.record_repair(tool_name, issue_count, status, argument_digest=argument_digest)

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            key for key, state in self._sessions.items()
            if now - state.last_seen > self._ttl
        ]
        for key in expired:
            del self._sessions[key]
            self._last_access.pop(key, None)

    def _evict_lru(self) -> None:
        if not self._last_access:
            return
        # Find least recently accessed
        oldest = min(self._last_access, key=lambda k: self._last_access[k])  # type: ignore[arg-type]
        self._sessions.pop(oldest, None)
        self._last_access.pop(oldest, None)

    def clear(self) -> None:
        self._sessions.clear()
        self._last_access.clear()

    @property
    def active_count(self) -> int:
        self._evict_expired()
        return len(self._sessions)

    def get_flag_status(self) -> list[str]:
        """Return list of session IDs that have been flagged for looping."""
        return [key[0] for key, state in self._sessions.items() if state.flagged]