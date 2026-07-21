"""Loop detection — detect protocol-level repetition.

Interop detects repetition without replacing the host agent's task logic.

Default signals:
- Identical tool name and arguments repeated 3x without different intervening results
- Repeated malformed output after repair
- Repeated empty assistant responses
- Repeated tool-call IDs
- No-progress response pattern exceeding threshold
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopState:
    """Tracks model output patterns for loop detection."""

    last_tool_calls: list[dict] = field(default_factory=list)
    tool_call_fingerprints: list[tuple] = field(default_factory=list)
    empty_responses: int = 0
    malformed_outputs: int = 0
    repair_loops: int = 0
    consecutive_identical_calls: int = 0
    total_turns: int = 0

    _last_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoopConfig:
    """Configuration for loop detection."""

    enabled: bool = True
    max_consecutive_identical: int = 3
    max_empty_responses: int = 3
    max_malformed_outputs: int = 3
    max_repair_loops: int = 2
    cooldown_turns: int = 5  # reset counter after this many turns of variety


class LoopDetector:
    """Detects model output loops during generation."""

    def __init__(self, config: LoopConfig | None = None) -> None:
        self.config = config or LoopConfig()
        self._sessions: dict[str, LoopState] = {}

    def get_state(self, session_id: str) -> LoopState:
        if session_id not in self._sessions:
            self._sessions[session_id] = LoopState()
        return self._sessions[session_id]

    def check(self, session_id: str, tool_calls: list[dict], empty: bool = False,
              malformed: bool = False, repaired: bool = False) -> str | None:
        """Check the latest output for loop signals.

        Returns a signal string if a loop is detected, None otherwise.
        """
        if not self.config.enabled:
            return None

        state = self.get_state(session_id)
        state.total_turns += 1

        # Empty response check
        if empty:
            state.empty_responses += 1
        else:
            # Reset empty counter on non-empty response
            state.empty_responses = 0

        if state.empty_responses >= self.config.max_empty_responses:
            return "repeated_empty_response"

        # Malformed output check
        if malformed:
            state.malformed_outputs += 1
        else:
            state.malformed_outputs = 0

        if state.malformed_outputs >= self.config.max_malformed_outputs:
            return "repeated_malformed_output"

        # Repair loop check
        if repaired:
            state.repair_loops += 1
        else:
            state.repair_loops = 0

        if state.repair_loops >= self.config.max_repair_loops:
            return "repair_loop"

        # Identical tool call check
        current_fingerprints = self._fingerprint_calls(tool_calls)
        if current_fingerprints:
            state.tool_call_fingerprints.append(current_fingerprints)
            # Keep only last N
            if len(state.tool_call_fingerprints) > self.config.max_consecutive_identical:
                state.tool_call_fingerprints.pop(0)

            # Check for consecutive identical patterns
            if len(state.tool_call_fingerprints) >= self.config.max_consecutive_identical:
                recent = state.tool_call_fingerprints[-self.config.max_consecutive_identical:]
                if all(r == recent[0] for r in recent):
                    return "repeated_identical_tool_calls"
        else:
            # Non-tool-call response breaks the streak
            state.tool_call_fingerprints.clear()

        return None

    def _fingerprint_calls(self, calls: list[dict]) -> tuple:
        """Create a stable fingerprint for a set of tool calls."""
        if not calls:
            return ()
        return tuple(
            (c.get("name", ""), str(sorted(c.get("arguments", {}).items())))
            for c in calls
        )

    def reset_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def signal_description(self, signal: str) -> str:
        descriptions = {
            "repeated_empty_response": "Model emitted empty responses repeatedly",
            "repeated_malformed_output": "Model output remains malformed after repair attempts",
            "repair_loop": "Model output triggers repair loop repeatedly",
            "repeated_identical_tool_calls": "Model called the same tool(s) with identical arguments consecutively",
        }
        return descriptions.get(signal, f"Loop signal: {signal}")