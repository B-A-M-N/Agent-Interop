"""Stream state machine — explicit FSM replacing ad hoc streaming.

States:
    IDLE → MESSAGE_STARTED → BLOCK_STARTED
        → TEXT_STREAMING | REASONING_STREAMING | TOOL_CALL_STREAMING
        → TOOL_CALL_VALIDATING
        → BLOCK_FINISHED
        → MESSAGE_FINISHED

Invariants:
- A content delta cannot precede its block-start event.
- A block cannot stop twice.
- A message cannot stop while a block remains open.
- Tool arguments may be buffered until valid.
- Partial tool JSON must never be exposed as ordinary assistant text.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from interop.types import CanonicalEvent, ContentBlock


class StreamState(Enum):
    IDLE = auto()
    MESSAGE_STARTED = auto()
    BLOCK_STARTED = auto()
    TEXT_STREAMING = auto()
    REASONING_STREAMING = auto()
    TOOL_CALL_STREAMING = auto()
    TOOL_CALL_VALIDATING = auto()
    BLOCK_FINISHED = auto()
    MESSAGE_FINISHED = auto()
    ERROR = auto()


# ─── Valid transitions ──────────────────────────────────────────────────────

_VALID_TRANSITIONS: dict[StreamState, set[StreamState]] = {
    StreamState.IDLE: {StreamState.MESSAGE_STARTED, StreamState.ERROR},
    StreamState.MESSAGE_STARTED: {
        StreamState.BLOCK_STARTED, StreamState.MESSAGE_FINISHED, StreamState.ERROR,
    },
    StreamState.BLOCK_STARTED: {
        StreamState.TEXT_STREAMING, StreamState.REASONING_STREAMING,
        StreamState.TOOL_CALL_STREAMING, StreamState.MESSAGE_FINISHED, StreamState.ERROR,
    },
    StreamState.TEXT_STREAMING: {
        StreamState.TEXT_STREAMING, StreamState.BLOCK_FINISHED, StreamState.ERROR,
    },
    StreamState.REASONING_STREAMING: {
        StreamState.REASONING_STREAMING, StreamState.BLOCK_FINISHED, StreamState.ERROR,
    },
    StreamState.TOOL_CALL_STREAMING: {
        StreamState.TOOL_CALL_STREAMING, StreamState.TOOL_CALL_VALIDATING,
        StreamState.BLOCK_FINISHED, StreamState.ERROR,
    },
    StreamState.TOOL_CALL_VALIDATING: {
        StreamState.BLOCK_FINISHED, StreamState.ERROR,
    },
    StreamState.BLOCK_FINISHED: {
        StreamState.BLOCK_STARTED, StreamState.MESSAGE_FINISHED, StreamState.ERROR,
    },
    StreamState.MESSAGE_FINISHED: {StreamState.MESSAGE_STARTED, StreamState.ERROR},
    StreamState.ERROR: set(),
}


# ─── Stream FSM ──────────────────────────────────────────────────────────────


class StreamFSM:
    """Explicit state machine for streaming responses."""

    def __init__(self) -> None:
        self._state = StreamState.IDLE
        self._block_index = 0
        self._open_blocks = 0
        self._tool_call_buffer: list[str] = []
        self._current_block_type: str = ""

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def block_index(self) -> int:
        return self._block_index

    @property
    def open_blocks(self) -> int:
        return self._open_blocks

    def transition(self, new_state: StreamState) -> None:
        """Attempt to transition to new_state."""
        allowed = _VALID_TRANSITIONS.get(self._state, set())
        if new_state not in allowed:
            raise StreamProtocolError(
                f"Invalid stream transition: {self._state.name} → {new_state.name}"
            )
        self._state = new_state

    def start_message(self) -> None:
        self.transition(StreamState.MESSAGE_STARTED)

    def start_block(self, block_type: str) -> None:
        self.transition(StreamState.BLOCK_STARTED)
        self._current_block_type = block_type
        self._open_blocks += 1

    def stream_text(self) -> None:
        self.transition(StreamState.TEXT_STREAMING)

    def stream_reasoning(self) -> None:
        self.transition(StreamState.REASONING_STREAMING)

    def stream_tool_call(self) -> None:
        self.transition(StreamState.TOOL_CALL_STREAMING)

    def buffer_tool_json(self, chunk: str) -> None:
        self._tool_call_buffer.append(chunk)

    def flush_tool_buffer(self) -> str:
        result = "".join(self._tool_call_buffer)
        self._tool_call_buffer.clear()
        return result

    def validate_tool_call(self) -> None:
        self.transition(StreamState.TOOL_CALL_VALIDATING)

    def finish_block(self) -> None:
        self.transition(StreamState.BLOCK_FINISHED)
        self._open_blocks -= 1
        self._block_index += 1
        self._current_block_type = ""

    def finish_message(self) -> None:
        if self._open_blocks > 0:
            raise StreamProtocolError(
                f"Cannot finish message with {self._open_blocks} open block(s)"
            )
        self.transition(StreamState.MESSAGE_FINISHED)

    def set_error(self) -> None:
        self._state = StreamState.ERROR

    def reset(self) -> None:
        self._state = StreamState.IDLE
        self._block_index = 0
        self._open_blocks = 0
        self._tool_call_buffer.clear()
        self._current_block_type = ""

    def events_from_text(self, text: str) -> list[CanonicalEvent]:
        """Convert a text chunk into the appropriate stream events."""
        events: list[CanonicalEvent] = []
        if self._state == StreamState.MESSAGE_STARTED:
            self.start_block("text")
            events.append(CanonicalEvent(type="text", index=self._block_index - 1, partial=text))
            self.stream_text()
        elif self._state in (StreamState.TEXT_STREAMING, StreamState.BLOCK_STARTED):
            if self._state == StreamState.BLOCK_STARTED:
                events.append(CanonicalEvent(type="text", index=self._block_index, partial=""))
                self.stream_text()
            events.append(CanonicalEvent(type="text_delta", index=self._block_index, partial=text))
        return events

    def events_from_tool_call(self, tool_call_block: ContentBlock) -> list[CanonicalEvent]:
        """Convert a complete tool call into stream events."""
        events: list[CanonicalEvent] = []
        if self._state in (StreamState.MESSAGE_STARTED, StreamState.BLOCK_FINISHED):
            self.start_block("tool_use")
            events.append(CanonicalEvent(
                type="tool_use", index=self._block_index - 1,
                content_block=tool_call_block,
            ))
            self.finish_block()
        return events

    def end_message_events(self) -> list[CanonicalEvent]:
        """Emit events for clean message end."""
        events: list[CanonicalEvent] = []
        try:
            self.finish_message()
            events.append(CanonicalEvent(type="message_stop"))
        except StreamProtocolError:
            pass
        return events


class StreamProtocolError(Exception):
    """Raised on invalid stream state transitions."""
    pass