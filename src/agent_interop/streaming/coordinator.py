"""
Stream coordinator — manages text streaming, error handling, and clean
termination.

The coordinator tracks stream state but does NOT perform tool-call
accumulation or repair — that happens in gateway.py's streaming path
once complete tool-call candidates are available.

Tool-call fragments are buffered per-call by PendingToolCallAccumulator
until the complete candidate is available for validation and repair.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_interop.abi import CanonicalError, CanonicalEvent


class ToolCallLimitExceeded(Exception):
    """Raised when a tool-call stream exceeds its configured size limits."""


@dataclass
class StreamLimits:
    """Hard limits applied to streaming tool-call accumulation.

    These guard against pathologically large tool-call arguments or
    unbounded numbers of parallel tool calls that would otherwise
    permit a runaway model output to consume unbounded memory.
    """

    max_frame_bytes: int = 1_048_576  # 1 MiB per SSE/NDJSON frame
    max_accumulated_arg_bytes: int = 4_194_304  # 4 MiB per tool call
    max_simultaneous_tool_calls: int = 64


@dataclass
class MalformedFrame:
    """A stream frame that could not be parsed as JSON.

    Preserves the raw frame content, event name, stream position,
    parse error, and the tool-call accumulator state at the time of
    failure.  Downstream code uses ``tool_call_accumulator_open`` to
    decide whether the partial candidates should be failed and the
    stream terminated.
    """

    raw_frame: str = ""
    stream_ordinal: int = 0
    event_name: str = ""
    parse_error: str = ""
    tool_call_accumulator_open: bool = False
    framing: str = ""  # "sse" | "ndjson"


@dataclass(frozen=True, order=True)
class ToolStreamKey:
    """Key for accumulating streaming tool-call fragments.

    Identity is (choice_index, tool_index) to correctly handle parallel
    and fragmented calls across choices.
    """

    choice_index: int = 0
    tool_index: int = 0


@dataclass
class PendingToolCall:
    """Accumulates fragments for a single streaming tool call.

    Fragments are buffered until the call is complete, at which point
    they can be validated and repaired as a unit.
    """

    key: ToolStreamKey
    call_id: str | None
    name_fragments: list[str] = field(default_factory=list)
    argument_fragments: list[str] = field(default_factory=list)
    started: bool = False
    completed: bool = False

    @property
    def choice_index(self) -> int:
        return self.key.choice_index

    @property
    def tool_index(self) -> int:
        return self.key.tool_index

    @property
    def ordinal(self) -> int:
        return self.key.tool_index

    @property
    def assembled_name(self) -> str:
        return "".join(self.name_fragments)

    @property
    def assembled_arguments(self) -> str:
        return "".join(self.argument_fragments)


class PendingToolCallAccumulator:
    """Buffers streaming tool-call fragments for later validation.

    Text can stream through immediately. Tool-call fragments are
    accumulated until the call is complete. Keyed by (choice_index, tool_index).
    """

    def __init__(self, limits: StreamLimits | None = None) -> None:
        self._pending: dict[ToolStreamKey, PendingToolCall] = {}
        self._completed: list[PendingToolCall] = []
        self.limits = limits or StreamLimits()

    def start_call(self, key: ToolStreamKey, call_id: str | None) -> PendingToolCall:
        if key in self._pending:
            call = self._pending[key]
            if call_id and not call.call_id:
                call.call_id = call_id
            return call
        if len(self._pending) >= self.limits.max_simultaneous_tool_calls:
            raise ToolCallLimitExceeded(
                f"Exceeded max_simultaneous_tool_calls={self.limits.max_simultaneous_tool_calls}"
            )
        call = PendingToolCall(key=key, call_id=call_id, started=True)
        self._pending[key] = call
        return call

    def feed_name(self, key: ToolStreamKey, fragment: str) -> None:
        call = self._pending.get(key)
        if call is not None:
            call.name_fragments.append(fragment)
        else:
            # Auto-create if not started yet
            self.start_call(key, None).name_fragments.append(fragment)

    def feed_arguments(self, key: ToolStreamKey, fragment: str) -> None:
        call = self._pending.get(key)
        if call is None:
            call = self.start_call(key, None)
        new_total = len(call.assembled_arguments.encode("utf-8")) + len((fragment or "").encode("utf-8"))
        if new_total > self.limits.max_accumulated_arg_bytes:
            raise ToolCallLimitExceeded(
                f"Tool call {key!s} exceeds max_accumulated_arg_bytes={self.limits.max_accumulated_arg_bytes}"
            )
        call.argument_fragments.append(fragment)

    def complete_call(self, key: ToolStreamKey) -> PendingToolCall | None:
        call = self._pending.pop(key, None)
        if call is not None:
            call.completed = True
            self._completed.append(call)
        return call

    def complete_choice(self, choice_index: int) -> list[PendingToolCall]:
        """Complete all pending calls for a choice, in tool-index order."""
        keys_to_complete = sorted(
            key for key in self._pending
            if key.choice_index == choice_index
        )
        completed = []
        for key in keys_to_complete:
            call = self.complete_call(key)
            if call is not None:
                completed.append(call)
        return completed

    @property
    def has_pending(self) -> bool:
        return len(self._pending) > 0

    @property
    def completed_calls(self) -> list[PendingToolCall]:
        return list(self._completed)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def complete_all_pending(self) -> list[PendingToolCall]:
        """Complete all pending tool calls and move them to the completed list."""
        keys = sorted(self._pending.keys())
        completed = []
        for key in keys:
            call = self.complete_call(key)
            if call is not None:
                completed.append(call)
        return completed

    def fail_all_pending(self, reason: str = "") -> list[PendingToolCall]:
        """Fail all pending tool calls due to a stream error (e.g. malformed frame).

        Moves them to the completed list with ``completed=False`` so that
        callers can inspect the partial fragments but must not treat them
        as valid candidates.
        """
        failed = []
        for key in sorted(self._pending.keys()):
            call = self._pending.pop(key)
            call.completed = False
            self._completed.append(call)
            failed.append(call)
        return failed

    def drain_completed(self) -> list[PendingToolCall]:
        """Return all completed calls and clear the completed list.

        Used by the streaming path in gateway.py to consume completed
        tool calls for validation without re-emitting previously drained calls.
        """
        calls = list(self._completed)
        self._completed.clear()
        return calls

    def reject_pending(self) -> list[ToolStreamKey]:
        """Reject (clear) all pending incomplete calls and return their keys.

        Logs a warning via the caller — this method simply clears state.
        """
        calls: list[ToolStreamKey] = list(self._pending)
        self._pending.clear()
        return calls

    def reset(self) -> None:
        self._pending.clear()
        self._completed.clear()


class StreamCoordinator:
    """Coordinates stream events, accumulation, and termination.

    Tracks stream lifecycle and delegates tool-call fragment
    accumulation to PendingToolCallAccumulator.
    """

    def __init__(self, protocol, limits: StreamLimits | None = None) -> None:
        self.protocol = protocol
        self._finished = False
        self._error: str | None = None
        # Propagate config-derived limits into the accumulator so per-tool-call
        # argument size and parallel-call caps actually take effect. A None
        # default preserves the existing behavior for callers that don't pass
        # limits (PendingToolCallAccumulator falls back to StreamLimits()).
        self.tool_accumulator = PendingToolCallAccumulator(limits=limits)
        self._has_emitted_tool_calls = False
        self._turn_rejected = False

    def mark_tool_calls_emitted(self) -> None:
        """Record that tool calls were emitted during this stream."""
        self._has_emitted_tool_calls = True

    @property
    def has_emitted_tool_calls(self) -> bool:
        """Whether tool calls were emitted during this stream."""
        return self._has_emitted_tool_calls

    def mark_turn_rejected(self) -> None:
        """Record that the whole tool batch was rejected for this stream.

        The caller's generic end-of-turn handling (stop-reason computation,
        terminal message_stop, evidence write-back, finalize_response) is then
        skipped because the shared rejection helper already emitted the
        terminal error + message_stop(INVALID_OUTPUT) and finalized the record
        as failed.
        """
        self._turn_rejected = True

    @property
    def turn_rejected(self) -> bool:
        """Whether the whole tool batch was rejected for this stream."""
        return self._turn_rejected

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def has_error(self) -> bool:
        return self._error is not None

    def set_error(self, error: str) -> list[CanonicalEvent]:
        self._error = error
        return [CanonicalEvent(type="error", error=CanonicalError(message=error))]

    @property
    def has_pending_tool_calls(self) -> bool:
        return self.tool_accumulator.has_pending

    @property
    def completed_tool_calls(self) -> list[PendingToolCall]:
        return self.tool_accumulator.completed_calls
