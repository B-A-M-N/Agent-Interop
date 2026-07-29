"""Proper SSE decoder.

SSE records can span multiple lines (event:, id:, retry:, data:)
and are terminated by a blank line. This decoder accumulates lines
within a record and emits ``SSEFrame`` only at record boundaries.

Enforces max frame size to prevent memory exhaustion from unbounded
upstream responses (item 82).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("agent_interop.transport.sse")

# Default max data per SSE frame: 1 MiB
DEFAULT_MAX_SSE_DATA_BYTES = 1_048_576
# Default max total buffered data before flush: 4 MiB
DEFAULT_MAX_SSE_BUFFER_BYTES = 4_194_304


@dataclass
class SSEFrame:
    """A single SSE frame."""
    event: str | None = None
    data: str = ""
    id: str | None = None
    raw: str = ""


class SSEFrameTooLargeError(Exception):
    """Raised when an SSE frame exceeds the configured byte limit."""


class SSEDecoder:
    """Accumulates SSE lines and emits frames at record boundaries.

    Configurable byte limits prevent unbounded buffering of malformed
    upstream responses. ``max_data_bytes`` caps the data portion of a
    single frame; ``max_buffer_bytes`` caps the total buffered content
    across frames.
    """

    def __init__(
        self,
        max_data_bytes: int = DEFAULT_MAX_SSE_DATA_BYTES,
        max_buffer_bytes: int = DEFAULT_MAX_SSE_BUFFER_BYTES,
    ) -> None:
        self._event: str | None = None
        self._data: list[str] = []
        self._id: str | None = None
        self._raw: list[str] = []
        self._data_bytes: int = 0
        self._buffer_bytes: int = 0
        self._max_data_bytes = max_data_bytes
        self._max_buffer_bytes = max_buffer_bytes

    def feed(self, line: str) -> SSEFrame | None:
        """Feed a line to the decoder.

        Args:
            line: A single line from the stream (including newline).

        Returns:
            An ``SSEFrame`` if a complete record was terminated by a
            blank line, or ``None`` if the record is still accumulating.

        Raises:
            SSEFrameTooLargeError: If adding this line would exceed
                max_buffer_bytes or max_data_bytes.
        """
        line_bytes = len(line.encode("utf-8"))

        # Check total buffer limit
        if self._buffer_bytes + line_bytes > self._max_buffer_bytes:
            raise SSEFrameTooLargeError(
                f"SSE buffer exceeds {self._max_buffer_bytes} bytes "
                f"(current={self._buffer_bytes}, adding={line_bytes})"
            )

        self._buffer_bytes += line_bytes
        self._raw.append(line)
        line = line.rstrip("\r\n")

        if line == "":
            # Blank line — emit the accumulated frame
            return self._emit()

        if line.startswith("event:"):
            self._event = line[len("event:"):].strip()
        elif line.startswith("id:"):
            self._id = line[len("id:"):].strip()
        elif line.startswith("retry:"):
            pass  # retry handled by client
        elif line.startswith("data:"):
            data_part = line[len("data:"):]
            data_part = data_part.removeprefix(" ")
            data_bytes = len(data_part.encode("utf-8"))
            if self._data_bytes + data_bytes > self._max_data_bytes:
                raise SSEFrameTooLargeError(
                    f"SSE data exceeds {self._max_data_bytes} bytes "
                    f"(current={self._data_bytes}, adding={data_bytes})"
                )
            self._data_bytes += data_bytes
            self._data.append(data_part)
        # Else: ignore unknown fields

        return None

    def _emit(self) -> SSEFrame:
        frame = SSEFrame(
            event=self._event,
            data="\n".join(self._data),
            id=self._id,
            raw="".join(self._raw),
        )
        # Reset accumulators
        self._event = None
        self._data = []
        self._id = None
        self._raw = []
        self._data_bytes = 0
        # Note: buffer_bytes is cumulative across frames — reset on full close
        return frame

    def flush(self) -> SSEFrame | None:
        """Emit any accumulated data as a frame.

        Call this at end-of-stream to emit a partial record.
        """
        if self._data or self._event or self._id:
            return self._emit()
        return None

    def reset_buffer_counter(self) -> None:
        """Call after a fully consumed frame to allow new buffering."""
        self._buffer_bytes = sum(len(r.encode("utf-8")) for r in self._raw)
