"""NDJSON decoder for Ollama native streaming.

Ollama's ``/api/chat`` endpoint uses newline-delimited JSON (NDJSON)
rather than standard SSE. Each line is a complete JSON object.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any

logger = logging.getLogger("agent_interop.transport.ndjson")


class MalformedNDJSONLine:
    """A single NDJSON line that could not be parsed.

    Returned alongside parsed frames so callers can act on parse
    failures rather than silently dropping them.
    """

    __slots__ = ("line", "ordinal", "parse_error")

    def __init__(self, line: str, ordinal: int, parse_error: str) -> None:
        self.line = line
        self.ordinal = ordinal
        self.parse_error = parse_error


class NDJSONDecoder:
    """Decodes newline-delimited JSON streams.

    Returns a heterogeneous iterator of parsed JSON dicts and
    ``MalformedNDJSONLine`` markers so callers can surface parse
    failures to the streaming coordinator.
    """

    def __init__(self, max_frame_bytes: int = 1_048_576) -> None:
        self._buffer = ""
        self._ordinal = 0
        self._max_frame_bytes = max_frame_bytes

    def feed(self, chunk: str) -> Iterator[Any]:
        """Feed a chunk of NDJSON text.

        Yields a mix of dict (successfully parsed frames) and
        ``MalformedNDJSONLine`` (parse failures or oversized lines).
        """
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            self._ordinal += 1
            if len(line) > self._max_frame_bytes:
                logger.warning(
                    "ndjson line #%d exceeds max_frame_bytes=%d (len=%d)",
                    self._ordinal, self._max_frame_bytes, len(line),
                )
                yield MalformedNDJSONLine(
                    line=line[: self._max_frame_bytes],
                    ordinal=self._ordinal,
                    parse_error=f"line exceeds max_frame_bytes={self._max_frame_bytes}",
                )
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "ndjson parse error #%d: %s", self._ordinal, exc,
                )
                yield MalformedNDJSONLine(
                    line=line[: self._max_frame_bytes],
                    ordinal=self._ordinal,
                    parse_error=str(exc),
                )

    def flush(self) -> list[Any]:
        """Parse any remaining buffered data.

        Returns a list of parsed JSON objects or ``MalformedNDJSONLine``
        markers for unparseable trailing data.
        """
        out: list[Any] = []
        tail = self._buffer.strip()
        self._buffer = ""
        if not tail:
            return out
        self._ordinal += 1
        if len(tail) > self._max_frame_bytes:
            out.append(MalformedNDJSONLine(
                line=tail[: self._max_frame_bytes],
                ordinal=self._ordinal,
                parse_error=f"line exceeds max_frame_bytes={self._max_frame_bytes}",
            ))
            return out
        try:
            out.append(json.loads(tail))
        except json.JSONDecodeError as exc:
            logger.warning("ndjson flush parse error: %s", exc)
            out.append(MalformedNDJSONLine(
                line=tail[: self._max_frame_bytes],
                ordinal=self._ordinal,
                parse_error=str(exc),
            ))
        return out