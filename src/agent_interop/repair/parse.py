"""JSON syntax recovery for tool-call arguments.

Handles truncated, trailing-comma, and control-char-escaped JSON before
schema validation ever runs. This is Layer 1 of the repair pipeline:
pre-parse recovery, adapted from command-code's ``parseToolArgs``.

The strategy, in order of cost:
  1. Plain json.loads
  2. Escape control chars inside string values, then json.loads
  3. Repair truncated JSON by appending missing closers
  4. Escape + repair (combine both)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

_CONTROL_CHAR_RE = re.compile(r'[\x00-\x1f]')

_TRAILING_COMMA_RE = re.compile(r',\s*([]}])')


@dataclass(frozen=True)
class ParseRecovery:
    """Structured result of JSON syntax recovery.

    Carries the parsed value, the original raw input, the recovered text
    (if repair occurred), the repair steps taken, and a confidence level.
    """

    value: dict[str, Any] | None = None
    original: str = ""
    recovered: str | None = None
    status: Literal["unchanged", "recovered", "rejected"] = "unchanged"
    steps: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"] = "high"


def parse_tool_args(raw: str | dict | None, max_bytes: int = 65536) -> ParseRecovery:
    """Parse a raw tool-arguments string into a dict.

    Returns a ParseRecovery with structured result. Accepts dicts directly
    (no-op). Handles truncated JSON, trailing commas, and unescaped control
    characters. Tracks repair provenance.

    Args:
        raw: The raw arguments string or dict.
        max_bytes: Maximum input size in bytes for syntax recovery.
            Passed through from RepairPolicy.max_input_bytes.
    """
    if raw is None:
        return ParseRecovery(original="", status="rejected")
    if isinstance(raw, dict):
        return ParseRecovery(value=raw, original=str(raw), status="unchanged")
    if isinstance(raw, list):
        return ParseRecovery(original=str(raw), status="rejected")
    if not isinstance(raw, str):
        return ParseRecovery(original=str(raw), status="rejected")

    original = raw
    raw = raw.strip()
    if not raw:
        return ParseRecovery(value={}, original=original, status="unchanged")

    # 1. Plain parse
    result = _try_parse(raw)
    if result is not None:
        return ParseRecovery(value=result, original=original, status="unchanged")

    # 1b. Strip trailing commas (common model artifact: {"a": 1,})
    no_commas = _TRAILING_COMMA_RE.sub(r"\1", raw)
    if no_commas != raw:
        result = _try_parse(no_commas)
        if result is not None:
            return ParseRecovery(
                value=result, original=original, recovered=no_commas,
                status="recovered", steps=("strip_trailing_comma",), confidence="high",
            )

    # 2. Escape control chars then parse
    escaped = _escape_control_chars(no_commas)
    if escaped != no_commas:
        result = _try_parse(escaped)
        if result is not None:
            return ParseRecovery(
                value=result, original=original, recovered=escaped,
                status="recovered", steps=("escape_control_chars",), confidence="high",
            )

    # 3. Repair truncated JSON (with policy-controlled max_bytes)
    repaired = _repair_truncated(no_commas, max_bytes=max_bytes)
    if repaired is not None and repaired != no_commas:
        result = _try_parse(repaired)
        if result is not None:
            return ParseRecovery(
                value=result, original=original, recovered=repaired,
                status="recovered", steps=("repair_truncated",), confidence="medium",
            )

    # 4. Escape + repair
    repaired_escaped = _repair_truncated(escaped, max_bytes=max_bytes)
    if repaired_escaped is not None and repaired_escaped != escaped:
        result = _try_parse(repaired_escaped)
        if result is not None:
            return ParseRecovery(
                value=result, original=original, recovered=repaired_escaped,
                status="recovered", steps=("escape_control_chars", "repair_truncated"),
                confidence="low",
            )

    return ParseRecovery(original=original, status="rejected")


def _try_parse(value: str | None) -> dict[str, Any] | None:
    """Attempt json.loads, returning None on failure."""
    if not isinstance(value, str):
        return None
    try:
        data = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        return data
    return None


def _escape_control_chars(s: str) -> str:
    """Escape unescaped control characters inside JSON string values."""
    out: list[str] = []
    in_string = False
    escaped = False

    for ch in s:
        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue
            if ch == "\\":
                out.append(ch)
                escaped = True
                continue
            if ch == '"':
                out.append(ch)
                in_string = False
                continue
            if _CONTROL_CHAR_RE.match(ch):
                out.append(_escape_one(ch))
                continue
            out.append(ch)
        else:
            if ch == '"':
                in_string = True
            out.append(ch)

    return "".join(out)


def _escape_one(ch: str) -> str:
    """Escape a single control character for JSON."""
    if ch == "\n":
        return "\\n"
    if ch == "\r":
        return "\\r"
    if ch == "\t":
        return "\\t"
    return f"\\u{ord(ch):04x}"


def _repair_truncated(s: str, max_bytes: int = 65536) -> str | None:
    """Attempt to close truncated JSON by counting unmatched delimiters.

    Only counts delimiters outside of strings. Appends the missing closing
    tokens in the right order.

    Returns None if the repair is unsafe:
    - Mismatched closing delimiters (e.g., {"a": [1, 2}})
    - Input exceeds max_bytes
    - Unterminated string (too risky — can change file content/shell meaning)

    Only enables automatic delimiter closure when the repair is uniquely
    determined.
    """
    # Enforce byte limit before scanning
    if len(s.encode("utf-8")) > max_bytes:
        return None

    # Strip a trailing comma (common truncation artifact).
    stripped = s.rstrip()
    if stripped.endswith(","):
        s = stripped[:-1]

    stack: list[str] = []
    in_string = False
    escaped = False
    has_mismatch = False

    for ch in s:
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                # Mismatched delimiter — unsafe to repair
                has_mismatch = True
                break

    # Reject mismatched delimiters: {"a": [1, 2}} is contradictory
    if has_mismatch:
        return None

    # Do NOT auto-close unterminated strings — too risky for file paths/commands
    if in_string:
        return None

    # Close in reverse order of opening.
    suffix = "".join(reversed(stack))
    return s + suffix if suffix else s
