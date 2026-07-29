"""String-aware balanced JSON scanner.

Replaces the regex-based ``_GENERIC_JSON_RE`` from profiles.py, which
cannot safely parse nested JSON — nested arrays, nested objects, braces
inside strings, escaped quotes, multiple adjacent calls, or truncated output.

This scanner tracks nesting depth, string state, and escape state to emit
balanced ``{…}`` and ``[ … ]`` candidates from arbitrary text. It then
classifies parsed objects as tool-call candidates based on key heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# ─── Scanner types ────────────────────────────────────────────────────────────


@dataclass
class BalancedSpan:
    """A balanced JSON span extracted from text."""

    start: int
    end: int
    text: str
    kind: str = "object"  # "object" or "array"

    def parse(self) -> dict[str, Any] | list[Any] | None:
        try:
            return json.loads(self.text)
        except (json.JSONDecodeError, ValueError):
            return None


@dataclass(frozen=True)
class JsonFieldSpan:
    """Exact span of a JSON field value, preserving raw text verbatim.

    Used to extract the raw ``arguments`` value from a wrapper object
    without requiring the value itself to be valid JSON.
    """

    key: str
    value_start: int
    value_end: int
    raw_value: str


@dataclass
class ToolCallCandidate:
    """A classified tool-call candidate from scanned JSON."""

    span: BalancedSpan
    name: str | None = None
    raw_arguments: dict[str, Any] | list[Any] | None = None
    raw_arguments_text: str | None = None
    confidence: float = 0.0

    @property
    def text(self) -> str:
        return self.span.text


# ─── Known keys that suggest an object is a tool call ─────────────────────────

_TOOL_NAME_KEYS = {"name", "tool", "function"}
_TOOL_ARGS_KEYS = {"arguments", "input", "parameters", "args"}


def _classify_as_tool_call(
    obj: dict[str, Any],
    span: BalancedSpan,
) -> ToolCallCandidate | None:
    """Heuristically classify a parsed JSON object as a tool-call candidate.

    Returns a ToolCallCandidate if the object looks like a tool call,
    or None if it doesn't.
    """
    name: str | None = None
    raw_args: dict[str, Any] | list[Any] | None = None
    raw_args_text: str | None = None
    confidence = 0.0

    # Try to extract name from known keys
    for key in _TOOL_NAME_KEYS:
        val = obj.get(key)
        if isinstance(val, str):
            name = val
            confidence += 0.3
            break

    # Try to extract arguments from known keys
    for key in _TOOL_ARGS_KEYS:
        val = obj.get(key)
        if isinstance(val, (dict, list)):
            raw_args = val
            raw_args_text = _try_extract_raw_args_text(span, preferred_key=key)
            confidence += 0.3
            break

    # Additional confidence signals
    if name and raw_args:
        confidence += 0.2
    if len(obj) <= 3:
        confidence += 0.1
    if name and not raw_args and len(obj) == 1:
        # {"name": "foo"} — might be truncated
        confidence -= 0.2

    if confidence >= 0.3:
        if raw_args_text is None:
            raw_args_text = _try_extract_raw_args_text(span)
        return ToolCallCandidate(
            span=span,
            name=name,
            raw_arguments=raw_args,
            raw_arguments_text=raw_args_text,
            confidence=min(confidence, 1.0),
        )
    return None


def _try_extract_raw_args_text(
    span: BalancedSpan,
    preferred_key: str | None = None,
) -> str | None:
    """Try to extract raw arguments text from a span using field-span scanner."""
    scanner = BalancedJsonScanner()
    fields = scanner.extract_field_spans(span.text)
    args_keys: list[str] = list(_TOOL_ARGS_KEYS)
    if preferred_key and preferred_key in args_keys:
        args_keys = [preferred_key] + [k for k in args_keys if k != preferred_key]
    for field in fields:
        if field.key in args_keys:
            return field.raw_value
    return None


# ─── Balanced scanner ─────────────────────────────────────────────────────────


class BalancedJsonScanner:
    """Scans text for balanced JSON objects/arrays with string awareness.

    Handles:
    - Nested objects and arrays
    - Strings containing braces, brackets, escapes
    - Multiple adjacent calls
    - Truncated output (partial JSON at end of text)
    """

    def __init__(self) -> None:
        self._spans: list[BalancedSpan] = []

    def scan(self, text: str) -> list[BalancedSpan]:
        """Scan text for all balanced JSON structures.

        Returns spans in order of discovery (which is typically left-to-right
        in the text, but nested spans inside outer spans are yielded first).
        """
        self._spans = []
        self._scan_range(text, 0, len(text))
        return list(self._spans)

    def _scan_range(self, text: str, start: int, end: int) -> None:
        """Recursively scan a range of text for balanced JSON."""
        i = start
        while i < end:
            ch = text[i]
            if ch == "{":
                i = self._try_scan_object(text, i, end)
            elif ch == "[":
                i = self._try_scan_array(text, i, end)
            elif ch == '"':
                i = self._skip_string(text, i, end)
            else:
                i += 1

    def _try_scan_object(self, text: str, start: int, end: int) -> int:
        """Try to scan a balanced object starting at start (which is '{').

        Returns the position after the closing '}' (or end if unbalanced).
        """
        depth = 1
        i = start + 1
        in_string = False
        escaped = False

        while i < end:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        span = BalancedSpan(
                            start=start,
                            end=i + 1,
                            text=text[start : i + 1],
                            kind="object",
                        )
                        self._spans.append(span)
                        return i + 1
            i += 1

        # Unbalanced — record as truncated span
        if depth > 0:
            truncated_text = text[start:end]
            span = BalancedSpan(
                start=start,
                end=end,
                text=truncated_text,
                kind="object",
            )
            self._spans.append(span)
        return end

    def _try_scan_array(self, text: str, start: int, end: int) -> int:
        """Try to scan a balanced array starting at start (which is '[')."""
        depth = 1
        i = start + 1
        in_string = False
        escaped = False

        while i < end:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        span = BalancedSpan(
                            start=start,
                            end=i + 1,
                            text=text[start : i + 1],
                            kind="array",
                        )
                        self._spans.append(span)
                        return i + 1
            i += 1

        # Unbalanced
        if depth > 0:
            truncated_text = text[start:end]
            span = BalancedSpan(
                start=start,
                end=end,
                text=truncated_text,
                kind="array",
            )
            self._spans.append(span)
        return end

    def _skip_string(self, text: str, start: int, end: int) -> int:
        """Skip past a string literal starting at start (which is '"').

        Returns the position after the closing '"'.
        """
        i = start + 1
        escaped = False
        while i < end:
            ch = text[i]
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                return i + 1
            i += 1
        return end

    def extract_field_spans(self, text: str) -> list[JsonFieldSpan]:
        """Extract exact value spans for each key in a JSON object.

        Scans the text linearly, tracking string-awareness, to identify
        the raw value substring associated with each top-level key.
        Does NOT require the overall text to be valid JSON.
        """
        fields: list[JsonFieldSpan] = []
        i = 0
        end = len(text)

        # Expect opening '{'
        i = self._skip_ws(text, i, end)
        if i >= end or text[i] != "{":
            return fields
        i += 1

        in_string = False
        escaped = False
        key = ""
        expecting_value = False

        while i < end:
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"' and not expecting_value:
                    # Start of a key
                    key_start = i + 1
                    key_end = key_start
                    ki = i + 1
                    while ki < end:
                        kch = text[ki]
                        if kch == "\\":
                            ki += 2
                            continue
                        if kch == '"':
                            key_end = ki
                            ki += 1
                            break
                        ki += 1
                    key = text[key_start:key_end]
                    i = ki
                    # skip colon
                    i = self._skip_ws(text, i, end)
                    if i < end and text[i] == ":":
                        expecting_value = True
                        i += 1
                        i = self._skip_ws(text, i, end)
                    continue
                elif ch == '"' and expecting_value:
                    # String value — scan to closing quote
                    val_start = i
                    vi = i + 1
                    esc = False
                    while vi < end:
                        vch = text[vi]
                        if esc:
                            esc = False
                        elif vch == "\\":
                            esc = True
                        elif vch == '"':
                            vi += 1
                            fields.append(JsonFieldSpan(
                                key=key,
                                value_start=val_start,
                                value_end=vi,
                                raw_value=text[val_start:vi],
                            ))
                            break
                        vi += 1
                    i = vi
                    expecting_value = False
                    continue
                elif expecting_value and ch in ("{", "["):
                    # Object or array value — use depth tracking
                    val_start = i
                    depth = 1
                    vi = i + 1
                    pair = {"{": "}", "[": "]"}[ch]
                    esc = False
                    instr = False
                    while vi < end and depth > 0:
                        vch = text[vi]
                        if instr:
                            if esc:
                                esc = False
                            elif vch == "\\":
                                esc = True
                            elif vch == '"':
                                instr = False
                        else:
                            if vch == '"':
                                instr = True
                            elif vch in ("{", "["):
                                depth += 1
                            elif vch == pair:
                                depth -= 1
                        vi += 1
                    fields.append(JsonFieldSpan(
                        key=key,
                        value_start=val_start,
                        value_end=vi,
                        raw_value=text[val_start:vi],
                    ))
                    i = vi
                    expecting_value = False
                    continue
                elif expecting_value:
                    # Number, boolean, null literal — scan until delimiter
                    val_start = i
                    vi = i
                    while vi < end and text[vi] not in (",", "}", "]", "\n", " "):
                        vi += 1
                    if vi > val_start:
                        fields.append(JsonFieldSpan(
                            key=key,
                            value_start=val_start,
                            value_end=vi,
                            raw_value=text[val_start:vi],
                        ))
                    i = vi
                    expecting_value = False
                    continue
                elif ch in ("}", "]"):
                    # End of object — done
                    break
                elif ch == ",":
                    expecting_value = False
                    key = ""
            i += 1

        return fields

    @staticmethod
    def _skip_ws(text: str, i: int, end: int) -> int:
        while i < end and text[i] in (" ", "\t", "\n", "\r"):
            i += 1
        return i

    @staticmethod
    def extract_tool_calls(text: str) -> list[ToolCallCandidate]:
        """Convenience: scan text for JSON, parse, and classify as tool calls."""
        scanner = BalancedJsonScanner()
        spans = scanner.scan(text)
        candidates: list[ToolCallCandidate] = []

        for span in spans:
            if span.kind != "object":
                continue
            parsed = span.parse()
            if not isinstance(parsed, dict):
                continue
            candidate = _classify_as_tool_call(parsed, span)
            if candidate is not None:
                candidate.span = span
                candidates.append(candidate)

        return candidates