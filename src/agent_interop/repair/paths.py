"""Immutable JSON path helpers for the repair pipeline.

These operate on nested dicts/lists without mutating the original.
All mutations produce a new copy with the change applied.
"""

from __future__ import annotations

import copy
from typing import Any

_MISSING = object()


def _json_type(value: Any) -> str:
    """Return JSON Schema type name for a Python value.

    Distinguishes boolean from integer (item 53: JSON semantic types).
    """
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _sort_key(key: Any) -> tuple[int, str]:
    """Sort key for deterministic dict-key iteration in diff."""
    if isinstance(key, str):
        return (0, key)
    return (1, str(key))


def get_at_path(instance: Any, path: list[str | int]) -> Any:
    """Get the value at a JSON path. Returns _MISSING if unreachable."""
    current = instance
    for segment in path:
        if isinstance(current, dict) and isinstance(segment, str):
            if segment in current:
                current = current[segment]
            else:
                return _MISSING
        elif isinstance(current, list) and isinstance(segment, int):
            if 0 <= segment < len(current):
                current = current[segment]
            else:
                return _MISSING
        else:
            return _MISSING
    return current


def set_at_path(instance: dict[str, Any], path: list[str | int], value: Any) -> dict[str, Any]:
    """Return a deep copy of instance with value set at path.

    Does NOT silently fabricate intermediate containers — if a path
    segment cannot be traversed, the operation fails and returns the
    original instance unchanged. Never mutates input.

    For list indices, only existing indices or index == len(list)
    (append) are accepted. Out-of-range indices are rejected.
    """
    if not path:
        if isinstance(value, dict):
            return value
        return instance

    result = copy.deepcopy(instance)
    current = result
    for segment in path[:-1]:
        if isinstance(current, dict) and isinstance(segment, str):
            if segment not in current or not isinstance(current[segment], (dict, list)):
                # Cannot traverse further — return unchanged copy (item 51)
                return result
            current = current[segment]
        elif isinstance(current, list) and isinstance(segment, int):
            if 0 <= segment < len(current) and isinstance(current[segment], (dict, list)):
                current = current[segment]
            else:
                # Cannot traverse — return unchanged copy (item 50, 51)
                return result
        else:
            # Type mismatch — return unchanged copy (item 50)
            return result

    last = path[-1]
    if isinstance(current, dict) and isinstance(last, str):
        current[last] = value
    elif isinstance(current, list) and isinstance(last, int):
        if 0 <= last <= len(current):
            if last == len(current):
                current.append(value)
            else:
                current[last] = value
        # else: out-of-range — silently skip (already deep-copied)

    return result


def delete_at_path(instance: dict[str, Any], path: list[str | int]) -> dict[str, Any]:
    """Return a deep copy of instance with the key at path removed.

    Never mutates input. Returns unchanged copy if path doesn't exist.
    """
    if not path:
        return instance

    result = copy.deepcopy(instance)
    current = result
    for segment in path[:-1]:
        if isinstance(current, dict) and isinstance(segment, str):
            if segment in current:
                current = current[segment]
            else:
                return result
        elif isinstance(current, list) and isinstance(segment, int):
            if 0 <= segment < len(current):
                current = current[segment]
            else:
                return result
        else:
            return result

    last = path[-1]
    if isinstance(current, dict) and isinstance(last, str) and last in current:
        del current[last]
    elif isinstance(current, list) and isinstance(last, int) and 0 <= last < len(current):
        current.pop(last)

    return result


def diff_paths(before: Any, after: Any, prefix: tuple[str | int, ...] = ()) -> set[tuple[str | int, ...]]:
    """Return the set of paths that differ between two JSON values.

    Works recursively through dicts and lists. Each path is a tuple of
    segments (str for object keys, int for array indices).

    Uses JSON semantic types for comparison (bool vs int are distinct,
    matching JSON Schema type semantics, item 53).
    """
    changed: set[tuple[str | int, ...]] = set()

    # Type mismatch (JSON semantics: bool ≠ int, item 53)
    if _json_type(before) != _json_type(after):
        changed.add(prefix)
        return changed

    if isinstance(before, dict) and isinstance(after, dict):
        all_keys = set(before.keys()) | set(after.keys())
        for key in sorted(all_keys, key=_sort_key):
            if key not in before or key not in after:
                changed.add(prefix + (key,))
            elif before[key] != after[key]:
                sub = diff_paths(before[key], after[key], prefix + (key,))
                changed.update(sub)
        return changed

    if isinstance(before, list) and isinstance(after, list):
        # Element-wise diff — report index paths for each differing element
        max_len = max(len(before), len(after))
        for i in range(max_len):
            if i >= len(before) or i >= len(after):
                changed.add(prefix + (i,))
            elif before[i] != after[i]:
                sub = diff_paths(before[i], after[i], prefix + (i,))
                changed.update(sub)
        return changed

    if before != after:
        changed.add(prefix)

    return changed
