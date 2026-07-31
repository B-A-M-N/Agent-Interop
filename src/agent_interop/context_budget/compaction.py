"""Safe context-compaction policy markers.

Actual model summarization belongs to controlled execution; keeping these
invariants central prevents an adapter from silently losing required state.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalMessage


def is_required_message(message: CanonicalMessage, index: int, last_index: int) -> bool:
    return index == last_index or message.role in {"system", "developer", "user", "tool"}
