"""Virtualization policy for tool results.

Decides which tool results should be virtualized (stored in the ContextStore
and replaced with a bounded handle) vs. passed through verbatim.

Default policy: virtualize ALL large results by default, preserving special
policy for things that need exact representation. This is safer than silent
truncation because nothing is lost — the full result is retained behind a ref.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass

from agent_interop.abi import CanonicalTextBlock, CanonicalToolResultBlock


@dataclass(frozen=True)
class VirtualizationDecision:
    """Whether and how to virtualize a tool result."""

    should_virtualize: bool
    reason: str = ""
    max_inline_lines: int = 50


class VirtualizationPolicy:
    """Decide whether a tool result should be virtualized."""

    def __init__(self, max_inline_lines: int = 50, max_inline_bytes: int = 8000) -> None:
        self._max_inline_lines = max_inline_lines
        self._max_inline_bytes = max_inline_bytes

    def decide(self, result: CanonicalToolResultBlock, tool_name: str = "") -> VirtualizationDecision:
        content = self._extract_text(result)
        if not content:
            return VirtualizationDecision(False, "empty")
        lines = content.splitlines()
        byte_size = len(content.encode("utf-8", "replace"))
        if len(lines) <= self._max_inline_lines and byte_size <= self._max_inline_bytes:
            return VirtualizationDecision(False, "small")
        return VirtualizationDecision(
            True,
            f"large: {len(lines)} lines, {byte_size} bytes",
            self._max_inline_lines,
        )

    def _extract_text(self, result: CanonicalToolResultBlock) -> str:
        """P1.3: produce a deterministic, model-readable text representation of
        a tool result WITHOUT arbitrarily flattening structured content via
        ``str(...)``. Strings pass through verbatim; structured blocks are
        serialized to canonical JSON so search/paging stay lossless."""
        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, CanonicalTextBlock):
                    parts.append(block.text)
                elif is_dataclass(block):
                    parts.append(json.dumps(asdict(block), default=str, sort_keys=True))
                else:
                    parts.append(json.dumps(block, default=str, sort_keys=True))
            return "\n".join(parts)
        if is_dataclass(content):
            return json.dumps(asdict(content), default=str, sort_keys=True)
        return json.dumps(content, default=str, sort_keys=True)


def default_virtualization_policy() -> VirtualizationPolicy:
    return VirtualizationPolicy()
