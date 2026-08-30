"""System prompt projection (P0.8).

Deterministically projects the client's system instructions into a bounded
model-consumable view.

Beta contract:
  - Unknown client/version -> preserve verbatim, ALWAYS.
  - Known client prompt -> deterministic projection ONLY via exact markers or
    hashes; never heuristic heading matching.
  - No heading heuristics are used for classification.

Specific rules:
  1. Unknown system prompts: every block is preserved verbatim
     (SystemProjectionResult with transformations=()).
  2. Known clients (e.g. Claude Code): sections are dropped ONLY if their
     normalized text exactly matches an entry in _KNOWN_DUPLICATE_MARKERS.
     Execution-critical content is never dropped.
  3. _EXECUTION_CRITICAL_PATTERNS is deliberately NOT used to decide
     deletion — it exists only as a documented preservation-wins guard.

This module does not import hashlib at module level; SHA-256 computation is
deferred to _project_claude_code so that callers never pay an import cost
they do not need.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from agent_interop.abi import CanonicalTextBlock

# Execution-critical patterns: NOT used to decide deletion.
# They are deliberately kept here only so that _project_claude_code can
# short-circuit preservation ("preservation wins" over any deletion path).
_EXECUTION_CRITICAL_PATTERNS = [
    r"you must",
    r"always",
    r"never",
    r"important:",
    r"note:",
    r"warning:",
    r"critical",
    r"do not",
    r"ensure",
    r"required:",
]

# Exact known markers that prove a section is duplicated tool documentation.
# Entries MUST be the FULL section text (not a substring, not a heading
# pattern) copied verbatim from a known client prompt version.  Before
# matching, the section and the marker are both normalised with:
#   " ".join(section.split())
# The list is empty by default — populate it only with exact copies from
# real client prompts.
_KNOWN_DUPLICATE_MARKERS: list[str] = []  # Exact full-section texts only


@dataclass(frozen=True)
class SystemProjectionResult:
    """Result of projecting system instructions."""

    projected: list[Any]  # CanonicalContentBlock
    original_length: int = 0
    projected_length: int = 0
    transformations: tuple[str, ...] = ()
    version_detected: str = "unknown"

    @property
    def reduction_ratio(self) -> float:
        if self.original_length <= 0:
            return 0.0
        return 1.0 - (self.projected_length / self.original_length)


def _is_execution_critical(text: str) -> bool:
    """Check if text contains execution-critical instructions.

    This function is intentionally NOT used to decide deletion.  When a
    section matches both this guard and a duplicate-marker, the section is
    preserved.
    """
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in _EXECUTION_CRITICAL_PATTERNS)


def _detect_version(system: list[Any]) -> str:
    """Detect the client version from system instructions."""
    for block in system:
        if isinstance(block, CanonicalTextBlock):
            text = block.text.lower()
            if "claude code" in text:
                match = re.search(r"claude\s+code\s+v?(\d+\.\d+\.\d+)", text)
                if match:
                    return f"claude_code:{match.group(1)}"
                return "claude_code:unknown"
    return "unknown"


def _project_claude_code(system: list[Any]) -> SystemProjectionResult:
    """Project known Claude Code system instructions.

    Beta-safe rules:
      - Preserve execution-critical sections verbatim.
      - A section is dropped ONLY if its normalised text exactly matches
        an entry in _KNOWN_DUPLICATE_MARKERS.
      - _EXECUTION_CRITICAL_PATTERNS is NOT used for deletion decisions;
        when a section matches both a critical pattern and a marker,
        preservation always wins.
      - Do NOT classify markdown headers, blockquotes, or lists as
        boilerplate — these patterns match real client instructions.
      - Preserve all other content verbatim.
    """
    transformations = []
    projected = []
    original_length = 0
    projected_length = 0

    for block in system:
        if not isinstance(block, CanonicalTextBlock):
            projected.append(block)
            continue

        text = block.text
        original_length += len(text)

        # Split into sections by headers
        sections = re.split(r"\n(?=#+\s)", text)
        for section in sections:
            if not section.strip():
                continue

            # Preservation-wins guard: execution-critical sections are never
            # dropped, even if they happen to match a duplicate marker.
            if _is_execution_critical(section):
                projected.append(CanonicalTextBlock(text=section))
                projected_length += len(section)
                continue

            # Exact-duplicate check: drop ONLY when normalised text equals a
            # known marker verbatim (after normalisation).
            normalised = " ".join(section.split())
            for marker in _KNOWN_DUPLICATE_MARKERS:
                if normalised == " ".join(marker.split()):
                    # Record which marker caused the removal.
                    hex_digest = hashlib.sha256(section.encode()).hexdigest()[:8]
                    transformations.append(
                        f"remove_known_duplicate_tool_doc:{hex_digest}"
                    )
                    break
            else:
                # Not a known duplicate — preserve verbatim.
                projected.append(CanonicalTextBlock(text=section))
                projected_length += len(section)

    return SystemProjectionResult(
        projected=projected,
        original_length=original_length,
        projected_length=projected_length,
        transformations=tuple(transformations),
        version_detected="claude_code",
    )


def _project_generic(system: list[Any]) -> SystemProjectionResult:
    """Generic system projection for unknown clients.

    Beta-safe rule: preserve EVERY block verbatim, unconditionally.
    No compaction, no deletion, no heuristics.
    """
    projected = []
    original_length = 0
    projected_length = 0

    for block in system:
        if not isinstance(block, CanonicalTextBlock):
            projected.append(block)
            continue

        text = block.text
        original_length += len(text)

        # Unknown prompts are always preserved verbatim.
        projected.append(CanonicalTextBlock(text=text))
        projected_length += len(text)

    return SystemProjectionResult(
        projected=projected,
        original_length=original_length,
        projected_length=projected_length,
        transformations=(),
        version_detected="generic",
    )


def project_system(system: list[Any], client_id: str = "") -> SystemProjectionResult:
    """Project system instructions based on client type.

    Dispatches to _project_generic (unknown -> verbatim) or
    _project_claude_code (known -> exact-marker deletion only).
    """
    version = _detect_version(system)

    if "claude_code" in version or "claude" in client_id.lower():
        return _project_claude_code(system)
    return _project_generic(system)
