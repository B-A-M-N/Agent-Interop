"""Tests for the system-projection hardening (P0-fix).

Verifies the beta contract:
  - Unknown client/version -> verbatim, ALWAYS.
  - Known client prompt -> deterministic projection ONLY via exact markers.
  - No heuristic heading matching.
"""

from __future__ import annotations

import hashlib
import textwrap

import pytest

from agent_interop.abi import CanonicalTextBlock
from agent_interop.projection.system import (
    _KNOWN_DUPLICATE_MARKERS,
    project_system,
)


@pytest.fixture(autouse=True)
def _restore_known_markers():
    """Restore _KNOWN_DUPLICATE_MARKERS after each test."""
    snapshot = list(_KNOWN_DUPLICATE_MARKERS)
    yield
    _KNOWN_DUPLICATE_MARKERS.clear()
    _KNOWN_DUPLICATE_MARKERS.extend(snapshot)


# ---------------------------------------------------------------------------
# (a) Generic prompt round-trips byte-identical
# ---------------------------------------------------------------------------

class TestGenericVerbatim:
    def test_generic_preserves_everything(self):
        """Unknown client -> every block preserved verbatim, transformations==()."""
        prompt = textwrap.dedent(
            """\
            # Environment
            foo
            # State
            bar
            """
        )
        system = [CanonicalTextBlock(text=prompt)]
        result = project_system(system, client_id="unknown_client")

        assert result.version_detected == "generic"
        assert result.transformations == ()
        # Concatenation of projected block texts equals input
        rebuilt = "".join(b.text for b in result.projected)
        assert rebuilt == prompt


class TestClaudeCodePreservation:
    # -----------------------------------------------------------------------
    # (b) Claude-code prompt with tool-ish headings preserved verbatim
    # -----------------------------------------------------------------------
    def test_claude_code_tool_headings_preserved(self):
        """Sections with tool-like headings are NOT deleted unless they match
        an exact marker."""
        prompt = textwrap.dedent(
            """\
            Some intro.

            # Tools
            Tool documentation here.

            # API
            API docs here.

            # Functions
            Function definitions here.
            """
        )
        system = [CanonicalTextBlock(text=prompt)]
        result = project_system(system, client_id="claude_code")

        assert result.version_detected == "claude_code"
        assert result.transformations == ()
        all_text = " ".join(b.text for b in result.projected)
        assert "# Tools" in all_text
        assert "# API" in all_text
        assert "# Functions" in all_text

    def test_claude_code_detects_version(self):
        """Claude Code v1.2.3 is detected as a known client."""
        system = [CanonicalTextBlock(text="claude code v1.2.3\n\nSome instructions.")]
        result = project_system(system, client_id="")
        assert result.version_detected == "claude_code"


# ---------------------------------------------------------------------------
# (c) Exact marker causes deletion, other sections preserved
# ---------------------------------------------------------------------------

class TestExactMarkerDeletion:
    def test_exact_section_dropped(self):
        """When _KNOWN_DUPLICATE_MARKERS contains an exact section text,
        that section is dropped and a transformation is recorded."""
        # Build section and prompt so the regex split cleanly separates sections.
        # The regex r'\n(?=#+\s)' splits on newline + heading, so each
        # heading-started region must be its own block with no trailing
        # non-heading content leaking into it.
        intro = "Some intro."
        section = "# Tool Reference\ndetailed tool docs"
        outro = "# Outro\nsome final text"

        full_prompt = f"{intro}\n\n{section}\n\n{outro}"
        _KNOWN_DUPLICATE_MARKERS.append(section)

        system = [CanonicalTextBlock(text=full_prompt)]
        result = project_system(system, client_id="claude_code")

        assert result.version_detected == "claude_code"
        assert len(result.transformations) == 1
        assert result.transformations[0].startswith("remove_known_duplicate_tool_doc:")
        # The dropped section should NOT appear in the output.
        all_text = "".join(b.text for b in result.projected)
        assert "# Tool Reference" not in all_text
        # Other sections are preserved.
        assert "Some intro." in all_text
        assert "# Outro" in all_text

    def test_transformation_contains_sha256_prefix(self):
        """The transformation record contains the first 8 hex chars of SHA-256."""
        section = "# Duplicated\ncontent"
        _KNOWN_DUPLICATE_MARKERS.append(section)
        hex_prefix = hashlib.sha256(section.encode()).hexdigest()[:8]

        system = [CanonicalTextBlock(text=section)]
        result = project_system(system, client_id="claude_code")

        assert f"remove_known_duplicate_tool_doc:{hex_prefix}" in result.transformations


# ---------------------------------------------------------------------------
# (d) Substring match does NOT cause deletion (exactness)
# ---------------------------------------------------------------------------

class TestSubstringNotDropped:
    def test_section_containing_marker_as_substring_not_dropped(self):
        """A section that merely CONTAINS a marker as a substring is NOT
        dropped — deletion requires exact normalised match."""
        marker = "# Tool Reference"
        _KNOWN_DUPLICATE_MARKERS.append(marker)

        # This section CONTAINS the marker but is longer — not an exact match.
        section = "# Tool Reference\nextra content here\nmore stuff"

        system = [CanonicalTextBlock(text=section)]
        result = project_system(system, client_id="claude_code")

        all_text = "".join(b.text for b in result.projected)
        assert "# Tool Reference" in all_text
        assert "extra content here" in all_text
        # No deletions occurred.
        assert result.transformations == ()


# ---------------------------------------------------------------------------
# (e) Execution-critical content never dropped, even if it matches a marker
# ---------------------------------------------------------------------------

class TestExecutionCriticalPreservation:
    def test_execution_critical_never_dropped(self):
        """Execution-critical content is never dropped even when it exactly
        equals a duplicate marker."""
        critical = "You must never delete files without confirmation."
        _KNOWN_DUPLICATE_MARKERS.append(critical)

        system = [CanonicalTextBlock(text=critical)]
        result = project_system(system, client_id="claude_code")

        all_text = "".join(b.text for b in result.projected)
        assert critical in all_text
        # No transformation should be recorded.
        assert result.transformations == ()

    def test_execution_critical_with_other_sections(self):
        """When a prompt has both execution-critical and duplicate sections,
        the critical one survives."""
        dup = "# Duplicated\ntool docs here"
        _KNOWN_DUPLICATE_MARKERS.append(dup)

        prompt = textwrap.dedent(
            """\
            You must always verify your work.

            # Duplicated
            tool docs here
            """
        )
        system = [CanonicalTextBlock(text=prompt)]
        result = project_system(system, client_id="claude_code")

        all_text = "".join(b.text for b in result.projected)
        assert "must always verify" in all_text
        # The duplicate section is removed.
        assert "# Duplicated" not in all_text
        # One transformation recorded.
        assert len(result.transformations) == 1
