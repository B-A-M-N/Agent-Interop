"""P0.8 regression: system prompt projection."""

from __future__ import annotations

from agent_interop.abi import CanonicalTextBlock
from agent_interop.projection import project_system
from agent_interop.projection.system import (
    SystemProjectionResult,
    _detect_version,
    _is_execution_critical,
)


class TestSystemProjection:
    def test_detect_claude_code_version(self):
        system = [CanonicalTextBlock(text="Claude Code v2.1.220")]
        assert "claude_code" in _detect_version(system)

    def test_detect_unknown(self):
        system = [CanonicalTextBlock(text="some generic system")]
        assert _detect_version(system) == "unknown"

    def test_execution_critical_preserved(self):
        assert _is_execution_critical("You MUST always verify your work")
        assert _is_execution_critical("IMPORTANT: do not delete files")
        assert not _is_execution_critical("Here is some context")

    def test_no_heuristic_deletion_without_markers(self):
        # Beta contract: heuristic heading-based deletion is disabled.
        # Only exact duplicate markers in _KNOWN_DUPLICATE_MARKERS trigger deletion.
        from agent_interop.projection.system import _KNOWN_DUPLICATE_MARKERS

        # Markers list is empty by default -> nothing deleted.
        assert _KNOWN_DUPLICATE_MARKERS == []

    def test_project_claude_code(self):
        system = [
            CanonicalTextBlock(text="# Environment\n<env>details</env>"),
            CanonicalTextBlock(text="You MUST always verify your work."),
            CanonicalTextBlock(text="# Tool Reference\nDetailed docs here..."),
        ]
        result = project_system(system, client_id="claude_code")
        assert isinstance(result, SystemProjectionResult)
        # execution-critical preserved
        assert any("MUST" in getattr(b, "text", "") for b in result.projected)
        # P0-32: For beta, heuristic heading-based deletion is disabled.
        # All content is preserved verbatim when no exact duplicate markers match.
        assert any("Tool Reference" in getattr(b, "text", "") for b in result.projected)

    def test_project_generic(self):
        system = [
            CanonicalTextBlock(text="You MUST follow these rules."),
            CanonicalTextBlock(text="Line " * 100),
        ]
        result = project_system(system, client_id="unknown")
        assert result.version_detected == "generic"
        assert result.reduction_ratio >= 0

    def test_empty_system(self):
        result = project_system([], client_id="claude_code")
        assert result.projected == []

    def test_projection_preserves_content_for_beta(self):
        # P0-32: For beta, unknown prompts and heuristic-matched content
        # are preserved verbatim. No reduction occurs without exact markers.
        system = [
            CanonicalTextBlock(text="# Tool Reference\n" + "x" * 1000),
            CanonicalTextBlock(text="# Environment\n<env>data</env>"),
        ]
        result = project_system(system, client_id="claude_code")
        assert result.projected_length == result.original_length
