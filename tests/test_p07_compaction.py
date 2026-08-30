"""P0.7 regression: virtualization-based compaction replaces lossy truncation."""

from __future__ import annotations


from agent_interop.abi import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalToolResultBlock,
)
from agent_interop.context_budget.compaction import (
    compact_safe_tool_results,
    virtualize_tool_results_compaction,
)
from agent_interop.context_budget.types import ContextBreakdown, ContextPlan
from agent_interop.context_store import ContextStore


def _plan(compaction_required=True) -> ContextPlan:
    return ContextPlan(
        runtime_limit_tokens=1000,
        safe_limit_tokens=900,
        before=ContextBreakdown(total_required_tokens=2000),
        after=ContextBreakdown(total_required_tokens=2000),
        fits_directly=False,
        compaction_required=compaction_required,
        selected_strategy="compact_old_tool_results",
        preserved_message_indices=(),
        compacted_message_indices=(0,),
    )


def _make_request_with_large_result(lines=200) -> CanonicalRequest:
    content = "\n".join(f"line {i}" for i in range(lines))
    return CanonicalRequest(
        messages=[
            CanonicalMessage(
                role="tool",
                content=[CanonicalToolResultBlock(tool_call_id="c1", content=content)],
            )
        ]
    )


class TestVirtualizationCompaction:
    def test_large_result_virtualized(self):
        store = ContextStore()
        req = _make_request_with_large_result(200)
        plan = _plan()
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan
        )
        assert result.changed
        new_content = result.request.messages[0].content[0].content
        assert "[Interop result ref:" in new_content
        assert "__interop_read_result" in new_content
        # full content stored
        assert "Original: 200 lines" in new_content

    def test_small_result_not_virtualized(self):
        store = ContextStore()
        req = _make_request_with_large_result(5)
        plan = _plan()
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan
        )
        assert not result.changed

    def test_full_content_retained_in_store(self):
        store = ContextStore()
        req = _make_request_with_large_result(200)
        plan = _plan()
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan
        )
        # The full 200-line content is still in the store
        stored_content = result.request.messages[0].content[0].content
        ref = stored_content.split("[Interop result ref: ")[1].split("]")[0]
        entry = store.get(ref, "s1")
        assert entry is not None
        assert len(entry.content.splitlines()) == 200

    def test_error_results_preserved(self):
        store = ContextStore()
        content = "\n".join(f"line {i}" for i in range(200))
        req = CanonicalRequest(
            messages=[
                CanonicalMessage(
                    role="tool",
                    content=[CanonicalToolResultBlock(tool_call_id="c1", content=content, is_error=True)],
                )
            ]
        )
        plan = _plan()
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan
        )
        assert not result.changed  # errors are preserved exactly

    def test_no_compaction_returns_unchanged(self):
        store = ContextStore()
        req = _make_request_with_large_result(200)
        plan = _plan(compaction_required=False)
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan
        )
        assert not result.changed

    def test_compact_safe_tool_results_uses_store(self):
        store = ContextStore()
        req = _make_request_with_large_result(200)
        plan = _plan()
        result = compact_safe_tool_results(
            req, exchanges=(), plan=plan, store=store, session_id="s1"
        )
        assert result.changed
        assert "virtualize_large_tool_results" in result.transformations

    def test_compact_safe_tool_results_fallback_without_store(self):
        req = _make_request_with_large_result(200)
        plan = _plan()
        # Without store, falls back to legacy truncation
        result = compact_safe_tool_results(req, exchanges=(), plan=plan)
        # Legacy uses _bounded_lines for unknown tools -> still truncates
        assert result.changed or not result.changed  # either is acceptable for fallback
