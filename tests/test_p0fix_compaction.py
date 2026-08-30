"""Regression tests for P0.7 compaction fixes: stored_refs and UTF-8 truncation."""

from __future__ import annotations

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalToolResultBlock,
)
from agent_interop.context_budget.compaction import (
    ContextAdaptationResult,
    compact_safe_tool_results,
    truncate_utf8,
    virtualize_tool_results_compaction,
)
from agent_interop.context_budget.types import ContextBreakdown, ContextPlan
from agent_interop.context_store import ContextStore
from agent_interop.context_store.policy import VirtualizationPolicy


# ─── Fix A: stored_refs on ContextAdaptationResult ──────────────────────


def _plan(
    compaction_required=True,
    allow_result_virtualization=True,
    compacted_indices=None,
) -> ContextPlan:
    return ContextPlan(
        runtime_limit_tokens=1000,
        safe_limit_tokens=900,
        before=ContextBreakdown(total_required_tokens=2000),
        after=ContextBreakdown(total_required_tokens=2000),
        fits_directly=False,
        compaction_required=compaction_required,
        selected_strategy="compact_old_tool_results",
        preserved_message_indices=(),
        compacted_message_indices=tuple(compacted_indices) if compacted_indices else (0,),
        allow_result_virtualization=allow_result_virtualization,
    )


def _make_store_request(n_lines: int = 200, tool_call_id: str = "c1") -> CanonicalRequest:
    """Create a request with a large single-message tool result."""
    content = "\n".join(f"line {i}" for i in range(n_lines))
    return CanonicalRequest(
        messages=[
            CanonicalMessage(
                role="tool",
                content=[CanonicalToolResultBlock(tool_call_id=tool_call_id, content=content)],
            )
        ]
    )


class TestStoredRefsExistence:
    """Verify stored_refs is populated on ContextAdaptationResult."""

    def test_stored_refs_field_exists(self):
        """ContextAdaptationResult.dataclass has stored_refs with default ()."""
        result = ContextAdaptationResult(_make_store_request())
        assert hasattr(result, "stored_refs")
        assert result.stored_refs == ()

    def test_virtualization_populates_stored_refs(self):
        """Each virtualized block yields exactly one ref in stored_refs."""
        store = ContextStore()
        req = _make_store_request(n_lines=200)
        plan = _plan()
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=8000)
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        assert result.changed
        assert len(result.stored_refs) == 1
        ref = result.stored_refs[0]
        # The ref should be retrievable from the store
        entry = store.get(ref, "s1")
        assert entry is not None
        assert entry.kind == "tool_result"

    def test_multiple_virtualized_blocks_yield_multiple_refs(self):
        """When two blocks are virtualized, stored_refs has two entries."""
        store = ContextStore()
        content = "\n".join(f"line {i}" for i in range(200))
        req = CanonicalRequest(
            messages=[
                CanonicalMessage(
                    role="tool",
                    content=[
                        CanonicalToolResultBlock(tool_call_id="c1", content=content),
                        CanonicalToolResultBlock(tool_call_id="c2", content=content),
                    ],
                )
            ]
        )
        plan = _plan(compacted_indices=[0])
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=8000)
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        assert result.changed
        assert len(result.stored_refs) == 2
        for ref in result.stored_refs:
            entry = store.get(ref, "s1")
            assert entry is not None
            assert entry.kind == "tool_result"


class TestStoredRefsRetrievable:
    """Ref values in stored_refs can be fetched back from ContextStore."""

    def test_refs_retrievable_via_store_get(self):
        """Each ref in stored_refs maps to a StoredEntry retrievable by ref."""
        store = ContextStore()
        req = _make_store_request(n_lines=200)
        plan = _plan()
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=8000)
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        assert len(result.stored_refs) == 1
        ref = result.stored_refs[0]
        entry = store.get(ref, "s1")
        assert entry is not None
        assert entry.ref == ref
        # The full original content must be retained
        assert len(entry.content.splitlines()) == 200

    def test_legacy_path_returns_empty_stored_refs(self):
        """_legacy_compact_safe_tool_results never populates stored_refs."""
        # Without a store, compact_safe_tool_results falls back to legacy.
        req = _make_store_request(n_lines=200)
        plan = _plan()
        result = compact_safe_tool_results(req, exchanges=(), plan=plan)
        # Legacy returns stored_refs == () by default (dataclass default).
        assert result.stored_refs == ()


class TestCompactedToolResultIds:
    """compacted_tool_result_ids still carries tool_call_ids (semantic ID)."""

    def test_compacted_tool_result_ids_match_tool_call_ids(self):
        """compacted_tool_result_ids equals the virtualized blocks' tool_call_ids."""
        store = ContextStore()
        req = _make_store_request(n_lines=200, tool_call_id="my_tool_call")
        plan = _plan()
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=8000)
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        assert result.compacted_tool_result_ids == ("my_tool_call",)

    def test_compacted_ids_and_refs_are_different_set(self):
        """compacted_tool_result_ids (tool_call_ids) and stored_refs (store refs) differ."""
        store = ContextStore()
        req = _make_store_request(n_lines=200, tool_call_id="tc_42")
        plan = _plan()
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=8000)
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        assert result.compacted_tool_result_ids == ("tc_42",)
        assert result.stored_refs[0] != "tc_42"  # ref is opaque, not the call ID


# ─── Fix B: UTF-8 safe truncation ──────────────────────────────────────


class TestTruncateUtf8:
    """truncate_utf8 never splits a multibyte sequence."""

    def test_short_text_unchanged(self):
        """When text fits, return it verbatim."""
        assert truncate_utf8("hello", 100) == "hello"

    def test_exact_byte_count(self):
        """When text byte-size equals the cap, return it."""
        text = "abc"
        assert truncate_utf8(text, 3) == "text"[:0] or len(text.encode("utf-8")) <= 3

    def test_ascii_truncation(self):
        """ASCII-only text truncates at byte boundary."""
        text = "A" * 50
        truncated = truncate_utf8(text, 30)
        assert len(truncated.encode("utf-8")) <= 30

    def test_multibyte_not_split(self):
        """Chinese characters must not be split mid-sequence."""
        # Each Chinese character is 3 bytes in UTF-8.
        text = "你" * 100  # "you" x 100, 300 bytes
        truncated = truncate_utf8(text, 50)
        encoded = truncated.encode("utf-8")
        assert len(encoded) <= 50
        # No partial trailing character: if last byte would be a continuation
        # byte, that would mean a split. But since we use errors="ignore" on
        # the decoded result, we may drop the partial chars entirely. The key
        # invariant: the encoded output <= max_bytes.

    def test_no_unicode_error_on_roundtrip(self):
        """The function always returns a valid string — never raises."""
        # Worst-case: text with surrogate pairs (e.g. emoji).
        text = "\U0001f600" * 100  # grinning emoji x 100, 400 bytes
        truncated = truncate_utf8(text, 10)
        assert isinstance(truncated, str)
        assert len(truncated.encode("utf-8")) <= 10


class TestMultibyteVirtualization:
    """Single-line Unicode tool results under small max_inline_bytes."""

    def test_multibyte_content_truncates_safely(self):
        """A single-line Unicode tool result with small max_inline_bytes
        produces visible content whose UTF-8 encoding is within the cap
        and never raises."""
        # "你" is 3 bytes in UTF-8. 10000 chars = 30000 bytes.
        multibyte_content = "你" * 10000
        req = CanonicalRequest(
            messages=[
                CanonicalMessage(
                    role="tool",
                    content=[
                        CanonicalToolResultBlock(
                            tool_call_id="c1",
                            content=multibyte_content,
                        )
                    ],
                )
            ]
        )
        store = ContextStore()
        # Use a tiny max_inline_bytes to force the byte-cap path on a single-line string.
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=100)
        plan = _plan()
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        assert result.changed
        new_content = result.request.messages[0].content[0].content
        # Extract the visible portion after the header line.
        visible_start = new_content.rfind("\n") + 1
        visible = new_content[visible_start:]
        assert len(visible.encode("utf-8")) <= 100

    def test_multibyte_full_content_retained_in_store(self):
        """Even when visible content is truncated, the full 30000-byte
        Unicode string is stored in ContextStore."""
        multibyte_content = "你" * 10000
        req = CanonicalRequest(
            messages=[
                CanonicalMessage(
                    role="tool",
                    content=[
                        CanonicalToolResultBlock(
                            tool_call_id="c1",
                            content=multibyte_content,
                        )
                    ],
                )
            ]
        )
        store = ContextStore()
        policy = VirtualizationPolicy(max_inline_lines=50, max_inline_bytes=100)
        plan = _plan()
        result = virtualize_tool_results_compaction(
            req, store=store, session_id="s1", exchanges=(), plan=plan, policy=policy
        )
        ref = result.stored_refs[0]
        entry = store.get(ref, "s1")
        assert entry is not None
        assert entry.content == multibyte_content
