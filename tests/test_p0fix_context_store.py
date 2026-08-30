"""P0-fix tests: store search LRU, ref-based schema paging, recall rendering."""

from __future__ import annotations

import json
import time

from agent_interop.context_store.executor import (
    InternalExecutionContext,
    InternalToolExecutor,
)
from agent_interop.context_store.store import ContextStore

# ─── Task 1: search() refreshes last_accessed_at ───────────────────────────


class TestSearchLruRefresh:
    def test_search_refreshes_last_accessed_at(self):
        """search() must update last_accessed_at so LRU eviction is accurate."""
        store = ContextStore()
        entry = store.store("s1", "searchable content", "tool_result", tool_call_id="c1")
        old_accessed = entry.last_accessed_at
        # Give the clock a tick so we can detect a change.
        time.sleep(0.01)
        results = store.search("s1", "searchable")
        assert len(results) == 1
        assert results[0].last_accessed_at > old_accessed

    def test_search_does_not_update_non_matching(self):
        """Only entries that match the query get their timestamp updated."""
        store = ContextStore()
        entry = store.store("s1", "old entry", "tool_result", tool_call_id="c1")
        time.sleep(0.01)
        matching_entry = store.store("s1", "searchable content", "tool_result", tool_call_id="c2")
        # Snapshot old_accessed BEFORE search.
        old_accessed = entry.last_accessed_at
        store.search("s1", "searchable")
        # Only the matching entry should have been touched; the non-matching
        # one stays at its old timestamp.  (get() also updates last_accessed_at
        # so we read it via the internal bucket to avoid double-update noise.)
        bucket = store._sessions["s1"]
        assert bucket.entries[entry.ref].last_accessed_at == old_accessed
        # The matching entry must have been updated.
        assert bucket.entries[matching_entry.ref].last_accessed_at > old_accessed


# ─── Task 2: oversized schema stored as ref, small schema inline ──────────


def _make_tool(name: str, schema: dict, description: str = ""):
    from agent_interop.abi import CanonicalTool

    return CanonicalTool(name=name, description=description, input_schema=schema)


class TestSchemaPaging:
    def _executor(self):
        store = ContextStore()
        return InternalToolExecutor(store)

    def test_oversized_schema_stored_as_ref(self):
        """A schema exceeding MAX_SCHEMA_CHARS is stored and returns a ref."""
        executor = self._executor()
        # Build a schema that is definitely > MAX_SCHEMA_CHARS chars.
        big_props = {f"field_{i}": {"type": "string", "description": f"description {i}" * 20}
                      for i in range(200)}
        tool = _make_tool("BigTool", {"type": "object", "properties": big_props})
        ctx = InternalExecutionContext(
            session_id="s1",
            authorized_tools={"BigTool": tool},
            withheld_tools=frozenset({"BigTool"}),
        )
        result = executor.execute(
            "__interop_get_tool_schema",
            {"name": "BigTool"},
            "s1",
            context=ctx,
        )
        assert not result.is_error
        assert result.ref, "Oversized schema must store a ref"
        # The content must mention the ref so the model can page.
        assert "ref " in result.content.lower() or f"ref {result.ref}" in result.content.lower()
        # The first 20 lines should be present.
        assert "BigTool" in result.content
        # There should be more lines than what's shown (20 preview lines).
        total_lines = len(result.content.splitlines())
        assert total_lines > 20

    def test_read_result_pages_oversized_schema(self):
        """_read_result can page through a ref'd schema via start_line/line_count."""
        executor = self._executor()
        big_props = {f"field_{i}": {"type": "string", "description": f"description {i}" * 20}
                      for i in range(200)}
        tool = _make_tool("BigTool", {"type": "object", "properties": big_props})
        ctx = InternalExecutionContext(
            session_id="s1",
            authorized_tools={"BigTool": tool},
            withheld_tools=frozenset({"BigTool"}),
        )
        result = executor.execute(
            "__interop_get_tool_schema",
            {"name": "BigTool"},
            "s1",
            context=ctx,
        )
        ref = result.ref
        assert ref
        # Read lines after the 20-line preview header — start at line 21.
        page = executor.execute(
            "__interop_read_result",
            {"ref": ref, "start_line": 21, "line_count": 10},
            "s1",
        )
        assert not page.is_error
        assert page.content  # must return content

    def test_small_schema_returns_inline_json(self):
        """A schema that fits within MAX_SCHEMA_CHARS is returned inline (no ref)."""
        executor = self._executor()
        tool = _make_tool("SmallTool", {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        })
        ctx = InternalExecutionContext(
            session_id="s1",
            authorized_tools={"SmallTool": tool},
            withheld_tools=frozenset({"SmallTool"}),
        )
        result = executor.execute(
            "__interop_get_tool_schema",
            {"name": "SmallTool"},
            "s1",
            context=ctx,
        )
        assert not result.is_error
        assert result.ref == "" or result.ref is None, "Small schema must not store a ref"
        assert '"SmallTool"' not in result.content  # name is in header, not inline schema
        assert '"name"' in result.content  # inline JSON schema present
        # Must contain valid-ish JSON (not truncated).
        assert "}" in result.content  # complete JSON object


# ─── Task 3 & 4: recall_history routes through render_fragment_text ───────


class TestRecallHistoryRendering:
    def test_history_fragment_routes_through_render_fragment_text(self):
        """recall_history on kind='history_fragment' renders human-readable text.

        Stores a real canonical-JSON fragment and asserts the output looks like
        "[user] hello" rather than raw JSON braces.
        """
        store = ContextStore()
        executor = InternalToolExecutor(store)
        # Real canonical-JSON fragment.
        canonical = json.dumps({
            "schema_version": 1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"}
                    ],
                }
            ],
        })
        entry = store.store(
            "s1", canonical, kind="history_fragment", tool_call_id="h1",
        )
        result = executor.execute(
            "__interop_recall_history",
            {"ref": entry.ref},
            "s1",
        )
        assert not result.is_error
        # Must NOT contain raw JSON braces (rendered text).
        assert "{" not in result.content or '"schema_version"' not in result.content
        # Must contain human-readable text with role prefix and message content.
        # render_fragment_text uses "user: ..." format (no brackets).
        assert "user:" in result.content
        assert "hello" in result.content
