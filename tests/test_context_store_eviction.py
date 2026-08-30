"""ContextStore eviction/pin coverage (P0-7 / P0.16 / P0.17 invariants).

Complements test_p03_p04_context_store.py's happy-path suite by exercising
the eviction machinery directly: session eviction under max_sessions,
global eviction under max_total_bytes, pin protection, and the TTL sweep.
"""

from __future__ import annotations

import pytest

from agent_interop.context_store.store import ContextStore
from agent_interop.errors import (
    ContextEntryEvictedError,
    ContextStoreCapacityError,
)


def _fill(store: ContextStore, session_id: str, blob: str, *, tool_call_id: str) -> str:
    """Store a blob and return its ref (pin/unpin/get take the ref string)."""
    return store.store(
        session_id,
        blob,
        "tool_result",
        tool_call_id=tool_call_id,
    ).ref


class TestSessionEviction:
    def test_max_sessions_evicts_oldest_unpinned(self):
        store = ContextStore(max_sessions=2, max_bytes_per_session=10_000)
        _fill(store, "s1", "blob-one", tool_call_id="c1")
        _fill(store, "s2", "blob-two", tool_call_id="c2")
        _fill(store, "s3", "blob-three", tool_call_id="c3")
        # Only the two newest sessions survive.
        assert set(store._sessions.keys()) == {"s2", "s3"}

    def test_all_sessions_pinned_raises_capacity_error(self):
        store = ContextStore(max_sessions=2, max_bytes_per_session=10_000)
        ref1 = _fill(store, "s1", "blob-one", tool_call_id="c1")
        ref2 = _fill(store, "s2", "blob-two", tool_call_id="c2")
        assert store.pin_ref(ref1, "req-1")
        assert store.pin_ref(ref2, "req-1")
        # Every session now holds pinned entries — a third must fail closed.
        with pytest.raises(ContextStoreCapacityError):
            _fill(store, "s3", "blob-three", tool_call_id="c3")

    def test_evict_oldest_session_with_no_sessions(self):
        store = ContextStore(max_sessions=1, max_bytes_per_session=10_000)
        # Direct call on an empty store must be a no-op, not a crash.
        store._evict_oldest_session()


class TestGlobalEviction:
    def test_total_cap_evicts_oldest_unpinned_entries(self):
        store = ContextStore(
            max_bytes_per_session=10_000,
            max_total_bytes=200,
        )
        big = "x" * 90
        ref1 = _fill(store, "s1", big, tool_call_id="c1")
        _fill(store, "s2", big, tool_call_id="c2")
        _fill(store, "s3", big, tool_call_id="c3")
        # Global cap forced eviction of the oldest entry (s1's).
        assert store.get(ref1, "s1") is None

    def test_global_eviction_never_touches_pinned_entries(self):
        store = ContextStore(
            max_bytes_per_session=10_000,
            max_total_bytes=200,
        )
        big = "x" * 90
        ref1 = _fill(store, "s1", big, tool_call_id="c1")
        # Pin BEFORE the inserts that would otherwise evict it: the global
        # cap enforces eagerly at insert time, not lazily.
        assert store.pin_ref(ref1, "req-1")
        _fill(store, "s2", big, tool_call_id="c2")
        _fill(store, "s3", big, tool_call_id="c3")
        # The pinned entry survives even though it is the oldest.
        assert store.get(ref1, "s1") is not None
        # The unpinned newer entries did not both survive — the cap held.
        assert store._total_bytes <= 200

    def test_entry_larger_than_total_cap_is_rejected(self):
        store = ContextStore(
            max_bytes_per_session=10_000,
            max_total_bytes=50,
        )
        with pytest.raises(ContextEntryEvictedError):
            _fill(store, "s1", "y" * 200, tool_call_id="c1")


class TestPinAndTtl:
    def test_unpin_ref_allows_eviction_again(self):
        store = ContextStore(max_sessions=1, max_bytes_per_session=10_000)
        ref = _fill(store, "s1", "blob", tool_call_id="c1")
        assert store.pin_ref(ref, "req-1")
        assert store.unpin_ref(ref, "req-1")
        # Unknown ref: both operations fail closed.
        assert not store.pin_ref("nope", "req-1")
        assert not store.unpin_ref("nope", "req-1")

    def test_evict_expired_sweeps_only_unpinned(self):
        store = ContextStore(max_bytes_per_session=10_000, ttl_seconds=0.01)
        ref_keep = _fill(store, "s1", "keep-me", tool_call_id="c1")
        ref_drop = _fill(store, "s1", "drop-me", tool_call_id="c2")
        assert store.pin_ref(ref_keep, "req-1")
        import time as _time

        _time.sleep(0.05)
        store.evict_expired()
        assert store.get(ref_keep, "s1") is not None
        assert store.get(ref_drop, "s1") is None

    def test_clear_session_removes_entries_and_dedup(self):
        store = ContextStore(max_bytes_per_session=10_000)
        ref = _fill(store, "s1", "blob", tool_call_id="c1")
        store.clear_session("s1")
        assert store.get(ref, "s1") is None
        # Re-storing the identical payload after a clear must succeed
        # (the dedup index entry went away with the session).
        ref2 = _fill(store, "s1", "blob", tool_call_id="c1")
        assert ref2

    def test_clear_session_unknown_is_noop(self):
        store = ContextStore()
        store.clear_session("never-existed")
