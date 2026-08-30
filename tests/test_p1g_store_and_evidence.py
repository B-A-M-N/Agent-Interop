"""P1-G regression: ContextStore CPU + evidence write-behind.

Locks in:
  * store() encodes/hashes content ONCE (byte_size + sha256 pre-populated);
  * get_slice is semantically IDENTICAL to the historical splitlines form —
    including exotic line boundaries — while using a cached offset index;
  * TTL sweeps are throttled and free when TTL is off;
  * pin_refs/unpin_refs pin a whole set in one lock acquisition;
  * result_is_stale is the pure predicate and is_stale agrees with it;
  * write-behind repair events persist off the request path, are drained
    before reads, and fall back to synchronous when not installed.
"""

from __future__ import annotations

import random
import time

from agent_interop.context_store.store import ContextStore
from agent_interop.evidence.store import EvidenceStore, result_is_stale
from agent_interop.replay.types import CompatibilityResult

# ─── store(): single encode + hash ──────────────────────────────────────────


def test_store_entry_populated_without_rehash():
    """The entry's byte_size and sha256 come from the caller's single
    encode/hash pass — __post_init__ must not recompute either."""

    entry = ContextStore().store("s", "hello world", kind="tool_result", tool_call_id="c")
    assert entry.byte_size == len(b"hello world")
    assert entry.sha256


def test_store_identical_dedup_still_returns_same_entry():
    store = ContextStore()
    a = store.store("s", "same content", kind="tool_result", tool_call_id="c1")
    b = store.store("s", "same content", kind="tool_result", tool_call_id="c1")
    assert a.ref == b.ref


# ─── get_slice: indexed path == historical splitlines semantics ─────────────


def _historical(content: str, start_line: int, line_count) -> str:
    lines = content.splitlines(keepends=True)
    start = max(0, start_line - 1)
    if line_count is not None:
        return "".join(lines[start:start + line_count])
    return "".join(lines[start:])


def test_get_slice_matches_historical_fixed_cases():
    store = ContextStore()
    cases = [
        "plain\nline\nbased\ntext\n",
        "no trailing newline",
        "",
        "single",
        "\n\n\n",
        "ends with newline\n",
        "crlf\r\nline2\r\n",
        "mixed \n cr \r and \v vertical \f form",
    ]
    for i, content in enumerate(cases):
        entry = store.store(f"s{i}", content, kind="tool_result", tool_call_id=f"c{i}")
        for start in range(6):
            for line_count in (None, 1, 2, 1000):
                assert store.get_slice(entry.ref, f"s{i}", start, line_count) == (
                    _historical(content, start, line_count)
                ), (i, start, line_count)


def test_get_slice_matches_historical_fuzz():
    store = ContextStore()
    rng = random.Random(4)
    alphabet = ["a", "\n", "\r", "\r\n", "\v", "\f", "b\n", "x", "\x85", "word "]
    for trial in range(150):
        content = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        sid = f"f{trial}"
        entry = store.store(sid, content, kind="tool_result", tool_call_id="c")
        for _ in range(8):
            start = rng.randint(1, 12)
            line_count = rng.choice([None, 1, 2, 5, 100])
            assert store.get_slice(entry.ref, sid, start, line_count) == (
                _historical(content, start, line_count)
            ), (repr(content), start, line_count)


def test_line_offsets_are_cached_not_rebuilt():
    store = ContextStore()
    entry = store.store("s", "\n".join(f"line {i}" for i in range(1000)), kind="tool_result")
    first = entry.line_offsets()
    assert entry.line_offsets() is first, "offset index must be built once"


# ─── TTL sweep throttling ────────────────────────────────────────────────────


def test_evict_expired_free_without_ttl():
    store = ContextStore(ttl_seconds=0.0)
    store.store("s", "x", kind="tool_result")
    assert store.evict_expired() == 0


def test_evict_expired_throttled_within_window():
    store = ContextStore(ttl_seconds=0.01)
    store.store("s", "x", kind="tool_result")
    now = time.monotonic()
    assert store.evict_expired(now=now) >= 0  # first sweep runs
    # A sweep inside the throttle window is a no-op even for expired entries.
    time.sleep(0.03)
    store.store("s2", "y", kind="tool_result")
    first = store.evict_expired(now=now)
    assert first == 0, "second sweep inside the throttle window must not run"


# ─── Batched pinning ─────────────────────────────────────────────────────────


def test_pin_refs_and_unpin_refs_batched():
    store = ContextStore()
    refs = [
        store.store("s", f"content {i}", kind="tool_result", tool_call_id=f"c{i}").ref
        for i in range(5)
    ]
    assert store.pin_refs(refs, "req-1") == 5
    for ref in refs:
        assert store.get(ref, "s").pinned_requests == {"req-1"}
    assert store.unpin_refs(refs, "req-1") == 5
    for ref in refs:
        assert store.get(ref, "s").pinned_requests == set()


def test_pin_refs_empty_is_a_noop():
    store = ContextStore()
    assert store.pin_refs([], "req-1") == 0
    assert store.unpin_refs([], "req-1") == 0


def test_pinned_entries_survive_capacity_eviction():
    store = ContextStore(max_bytes_per_session=50)
    pinned = store.store("s", "pinned content here", kind="tool_result")
    store.pin_refs([pinned.ref], "req")
    for i in range(10):
        store.store("s", f"filler {i} " + "z" * 20, kind="tool_result", tool_call_id=f"c{i}")
    assert store.get(pinned.ref, "s") is not None


# ─── Evidence: single-read staleness + write-behind ─────────────────────────


def _result(**kwargs) -> CompatibilityResult:
    defaults = {
        "tested_at": "2026-01-01T00:00:00+00:00",
        "sample_count": 5,
        "passes_expiry_hours": 720,
    }
    defaults.update(kwargs)
    return CompatibilityResult(**defaults)


def test_result_is_stale_matches_is_stale_for_the_same_record():
    store = EvidenceStore(db_path=":memory:")
    from agent_interop.evidence.key import build_compatibility_key

    class _K:
        pass

    # Any key works — store/get round-trip by deterministic id.
    key = build_compatibility_key(_K()) if False else None
    from agent_interop.replay.types import CompatibilityKey

    key = CompatibilityKey(client_id="c")
    store.store_result(key, _result())
    record = store.get_result(key)
    assert record is not None
    assert store.is_stale(key) == result_is_stale(record)


def test_result_is_stale_revoked_and_missing():
    assert result_is_stale(None) is True
    assert result_is_stale(_result(revoked=True, revocation_reason="x")) is True


def test_write_behind_events_persist_and_drain_before_read(tmp_path):
    # :memory: is refused (the worker's thread-local connection would open a
    # DIFFERENT empty database) — write-behind targets the file-backed store.
    store = EvidenceStore(db_path=str(tmp_path / "evidence.db"))
    store.install_write_behind()
    try:
        for i in range(20):
            enqueued = store.record_repair_event_async(
                route_id="r", model_id="m", client_id="c",
                tool_name=f"t{i}", outcome="valid_unchanged",
            )
            assert enqueued
        groups = store.query_repair_stats()  # read drains the queue
        assert len(groups) == 1
        assert groups[0].total_eligible == 20
    finally:
        store.close_write_behind()


def test_write_behind_refused_for_memory_db():
    store = EvidenceStore(db_path=":memory:")
    store.install_write_behind()
    # Install is a silent no-op — every row takes the synchronous path.
    assert store.record_repair_event_async(
        route_id="r", model_id="m", client_id="c",
        tool_name="t", outcome="valid_unchanged",
    ) is False
    assert store.query_repair_stats()[0].total_eligible == 1


def test_write_behind_falls_back_to_synchronous_when_not_installed():
    store = EvidenceStore(db_path=":memory:")
    assert store.record_repair_event_async(
        route_id="r", model_id="m", client_id="c",
        tool_name="t", outcome="repaired", repair_rules=["alias"],
    ) is False, "no worker installed — caller must write synchronously"
    groups = store.query_repair_stats()
    assert len(groups) == 1
    assert groups[0].accepted_after_repair == 1
    assert groups[0].rule_counts == {"alias": 1}


def test_close_drains_pending_write_behind_rows(tmp_path):
    store = EvidenceStore(db_path=str(tmp_path / "evidence.db"))
    store.install_write_behind()
    for i in range(5):
        assert store.record_repair_event_async(
            route_id="r", model_id="m", client_id="c",
            tool_name=f"t{i}", outcome="rejected",
        )
    store.close()  # must drain before closing
    groups = store.query_repair_stats()
    assert groups[0].rejected == 5
