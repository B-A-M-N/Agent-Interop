"""Bounded session store for virtualized client state.

This module holds the FULL client state (tool results, history fragments)
that Interop retains on behalf of the local model. The model never sees the
raw blobs — it sees bounded handles (refs) and retrieves slices through the
private ``__interop_read_result`` / ``__interop_recall_history`` /
``__interop_search_history`` tools.

Design
------
* Each stored blob gets an opaque, unguessable handle (ref).
* Refs are scoped to a session — cross-session access fails closed.
* The store is bounded: total bytes per session are capped; the oldest
  evictable entries are dropped when the cap is exceeded.
* Full content is retained byte-for-byte; virtualization is a VIEW, not a
  lossy transform. Nothing is discarded silently.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Lock
from typing import Any

_EXOTIC_BOUNDARIES_NON_ASCII = ("\x85", "\u2028", "\u2029")
_EXOTIC_BOUNDARIES_ASCII = ("\r", "\v", "\f", "\x1c", "\x1d", "\x1e")


def _has_exotic_line_boundaries(content: str) -> bool:
    """True when content contains a splitlines boundary other than "\n".

    Membership scans are C-level and short-circuit; on ASCII content only
    the six ASCII-range boundaries are possible, so the unicode scan is
    skipped entirely. A per-character Python loop or a unicode-class regex
    over a multi-hundred-kilobyte tool result costs more than the
    splitlines the index exists to avoid — the check must be effectively
    free for the indexed path to win.
    """
    if (
        "\r" in content or "\v" in content or "\f" in content
        or "\x1c" in content or "\x1d" in content or "\x1e" in content
    ):
        return True
    if not content.isascii():
        return any(sep in content for sep in _EXOTIC_BOUNDARIES_NON_ASCII)
    return False


@dataclass
class StoredEntry:
    """A single stored blob."""

    ref: str
    session_id: str
    kind: str  # "tool_result" | "history_fragment"
    content: str
    tool_call_id: str = ""
    tool_name: str = ""
    byte_size: int = 0
    sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    last_accessed_at: float = 0.0
    sequence: int = 0
    pinned_requests: set[str] = field(default_factory=set)
    # P1-G: 1-indexed character offset where each line starts, built lazily
    # on first get_slice and invalidated never (content is immutable after
    # insert). Turns repeated line paging from an O(content) split per read
    # into an O(lines-in-range) slice.
    _line_offsets: list[int] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = sha256(self.content.encode("utf-8", "replace")).hexdigest()
        if not self.byte_size:
            self.byte_size = len(self.content.encode("utf-8", "replace"))
        now = time.time()
        if not self.created_at:
            self.created_at = now
        if not self.last_accessed_at:
            self.last_accessed_at = now

    def line_offsets(self) -> list[int]:
        """Character offset of each line start (offsets[0] == 0)."""
        if self._line_offsets is None:
            offsets = [0]
            content = self.content
            find = content.find
            start = 0
            while True:
                idx = find("\n", start)
                if idx < 0:
                    break
                offsets.append(idx + 1)
                start = idx + 1
            self._line_offsets = offsets
        return self._line_offsets


@dataclass
class _SessionBucket:
    entries: dict[str, StoredEntry] = field(default_factory=dict)
    total_bytes: int = 0


class ContextStore:
    """Bounded, session-scoped store for virtualized client state.

    Thread-safe. Refs are opaque (256-bit random) so the model cannot forge
    access to another session's data.
    """

    def __init__(
        self,
        max_bytes_per_session: int = 2_000_000,
        max_total_bytes: int = 10_000_000,
        max_sessions: int = 100,
        max_entry_bytes: int = 0,
        ttl_seconds: float = 0.0,
    ) -> None:
        self._max_bytes = max_bytes_per_session
        self._max_total_bytes = max_total_bytes
        self._max_sessions = max_sessions
        # 0 means "no per-entry cap beyond the session/total caps"
        self._max_entry_bytes = max_entry_bytes or max_bytes_per_session
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, _SessionBucket] = {}
        self._lock = Lock()
        self._sequence_counter: int = 0
        self._dedup_index: dict[tuple[str, str, str, str], str] = {}  # (session_id, kind, tool_call_id, sha256) -> ref
        self._total_bytes: int = 0
        # P1-G: TTL sweep throttle (monotonic clock; last sweep time).
        self._last_sweep_at: float = 0.0

    # Minimum seconds between TTL sweeps. TTL precision of a few seconds is
    # irrelevant against multi-second eviction windows; scanning every entry
    # per request is pure lock-holding waste.
    _SWEEP_MIN_INTERVAL: float = 5.0

    def _bucket(self, session_id: str) -> _SessionBucket:
        if session_id not in self._sessions:
            self._ensure_session_capacity_locked()
            self._sessions[session_id] = _SessionBucket()
        return self._sessions[session_id]

    def _new_ref(self) -> str:
        return secrets.token_urlsafe(24)

    def _next_sequence(self) -> int:
        self._sequence_counter += 1
        return self._sequence_counter

    def _session_has_pinned_entries(self, bucket: _SessionBucket) -> bool:
        """Return True if any entry in this session is pinned by an active request."""
        return any(e.pinned_requests for e in bucket.entries.values())

    def _ensure_session_capacity_locked(self) -> None:
        """Make room for a new session without deadlocking.

        The lock is already held by the caller. We cannot call
        clear_session() (which re-acquires the lock), so we evict
        inline here. P0-7: only evict sessions with ZERO pinned entries.
        If every session has pinned state, raise ContextStoreCapacityError.
        """
        while len(self._sessions) >= self._max_sessions:
            if not self._sessions:
                return
            # P0-7: Find the oldest session with no pinned entries.
            # Never evict a session that has active pinned refs.
            evictable = [
                (sid, bucket)
                for sid, bucket in self._sessions.items()
                if not self._session_has_pinned_entries(bucket)
            ]
            if not evictable:
                # Every session has pinned state — cannot evict safely.
                from agent_interop.errors import ContextStoreCapacityError
                raise ContextStoreCapacityError(
                    f"Cannot create new session: max_sessions={self._max_sessions} "
                    "reached and all sessions have pinned entries"
                )
            oldest_session = min(
                evictable,
                key=lambda item: min(
                    (e.last_accessed_at for e in item[1].entries.values()),
                    default=0.0,
                ),
            )
            self._clear_session_locked(oldest_session[0])

    def _evict_oldest_session(self) -> None:
        """Evict the oldest session when max_sessions is exceeded.

        P0.16: Lock is held by the caller — must call the locked variant.
        P0-7: Only evict sessions with zero pinned entries.
        """
        if not self._sessions:
            return
        evictable = [
            (sid, bucket)
            for sid, bucket in self._sessions.items()
            if not self._session_has_pinned_entries(bucket)
        ]
        if not evictable:
            return  # All sessions have pinned state — cannot evict.
        oldest_session = min(
            evictable,
            key=lambda item: min(
                (e.last_accessed_at for e in item[1].entries.values()),
                default=0.0,
            ),
        )
        self._clear_session_locked(oldest_session[0])

    def _clear_session_locked(self, session_id: str) -> None:
        """Remove a session's entries and its dedup-index entries. Lock-free: the caller holds self._lock."""
        bucket = self._sessions.pop(session_id, None)
        if bucket is not None:
            self._total_bytes -= bucket.total_bytes
            for entry in bucket.entries.values():
                dedup_key = (entry.session_id, entry.kind, entry.tool_call_id, entry.sha256)
                self._dedup_index.pop(dedup_key, None)

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._clear_session_locked(session_id)

    def _cleanup_empty_sessions(self) -> None:
        """Remove sessions with no entries. Lock-free: caller holds self._lock."""
        empty = [sid for sid, bucket in self._sessions.items() if not bucket.entries]
        for sid in empty:
            self._sessions.pop(sid, None)

    def store(
        self,
        session_id: str,
        content: str,
        kind: str,
        tool_call_id: str = "",
        tool_name: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> StoredEntry:
        """Store a blob and return its handle. Deduplicates identical content.

        P0.17: refuses to store an entry larger than max_entry_bytes and
        verifies the ref survives eviction after insertion.
        """
        entry_size = len(content.encode("utf-8", "replace"))
        if entry_size > self._max_entry_bytes:
            from agent_interop.errors import ContextEntryTooLargeError
            raise ContextEntryTooLargeError(
                f"Entry {entry_size} bytes exceeds max_entry_bytes {self._max_entry_bytes}"
            )

        # P1-G: one encode + one hash over the content, shared by the size
        # check, the dedup key, and the entry itself (StoredEntry.__post_init__
        # skips both when pre-populated). The historical form encoded+hashed
        # the content twice before taking the lock.
        content_bytes = content.encode("utf-8", "replace")
        content_sha256 = sha256(content_bytes).hexdigest()

        with self._lock:
            dedup_key = (session_id, kind, tool_call_id, content_sha256)
            if dedup_key in self._dedup_index:
                existing_ref = self._dedup_index[dedup_key]
                bucket = self._sessions.get(session_id)
                if bucket and existing_ref in bucket.entries:
                    entry = bucket.entries[existing_ref]
                    entry.last_accessed_at = time.time()
                    return entry

            bucket = self._bucket(session_id)
            ref = self._new_ref()
            entry = StoredEntry(
                ref=ref,
                session_id=session_id,
                kind=kind,
                content=content,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                metadata=metadata or {},
                sequence=self._next_sequence(),
                byte_size=entry_size,
                sha256=content_sha256,
            )
            bucket.entries[ref] = entry
            bucket.total_bytes += entry.byte_size
            self._total_bytes += entry.byte_size
            self._dedup_index[dedup_key] = ref
            self._evict_if_needed(bucket)
            self._evict_global_if_needed()

            # P0.17: confirm the just-inserted ref still exists
            if ref not in bucket.entries:
                from agent_interop.errors import ContextEntryEvictedError
                raise ContextEntryEvictedError(
                    f"Entry {ref} was immediately evicted (size {entry_size}, session cap {self._max_bytes}, total cap {self._max_total_bytes})"
                )
            return entry

    def get(self, ref: str, session_id: str) -> StoredEntry | None:
        """Retrieve a blob by ref, scoped to the session. Cross-session fails closed."""
        with self._lock:
            bucket = self._sessions.get(session_id)
            if bucket is None:
                return None
            entry = bucket.entries.get(ref)
            if entry is not None:
                entry.last_accessed_at = time.time()
            return entry

    def get_slice(self, ref: str, session_id: str, start_line: int = 1, line_count: int | None = None) -> str | None:
        """Return a line-range slice of a stored blob.

        1-indexed. If ``line_count`` is None, return from ``start_line`` to end.

        P1-G: uses the entry's cached line-offset index — materializing the
        full ``splitlines`` list on every paged read charged O(total lines)
        per read even when the model asked for 40 lines of a 5000-line blob.
        Semantics match the historical form exactly (``splitlines`` splits
        on more boundaries than ``\\n`` — ``\\r\\n``/``\\x0b``/``\\x0c`` etc.).
        """
        entry = self.get(ref, session_id)
        if entry is None:
            return None
        content = entry.content
        # Historical semantics: splitlines() treats every universal newline
        # as a boundary, not just "\n". The offset index records only "\n"
        # boundaries, so fall back to the exact historical behavior when the
        # content contains any other boundary kind (\r, \v, \f, and the
        # unicode line separators). The common case (plain "\n" tool
        # output) takes the O(range) indexed path.
        if _has_exotic_line_boundaries(content):
            lines = content.splitlines(keepends=True)
            start = max(0, start_line - 1)
            if line_count is not None:
                end = start + line_count
                return "".join(lines[start:end])
            return "".join(lines[start:])
        offsets = entry.line_offsets()
        total_lines = len(offsets)
        start_idx = max(0, start_line - 1)
        if start_idx >= total_lines:
            return ""
        if line_count is None:
            return content[offsets[start_idx]:]
        end_idx = min(start_idx + line_count, total_lines)
        if start_idx == end_idx:
            return ""
        # Slice from the first line's start to the LAST line's end. The last
        # line runs to the start of the following line (or content end).
        end_offset = (
            offsets[end_idx] if end_idx < total_lines else len(content)
        )
        return content[offsets[start_idx]:end_offset]

    def search(self, session_id: str, query: str, max_results: int = 5) -> list[StoredEntry]:
        """Simple substring search across a session's stored blobs.

        P0.20: update last_accessed_at so LRU eviction reflects actual search use.
        """
        with self._lock:
            bucket = self._sessions.get(session_id)
            if bucket is None:
                return []
            query_lower = query.lower()
            results: list[tuple[int, StoredEntry]] = []
            for entry in bucket.entries.values():
                idx = entry.content.lower().find(query_lower)
                if idx >= 0:
                    entry.last_accessed_at = time.time()
                    results.append((idx, entry))
            results.sort(key=lambda t: t[0])
            return [e for _, e in results[:max_results]]

    def _evict_if_needed(self, bucket: _SessionBucket) -> None:
        """Drop oldest entries (by last_accessed_at) until under the byte cap.
        Pinned entries are never evicted. Lock-free: caller holds self._lock."""
        if bucket.total_bytes <= self._max_bytes:
            return
        ordered = sorted(
            (e for e in bucket.entries.values() if not e.pinned_requests),
            key=lambda e: e.last_accessed_at,
        )
        for entry in ordered:
            if bucket.total_bytes <= self._max_bytes:
                return
            bucket.total_bytes -= entry.byte_size
            self._total_bytes -= entry.byte_size
            bucket.entries.pop(entry.ref, None)
            dedup_key = (entry.session_id, entry.kind, entry.tool_call_id, entry.sha256)
            self._dedup_index.pop(dedup_key, None)

    def _evict_global_if_needed(self) -> None:
        """Evict oldest entries globally when total bytes exceed the cap.
        Lock-free: caller holds self._lock."""
        if self._total_bytes <= self._max_total_bytes:
            return
        all_entries: list[tuple[float, str, str]] = []  # (last_accessed_at, session_id, ref)
        for session_id, bucket in self._sessions.items():
            for entry in bucket.entries.values():
                if not entry.pinned_requests:
                    all_entries.append((entry.last_accessed_at, session_id, entry.ref))
        all_entries.sort(key=lambda t: t[0])
        for _, session_id, ref in all_entries:
            if self._total_bytes <= self._max_total_bytes:
                return
            target = self._sessions.get(session_id)
            if target and ref in target.entries:
                entry = target.entries[ref]
                target.total_bytes -= entry.byte_size
                self._total_bytes -= entry.byte_size
                target.entries.pop(ref, None)
                dedup_key = (entry.session_id, entry.kind, entry.tool_call_id, entry.sha256)
                self._dedup_index.pop(dedup_key, None)
        self._cleanup_empty_sessions()

    def pin_ref(self, ref: str, request_id: str) -> bool:
        """Pin a ref to prevent eviction. Returns True if successful."""
        with self._lock:
            for bucket in self._sessions.values():
                if ref in bucket.entries:
                    bucket.entries[ref].pinned_requests.add(request_id)
                    return True
            return False

    def unpin_ref(self, ref: str, request_id: str) -> bool:
        """Unpin a ref. Returns True if successful."""
        with self._lock:
            for bucket in self._sessions.values():
                if ref in bucket.entries:
                    bucket.entries[ref].pinned_requests.discard(request_id)
                    return True
            return False

    def pin_refs(
        self,
        refs: Iterable[str],
        request_id: str,
        session_id: str | None = None,
    ) -> int:
        """Pin every ref in one lock acquisition.

        P1-G: a request that virtualized several results pinned them one
        lock round-trip each; under concurrency that is a contended
        lock acquire per ref. Returns the count actually pinned.

        When ``session_id`` is given, only that session's bucket is
        consulted — a cross-session ref cannot legitimately be pinned by
        this request, so it is skipped instead of scanned for. Omitting
        ``session_id`` preserves the historical scan-all-sessions behavior
        for callers that genuinely do not know the owning session.
        """
        pinned = 0
        wanted = list(refs)
        if not wanted:
            return 0
        with self._lock:
            if session_id is not None:
                buckets = [
                    bucket for bucket in (self._sessions.get(session_id),)
                    if bucket is not None
                ]
            else:
                buckets = list(self._sessions.values())
            for bucket in buckets:
                for ref in wanted:
                    entry = bucket.entries.get(ref)
                    if entry is not None:
                        entry.pinned_requests.add(request_id)
                        pinned += 1
            return pinned

    def unpin_refs(
        self,
        refs: Iterable[str],
        request_id: str,
        session_id: str | None = None,
    ) -> int:
        """Unpin every ref in one lock acquisition. Returns count unpinned.

        Accepts the same optional ``session_id`` narrowing as
        :meth:`pin_refs`; the session must match for the unpin to find the
        entry, which also keeps a request from unpinning another session's
        identically-named ref.
        """
        unpinned = 0
        wanted = list(refs)
        if not wanted:
            return 0
        with self._lock:
            if session_id is not None:
                buckets = [
                    bucket for bucket in (self._sessions.get(session_id),)
                    if bucket is not None
                ]
            else:
                buckets = list(self._sessions.values())
            for bucket in buckets:
                for ref in wanted:
                    entry = bucket.entries.get(ref)
                    if entry is not None:
                        entry.pinned_requests.discard(request_id)
                        unpinned += 1
            return unpinned

    def evict_expired(self, now: float | None = None) -> int:
        """P1.1: remove entries older than TTL. Returns count evicted.

        P1-G: the request lifecycle is the TTL hook (the gateway calls this
        once per request), so a no-op TTL store must not touch the lock at
        all, and a sweep must not run more often than the throttle window —
        a busy server otherwise re-scans every entry on every request while
        nothing has expired.
        """
        if not self._ttl_seconds:
            return 0
        now = now or time.time()
        if now - self._last_sweep_at < self._SWEEP_MIN_INTERVAL:
            return 0
        self._last_sweep_at = now
        expired: list[tuple[str, str]] = []  # (session_id, ref)
        with self._lock:
            for session_id, bucket in self._sessions.items():
                for entry in bucket.entries.values():
                    if entry.pinned_requests:
                        continue
                    if now - entry.last_accessed_at > self._ttl_seconds:
                        expired.append((session_id, entry.ref))
            for session_id, ref in expired:
                target = self._sessions.get(session_id)
                if target and ref in target.entries:
                    entry = target.entries[ref]
                    target.total_bytes -= entry.byte_size
                    self._total_bytes -= entry.byte_size
                    target.entries.pop(ref, None)
                    dedup_key = (entry.session_id, entry.kind, entry.tool_call_id, entry.sha256)
                    self._dedup_index.pop(dedup_key, None)
            self._cleanup_empty_sessions()
        return len(expired)
