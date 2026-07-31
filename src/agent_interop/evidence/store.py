"""Evidence store — persist compatibility evidence using SQLite.

Stores exact compatibility tuple records, profile revisions, test-suite
revisions, captured sanitized failure cases, repair outcomes, and
benchmark aggregates.

Static YAML provides initial declarations. Verified evidence overrides
declarations only for an exact matching tuple.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from agent_interop.replay.types import (
    CompatibilityKey,
    CompatibilityQuirk,
    CompatibilityResult,
    EvidencePassFailBreakdown,
)

logger = logging.getLogger("agent_interop.evidence")


@dataclass(frozen=True)
class RepairStatsGroup:
    """Aggregated `interop repair stats` numbers for one (route, model,
    client) combination.

    ``total_eligible`` is every tool-call decision the repair pipeline
    evaluated (matches the existing ``candidate_count`` semantics on
    ``CompatibilityResult``). The other counts partition it:
    accepted_without_repair + accepted_after_repair + rejected ==
    total_eligible. ``rejected_with_partial_repair`` is a SUBSET of
    ``rejected`` (not an additional partition) — a rejected call where at
    least one repair rule fired and made SOME progress before the call
    still ultimately failed overall validation.
    """

    route_id: str = ""
    model_id: str = ""
    client_id: str = ""
    total_eligible: int = 0
    accepted_without_repair: int = 0
    accepted_after_repair: int = 0
    rejected: int = 0
    rejected_with_partial_repair: int = 0
    rule_counts: dict[str, int] = field(default_factory=dict)

# Schema version for migrations
_SCHEMA_VERSION = 8

# Columns added in schema version 2
_V2_COLUMNS = [
    ("tool_schema_fingerprint", "TEXT DEFAULT ''"),
    ("streaming", "INTEGER DEFAULT 0"),
    ("effective_tool_mode", "TEXT DEFAULT ''"),
    ("parser_id", "TEXT DEFAULT ''"),
    ("template_revision", "TEXT DEFAULT ''"),
    ("backend_serving_config", "TEXT DEFAULT ''"),
]

# Columns added in schema version 3 — evidence lifecycle
_V3_COLUMNS = [
    ("created_at_iso", "TEXT DEFAULT ''"),
    ("last_verified_at", "TEXT DEFAULT ''"),
    ("passes_expiry_hours", "INTEGER DEFAULT 720"),
    ("manually_verified", "INTEGER DEFAULT 0"),
    ("revoked", "INTEGER DEFAULT 0"),
    ("revocation_reason", "TEXT DEFAULT ''"),
    ("pf_total_samples", "INTEGER DEFAULT 0"),
    ("pf_tool_selection_pass", "INTEGER DEFAULT 0"),
    ("pf_tool_selection_fail", "INTEGER DEFAULT 0"),
    ("pf_valid_call_pass", "INTEGER DEFAULT 0"),
    ("pf_valid_call_fail", "INTEGER DEFAULT 0"),
    ("pf_task_completion_pass", "INTEGER DEFAULT 0"),
    ("pf_task_completion_fail", "INTEGER DEFAULT 0"),
    ("pf_streaming_equivalent_pass", "INTEGER DEFAULT 0"),
    ("pf_streaming_equivalent_fail", "INTEGER DEFAULT 0"),
    ("pf_history_round_trip_pass", "INTEGER DEFAULT 0"),
    ("pf_history_round_trip_fail", "INTEGER DEFAULT 0"),
]

# Columns added in schema version 4 — counter-based rate aggregation
_V4_COLUMNS = [
    ("last_observed_at", "TEXT DEFAULT ''"),
    ("no_selection_request_count", "INTEGER DEFAULT 0"),
    ("candidate_count", "INTEGER DEFAULT 0"),
    ("valid_unchanged_count", "INTEGER DEFAULT 0"),
    ("repaired_count", "INTEGER DEFAULT 0"),
    ("regenerated_count", "INTEGER DEFAULT 0"),
    ("accepted_count", "INTEGER DEFAULT 0"),
    ("rejected_count", "INTEGER DEFAULT 0"),
]

# Columns added in schema version 5 — recorded reviewer attestation
_V5_COLUMNS = [
    ("attestation", "TEXT DEFAULT ''"),
]

# Columns added in schema version 6 — conformance-level battery provenance
_V6_COLUMNS = [
    ("battery_version", "TEXT DEFAULT ''"),
]

# Schema v8 stores the complete immutable key representation. Earlier
# columns remain queryable indexes, while this prevents new key dimensions
# from being silently dropped during round-trip reconstruction.
_V8_COLUMNS = [
    ("key_json", "TEXT DEFAULT ''"),
]


class EvidenceStore:
    """SQLite-backed store for compatibility evidence."""

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            from agent_interop.paths import evidence_file

            db_path = str(evidence_file())
        self._db_path = db_path
        self._lock = threading.Lock()
        self._local = threading.local()
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Create the database and tables if they don't exist."""
        db_dir = os.path.dirname(self._db_path)
        if db_dir and db_dir != ":memory:":
            os.makedirs(db_dir, exist_ok=True)
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS compatibility_results (
                    id TEXT PRIMARY KEY,
                    client_id TEXT,
                    client_version TEXT,
                    client_protocol TEXT,
                    model_id TEXT,
                    model_digest TEXT,
                    quantization TEXT,
                    backend_kind TEXT,
                    backend_version TEXT,
                    upstream_protocol TEXT,
                    chat_template_digest TEXT,
                    profile_id TEXT,
                    profile_revision TEXT,
                    tool_schema_fingerprint TEXT,
                    streaming INTEGER,
                    effective_tool_mode TEXT,
                    parser_id TEXT,
                    template_revision TEXT,
                    backend_serving_config TEXT,
                    tested_at TEXT,
                    sample_count INTEGER,
                    tool_selection_rate REAL,
                    valid_call_rate_before_repair REAL,
                    valid_call_rate_after_repair REAL,
                    task_completion_rate REAL,
                    deterministic_repair_rate REAL,
                    regeneration_rate REAL,
                    rejection_rate REAL,
                    streaming_equivalent INTEGER,
                    history_round_trip_valid INTEGER,
                    verified_capabilities TEXT,
                    known_quirks TEXT,
                    created_at REAL,
                    created_at_iso TEXT,
                    last_verified_at TEXT,
                    passes_expiry_hours INTEGER DEFAULT 720,
                    manually_verified INTEGER DEFAULT 0,
                    revoked INTEGER DEFAULT 0,
                    revocation_reason TEXT DEFAULT '',
                    pf_total_samples INTEGER DEFAULT 0,
                    pf_tool_selection_pass INTEGER DEFAULT 0,
                    pf_tool_selection_fail INTEGER DEFAULT 0,
                    pf_valid_call_pass INTEGER DEFAULT 0,
                    pf_valid_call_fail INTEGER DEFAULT 0,
                    pf_task_completion_pass INTEGER DEFAULT 0,
                    pf_task_completion_fail INTEGER DEFAULT 0,
                    pf_streaming_equivalent_pass INTEGER DEFAULT 0,
                    pf_streaming_equivalent_fail INTEGER DEFAULT 0,
                    pf_history_round_trip_pass INTEGER DEFAULT 0,
                    pf_history_round_trip_fail INTEGER DEFAULT 0,
                    last_observed_at TEXT DEFAULT '',
                    no_selection_request_count INTEGER DEFAULT 0,
                    candidate_count INTEGER DEFAULT 0,
                    valid_unchanged_count INTEGER DEFAULT 0,
                    repaired_count INTEGER DEFAULT 0,
                    regenerated_count INTEGER DEFAULT 0,
                    accepted_count INTEGER DEFAULT 0,
                    rejected_count INTEGER DEFAULT 0,
                    attestation TEXT DEFAULT '',
                    battery_version TEXT DEFAULT '',
                    key_json TEXT DEFAULT ''
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS failure_cases (
                    id TEXT PRIMARY KEY,
                    result_id TEXT,
                    case_data TEXT,
                    created_at REAL,
                    FOREIGN KEY (result_id) REFERENCES compatibility_results(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                """
            )
            # Schema v7 — one row per tool-call decision, feeding
            # `interop repair stats`. Deliberately a separate, append-only
            # table from compatibility_results (which stores a single
            # merged-counter row per compatibility tuple, with no
            # per-rule or time-windowed breakdown at all): route/model/
            # client grouping, per-rule-type breakdown, and --since
            # filtering all require real per-event rows, not an
            # aggregate counter.
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS repair_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    timestamp_iso TEXT,
                    route_id TEXT DEFAULT '',
                    model_id TEXT DEFAULT '',
                    client_id TEXT DEFAULT '',
                    tool_name TEXT DEFAULT '',
                    outcome TEXT DEFAULT '',
                    repair_rules TEXT DEFAULT '[]'
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_repair_events_lookup "
                "ON repair_events (route_id, model_id, client_id, timestamp)"
            )
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )

            # Migrate existing databases: add columns introduced in V2/V3
            existing_cols = {
                row[1] for row in
                conn.execute("PRAGMA table_info(compatibility_results)").fetchall()
            }
            for col_name, col_def in _V2_COLUMNS:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE compatibility_results ADD COLUMN {col_name} {col_def}"
                    )
                    logger.info("Migrated evidence DB: added column %s", col_name)
            for col_name, col_def in _V3_COLUMNS:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE compatibility_results ADD COLUMN {col_name} {col_def}"
                    )
                    logger.info("Migrated evidence DB: added column %s", col_name)
            for col_name, col_def in _V4_COLUMNS:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE compatibility_results ADD COLUMN {col_name} {col_def}"
                    )
                    logger.info("Migrated evidence DB: added column %s", col_name)
            for col_name, col_def in _V5_COLUMNS:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE compatibility_results ADD COLUMN {col_name} {col_def}"
                    )
                    logger.info("Migrated evidence DB: added column %s", col_name)
            for col_name, col_def in _V6_COLUMNS:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE compatibility_results ADD COLUMN {col_name} {col_def}"
                    )
                    logger.info("Migrated evidence DB: added column %s", col_name)
            for col_name, col_def in _V8_COLUMNS:
                if col_name not in existing_cols:
                    conn.execute(
                        f"ALTER TABLE compatibility_results ADD COLUMN {col_name} {col_def}"
                    )
                    logger.info("Migrated evidence DB: added column %s", col_name)

    @contextmanager
    def _connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        try:
            yield self._local.conn
            self._local.conn.commit()
        except Exception:
            self._local.conn.rollback()
            raise

    def _make_result_id(self, key: CompatibilityKey) -> str:
        """Generate a deterministic ID from a compatibility key."""
        import hashlib

        content = json.dumps(asdict(key), sort_keys=True, separators=(",", ":"), default=str)
        return f"res_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    def store_result(
        self,
        key: CompatibilityKey,
        result: CompatibilityResult,
        failure_cases: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        """Store a compatibility result for an exact tuple.

        Returns the result ID.
        """
        result_id = self._make_result_id(key)

        columns = (
            "id", "client_id", "client_version", "client_protocol",
            "model_id", "model_digest", "quantization",
            "backend_kind", "backend_version", "upstream_protocol",
            "chat_template_digest", "profile_id", "profile_revision",
            "tool_schema_fingerprint", "streaming", "effective_tool_mode",
            "parser_id", "template_revision", "backend_serving_config",
            "tested_at", "sample_count",
            "tool_selection_rate", "valid_call_rate_before_repair",
            "valid_call_rate_after_repair", "task_completion_rate",
            "deterministic_repair_rate", "regeneration_rate",
            "rejection_rate", "streaming_equivalent",
            "history_round_trip_valid", "verified_capabilities",
            "known_quirks", "created_at",
            "created_at_iso", "last_verified_at", "passes_expiry_hours",
            "manually_verified", "revoked", "revocation_reason",
            "pf_total_samples", "pf_tool_selection_pass", "pf_tool_selection_fail",
            "pf_valid_call_pass", "pf_valid_call_fail",
            "pf_task_completion_pass", "pf_task_completion_fail",
            "pf_streaming_equivalent_pass", "pf_streaming_equivalent_fail",
            "pf_history_round_trip_pass", "pf_history_round_trip_fail",
            "last_observed_at",
            "no_selection_request_count", "candidate_count",
            "valid_unchanged_count", "repaired_count", "regenerated_count",
            "accepted_count", "rejected_count", "attestation", "battery_version", "key_json",
        )
        # Serialize pass/fail breakdown
        pf = result.pass_fail_breakdown
        pf_values: tuple[int, ...] = (
            (pf.total_samples, pf.tool_selection_pass, pf.tool_selection_fail,
             pf.valid_call_pass, pf.valid_call_fail,
             pf.task_completion_pass, pf.task_completion_fail,
             pf.streaming_equivalent_pass, pf.streaming_equivalent_fail,
             pf.history_round_trip_pass, pf.history_round_trip_fail)
            if pf is not None else (0,) * 11
        )
        values = (
            result_id,
            key.client_id, key.client_version, key.client_protocol,
            key.model_id, key.model_digest, key.quantization,
            key.backend_kind, key.backend_version, key.upstream_protocol,
            key.chat_template_digest, key.profile_id, key.profile_revision,
            key.tool_schema_fingerprint, int(key.streaming), key.effective_tool_mode,
            key.parser_id, key.template_revision, key.backend_serving_config,
            result.tested_at,
            result.sample_count,
            result.tool_selection_rate,
            result.valid_call_rate_before_repair,
            result.valid_call_rate_after_repair,
            result.task_completion_rate,
            result.deterministic_repair_rate,
            result.regeneration_rate,
            result.rejection_rate,
            int(result.streaming_equivalent),
            int(result.history_round_trip_valid),
            json.dumps(list(result.verified_capabilities)),
            json.dumps([q.__dict__ for q in result.known_quirks]),
            time.time(),
            result.created_at,
            result.last_verified_at,
            result.passes_expiry_hours,
            int(result.manually_verified),
            int(result.revoked),
            result.revocation_reason,
            *pf_values,
            result.last_observed_at,
            result.no_selection_request_count,
            result.candidate_count,
            result.valid_unchanged_count,
            result.repaired_count,
            result.regenerated_count,
            result.accepted_count,
            result.rejected_count,
            result.attestation,
            result.battery_version,
            json.dumps(asdict(key), sort_keys=True, default=str),
        )

        assert len(columns) == len(values), (
            f"Column/value count mismatch: {len(columns)} columns, {len(values)} values"
        )

        column_sql = ", ".join(columns)
        placeholders = ", ".join("?" for _ in columns)

        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    f"INSERT OR REPLACE INTO compatibility_results ({column_sql}) VALUES ({placeholders})",
                    values,
                )

                # Store associated failure cases
                for case in failure_cases:
                    case_id = f"fc_{hash(json.dumps(case, sort_keys=True, default=str)) % (2**32):08x}"
                    conn.execute(
                        "INSERT OR REPLACE INTO failure_cases VALUES (?, ?, ?, ?)",
                        (case_id, result_id, json.dumps(case), time.time()),
                    )

        logger.info("Stored compatibility result: %s", result_id)
        return result_id

    def get_result(self, key: CompatibilityKey) -> CompatibilityResult | None:
        """Retrieve the latest result for an exact compatibility tuple."""
        result_id = self._make_result_id(key)

        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM compatibility_results WHERE id = ?",
                (result_id,),
            ).fetchone()

        if row is None:
            return None

        # Deserialize pass/fail breakdown
        pass_fail = EvidencePassFailBreakdown(
            total_samples=row["pf_total_samples"],
            tool_selection_pass=row["pf_tool_selection_pass"],
            tool_selection_fail=row["pf_tool_selection_fail"],
            valid_call_pass=row["pf_valid_call_pass"],
            valid_call_fail=row["pf_valid_call_fail"],
            task_completion_pass=row["pf_task_completion_pass"],
            task_completion_fail=row["pf_task_completion_fail"],
            streaming_equivalent_pass=row["pf_streaming_equivalent_pass"],
            streaming_equivalent_fail=row["pf_streaming_equivalent_fail"],
            history_round_trip_pass=row["pf_history_round_trip_pass"],
            history_round_trip_fail=row["pf_history_round_trip_fail"],
        )

        return CompatibilityResult(
            tested_at=row["tested_at"],
            sample_count=row["sample_count"],
            tool_selection_rate=row["tool_selection_rate"],
            valid_call_rate_before_repair=row["valid_call_rate_before_repair"],
            valid_call_rate_after_repair=row["valid_call_rate_after_repair"],
            task_completion_rate=row["task_completion_rate"],
            deterministic_repair_rate=row["deterministic_repair_rate"],
            regeneration_rate=row["regeneration_rate"],
            rejection_rate=row["rejection_rate"],
            streaming_equivalent=bool(row["streaming_equivalent"]),
            history_round_trip_valid=bool(row["history_round_trip_valid"]),
            verified_capabilities=frozenset(json.loads(row["verified_capabilities"])),
            known_quirks=tuple(
                CompatibilityQuirk(**q) for q in json.loads(row["known_quirks"])
            ),
            created_at=row["created_at_iso"],
            last_verified_at=row["last_verified_at"],
            passes_expiry_hours=row["passes_expiry_hours"],
            manually_verified=bool(row["manually_verified"]),
            revoked=bool(row["revoked"]),
            revocation_reason=row["revocation_reason"],
            last_observed_at=row["last_observed_at"],
            no_selection_request_count=row["no_selection_request_count"],
            candidate_count=row["candidate_count"],
            valid_unchanged_count=row["valid_unchanged_count"],
            repaired_count=row["repaired_count"],
            regenerated_count=row["regenerated_count"],
            accepted_count=row["accepted_count"],
            rejected_count=row["rejected_count"],
            attestation=row["attestation"],
            battery_version=row["battery_version"],
            pass_fail_breakdown=pass_fail,
        )

    def query_results(
        self,
        *,
        model_id: str | None = None,
        backend_kind: str | None = None,
        profile_id: str | None = None,
    ) -> list[tuple[CompatibilityKey, CompatibilityResult]]:
        """Query stored results with optional filters."""
        query = "SELECT * FROM compatibility_results WHERE 1=1"
        params: list[Any] = []

        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if backend_kind:
            query += " AND backend_kind = ?"
            params.append(backend_kind)
        if profile_id:
            query += " AND profile_id = ?"
            params.append(profile_id)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            key_data = json.loads(row["key_json"]) if row["key_json"] else {}
            key = CompatibilityKey(**key_data) if key_data else CompatibilityKey(
                client_id=row["client_id"],
                client_version=row["client_version"],
                client_protocol=row["client_protocol"],
                model_id=row["model_id"],
                model_digest=row["model_digest"],
                quantization=row["quantization"],
                backend_kind=row["backend_kind"],
                backend_version=row["backend_version"],
                upstream_protocol=row["upstream_protocol"],
                chat_template_digest=row["chat_template_digest"],
                profile_id=row["profile_id"],
                profile_revision=row["profile_revision"],
                tool_schema_fingerprint=row["tool_schema_fingerprint"],
                streaming=bool(row["streaming"]),
                effective_tool_mode=row["effective_tool_mode"],
                parser_id=row["parser_id"],
                template_revision=row["template_revision"],
                backend_serving_config=row["backend_serving_config"],
            )
            # Deserialize pass/fail breakdown
            pf = EvidencePassFailBreakdown(
                total_samples=row["pf_total_samples"],
                tool_selection_pass=row["pf_tool_selection_pass"],
                tool_selection_fail=row["pf_tool_selection_fail"],
                valid_call_pass=row["pf_valid_call_pass"],
                valid_call_fail=row["pf_valid_call_fail"],
                task_completion_pass=row["pf_task_completion_pass"],
                task_completion_fail=row["pf_task_completion_fail"],
                streaming_equivalent_pass=row["pf_streaming_equivalent_pass"],
                streaming_equivalent_fail=row["pf_streaming_equivalent_fail"],
                history_round_trip_pass=row["pf_history_round_trip_pass"],
                history_round_trip_fail=row["pf_history_round_trip_fail"],
            )
            result = CompatibilityResult(
                tested_at=row["tested_at"],
                sample_count=row["sample_count"],
                tool_selection_rate=row["tool_selection_rate"],
                valid_call_rate_before_repair=row["valid_call_rate_before_repair"],
                valid_call_rate_after_repair=row["valid_call_rate_after_repair"],
                task_completion_rate=row["task_completion_rate"],
                deterministic_repair_rate=row["deterministic_repair_rate"],
                regeneration_rate=row["regeneration_rate"],
                rejection_rate=row["rejection_rate"],
                streaming_equivalent=bool(row["streaming_equivalent"]),
                history_round_trip_valid=bool(row["history_round_trip_valid"]),
                verified_capabilities=frozenset(json.loads(row["verified_capabilities"])),
                known_quirks=tuple(
                    CompatibilityQuirk(**q) for q in json.loads(row["known_quirks"])
                ),
                created_at=row["created_at_iso"],
                last_verified_at=row["last_verified_at"],
                passes_expiry_hours=row["passes_expiry_hours"],
                manually_verified=bool(row["manually_verified"]),
                revoked=bool(row["revoked"]),
                revocation_reason=row["revocation_reason"],
                last_observed_at=row["last_observed_at"],
                no_selection_request_count=row["no_selection_request_count"],
                candidate_count=row["candidate_count"],
                valid_unchanged_count=row["valid_unchanged_count"],
                repaired_count=row["repaired_count"],
                regenerated_count=row["regenerated_count"],
                accepted_count=row["accepted_count"],
                rejected_count=row["rejected_count"],
                attestation=row["attestation"],
                battery_version=row["battery_version"],
                pass_fail_breakdown=pf,
            )
            results.append((key, result))

        return results

    # ── Evidence lifecycle operations ────────────────────────────────────

    def is_stale(self, key: CompatibilityKey) -> bool:
        """Check whether stored evidence has exceeded its expiry window.

        Returns True if the evidence is missing, expired, or revoked.

        Staleness for TRUST purposes is measured from ``last_verified_at``
        first — this timestamp is set ONLY by manual verification /
        certification (``mark_verified`` / ``interop certify``) and never by
        live traffic. Falling back to ``tested_at`` then ``created_at``
        handles older or automated-only records. Because live write-back
        never touches ``last_verified_at``, ordinary traffic cannot keep a
        once-certified record looking freshly-verified forever.
        """
        result = self.get_result(key)
        if result is None:
            return True
        if result.revoked:
            return True
        from datetime import datetime

        reference = result.last_verified_at or result.tested_at or result.created_at
        if not reference:
            return True
        try:
            tested = datetime.fromisoformat(reference)
            now = datetime.now(UTC)
            if tested.tzinfo is None:
                tested = tested.replace(tzinfo=UTC)
            age_hours = (now - tested).total_seconds() / 3600.0
        except (ValueError, OverflowError):
            return True
        return age_hours > result.passes_expiry_hours

    def mark_verified(self, key: CompatibilityKey, attestation: str = "") -> None:
        """Mark evidence as manually verified, updating last_verified_at.

        ``attestation`` records the reviewer's note for why this exact
        compatibility tuple was approved — surfaced by ``interop evidence
        show`` so a later reader knows what was actually checked, not just
        that a boolean got flipped.
        """
        from dataclasses import replace
        from datetime import datetime

        result = self.get_result(key)
        if result is None:
            return
        now = datetime.now(UTC).isoformat()
        updated = replace(
            result,
            last_verified_at=now,
            manually_verified=True,
            attestation=attestation or result.attestation,
        )
        self.store_result(key, updated)

    def revoke(self, key: CompatibilityKey, reason: str) -> None:
        """Revoke evidence, marking it untrusted with a reason."""
        from dataclasses import replace

        result = self.get_result(key)
        if result is None:
            return
        updated = replace(result, revoked=True, revocation_reason=reason)
        self.store_result(key, updated)

    def unrevoke(self, key: CompatibilityKey) -> None:
        """Restore previously-revoked evidence."""
        from dataclasses import replace

        result = self.get_result(key)
        if result is None:
            return
        updated = replace(result, revoked=False, revocation_reason="")
        self.store_result(key, updated)

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # ── Repair stats (schema v7) ────────────────────────────────────────

    def record_repair_event(
        self,
        *,
        route_id: str,
        model_id: str,
        client_id: str,
        tool_name: str,
        outcome: str,
        repair_rules: Sequence[str] = (),
    ) -> None:
        """Persist one row per tool-call decision — the source of truth
        for `interop repair stats`. ``outcome`` is one of
        "valid_unchanged" | "repaired" | "regenerated" | "rejected".

        Never raises: a persistence failure here must not break the
        request that triggered it, mirroring
        Gateway._record_evidence_observation's own swallow-and-log
        discipline at the call site.
        """
        now = time.time()
        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    "INSERT INTO repair_events "
                    "(timestamp, timestamp_iso, route_id, model_id, client_id, "
                    "tool_name, outcome, repair_rules) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        now,
                        datetime.fromtimestamp(now, UTC).isoformat(),
                        route_id, model_id, client_id, tool_name, outcome,
                        json.dumps(list(repair_rules)),
                    ),
                )

    def query_repair_stats(
        self,
        *,
        route_id: str | None = None,
        model_id: str | None = None,
        client_id: str | None = None,
        since: str | None = None,
    ) -> list[RepairStatsGroup]:
        """Aggregate recorded repair events, grouped by (route, model,
        client) — every filter is optional and independently composable.
        ``since`` is an ISO-8601 timestamp lower bound (inclusive).

        Never collapses distinct (route, model, client) combinations into
        one row — same "never silently pick one record" discipline as
        `interop evidence list` — so a caller filtering by only
        ``route_id`` gets one group per distinct model/client pair
        observed for that route, not a merged total.
        """
        query = (
            "SELECT route_id, model_id, client_id, outcome, repair_rules "
            "FROM repair_events WHERE 1=1"
        )
        params: list[Any] = []
        if route_id:
            query += " AND route_id = ?"
            params.append(route_id)
        if model_id:
            query += " AND model_id = ?"
            params.append(model_id)
        if client_id:
            query += " AND client_id = ?"
            params.append(client_id)
        if since:
            query += " AND timestamp_iso >= ?"
            params.append(since)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()

        acc: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (row["route_id"], row["model_id"], row["client_id"])
            g = acc.setdefault(key, {
                "total_eligible": 0,
                "accepted_without_repair": 0,
                "accepted_after_repair": 0,
                "rejected": 0,
                "rejected_with_partial_repair": 0,
                "rule_counts": {},
            })
            g["total_eligible"] += 1
            outcome = row["outcome"]
            try:
                rules = json.loads(row["repair_rules"] or "[]")
            except (json.JSONDecodeError, ValueError):
                rules = []
            if outcome == "valid_unchanged":
                g["accepted_without_repair"] += 1
            elif outcome in ("repaired", "regenerated"):
                g["accepted_after_repair"] += 1
            elif outcome == "rejected":
                g["rejected"] += 1
                if rules:
                    g["rejected_with_partial_repair"] += 1
            for rule in rules:
                g["rule_counts"][rule] = g["rule_counts"].get(rule, 0) + 1

        return [
            RepairStatsGroup(route_id=k[0], model_id=k[1], client_id=k[2], **v)
            for k, v in sorted(acc.items())
        ]


def capability_source(result: CompatibilityResult) -> str:
    """Classify what kind of evidence a stored CompatibilityResult
    represents: "revoked" | "stale" | "manually_approved" | "observed".

    Derived from CompatibilityResult's existing lifecycle fields rather
    than stored as its own column — there is exactly one place that can
    disagree with itself, matching this dataclass's existing "rates are
    DERIVED from counters" design. "declared" (the static /v1/capabilities
    profile-metadata block) is a separate code path's own label — it is
    never backed by a CompatibilityResult, so it can't come from here.

    Precedence: revoked overrides everything (untrusted regardless of
    anything else); a battery_version mismatch overrides manual approval
    (an approval made against a since-changed test battery no longer
    means what it claimed to); manually_approved overrides plain
    "observed" (a human explicitly attested this exact tuple).
    """
    from agent_interop.testing.levels import BATTERY_VERSION

    if result.revoked:
        return "revoked"
    if result.battery_version and result.battery_version != BATTERY_VERSION:
        return "stale"
    if result.manually_verified:
        return "manually_approved"
    return "observed"


# ─── Module-level default ──────────────────────────────────────────────────

_default_store: EvidenceStore | None = None


def get_default_store() -> EvidenceStore:
    global _default_store
    if _default_store is None:
        _default_store = EvidenceStore()
    return _default_store


def close_default_store() -> None:
    """Close the module-level default store, if open."""
    global _default_store
    if _default_store is not None:
        try:
            _default_store.close()
        except Exception:
            pass
        _default_store = None
