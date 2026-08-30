"""Durable, bounded bootstrap-qualification state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agent_interop.qualification.state import ProbeOutcome, QualificationRecord, QualificationState


def _serialize_record(record: QualificationRecord) -> dict[str, Any]:
    """P0.29: Serialize record with ProbeOutcome enum and tested_probes set."""
    return {
        "model_digest": record.model_digest,
        "state": record.state.value,
        "native_forced_tool": record.native_forced_tool.value,
        "prompted_forced_tool": record.prompted_forced_tool.value,
        "no_tool_compliant": record.no_tool_compliant.value,
        "continuation": record.continuation.value,
        "tested_probes": sorted(record.tested_probes),
        "battery_revision": record.battery_revision,
        "template_digest": record.template_digest,
    }


def _deserialize_record(digest: str, value: dict[str, Any]) -> QualificationRecord | None:
    """P0.29: Deserialize record with ProbeOutcome enum and tested_probes."""
    try:
        return QualificationRecord(
            model_digest=digest,
            state=QualificationState(value.get("state", QualificationState.UNKNOWN.value)),
            native_forced_tool=ProbeOutcome(value.get("native_forced_tool", ProbeOutcome.UNKNOWN.value)),
            prompted_forced_tool=ProbeOutcome(value.get("prompted_forced_tool", ProbeOutcome.UNKNOWN.value)),
            no_tool_compliant=ProbeOutcome(value.get("no_tool_compliant", ProbeOutcome.UNKNOWN.value)),
            continuation=ProbeOutcome(value.get("continuation", ProbeOutcome.UNKNOWN.value)),
            tested_probes=frozenset(value.get("tested_probes", [])),
            battery_revision=value.get("battery_revision", ""),
            template_digest=value.get("template_digest", ""),
        )
    except (ValueError, KeyError):
        return None


class QualificationStore:
    """Atomic JSON cache keyed by immutable served-model digest.

    The store contains only side-effect-free probe outcomes. It never stores
    prompts, model output, tool arguments, credentials, or client content.

    P0.29 evidence discipline: records carry ``battery_revision`` and
    ``template_digest`` produced by bootstrap.  ``record_is_current`` lets
    callers check whether a cached record is still valid for the
    _current_ battery + template, before the gateway-side comparison lands.
    """

    # P0-56: v2 — tri-state ProbeOutcome evidence + battery/template
    # currency fields.  A v1 payload predates record_is_current, so its
    # records cannot prove which battery/template produced them; loading
    # them as current would let stale evidence qualify a model.
    schema_version = 2

    def __init__(self, path: Path, max_records: int = 1024) -> None:
        self.path = path.expanduser()
        self.max_records = max_records
        self._records = self._load()

    def _load(self) -> dict[str, QualificationRecord]:
        try:
            payload = json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict) or payload.get("schema_version") != self.schema_version:
            return {}
        records: dict[str, QualificationRecord] = {}
        for digest, value in payload.get("records", {}).items():
            if not isinstance(digest, str) or not isinstance(value, dict):
                continue
            record = _deserialize_record(digest, value)
            if record is not None:
                records[digest] = record
        return records

    def get(self, model_digest: str) -> QualificationRecord | None:
        return self._records.get(model_digest)

    def put(self, record: QualificationRecord) -> None:
        if not record.model_digest:
            return
        self._records[record.model_digest] = record
        while len(self._records) > self.max_records:
            # The cache is only an operational optimization; evicting a
            # deterministic key causes a safe requalification, never a
            # compatibility promotion.
            self._records.pop(min(self._records))
        payload = {
            "schema_version": self.schema_version,
            "records": {digest: _serialize_record(value) for digest, value in sorted(self._records.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
        os.replace(temporary, self.path)


def record_is_current(
    record, *, battery_revision: str, template_digest: str
) -> bool:
    """A record is trustworthy only for the exact battery + template that produced it.

    Uses ``getattr`` with empty-string defaults so duck-typed objects that
    lack the new fields are treated as stale rather than crashing the
    comparison path.
    """
    return bool(
        getattr(record, "battery_revision", "") == battery_revision
        and getattr(record, "template_digest", "") == template_digest
    )
