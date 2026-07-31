"""Durable, bounded bootstrap-qualification state."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from agent_interop.qualification.state import QualificationRecord, QualificationState


class QualificationStore:
    """Atomic JSON cache keyed by immutable served-model digest.

    The store contains only side-effect-free probe outcomes. It never stores
    prompts, model output, tool arguments, credentials, or client content.
    """

    schema_version = 1

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
            try:
                records[digest] = QualificationRecord(
                    model_digest=digest,
                    state=QualificationState(value.get("state", QualificationState.UNKNOWN.value)),
                    native_forced_tool=bool(value.get("native_forced_tool", False)),
                    prompted_forced_tool=bool(value.get("prompted_forced_tool", False)),
                    no_tool_compliant=bool(value.get("no_tool_compliant", False)),
                    continuation=bool(value.get("continuation", False)),
                )
            except ValueError:
                continue
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
            "records": {digest: asdict(value) for digest, value in sorted(self._records.items())},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2))
        os.replace(temporary, self.path)
