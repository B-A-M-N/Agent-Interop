"""Bounded diagnostic replay-case retention with optional durable storage."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import asdict, replace
from pathlib import Path

from agent_interop.replay.types import CompatibilityKey, ReplayCase, ReplayInvariant


class DiagnosticCaseStore:
    """LRU case store, optionally mirrored as sanitized JSON on disk."""

    def __init__(
        self,
        retention_count: int = 100,
        directory: Path | None = None,
        max_case_bytes: int = 1024 * 1024,
    ) -> None:
        self.retention_count = max(1, retention_count)
        self.directory = directory
        self.max_case_bytes = max(1024, max_case_bytes)
        self._cases: OrderedDict[str, ReplayCase] = OrderedDict()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._prune_disk()

    def put(self, case: ReplayCase) -> None:
        case = self._bounded_case(case)
        self._cases[case.case_id] = case
        self._cases.move_to_end(case.case_id)
        while len(self._cases) > self.retention_count:
            self._cases.popitem(last=False)
        if self.directory is not None:
            self._write_case(case)
            self._prune_disk()

    def get(self, case_id: str) -> ReplayCase | None:
        case = self._cases.get(case_id)
        if case is not None:
            self._cases.move_to_end(case_id)
        if case is not None:
            return case
        return self.load(case_id)

    def list_ids(self) -> tuple[str, ...]:
        disk_ids = () if self.directory is None else tuple(path.stem for path in sorted(self.directory.glob("*.json")))
        return tuple(dict.fromkeys((*self._cases.keys(), *disk_ids)))

    def load(self, case_id: str) -> ReplayCase | None:
        if self.directory is None:
            return None
        path = self.directory / f"{case_id}.json"
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        key_data = data.get("compatibility_key", {})
        invariants = tuple(ReplayInvariant(**item) for item in data.get("expected_invariants", []) if isinstance(item, dict))
        case = ReplayCase(
            format_version=str(data.get("format_version", "interop.replay.v1")),
            case_id=str(data.get("case_id", case_id)),
            client_protocol=str(data.get("client_protocol", "")),
            upstream_protocol=str(data.get("upstream_protocol", "")),
            compatibility_key=CompatibilityKey(**key_data) if isinstance(key_data, dict) else CompatibilityKey(),
            inbound_request=data.get("inbound_request", {}),
            upstream_request=data.get("upstream_request", {}),
            raw_upstream_response=data.get("raw_upstream_response"),
            raw_upstream_frames=tuple(data.get("raw_upstream_frames", ())),
            expected_invariants=invariants,
            diagnostics=data.get("diagnostics", {}),
        )
        self._cases[case.case_id] = case
        self._cases.move_to_end(case.case_id)
        return case

    def _write_case(self, case: ReplayCase) -> None:
        assert self.directory is not None
        destination = self.directory / f"{case.case_id}.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_bytes(self._serialized(case))
        os.replace(temporary, destination)

    def _prune_disk(self) -> None:
        assert self.directory is not None
        cases = sorted(self.directory.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in cases[self.retention_count:]:
            path.unlink(missing_ok=True)

    def _bounded_case(self, case: ReplayCase) -> ReplayCase:
        """Ensure a pathological response cannot exhaust diagnostic storage."""
        raw = self._serialized(case)
        if len(raw) <= self.max_case_bytes:
            return case
        return replace(
            case,
            raw_upstream_response=None,
            raw_upstream_frames=(),
            diagnostics={
                "capture_truncated": True,
                "original_case_bytes": len(raw),
                "max_case_bytes": self.max_case_bytes,
                "response_status": case.diagnostics.get("response", {}),
                "build": case.diagnostics.get("build", {}),
            },
        )

    @staticmethod
    def _serialized(case: ReplayCase) -> bytes:
        return json.dumps(asdict(case), default=str, sort_keys=True, indent=2).encode()
