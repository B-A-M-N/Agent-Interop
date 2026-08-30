"""Regression tests for evidence invalidation: battery_revision + template_digest.

Covers the store/data-contract half of review finding 24:
  (a) QualificationStore put/get round-trip preserves battery_revision &
      template_digest
  (b) record_is_current returns True on exact match, False on battery
      mismatch, False on template mismatch, False on missing attrs
  (c) merge() carries forward battery_revision/template_digest from the
      old record
"""

from __future__ import annotations

from agent_interop.qualification.revision import QUALIFICATION_BATTERY_REVISION
from agent_interop.qualification.state import ProbeOutcome, QualificationRecord
from agent_interop.qualification.store import (
    QualificationStore,
    record_is_current,
)


# ─── (a) store put/get preserves battery_revision & template_digest ────────


class TestStoreRoundTrip:
    """A record put through QualificationStore must survive a get with all fields."""

    def test_battery_revision_survives_round_trip(self, tmp_path):
        store = QualificationStore(path=tmp_path / "qual.json")
        rec = QualificationRecord(
            model_digest="round-trip-digest",
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest="template-digest-01",
        )
        store.put(rec)
        retrieved = store.get("round-trip-digest")
        assert retrieved is not None
        assert retrieved.battery_revision == QUALIFICATION_BATTERY_REVISION
        assert retrieved.template_digest == "template-digest-01"

    def test_template_digest_survives_round_trip(self, tmp_path):
        store = QualificationStore(path=tmp_path / "qual2.json")
        rec = QualificationRecord(
            model_digest="rt-digest-2",
            native_forced_tool=ProbeOutcome.PASSED,
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest="my-unique-template-sha",
        )
        store.put(rec)
        retrieved = store.get("rt-digest-2")
        assert retrieved is not None
        assert retrieved.template_digest == "my-unique-template-sha"
        assert retrieved.battery_revision == QUALIFICATION_BATTERY_REVISION


# ─── (b) record_is_current: exact match, mismatch, missing attrs ──────────


class TestRecordIsCurrent:
    """record_is_current must implement the three-way gate."""

    def test_exact_match_returns_true(self):
        rec = QualificationRecord(
            model_digest="d1",
            battery_revision="rev-1",
            template_digest="tmpl-1",
        )
        assert record_is_current(rec, battery_revision="rev-1", template_digest="tmpl-1")

    def test_battery_mismatch_returns_false(self):
        rec = QualificationRecord(
            model_digest="d1",
            battery_revision="rev-1",
            template_digest="tmpl-1",
        )
        assert not record_is_current(rec, battery_revision="rev-2", template_digest="tmpl-1")

    def test_template_mismatch_returns_false(self):
        rec = QualificationRecord(
            model_digest="d1",
            battery_revision="rev-1",
            template_digest="tmpl-1",
        )
        assert not record_is_current(rec, battery_revision="rev-1", template_digest="tmpl-2")

    def test_missing_battery_attr_returns_false(self):
        """Duck-typed object without battery_revision must be treated as stale."""

        class MinimalRecord:
            model_digest = "d1"

        assert not record_is_current(
            MinimalRecord(), battery_revision="rev-1", template_digest="tmpl-1"
        )

    def test_missing_template_attr_returns_false(self):
        """Duck-typed object without template_digest must be treated as stale."""

        class MinimalRecord:
            model_digest = "d1"
            battery_revision = "rev-1"

        assert not record_is_current(
            MinimalRecord(), battery_revision="rev-1", template_digest="tmpl-1"
        )


# ─── (c) merge carries forward battery_revision & template_digest ──────────


class TestMergeCarriesForward:
    """QualificationRecord.merge() must copy battery_revision and template_digest."""

    def test_merge_carries_battery_revision(self):
        rec = QualificationRecord(
            model_digest="d1",
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest="tmpl-for-merge",
        )
        merged = rec.merge({"native_forced_tool": ProbeOutcome.PASSED})
        assert merged.battery_revision == QUALIFICATION_BATTERY_REVISION
        assert merged.template_digest == "tmpl-for-merge"

    def test_merge_does_not_overwrite_existing_fields(self):
        rec = QualificationRecord(
            model_digest="d2",
            native_forced_tool=ProbeOutcome.PASSED,
            battery_revision="rev-abc",
            template_digest="tmpl-xyz",
        )
        # Merge unrelated probe outcome — battery + template must survive unchanged
        merged = rec.merge({"prompted_forced_tool": ProbeOutcome.FAILED})
        assert merged.battery_revision == "rev-abc"
        assert merged.template_digest == "tmpl-xyz"
        assert merged.native_forced_tool == ProbeOutcome.PASSED
        assert merged.prompted_forced_tool == ProbeOutcome.FAILED
