"""Tests for `interop repair stats` — the CLI surface over
EvidenceStore.record_repair_event/query_repair_stats.

Turns repair-pipeline effectiveness into measurable, queryable evidence:
accepted without repair, accepted after repair, rejected, and rejected-
with-partial-repair, grouped by route/model/client, with a --json escape
hatch for scripting.
"""

from __future__ import annotations

import json

from agent_interop.evidence.store import EvidenceStore


def _store() -> EvidenceStore:
    return EvidenceStore(db_path=":memory:")


def _seed(store: EvidenceStore) -> None:
    store.record_repair_event(
        route_id="r1", model_id="qwen3-coder", client_id="claude_code",
        tool_name="read_file", outcome="valid_unchanged",
    )
    store.record_repair_event(
        route_id="r1", model_id="qwen3-coder", client_id="claude_code",
        tool_name="edit_file", outcome="repaired", repair_rules=["rename_aliased_fields"],
    )
    store.record_repair_event(
        route_id="r1", model_id="qwen3-coder", client_id="claude_code",
        tool_name="edit_file", outcome="rejected", repair_rules=["rename_aliased_fields"],
    )
    store.record_repair_event(
        route_id="r1", model_id="qwen3-coder", client_id="claude_code",
        tool_name="write_file", outcome="rejected",
    )
    # a distinct group (different route) must never be merged into r1's.
    store.record_repair_event(
        route_id="r2", model_id="other-model", client_id="codex",
        tool_name="read_file", outcome="valid_unchanged",
    )


class TestQueryRepairStatsAggregation:
    def test_groups_by_route_model_client_and_partitions_outcomes(self):
        store = _store()
        _seed(store)
        groups = store.query_repair_stats()
        assert len(groups) == 2

        r1 = next(g for g in groups if g.route_id == "r1")
        assert r1.total_eligible == 4
        assert r1.accepted_without_repair == 1
        assert r1.accepted_after_repair == 1
        assert r1.rejected == 2
        # only one of the two rejections had a repair rule fire first.
        assert r1.rejected_with_partial_repair == 1
        assert r1.rule_counts == {"rename_aliased_fields": 2}

        r2 = next(g for g in groups if g.route_id == "r2")
        assert r2.total_eligible == 1
        assert r2.accepted_without_repair == 1

    def test_filters_are_independently_composable(self):
        store = _store()
        _seed(store)
        assert len(store.query_repair_stats(route_id="r1")) == 1
        assert len(store.query_repair_stats(client_id="codex")) == 1
        assert len(store.query_repair_stats(model_id="does-not-exist")) == 0

    def test_since_filters_out_older_events(self):
        store = _store()
        _seed(store)
        # everything just seeded is "now" — a since far in the future
        # excludes it all without needing to fake the clock.
        groups = store.query_repair_stats(since="2999-01-01T00:00:00+00:00")
        assert groups == []


class TestRepairStatsCli:
    def _runner_app(self, store: EvidenceStore, monkeypatch):
        from typer.testing import CliRunner

        from agent_interop.cli import app

        monkeypatch.setattr("agent_interop.evidence.store.get_default_store", lambda: store)
        return CliRunner(), app

    def test_stats_table_output(self, monkeypatch):
        store = _store()
        _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result = runner.invoke(app, ["repair", "stats"])
        assert result.exit_code == 0
        assert "r1" in result.stdout
        assert "r2" in result.stdout
        # Rich truncates wide columns at the default 80-col test width —
        # rule_counts content is fully covered by the --json test below.
        assert "Repair Pipeline Stats" in result.stdout

    def test_stats_json_output_is_parseable_and_complete(self, monkeypatch):
        store = _store()
        _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result = runner.invoke(app, ["repair", "stats", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 2
        r1 = next(g for g in data if g["route_id"] == "r1")
        assert r1["total_eligible"] == 4
        assert r1["accepted_without_repair"] == 1
        assert r1["accepted_after_repair"] == 1
        assert r1["rejected"] == 2
        assert r1["rejected_with_partial_repair"] == 1
        assert r1["rule_counts"] == {"rename_aliased_fields": 2}

    def test_stats_filtered_by_route(self, monkeypatch):
        store = _store()
        _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result = runner.invoke(app, ["repair", "stats", "--route", "r1", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 1
        assert data[0]["route_id"] == "r1"

    def test_stats_empty_store_reports_no_events(self, monkeypatch):
        store = _store()
        runner, app = self._runner_app(store, monkeypatch)
        result = runner.invoke(app, ["repair", "stats"])
        assert result.exit_code == 0
        assert "No repair events recorded" in result.stdout

    def test_invalid_since_exits_nonzero(self, monkeypatch):
        store = _store()
        runner, app = self._runner_app(store, monkeypatch)
        result = runner.invoke(app, ["repair", "stats", "--since", "not-a-date"])
        assert result.exit_code != 0

    def test_bare_date_since_is_accepted(self, monkeypatch):
        store = _store()
        _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result = runner.invoke(app, ["repair", "stats", "--since", "2020-01-01", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert len(data) == 2
