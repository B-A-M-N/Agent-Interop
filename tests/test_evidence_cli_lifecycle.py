"""Tests for the evidence lifecycle CLI (re-audit P1#9).

Previously the CLI only exposed list/show/revoke — mark_verified() existed
on EvidenceStore but had no supported way for an operator to actually call
it, so 'interop certify' could observe evidence but nothing could ever
legitimately promote it to manually_verified. These tests exercise the new
'review', 'approve', and 'unrevoke' actions end-to-end through the real
Typer CLI, plus the underlying store.mark_verified(attestation=...) and the
attestation column round-trip.
"""

from __future__ import annotations

from agent_interop.evidence.store import EvidenceStore
from agent_interop.replay.types import CompatibilityKey, CompatibilityResult


def _store() -> EvidenceStore:
    return EvidenceStore(db_path=":memory:")


def _seed(store: EvidenceStore, **key_overrides) -> CompatibilityKey:
    defaults = {
        "client_id": "claude_code", "client_version": "1.0", "model_id": "test-model",
        "backend_kind": "ollama",
    }
    defaults.update(key_overrides)
    key = CompatibilityKey(**defaults)
    store.store_result(key, CompatibilityResult(sample_count=10, tested_at="2026-01-01T00:00:00"))
    return key


class TestStoreAttestation:
    def test_mark_verified_records_attestation(self):
        store = _store()
        key = _seed(store)
        store.mark_verified(key, attestation="Manually ran claude-code v1.2 against this route.")
        result = store.get_result(key)
        assert result is not None
        assert result.manually_verified is True
        assert result.attestation == "Manually ran claude-code v1.2 against this route."

    def test_mark_verified_without_attestation_preserves_prior_one(self):
        """Re-approving without a fresh attestation must not silently blank
        out a previously recorded one."""
        store = _store()
        key = _seed(store)
        store.mark_verified(key, attestation="first review")
        store.mark_verified(key)  # no attestation passed this time
        result = store.get_result(key)
        assert result is not None
        assert result.attestation == "first review"

    def test_attestation_round_trips_through_query_results(self):
        """query_results() has its own independent row-mapping — a field
        added only to get_result() would silently vanish from 'list'/filter
        based lookups, which approve/revoke/unrevoke all use."""
        store = _store()
        key = _seed(store)
        store.mark_verified(key, attestation="checked via review")
        matches = store.query_results(model_id="test-model")
        assert len(matches) == 1
        _, result = matches[0]
        assert result.attestation == "checked via review"


class TestEvidenceCliActions:
    def _runner_app(self, store: EvidenceStore, monkeypatch):
        from typer.testing import CliRunner

        from agent_interop.cli import app

        monkeypatch.setattr("agent_interop.evidence.store.get_default_store", lambda: store)
        return CliRunner(), app

    def test_review_shows_full_compatibility_tuple(self, monkeypatch):
        store = _store()
        key = _seed(
            store, upstream_protocol="openai_chat", profile_id="qwen3-coder",
            profile_revision="rev-3", tool_schema_fingerprint="abc123",
            parser_id="tool_call_envelope", effective_tool_mode="prompted",
        )
        runner, app = self._runner_app(store, monkeypatch)
        result_id = store._make_result_id(key)

        result = runner.invoke(app, ["evidence", "review", "--id", result_id])
        assert result.exit_code == 0, result.output
        assert "claude_code" in result.output
        assert "qwen3-coder" in result.output
        assert "rev-3" in result.output
        assert "abc123" in result.output
        assert "tool_call_envelope" in result.output
        assert "prompted" in result.output
        # review must point the operator at the actual approve command.
        assert "evidence approve" in result.output

    def test_approve_without_attestation_is_rejected(self, monkeypatch):
        store = _store()
        key = _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result_id = store._make_result_id(key)

        result = runner.invoke(app, ["evidence", "approve", "--id", result_id])
        assert result.exit_code != 0
        assert "attestation" in result.output.lower()
        stored = store.get_result(key)
        assert stored is not None
        assert stored.manually_verified is False

    def test_approve_with_attestation_marks_verified(self, monkeypatch):
        store = _store()
        key = _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result_id = store._make_result_id(key)

        result = runner.invoke(
            app, ["evidence", "approve", "--id", result_id, "--attestation", "reviewed manually"],
        )
        assert result.exit_code == 0, result.output
        stored = store.get_result(key)
        assert stored is not None
        assert stored.manually_verified is True
        assert stored.attestation == "reviewed manually"

    def test_unrevoke_clears_revocation(self, monkeypatch):
        store = _store()
        key = _seed(store)
        store.revoke(key, reason="flaky")
        runner, app = self._runner_app(store, monkeypatch)
        result_id = store._make_result_id(key)

        result = runner.invoke(app, ["evidence", "unrevoke", "--id", result_id])
        assert result.exit_code == 0, result.output
        stored = store.get_result(key)
        assert stored is not None
        assert stored.revoked is False
        assert stored.revocation_reason == ""

    def test_show_still_works_unchanged(self, monkeypatch):
        """'show' remains a supported alias-equivalent of 'review' (same
        display), not removed by adding the new actions."""
        store = _store()
        key = _seed(store)
        runner, app = self._runner_app(store, monkeypatch)
        result_id = store._make_result_id(key)

        result = runner.invoke(app, ["evidence", "show", "--id", result_id])
        assert result.exit_code == 0, result.output
        assert "test-model" in result.output


class TestEvidenceRouteFilter:
    """REVISION #5: 'evidence list/show' can be filtered by --route (a
    gateway-config concept), resolved via --config to that route's
    upstream_model — the same identifier CompatibilityKey.model_id is
    built from (see evidence/key.py: model_id = route.upstream_model)."""

    def _runner_app(self, store: EvidenceStore, monkeypatch):
        from typer.testing import CliRunner

        from agent_interop.cli import app

        monkeypatch.setattr("agent_interop.evidence.store.get_default_store", lambda: store)
        return CliRunner(), app

    def _write_config(self, tmp_path, *, route_id: str, upstream_model: str) -> str:
        config_path = tmp_path / "interop.yaml"
        config_path.write_text(
            "routes:\n"
            f"  {route_id}:\n"
            f"    upstream_model: {upstream_model}\n"
            "    aliases: [test-model]\n"
            "    upstream:\n"
            "      kind: ollama\n"
            "      base_url: http://127.0.0.1:11434\n"
        )
        return str(config_path)

    def test_route_resolves_to_matching_evidence(self, tmp_path, monkeypatch):
        store = _store()
        key = _seed(store, model_id="qwen3-coder")
        _seed(store, model_id="some-other-model", client_id="other_client")
        config_path = self._write_config(tmp_path, route_id="r1", upstream_model="qwen3-coder")
        runner, app = self._runner_app(store, monkeypatch)

        result = runner.invoke(app, ["evidence", "list", "--route", "r1", "--config", config_path])
        assert result.exit_code == 0, result.output
        assert store._make_result_id(key)[:20] in result.output
        assert "some-other-model" not in result.output

    def test_unknown_route_errors(self, tmp_path, monkeypatch):
        store = _store()
        _seed(store)
        config_path = self._write_config(tmp_path, route_id="r1", upstream_model="qwen3-coder")
        runner, app = self._runner_app(store, monkeypatch)

        result = runner.invoke(app, ["evidence", "list", "--route", "does-not-exist", "--config", config_path])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_conflicting_model_and_route_errors(self, tmp_path, monkeypatch):
        store = _store()
        _seed(store)
        config_path = self._write_config(tmp_path, route_id="r1", upstream_model="qwen3-coder")
        runner, app = self._runner_app(store, monkeypatch)

        result = runner.invoke(
            app,
            ["evidence", "list", "--route", "r1", "--config", config_path, "--model", "totally-different"],
        )
        assert result.exit_code != 0
        assert "conflict" in result.output.lower()

    def test_multiple_distinct_keys_for_same_route_all_listed(self, tmp_path, monkeypatch):
        """The actual REVISION #5 guarantee: several distinct compatibility
        keys sharing a model_id (different client protocols/ids here) are
        ALL shown as separate rows — never silently collapsed to one."""
        store = _store()
        key_a = _seed(store, model_id="qwen3-coder", client_id="claude_code")
        key_b = _seed(store, model_id="qwen3-coder", client_id="codex")
        config_path = self._write_config(tmp_path, route_id="r1", upstream_model="qwen3-coder")
        runner, app = self._runner_app(store, monkeypatch)

        result = runner.invoke(app, ["evidence", "list", "--route", "r1", "--config", config_path])
        assert result.exit_code == 0, result.output
        assert store._make_result_id(key_a)[:20] in result.output
        assert store._make_result_id(key_b)[:20] in result.output
