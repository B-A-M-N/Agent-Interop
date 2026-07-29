"""Tests for the EvidenceStore — verifying store/retrieve round-trip with real SQLite."""

from __future__ import annotations

from agent_interop.evidence.store import EvidenceStore
from agent_interop.replay.types import CompatibilityKey, CompatibilityQuirk, CompatibilityResult


def _make_key(**overrides) -> CompatibilityKey:
    defaults = {
        "client_id": "claude_code",
        "client_version": "1.0.0",
        "client_protocol": "anthropic_messages",
        "model_id": "qwen3-coder",
        "model_digest": "sha256:abc123",
        "quantization": "q4_k_m",
        "backend_kind": "ollama",
        "backend_version": "0.5.1",
        "upstream_protocol": "openai_chat",
        "chat_template_digest": "tmpl_hash_abc",
        "profile_id": "qwen-coder-ollama",
        "profile_revision": "r1",
        "tool_schema_fingerprint": "fp_12345678",
        "streaming": False,
        "effective_tool_mode": "native",
        "parser_id": "hermes",
        "template_revision": "t1",
        "backend_serving_config": "",
    }
    defaults.update(overrides)
    return CompatibilityKey(**defaults)


def _make_result(**overrides) -> CompatibilityResult:
    defaults = {
        "tested_at": "2026-07-24T12:00:00+00:00",
        "sample_count": 10,
        "tool_selection_rate": 0.9,
        "valid_call_rate_before_repair": 0.7,
        "valid_call_rate_after_repair": 0.95,
        "task_completion_rate": 0.85,
        "deterministic_repair_rate": 0.25,
        "regeneration_rate": 0.0,
        "rejection_rate": 0.05,
        "streaming_equivalent": True,
        "history_round_trip_valid": True,
        "verified_capabilities": frozenset({"native", "textual"}),
        "known_quirks": (),
    }
    defaults.update(overrides)
    return CompatibilityResult(**defaults)


class TestEvidenceStoreRoundTrip:
    def test_store_and_retrieve(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        key = _make_key()
        result = _make_result()

        result_id = store.store_result(key, result)
        assert result_id
        assert result_id.startswith("res_")

        retrieved = store.get_result(key)
        assert retrieved is not None
        assert retrieved.sample_count == 10
        assert retrieved.tool_selection_rate == 0.9
        assert retrieved.valid_call_rate_after_repair == 0.95
        assert retrieved.task_completion_rate == 0.85
        assert retrieved.streaming_equivalent is True
        assert retrieved.history_round_trip_valid is True
        assert "native" in retrieved.verified_capabilities
        assert "textual" in retrieved.verified_capabilities

    def test_store_with_quirks(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        key = _make_key()
        quirk = CompatibilityQuirk(description="Emits trailing comma in JSON")
        result = _make_result(known_quirks=(quirk,))

        store.store_result(key, result)
        retrieved = store.get_result(key)
        assert retrieved is not None
        assert len(retrieved.known_quirks) == 1
        assert retrieved.known_quirks[0].description == "Emits trailing comma in JSON"

    def test_store_with_failure_cases(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        key = _make_key()
        result = _make_result()
        cases = [{"test": "explicit_tool", "error": "wrong name"}]

        result_id = store.store_result(key, result, failure_cases=cases)
        assert result_id

    def test_missing_key_returns_none(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        key = _make_key(model_id="nonexistent")
        assert store.get_result(key) is None

    def test_different_keys_are_separate(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        key1 = _make_key(model_id="model_a")
        key2 = _make_key(model_id="model_b")
        result1 = _make_result(sample_count=5)
        result2 = _make_result(sample_count=15)

        store.store_result(key1, result1)
        store.store_result(key2, result2)

        r1 = store.get_result(key1)
        r2 = store.get_result(key2)
        assert r1 is not None and r1.sample_count == 5
        assert r2 is not None and r2.sample_count == 15

    def test_streaming_vs_nonstreaming_keys_differ(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        key_stream = _make_key(streaming=True)
        key_nostream = _make_key(streaming=False)
        result_s = _make_result(sample_count=3)
        result_n = _make_result(sample_count=7)

        store.store_result(key_stream, result_s)
        store.store_result(key_nostream, result_n)

        rs = store.get_result(key_stream)
        rn = store.get_result(key_nostream)
        assert rs is not None and rs.sample_count == 3
        assert rn is not None and rn.sample_count == 7

    def test_reopen_database_persists(self, tmp_path):
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)
        key = _make_key()
        result = _make_result(sample_count=42)
        store.store_result(key, result)
        del store

        store2 = EvidenceStore(db_path=db_path)
        retrieved = store2.get_result(key)
        assert retrieved is not None
        assert retrieved.sample_count == 42

    def test_one_field_difference_separates(self, tmp_path):
        """Keys differing by exactly one field produce different result IDs."""
        db_path = str(tmp_path / "test_evidence.db")
        store = EvidenceStore(db_path=db_path)

        base = _make_key()
        different_profile = _make_key(profile_revision="r2")

        store.store_result(base, _make_result(sample_count=1))
        store.store_result(different_profile, _make_result(sample_count=2))

        assert store.get_result(base).sample_count == 1
        assert store.get_result(different_profile).sample_count == 2
