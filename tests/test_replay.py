"""Tests for the replay subsystem: capture, compare, and runner."""

from __future__ import annotations

import asyncio

from agent_interop.abi import CanonicalModelReference, CanonicalRequest, CanonicalTool
from agent_interop.replay.capture import capture_case, sanitize_body, sanitize_headers
from agent_interop.replay.compare import compare_policies, summarize_comparisons
from agent_interop.replay.runner import (
    _check_arguments_valid,
    _check_tool_identity,
    _extract_candidates_from_raw,
    replay_all_policies,
    replay_case,
)
from agent_interop.replay.types import (
    REPAIR_POLICIES,
    CompatibilityKey,
    ReplayCase,
    ReplayInvariant,
    ReplayResult,
)

# ─── Sanitization ────────────────────────────────────────────────────────────


class TestSanitizeHeaders:
    def test_redacts_authorization(self):
        headers = {"Authorization": "Bearer secret-token", "Content-Type": "application/json"}
        result = sanitize_headers(headers)
        assert result["Authorization"] == "[REDACTED]"
        assert result["Content-Type"] == "application/json"

    def test_redacts_x_api_key(self):
        headers = {"x-api-key": "sk-12345", "accept": "application/json"}
        result = sanitize_headers(headers)
        assert result["x-api-key"] == "[REDACTED]"

    def test_case_insensitive(self):
        headers = {"AUTHORIZATION": "secret"}
        result = sanitize_headers(headers)
        assert result["AUTHORIZATION"] == "[REDACTED]"

    def test_non_sensitive_preserved(self):
        headers = {"content-type": "application/json", "accept": "*/*"}
        result = sanitize_headers(headers)
        assert result["content-type"] == "application/json"
        assert result["accept"] == "*/*"


class TestSanitizeBody:
    def test_redacts_token_fields(self):
        body = {"model": "test", "api_key": "secret", "messages": []}
        result = sanitize_body(body)
        assert result["api_key"] == "[REDACTED]"
        assert result["model"] == "test"

    def test_redacts_password(self):
        body = {"username": "admin", "password": "hunter2"}
        result = sanitize_body(body)
        assert result["password"] == "[REDACTED]"
        assert result["username"] == "admin"

    def test_non_sensitive_preserved(self):
        body = {"model": "test", "temperature": 0.7, "messages": [{"role": "user"}]}
        result = sanitize_body(body)
        assert result["temperature"] == 0.7


# ─── Capture ─────────────────────────────────────────────────────────────────


class TestCaptureCase:
    def test_captures_basic_case(self):
        case = capture_case(
            client_protocol="anthropic_messages",
            upstream_protocol="ollama_chat",
            inbound_request={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
            upstream_request={"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert isinstance(case, ReplayCase)
        assert case.client_protocol == "anthropic_messages"
        assert case.upstream_protocol == "ollama_chat"

    def test_sensitive_data_redacted(self):
        case = capture_case(
            client_protocol="anthropic_messages",
            upstream_protocol="ollama_chat",
            inbound_request={"model": "test", "api_key": "secret"},
            upstream_request={"model": "test", "api_key": "secret"},
        )
        # api_key should be redacted in the captured request
        assert case.inbound_request.get("api_key") == "[REDACTED]"

    def test_generates_case_id(self):
        case = capture_case(
            client_protocol="anthropic_messages",
            upstream_protocol="ollama_chat",
            inbound_request={"model": "test"},
            upstream_request={"model": "test"},
        )
        # Should generate a deterministic ID
        assert case.case_id.startswith("case_")

    def test_explicit_case_id_preserved(self):
        case = capture_case(
            client_protocol="anthropic_messages",
            upstream_protocol="ollama_chat",
            inbound_request={},
            upstream_request={},
            case_id="my-custom-id",
        )
        assert case.case_id == "my-custom-id"


# ─── Compare ─────────────────────────────────────────────────────────────────


class TestComparePolicies:
    def test_repair_helped(self):
        """When baseline fails but repair succeeds, repair_helped should be True."""
        results = {
            "repair_disabled": ReplayResult(
                policy_name="repair_disabled",
                executable=False,
            ),
            "safe_only": ReplayResult(
                policy_name="safe_only",
                executable=True,
                arguments_valid=True,
            ),
        }
        comparison = compare_policies(results)
        assert comparison.repair_helped is True

    def test_repair_not_needed(self):
        """When baseline already works, repair_helped should be False."""
        results = {
            "repair_disabled": ReplayResult(
                policy_name="repair_disabled",
                executable=True,
            ),
        }
        comparison = compare_policies(results)
        assert comparison.repair_helped is False

    def test_best_policy_selected(self):
        """best_policy should pick the policy with the best outcome."""
        results = {
            "repair_disabled": ReplayResult(
                policy_name="repair_disabled",
                executable=False,
            ),
            "aggressive": ReplayResult(
                policy_name="aggressive",
                executable=True,
            ),
        }
        comparison = compare_policies(results)
        assert comparison.best_policy == "aggressive"


class TestSummarizeComparisons:
    def test_summary_counts(self):
        results = {
            "repair_disabled": ReplayResult(policy_name="repair_disabled", executable=False),
            "safe_only": ReplayResult(policy_name="safe_only", executable=True, arguments_valid=True),
        }
        comparisons = [compare_policies(results)]
        summary = summarize_comparisons(comparisons)
        assert summary["total_cases"] == 1
        assert summary["repair_helped_count"] == 1

    def test_empty_comparisons(self):
        summary = summarize_comparisons([])
        assert summary["total_cases"] == 0


# ─── Replay Types ────────────────────────────────────────────────────────────


class TestReplayTypes:
    def test_replay_case_serialization(self):
        """ReplayCase should be JSON-serializable for file storage."""
        case = ReplayCase(
            case_id="test-123",
            client_protocol="anthropic_messages",
            upstream_protocol="ollama_chat",
            inbound_request={"model": "test", "messages": []},
            upstream_request={"model": "test", "messages": []},
        )
        # Should be serializable via asdict or __dict__
        import dataclasses
        data = dataclasses.asdict(case)
        assert data["case_id"] == "test-123"
        assert data["client_protocol"] == "anthropic_messages"

    def test_compatibility_key_dimensions(self):
        """CompatibilityKey should have all dimensions."""
        key = CompatibilityKey(
            client_id="test",
            client_version="1.0",
            client_protocol="anthropic_messages",
            model_id="model",
            backend_kind="ollama",
            upstream_protocol="ollama_chat",
            profile_id="default",
        )
        assert key.client_id == "test"
        assert key.upstream_protocol == "ollama_chat"
        assert key.backend_kind == "ollama"

    def test_replay_invariant_types(self):
        """ReplayInvariant should support all invariant types."""
        inv = ReplayInvariant(type="tool_name", expected="read_file", description="Must call read_file")
        assert inv.type == "tool_name"
        assert inv.expected == "read_file"


# ─── Runner: replay_case / replay_all_policies ──────────────────────────────
#
# Despite the module docstring above ("capture, compare, and runner"),
# replay/runner.py had no tests at all until now — the module driving the
# actual repair-policy comparison the CLI's `interop replay` command
# depends on was completely unverified.

READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)


def _run(coro):
    return asyncio.run(coro)


class TestExtractCandidatesFromRaw:
    def test_openai_format(self):
        raw = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                    }],
                },
            }],
        }
        candidates = _extract_candidates_from_raw(raw, [READ_FILE_TOOL])
        assert len(candidates) == 1
        assert candidates[0].name == "read_file"
        assert candidates[0].id == "call_1"
        assert candidates[0].source_protocol == "openai_chat"

    def test_anthropic_format(self):
        raw = {
            "content": [
                {"type": "text", "text": "I'll read that."},
                {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "/tmp/x"}},
            ],
        }
        candidates = _extract_candidates_from_raw(raw, [READ_FILE_TOOL])
        assert len(candidates) == 1
        assert candidates[0].name == "read_file"
        assert candidates[0].id == "toolu_1"
        assert candidates[0].source_protocol == "anthropic_messages"

    def test_no_tool_calls_returns_empty(self):
        raw = {"choices": [{"message": {"content": "just text, no tools"}}]}
        assert _extract_candidates_from_raw(raw, [READ_FILE_TOOL]) == []

    def test_empty_raw_response(self):
        assert _extract_candidates_from_raw({}, [READ_FILE_TOOL]) == []


class TestCheckToolIdentity:
    def _decision(self, name: str):
        from agent_interop.abi import (
            CanonicalToolCallBlock,
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )

        return ToolCallDecision(
            candidate=RawToolCallCandidate(id="1", name=name, raw_arguments="{}"),
            outcome=RepairOutcome(status=RepairStatus.VALID_UNCHANGED, call_name=name, accepted={}),
            accepted_block=CanonicalToolCallBlock(id="1", name=name, arguments={}),
        )

    def test_no_invariant_always_true(self):
        assert _check_tool_identity([self._decision("read_file")], ()) is True
        assert _check_tool_identity([], ()) is True

    def test_matching_name_invariant_true(self):
        inv = (ReplayInvariant(type="tool_name", expected="read_file"),)
        assert _check_tool_identity([self._decision("read_file")], inv) is True

    def test_mismatched_name_invariant_false(self):
        inv = (ReplayInvariant(type="tool_name", expected="read_file"),)
        assert _check_tool_identity([self._decision("get_weather")], inv) is False


class TestCheckArgumentsValid:
    def _decision(self, name: str, arguments: dict):
        from agent_interop.abi import (
            CanonicalToolCallBlock,
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )

        return ToolCallDecision(
            candidate=RawToolCallCandidate(id="1", name=name, raw_arguments="{}"),
            outcome=RepairOutcome(status=RepairStatus.VALID_UNCHANGED, call_name=name, accepted=arguments),
            accepted_block=CanonicalToolCallBlock(id="1", name=name, arguments=arguments),
        )

    def test_valid_arguments(self):
        decision = self._decision("read_file", {"path": "/tmp/x"})
        assert _check_arguments_valid(decision, [READ_FILE_TOOL]) is True

    def test_invalid_arguments_missing_required(self):
        decision = self._decision("read_file", {"wrong_field": "x"})
        assert _check_arguments_valid(decision, [READ_FILE_TOOL]) is False

    def test_unknown_tool_name(self):
        decision = self._decision("nonexistent", {"path": "/tmp/x"})
        assert _check_arguments_valid(decision, [READ_FILE_TOOL]) is False

    def test_no_accepted_block(self):
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )

        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            outcome=RepairOutcome(status=RepairStatus.REJECTED, call_name="read_file", accepted=None, error="bad"),
            accepted_block=None,
        )
        assert _check_arguments_valid(decision, [READ_FILE_TOOL]) is False


class TestReplayCase:
    def _case(self, **overrides) -> ReplayCase:
        defaults = {
            "case_id": "case-1",
            "canonical_request": CanonicalRequest(
                model=CanonicalModelReference(requested_name="m"),
                tools=[READ_FILE_TOOL],
            ),
            "raw_upstream_response": {
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                        }],
                    },
                }],
            },
            "tool_registry": (READ_FILE_TOOL,),
        }
        defaults.update(overrides)
        return ReplayCase(**defaults)

    def test_no_canonical_request_produces_diagnostic(self):
        case = self._case(canonical_request=None)
        result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert result.executable is False
        assert "No canonical request" in result.diagnostics[0]

    def test_no_raw_response_produces_diagnostic(self):
        case = self._case(raw_upstream_response=None)
        result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert result.executable is False
        assert "No raw upstream response" in result.diagnostics[0]

    def test_no_candidates_and_no_tool_invariant_is_executable(self):
        """A case that never expected a tool call and got none is a PASS,
        not a failure — executable defaults True when nothing required a
        tool call in the first place."""
        case = self._case(raw_upstream_response={"choices": [{"message": {"content": "hi"}}]})
        result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert result.executable is True
        assert "No tool candidates found" in result.diagnostics[0]

    def test_no_candidates_but_tool_invariant_expected_is_not_executable(self):
        case = self._case(
            raw_upstream_response={"choices": [{"message": {"content": "hi"}}]},
            expected_invariants=(ReplayInvariant(type="tool_name", expected="read_file"),),
        )
        result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert result.executable is False

    def test_valid_tool_call_is_executable_and_arguments_valid(self):
        case = self._case()
        result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert result.executable is True
        assert result.arguments_valid is True
        assert result.output_tool_name == "read_file"
        assert result.output_arguments == {"path": "/tmp/x"}

    def test_repair_disabled_policy_rejects_malformed_json(self):
        """With repair disabled, malformed JSON must fail — proving the
        policy comparison this whole subsystem exists for actually
        produces different results per policy."""
        case = self._case(
            raw_upstream_response={
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "/tmp/x",}'},
                        }],
                    },
                }],
            },
        )
        disabled_result = _run(replay_case(case, "repair_disabled", REPAIR_POLICIES["repair_disabled"]))
        assert disabled_result.executable is False

        repaired_result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert repaired_result.executable is True

    def test_undeclared_tool_call_is_rejected(self):
        case = self._case(
            raw_upstream_response={
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "delete_everything", "arguments": "{}"},
                        }],
                    },
                }],
            },
        )
        result = _run(replay_case(case, "safe_shape", REPAIR_POLICIES["safe_shape"]))
        assert result.executable is False
        assert result.diagnostics


class TestReplayAllPolicies:
    def test_runs_every_standard_policy(self):
        case = ReplayCase(
            case_id="case-1",
            canonical_request=CanonicalRequest(
                model=CanonicalModelReference(requested_name="m"), tools=[READ_FILE_TOOL],
            ),
            raw_upstream_response={
                "choices": [{
                    "message": {
                        "tool_calls": [{
                            "id": "call_1", "type": "function",
                            "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                        }],
                    },
                }],
            },
            tool_registry=(READ_FILE_TOOL,),
        )
        results = _run(replay_all_policies(case))
        assert set(results.keys()) == set(REPAIR_POLICIES.keys())
        for policy_name, result in results.items():
            assert result.policy_name == policy_name
            assert result.case_id == "case-1"
