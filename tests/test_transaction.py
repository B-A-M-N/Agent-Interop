"""Phase 4 gate: Universal Tool Transaction Service tests.

Validates:
1. tool_transaction_service accepts a RawToolCallCandidate with already-valid args
2. Malformed JSON (trailing comma) is recovered via the repair pipeline
3. Unknown tool names produce rejection
4. Empty name produces rejection
5. The accepted_block has the correct shape
6. raw_arguments are preserved verbatim on the candidate after processing
7. None raw_arguments produce rejection
8. Regeneration path (when enabled) produces REGENERATED status
"""

from __future__ import annotations

import asyncio

from agent_interop.abi import (
    CanonicalTool,
    CanonicalToolChoice,
    RawToolCallCandidate,
    RepairOutcome,
    RepairStatus,
    SchemaIssue,
    ToolCallCorrection,
    ToolCallDecision,
)
from agent_interop.config import FieldAliasPolicy, RepairPolicy
from agent_interop.transaction import (
    ToolTransactionContext,
    _build_correction,
    tool_transaction_service,
    validate_batch_choice,
)


def _call_service(*args, **kwargs):
    """Synchronous wrapper for the now-async tool_transaction_service."""
    return asyncio.run(tool_transaction_service(*args, **kwargs))

# ── Sample tools ──────────────────────────────────────────────────────────

READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description="Read the contents of a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path"},
        },
        "required": ["path"],
    },
)

GET_WEATHER_TOOL = CanonicalTool(
    name="get_weather",
    description="Get the current weather for a city",
    input_schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
)

SAMPLE_TOOLS = [READ_FILE_TOOL, GET_WEATHER_TOOL]


# ── Phase 4 gate tests ────────────────────────────────────────────────────


class TestPhase4Gate:
    """The gate tests specified in PROPOSED_CHANGES.md Phase 4."""

    def test_tool_transaction_preserves_raw_malformed(self) -> None:
        """A RawToolCallCandidate with malformed JSON survives the transaction
        service as a RawToolCallCandidate with its raw_arguments preserved.
        The service does not 'fix' it into an empty dict before processing."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="read_file",
            raw_arguments='{"path":"/tmp/x",}',  # trailing comma
            source_protocol="ollama_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_accepted, f"Expected accepted, got {decision.outcome.status}"
        assert decision.accepted_block is not None
        assert decision.accepted_block.arguments == {"path": "/tmp/x"}
        # raw_arguments must be preserved verbatim on the candidate
        assert decision.candidate.raw_arguments == '{"path":"/tmp/x",}'


class TestTransactionService:
    """Additional transaction service behavior tests."""

    def test_valid_args_accepted_unchanged(self) -> None:
        """Already-valid arguments pass through unchanged."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="get_weather",
            raw_arguments='{"city": "London", "units": "celsius"}',
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_accepted
        assert decision.accepted_block is not None
        assert decision.accepted_block.arguments == {"city": "London", "units": "celsius"}
        assert decision.outcome.status == RepairStatus.VALID_UNCHANGED

    def test_unknown_tool_name_rejected(self) -> None:
        """An unrecognized tool name produces rejection."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="nonexistent_tool",
            raw_arguments='{"path": "/tmp/x"}',
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_rejected
        assert decision.accepted_block is None
        assert "not found" in decision.outcome.error.lower()

    def test_empty_name_rejected(self) -> None:
        """An empty tool name produces immediate rejection."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="",
            raw_arguments="{}",
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_rejected
        assert decision.accepted_block is None

    def test_none_name_rejected(self) -> None:
        """A None tool name produces immediate rejection."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name=None,
            raw_arguments="{}",
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_rejected
        assert decision.accepted_block is None

    def test_none_arguments_rejected(self) -> None:
        """None arguments result in no valid args, which may be rejected."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="read_file",
            raw_arguments=None,
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        # read_file requires "path" — None args can't satisfy that
        assert decision.is_rejected
        assert decision.accepted_block is None

    def test_missing_required_field_rejected(self) -> None:
        """Valid JSON but missing required schema field is rejected."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="read_file",
            raw_arguments='{"units": "celsius"}',
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        # read_file requires 'path', not 'units'
        assert decision.is_rejected
        assert decision.accepted_block is None

    def test_dict_arguments_passed_through(self) -> None:
        """Pre-parsed dict arguments work correctly."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="get_weather",
            raw_arguments={"city": "Tokyo"},
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_accepted
        assert decision.accepted_block is not None
        assert decision.accepted_block.arguments == {"city": "Tokyo"}

    def test_accepted_block_shape(self) -> None:
        """The accepted_block has the correct type and fields."""
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="read_file",
            raw_arguments='{"path": "/etc/hosts"}',
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS)
        assert decision.is_accepted
        block = decision.accepted_block
        assert block is not None
        assert block.type == "tool_call"
        assert block.name == "read_file"
        assert block.arguments == {"path": "/etc/hosts"}
        assert block.id == "tc_1"

    def test_repair_steps_recorded_on_accepted(self) -> None:
        """Repair steps are recorded when repair is needed."""
        from agent_interop.replay.types import CompatibilityKey
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="read_file",
            raw_arguments='{"file_path": "/tmp/x"}',  # alias for "path"
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(
            candidate,
            tools=SAMPLE_TOOLS,
            context=ToolTransactionContext(
                client_id="claude_code",
                repair_policy=RepairPolicy(field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK),
                compatibility_key=CompatibilityKey(client_id="claude_code", model_id="test-model"),
                compatibility_verified=True,
            ),
        )
        # The rename_aliased_fields rule should map file_path → path
        assert decision.is_accepted, f"Expected accepted, got {decision.outcome.status}: {decision.outcome.error}"
        assert decision.accepted_block is not None
        assert decision.accepted_block.id == "tc_1"
        assert decision.accepted_block.arguments == {"path": "/tmp/x"}


class TestToolCallDecision:
    """ToolCallDecision dataclass contract."""

    def test_rejected_decision(self) -> None:
        """A rejected decision has is_rejected=True and no accepted_block."""
        outcome = RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name="bad_tool",
            accepted=None,
            error="Tool not found",
        )
        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(
                id="tc_1", name="bad_tool", raw_arguments="{}",
                source_protocol="test", source_index=0,
            ),
            outcome=outcome,
            accepted_block=None,
        )
        assert decision.is_rejected
        assert not decision.is_accepted
        assert decision.accepted_block is None

    def test_accepted_decision(self) -> None:
        """An accepted decision has is_accepted=True and an accepted_block."""
        from agent_interop.abi import CanonicalToolCallBlock

        outcome = RepairOutcome(
            status=RepairStatus.VALID_UNCHANGED,
            call_name="read_file",
            accepted={"path": "/tmp/x"},
        )
        block = CanonicalToolCallBlock(
            id="tc_1", name="read_file", arguments={"path": "/tmp/x"},
        )
        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(
                id="tc_1", name="read_file", raw_arguments='{"path":"/tmp/x"}',
                source_protocol="test", source_index=0,
            ),
            outcome=outcome,
            accepted_block=block,
        )
        assert decision.is_accepted
        assert not decision.is_rejected
        assert decision.accepted_block is not None


class TestArrayRootRejection:
    """Bug 1: top-level list arguments are rejected uniformly, even when the
    tool schema declares an array root. The ``_raw_array`` wrapping path and the
    ``_schema_allows_array_root`` helper have been removed — the canonical ABI
    requires object-root tool schemas."""

    def test_array_root_schema_rejects_list_args(self) -> None:
        """A tool whose schema genuinely declares an array root still rejects
        list-shaped arguments — we do not silently wrap them in ``_raw_array``."""
        array_tool = CanonicalTool(
            name="append_items",
            description="Append items",
            input_schema={
                "type": "array",
                "items": {"type": "string"},
            },
        )
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="append_items",
            raw_arguments=["a", "b", "c"],  # top-level list
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=[array_tool])
        assert decision.is_rejected, (
            f"Expected rejected, got {decision.outcome.status}"
        )
        assert decision.accepted_block is None
        assert decision.outcome.status == RepairStatus.REJECTED
        # The rejection message references the object-root contract.
        assert "object-root" in decision.outcome.error

    def test_array_root_schema_never_wraps_raw_array(self) -> None:
        """No ``_raw_array`` key appears anywhere in accepted output for a
        list-shaped argument."""
        array_tool = CanonicalTool(
            name="append_items",
            description="Append items",
            input_schema={"type": "array", "items": {"type": "string"}},
        )
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="append_items",
            raw_arguments=["x", "y"],
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=[array_tool])
        # accepted_block is None on rejection, so no _raw_array key anywhere.
        assert decision.accepted_block is None
        assert decision.outcome.accepted is None
        if decision.outcome.accepted is not None:
            assert "_raw_array" not in decision.outcome.accepted

    def test_oneof_array_branch_also_rejected(self) -> None:
        """A schema with an array branch under oneOf also rejects list args."""
        tool = CanonicalTool(
            name="flex_input",
            description="Flexible",
            input_schema={
                "oneOf": [
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                    {"type": "array", "items": {"type": "string"}},
                ],
            },
        )
        candidate = RawToolCallCandidate(
            id="tc_1",
            name="flex_input",
            raw_arguments=["only", "a", "list"],
            source_protocol="openai_chat",
            source_index=0,
        )
        decision = _call_service(candidate, tools=[tool])
        assert decision.is_rejected
        assert decision.accepted_block is None


class TestBuildCorrectionRegression:
    """Bug 3: ``_build_correction`` had a duplicate line. After removing it the
    function must still produce a correct ``ToolCallCorrection``."""

    def test_build_correction_uses_final_issues(self) -> None:
        """final_issues (post-repair) populate the correction."""
        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="test", source_index=0,
        )
        outcome = RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name="read_file",
            accepted=None,
            error="missing path",
            final_issues=[
                SchemaIssue(
                    path=["path"],
                    keyword="required",
                    message=" 'path' is a required property",
                    expected="string",
                    actual="absent",
                ),
            ],
        )
        correction = _build_correction(candidate, outcome)
        assert correction is not None
        assert isinstance(correction, ToolCallCorrection)
        assert correction.tool_name == "read_file"
        assert correction.schema_keyword == "required"
        assert correction.issue_path == "path"

    def test_build_correction_falls_back_to_initial_issues(self) -> None:
        """When final_issues is empty, initial_issues are used."""
        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="test", source_index=0,
        )
        outcome = RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name="read_file",
            accepted=None,
            error="bad",
            initial_issues=[
                SchemaIssue(
                    path=[], keyword="type", message="bad type",
                    expected="string", actual="integer",
                ),
            ],
        )
        correction = _build_correction(candidate, outcome)
        assert correction is not None
        assert correction.schema_keyword == "type"
        assert correction.issue_path == "root"

    def test_build_correction_none_when_no_issues(self) -> None:
        """No issues → no correction."""
        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="test", source_index=0,
        )
        outcome = RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name="read_file",
            accepted=None,
            error="bad",
        )
        assert _build_correction(candidate, outcome) is None

class TestValidateBatchChoice:
    """validate_batch_choice() had no dedicated tests at all despite
    enforcing the tool_choice contract (none/named/required/auto) — a
    security/correctness-relevant gate that decides whether a batch of
    tool calls is even allowed to proceed."""

    def test_auto_always_valid_with_candidates(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.auto(), SAMPLE_TOOLS)
        assert err == ""

    def test_auto_always_valid_with_no_candidates(self) -> None:
        err = validate_batch_choice([], CanonicalToolChoice.auto(), SAMPLE_TOOLS)
        assert err == ""

    # ── none ──────────────────────────────────────────────────────────

    def test_none_valid_with_no_candidates(self) -> None:
        err = validate_batch_choice([], CanonicalToolChoice.none(), SAMPLE_TOOLS)
        assert err == ""

    def test_none_rejects_any_candidate(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.none(), SAMPLE_TOOLS)
        assert err != ""
        assert "read_file" in err

    def test_none_error_lists_all_candidate_names(self) -> None:
        candidates = [
            RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            RawToolCallCandidate(id="2", name="get_weather", raw_arguments="{}"),
        ]
        err = validate_batch_choice(candidates, CanonicalToolChoice.none(), SAMPLE_TOOLS)
        assert "read_file" in err
        assert "get_weather" in err

    # ── named ─────────────────────────────────────────────────────────

    def test_named_no_name_on_choice_rejected(self) -> None:
        err = validate_batch_choice([], CanonicalToolChoice.named(""), SAMPLE_TOOLS)
        assert "no tool name" in err.lower()

    def test_named_undeclared_tool_rejected(self) -> None:
        err = validate_batch_choice([], CanonicalToolChoice.named("delete_everything"), SAMPLE_TOOLS)
        assert "not in the declared tools" in err

    def test_named_no_candidates_rejected(self) -> None:
        err = validate_batch_choice([], CanonicalToolChoice.named("read_file"), SAMPLE_TOOLS)
        assert "not called" in err

    def test_named_mismatched_candidate_rejected(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="get_weather", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.named("read_file"), SAMPLE_TOOLS)
        assert "does not match" in err

    def test_named_matching_candidate_valid(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.named("read_file"), SAMPLE_TOOLS)
        assert err == ""

    def test_named_matches_via_case_canonicalization(self) -> None:
        """Candidate names are canonicalized before comparison — a safely
        normalizable case difference must not cause a false rejection."""
        candidates = [RawToolCallCandidate(id="1", name="Read_File", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.named("read_file"), SAMPLE_TOOLS)
        assert err == ""

    def test_named_one_matching_one_mismatched_rejected(self) -> None:
        candidates = [
            RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            RawToolCallCandidate(id="2", name="get_weather", raw_arguments="{}"),
        ]
        err = validate_batch_choice(candidates, CanonicalToolChoice.named("read_file"), SAMPLE_TOOLS)
        assert err != ""

    def test_named_empty_candidate_name_skipped(self) -> None:
        """A candidate with no name at all can't violate a named-choice
        check (nothing to compare) — current behavior skips it rather
        than treating it as a mismatch."""
        candidates = [RawToolCallCandidate(id="1", name=None, raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.named("read_file"), SAMPLE_TOOLS)
        assert err == ""

    # ── required ──────────────────────────────────────────────────────

    def test_required_no_candidates_rejected(self) -> None:
        err = validate_batch_choice([], CanonicalToolChoice.required(), SAMPLE_TOOLS)
        assert "missing" in err.lower()

    def test_required_undeclared_tool_rejected(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="delete_everything", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.required(), SAMPLE_TOOLS)
        assert "not a declared tool" in err

    def test_required_declared_tool_valid(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.required(), SAMPLE_TOOLS)
        assert err == ""

    def test_required_multiple_declared_tools_valid(self) -> None:
        candidates = [
            RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            RawToolCallCandidate(id="2", name="get_weather", raw_arguments="{}"),
        ]
        err = validate_batch_choice(candidates, CanonicalToolChoice.required(), SAMPLE_TOOLS)
        assert err == ""

    def test_required_one_undeclared_among_valid_rejected(self) -> None:
        candidates = [
            RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            RawToolCallCandidate(id="2", name="delete_everything", raw_arguments="{}"),
        ]
        err = validate_batch_choice(candidates, CanonicalToolChoice.required(), SAMPLE_TOOLS)
        assert err != ""

    def test_required_matches_via_case_canonicalization(self) -> None:
        candidates = [RawToolCallCandidate(id="1", name="READ_FILE", raw_arguments="{}")]
        err = validate_batch_choice(candidates, CanonicalToolChoice.required(), SAMPLE_TOOLS)
        assert err == ""


class TestRegenerationPath:
    """The module docstring has claimed 'Regeneration path (when enabled)
    produces REGENERATED status' as tested since this file's original
    Phase 4 gate — but no test ever actually exercised it. Covers the real
    wiring: tool_transaction_service -> _attempt_regeneration ->
    RegenerationOrchestrator -> regenerate_fn -> re-run through repair_one."""

    def test_regeneration_produces_regenerated_status_on_required_choice(self) -> None:
        from agent_interop.transaction import RepairBudget, ToolTransactionContext

        candidate = RawToolCallCandidate(
            id="tc_1",
            name="read_file",
            raw_arguments="{}",  # missing required "path" — deterministic repair can't invent it
            source_protocol="ollama_chat",
            source_index=0,
        )

        async def fake_regenerate(prompt: str) -> str:
            return '{"name": "read_file", "arguments": {"path": "/tmp/recovered.txt"}}'

        context = ToolTransactionContext(
            request_id="req-1",
            tool_choice=CanonicalToolChoice.required(),
            repair_policy=RepairPolicy(max_regenerations=1),
            regenerate_fn=fake_regenerate,
            budget=RepairBudget(),
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS, context=context)
        assert decision.is_accepted, f"expected accepted, got {decision.outcome.status}: {decision.outcome.error}"
        assert decision.outcome.status == RepairStatus.REGENERATED
        assert decision.accepted_block is not None
        assert decision.accepted_block.arguments == {"path": "/tmp/recovered.txt"}

    def test_regeneration_not_attempted_under_auto_choice(self) -> None:
        """AUTO mode alone must not trigger hidden regeneration — only
        REQUIRED/NAMED make the call mandatory enough to justify it."""
        from agent_interop.transaction import RepairBudget, ToolTransactionContext

        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="ollama_chat", source_index=0,
        )
        called = {"n": 0}

        async def fake_regenerate(prompt: str) -> str:
            called["n"] += 1
            return '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'

        context = ToolTransactionContext(
            request_id="req-1",
            tool_choice=CanonicalToolChoice.auto(),
            repair_policy=RepairPolicy(max_regenerations=1),
            regenerate_fn=fake_regenerate,
            budget=RepairBudget(),
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS, context=context)
        assert called["n"] == 0
        assert decision.is_rejected

    def test_regeneration_not_attempted_when_max_regenerations_zero(self) -> None:
        from agent_interop.transaction import RepairBudget, ToolTransactionContext

        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="ollama_chat", source_index=0,
        )
        called = {"n": 0}

        async def fake_regenerate(prompt: str) -> str:
            called["n"] += 1
            return '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'

        context = ToolTransactionContext(
            request_id="req-1",
            tool_choice=CanonicalToolChoice.required(),
            repair_policy=RepairPolicy(max_regenerations=0),
            regenerate_fn=fake_regenerate,
            budget=RepairBudget(),
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS, context=context)
        assert called["n"] == 0
        assert decision.is_rejected

    def test_regeneration_budget_exhausted_skips_attempt(self) -> None:
        from agent_interop.transaction import RepairBudget, ToolTransactionContext

        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="ollama_chat", source_index=0,
        )
        called = {"n": 0}

        async def fake_regenerate(prompt: str) -> str:
            called["n"] += 1
            return '{"name": "read_file", "arguments": {"path": "/tmp/x"}}'

        budget = RepairBudget(regeneration_attempts=1)  # already at the max of 1
        context = ToolTransactionContext(
            request_id="req-1",
            tool_choice=CanonicalToolChoice.required(),
            repair_policy=RepairPolicy(max_regenerations=1),
            regenerate_fn=fake_regenerate,
            budget=budget,
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS, context=context)
        assert called["n"] == 0
        assert decision.is_rejected

    def test_regeneration_fn_raising_falls_back_to_original_outcome(self) -> None:
        from agent_interop.transaction import RepairBudget, ToolTransactionContext

        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="ollama_chat", source_index=0,
        )

        async def failing_regenerate(prompt: str) -> str:
            raise RuntimeError("backend unavailable")

        context = ToolTransactionContext(
            request_id="req-1",
            tool_choice=CanonicalToolChoice.required(),
            repair_policy=RepairPolicy(max_regenerations=1),
            regenerate_fn=failing_regenerate,
            budget=RepairBudget(),
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS, context=context)
        assert decision.is_rejected
        assert decision.outcome.status != RepairStatus.REGENERATED

    def test_regeneration_producing_still_invalid_output_stays_rejected(self) -> None:
        """A regenerated response that STILL fails schema validation must
        not be force-accepted — it goes back through the full repair
        pipeline, which can reject it again."""
        from agent_interop.transaction import RepairBudget, ToolTransactionContext

        candidate = RawToolCallCandidate(
            id="tc_1", name="read_file", raw_arguments="{}",
            source_protocol="ollama_chat", source_index=0,
        )

        async def bad_regenerate(prompt: str) -> str:
            # Still missing the required "path" field.
            return '{"name": "read_file", "arguments": {}}'

        context = ToolTransactionContext(
            request_id="req-1",
            tool_choice=CanonicalToolChoice.required(),
            repair_policy=RepairPolicy(max_regenerations=1),
            regenerate_fn=bad_regenerate,
            budget=RepairBudget(),
        )
        decision = _call_service(candidate, tools=SAMPLE_TOOLS, context=context)
        assert decision.is_rejected
        assert decision.outcome.status != RepairStatus.REGENERATED
