"""Integration tests for previously-untested components.

Covers:
- Loop detection
- Tool-call extraction from model output
- Request validation with backend constraints
- Execution coordinator lifecycle
- Provider metadata round-trip through upstream codec
"""

from __future__ import annotations

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalReasoningBlock,
    CanonicalTextBlock,
    CanonicalTool,
)
from agent_interop.execution import InteropRequestExecution
from agent_interop.extraction import ToolCallEnvelopeExtractor
from agent_interop.request_validation import (
    BackendConstraints,
    validate_tool_contract,
)
from agent_interop.session import SessionManager

# ─── Loop detection: argument-digest disambiguation ───────────────────────
#
# Regression tests for the argument-digest bug in SessionState.record_repair.
# Previously every repair/rejection was recorded with an empty argument_digest
# and a hardcoded result_class of "success", so (a) repairing the same tool
# three times with DIFFERENT arguments was falsely flagged as a loop, and (b)
# rejections were conflated with successes in the signature window.


class TestArgumentDigestLoopDetection:
    def test_different_argument_repairs_not_flagged(self):
        """Same tool, three DIFFERENT argument digests — NOT a loop.

        With the old bug (empty digest) these three distinct repairs would
        collapse to one signature and be flagged. With the fix each digest is
        distinct, so no signature repeats.
        """
        mgr = SessionManager()
        mgr.begin_request("sess-1")
        mgr.record_repair("sess-1", "", "read_file", 1, "repaired", argument_digest="digest_a")
        mgr.record_repair("sess-1", "", "read_file", 1, "repaired", argument_digest="digest_b")
        mgr.record_repair("sess-1", "", "read_file", 1, "repaired", argument_digest="digest_c")
        state = mgr.get("sess-1")
        assert state is not None
        assert not state.flagged

    def test_identical_argument_repairs_flagged(self):
        """Same tool, SAME argument digest three times — IS a loop.

        Proves the signature mechanism still fires when arguments are
        genuinely identical (digest disambiguation must not mask real loops).
        """
        mgr = SessionManager()
        mgr.begin_request("sess-2")
        for _ in range(3):
            mgr.record_repair(
                "sess-2", "", "read_file", 1, "repaired", argument_digest="same_digest"
            )
        state = mgr.get("sess-2")
        assert state is not None
        assert state.flagged


# ─── Tool-Call Extraction ──────────────────────────────────────────────────


class TestToolCallExtraction:
    def test_basic_envelope_extraction(self):
        extractor = ToolCallEnvelopeExtractor()
        text = 'Hello\n<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>\nDone'
        tools = [CanonicalTool(name="read_file", input_schema={"type": "object", "properties": {"path": {"type": "string"}}})]
        result = extractor.extract(
            [CanonicalTextBlock(text=text)],
            tools=tools,
            envelope="tool_call",
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].name == "read_file"

    def test_fenced_code_not_extracted(self):
        extractor = ToolCallEnvelopeExtractor()
        text = '```json\n<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>\n```'
        tools = [CanonicalTool(name="read_file", input_schema={"type": "object"})]
        result = extractor.extract(
            [CanonicalTextBlock(text=text)],
            tools=tools,
            envelope="tool_call",
        )
        assert len(result.candidates) == 0

    def test_unclosed_fenced_code_not_extracted(self):
        """Regression: a fence that never closes (truncated generation, or
        a model that starts an example and never finishes the markdown) was
        NOT masked at all — the tool call inside it matched the primary
        regex directly and was executed. An unclosed fence is not "not a
        fence"; it's a fence whose closing marker never arrived."""
        extractor = ToolCallEnvelopeExtractor()
        text = (
            'Here is an example:\n```json\n'
            '<tool_call>{"name": "delete_file", "arguments": {"path": "/etc/passwd"}}</tool_call>'
        )
        tools = [CanonicalTool(name="delete_file", input_schema={"type": "object"})]
        result = extractor.extract(
            [CanonicalTextBlock(text=text)],
            tools=tools,
            envelope="tool_call",
        )
        assert len(result.candidates) == 0

    def test_real_call_after_closed_fence_still_extracted(self):
        """The unclosed-fence fix must not swallow legitimate content that
        follows a properly CLOSED fence earlier in the same message."""
        extractor = ToolCallEnvelopeExtractor()
        text = (
            'Example:\n```json\n{"x": 1}\n```\n'
            'Now for real: <tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'
        )
        tools = [CanonicalTool(name="read_file", input_schema={"type": "object"})]
        result = extractor.extract(
            [CanonicalTextBlock(text=text)],
            tools=tools,
            envelope="tool_call",
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].name == "read_file"

    def test_multiple_tool_calls(self):
        extractor = ToolCallEnvelopeExtractor()
        text = (
            '<tool_call>{"name": "read_file", "arguments": {"path": "/a"}}</tool_call>\n'
            '<tool_call>{"name": "read_file", "arguments": {"path": "/b"}}</tool_call>'
        )
        tools = [CanonicalTool(name="read_file", input_schema={"type": "object", "properties": {"path": {"type": "string"}}})]
        result = extractor.extract(
            [CanonicalTextBlock(text=text)],
            tools=tools,
            envelope="tool_call",
        )
        assert len(result.candidates) == 2


# ─── Request Validation with Backend Constraints ───────────────────────────


class TestBackendConstraints:
    def test_default_validation_passes(self):
        tools = [CanonicalTool(name="read_file", input_schema={"type": "object"})]
        valid, _issues = validate_tool_contract(tools, None)
        assert valid

    def test_custom_name_length_limit(self):
        constraints = BackendConstraints(max_name_length=10)
        tools = [CanonicalTool(name="a_very_long_tool_name", input_schema={"type": "object"})]
        valid, issues = validate_tool_contract(tools, None, backend_constraints=constraints)
        assert not valid
        assert any("exceeds" in i.message for i in issues)

    def test_custom_name_pattern(self):
        import re
        constraints = BackendConstraints(name_pattern=re.compile(r"^[a-z_]+$"))
        tools = [CanonicalTool(name="ReadFile", input_schema={"type": "object"})]
        valid, issues = validate_tool_contract(tools, None, backend_constraints=constraints)
        assert not valid
        assert any("not allowed" in i.message for i in issues)

    def test_max_tools_limit(self):
        constraints = BackendConstraints(max_tools=2)
        tools = [
            CanonicalTool(name=f"tool_{i}", input_schema={"type": "object"})
            for i in range(5)
        ]
        valid, issues = validate_tool_contract(tools, None, backend_constraints=constraints)
        assert not valid
        assert any("Too many tools" in i.message for i in issues)


# ─── Execution Coordinator Lifecycle ───────────────────────────────────────


class TestExecutionCoordinator:
    def test_finalize_response_accepted(self):
        from agent_interop.abi import CanonicalResponse, CanonicalStopReason
        exec_ = InteropRequestExecution()
        resp = CanonicalResponse(content=[], stop_reason=CanonicalStopReason.END_TURN)
        exec_.finalize_response(resp)
        assert exec_.response_outcome == "accepted"
        assert exec_.finished_at is not None

    def test_finalize_response_error(self):
        from agent_interop.abi import CanonicalError, CanonicalResponse, CanonicalStopReason
        exec_ = InteropRequestExecution()
        resp = CanonicalResponse(
            content=[],
            stop_reason=CanonicalStopReason.INVALID_OUTPUT,
            error=CanonicalError(code="TEST", message="test"),
        )
        exec_.finalize_response(resp)
        assert exec_.response_outcome == "error"

    def test_finalize_none_is_error(self):
        exec_ = InteropRequestExecution()
        exec_.finalize_error(None)
        assert exec_.response_outcome == "error"

    def test_finalize_error_explicit(self):
        exec_ = InteropRequestExecution()
        exec_.finalize_error()
        assert exec_.response_outcome == "error"

    def test_finalize_cancelled(self):
        exec_ = InteropRequestExecution()
        exec_.finalize_cancelled()
        assert exec_.response_outcome == "cancelled"

    def test_record_tool_decision(self):
        from agent_interop.abi import RepairOutcome, RepairStatus
        exec_ = InteropRequestExecution()
        outcome = RepairOutcome(status=RepairStatus.REPAIRED, call_name="read_file")
        exec_.record_tool_decision("read_file", "tc_1", outcome, accepted=True)
        assert len(exec_.tool_decisions) == 1
        assert exec_.tool_decisions[0].tool_name == "read_file"
        assert exec_.tool_decisions[0].accepted is True

    def test_record_malformed_frame(self):
        exec_ = InteropRequestExecution()
        exec_.record_malformed_frame(5, "parse error", "raw data here")
        assert len(exec_.raw_frame_evidence) == 1
        assert exec_.raw_frame_evidence[0]["ordinal"] == 5

    def test_to_sanitized_dict(self):
        exec_ = InteropRequestExecution()
        exec_.finalize_error()
        d = exec_.to_sanitized_dict()
        assert d["response_outcome"] == "error"
        assert "elapsed_ms" in d


# ─── Provider Metadata Round-Trip (Upstream Codec) ─────────────────────────


class TestProviderMetadataRoundTrip:
    def test_reasoning_content_decoded(self):
        from agent_interop.upstreams.openai_chat import OpenAIChatCodec
        codec = OpenAIChatCodec()
        body = {
            "choices": [{
                "message": {
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think step by step...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        decoded = codec.decode_response(body)
        assert len(decoded.content) == 2
        assert isinstance(decoded.content[0], CanonicalReasoningBlock)
        assert decoded.content[0].content == "Let me think step by step..."
        assert isinstance(decoded.content[1], CanonicalTextBlock)
        assert decoded.content[1].text == "The answer is 42."

    def test_reasoning_content_rendered(self):
        from agent_interop.upstreams.openai_chat import _render_message
        msg = CanonicalMessage(
            role="assistant",
            content=[
                CanonicalReasoningBlock(content="thinking..."),
                CanonicalTextBlock(text="Hello"),
            ],
        )
        rendered = _render_message(msg)
        assert rendered["role"] == "assistant"
        assert rendered["content"] == "Hello"
        assert rendered["reasoning_content"] == "thinking..."

    def test_no_reasoning_when_absent(self):
        from agent_interop.upstreams.openai_chat import OpenAIChatCodec
        codec = OpenAIChatCodec()
        body = {
            "choices": [{
                "message": {"content": "Just text."},
                "finish_reason": "stop",
            }],
        }
        decoded = codec.decode_response(body)
        assert len(decoded.content) == 1
        assert isinstance(decoded.content[0], CanonicalTextBlock)


# ─── Argument digest canonicalization ──────────────────────────────────────
#
# Regression test for the bug where ``Gateway._compute_argument_digest`` re-
# serialized a JSON *string* argument verbatim (quote-wrapped, escaped) instead
# of parsing it first. Two semantically-identical JSON strings with different
# key order / whitespace must produce the SAME digest, otherwise a model that
# regenerates the same logical arguments in a different encoding would be
# treated as a fresh signature and a real repair loop could go undetected.


class TestArgumentDigestCanonicalization:
    def _gateway(self):
        from agent_interop.gateway import Gateway

        return Gateway.__new__(Gateway)

    def test_string_same_args_different_key_order(self):
        """JSON string arguments: key-order-insensitive, whitespace-insensitive."""
        gw = self._gateway()
        a = gw._compute_argument_digest('{"a":1,"b":2}')
        b = gw._compute_argument_digest('{"b": 2, "a": 1}')
        assert a == b
        assert a != ""  # non-trivial arguments must not collapse to empty

    def test_string_vs_equivalent_dict(self):
        """A JSON string and its parsed dict form must digest identically."""
        gw = self._gateway()
        from_str = gw._compute_argument_digest('{"x":1,"y":2}')
        from_dict = gw._compute_argument_digest({"y": 2, "x": 1})
        assert from_str == from_dict

    def test_dict_different_key_order(self):
        """Dict arguments are already canonicalized via sort_keys (sanity check)."""
        gw = self._gateway()
        a = gw._compute_argument_digest({"b": 2, "a": 1})
        b = gw._compute_argument_digest({"a": 1, "b": 2})
        assert a == b

    def test_nested_string_args(self):
        """Nested structures inside a JSON string must also canonicalize."""
        gw = self._gateway()
        a = gw._compute_argument_digest('{"outer":{"b":2,"a":1}}')
        b = gw._compute_argument_digest('{"outer": {"a": 1, "b": 2}}')
        assert a == b

    def test_empty_and_none_arguments(self):
        """Falsy arguments collapse to the empty digest (argument-less calls)."""
        gw = self._gateway()
        assert gw._compute_argument_digest(None) == ""
        assert gw._compute_argument_digest("") == ""
        assert gw._compute_argument_digest({}) == ""

    def test_malformed_json_string_does_not_raise(self):
        """A non-JSON string must not crash the non-fatal bookkeeping path."""
        gw = self._gateway()
        # Should not raise; produces a stable fallback digest.
        digest = gw._compute_argument_digest("not json at all")
        assert isinstance(digest, str)
        assert len(digest) == 16
