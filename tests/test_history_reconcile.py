"""Tests for the HistoryReconciliationResult contract.

The gateway must:
* Replace request.messages with the reconciled version.
* Reject the request when the history is unsafe (orphan results,
  unpaired calls, duplicate IDs, result-before-call).
* Synthesize consistent IDs for missing tool_call_id / tool result
  tool_call_id values.
* Pair empty-ID results with the immediately preceding assistant group.
"""

from __future__ import annotations

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)
from agent_interop.history.reconcile import ToolExchange, reconcile_history


def _call_msg(call_id: str = "tc_1", name: str = "read_file") -> CanonicalMessage:
    return CanonicalMessage(
        role="assistant",
        content=[CanonicalToolCallBlock(
            id=call_id,
            name=name,
            arguments={"path": "/tmp/x"},
        )],
    )


def _result_msg(call_id: str = "tc_1") -> CanonicalMessage:
    return CanonicalMessage(
        role="tool",
        content=[CanonicalToolResultBlock(
            tool_call_id=call_id,
            content="file content",
        )],
    )


def _user_msg(text: str = "hello") -> CanonicalMessage:
    return CanonicalMessage(role="user", content=[CanonicalTextBlock(text=text)])


class TestHistoryReconciliation:
    def test_safe_history_returns_unchanged_messages(self):
        msgs = [_user_msg(), _call_msg(), _result_msg()]
        result = reconcile_history(msgs)
        assert result.is_safe
        assert result.messages == msgs
        assert result.diagnostics == []

    def test_missing_call_id_paired_with_preceding_call(self):
        """Adjacent empty-ID call and result should be paired."""
        msgs = [
            _user_msg(),
            CanonicalMessage(
                role="assistant",
                content=[CanonicalToolCallBlock(
                    id="",
                    name="read_file",
                    arguments={"path": "/tmp/x"},
                )],
            ),
            _result_msg(""),
        ]
        result = reconcile_history(msgs)
        # The empty-ID result should be paired with the empty-ID call
        # from the immediately preceding assistant message.
        asst = result.messages[1]
        first_block = asst.content[0]
        assert isinstance(first_block, CanonicalToolCallBlock)
        new_id = first_block.id
        assert new_id
        assert any("synthesized_id" in a for a in result.actions)
        # Result should adopt the same synthesized ID — history is safe
        tool_msg = result.messages[2]
        result_block = tool_msg.content[0]
        assert isinstance(result_block, CanonicalToolResultBlock)
        assert result_block.tool_call_id == new_id
        assert result.is_safe

    def test_orphan_result_marks_unsafe(self):
        """A result referencing a call that never appeared is unsafe."""
        msgs = [
            _user_msg(),
            _call_msg("tc_real"),
            _result_msg("tc_orphan"),
        ]
        result = reconcile_history(msgs)
        assert not result.is_safe
        assert any("result_before_call" in d or "orphan" in d for d in result.diagnostics)

    def test_unpaired_call_with_continuation_marks_unsafe(self):
        """A call with no result, followed by more messages, is unsafe."""
        msgs = [
            _user_msg(),
            _call_msg("tc_unpaired"),
            _user_msg("continue"),
        ]
        result = reconcile_history(msgs)
        assert not result.is_safe
        assert any("unpaired_call" in d for d in result.diagnostics)

    def test_unpaired_call_at_end_is_safe(self):
        """A call at the end of history (no continuation) is safe."""
        msgs = [
            _user_msg(),
            _call_msg("tc_last"),
        ]
        result = reconcile_history(msgs)
        assert result.is_safe

    def test_duplicate_call_id_marks_unsafe(self):
        msgs = [
            _user_msg(),
            _call_msg("tc_dup"),
            _call_msg("tc_dup"),
            _result_msg("tc_dup"),
        ]
        result = reconcile_history(msgs)
        assert any("duplicate_call_id" in d for d in result.diagnostics)
        assert not result.is_safe

    def test_result_before_call_marks_unsafe(self):
        """A result referencing a call that appears later is unsafe."""
        msgs = [
            _user_msg(),
            _result_msg("tc_future"),
            _call_msg("tc_future"),
        ]
        result = reconcile_history(msgs)
        assert not result.is_safe
        assert any("result_before_call" in d for d in result.diagnostics)

    def test_duplicate_result_marks_unsafe(self):
        """Multiple results for the same call is unsafe."""
        msgs = [
            _user_msg(),
            _call_msg("tc_1"),
            _result_msg("tc_1"),
            _result_msg("tc_1"),
        ]
        result = reconcile_history(msgs)
        assert not result.is_safe
        assert any("duplicate_result" in d for d in result.diagnostics)

    def test_ambiguous_empty_result_with_parallel_calls(self):
        """Empty result ID with multiple unresolved calls is ambiguous/unsafe."""
        msgs = [
            _user_msg(),
            CanonicalMessage(
                role="assistant",
                content=[
                    CanonicalToolCallBlock(id="", name="read_file", arguments={"path": "/a"}),
                    CanonicalToolCallBlock(id="", name="write_file", arguments={"path": "/b", "content": "x"}),
                ],
            ),
            CanonicalMessage(
                role="tool",
                content=[CanonicalToolResultBlock(tool_call_id="", content="result")],
            ),
        ]
        result = reconcile_history(msgs)
        assert not result.is_safe
        assert any("ambiguous" in d for d in result.diagnostics)


class TestReconciliationExchanges:
    """Tests for the structured call/result exchange view on the result."""

    def test_paired_call_and_result(self):
        """A call followed by its result yields one fully-populated exchange."""
        call_id = "tc_pair"
        msgs = [
            _user_msg(),
            CanonicalMessage(
                role="assistant",
                content=[CanonicalToolCallBlock(
                    id=call_id,
                    name="read_file",
                    arguments={"path": "/tmp/x"},
                )],
            ),
            CanonicalMessage(
                role="tool",
                content=[CanonicalToolResultBlock(
                    tool_call_id=call_id,
                    content="file content",
                )],
            ),
        ]
        result = reconcile_history(msgs)
        assert len(result.exchanges) == 1
        exchange = result.exchanges[0]
        assert isinstance(exchange, ToolExchange)
        assert exchange.call_id == call_id
        assert exchange.call is not None
        assert exchange.call.name == "read_file"
        assert exchange.result is not None
        assert exchange.result.tool_call_id == call_id
        # Call is in message index 1, result in index 2 (index 0 is the user).
        assert exchange.call_message_index == 1
        assert exchange.result_message_index == 2

    def test_unresolved_call_has_no_result(self):
        """A pending (unresolved) call yields an exchange with result=None."""
        call_id = "tc_pending"
        msgs = [
            _user_msg(),
            CanonicalMessage(
                role="assistant",
                content=[CanonicalToolCallBlock(
                    id=call_id,
                    name="read_file",
                    arguments={"path": "/tmp/x"},
                )],
            ),
        ]
        result = reconcile_history(msgs)
        assert len(result.exchanges) == 1
        exchange = result.exchanges[0]
        assert exchange.call_id == call_id
        assert exchange.call is not None
        assert exchange.result is None
        assert exchange.result_message_index is None
        assert exchange.call_message_index == 1


class TestDeterministicSynthesizedIds:
    """MVP-13: synthesized IDs must be stable across repeated reconciliation
    of the SAME request (retries, replay) — a random UUID fallback means
    the same malformed history produces a different ID every time."""

    def _msg_missing_call_id(self) -> CanonicalMessage:
        return CanonicalMessage(
            role="assistant",
            content=[CanonicalToolCallBlock(id="", name="read_file", arguments={"path": "/tmp/x"})],
        )

    def test_missing_call_id_is_stable_given_session_and_request(self):
        msgs = [self._msg_missing_call_id()]
        r1 = reconcile_history(msgs, session_id="sess-abc", request_id="req-123")
        r2 = reconcile_history(msgs, session_id="sess-abc", request_id="req-123")

        id1 = r1.messages[0].content[0].id
        id2 = r2.messages[0].content[0].id
        assert id1 == id2
        assert id1  # actually synthesized, not left empty

    def test_missing_call_id_differs_across_sessions(self):
        msgs = [self._msg_missing_call_id()]
        r1 = reconcile_history(msgs, session_id="sess-a", request_id="req-1")
        r2 = reconcile_history(msgs, session_id="sess-b", request_id="req-1")

        assert r1.messages[0].content[0].id != r2.messages[0].content[0].id

    def test_missing_call_id_random_fallback_when_no_context(self):
        """With neither session_id nor request_id, the documented fallback
        (random UUID) still applies — no context means no stable identity
        to derive a digest from."""
        msgs = [self._msg_missing_call_id()]
        r1 = reconcile_history(msgs)
        r2 = reconcile_history(msgs)

        assert r1.messages[0].content[0].id != r2.messages[0].content[0].id

    def test_orphan_result_id_is_stable_given_session_and_request(self):
        msgs = [
            _user_msg(),
            CanonicalMessage(
                role="tool",
                content=[CanonicalToolResultBlock(tool_call_id="", content="orphaned")],
            ),
        ]
        r1 = reconcile_history(msgs, session_id="sess-abc", request_id="req-123")
        r2 = reconcile_history(msgs, session_id="sess-abc", request_id="req-123")

        id1 = r1.messages[1].content[0].tool_call_id
        id2 = r2.messages[1].content[0].tool_call_id
        assert id1 == id2
        assert id1
        assert r1.is_safe is False  # orphan result remains a rejection regardless
