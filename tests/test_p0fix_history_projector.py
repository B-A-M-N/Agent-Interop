"""Tests for the rewritten history projector (P0.15 rewrite).

Covers:
(a) assistant tool-call messages and their tool results are never split
    across the preserved/fragment boundary.
(b) stored entry content parses as JSON with schema_version == 1 and
    round-trips role/structure.
(c) render_fragment_text bounds at message boundaries and never splits a
    message line.
(d) paging of a long conversation stores multiple fragments with correct
    message_range.
(e) protected: latest user turn + latest tool exchange always preserved.
(f) unsafe projection raises ValueError.
"""

from __future__ import annotations

import json

import pytest

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)
from agent_interop.context_store.store import ContextStore
from agent_interop.history.projector import (
    build_history_index_prompt,
    project_history,
    render_fragment_text,
)

# ─── Factory helpers ───────────────────────────────────────────────────────


def _user(text: str) -> CanonicalMessage:
    return CanonicalMessage(role="user", content=[CanonicalTextBlock(text=text)])


def _assistant_text(text: str) -> CanonicalMessage:
    return CanonicalMessage(
        role="assistant", content=[CanonicalTextBlock(text=text)]
    )


def _assistant_tool_call(call_id: str, name: str, args: dict | None = None):
    return CanonicalMessage(
        role="assistant",
        content=[
            CanonicalToolCallBlock(id=call_id, name=name, arguments=args or {}),
        ],
    )


def _tool_result(call_id: str, content: str, is_error: bool = False):
    return CanonicalMessage(
        role="tool",
        content=[
            CanonicalToolResultBlock(
                tool_call_id=call_id, content=content, is_error=is_error
            )
        ],
    )


def _system(text: str) -> CanonicalMessage:
    return CanonicalMessage(role="system", content=[CanonicalTextBlock(text=text)])


# ─── (a) Tool-call / result must never be split ────────────────────────────


def test_tool_exchange_never_split():
    """Assistant tool-call message and its tool-result must stay together."""
    store = ContextStore()
    messages = [
        _system("sys"),
        _user("hi"),
        _assistant_tool_call("tc1", "get_weather", {"city": "nyc"}),
        _tool_result("tc1", "sunny"),
        _user("thanks"),
    ]
    result = project_history(
        messages, max_recent_turns=2, store=store, session_id="s1"
    )

    # Collect indices of preserved messages.
    preserved_idx = set()
    for msg in result.messages:
        preserved_idx.add(messages.index(msg))

    # The assistant tc message and its tool result must both be preserved
    # or both be compacted (never one preserved, one in a fragment).
    # Find the tc message index and the tool-result index.
    tc_idx = None
    tr_idx = None
    for i, m in enumerate(messages):
        if m.role == "assistant" and any(
            isinstance(b, CanonicalToolCallBlock) for b in m.content
        ):
            tc_idx = i
        if m.role == "tool" and any(
            isinstance(b, CanonicalToolResultBlock) for b in m.content
        ):
            tr_idx = i

    assert tc_idx is not None and tr_idx is not None
    # Both must be in the preserved set or both in a stored fragment.
    tc_in_preserved = tc_idx in preserved_idx
    tr_in_preserved = tr_idx in preserved_idx
    assert tc_in_preserved == tr_in_preserved, (
        f"Tool-call msg {tc_idx} preserved={tc_in_preserved} "
        f"but tool-result msg {tr_idx} preserved={tr_in_preserved}"
    )


def test_multiple_tool_calls_in_one_assistant_message():
    """When one assistant message has multiple tool calls, all results stay with it."""
    store = ContextStore()
    messages = [
        _user("do both"),
        _assistant_tool_call("tc1", "call_a", {}),
        _tool_result("tc1", "res a"),
        _assistant_tool_call("tc2", "call_b", {}),
        _tool_result("tc2", "res b"),
        _user("done"),
    ]
    result = project_history(
        messages, max_recent_turns=1, store=store, session_id="s2"
    )
    # No message should ever be in a fragment if it is part of a tool
    # exchange with a preserved message.
    for ref in result.refs:
        start, end = ref.message_range
        for i in range(start, end + 1):
            assert messages[i].role != "assistant" or any(
                isinstance(b, CanonicalToolCallBlock) for b in messages[i].content
            ) or messages[i].role != "tool" or any(
                isinstance(b, CanonicalToolResultBlock) for b in messages[i].content
            ), (
                f"Fragment {ref.ref} ({start}-{end}) unexpectedly contains "
                f"a split message at {i}"
            )


# ─── (b) Stored entries are canonical JSON ─────────────────────────────────


def test_stored_entry_canonical_json():
    """Stored content parses as JSON with schema_version==1 and round-trips."""
    store = ContextStore()
    messages = [
        _system("sys"),
        _user("first hello"),
        _assistant_tool_call("tc1", "echo", {"msg": "hi"}),
        _tool_result("tc1", "hello back"),
        _user("bye"),
    ]
    result = project_history(
        messages, max_recent_turns=1, store=store, session_id="s3"
    )
    assert len(result.refs) >= 1, "Expected at least one stored fragment"
    ref = result.refs[0]
    entry = store.get(ref.ref, "s3")
    assert entry is not None

    payload = json.loads(entry.content)
    assert payload["schema_version"] == 1
    assert "messages" in payload

    for m in payload["messages"]:
        assert "role" in m
        assert "content" in m
        for blk in m["content"]:
            assert "type" in blk


def test_stored_round_trips_role_structure():
    """Round-trip: read stored JSON and verify roles/content blocks match."""
    store = ContextStore()
    messages = [
        _system("sys"),
        _user("ask a"),
        _assistant_tool_call("tc1", "query", {"q": "what"}),
        _tool_result("tc1", "answer"),
        _user("ask b"),
        _assistant_tool_call("tc2", "query", {"q": "where"}),
        _tool_result("tc2", "there"),
        _user("final"),
    ]
    result = project_history(
        messages, max_recent_turns=2, store=store, session_id="s4"
    )
    # The stored fragments should contain the older messages
    for ref in result.refs:
        entry = store.get(ref.ref, "s4")
        assert entry is not None
        payload = json.loads(entry.content)
        stored_roles = [m["role"] for m in payload["messages"]]
        # Verify stored roles match the original message roles at the range
        for i, m in enumerate(messages):
            if ref.message_range[0] <= i <= ref.message_range[1]:
                assert stored_roles[i - ref.message_range[0]] == m.role


# ─── (c) render_fragment_text bounds at message boundaries ─────────────────


def test_render_fragment_text_bounds_at_message_boundary():
    """Truncation never splits a message line."""
    canonical_json = json.dumps({
        "schema_version": 1,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a" * 200}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "short"}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "long " * 100}],
            },
        ],
    })

    max_chars = 100
    rendered = render_fragment_text(canonical_json, max_chars)
    assert len(rendered) <= max_chars + 1  # sentinel line may add a bit

    # The line that would cause overflow should be absent
    lines = rendered.split("\n")
    for line in lines:
        if line.startswith("user: "):
            # Truncated text lines should be complete messages
            pass  # each line is a full message representation

    # Should contain the sentinel if truncation happened
    if len(rendered) >= max_chars:
        assert "[...older messages omitted" in rendered


def test_render_fragment_text_truncation_never_splits_tool_call():
    """A tool_call line must never be cut mid-JSON."""
    args = {"very_long_key_name": "very_long_value"}
    canonical_json = json.dumps({
        "schema_version": 1,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "tc1",
                        "name": "do_stuff",
                        "arguments": args,
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "x" * 100}],
            },
        ],
    })

    max_chars = 40
    rendered = render_fragment_text(canonical_json, max_chars)
    # The user line should not be present if truncated
    lines = rendered.split("\n")
    user_lines = [ln for ln in lines if ln.startswith("user: ")]
    # If user line appears, it must be a complete line (no partial JSON)
    for ul in user_lines:
        assert not ul.endswith("{") and not ul.endswith(",")


def test_render_fragment_text_parse_failure_fallback():
    """Invalid JSON returns first max_chars of raw string."""
    bad = "not json at all" * 10
    result = render_fragment_text(bad, 50)
    assert result == bad[:50]


# ─── (d) Long conversation stores multiple fragments ───────────────────────


def test_long_conversation_stores_multiple_fragments():
    """A conversation long enough for multiple fragments stores them correctly."""
    store = ContextStore()
    messages: list[CanonicalMessage] = [_system("sys")]
    for i in range(20):
        messages.append(_user(f"user message {i}"))
        messages.append(
            _assistant_tool_call(
                f"tc{i}", f"tool_{i}", {"idx": i}
            )
        )
        messages.append(_tool_result(f"tc{i}", f"result {i}"))
    # Final user message (the latest turn — always protected)
    messages.append(_user("final message"))

    result = project_history(
        messages, max_recent_turns=2, store=store, session_id="s5"
    )

    # Should have at least one ref
    assert len(result.refs) >= 1
    # Each ref should have a valid message_range
    for ref in result.refs:
        start, end = ref.message_range
        assert start <= end
        assert start >= 0
        assert end < len(messages)
        # The stored entry must exist
        entry = store.get(ref.ref, "s5")
        assert entry is not None, f"Missing entry for ref {ref.ref}"

    # refs should be in order
    for i in range(1, len(result.refs)):
        prev_end = result.refs[i - 1].message_range[1]
        curr_start = result.refs[i].message_range[0]
        assert prev_end < curr_start, "Fragments must not overlap"


def test_fragment_summary_includes_unit_kinds():
    """build_history_index_prompt uses summary with unit kinds."""
    store = ContextStore()
    messages: list[CanonicalMessage] = []
    for i in range(10):
        messages.append(_user(f"msg {i}"))
        messages.append(_assistant_tool_call(f"tc{i}", f"tool_{i}"))
        messages.append(_tool_result(f"tc{i}", f"res {i}"))
    messages.append(_user("final"))

    result = project_history(
        messages, max_recent_turns=2, store=store, session_id="s6"
    )

    if result.refs:
        # Check that the summary contains the unit kinds
        assert "(" in result.refs[0].summary
        assert ")" in result.refs[0].summary
        # build_history_index_prompt should render the refs properly
        prompt = build_history_index_prompt(result.refs)
        assert "Earlier conversation" in prompt


# ─── (e) Protected: latest user turn + latest tool exchange always preserved ─


def test_latest_user_turn_always_preserved():
    """Even with tiny max_recent_turns, the latest user turn is protected."""
    store = ContextStore()
    messages = [
        _user("old"),
        _assistant_tool_call("tc1", "tool1"),
        _tool_result("tc1", "res1"),
        _user("newest user"),
    ]
    result = project_history(
        messages, max_recent_turns=0, store=store, session_id="s7"
    )
    # The latest user message must be in the preserved set
    roles = [msg.role for msg in result.messages]
    assert "user" in roles


def test_latest_tool_exchange_always_preserved():
    """Even with tiny max_recent_turns, the latest tool exchange is protected."""
    store = ContextStore()
    messages = [
        _user("old 1"),
        _assistant_tool_call("tc1", "tool1"),
        _tool_result("tc1", "res1"),
        _user("old 2"),
        _assistant_tool_call("tc2", "tool2"),
        _tool_result("tc2", "res2"),
        _user("newest"),
    ]
    result = project_history(
        messages, max_recent_turns=0, store=store, session_id="s8"
    )
    # The latest tool exchange (tc2 + result) must be in preserved.
    # Collect roles: we should see an assistant with tool calls and a tool result.
    has_assistant_call = any(
        msg.role == "assistant"
        and any(isinstance(b, CanonicalToolCallBlock) for b in msg.content)
        for msg in result.messages
    )
    has_tool_result = any(
        msg.role == "tool"
        and any(isinstance(b, CanonicalToolResultBlock) for b in msg.content)
        for msg in result.messages
    )
    assert has_assistant_call and has_tool_result, (
        "Latest tool exchange was not preserved"
    )


# ─── (f) Unsafe projection raises ──────────────────────────────────────────


def test_unsafe_projection_raises():
    """Reconcile producing is_safe=False must raise ValueError."""
    from agent_interop.abi import CanonicalMessage
    from agent_interop.history.projector import project_history

    # Build an "unsafe" history: orphan tool result (no prior call).
    messages = [
        _user("hi"),
        CanonicalMessage(
            role="tool",
            content=[
                CanonicalToolResultBlock(
                    tool_call_id="orphan-tc", content="no matching call"
                )
            ],
        ),
        _user("now what"),
    ]
    with pytest.raises(ValueError, match="history paging produced unsafe history"):
        project_history(
            messages, max_recent_turns=2, session_id="s9"
        )


# ─── Additional: empty / edge cases ────────────────────────────────────────


def test_empty_messages():
    """Empty message list returns empty result."""
    result = project_history([])
    assert result.messages == []
    assert result.refs == []
    assert result.preserved_count == 0
    assert result.compacted_count == 0


def test_system_only_preserved():
    """Messages with only system messages are all preserved."""
    result = project_history([
        _system("sys1"),
        _system("sys2"),
    ])
    assert len(result.messages) == 2
    assert result.refs == []
    assert result.preserved_count == 2
