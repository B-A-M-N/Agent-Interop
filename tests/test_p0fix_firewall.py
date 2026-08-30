"""Tests for the identity-based output firewall in private_loop.py.

Scenario matrix
---------------
(a) identity mode, marker syntax in prose with empty refs → no raise (fix)
(b) identity mode, actual ref in text → raises
(c) tool result with identity call_id → raises + filter drops it
(d) internal tool name in call block → raises
(e) identity=None preserves legacy pattern behaviour
(f) assert_model_request_invariant detects ref inside CanonicalToolResultBlock
"""

from __future__ import annotations

import pytest

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)
from agent_interop.private_loop import (
    InternalIdentity,
    assert_model_request_invariant,
    assert_no_internal_leakage,
    filter_public_blocks,
)

# ---------------------------------------------------------------------------
# Helpers — build canonical blocks for tests
# ---------------------------------------------------------------------------

MARKER_TEXT = "[Interop result ref: totally-unrelated]"


def _text_block(text: str) -> CanonicalTextBlock:
    return CanonicalTextBlock(text=text)


def _tool_call(name: str, call_id: str = "tc1") -> CanonicalToolCallBlock:
    return CanonicalToolCallBlock(name=name, id=call_id)


def _tool_result(content: str, call_id: str = "tc1") -> CanonicalToolResultBlock:
    return CanonicalToolResultBlock(tool_call_id=call_id, content=content)


def _response(blocks) -> CanonicalResponse:
    return CanonicalResponse(content=list(blocks))


# ---------------------------------------------------------------------------
# (a) identity mode: marker text with empty refs must NOT raise
# ---------------------------------------------------------------------------

def test_identity_mode_marker_text_no_leak_with_empty_refs():
    """False-positive fix: quoting the marker syntax in prose is not a leak."""
    identity = InternalIdentity(
        call_ids=frozenset(),
        refs=frozenset(),
        tool_names=frozenset(),
    )
    blocks = [_text_block(f"Here is a ref: {MARKER_TEXT}")]
    resp = _response(blocks)
    # Should NOT raise — no identity refs match
    assert_no_internal_leakage(resp, identity=identity)


# ---------------------------------------------------------------------------
# (b) identity mode: text containing an actual identity ref raises
# ---------------------------------------------------------------------------

def test_identity_mode_actual_ref_in_text_raises():
    """A real 24-byte token ref inside text must be caught."""
    actual_ref = "aaaaaaaaabbbbbbbbbbcccccccccc"  # 24 bytes
    identity = InternalIdentity(
        call_ids=frozenset(),
        refs=frozenset([actual_ref]),
        tool_names=frozenset(),
    )
    blocks = [_text_block(f"Data chunk: {actual_ref}")]
    resp = _response(blocks)
    with pytest.raises(ValueError, match="internal ref leaked"):
        assert_no_internal_leakage(resp, identity=identity)


# ---------------------------------------------------------------------------
# (c) tool result whose tool_call_id is in identity.call_ids → raise & drop
# ---------------------------------------------------------------------------

def test_identity_mode_private_result_raises_and_is_dropped():
    """Private call results must not reach the client."""
    private_id = "tc_private"
    identity = InternalIdentity(
        call_ids=frozenset([private_id]),
        refs=frozenset(),
        tool_names=frozenset(),
    )
    blocks = [_tool_result("private data", private_id)]
    resp = _response(blocks)

    with pytest.raises(ValueError, match="private call result"):
        assert_no_internal_leakage(resp, identity=identity)

    # filter_public_blocks must also drop it
    filtered = filter_public_blocks(blocks, identity=identity)
    assert len(filtered) == 0


def test_identity_mode_public_result_survives_filter():
    """Public tool results are NOT dropped."""
    identity = InternalIdentity(
        call_ids=frozenset(["tc_private"]),
        refs=frozenset(),
        tool_names=frozenset(),
    )
    blocks = [_tool_result("public data", "tc_public")]
    filtered = filter_public_blocks(blocks, identity=identity)
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# (d) internal tool name call raises
# ---------------------------------------------------------------------------

def test_identity_mode_internal_tool_call_raises():
    """Internal tool calls must not reach the client."""
    identity = InternalIdentity(
        call_ids=frozenset(),
        refs=frozenset(),
        tool_names=frozenset(["__interop_read_result"]),
    )
    blocks = [_tool_call("__interop_read_result", "tc1")]
    resp = _response(blocks)
    with pytest.raises(ValueError, match="internal tool call"):
        assert_no_internal_leakage(resp, identity=identity)


def test_identity_mode_internal_tool_prefix_also_catches():
    """Even without the name in tool_names, the __interop_ prefix catches it."""
    identity = InternalIdentity(
        call_ids=frozenset(),
        refs=frozenset(),
        tool_names=frozenset(),
    )
    blocks = [_tool_call("__interop_read_result", "tc1")]
    resp = _response(blocks)
    with pytest.raises(ValueError, match="internal tool call"):
        assert_no_internal_leakage(resp, identity=identity)


def test_identity_mode_client_tool_survives():
    """A regular client tool call is fine."""
    identity = InternalIdentity(
        call_ids=frozenset(),
        refs=frozenset(),
        tool_names=frozenset(["__interop_read_result"]),
    )
    blocks = [_tool_call("write_file", "tc1")]
    resp = _response(blocks)
    assert_no_internal_leakage(resp, identity=identity)


# ---------------------------------------------------------------------------
# (e) identity=None preserves legacy pattern behaviour
# ---------------------------------------------------------------------------

def test_legacy_mode_marker_text_raises():
    """When identity is None the marker text pattern must still raise."""
    blocks = [_text_block(MARKER_TEXT)]
    resp = _response(blocks)
    with pytest.raises(ValueError, match="internal retrieval ref"):
        assert_no_internal_leakage(resp)


def test_legacy_mode_filter_drops_marker_in_tool_result():
    """Legacy filter must drop tool-result blocks containing the marker pattern."""
    blocks = [_tool_result(MARKER_TEXT)]
    filtered = filter_public_blocks(blocks)
    assert len(filtered) == 0


def test_legacy_mode_text_blocks_pass_through():
    """Legacy filter does NOT drop text blocks (that was never its job)."""
    blocks = [_text_block(MARKER_TEXT)]
    filtered = filter_public_blocks(blocks)
    assert len(filtered) == 1


def test_legacy_mode_clean_text_survives():
    """Legacy mode with clean content must pass through."""
    blocks = [_text_block("Hello world")]
    resp = _response(blocks)
    assert_no_internal_leakage(resp)
    filtered = filter_public_blocks(blocks)
    assert len(filtered) == 1


# ---------------------------------------------------------------------------
# (f) assert_model_request_invariant: ref inside CanonicalToolResultBlock
# ---------------------------------------------------------------------------

def test_invariant_detects_ref_in_tool_result_no_tool():
    """Ref inside a tool-result block raises when __interop_read_result is absent."""
    actual_ref = "aaaaaaaaabbbbbbbbbbcccccccccc"
    blocks = [_tool_result(f"[Interop result ref: {actual_ref}]")]
    msg = CanonicalMessage(role="assistant", content=blocks)
    req = CanonicalRequest(
        messages=[msg],
        tools=[],  # no __interop_read_result
    )

    class _Invocation:
        model_request = req

    with pytest.raises(ValueError, match="Invariant violation"):
        assert_model_request_invariant(_Invocation())


def test_invariant_passes_when_tool_present():
    """Same request passes when __interop_read_result is in the tool surface."""
    actual_ref = "aaaaaaaaabbbbbbbbbbcccccccccc"
    blocks = [_tool_result(f"[Interop result ref: {actual_ref}]")]
    msg = CanonicalMessage(role="assistant", content=blocks)
    req = CanonicalRequest(
        messages=[msg],
        tools=[CanonicalTool(name="__interop_read_result")],
    )

    class _Invocation:
        model_request = req

    # Should NOT raise
    assert_model_request_invariant(_Invocation())


# ---------------------------------------------------------------------------
# Integration: mixed response with identity mode
# ---------------------------------------------------------------------------

def test_identity_mode_clean_response_survives():
    """A clean response with public blocks passes identity mode."""
    identity = InternalIdentity(
        call_ids=frozenset(["tc_private"]),
        refs=frozenset(["aabbccddee"],),
        tool_names=frozenset(["__interop_read_result"]),
    )
    blocks = [
        _text_block("Here is the result."),
        _tool_call("write_file", "tc_public"),
    ]
    resp = _response(blocks)
    assert_no_internal_leakage(resp, identity=identity)
    filtered = filter_public_blocks(blocks, identity=identity)
    assert len(filtered) == 2
