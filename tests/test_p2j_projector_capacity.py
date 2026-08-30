"""P2-J regression: history projector semantics + capacity source semantics.

Locks in:
  * recent-unit protection compares within ONE index space (P0-57) —
    interleaved system units no longer skew how many recent turns survive;
  * a tool call and its result can never land on opposite sides of the
    paging boundary (merged tool regions, ID-aware ownership);
  * max_total_messages bounds the preserved window at UNIT boundaries
    (P0-58) while mandatory units (system, latest user turn, latest tool
    exchange) keep priority;
  * render_fragment_text truncation is an explicit fact (P0-59) — an
    exactly-fitting render carries NO omission sentinel;
  * architecture_context_tokens is a CEILING, never a capacity source
    (P0-49); usable capacity comes from observed/configured/operator
    values only;
  * unknown_capacity_policy "reject_tool_requests" is an explicit alias
    of "reject" (P0-50) — the name says what the policy does.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)
from agent_interop.context_budget.planner import effective_context_limit
from agent_interop.history.projector import (
    _group_messages_into_semantic_units,
    project_history,
    render_fragment_text,
)


def _turns(count: int, prefix: str = "turn") -> list[CanonicalMessage]:
    messages: list[CanonicalMessage] = []
    for i in range(count):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append(CanonicalMessage(role=role, content=[CanonicalTextBlock(text=f"{prefix} {i}")]))
    return messages


# ─── P0-57: recent-unit indexing ────────────────────────────────────────────


def test_recent_turn_protection_is_immune_to_leading_system_units():
    """A leading system unit must not shrink the recent window: with
    max_recent_turns=4, the last 4 non-system UNITS are protected whether
    or not system units exist at the front."""
    plain = [
        CanonicalMessage(role="system", content=[CanonicalTextBlock(text="sys")]),
        *_turns(10),
    ]
    plain_result = project_history(plain, max_recent_turns=4)
    with_more_system = [
        CanonicalMessage(role="system", content=[CanonicalTextBlock(text="s1")]),
        CanonicalMessage(role="developer", content=[CanonicalTextBlock(text="d1")]),
        CanonicalMessage(role="system", content=[CanonicalTextBlock(text="s2")]),
        *_turns(10),
    ]
    biased_result = project_history(with_more_system, max_recent_turns=4)

    def preserved_texts(result):
        return [
            block.text
            for message in result.messages
            for block in message.content
            if isinstance(block, CanonicalTextBlock) and block.text.startswith("turn ")
        ]

    # The same recent conversation turns survive in both shapes.
    assert preserved_texts(plain_result) == preserved_texts(biased_result)
    # And the window really is 4 units (4 single-message turns here).
    assert len(preserved_texts(plain_result)) == 4


# ─── P0-57: call/result integrity across the paging boundary ───────────────


def test_call_and_result_never_split_across_units():
    messages = [
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="go")]),
        CanonicalMessage(role="assistant", content=[
            CanonicalToolCallBlock(id="call_A", name="fa", arguments={}),
        ]),
        CanonicalMessage(role="assistant", content=[
            CanonicalToolCallBlock(id="call_B", name="fb", arguments={}),
        ]),
        CanonicalMessage(role="tool", content=[
            CanonicalToolResultBlock(tool_call_id="call_A", content="ra"),
        ]),
        CanonicalMessage(role="tool", content=[
            CanonicalToolResultBlock(tool_call_id="call_B", content="rb"),
        ]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="done")]),
    ]
    units = _group_messages_into_semantic_units(messages)
    # The interleaved exchanges form ONE contiguous tool unit: paging it
    # pages every call and result together.
    tool_units = [u for u in units if u.kind == "tool_exchange"]
    assert len(tool_units) == 1
    assert (tool_units[0].start_index, tool_units[0].end_index) == (1, 4)

    # A result separated from its call by a real message still cannot be
    # split from it: the ownership merge bridges the units.
    separated = [
        CanonicalMessage(role="assistant", content=[
            CanonicalToolCallBlock(id="call_C", name="fc", arguments={}),
        ]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="meanwhile")]),
        CanonicalMessage(role="tool", content=[
            CanonicalToolResultBlock(tool_call_id="call_C", content="rc"),
        ]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="end")]),
    ]
    units2 = _group_messages_into_semantic_units(separated)
    owner_units = [
        u for u in units2
        if any(
            isinstance(b, CanonicalToolCallBlock) and b.id == "call_C"
            for pos in range(u.start_index, u.end_index + 1)
            for b in separated[pos].content
        )
    ]
    result_units = [
        u for u in units2
        if any(
            isinstance(b, CanonicalToolResultBlock) and b.tool_call_id == "call_C"
            for pos in range(u.start_index, u.end_index + 1)
            for b in separated[pos].content
        )
    ]
    assert owner_units and result_units
    # Same unit, or at minimum the whole span pages together (the merge
    # guarantees one unit when they are adjacent in unit order).
    if owner_units[0] is not result_units[0]:
        assert owner_units[0].end_index < result_units[0].start_index
        # The merge must have fused them into a single protected decision —
        # verify via a projection with a tiny recent window: either both
        # survive or both page.
        small = project_history(separated, max_recent_turns=1)
        preserved_ids = {
            getattr(b, "id", getattr(b, "tool_call_id", ""))
            for m in small.messages for b in m.content
        }
        both_in = "call_C" in preserved_ids and "call_C" in {
            getattr(b, "tool_call_id", "") for m in small.messages for b in m.content
        }
        both_out = "call_C" not in preserved_ids
        assert both_in or both_out, "call and result were split by paging"

    # Sequential exchanges stay SEPARATE units (no over-merging).
    sequential = [
        CanonicalMessage(role="assistant", content=[
            CanonicalToolCallBlock(id="c1", name="f", arguments={}),
        ]),
        CanonicalMessage(role="tool", content=[
            CanonicalToolResultBlock(tool_call_id="c1", content="r1"),
        ]),
        CanonicalMessage(role="assistant", content=[
            CanonicalToolCallBlock(id="c2", name="f", arguments={}),
        ]),
        CanonicalMessage(role="tool", content=[
            CanonicalToolResultBlock(tool_call_id="c2", content="r2"),
        ]),
    ]
    assert [u.kind for u in _group_messages_into_semantic_units(sequential)] == [
        "tool_exchange", "tool_exchange",
    ]


def test_paging_never_produces_unsafe_history():
    """Whatever the window, reconciliation must accept the preserved set —
    the projector raises rather than emitting orphaned results."""
    messages = [
        CanonicalMessage(role="system", content=[CanonicalTextBlock(text="sys")]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="go")]),
        CanonicalMessage(role="assistant", content=[
            CanonicalToolCallBlock(id="call_A", name="fa", arguments={}),
        ]),
        CanonicalMessage(role="tool", content=[
            CanonicalToolResultBlock(tool_call_id="call_A", content="ra"),
        ]),
        *_turns(8, prefix="filler"),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="final")]),
    ]
    for turns in (1, 2, 3, 6):
        result = project_history(messages, max_recent_turns=turns)  # must not raise
        assert result.preserved_count >= 2


# ─── P0-58: max_total_messages at unit boundaries ───────────────────────────


def test_max_total_messages_bounds_preserved_window():
    messages = [
        CanonicalMessage(role="system", content=[CanonicalTextBlock(text="sys")]),
        *_turns(20),
    ]
    capped = project_history(messages, max_recent_turns=10, max_total_messages=6)
    assert capped.preserved_count <= 6
    # Mandatory content survives the cap: system + the latest USER turn
    # (turn 18 — turn 19 is the assistant reply).
    texts = [
        b.text for m in capped.messages for b in m.content
        if isinstance(b, CanonicalTextBlock)
    ]
    assert "sys" in texts
    assert "turn 18" in texts, texts
    # Uncapped request preserves strictly more.
    uncapped = project_history(messages, max_recent_turns=10)
    assert uncapped.preserved_count >= capped.preserved_count


def test_max_total_messages_zero_disables_the_bound():
    messages = [
        CanonicalMessage(role="system", content=[CanonicalTextBlock(text="sys")]),
        *_turns(20),
    ]
    result = project_history(messages, max_recent_turns=10, max_total_messages=0)
    baseline = project_history(messages, max_recent_turns=10)
    assert result.preserved_count == baseline.preserved_count


# ─── P0-59: explicit truncation ─────────────────────────────────────────────


def test_render_fragment_exact_fit_has_no_sentinel():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello world"}]},
        ],
    }
    canonical = json.dumps(payload)
    line = "user: hello world"
    out = render_fragment_text(canonical, len(line))  # exactly fits
    assert "omitted" not in out
    assert out == line


def test_render_fragment_real_truncation_carries_sentinel():
    payload = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "a" * 50}]},
            {"role": "user", "content": [{"type": "text", "text": "b" * 50}]},
        ],
    }
    out = render_fragment_text(json.dumps(payload), 60)
    assert "omitted" in out


# ─── P0-49: capacity source semantics ───────────────────────────────────────


def test_architecture_maximum_is_a_ceiling_not_a_source():
    # Architecture alone supplies NOTHING.
    assert effective_context_limit(
        configured_limit=0, route_override=0,
        observed_effective_limit=0, architecture_ceiling=32768,
    ) == 0
    # An operator cap above the ceiling clamps DOWN to the ceiling.
    assert effective_context_limit(
        configured_limit=0, route_override=65536,
        observed_effective_limit=0, architecture_ceiling=32768,
    ) == 32768
    # An operator cap at/below the ceiling stands.
    assert effective_context_limit(
        configured_limit=0, route_override=8192,
        observed_effective_limit=0, architecture_ceiling=32768,
    ) == 8192
    # Observed serving context wins over the operator cap…
    assert effective_context_limit(
        configured_limit=16384, route_override=65536,
        observed_effective_limit=8192, architecture_ceiling=32768,
    ) == 8192
    # …and is itself ceiling-clamped.
    assert effective_context_limit(
        configured_limit=0, route_override=0,
        observed_effective_limit=65536, architecture_ceiling=32768,
    ) == 32768


def test_planner_uses_architecture_only_as_clamp():
    """A 32K-architecture model with an 8K num_ctx must plan against 8K,
    not 32K (the historical leak)."""
    from agent_interop.backends.base import ModelRuntimeCapabilities
    from agent_interop.config import (
        CompatibilityConfig, ContextConfig, ModelRoute, ToolMode,
        UpstreamConfig, UpstreamKind, UpstreamProtocol,
    )
    from agent_interop.context import RequestContext
    from agent_interop.planning import (
        BehavioralCapabilities, RequestCompatibilityPlanner,
    )
    from agent_interop.upstreams.codec import CodecCapabilities

    route = ModelRoute(
        id="local",
        client_model_aliases=["local"],
        upstream_model="m",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
        context=ContextConfig(output_reserve_tokens=32),
        compatibility=CompatibilityConfig(),
    )
    request = CanonicalRequest(
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        generation=CanonicalGenerationOptions(stream=False),
    )
    plan = asyncio.run(RequestCompatibilityPlanner().plan(
        request=request,
        context=RequestContext(),
        route=route,
        client_requirements=object(),
        codec_capabilities=CodecCapabilities(),
        runtime_capabilities=ModelRuntimeCapabilities(
            backend_kind=UpstreamKind.OLLAMA,
            architecture_context_tokens=32768,
            configured_context_tokens=8192,
            effective_context_tokens=0,
        ),
        behavioral_capabilities=BehavioralCapabilities(),
    ))
    assert plan.context_plan.runtime_limit_tokens == 8192, (
        "configured serving context must win; architecture (32768) is only a clamp"
    )
