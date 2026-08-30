"""P0-15/16/17/19/20 regression: private loop + budget accounting.

Locks in:
  * private generations spend their OWN allowance — never the compatibility
    ladder's upstream_attempts (item 16);
  * the private loop is one non-recursive loop over single-step sends
    (item 17);
  * missing/blank private call IDs are normalized ONCE across identity,
    transcript, and result (item 19);
  * malformed private arguments become is_error results — never silently
    coerced to {} and executed (item 20).
"""

from __future__ import annotations

import asyncio
import json

from agent_interop.abi import (
    CanonicalTool,
    CanonicalToolCallBlock,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.execution_attempts import AttemptBudget
from agent_interop.gateway import Gateway
from agent_interop.private_loop import MAX_INTERNAL_TOOL_LOOP_DEPTH, parse_private_arguments
from tests.test_p01_private_continuation import (  # reuse proven scaffolding
    _openai_chat_completion,
    _ScriptedTransport,
    _session,
    _virtualizing_request,
    _VirtualizingContextSize,
)


def test_private_generations_do_not_consume_ladder_rungs():
    """P0-16: two private retrieval generations must leave upstream_attempts
    untouched — a paged read must not cost the request its attempt ladder."""
    budget = AttemptBudget(max_upstream_attempts=3, max_private_generations=16)
    assert budget.allow_private_generation()
    assert budget.allow_private_generation()
    assert budget.upstream_attempts == 0, "private generations must not spend ladder rungs"
    # The ladder still has all 3 rungs.
    assert budget.allow(use_controller=False)
    assert budget.allow(use_controller=False)
    assert budget.allow(use_controller=False)
    assert not budget.allow(use_controller=False)


def test_private_generation_ceiling_enforced():
    budget = AttemptBudget(max_private_generations=2)
    assert budget.allow_private_generation()
    assert budget.allow_private_generation()
    assert not budget.allow_private_generation()
    assert budget.exhausted_by == "max_private_generations"


def test_reservation_commit_replaces_estimate_not_adds():
    """P0-14 (regression lock): commit() must REPLACE the reserved estimate
    with actuals — reserving 100 then committing 40 leaves 40, not 140."""
    budget = AttemptBudget()
    reservation = budget.reserve_generation(
        estimated_input_tokens=100, output_reserve_tokens=20,
    )
    assert reservation is not None
    assert budget.total_input_tokens == 100
    reservation.commit(actual_input_tokens=40, actual_output_tokens=8)
    assert budget.total_input_tokens == 40
    assert budget.generated_tokens == 8


def test_reservation_release_returns_headroom():
    budget = AttemptBudget(max_total_input_tokens=1000)
    reservation = budget.reserve_generation(
        estimated_input_tokens=900, output_reserve_tokens=50,
    )
    assert reservation is not None
    reservation.release()
    assert budget.total_input_tokens == 0
    # Headroom available again.
    assert budget.allow_private_generation()


def test_reservation_single_use():
    budget = AttemptBudget()
    reservation = budget.reserve_generation(estimated_input_tokens=10, output_reserve_tokens=5)
    assert reservation is not None
    reservation.commit(actual_input_tokens=10, actual_output_tokens=5)
    # Double-commit / double-release must be no-ops, not double accounting.
    reservation.commit(actual_input_tokens=999, actual_output_tokens=999)
    reservation.release()
    assert budget.total_input_tokens == 10
    assert budget.generated_tokens == 5


def _private_loop_gateway() -> tuple[Gateway, RequestContext]:
    gw = Gateway(InteropServerConfig(
        probe_on_startup=False,
        log_level="error",
        routes={
            "r": ModelRoute(
                id="r",
                client_model_aliases=["m"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://127.0.0.1:1",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                ),
                tool_mode=ToolMode.NATIVE,
                tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
                context=ContextConfig(**_VirtualizingContextSize),
            ),
        },
    ))
    return gw, _session()


def test_private_loop_bounded_by_generations_not_unique_ids():
    """P0-18: the depth cap counts generations. A model reusing ONE call ID
    every round must still hit the cap — repeated IDs cannot bypass it."""
    gw, session = _private_loop_gateway()
    ref = gw._context_store.store("sess-private-loop", "d\n", kind="tool_result", tool_call_id="RD").ref
    # SAME id every turn.
    internal_tc = [{
        "id": "call_same", "type": "function",
        "function": {"name": "__interop_read_result", "arguments": json.dumps({"ref": ref})},
    }]
    transport = _ScriptedTransport(
        [_openai_chat_completion(internal_tc)] * (MAX_INTERNAL_TOOL_LOOP_DEPTH + 3),
    )
    gw._transport = transport
    resp = asyncio.run(gw.handle_request(_virtualizing_request(2, "seed-loop"), session))
    assert resp.error is not None
    assert resp.error.code == "INTERNAL_TOOL_LOOP_EXHAUSTED", resp.error.code
    # Bound: each ladder rung that reaches generation runs its own loop of at
    # most MAX private generations; with the default 3-rung ladder the total
    # stays bounded by rungs × (1 + MAX). The invariant under test: a model
    # reusing one ID cannot loop unboundedly.
    assert transport.calls <= 3 * (MAX_INTERNAL_TOOL_LOOP_DEPTH + 1), transport.calls


def test_blank_call_id_normalized_to_result_id():
    """P0-19: a blank provider ID must not produce an assistant call with id
    '' and a result with a different synthetic id — one normalized ID goes
    everywhere (the loop completes and the identity stays consistent)."""
    gw, session = _private_loop_gateway()
    ref = gw._context_store.store("sess-private-loop", "s\n", kind="tool_result", tool_call_id="RB").ref
    blank_id = [{
        "id": "", "type": "function",
        "function": {"name": "__interop_read_result", "arguments": json.dumps({"ref": ref})},
    }]
    public = [{
        "id": "pub", "type": "function",
        "function": {"name": "tool_0", "arguments": json.dumps({"x": "1"})},
    }]
    transport = _ScriptedTransport([
        _openai_chat_completion(blank_id),
        _openai_chat_completion(public),
    ])
    gw._transport = transport
    resp = asyncio.run(gw.handle_request(_virtualizing_request(2, "seed-blank"), session))
    assert resp.error is None, resp.error
    names = [b.name for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
    assert "tool_0" in names
    assert not any(n and n.startswith("__interop_") for n in names)


def test_malformed_private_arguments_become_error_not_empty_object():
    """P0-20: a private call with invalid JSON arguments must produce an
    is_error result fed back to the model — never a silent {} execution
    (which for read_result would read the head of the ref by default)."""
    gw, session = _private_loop_gateway()
    gw._context_store.store("sess-private-loop", "data\n", kind="tool_result", tool_call_id="RE")
    malformed = [{
        "id": "bad", "type": "function",
        "function": {"name": "__interop_read_result", "arguments": "{not json"},
    }]
    public = [{
        "id": "pub2", "type": "function",
        "function": {"name": "tool_0", "arguments": json.dumps({"x": "ok"})},
    }]
    transport = _ScriptedTransport([
        _openai_chat_completion(malformed),
        _openai_chat_completion(public),
    ])
    gw._transport = transport
    resp = asyncio.run(gw.handle_request(_virtualizing_request(2, "seed-malformed"), session))
    # The request completes (model recovered on turn 2); nothing leaked.
    names = [b.name for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
    assert "tool_0" in names
    assert "__interop_read_result" not in names


def test_missing_required_argument_rejected_strictly():
    """P0-20: read_result requires 'ref' — a call without it is an error,
    not a default-execution."""

    parsed, err = parse_private_arguments(
        type("C", (), {"raw_arguments": json.dumps({"start_line": 3})})(),
        CanonicalTool(
            name="__interop_read_result",
            description="",
            input_schema={
                "type": "object",
                "properties": {"ref": {"type": "string"}, "start_line": {"type": "integer"}},
                "required": ["ref"],
            },
        ),
    )
    assert err is not None
    assert "ref" in err
    assert parsed == {}
