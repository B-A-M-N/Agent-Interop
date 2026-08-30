"""P1-F regression: single serialization pass + linear tool-surface budget.

Locks in:
  * build_request_cost_snapshot prices the request ONCE and its fingerprint
    equals the gateway's canonical form;
  * price_tool_subset is byte-exact against estimate_tool_schema_tokens for
    arbitrary subsets (no re-serialization);
  * the schema-budget selection loop is O(n) but selects IDENTICALLY to the
    historical O(n²) re-serialization form;
  * rank_tools hoists query tokenization and consumes a registry term cache;
  * the compatibility key fingerprint reaches the evidence path from the
    snapshot (requirements carry it), not a second serialization.
"""

from __future__ import annotations

import random

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.context_budget.estimator import (
    build_request_cost_snapshot,
    estimate_request_context,
    estimate_tool_schema_tokens,
    price_tool_subset,
)
from agent_interop.tool_surface.lexical import build_tool_terms_cache, rank_tools
from agent_interop.tool_surface.selector import _single_tool_item_bytes


def _tools(n: int, seed: int = 0) -> list[CanonicalTool]:
    rng = random.Random(seed)
    return [
        CanonicalTool(
            name=f"tool_{i}",
            description=f"tool {i} " + "x" * rng.randint(0, 120),
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "p" * rng.randint(0, 60)},
                    "n": {"type": "integer"},
                },
            },
        )
        for i in range(n)
    ]


def test_snapshot_matches_estimator_for_full_surface():
    tools = _tools(10, seed=1)
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        system=[CanonicalTextBlock(text="system prompt")],
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hello")])],
        tools=tools,
    )
    snapshot = build_request_cost_snapshot(request)
    assert snapshot.tool_schema_tokens == estimate_tool_schema_tokens(tools).input_tokens
    # The fingerprint equals the canonical evidence-key form.
    import hashlib
    import json
    canonical = sorted(
        ({"name": t.name, "schema": t.input_schema} for t in tools),
        key=lambda entry: entry["name"],
    )
    assert snapshot.tool_schema_fingerprint == hashlib.sha256(
        json.dumps(canonical, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def test_price_tool_subset_is_exact_for_every_subset():
    tools = _tools(12, seed=2)
    snapshot = build_request_cost_snapshot(CanonicalRequest(tools=tools))
    rng = random.Random(9)
    for _ in range(300):
        subset = tuple(t for t in tools if rng.random() < 0.5)
        assert price_tool_subset(snapshot, subset) == estimate_tool_schema_tokens(subset).input_tokens
    assert price_tool_subset(snapshot, ()) == 0


def test_estimate_request_context_with_snapshot_matches_direct():
    tools = _tools(8, seed=3)
    visible = tools[:5]
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        system=[CanonicalTextBlock(text="sys")],
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        tools=tools,
    )
    snapshot = build_request_cost_snapshot(request)
    direct = estimate_request_context(request, visible_tools=visible, output_reserve_tokens=256)
    via_snapshot = estimate_request_context(
        request, visible_tools=visible, output_reserve_tokens=256, snapshot=snapshot,
    )
    assert direct.total_required_tokens == via_snapshot.total_required_tokens
    assert direct.tool_schema_tokens == via_snapshot.tool_schema_tokens


def _historical_budget_selection(visible, max_schema):
    """The pre-P1.3 O(n²) form: re-serialize the accumulating list per tool."""
    budgeted = []
    for tool in visible:
        if estimate_tool_schema_tokens((*budgeted, tool)).input_tokens > max_schema:
            continue
        budgeted.append(tool)
    return tuple(budgeted)


def test_incremental_budget_selection_matches_historical():
    """The linear budget loop must select exactly the same tools the O(n²)
    form selected — the optimization changes cost, not outcomes."""
    from agent_interop.abi import CanonicalToolChoice
    from agent_interop.config import ToolSurfaceConfig, ToolSurfaceMode
    from agent_interop.tool_surface.selector import ToolSurfacePlanner

    rng = random.Random(5)
    for trial in range(6):
        tools = _tools(rng.randint(3, 15), seed=trial)
        request = CanonicalRequest(
            model=CanonicalModelReference(requested_name="m"),
            messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="work")])],
            tools=tools,
            tool_choice=CanonicalToolChoice.auto(),
        )
        snapshot = build_request_cost_snapshot(CanonicalRequest(tools=tools))
        for max_schema in (80, 200, 600, 5000):
            # Surface WITHOUT budget enforcement, then apply both forms.
            plan = ToolSurfacePlanner().plan(
                request,
                ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=len(tools), max_schema_tokens=0),
                cost_snapshot=snapshot,
            )
            order = list(plan.visible_tools)
            rng.shuffle(order)
            # The live planner with a budget uses the incremental loop.
            budgeted_plan = ToolSurfacePlanner().plan(
                request,
                ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=len(tools), max_schema_tokens=max_schema),
                cost_snapshot=snapshot,
            )
            # Compare incremental vs historical on the order the planner
            # produced (visible list order is deterministic).
            assert tuple(budgeted_plan.visible_tools) == _historical_budget_selection(
                plan.visible_tools, max_schema,
            ), (trial, max_schema)


def test_rank_tools_with_registry_cache_matches_uncached():
    tools = tuple(_tools(10, seed=6))
    cache = build_tool_terms_cache(tools)
    query = "search the file system for tool_3 config"
    assert rank_tools(query, tools, tool_terms=cache) == rank_tools(query, tools)
    # Exact-name match still scores highest.
    assert rank_tools(query, tools, tool_terms=cache)[0].name == "tool_3"


def test_unknown_tool_falls_back_to_serialization():
    """A tool absent from the snapshot (added post-snapshot, e.g. a private
    capability) is priced by direct serialization — never silently 0."""
    known = _tools(3, seed=7)
    snapshot = build_request_cost_snapshot(CanonicalRequest(tools=known))
    stranger = CanonicalTool(name="late_tool", description="d", input_schema={"type": "object"})
    assert price_tool_subset(snapshot, (stranger,)) == estimate_tool_schema_tokens((stranger,)).input_tokens


def test_single_tool_item_bytes_matches_snapshot():
    tools = _tools(5, seed=8)
    snapshot = build_request_cost_snapshot(CanonicalRequest(tools=tools))
    for tool in tools:
        assert _single_tool_item_bytes(tool) == snapshot.tool_item_bytes[tool.name]


def test_budget_loop_performs_no_repeated_serializations(monkeypatch):
    """P1.3: with a cost snapshot, the schema-budget loop prices candidates
    by arithmetic — the only serialization in plan() is the fingerprint.
    The historical form serialized once PER TOOL (O(n²) total)."""
    import json as _json

    from agent_interop.config import ToolSurfaceMode
    from agent_interop.tool_surface import selector as selector_mod

    calls = {"dumps": 0}
    real_dumps = _json.dumps

    def counting_dumps(*args, **kwargs):
        calls["dumps"] += 1
        return real_dumps(*args, **kwargs)

    monkeypatch.setattr(selector_mod.json, "dumps", counting_dumps)

    tools = _tools(20, seed=10)
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="use tool_7 now")])],
        tools=tools,
        tool_choice=CanonicalToolChoice.auto(),
    )
    snapshot = build_request_cost_snapshot(request)
    from agent_interop.config import ToolSurfaceConfig as TSC
    from agent_interop.tool_surface.selector import ToolSurfacePlanner

    planner = ToolSurfacePlanner()
    # Warm the registry rank cache so its one-time build is not counted.
    planner.plan(
        request, TSC(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=20, max_schema_tokens=5000),
        cost_snapshot=snapshot,
    )
    calls["dumps"] = 0
    plan = planner.plan(
        request, TSC(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=20, max_schema_tokens=5000),
        cost_snapshot=snapshot,
    )
    assert plan.visible_tools, "surface must be non-empty"
    # One serialization = the visible-surface fingerprint. The budget loop,
    # per-tool pricing, and subset totals must all be arithmetic.
    assert calls["dumps"] <= 2, (
        f"plan() serialized {calls['dumps']}x — the O(n²) budget loop regressed"
    )
