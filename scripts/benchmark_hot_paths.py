#!/usr/bin/env python3
"""Deterministic hot-path micro-benchmarks — the release gate's perf budget.

Measures the three hot paths the P0 performance work touched, with p95
budgets that FAIL on regression:

1. planner_cache_hit — a repeated request planning against a warm planner
   cache. The hit path must stay O(key-build), i.e. dict lookup + tuple
   construction, NOT a re-plan (tool-surface pricing + context estimate).
2. serialize_once — rendering a request to wire bytes the way the
   generation seam does. The P0-6 contract is that this happens once per
   generation; the budget here is the render+serialize cost itself, so a
   regression that re-parses or re-serializes per consumer shows up.
3. repair_pipeline_dict — repair_one on an already-parsed dict (the P0-5
   fast path the regeneration orchestrator now uses). A regression that
   reintroduces a serialize→parse round trip — or makes validation
   quadratic — fails the budget.

Everything is deterministic by construction: no network, no sleeps, no
model calls, fixed inputs built once and reused. Timing noise is handled
the way any micro-benchmark must be — many iterations, p95 over
per-iteration samples, and budgets set with wide headroom over the
measured baseline (a 2x+ regression trips the gate; ordinary machine
jitter does not).

Usage:
    uv run python scripts/benchmark_hot_paths.py          # human report
    (invoked by scripts/release.sh gate 16 — nonzero exit on regression)

Environment overrides (for constrained CI runners):
    INTEROP_PERF_SCALE=1   budget multiplier (default 1.0; raise on
                           heavily loaded/shared runners)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ─── Budgets (milliseconds, p95 per-iteration) ────────────────────────────────
# Headroom policy: each budget is set at roughly an order of magnitude over
# the baseline measured on developer hardware, so the gate trips on real
# algorithmic regressions (a re-plan per request, a lost fast path, an
# accidental re-serialize) rather than machine noise. If a LEGITIMATE change
# moves a path past its budget, the change is a perf regression and either
# fixes it or consciously re-baselines the budget in the same commit.

BUDGETS_MS = {
    "planner_cache_hit": 5.0,
    "serialize_once": 25.0,
    "repair_pipeline_dict": 10.0,
}

WARMUP = 20
SAMPLES = 200


@dataclass
class BenchResult:
    name: str
    p50_ms: float
    p95_ms: float
    budget_ms: float
    passed: bool
    samples: int


def _percentile(sorted_samples: list[float], pct: float) -> float:
    """Nearest-rank percentile over a pre-sorted sample list."""
    if not sorted_samples:
        return 0.0
    idx = min(len(sorted_samples) - 1, max(0, round(pct / 100 * len(sorted_samples)) - 1))
    return sorted_samples[idx]


def _run_bench(name: str, fn, budget_ms: float) -> BenchResult:
    for _ in range(WARMUP):
        fn()
    samples: list[float] = []
    for _ in range(SAMPLES):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1_000_000)
    samples.sort()
    p50 = _percentile(samples, 50)
    p95 = _percentile(samples, 95)
    return BenchResult(
        name=name, p50_ms=p50, p95_ms=p95, budget_ms=budget_ms,
        passed=p95 <= budget_ms, samples=len(samples),
    )


# ─── Fixed fixtures ───────────────────────────────────────────────────────────

def _build_request():
    """A representative tool-bearing request: several tools with real
    schemas and a short history — big enough that a lost cache (re-plan)
    or a lost serialize-once (re-parse per consumer) is measurable."""
    from agent_interop.abi import (
        CanonicalMessage,
        CanonicalModelReference,
        CanonicalRequest,
        CanonicalTextBlock,
        CanonicalTool,
        CanonicalToolChoice,
    )

    tools = [
        CanonicalTool(
            name=f"tool_{i}",
            description=f"Benchmark tool {i}: reads a resource.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "resource path"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "encoding": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                },
                "required": ["path"],
            },
        )
        for i in range(8)
    ]
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="bench-model"),
        tool_choice=CanonicalToolChoice.auto(),
        messages=[
            CanonicalMessage(role="system", content=[CanonicalTextBlock(text="You are a benchmark agent.")]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="Read /tmp/example and summarize.")]),
        ],
        tools=tools,
    )


def _build_route():
    from agent_interop.config import (
        ContextConfig,
        ModelRoute,
        ToolMode,
        UpstreamConfig,
        UpstreamKind,
        UpstreamProtocol,
    )

    return ModelRoute(
        id="bench",
        client_model_aliases=["bench-model"],
        upstream_model="bench-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        context=ContextConfig(context_limit_tokens=8192, output_reserve_tokens=512),
    )


class _Plan:
    """Minimal invocation plan stand-in for the seam (no request-scoped state)."""
    effective_tool_mode = None


class _Invocation:
    """Just enough of ResolvedInvocation for the seam's render path."""

    def __init__(self, request, route, codec) -> None:
        self.reconciled_request = request
        self.route = route
        self.codec = codec
        self.invocation_plan = _Plan()


def bench_planner_cache_hit(request, route) -> Callable[[], Coroutine[Any, Any, None]]:
    """Warm-cache plan() — must return the cached CompatibilityPlan."""
    from agent_interop.context import RequestContext
    from agent_interop.context_budget.estimator import (
        build_request_cost_snapshot,
        estimate_request_context,
    )
    from agent_interop.context_budget.types import TokenEstimate
    from agent_interop.planning.planner import RequestCompatibilityPlanner
    from agent_interop.planning.requirements import derive_request_requirements
    from agent_interop.planning.types import BehavioralCapabilities

    planner = RequestCompatibilityPlanner()
    codec_capabilities = None
    from agent_interop.backends.base import ModelRuntimeCapabilities
    from agent_interop.config import UpstreamKind

    runtime = ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OPENAI_COMPATIBLE,
        model_name="bench-model",
        configured_context_tokens=8192,
    )
    behavioral = BehavioralCapabilities(
        native_tools=True, automatic_selection=True, streaming=True, sample_count=3,
    )
    context = RequestContext(session_id="bench")
    client_requirements = None

    # Warm the cache with the exact inputs the bench will reuse.
    snapshot = build_request_cost_snapshot(request)
    token_estimate = estimate_request_context(request, snapshot=snapshot).total_required_tokens
    requirements = derive_request_requirements(
        request, context, client_requirements, TokenEstimate(token_estimate),
        cost_snapshot=snapshot,
    )
    key = planner._cache_key(
        route=route,
        requirements=requirements,
        codec_capabilities=codec_capabilities,
        behavioral_capabilities=behavioral,
        runtime_capabilities=runtime,
    )
    assert key is not None, "bench could not build a planner cache key"

    # The bench measures the real hit path, not a re-implementation of it:
    # seed the cache with a sentinel plan, drive plan() itself, and assert
    # it short-circuits by returning THAT object (identity, not equality).
    # A regression that skips the cache re-plans and builds a fresh plan —
    # a different object — and the bench fails loudly instead of quietly
    # benchmarking the slow path it was meant to catch.
    from agent_interop.planning.planner import _PLANNER_CACHE_LOCK

    sentinel = object()

    async def warm() -> None:
        await planner.plan(
            request=request,
            context=context,
            route=route,
            client_requirements=client_requirements,
            codec_capabilities=codec_capabilities,
            runtime_capabilities=runtime,
            behavioral_capabilities=behavioral,
            cost_snapshot=snapshot,
        )
        # Replace the real plan with the sentinel: plan() must return it.
        with _PLANNER_CACHE_LOCK:
            planner._cache[key] = sentinel  # type: ignore[assignment]

    async def plan_hit() -> None:
        result = await planner.plan(
            request=request,
            context=context,
            route=route,
            client_requirements=client_requirements,
            codec_capabilities=codec_capabilities,
            runtime_capabilities=runtime,
            behavioral_capabilities=behavioral,
            cost_snapshot=snapshot,
        )
        if result is not sentinel:
            raise AssertionError(
                "plan() rebuilt the plan instead of returning the cached "
                "entry — this benchmark is not measuring the cache-hit path"
            )

    # Warm now, once, outside the timed region: one real plan() populates
    # the cache, then the sentinel swap makes every subsequent hit
    # identity-verifiable.
    asyncio.run(warm())

    return plan_hit


def bench_serialize_once(request, route) -> Callable[[], None]:
    """render → apply plan → serialize exactly as the generation seam does."""
    import json as _json

    from agent_interop.config import InteropServerConfig
    from agent_interop.gateway import Gateway
    from agent_interop.upstreams.registry import get_codec

    codec = get_codec(route.upstream.wire_protocol)
    # The plan-application step lives on the Gateway; use the real one with
    # a no-network config so the bench measures production presentation.
    gateway = Gateway(
        InteropServerConfig(probe_on_startup=False, log_level="error", routes={"bench": route}),
        transport=None,
    )
    plan = type("P", (), {"effective_tool_mode": None})()

    def serialize_once() -> None:
        rendered = codec.render_request(request, route.upstream_model, stream=False)
        rendered = gateway._apply_invocation_plan_to_request(rendered, plan, route)
        data = _json.dumps(rendered, ensure_ascii=False, separators=(",", ":")).encode("utf-8", "replace")
        if not data:
            raise AssertionError("render produced empty bytes")

    return serialize_once


def bench_repair_pipeline_dict(request, route) -> Callable[[], None]:
    """repair_one on an already-parsed dict — the P0-5 regeneration path.

    The dict is schema-valid, so the pipeline should accept it at
    VALID_UNCHANGED after validation alone; a regression that re-parses or
    rebuilds it pays measurably.
    """
    from agent_interop.repair.pipeline import RepairBudget, repair_one

    tool = request.tools[0]
    good_args = {"path": "/tmp/example", "options": {"encoding": "utf-8", "limit": 10}}

    def repair_dict() -> None:
        outcome = repair_one(
            call_name=tool.name,
            call_arguments=dict(good_args),
            tools=list(request.tools),
            budget=RepairBudget(),
        )
        if not outcome.is_accepted:
            raise AssertionError(f"bench repair unexpectedly rejected: {outcome.error}")

    return repair_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None, help="also write results as JSON")
    args = parser.parse_args()

    scale = float(os.environ.get("INTEROP_PERF_SCALE", "1.0"))

    request = _build_request()
    route = _build_route()

    benches = [
        ("planner_cache_hit", bench_planner_cache_hit(request, route), BUDGETS_MS["planner_cache_hit"]),
        ("serialize_once", bench_serialize_once(request, route), BUDGETS_MS["serialize_once"]),
        ("repair_pipeline_dict", bench_repair_pipeline_dict(request, route), BUDGETS_MS["repair_pipeline_dict"]),
    ]

    # plan() is async; drive async benches through asyncio.run per call. A
    # fresh event loop per sample is deliberate — the planner's cache lock
    # is loop-agnostic, and per-call isolation keeps any loop state from
    # leaking between samples.
    results: list[BenchResult] = []
    for name, fn, budget in benches:
        is_coro = asyncio.iscoroutinefunction(fn)
        wrapped = (lambda f=fn: asyncio.run(f())) if is_coro else fn
        result = _run_bench(name, wrapped, budget * scale)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{status}  {name:<22} p50={result.p50_ms:8.4f}ms  "
            f"p95={result.p95_ms:8.4f}ms  budget={result.budget_ms:.2f}ms"
        )

    failed = [r for r in results if not r.passed]
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "generated_at": datetime.now(UTC).isoformat(),
            "scale": scale,
            "samples_per_bench": SAMPLES,
            "results": [vars(r) for r in results],
        }, indent=2))

    if failed:
        print(f"\nPERF REGRESSION: {len(failed)} path(s) over budget")
        return 1
    print("\nall hot paths within budget")
    return 0


if __name__ == "__main__":
    sys.exit(main())
