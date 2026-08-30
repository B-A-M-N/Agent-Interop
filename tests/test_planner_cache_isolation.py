"""P0-planner-cache: cross-request plan-contamination regression suite.

The planner cache previously keyed on (route id, tool fingerprint, message
size, runtime capacity, behavioral tuple) — none of which captured the
request CONTRACT (tool_choice mode, streaming, requested capabilities,
client profile, policy configs). Two requests with the same tools and the
same size could inherit each other's plans.

The fix anchors the key on the derived ``RequestRequirements`` vector plus
the full route-policy/config fingerprint. These tests walk a table of
single-dimension changes: for each pair, either the cache key differs, or
the resulting plan is byte-for-byte identical (proving sharing is safe).
"""

from __future__ import annotations

import asyncio
import dataclasses

import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
)
from agent_interop.config import (
    CompatibilityConfig,
    ContextConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.planning import RequestCompatibilityPlanner
from agent_interop.planning.types import BehavioralCapabilities
from agent_interop.upstreams.registry import get_codec


def _route(**overrides) -> ModelRoute:
    route = ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
        context=ContextConfig(context_limit_tokens=32768),
    )
    for key, value in overrides.items():
        setattr(route, key, value)
    return route




def _request(**overrides) -> CanonicalRequest:
    defaults: dict = {
        "model": CanonicalModelReference(requested_name="m"),
        # stream defaults to True on the abi type — pin the baseline OFF so
        # the stream_vs_nonstream mutation is a real single-dimension change.
        "generation": CanonicalGenerationOptions(max_output_tokens=256, stream=False),
        "messages": [
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")]),
        ],
        "tools": [CanonicalTool(
            name="read_file",
            description="read",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )],
        "tool_choice": CanonicalToolChoice.auto(),
    }
    defaults.update(overrides)
    return CanonicalRequest(**defaults)


def _capabilities() -> BehavioralCapabilities:
    return BehavioralCapabilities(
        native_tools=True, prompted_tools=True, forced_selection=True,
        sequential_tool_use=True, tool_result_continuation=True,
        streaming=True, sample_count=1,
    )


def _runtime_capabilities():
    from agent_interop.backends.base import ModelRuntimeCapabilities

    return ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OPENAI_COMPATIBLE,
        model_name="fake-model",
        configured_context_tokens=32768,
        effective_context_tokens=32768,
        architecture_context_tokens=32768,
    )


def _plan(route: ModelRoute, request: CanonicalRequest, context: RequestContext):
    planner = RequestCompatibilityPlanner()
    return asyncio.run(planner.plan(
        request=request,
        context=context,
        route=route,
        client_requirements=None,
        codec_capabilities=get_codec(route.upstream.wire_protocol).capabilities(),
        runtime_capabilities=_runtime_capabilities(),
        behavioral_capabilities=_capabilities(),
    ))


# A warm second call through the SAME planner must hit the cache and
# return an equal plan — that is the premise every isolation case relies on.
def test_warm_repeat_hits_cache_with_equal_plan():
    route, request, context = _route(), _request(), RequestContext()
    planner = RequestCompatibilityPlanner()

    async def both():
        first = await planner.plan(
            request=request, context=context, route=route,
            client_requirements=None,
            codec_capabilities=get_codec(route.upstream.wire_protocol).capabilities(),
            runtime_capabilities=_runtime_capabilities(),
            behavioral_capabilities=_capabilities(),
        )
        assert planner._cache, "first plan must populate the cache"
        second = await planner.plan(
            request=request, context=context, route=route,
            client_requirements=None,
            codec_capabilities=get_codec(route.upstream.wire_protocol).capabilities(),
            runtime_capabilities=_runtime_capabilities(),
            behavioral_capabilities=_capabilities(),
        )
        return first, second

    first, second = asyncio.run(both())
    assert first == second


# ─── Single-dimension isolation table ────────────────────────────────────
#
# Each entry: (label, mutate_a, mutate_b). ``mutate_a`` shapes request A,
# ``mutate_b`` request B. The two plans must NOT be cache-shared unless
# the plans are deep-equal. We assert on the key directly (the mechanism)
# AND on plan equality only when keys collide (the tolerance).


def _key_for(planner: RequestCompatibilityPlanner, route, request, context):
    from agent_interop.context_budget import build_request_cost_snapshot
    from agent_interop.context_budget.estimator import estimate_request_context
    from agent_interop.context_budget.types import TokenEstimate
    from agent_interop.planning.requirements import derive_request_requirements

    snapshot = build_request_cost_snapshot(request)
    estimate = estimate_request_context(request, snapshot=snapshot)
    requirements = derive_request_requirements(
        request, context, None, TokenEstimate(estimate.total_required_tokens),
        cost_snapshot=snapshot,
    )
    return planner._cache_key(
        route=route,
        requirements=requirements,
        codec_capabilities=get_codec(route.upstream.wire_protocol).capabilities(),
        behavioral_capabilities=_capabilities(),
        runtime_capabilities=_runtime_capabilities(),
    )


CASES = [
    ("auto_vs_required",
     lambda r: r,  # auto (default)
     lambda r: dataclasses.replace(r, tool_choice=CanonicalToolChoice.required())),
    ("auto_vs_named",
     lambda r: r,
     lambda r: dataclasses.replace(r, tool_choice=CanonicalToolChoice.named("read_file"))),
    ("stream_vs_nonstream",
     lambda r: r,
     lambda r: dataclasses.replace(
         r, generation=dataclasses.replace(r.generation, stream=True))),
    ("with_vs_without_tool_result_history",
     lambda r: r,
     lambda r: dataclasses.replace(r, messages=[
         *r.messages,
         CanonicalMessage(
             role="assistant",
             content=[CanonicalToolCallBlock(
                 id="call_1", name="read_file", arguments={"path": "x"},
             )],
         ),
         CanonicalMessage(
             role="tool",
             content=[CanonicalToolResultBlock(tool_call_id="call_1", content="data")],
         ),
     ])),
    ("parallel_capability_requested",
     lambda r: r,
     lambda r: dataclasses.replace(
         r,
         requested_capabilities=dataclasses.replace(
             r.requested_capabilities, parallel_tools=True),
     )),
    ("reasoning_capability_requested",
     lambda r: r,
     lambda r: dataclasses.replace(
         r,
         requested_capabilities=dataclasses.replace(
             r.requested_capabilities, reasoning=True),
     )),
    ("structured_output_requested",
     lambda r: r,
     lambda r: dataclasses.replace(
         r,
         requested_capabilities=dataclasses.replace(
             r.requested_capabilities, structured_output=True),
     )),
]


@pytest.mark.parametrize("label,mutate_a,mutate_b", CASES, ids=[c[0] for c in CASES])
def test_request_dimension_change_isolates_cache_entries(label, mutate_a, mutate_b):
    route = _route()
    context = RequestContext()
    request_a = mutate_a(_request())
    request_b = mutate_b(_request())
    planner = RequestCompatibilityPlanner()
    key_a = _key_for(planner, route, request_a, context)
    key_b = _key_for(planner, route, request_b, context)
    assert key_a is not None and key_b is not None
    # Different contract ⇒ different cache entry. (If the mutation turned
    # out to be plan-neutral, plans would be equal — but the KEY pairs the
    # requirement vector, so equality here would mean the derivation
    # itself is blind to the dimension, which is exactly what we guard.)
    assert key_a != key_b, (
        f"{label}: two different request contracts share a planner cache key"
    )


def test_tool_result_history_changes_requirements():
    """The simplest real contamination: a follow-up turn carrying tool
    results needs continuation support; a fresh turn does not. Same route,
    same tools, similar size — the cache must still separate them."""
    route = _route()
    context = RequestContext()
    plain = _request()
    continuation = _request(messages=[
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="list files")]),
        CanonicalMessage(
            role="assistant",
            content=[CanonicalToolCallBlock(
                id="call_1", name="read_file", arguments={"path": "x"},
            )],
        ),
        CanonicalMessage(
            role="tool",
            content=[CanonicalToolResultBlock(tool_call_id="call_1", content="contents")],
        ),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="now what?")]),
    ])
    planner = RequestCompatibilityPlanner()
    key_plain = _key_for(planner, route, plain, context)
    key_cont = _key_for(planner, route, continuation, context)
    assert key_plain != key_cont
    plan_plain = _plan(route, plain, context)
    plan_cont = _plan(route, continuation, context)
    assert (
        plan_plain.requirements.tool_result_continuation_required
        != plan_cont.requirements.tool_result_continuation_required
    )


def test_client_identity_changes_cache_key():
    from agent_interop.context import RequestContext

    route = _route()
    request = _request()
    planner = RequestCompatibilityPlanner()
    ctx_a = RequestContext(client_id="claude-code", client_version="2.1.0")
    ctx_b = RequestContext(client_id="generic-openai", client_version="1.0")
    key_a = _key_for(planner, route, request, ctx_a)
    key_b = _key_for(planner, route, request, ctx_b)
    assert key_a != key_b, "client profile is a requirements source and must split keys"


def test_route_policy_change_changes_cache_key():
    request = _request()
    context = RequestContext()
    planner = RequestCompatibilityPlanner()
    base = _key_for(planner, _route(), request, context)

    surface = ToolSurfaceConfig(mode=ToolSurfaceMode.TRANSPARENT)
    key_surface = _key_for(planner, _route(tool_surface=surface), request, context)
    assert key_surface != base

    compat = CompatibilityConfig(mode="direct")
    key_compat = _key_for(planner, _route(compatibility=compat), request, context)
    assert key_compat != base

    ctx = ContextConfig(strategy="strict_bounded", context_limit_tokens=32768)
    key_ctx = _key_for(planner, _route(context=ctx), request, context)
    assert key_ctx != base

    key_toolmode = _key_for(planner, _route(tool_mode=ToolMode.NATIVE), request, context)
    assert key_toolmode != base


def test_runtime_capacity_change_changes_cache_key():
    request = _request()
    context = RequestContext()
    route = _route()
    planner = RequestCompatibilityPlanner()

    def key_with(effective: int):
        from agent_interop.backends.base import ModelRuntimeCapabilities

        runtime = ModelRuntimeCapabilities(
            backend_kind=UpstreamKind.OPENAI_COMPATIBLE,
            model_name="fake-model",
            configured_context_tokens=32768,
            effective_context_tokens=effective,
            architecture_context_tokens=32768,
        )
        from agent_interop.context_budget import build_request_cost_snapshot
        from agent_interop.context_budget.estimator import estimate_request_context
        from agent_interop.context_budget.types import TokenEstimate
        from agent_interop.planning.requirements import derive_request_requirements
        snapshot = build_request_cost_snapshot(request)
        estimate = estimate_request_context(request, snapshot=snapshot)
        requirements = derive_request_requirements(
            request, context, None, TokenEstimate(estimate.total_required_tokens),
            cost_snapshot=snapshot,
        )
        return planner._cache_key(
            route=route,
            requirements=requirements,
            codec_capabilities=get_codec(route.upstream.wire_protocol).capabilities(),
            behavioral_capabilities=_capabilities(),
            runtime_capabilities=runtime,
        )

    assert key_with(32768) != key_with(8192), (
        "a replan with static capabilities and a live request with inspected "
        "capacity must not share an entry"
    )
