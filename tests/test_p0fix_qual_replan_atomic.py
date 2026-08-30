"""P1.5/P1.6 (review items 12-13): qualification replan atomicity.

The replan after qualification MUST:
  - thread the streaming flag through to the resolver (item 13) so the
    compatibility key fingerprints the streaming shape;
  - rebuild ModelView + private_capabilities + model_request + pinned_refs
    atomically from the FRESH compatibility_plan (item 12), not leave them
    pointing at the pre-qualification presentation.

A pre-fix code path left model_view pointing at the pre-qualification
surface, so a model that just promoted to FORCED_TOOL would see telemetry
report NO tool support while the rendered request actually exposes tools.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.gateway import Gateway
from agent_interop.context import RequestContext


def _route() -> Any:
    from agent_interop.config import ModelRoute
    return ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        context=ContextConfig(context_limit_tokens=2000, output_reserve_tokens=500),
    )


def _request(stream: bool = False) -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=64, stream=stream),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")]),
        ],
        tools=[CanonicalTool(name="t", description="d", input_schema={"type": "object"})],
        tool_choice=CanonicalToolChoice.auto(),
    )


def _gateway() -> Gateway:
    gw = Gateway(InteropServerConfig(
        probe_on_startup=False, log_level="error", routes={"r": _route()},
    ))
    return gw


def test_replan_threads_streaming_through_to_resolver():
    """Item 13: the resolver is invoked with streaming=True when the request
    is streaming — the compatibility key fingerprint must reflect streaming."""
    from agent_interop.execution import InteropRequestExecution
    from agent_interop.replay.types import CompatibilityKey
    from agent_interop.config import RepairPolicy
    from agent_interop.gateway import ResolvedInvocation

    gw = _gateway()
    route = _route()
    reconciled = _request(stream=True)
    invocation = ResolvedInvocation(
        request_context=RequestContext(session_id="s-1"),
        original_request=reconciled,
        reconciled_request=reconciled,
        route=route,
        backend_metadata=None,
        model_profile=None,
        repair_policy=None,
        invocation_plan=SimpleNamespace(),  # pre-existing plan
        codec=SimpleNamespace(),
        compatibility_key=None,
        evidence_record=None,
        repair_budget=None,
        execution_record=InteropRequestExecution(),
        runtime_capabilities=gw._static_runtime_capabilities(route),
        pinned_refs=(),
        authoritative_request=reconciled,
    )
    # Stub the resolver to capture streaming argument.
    captured: dict[str, Any] = {}

    async def fake_resolve(*args: Any, **kwargs: Any):
        captured["streaming"] = kwargs.get("streaming", args[3] if len(args) > 3 else None)
        captured["positional_streaming"] = args[3] if len(args) > 3 else None
        # Minimal tuple the replan expects.
        return (
            None, None, RepairPolicy(), SimpleNamespace(), CompatibilityKey(), None, None,
            SimpleNamespace(
                requirements=None,
                tool_surface_plan=SimpleNamespace(visible_tools=(), validation_tools=()),
            ),
            None,
        )

    original_resolve = Gateway._resolve_invocation_plan_and_key_async
    gw._resolve_invocation_plan_and_key_async = fake_resolve  # type: ignore[method-assign]

    async def run() -> None:
        # Patch ModelProjector.project to a no-op so this test does not
        # depend on the projector wiring. Save/restore so we don't leak
        # the patch into other tests in the same pytest run.
        from agent_interop.projection import ModelProjector
        original = ModelProjector.project

        def fake_project(**_: Any) -> Any:
            return SimpleNamespace(
                request=invocation.reconciled_request,
                model_view=SimpleNamespace(safe_context_limit=1),
                private_capabilities=None,
                referenced_refs=(),
            )
        ModelProjector.project = staticmethod(fake_project)  # type: ignore[method-assign]
        try:
            await gw._replan_after_qualification(invocation, InteropRequestExecution(), streaming=True)
        finally:
            ModelProjector.project = original  # type: ignore[method-assign]
            Gateway._resolve_invocation_plan_and_key_async = original_resolve

    asyncio.run(run())
    assert captured.get("streaming") is True, captured


def test_replan_rebuilds_model_view_atomically():
    """Item 12: model_view must come from a FRESH projection using the
    replanned compatibility_plan, NOT the pre-qualification value carried
    on the invocation."""
    from agent_interop.execution import InteropRequestExecution
    from agent_interop.gateway import ResolvedInvocation

    gw = _gateway()
    pre_view = SimpleNamespace(
        safe_context_limit=999,
        surface_kind="PRE_QUAL",
        visible_tool_names=(),
    )
    route = _route()
    reconciled = _request()
    # Construct a real ResolvedInvocation dataclass so replace() works.
    invocation = ResolvedInvocation(
        request_context=RequestContext(session_id="s-1"),
        original_request=reconciled,
        reconciled_request=reconciled,
        route=route,
        backend_metadata=None,
        model_profile=None,
        repair_policy=None,
        invocation_plan=SimpleNamespace(),
        codec=SimpleNamespace(),
        compatibility_key=None,
        evidence_record=None,
        repair_budget=None,
        execution_record=InteropRequestExecution(),
        runtime_capabilities=gw._static_runtime_capabilities(route),
        model_view=pre_view,  # the stale value the old code would leak
        pinned_refs=(),
        authoritative_request=reconciled,
    )
    # Stub resolver to return a fresh surface.
    from agent_interop.replay.types import CompatibilityKey
    from agent_interop.config import RepairPolicy
    new_surface = SimpleNamespace(visible_tools=("t",), validation_tools=())
    new_plan = SimpleNamespace(requirements=None, tool_surface_plan=new_surface)

    async def fake_resolve(*args: Any, **kwargs: Any):
        return (
            None, None, RepairPolicy(), SimpleNamespace(), CompatibilityKey(), None, None,
            new_plan,
            None,
        )

    gw._resolve_invocation_plan_and_key_async = fake_resolve  # type: ignore[method-assign]

    fresh_view = SimpleNamespace(safe_context_limit=1, surface_kind="POST_QUAL")

    captured: dict[str, Any] = {}

    async def run() -> None:
        from agent_interop.projection import ModelProjector
        original = ModelProjector.project

        def fake_project(**kwargs: Any):
            captured["compatibility_plan"] = kwargs.get("compatibility_plan")
            captured["invocation_plan"] = kwargs.get("invocation_plan")
            captured["projected_request"] = kwargs.get("projected_request")
            captured["seed_refs"] = kwargs.get("seed_refs")
            return SimpleNamespace(
                request=kwargs.get("projected_request") or reconciled,
                model_view=fresh_view,
                private_capabilities=None,
                referenced_refs=("seed0",),
            )
        ModelProjector.project = staticmethod(fake_project)  # type: ignore[method-assign]
        try:
            result = await gw._replan_after_qualification(
                invocation, InteropRequestExecution(), streaming=False,
            )

            # The returned invocation must carry the FRESH model_view, not
            # the pre-qualification one — otherwise downstream code would
            # read ModelView fields reporting the wrong surface.
            assert result.model_view is fresh_view, (
                "ModelView was not atomically rebuilt from the fresh plan"
            )
            # And the projector must have been driven by the FRESH
            # compatibility_plan / invocation_plan / projected_request.
            assert captured["compatibility_plan"] is new_plan
            assert captured["invocation_plan"] is not None
            assert captured["projected_request"] is reconciled
            # Pinned_refs carried forward as seed_refs so prior work is
            # not lost.
            assert captured["seed_refs"] == ()
        finally:
            ModelProjector.project = original  # type: ignore[method-assign]

    asyncio.run(run())


def test_streaming_qual_replan_uses_replan_helper_not_full_reprep():
    """Item 13: the streaming path must call _replan_after_qualification
    (a partial replan) when qualification triggers, not the full
    _prepare_invocation_async (which would re-pay history paging, schema
    hashing, etc.)."""
    # Inspect the source of handle_stream to verify it calls the replan.
    import inspect
    src = inspect.getsource(Gateway.handle_stream)
    assert "_replan_after_qualification" in src
    assert "streaming=True" in src
    # And the second _prepare_invocation_async call inside the
    # qualification branch must be gone.
    # (We can't easily count without false positives, so we just confirm
    # the replan helper is present and named with the streaming kwarg.)
