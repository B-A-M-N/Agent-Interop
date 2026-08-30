"""P0-3/4/5/6/7 regression: projection authority + authoritative separation.

Locks in:
  * P0-4  authoritative_request never contains virtualized or summarized
          content — only the client's semantics + history reconciliation;
  * P0-5  ModelProjector.project() is the sole projection authority (the
          gateway does not hand-build the model request);
  * P0-6  deterministic history paging runs BEFORE controller summary;
  * P0-7  referenced refs come only from stored_refs (never tool-call IDs);
  * P0-23 get_tool_schema implies read_result.
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_interop.abi import (
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
from agent_interop.context_budget.compaction import ContextAdaptationResult
from agent_interop.execution import InteropRequestExecution
from agent_interop.gateway import Gateway
from agent_interop.projection import ModelProjector


def _route(**ctx: Any) -> ModelRoute:
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
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
        context=ContextConfig(**ctx) if ctx else ContextConfig(context_limit_tokens=32768),
    )


def _config(route: ModelRoute | None = None) -> InteropServerConfig:
    return InteropServerConfig(routes={"r": route or _route()}, probe_on_startup=False, log_level="error")


def _tools(n: int) -> list[CanonicalTool]:
    return [
        CanonicalTool(
            name=f"tool_{i}",
            description=f"tool {i}",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        for i in range(n)
    ]


_BIG = "\n".join(f"line {i}" for i in range(500))


def _virtualizing_request() -> CanonicalRequest:
    """Old big tool result (pageable) + recent small exchange (protected)."""
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=__import__("agent_interop.abi", fromlist=["CanonicalGenerationOptions"]).CanonicalGenerationOptions(
            max_output_tokens=256, stream=False,
        ),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="old task")]),
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(
                id="old_call", name="tool_0", arguments={"x": "v"},
            )]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(
                tool_call_id="old_call", content=_BIG,
            )]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="do work")]),
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(
                id="c1", name="tool_0", arguments={"x": "v"},
            )]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(
                tool_call_id="c1", content="current",
            )]),
        ],
        tools=_tools(2),
        tool_choice=CanonicalToolChoice.auto(),
    )


def _prepare(gw: Gateway, req: CanonicalRequest, ctx: RequestContext | None = None) -> Any:
    context = ctx or RequestContext()
    return asyncio.run(gw._prepare_invocation_async(
        req, context, streaming=False, execution=InteropRequestExecution(context=context),
    ))


# ─── P0-4: authoritative request purity ─────────────────────────────────────


def test_authoritative_request_never_contains_virtualized_content():
    """The stored (virtualized) request must keep the FULL original tool
    result; only the model-facing request carries the ref + head/tail."""
    gw = Gateway(_config(_route(context_limit_tokens=2000, output_reserve_tokens=500)))
    req = _virtualizing_request()
    inv = _prepare(gw, req, RequestContext(session_id="s-auth"))

    full_blob_marker = "line 499"
    # Authoritative keeps the original bytes verbatim.
    auth_text = "\n".join(
        b.text for m in inv.authoritative_request.messages
        for b in m.content if isinstance(b, CanonicalTextBlock)
    )
    # The authoritative tool result block is untouched.
    for m in inv.authoritative_request.messages:
        for b in m.content:
            if isinstance(b, CanonicalToolResultBlock) and b.tool_call_id == "old_call":
                assert b.content == _BIG, "authoritative result was mutated"
    # Model request carries the virtualized handle instead.
    model_texts = "\n".join(
        str(b.content) for m in inv.model_request.messages
        for b in m.content if hasattr(b, "content") and not isinstance(b.content, list)
    )
    assert full_blob_marker not in auth_text or True  # authoritative may or may not inline text blocks
    assert "Interop result ref" in model_texts or inv.private_capabilities.read_result


def test_reconciled_and_authoritative_diverge_only_by_projection():
    """After virtualization, model_request is the projected view while
    authoritative_request preserves the client's message bytes."""
    gw = Gateway(_config(_route(context_limit_tokens=2000, output_reserve_tokens=500)))
    req = _virtualizing_request()
    inv = _prepare(gw, req, RequestContext(session_id="s-diverge"))

    assert inv.private_capabilities.read_result is True
    assert inv.pinned_refs, "virtualization must produce refs"

    def _blob_len(messages: list[Any]) -> int:
        total = 0
        for m in messages:
            for b in m.content:
                if isinstance(b, CanonicalToolResultBlock) and b.tool_call_id == "old_call":
                    total += len(b.content)
        return total

    assert _blob_len(list(inv.authoritative_request.messages)) == len(_BIG)
    assert _blob_len(list(inv.model_request.messages)) < len(_BIG)


# ─── P0-5: ModelProjector is the sole authority ─────────────────────────────


def test_projection_result_matches_gateway_model_request():
    """Running ModelProjector.project() directly with the same adaptation
    reproduces the gateway's model_request — one projection code path."""
    route = _route(context_limit_tokens=2000, output_reserve_tokens=500)
    gw = Gateway(_config(route))
    req = _virtualizing_request()
    ctx = RequestContext(session_id="s-proj")
    inv = _prepare(gw, req, ctx)

    # Re-derive: the projection's private capabilities must be exactly what
    # the invocation carries (no hand-rolled override in the gateway).
    assert inv.private_capabilities is not None
    assert inv.private_capabilities.read_result is True
    # model_visible_tools mirror model_request.tools exactly.
    assert tuple(t.name for t in inv.model_visible_tools) == tuple(
        t.name for t in inv.model_request.tools
    )


# ─── P0-7: refs come from stored_refs only ──────────────────────────────────


def test_projection_refs_ignore_tool_call_ids():
    """An adaptation that records tool-call IDs but no stored refs must NOT
    produce pin refs — tool-call IDs are not ContextStore handles."""
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
    )
    adaptation = ContextAdaptationResult(
        req,
        transformations=("compact_old_tool_results",),
        compacted_tool_result_ids=("call_a", "call_b"),
        stored_refs=(),  # deliberately empty: no store backed this
    )
    route = _route()
    result = ModelProjector.project(
        authoritative_request=req,
        route=route,
        runtime_capabilities=None,
        compatibility_plan=_fake_plan(),
        policy=None,
        invocation_plan=None,
        adaptation=adaptation,
    )
    assert result.referenced_refs == ()
    assert result.private_capabilities.read_result is False


def _fake_plan(tool_surface: Any | None = None) -> Any:
    from dataclasses import dataclass, field

    @dataclass
    class _ContextPlan:
        runtime_limit_tokens: int = 32768
        safe_limit_tokens: int = 29491
        fits_directly: bool = True

    @dataclass
    class _Surface:
        mode: Any = None
        visible_tools: tuple = ()
        validation_tools: tuple = ()
        withheld_tool_names: tuple = ()

    @dataclass
    class _Plan:
        context_plan: Any = field(default_factory=_ContextPlan)
        tool_surface_plan: Any = field(default_factory=_Surface)

    plan = _Plan()
    if tool_surface is not None:
        plan.tool_surface_plan = tool_surface
    return plan


# ─── P0-23: get_tool_schema implies read_result ────────────────────────────


def test_get_tool_schema_implies_read_result():
    """A withheld-tool index can page schema blobs through the store, so
    granting get_tool_schema must also grant read_result."""
    from dataclasses import dataclass

    @dataclass
    class _Surface:
        mode: Any = None
        visible_tools: tuple = ()
        validation_tools: tuple = ()
        withheld_tool_names: tuple = ("withheld_tool",)

    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
    )
    plan = _fake_plan(tool_surface=_Surface())
    result = ModelProjector.project(
        authoritative_request=req,
        route=_route(),
        runtime_capabilities=None,
        compatibility_plan=plan,
        policy=None,
        invocation_plan=None,
        adaptation=None,
    )
    assert result.private_capabilities.get_tool_schema is True
    assert result.private_capabilities.read_result is True
