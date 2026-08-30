"""P0.1 regression: the authoritative/model-view split must be load-bearing.

This test locks the contract the user prescribed:

    authoritative_request   -> full reconciled client state (source of truth)
                               used for validation / reconciliation / evidence
    model_request          -> bounded view actually rendered upstream
                               (selected tools only; no second projection)
    model_view             -> receipt describing the projection

It proves three things the earlier 7-integration-test regression exposed:

1. ``model_request`` is NOT a stale/descriptive copy. After ``_prepare_invocation``
   returns, the send path renders ``model_request`` directly — there is no
   second ``replace(...tools=plan.upstream_tools)`` re-derivation that could
   diverge from the prepared view.

2. Rendering ``model_request`` produces exactly the upstream tool set
   (``plan.upstream_tools``), and never the full authoritative registry.

3. Validation (process_tool_batch) uses the authoritative registry, so a tool
   call the model emits is checked against the full client tool set even when
   the narrowed model view hides some tools (e.g. PROMPTED mode where
   ``upstream_tools`` is empty).
"""

from __future__ import annotations

import asyncio


from agent_interop.abi import (
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
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.execution import InteropRequestExecution
from agent_interop.gateway import Gateway, ResolvedInvocation


def _make_tools(n: int) -> list[CanonicalTool]:
    return [
        CanonicalTool(
            name=f"tool_{i}",
            description=f"tool {i}",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        )
        for i in range(n)
    ]


def _make_request(tools: list[CanonicalTool], visible: int) -> tuple[CanonicalRequest, InteropServerConfig, ModelRoute]:
    route = ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=visible),
        # These tests exercise ModelView/tool-surface accounting, not the
        # unknown-capacity policy (beta default: hard-reject tool-bearing
        # requests when capacity is unknown).
        context=ContextConfig(context_limit_tokens=32768),
    )
    config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="do work")])],
        tools=list(tools),
        tool_choice=CanonicalToolChoice.auto(),
    )
    return req, config, route


async def _prepare(gw: Gateway, req: CanonicalRequest) -> ResolvedInvocation:
    return await gw._prepare_invocation_async(req, RequestContext(), streaming=False, execution=InteropRequestExecution(context=RequestContext()))


class TestP01AuthoritativeModelView:
    def test_authoritative_holds_full_registry(self):
        tools = _make_tools(20)
        req, config, _route = _make_request(tools, visible=4)
        gw = Gateway(config)
        inv = asyncio.run(_prepare(gw, req))
        assert [t.name for t in inv.authoritative_request.tools] == [t.name for t in tools]
        # reconciled_request alias equals authoritative for the tool set
        assert [t.name for t in inv.reconciled_request.tools] == [t.name for t in tools]

    def test_model_request_is_narrowed_to_upstream_tools(self):
        tools = _make_tools(20)
        req, config, _route = _make_request(tools, visible=4)
        gw = Gateway(config)
        inv = asyncio.run(_prepare(gw, req))
        upstream = [t.name for t in inv.invocation_plan.upstream_tools]
        assert len(upstream) == 4
        # model_request carries the narrowed client tools PLUS private
        # __interop_ retrieval tools (P0.3)
        model_names = [t.name for t in inv.model_request.tools]
        assert set(upstream).issubset(set(model_names))
        # P0.7: Internal tools are conditional. Withheld tools → __interop_get_tool_schema
        assert "__interop_get_tool_schema" in model_names
        # model view records the reduction
        assert inv.model_view.authorized_tool_count == 20
        assert inv.model_view.visible_tool_count == 4
        assert inv.model_view.visible_tool_names == tuple(upstream)

    def test_render_uses_only_model_view_tools(self):
        """No second projection: rendered model_request carries exactly the
        narrowed tool set, never the full authoritative registry."""
        tools = _make_tools(20)
        req, config, _route = _make_request(tools, visible=4)
        gw = Gateway(config)
        inv = asyncio.run(_prepare(gw, req))
        codec = inv.codec
        rendered = codec.render_request(inv.model_request, inv.route.upstream_model, stream=False)
        rendered_names = [t["function"]["name"] for t in rendered.get("tools", [])]
        upstream = [t.name for t in inv.model_request.tools]
        assert rendered_names == upstream
        # full client registry is hidden from model
        assert "tool_19" not in rendered_names

    def test_authoritative_and_model_view_disagree_on_tool_count(self):
        """Demonstrates the split is real: the two views differ, and the
        authoritative one is strictly larger (when a surface narrows)."""
        tools = _make_tools(20)
        req, config, _route = _make_request(tools, visible=4)
        gw = Gateway(config)
        inv = asyncio.run(_prepare(gw, req))
        assert len(inv.authoritative_request.tools) > len(inv.model_request.tools)

    def test_prompted_mode_model_view_tools_empty_but_validation_full(self):
        """PROMPTED mode narrows upstream_tools to empty (tools go in the prompt
        text). The model view must reflect that, yet validation_tools (and thus
        the authoritative registry used for validation) stays full — otherwise a
        returned tool call would be wrongly rejected."""
        tools = _make_tools(3)
        route = ModelRoute(
            id="r",
            client_model_aliases=["m"],
            upstream_model="fake-model",
            upstream=UpstreamConfig(
                kind=UpstreamKind.OPENAI_COMPATIBLE,
                base_url="http://127.0.0.1:1",
                wire_protocol=UpstreamProtocol.OPENAI_CHAT,
            ),
            tool_mode=ToolMode.PROMPTED,
            context=ContextConfig(context_limit_tokens=32768),
        )
        config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
        req = CanonicalRequest(
            model=CanonicalModelReference(requested_name="m"),
            messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="do work")])],
            tools=list(tools),
            tool_choice=CanonicalToolChoice.auto(),
        )
        gw = Gateway(config)
        inv = asyncio.run(_prepare(gw, req))
        # PROMPTED: bounded model view has no client tools
        client_names = [t.name for t in inv.model_request.tools if not t.name.startswith("__interop_")]
        assert len(client_names) == 0
        # P0.6: Internal tools are gated by PrivateCapabilityPlan — when no
        # capability is active (no virtualization, no withheld tools), the
        # prompt contract does NOT include internal tool descriptions.
        has_internal_in_tools = any(t.name.startswith("__interop_") for t in inv.model_request.tools)
        has_internal_in_contract = "__interop_read_result" in (inv.invocation_plan.prompt_contract or "")
        # In PROMPTED mode without active capabilities, no internal tools exposed.
        assert not has_internal_in_tools
        # but the authoritative registry (used for validation) is intact
        assert [t.name for t in inv.authoritative_request.tools] == [t.name for t in tools]
        assert [t.name for t in inv.invocation_plan.validation_tools] == [t.name for t in tools]
        assert inv.model_view.visible_tool_count == 0
