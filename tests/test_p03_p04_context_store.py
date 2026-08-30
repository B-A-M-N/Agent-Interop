"""P0.3/P0.4 regression tests: context virtualization + tool authority."""

from __future__ import annotations

import asyncio

import pytest

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
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
from agent_interop.context_store import (
    ContextStore,
    InternalToolExecutor,
    VirtualizationPolicy,
    all_internal_tools,
    internal_tool_names,
)
from agent_interop.enums import RESERVED_INTERNAL_TOOL_PREFIX, ToolAuthority
from agent_interop.execution import InteropRequestExecution
from agent_interop.gateway import Gateway

# ─── ContextStore ─────────────────────────────────────────────────────────


class TestContextStore:
    def test_store_and_retrieve(self):
        store = ContextStore()
        entry = store.store("sess-1", "hello world", "tool_result", tool_call_id="c1")
        assert entry.ref
        assert entry.sha256
        retrieved = store.get(entry.ref, "sess-1")
        assert retrieved is not None
        assert retrieved.content == "hello world"

    def test_cross_session_isolation(self):
        store = ContextStore()
        entry = store.store("sess-1", "secret", "tool_result")
        assert store.get(entry.ref, "sess-2") is None

    def test_get_slice(self):
        store = ContextStore()
        content = "\n".join(f"line {i}" for i in range(100))
        entry = store.store("s1", content, "tool_result")
        slice_ = store.get_slice(entry.ref, "s1", start_line=1, line_count=5)
        assert slice_ == "line 0\nline 1\nline 2\nline 3\nline 4\n"

    def test_search(self):
        store = ContextStore()
        store.store("s1", "the quick brown fox", "tool_result")
        store.store("s1", "lazy dog", "tool_result")
        store.store("s1", "foxes are quick", "tool_result")
        matches = store.search("s1", "fox")
        assert len(matches) == 2

    def test_unknown_ref_fails_closed(self):
        store = ContextStore()
        assert store.get("nonexistent", "s1") is None
        assert store.get_slice("nonexistent", "s1") is None


# ─── Internal tools ────────────────────────────────────────────────────────


class TestInternalTools:
    def test_tool_names(self):
        names = internal_tool_names()
        assert "__interop_read_result" in names
        assert "__interop_recall_history" in names
        assert "__interop_search_history" in names

    def test_all_internal_tools_have_tiny_schemas(self):
        for tool in all_internal_tools():
            assert tool.name.startswith(RESERVED_INTERNAL_TOOL_PREFIX)
            # schemas should be small
            props = tool.input_schema.get("properties", {})
            assert len(props) <= 3


class TestInternalToolExecutor:
    def test_read_result(self):
        store = ContextStore()
        executor = InternalToolExecutor(store)
        content = "\n".join(f"line {i}" for i in range(100))
        entry = store.store("s1", content, "tool_result", tool_call_id="c1")
        result = executor.execute("__interop_read_result", {"ref": entry.ref, "start_line": 1, "line_count": 5}, "s1")
        assert not result.is_error
        assert "line 0" in result.content
        assert "line 4" in result.content

    def test_read_result_unknown_ref(self):
        store = ContextStore()
        executor = InternalToolExecutor(store)
        result = executor.execute("__interop_read_result", {"ref": "bad"}, "s1")
        assert result.is_error

    def test_recall_history(self):
        store = ContextStore()
        executor = InternalToolExecutor(store)
        entry = store.store("s1", "old conversation", "history_fragment")
        result = executor.execute("__interop_recall_history", {"ref": entry.ref}, "s1")
        assert not result.is_error
        assert result.content == "old conversation"

    def test_search_history(self):
        store = ContextStore()
        executor = InternalToolExecutor(store)
        store.store("s1", "the quick brown fox", "tool_result")
        result = executor.execute("__interop_search_history", {"query": "fox"}, "s1")
        assert not result.is_error
        assert "match" in result.content.lower()

    def test_unknown_internal_tool(self):
        store = ContextStore()
        executor = InternalToolExecutor(store)
        result = executor.execute("__interop_nonexistent", {}, "s1")
        assert result.is_error


# ─── Virtualization policy ─────────────────────────────────────────────────


class TestVirtualizationPolicy:
    def test_small_result_not_virtualized(self):
        policy = VirtualizationPolicy(max_inline_lines=50)
        block = CanonicalToolResultBlock(tool_call_id="c1", content="small output")
        decision = policy.decide(block)
        assert not decision.should_virtualize

    def test_large_result_virtualized(self):
        policy = VirtualizationPolicy(max_inline_lines=10)
        content = "\n".join(f"line {i}" for i in range(100))
        block = CanonicalToolResultBlock(tool_call_id="c1", content=content)
        decision = policy.decide(block)
        assert decision.should_virtualize


# ─── Tool authority ────────────────────────────────────────────────────────


class TestToolAuthority:
    def test_classify_internal(self):
        Gateway(InteropServerConfig(routes={}, probe_on_startup=False, log_level="error"), allow_invalid_config=True)
        assert Gateway._classify_tool_authority("__interop_read_result") == ToolAuthority.INTEROP_INTERNAL

    def test_classify_client(self):
        Gateway(InteropServerConfig(routes={}, probe_on_startup=False, log_level="error"), allow_invalid_config=True)
        assert Gateway._classify_tool_authority("Read") == ToolAuthority.CLIENT

    def test_reserved_namespace_collision_raises(self):
        gw = Gateway(InteropServerConfig(routes={}, probe_on_startup=False, log_level="error"), allow_invalid_config=True)
        with pytest.raises(ValueError, match="reserved"):
            gw._check_reserved_namespace_collision([CanonicalTool(name="__interop_evil", description="x", input_schema={})])

    def test_valid_client_tools_pass(self):
        gw = Gateway(InteropServerConfig(routes={}, probe_on_startup=False, log_level="error"), allow_invalid_config=True)
        gw._check_reserved_namespace_collision([CanonicalTool(name="Read", description="x", input_schema={})])


# ─── Gateway integration ───────────────────────────────────────────────────


def _make_route(tool_mode=ToolMode.NATIVE, visible=4):
    return ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=tool_mode,
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=visible),
        context=ContextConfig(context_limit_tokens=32768),
    )


def _make_request(tools):
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="do work")])],
        tools=list(tools),
        tool_choice=CanonicalToolChoice.auto(),
    )


class TestGatewayIntegration:
    def test_internal_tools_on_model_surface(self):
        route = _make_route(visible=4)
        config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
        tools = [CanonicalTool(name=f"t{i}", description=f"t{i}", input_schema={"type": "object", "properties": {}}) for i in range(10)]
        req = _make_request(tools)
        gw = Gateway(config)
        ctx = RequestContext()
        inv = asyncio.run(gw._prepare_invocation_async(req, ctx, streaming=False, execution=InteropRequestExecution(context=ctx)))
        model_names = [t.name for t in inv.model_request.tools]
        # P0.7: Internal tools are conditional. Withheld tools → __interop_get_tool_schema
        assert "__interop_get_tool_schema" in model_names

    def test_reserved_namespace_rejected_in_request(self):
        route = _make_route()
        config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
        tools = [CanonicalTool(name="__interop_evil", description="x", input_schema={"type": "object", "properties": {}})]
        req = _make_request(tools)
        gw = Gateway(config)
        ctx = RequestContext()
        with pytest.raises(ValueError, match="reserved"):
            asyncio.run(gw._prepare_invocation_async(req, ctx, streaming=False, execution=InteropRequestExecution(context=ctx)))

    def test_partition_by_authority(self):
        route = _make_route(visible=4)
        config = InteropServerConfig(routes={"r": route}, probe_on_startup=False, log_level="error")
        gw = Gateway(config)

        class FakeBlock:
            def __init__(self, name, id="id"):
                self.name = name
                self.id = id
                self.arguments = {}

        class FakeDecision:
            accepted_blocks = [FakeBlock("Read"), FakeBlock("__interop_read_result"), FakeBlock("Write")]

        client, internal = gw._partition_accepted_by_authority(FakeDecision(), "s1")
        assert len(client) == 2
        assert len(internal) == 1
        assert internal[0].id == "id"
