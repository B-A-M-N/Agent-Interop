"""P0.2 regression: the native tool-surface reduction MUST reach the rendered
upstream request on the wire.

Background
----------
``tool_surface_plan.visible_tools`` selects a small bounded tool set (e.g. 4 of
20 client tools), and ``InvocationPlan.upstream_tools`` carries exactly that set
while ``InvocationPlan.validation_tools`` retains the full client registry (20).
But both ``_handle_request_send`` and ``_handle_stream_send`` used to deep-copy
the full ``reconciled_request`` and render it verbatim, and
``_apply_invocation_plan_to_request`` was a literal ``pass`` for NATIVE mode —
so the model received all 20 tools and the schema-context reduction never
happened on the wire. That defeats the entire low-resource premise: a 16K model
still has to hold the full client tool schemas.

Fix (gateway.py)
----------------
Render ``replace(canonical, tools=list(plan.upstream_tools))`` instead of the
full reconciled request, in BOTH the non-streaming and streaming send paths.

This test reuses the project's own ``FakeMatrixUpstream`` HTTP harness (a real
upstream the gateway talks to), captures the exact request body the gateway
transmits, and asserts the tool count — rather than trusting a planning
dataclass.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
import uvicorn

from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.server.app import create_app


# ─── capturing fake upstream (records the exact rendered body) ───────────────


class _CapturingUpstream:
    """Minimal Ollama-Chat upstream that records the last request body."""

    def __init__(self) -> None:
        self.last_request: dict[str, Any] | None = None
        self._port = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def _app(self) -> FastAPI:
        upstream = self

        app = FastAPI()

        @app.post("/api/chat")
        async def ollama_chat(request: FastAPIRequest):
            body = await request.json()
            upstream.last_request = body
            return JSONResponse(
                {
                    "model": "fake-model",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                    "done_reason": "stop",
                }
            )

        return app

    async def start(self) -> int:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self._port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(app=self._app(), host="127.0.0.1", port=self._port, log_level="error")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        for _ in range(50):
            if self._server.started:
                break
            await asyncio_sleep()
        else:
            raise RuntimeError("capturing upstream failed to start")
        return self._port

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
            if self._thread:
                await asyncio_to_thread_join(self._thread)
            self._server = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"


def asyncio_sleep() -> Any:
    import asyncio

    return asyncio.sleep(0.05)


def asyncio_to_thread_join(thread: threading.Thread) -> Any:
    import asyncio

    return asyncio.to_thread(thread.join, 5)


# ─── app builder ────────────────────────────────────────────────────────────


@asynccontextmanager
async def _make_app(upstream_url: str, visible: int) -> AsyncGenerator[FastAPI, None]:
    from asgi_lifespan import LifespanManager

    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "qwen": ModelRoute(
                id="qwen",
                client_model_aliases=["qwen2.5-coder"],
                upstream_model="qwen2.5-coder",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url=upstream_url,
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    timeout_seconds=30.0,
                ),
                context=ContextConfig(context_limit_tokens=32768),
                tool_mode=ToolMode.NATIVE,
                translation_mode=TranslationMode.CANONICAL,
                tool_surface=ToolSurfaceConfig(
                    mode=ToolSurfaceMode.DYNAMIC,
                    max_initial_tools=visible,
                    max_schema_tokens=4096,
                ),
            ),
        },
    )
    app = create_app(config=config)
    async with LifespanManager(app) as manager:
        yield manager.app


# ─── tools ──────────────────────────────────────────────────────────────────


def _make_tools(count: int) -> list[dict[str, Any]]:
    return [
        {
            "name": f"client_tool_{i:02d}",
            "description": f"client tool {i}",
            "input_schema": {
                "type": "object",
                "properties": {"arg": {"type": "string"}},
                "required": ["arg"],
            },
        }
        for i in range(count)
    ]


def _anthropic_body(tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": "qwen2.5-coder",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "use the read tool"}],
        "tools": tools,
    }


# ─── the regression ─────────────────────────────────────────────────────────


class TestNativeToolProjection:
    @pytest_asyncio.fixture
    async def upstream(self):
        srv = _CapturingUpstream()
        await srv.start()
        try:
            yield srv
        finally:
            await srv.stop()

    @pytest.mark.asyncio
    async def test_rendered_request_carries_only_visible_tools(self, upstream):
        """20 client tools, visible_tools == 4: Ollama must receive exactly 4
        client-visible tools PLUS the 4 internal __interop_ tools = 8 total."""
        client_tools = _make_tools(20)
        async with _make_app(upstream.url, visible=4) as app:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/v1/messages", json=_anthropic_body(client_tools))
                assert resp.status_code == 200, resp.text
        assert upstream.last_request is not None, "upstream never received a request"
        rendered = upstream.last_request
        assert "tools" in rendered, "rendered Ollama body has no tools key"
        rendered_names = {t["function"]["name"] for t in rendered["tools"]}
        declared_names = {t["name"] for t in client_tools}
        # P0.7: Internal tools are conditional. With 20 declared and 4 visible,
        # there are 16 withheld tools, so __interop_get_tool_schema should be present.
        # Client tools: exactly 4 visible
        client_visible = rendered_names - {"__interop_read_result", "__interop_recall_history", "__interop_search_history", "__interop_get_tool_schema"}
        assert len(client_visible) == 4, f"expected 4 visible client tools, got {len(client_visible)}: {client_visible}"
        assert client_visible <= declared_names

    @pytest.mark.asyncio
    async def test_streaming_path_identical_view(self, upstream):
        """P0.2 must apply identically to the streaming send path."""
        client_tools = _make_tools(20)
        body = _anthropic_body(client_tools)
        body["stream"] = True
        async with _make_app(upstream.url, visible=4) as app:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                async with client.stream("POST", "/v1/messages", json=body) as resp:
                    assert resp.status_code == 200, resp.text
                    # drain
                    async for _ in resp.aiter_bytes():
                        pass
        assert upstream.last_request is not None
        rendered = upstream.last_request
        rendered_names = {t["function"]["name"] for t in rendered["tools"]}
        declared_names = {t["name"] for t in client_tools}
        # P0.7: Internal tools are conditional. __interop_get_tool_schema is
        # present because there are withheld tools (20 declared, 4 visible).
        assert "__interop_get_tool_schema" in rendered_names
        # Client tools: exactly 4 visible
        client_visible = rendered_names - {"__interop_read_result", "__interop_recall_history", "__interop_search_history", "__interop_get_tool_schema"}
        assert len(client_visible) == 4, f"expected 4 visible client tools, got {len(client_visible)}"
        assert client_visible <= declared_names
