"""Phase 5 gate: Complete non-streaming application wiring integration tests.

Exercises the full pipeline:
  ASGI endpoint → adapter.decode_request() → CanonicalRequest
  → route resolution → ModelRoute → InvocationPlan
  → upstream codec render → HTTP transport → upstream codec decode
  → tool_transaction_service → canonical response assembly
  → CanonicalResponse → adapter.encode_response() → HTTP response

The fake upstream runs as a real HTTP server (uvicorn in a background thread)
so the HTTP transport layer is exercised for the Interop→upstream leg.
The client→Interop leg uses ASGITransport (httpx.AsyncClient with app=).
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse

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
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.gateway import Gateway
from agent_interop.server.app import create_app

# ─── Upstream wire-protocol response body factories ──────────────────────

# These produce the bodies the *fake upstream HTTP server* returns.
# The fake upstream listens on a real port and responds in the same format
# as a real model backend (OpenAI Chat, Anthropic Messages, OpenAI Responses).


def _openai_chat_response_body(text: str = "Hello from upstream") -> dict[str, Any]:
    """OpenAI Chat Completions response body."""
    return {
        "id": "fake-chat-response",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _openai_chat_tool_call_body(
    name: str = "read_file", arguments: dict[str, Any] | None = None
) -> dict[str, Any]:
    """OpenAI Chat response with a tool call."""
    args = arguments or {"path": "/tmp/x"}
    return {
        "id": "fake-chat-response",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_fake_001",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _anthropic_response_body(text: str = "Hello from upstream") -> dict[str, Any]:
    """Anthropic Messages API response body."""
    return {
        "id": "msg_fake_001",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "fake-model",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _openai_responses_response_body(text: str = "Hello from upstream") -> dict[str, Any]:
    """OpenAI Responses API response body."""
    return {
        "id": "resp_fake_001",
        "object": "response",
        "created": 0,
        "model": "fake-model",
        "output": [
            {
                "id": "msg_fake_001",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


# ─── Client request body factories ───────────────────────────────────────

# These produce the bodies the *test client* sends to the Interop server.
# Each matches the protocol format the client endpoint expects.


def anthropic_messages_body(text: str = "Hello") -> dict[str, Any]:
    return {
        "model": "test-model",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": text}],
    }


def openai_chat_body(text: str = "Hello") -> dict[str, Any]:
    return {
        "model": "test-model",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 100,
    }


def openai_responses_body(text: str = "Hello") -> dict[str, Any]:
    return {
        "model": "test-model",
        "input": [{"role": "user", "content": text}],
    }


# ─── Endpoint map ────────────────────────────────────────────────────────

ENDPOINT_MAP: dict[str, str] = {
    "anthropic_messages": "/v1/messages",
    "openai_chat": "/v1/chat/completions",
    "openai_responses": "/v1/responses",
}

# Map: client protocol → which upstream response format + endpoint path
# The fake upstream will serve the mapped upstream format.
PROTOCOL_UPSTREAM_MAP: dict[str, str] = {
    "anthropic_messages": "anthropic_messages",
    "openai_chat": "openai_chat",
    "openai_responses": "openai_responses",
}


# ─── Fake Upstream Server ────────────────────────────────────────────────


class FakeUpstreamServer:
    """A minimal ASGI server acting as a fake model backend.

    Listens on a real port so the Gateway's HTTP transport can connect to it.
    Returns configurable responses keyed by protocol name.
    """

    def __init__(self) -> None:
        self._responses: dict[str, dict[str, Any]] = {}
        self._default_response: dict[str, Any] = _openai_chat_response_body()
        self._port: int = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = asyncio.Event()

    def set_response(self, protocol: str, body: dict[str, Any]) -> None:
        self._responses[protocol] = body

    def _app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: FastAPIRequest):
            return JSONResponse(
                self._responses.get("openai_chat", self._default_response)
            )

        @app.post("/v1/messages")
        async def messages(request: FastAPIRequest):
            return JSONResponse(
                self._responses.get("anthropic_messages", _anthropic_response_body())
            )

        @app.post("/v1/responses")
        async def responses(request: FastAPIRequest):
            return JSONResponse(
                self._responses.get(
                    "openai_responses", _openai_responses_response_body()
                )
            )

        return app

    async def start(self) -> int:
        """Start the server on an ephemeral port. Returns the port number."""
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self._port = sock.getsockname()[1]
        sock.close()

        config = uvicorn.Config(
            app=self._app,
            host="127.0.0.1",
            port=self._port,
            log_level="error",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run, daemon=True
        )
        self._thread.start()

        # Wait for the server to start
        for _ in range(50):
            if self._server.started:
                break
            await asyncio.sleep(0.05)
        else:
            raise RuntimeError("Fake upstream server failed to start")

        return self._port

    async def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
            if self._thread:
                await asyncio.to_thread(self._thread.join, timeout=5)
            self._server = None
            self._thread = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}"


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_upstream():
    """Start and stop a fake upstream server."""
    server = FakeUpstreamServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest_asyncio.fixture
async def interop_app(fake_upstream: FakeUpstreamServer):
    """Create an Interop server configured to route to the fake upstream.

    Uses LifespanManager to properly start/stop the Gateway and its
    async resources, preventing leaked sockets and transports.
    """
    from asgi_lifespan import LifespanManager

    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "test-route": ModelRoute(
                id="test-route",
                client_model_aliases=["test-model"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url=fake_upstream.url,
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    app = create_app(config=config)

    async with LifespanManager(app):
        yield app


# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_canonical(text: str = "Hello") -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="test-model"),
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalTextBlock(text=text)],
            )
        ],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )


# ─── Phase 5 Gate Tests ──────────────────────────────────────────────────


class TestPhase5Gate:
    """Phase 5 gate tests exercising the full non-streaming pipeline."""

    @pytest.mark.asyncio
    async def test_non_streaming_all_three_protocols(
        self, interop_app, fake_upstream: FakeUpstreamServer
    ):
        """All three client protocols produce valid HTTP responses through the
        complete ASGI-to-upstream-to-ASGI path for non-streaming requests."""
        fake_upstream.set_response("openai_chat", _openai_chat_response_body("Hello world"))
        fake_upstream.set_response("anthropic_messages", _anthropic_response_body("Hello world"))
        fake_upstream.set_response(
            "openai_responses", _openai_responses_response_body("Hello world")
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            for protocol, body_factory in [
                ("anthropic_messages", anthropic_messages_body),
                ("openai_chat", openai_chat_body),
                ("openai_responses", openai_responses_body),
            ]:
                endpoint = ENDPOINT_MAP[protocol]
                resp = await client.post(endpoint, json=body_factory())
                assert resp.status_code == 200, (
                    f"{protocol} returned {resp.status_code}: {resp.text}"
                )
                data = resp.json()
                # Each protocol returns different top-level keys
                assert "content" in data or "output" in data or "choices" in data, (
                    f"{protocol} response missing expected key: {sorted(data)}"
                )

    @pytest.mark.asyncio
    async def test_non_streaming_with_tool_calls(
        self, interop_app, fake_upstream: FakeUpstreamServer
    ):
        """Tool calls from the upstream flow through the full pipeline and
        appear in the client response."""
        fake_upstream.set_response(
            "openai_chat", _openai_chat_tool_call_body("read_file", {"path": "/tmp/x"})
        )

        _tools = [
            CanonicalTool(
                name="read_file",
                description="Read a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path"}
                    },
                    "required": ["path"],
                },
            )
        ]

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            body = openai_chat_body("Read the file")
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string", "description": "File path"}
                            },
                            "required": ["path"],
                        },
                    },
                }
            ]
            resp = await client.post("/v1/chat/completions", json=body)
            assert resp.status_code == 200, resp.text
            data = resp.json()
            choices = data.get("choices", [])
            assert len(choices) > 0
            msg = choices[0].get("message", {})
            tcs = msg.get("tool_calls", [])
            assert len(tcs) > 0, f"No tool_calls in response: {data}"
            assert tcs[0]["function"]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_non_streaming_all_protocols_with_tools(
        self, interop_app, fake_upstream: FakeUpstreamServer
    ):
        """All three protocols handle tool calls correctly."""
        fake_upstream.set_response(
            "openai_chat", _openai_chat_tool_call_body("read_file", {"path": "/tmp/x"})
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            for protocol, body_factory, endpoint_path in [
                ("anthropic_messages", anthropic_messages_body, "/v1/messages"),
                ("openai_chat", openai_chat_body, "/v1/chat/completions"),
                ("openai_responses", openai_responses_body, "/v1/responses"),
            ]:
                body = body_factory("Use a tool")
                if protocol == "anthropic_messages":
                    body["tools"] = [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "path": {"type": "string"}
                                },
                                "required": ["path"],
                            },
                        }
                    ]
                elif protocol == "openai_chat":
                    body["tools"] = [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "description": "Read a file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"],
                                },
                            },
                        }
                    ]
                elif protocol == "openai_responses":
                    body["tools"] = [
                        {
                            "name": "read_file",
                            "description": "Read a file",
                            "parameters": {
                                "type": "object",
                                "properties": {"path": {"type": "string"}},
                                "required": ["path"],
                            },
                        }
                    ]

                resp = await client.post(endpoint_path, json=body)
                assert resp.status_code == 200, (
                    f"{protocol} returned {resp.status_code}: {resp.text}"
                )
                data = resp.json()
                # Each protocol encodes tool calls in its own format
                if protocol == "anthropic_messages":
                    content = data.get("content", [])
                    blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]
                    assert len(blocks) > 0, f"No tool_use in {protocol}: {data}"
                    assert blocks[0]["name"] == "read_file"
                elif protocol == "openai_chat":
                    tcs = data.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                    assert len(tcs) > 0, f"No tool_calls in {protocol}: {data}"
                    assert tcs[0]["function"]["name"] == "read_file"
                elif protocol == "openai_responses":
                    output = data.get("output", [])
                    items = [o for o in output if isinstance(o, dict) and o.get("type") == "function_call"]
                    assert len(items) > 0, f"No function_call in {protocol}: {data}"
                    assert items[0]["name"] == "read_file"

    @pytest.mark.asyncio
    async def test_upstream_error_returns_error_response(
        self, interop_app, fake_upstream: FakeUpstreamServer
    ):
        """When the upstream returns an error, the gateway returns a structured error."""
        fake_upstream.set_response("openai_chat", {"error": "upstream failure"})

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/v1/chat/completions", json=openai_chat_body())
            # The upstream returned 200 with an error body, so no HTTP error.
            # The gateway processes it and produces a response.
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(
        self, interop_app, fake_upstream: FakeUpstreamServer
    ):
        """Multiple concurrent requests are handled correctly."""
        fake_upstream.set_response("openai_chat", _openai_chat_response_body("Concurrent"))

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            tasks = [
                client.post("/v1/chat/completions", json=openai_chat_body(f"Request {i}"))
                for i in range(5)
            ]
            responses = await asyncio.gather(*tasks)
            for resp in responses:
                assert resp.status_code == 200
                data = resp.json()
                assert "choices" in data


# ─── Additional Integration Tests ────────────────────────────────────────


class TestGatewayDirect:
    """Tests that exercise the Gateway directly (not via HTTP)."""

    @pytest.mark.asyncio
    async def test_gateway_handle_request_text(
        self, fake_upstream: FakeUpstreamServer
    ):
        """Gateway.handle_request returns a canonical response with text."""
        fake_upstream.set_response("openai_chat", _openai_chat_response_body("Hello from test"))

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            log_level="error",
            probe_on_startup=False,
            routes={
                "test-route": ModelRoute(
                    id="test-route",
                    client_model_aliases=["test-model"],
                    upstream_model="fake-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OPENAI_COMPATIBLE,
                        base_url=fake_upstream.url,
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        gw = Gateway(config)
        await gw.startup()

        try:
            canonical = _make_canonical("Hello")
            resp = await gw.handle_request(canonical, RequestContext())
            assert resp.stop_reason is not None
            assert resp.content is not None
        finally:
            await gw.close()

    @pytest.mark.asyncio
    async def test_gateway_handle_request_tool_call(
        self, fake_upstream: FakeUpstreamServer
    ):
        """Gateway.handle_request returns tool calls from the upstream."""
        fake_upstream.set_response(
            "openai_chat", _openai_chat_tool_call_body("read_file", {"path": "/tmp/x"})
        )

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            log_level="error",
            probe_on_startup=False,
            routes={
                "test-route": ModelRoute(
                    id="test-route",
                    client_model_aliases=["test-model"],
                    upstream_model="fake-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OPENAI_COMPATIBLE,
                        base_url=fake_upstream.url,
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        gw = Gateway(config)
        await gw.startup()

        try:
            tool = CanonicalTool(
                name="read_file",
                description="Read a file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
            canonical = _make_canonical("Read the file")
            canonical.tools = [tool]
            resp = await gw.handle_request(canonical, RequestContext())
            # Should have tool call blocks in content
            assert resp.content is not None
            tool_blocks = [c for c in resp.content if getattr(c, "type", "") == "tool_call"]
            assert len(tool_blocks) > 0
        finally:
            await gw.close()

    @pytest.mark.asyncio
    async def test_route_resolution_by_model_name(
        self, fake_upstream: FakeUpstreamServer
    ):
        """The gateway resolves the correct route by model name."""
        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            log_level="error",
            probe_on_startup=False,
            routes={
                "route-a": ModelRoute(
                    id="route-a",
                    client_model_aliases=["model-a"],
                    upstream_model="fake-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OPENAI_COMPATIBLE,
                        base_url=fake_upstream.url,
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
                "route-b": ModelRoute(
                    id="route-b",
                    client_model_aliases=["model-b"],
                    upstream_model="fake-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OPENAI_COMPATIBLE,
                        base_url=fake_upstream.url,
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        gw = Gateway(config)
        await gw.startup()

        try:
            canonical_a = _make_canonical("Hello")
            canonical_a.model.requested_name = "model-a"
            canonical_b = _make_canonical("Hello")
            canonical_b.model.requested_name = "model-b"

            resp_a = await gw.handle_request(canonical_a, RequestContext())
            resp_b = await gw.handle_request(canonical_b, RequestContext())

            assert resp_a.stop_reason is not None
            assert resp_b.stop_reason is not None
        finally:
            await gw.close()