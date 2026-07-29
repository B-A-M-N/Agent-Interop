"""Phase 7 gate: Full protocol matrix integration tests.

Exercises every combination of (client protocol × upstream protocol × tool mode
× streaming flag) from the cross-product table in PROPOSED_CHANGES.md.

Each test runs the complete ASGI pipeline:
  ASGI endpoint → adapter.decode_request() → CanonicalRequest
  → route resolution → codec render → HTTP transport → codec decode
  → tool_transaction_service → canonical response assembly
  → adapter.encode_response() → HTTP response

The fake upstream runs as a real HTTP server (uvicorn in a background thread)
so the HTTP transport layer is exercised.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.server.app import create_app

# ─── Fake Upstream (supports both formats) ────────────────────────────────


class FakeMatrixUpstream:
    """A fake upstream that serves both OpenAI Chat and Ollama Chat formats.

    Has dedicated endpoints:
      /v1/chat/completions — OpenAI Chat format
      /api/chat           — Ollama Chat format (NDJSON for streaming)
    """

    def __init__(self) -> None:
        self._text_response: str = "Hello from upstream"
        self._tool_calls: list[dict[str, Any]] = []
        self._port: int = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def set_text_response(self, text: str) -> None:
        self._text_response = text
        self._tool_calls = []

    def set_tool_call_response(
        self, name: str = "test_tool", arguments: dict[str, Any] | None = None
    ) -> None:
        self._text_response = ""
        self._tool_calls = [
            {
                "id": "tc_matrix_001",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {"key": "value"}),
                },
            }
        ]

    def _build_openai_body(self, stream: bool = False) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self._text_response or None,
        }
        finish = "stop"
        if self._tool_calls:
            message["tool_calls"] = self._tool_calls
            finish = "tool_calls"
        return {
            "id": "matrix-chat-response",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _build_ollama_body(self, stream: bool = False) -> dict[str, Any]:
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": self._text_response or "",
        }
        if self._tool_calls:
            msg["tool_calls"] = self._tool_calls
        return {
            "model": "fake-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": msg,
            "done": True,
            "done_reason": "stop" if not self._tool_calls else "tool_calls",
        }

    async def _stream_openai(self) -> AsyncIterator[bytes]:
        if self._text_response:
            chunk = {
                "choices": [{
                    "delta": {"content": self._text_response},
                    "index": 0,
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        for tc in self._tool_calls:
            fn = tc.get("function", {})
            chunk = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 0,
                            "id": tc.get("id", "tc_matrix_001"),
                            "type": "function",
                            "function": {
                                "name": fn.get("name", ""),
                                "arguments": fn.get("arguments", "{}"),
                            },
                        }]
                    },
                    "index": 0,
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
        finish = "tool_calls" if self._tool_calls else "stop"
        chunk = {"choices": [{"delta": {}, "index": 0, "finish_reason": finish}]}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def _stream_ollama(self) -> AsyncIterator[bytes]:
        if self._text_response:
            chunk = {
                "model": "fake-model",
                "message": {"role": "assistant", "content": self._text_response},
                "done": False,
            }
            yield f"{json.dumps(chunk)}\n".encode()
        for tc in self._tool_calls:
            fn = tc.get("function", {})
            chunk = {
                "model": "fake-model",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        }
                    }],
                },
                "done": False,
            }
            yield f"{json.dumps(chunk)}\n".encode()
        done = {
            "model": "fake-model",
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "tool_calls" if self._tool_calls else "stop",
        }
        yield f"{json.dumps(done)}\n".encode()

    def _app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def openai_chat(request: FastAPIRequest):
            body = await request.json()
            if body.get("stream", False):
                return StreamingResponse(
                    self._stream_openai(),
                    media_type="text/event-stream",
                )
            return JSONResponse(self._build_openai_body())

        @app.post("/api/chat")
        async def ollama_chat(request: FastAPIRequest):
            body = await request.json()
            if body.get("stream", False):
                return StreamingResponse(
                    self._stream_ollama(),
                    media_type="application/x-ndjson",
                )
            return JSONResponse(self._build_ollama_body())

        return app

    async def start(self) -> int:
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
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
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


# ─── Client protocol body factories ───────────────────────────────────────


def anthropic_messages_body(
    text: str = "Hello", stream: bool = False, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "test-model",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": text}],
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    return body


def openai_chat_body(
    text: str = "Hello", stream: bool = False, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 100,
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    return body


def openai_responses_body(
    text: str = "Hello", stream: bool = False, tools: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": "test-model",
        "input": [{"role": "user", "content": text}],
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    return body


# ─── Client endpoints ─────────────────────────────────────────────────────

CLIENT_ENDPOINTS: dict[str, str] = {
    "anthropic_messages": "/v1/messages",
    "openai_chat": "/v1/chat/completions",
    "openai_responses": "/v1/responses",
}

# ─── Tool payloads by client protocol ─────────────────────────────────────

ANTHROPIC_TOOL: list[dict[str, Any]] = [
    {
        "name": "test_tool",
        "description": "A test tool",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }
]

OPENAI_TOOL: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    }
]

RESPONSES_TOOL: list[dict[str, Any]] = [
    {
        "name": "test_tool",
        "description": "A test tool",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    }
]

TOOL_PAYLOADS: dict[str, list[dict[str, Any]]] = {
    "anthropic_messages": ANTHROPIC_TOOL,
    "openai_chat": OPENAI_TOOL,
    "openai_responses": RESPONSES_TOOL,
}

# ─── SSE collector ────────────────────────────────────────────────────────


def collect_sse(resp: httpx.Response) -> list[dict[str, Any]]:
    """Collect SSE data events from a streaming response."""
    events: list[dict[str, Any]] = []
    buffer = ""
    for chunk in resp.iter_text():
        buffer += chunk
        while "\n\n" in buffer:
            record, buffer = buffer.split("\n\n", 1)
            for line in record.split("\n"):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        continue
                    try:
                        events.append(json.loads(data_str))
                    except json.JSONDecodeError:
                        pass
    return events


# ─── Result type ──────────────────────────────────────────────────────────


class MatrixTestResult:
    """Result of a single matrix test cell."""

    def __init__(self, success: bool, error: str = "") -> None:
        self.success = success
        self.error = error


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_upstream():
    """Start and stop a fake upstream server."""
    server = FakeMatrixUpstream()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@asynccontextmanager
async def _create_interop_app(
    upstream_url: str,
    wire_protocol: UpstreamProtocol,
    tool_mode: ToolMode = ToolMode.AUTO,
) -> AsyncGenerator[FastAPI, None]:
    """Create an Interop server configured for a specific matrix cell.

    Uses LifespanManager to properly start/stop the Gateway and its
    async resources, preventing leaked sockets and transports.
    """
    from asgi_lifespan import LifespanManager

    # Pick a kind that supports the wire protocol
    if wire_protocol == UpstreamProtocol.OLLAMA_CHAT:
        kind = UpstreamKind.OLLAMA
    elif wire_protocol == UpstreamProtocol.ANTHROPIC_MESSAGES:
        kind = UpstreamKind.ANTHROPIC
    elif wire_protocol == UpstreamProtocol.OPENAI_RESPONSES:
        kind = UpstreamKind.OPENAI
    else:
        kind = UpstreamKind.OPENAI_COMPATIBLE

    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "matrix-route": ModelRoute(
                id="matrix-route",
                client_model_aliases=["test-model"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=kind,
                    base_url=upstream_url,
                    wire_protocol=wire_protocol,
                    timeout_seconds=30.0,
                ),
                tool_mode=tool_mode,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    app = create_app(config=config)

    async with LifespanManager(app) as manager:
        yield manager.app


# ─── Semantic check for PROMPTED-mode tool-call extraction ─────────────────


def _check_tool_call_in_response(
    data: dict[str, Any], client_protocol: str
) -> str | None:
    """Return an error string if the PROMPTED-mode response does not contain a
    real tool-call block for ``test_tool`` with arguments ``{"key": "value"}``,
    or if the literal ``<tool_call>`` envelope text leaked into visible text.

    Protocol-appropriate:
      - anthropic_messages: ``content`` has a ``tool_use`` block with ``input``
      - openai_chat: ``choices[0].message.tool_calls`` has a function call
      - openai_responses: ``output`` has a ``function_call`` item
    Returns None when the response is correct.
    """
    if client_protocol == "anthropic_messages":
        blocks = data.get("content", [])
        tool_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not any(
            b.get("name") == "test_tool" and b.get("input") == {"key": "value"}
            for b in tool_uses
        ):
            return f"expected tool_use test_tool(key=value), got content={blocks}"
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text" and "<tool_call>" in b.get("text", ""):
                return f"<tool_call> leaked into Anthropic text: {b['text']!r}"
    elif client_protocol == "openai_chat":
        msg = (data.get("choices") or [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls") or []
        if not any(
            tc.get("function", {}).get("name") == "test_tool"
            and json.loads(tc.get("function", {}).get("arguments", "{}")) == {"key": "value"}
            for tc in tool_calls
        ):
            return f"expected tool_call test_tool(key=value), got message={msg}"
        if "<tool_call>" in (msg.get("content") or ""):
            return f"<tool_call> leaked into OpenAI Chat content: {msg['content']!r}"
    elif client_protocol == "openai_responses":
        output = data.get("output", [])
        func_calls = [o for o in output if isinstance(o, dict) and o.get("type") == "function_call"]
        if not any(
            o.get("name") == "test_tool"
            and json.loads(o.get("arguments", "{}")) == {"key": "value"}
            for o in func_calls
        ):
            return f"expected function_call test_tool(key=value), got output={output}"
        for o in output:
            if isinstance(o, dict) and o.get("type") == "message":
                for part in o.get("content", []):
                    if "<tool_call>" in part.get("text", ""):
                        return f"<tool_call> leaked into Responses text: {part['text']!r}"
    else:
        return None  # unknown protocol: skip semantic check
    return None


# ─── Matrix test runner ───────────────────────────────────────────────────


async def run_matrix_test(
    app: FastAPI,
    fake_upstream_server: FakeMatrixUpstream,
    client_protocol: str,
    upstream_protocol: UpstreamProtocol,
    tool_mode: ToolMode,
    streaming: bool,
    use_tool_calls: bool = False,
) -> MatrixTestResult:
    """Run a single cell in the protocol matrix.

    Sends a request through the ASGI pipeline with the given pre-configured
    Interop server and validates the response.
    """

    # Configure the fake upstream response
    if use_tool_calls:
        if tool_mode == ToolMode.PROMPTED:
            # PROMPTED mode expects text with <tool_call> blocks, not native tool_calls
            fake_upstream_server.set_text_response(
                '<tool_call>{"name":"test_tool","arguments":{"key":"value"}}</tool_call>'
            )
        else:
            fake_upstream_server.set_tool_call_response(
                "test_tool", {"key": "value"}
            )
    else:
        fake_upstream_server.set_text_response("Hello matrix")

    # Build the client request body
    tool_payload = TOOL_PAYLOADS.get(client_protocol) if use_tool_calls else None
    body_factories = {
        "anthropic_messages": anthropic_messages_body,
        "openai_chat": openai_chat_body,
        "openai_responses": openai_responses_body,
    }
    body = body_factories[client_protocol](
        text="Hello",
        stream=streaming,
        tools=tool_payload,
    )

    endpoint = CLIENT_ENDPOINTS[client_protocol]

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        try:
            resp = await client.post(endpoint, json=body)
            if resp.status_code != 200:
                return MatrixTestResult(
                    False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                )

            if streaming:
                events = collect_sse(resp)
                if not events:
                    return MatrixTestResult(
                        False, "No SSE events in streaming response"
                    )
            else:
                data = resp.json()
                # Each protocol returns different top-level keys
                valid = (
                    "content" in data
                    or "output" in data
                    or "choices" in data
                )
                if not valid:
                    return MatrixTestResult(
                        False,
                        f"Missing expected response keys: {sorted(data)}",
                    )

                # PROMPTED-mode semantic check: the <tool_call> envelope must be
                # recovered as a real tool-call block and must not leak as text.
                if use_tool_calls and tool_mode == ToolMode.PROMPTED:
                    error = _check_tool_call_in_response(data, client_protocol)
                    if error is not None:
                        return MatrixTestResult(False, error)

            return MatrixTestResult(True)

        except Exception as exc:
            return MatrixTestResult(False, str(exc))


# ─── Phase 7 Gate Tests ──────────────────────────────────────────────────


class TestPhase7ProtocolMatrix:
    """Full protocol matrix: (client × upstream × mode × streaming)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "client_protocol,upstream_protocol,tool_mode,streaming,use_tool_calls",
        [
            # anthropic_messages × ollama_chat
            ("anthropic_messages", UpstreamProtocol.OLLAMA_CHAT, ToolMode.NATIVE, False, False),
            ("anthropic_messages", UpstreamProtocol.OLLAMA_CHAT, ToolMode.NATIVE, False, True),
            ("anthropic_messages", UpstreamProtocol.OLLAMA_CHAT, ToolMode.PROMPTED, False, False),
            ("anthropic_messages", UpstreamProtocol.OLLAMA_CHAT, ToolMode.PROMPTED, False, True),
            # anthropic_messages × openai_chat
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, False, False),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, False, True),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, True, False),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, True, True),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, False, False),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, False, True),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, True, False),
            ("anthropic_messages", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, True, True),
            # openai_chat × ollama_chat
            ("openai_chat", UpstreamProtocol.OLLAMA_CHAT, ToolMode.NATIVE, False, False),
            ("openai_chat", UpstreamProtocol.OLLAMA_CHAT, ToolMode.NATIVE, False, True),
            ("openai_chat", UpstreamProtocol.OLLAMA_CHAT, ToolMode.NATIVE, True, False),
            ("openai_chat", UpstreamProtocol.OLLAMA_CHAT, ToolMode.NATIVE, True, True),
            # openai_chat × openai_chat
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, False, False),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, False, True),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, True, False),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, True, True),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, False, False),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, False, True),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, True, False),
            ("openai_chat", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, True, True),
            # openai_responses × openai_chat
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, False, False),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, False, True),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, True, False),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.NATIVE, True, True),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, False, False),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, False, True),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, True, False),
            ("openai_responses", UpstreamProtocol.OPENAI_CHAT, ToolMode.PROMPTED, True, True),
        ],
        ids=lambda p: (
            f"{p}" if isinstance(p, str)
            else f"{p.value}" if isinstance(p, Enum)
            else str(p)
        ),
    )
    @pytest.mark.asyncio
    async def test_matrix_cell(
        self,
        fake_upstream: FakeMatrixUpstream,
        client_protocol: str,
        upstream_protocol: UpstreamProtocol,
        tool_mode: ToolMode,
        streaming: bool,
        use_tool_calls: bool,
    ):
        """Test a single cell in the protocol matrix."""
        async with _create_interop_app(
            upstream_url=fake_upstream.url,
            wire_protocol=upstream_protocol,
            tool_mode=tool_mode,
        ) as app:
            result = await run_matrix_test(
                app=app,
                fake_upstream_server=fake_upstream,
                client_protocol=client_protocol,
                upstream_protocol=upstream_protocol,
                tool_mode=tool_mode,
                streaming=streaming,
                use_tool_calls=use_tool_calls,
            )
        assert result.success, (
            f"Failed: {client_protocol}→{upstream_protocol.value}"
            f"({tool_mode.value},stream={streaming},tools={use_tool_calls}): "
            f"{result.error}"
        )


# ─── Selected individual tests for debugging ─────────────────────────────


class TestMatrixSanity:
    """Sanity checks for individual matrix cells — useful for debugging."""

    @pytest.mark.asyncio
    async def test_openai_to_openai_native_non_streaming(
        self, fake_upstream: FakeMatrixUpstream
    ):
        """OpenAI Chat → OpenAI Chat, NATIVE, non-streaming, text only."""
        async with _create_interop_app(
            upstream_url=fake_upstream.url,
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
            tool_mode=ToolMode.NATIVE,
        ) as app:
            fake_upstream.set_text_response("Hello openai→openai")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=openai_chat_body("Hello"),
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                assert "choices" in data
                # Semantic: response contains the model's text
                msg = data["choices"][0]["message"]
                assert msg.get("content") == "Hello openai→openai"

    @pytest.mark.asyncio
    async def test_openai_to_openai_with_tool_calls(
        self, fake_upstream: FakeMatrixUpstream
    ):
        """OpenAI Chat → OpenAI Chat, NATIVE, non-streaming, with tool calls."""
        async with _create_interop_app(
            upstream_url=fake_upstream.url,
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
            tool_mode=ToolMode.NATIVE,
        ) as app:
            fake_upstream.set_tool_call_response("test_tool", {"key": "value"})

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=openai_chat_body("Use a tool", tools=OPENAI_TOOL),
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                choices = data.get("choices", [])
                assert len(choices) > 0
                msg = choices[0].get("message", {})
                tcs = msg.get("tool_calls", [])
                assert len(tcs) > 0, f"No tool_calls: {data}"
                # Semantic: tool call has correct name and arguments
                tc_fn = tcs[0].get("function", {})
                assert tc_fn.get("name") == "test_tool"
                assert json.loads(tc_fn.get("arguments", "{}")) == {"key": "value"}

    @pytest.mark.asyncio
    async def test_anthropic_to_openai_non_streaming(
        self, fake_upstream: FakeMatrixUpstream
    ):
        """Anthropic Messages → OpenAI Chat, NATIVE, non-streaming."""
        async with _create_interop_app(
            upstream_url=fake_upstream.url,
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
            tool_mode=ToolMode.NATIVE,
        ) as app:
            fake_upstream.set_text_response("Hello anthropic→openai")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/v1/messages",
                    json=anthropic_messages_body("Hello"),
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                assert "content" in data
                # Semantic: Anthropic response contains the text block
                content_blocks = data.get("content", [])
                assert len(content_blocks) > 0
                assert content_blocks[0].get("text") == "Hello anthropic→openai"

    @pytest.mark.asyncio
    async def test_openai_streaming_text(
        self, fake_upstream: FakeMatrixUpstream
    ):
        """OpenAI Chat → OpenAI Chat, streaming, text."""
        async with _create_interop_app(
            upstream_url=fake_upstream.url,
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
            tool_mode=ToolMode.NATIVE,
        ) as app:
            fake_upstream.set_text_response("Hello streaming")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=openai_chat_body("Hello", stream=True),
                )
                assert resp.status_code == 200
                events = collect_sse(resp)
                assert len(events) > 0

    @pytest.mark.asyncio
    async def test_ollama_as_upstream(
        self, fake_upstream: FakeMatrixUpstream
    ):
        """Anthropic Messages → Ollama Chat, NATIVE, non-streaming."""
        async with _create_interop_app(
            upstream_url=fake_upstream.url,
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
            tool_mode=ToolMode.NATIVE,
        ) as app:
            fake_upstream.set_text_response("Hello ollama")

            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/v1/messages",
                    json=anthropic_messages_body("Hello"),
                )
                assert resp.status_code == 200, resp.text
                data = resp.json()
                assert "content" in data