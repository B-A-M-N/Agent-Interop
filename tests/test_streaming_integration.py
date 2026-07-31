"""Phase 6 gate: Streaming equivalence integration tests.

Verifies that for the same input and upstream response, the streaming and
non-streaming paths produce identical CanonicalToolCallBlock objects.

Exercises the full ASGI pipeline with streaming:
  ASGI endpoint (stream=True) → adapter.decode_request()
  → CanonicalRequest → gateway.handle_stream()
  → SSE stream → adapter.encode_stream_event() → SSE events

The fake upstream runs as a real HTTP server (uvicorn in a background thread)
so the HTTP transport layer is exercised for the Interop→upstream leg.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Self

import httpx
import pytest
import pytest_asyncio
import uvicorn
from fastapi import FastAPI
from fastapi import Request as FastAPIRequest
from fastapi.responses import JSONResponse, StreamingResponse

from agent_interop.abi import (
    CanonicalEvent,
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.config import (
    CompatibilityConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.execution import ExecutionState, InteropRequestExecution
from agent_interop.gateway import Gateway, ResolvedInvocation
from agent_interop.repair.invocation import build_invocation_plan
from agent_interop.server.app import create_app
from agent_interop.transport.http import PreparedUpstreamRequest
from agent_interop.upstreams.registry import get_codec

# ─── Fake Upstream Server (streaming-capable) ─────────────────────────────


class FakeStreamingUpstreamServer:
    """A minimal ASGI server acting as a fake model backend.

    Supports both streaming and non-streaming endpoints so the same server
    can be used for equivalence testing. For streaming, yields SSE chunks
    in OpenAI Chat format.
    """

    def __init__(self) -> None:
        self._text_response: str = "Hello from upstream"
        self._tool_calls: list[dict[str, Any]] = []
        self._port: int = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._started = asyncio.Event()

    def set_text_response(self, text: str) -> None:
        self._text_response = text
        self._tool_calls = []

    def set_tool_call_response(
        self, name: str = "read_file", arguments: dict[str, Any] | None = None
    ) -> None:
        self._text_response = ""
        self._tool_calls = [
            {
                "id": "tc_fake_001",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments or {"path": "/tmp/x"}),
                },
            }
        ]

    def _build_non_streaming_body(self) -> dict[str, Any]:
        """Build an OpenAI Chat non-streaming response body."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self._text_response or None,
        }
        finish = "stop"
        if self._tool_calls:
            message["tool_calls"] = self._tool_calls
            finish = "tool_calls"
        return {
            "id": "fake-chat-response",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    async def _streaming_body(self) -> AsyncIterator[bytes]:
        """Yield SSE chunks for the configured response."""
        if self._text_response:
            chunk = {
                "choices": [{
                    "delta": {"content": self._text_response},
                    "index": 0,
                    "finish_reason": None,
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()

        if self._tool_calls:
            for tc in self._tool_calls:
                fn = tc.get("function", {})
                chunk = {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": tc.get("id", "tc_fake_001"),
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

        # Finish chunk
        finish_reason = "tool_calls" if self._tool_calls else "stop"
        chunk = {
            "choices": [{"delta": {}, "index": 0, "finish_reason": finish_reason}]
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def _app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat_completions(request: FastAPIRequest):
            body = await request.json()
            if body.get("stream", False):
                return StreamingResponse(
                    self._streaming_body(),
                    media_type="text/event-stream",
                )
            return JSONResponse(self._build_non_streaming_body())

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


# ─── Fixtures ─────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fake_upstream():
    """Start and stop a fake upstream server (streaming-capable)."""
    server = FakeStreamingUpstreamServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


@pytest_asyncio.fixture
async def interop_app(fake_upstream: FakeStreamingUpstreamServer):
    """Create an Interop server configured to route to the fake upstream."""
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


# ─── Helpers ──────────────────────────────────────────────────────────────


async def _collect_sse(resp: httpx.Response) -> list[dict[str, Any]]:
    """Collect all SSE data events from a streaming response."""
    events: list[dict[str, Any]] = []
    buffer = ""
    async for chunk in resp.aiter_text():
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


def _extract_tool_calls_from_non_streaming(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract tool calls from a non-streaming OpenAI Chat response."""
    choices = data.get("choices", [])
    if not choices:
        return []
    msg = choices[0].get("message", {})
    return msg.get("tool_calls", [])


def _extract_tool_calls_from_streaming(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reassemble tool calls from streaming SSE events.

    OpenAI Chat streaming sends tool calls as delta fragments.
    We need to reassemble multi-chunk tool calls and filter out
    intermediate deltas, keeping only the final assembled view.
    """
    # For the streaming path, the gateway emits tool_use canonical events
    # which the adapter encodes. The OpenAI Chat adapter currently sends
    # finish_reason="tool_calls" on message_stop. Since the chat adapter
    # doesn't stream tool call deltas (it returns None for tool_use_delta),
    # the tool call info appears only in the final aggregated response.
    #
    # For equivalence testing, we check that the streaming path produces
    # canonical events that encode the same tool call info. We look for
    # the message_stop event which carries finish_reason, and verify
    # completeness via the events stream.
    #
    # The real equivalence check is done at the CanonicalEvent level
    # via gateway.handle_stream() → gateway.handle_request() comparison,
    # not just at the HTTP response level.
    tool_calls: list[dict[str, Any]] = []

    # Accumulate all tool call deltas across chunks
    accum: dict[int, dict[str, Any]] = {}
    for event in events:
        choices = event.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        raw_tcs = delta.get("tool_calls", [])
        for raw_tc in raw_tcs:
            idx = raw_tc.get("index", 0)
            if idx not in accum:
                accum[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
            tc_data = accum[idx]
            if raw_tc.get("id"):
                tc_data["id"] = raw_tc["id"]
            fn = raw_tc.get("function", {})
            if fn.get("name"):
                tc_data["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                tc_data["function"]["arguments"] += fn["arguments"]

    for idx in sorted(accum.keys()):
        tc = accum[idx]
        # Only include if we have a complete function call
        if tc["function"]["name"] and tc["function"]["arguments"]:
            tool_calls.append(tc)

    return tool_calls


def _deep_equal_tool_calls(
    calls_a: list[dict[str, Any]], calls_b: list[dict[str, Any]]
) -> bool:
    """Check if two lists of tool calls are semantically equal.

    Compares function names and arguments (parsed as JSON for deep equality).
    """
    if len(calls_a) != len(calls_b):
        return False
    for a, b in zip(calls_a, calls_b):
        if a.get("function", {}).get("name") != b.get("function", {}).get("name"):
            return False
        try:
            args_a = json.loads(a.get("function", {}).get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args_a = a.get("function", {}).get("arguments", {})
        try:
            args_b = json.loads(b.get("function", {}).get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            args_b = b.get("function", {}).get("arguments", {})
        if args_a != args_b:
            return False
    return True


# ─── Phase 6 Gate Tests ──────────────────────────────────────────────────


class TestPhase6Gate:
    """Phase 6 gate: streaming/non-streaming equivalence."""

    @pytest.mark.asyncio
    async def test_streaming_text_response(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        """Text-only responses: streaming path produces valid SSE events."""
        fake_upstream.set_text_response("Hello from streaming")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100, "stream": True},
            )
            assert resp.status_code == 200
            events = await _collect_sse(resp)
            assert len(events) > 0, f"Should have received SSE events, got: {events}"
            # Should have at least one text delta
            text_deltas = [
                e for e in events
                if e.get("choices", [{}])[0].get("delta", {}).get("content")
            ]
            assert len(text_deltas) > 0, "Should have text content in stream"

    @pytest.mark.asyncio
    async def test_streaming_tool_call_events(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        """Tool call responses: streaming path produces valid SSE events."""
        fake_upstream.set_tool_call_response("read_file", {"path": "/tmp/x"})

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Read the file"}],
                    "max_tokens": 100,
                    "stream": True,
                    "tools": [
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
                    ],
                },
            )
            assert resp.status_code == 200
            events = await _collect_sse(resp)
            assert len(events) > 0, "Should have received SSE events"

            # Should see a message_stop event (completion)
            stop_events = [e for e in events if e.get("choices", [{}])[0].get("finish_reason") == "tool_calls"]
            assert len(stop_events) > 0, (
                f"Should have finish_reason='tool_calls': {events}"
            )

    @pytest.mark.asyncio
    async def test_streaming_non_streaming_equivalence_text(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        """For text-only responses, streaming and non-streaming produce equivalent results."""
        fake_upstream.set_text_response("Hello equivalence")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            # Non-streaming
            resp_non = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
            )
            assert resp_non.status_code == 200
            non_data = resp_non.json()
            non_text = non_data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Streaming
            resp_stream = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100, "stream": True},
            )
            assert resp_stream.status_code == 200
            events = await _collect_sse(resp_stream)

            # Reassemble text from streaming deltas
            stream_text = ""
            for event in events:
                choices = event.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    stream_text += delta.get("content", "")

            assert stream_text == non_text, (
                f"Streaming text '{stream_text}' != non-streaming text '{non_text}'"
            )


class TestStreamResponseIdIsDistinctFromRequestId:
    """MVP-12 regression: the stream encoder used to be seeded with
    canonical.request_id (the CLIENT's inbound request identifier) as the
    response_id, conflating two different identities. The response ID must
    be freshly generated per response, not copied from the request."""

    @pytest.mark.asyncio
    async def test_response_id_differs_from_request_id(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        fake_upstream.set_text_response("Hello")
        request_id = "req-fixed-12345"

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/messages",
                headers={"x-request-id": request_id},
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            raw = resp.text

        message_start_ids = []
        for line in raw.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):].strip()
            if payload in ("", "[DONE]"):
                continue
            data = json.loads(payload)
            if data.get("type") == "message_start":
                message_start_ids.append(data["message"]["id"])

        assert message_start_ids, f"expected a message_start frame: {raw}"
        assert message_start_ids[0] != request_id
        assert message_start_ids[0].startswith("msg_")


class TestStreamExecutionFinalizationThroughASGI:
    """MVP-02 regression: streaming bookkeeping must complete through the
    REAL ASGI server path, not just direct Gateway generator iteration.

    server/app.py's event_stream() breaks out of the Gateway's async
    generator as soon as it sees message_stop. Bookkeeping that only ran
    after that yield (evidence write-back, exec_record.finalize_response)
    was therefore never guaranteed to execute through the actual request
    path — only a test that drives the ASGI app end-to-end can catch that.
    """

    @pytest.mark.asyncio
    async def test_successful_stream_leaves_execution_succeeded(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer, monkeypatch
    ):
        fake_upstream.set_text_response("Hello from streaming")

        # Note: _log_summary() is NOT a reliable observation point here. It
        # only runs after handle_stream's outer `async for` loop resumes past
        # its last yield to discover the inner generator is exhausted — and
        # the ASGI consumer never resumes after message_stop (it breaks
        # immediately, then aclosing() throws GeneratorExit into the
        # suspended generator). What we actually care about — did the state
        # transition happen — is driven by finalize_response/finalize_error
        # themselves, which now run BEFORE the terminal yield.
        observed_states: list[ExecutionState] = []
        original_finalize_response = InteropRequestExecution.finalize_response
        original_finalize_error = InteropRequestExecution.finalize_error

        def capturing_finalize_response(self, response):
            original_finalize_response(self, response)
            observed_states.append(self.state)

        def capturing_finalize_error(self, error=None):
            original_finalize_error(self, error)
            observed_states.append(self.state)

        monkeypatch.setattr(InteropRequestExecution, "finalize_response", capturing_finalize_response)
        monkeypatch.setattr(InteropRequestExecution, "finalize_error", capturing_finalize_error)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            events = await _collect_sse(resp)
            assert len(events) > 0

        assert observed_states == [ExecutionState.SUCCEEDED], (
            f"expected the real ASGI streaming path to finalize as SUCCEEDED, "
            f"got {observed_states} (bookkeeping may not have run before the "
            f"server stopped consuming the generator at message_stop)"
        )

    @pytest.mark.asyncio
    async def test_streaming_non_streaming_equivalence_tool_calls(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        """For tool call responses, streaming and non-streaming produce equivalent tool calls."""
        fake_upstream.set_tool_call_response("read_file", {"path": "/tmp/x"})

        tools_payload = [
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

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            # Non-streaming
            resp_non = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Read the file"}],
                    "max_tokens": 100,
                    "tools": tools_payload,
                },
            )
            assert resp_non.status_code == 200
            non_data = resp_non.json()
            non_calls = _extract_tool_calls_from_non_streaming(non_data)

            # Streaming
            resp_stream = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Read the file"}],
                    "max_tokens": 100,
                    "stream": True,
                    "tools": tools_payload,
                },
            )
            assert resp_stream.status_code == 200
            events = await _collect_sse(resp_stream)
            stream_calls = _extract_tool_calls_from_streaming(events)

            assert len(stream_calls) == len(non_calls), (
                f"Streaming has {len(stream_calls)} tool calls, "
                f"non-streaming has {len(non_calls)}"
            )
            assert _deep_equal_tool_calls(stream_calls, non_calls), (
                f"Tool calls differ:\n  streaming: {stream_calls}\n  non-streaming: {non_calls}"
            )

    @pytest.mark.asyncio
    async def test_streaming_error_recovery(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        """Streaming endpoint handles errors gracefully without crashing."""
        # Don't configure any response — the upstream still returns valid HTTP
        # with a default response, so this test verifies the streaming path
        # doesn't crash with missing configuration.

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100, "stream": True},
            )
            # Should still get a valid streaming response
            assert resp.status_code == 200
            events = await _collect_sse(resp)
            assert len(events) > 0

    @pytest.mark.asyncio
    async def test_streaming_multiple_protocols(
        self, interop_app, fake_upstream: FakeStreamingUpstreamServer
    ):
        """All client protocols handle streaming responses."""
        fake_upstream.set_text_response("Hello streaming")

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=interop_app),
            base_url="http://test",
        ) as client:
            for endpoint, body in [
                ("/v1/messages", {
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                }),
                ("/v1/chat/completions", {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 100,
                    "stream": True,
                }),
            ]:
                resp = await client.post(endpoint, json=body)
                assert resp.status_code == 200, (
                    f"{endpoint} streaming returned {resp.status_code}"
                )
                events = await _collect_sse(resp)
                assert len(events) > 0, f"{endpoint} streaming produced no events"


# ─── Gateway Direct Streaming Tests ──────────────────────────────────────


class TestGatewayStreamingDirect:
    """Tests that exercise the Gateway's handle_stream directly."""

    @pytest.mark.asyncio
    async def test_gateway_handle_stream_text(
        self, fake_upstream: FakeStreamingUpstreamServer
    ):
        """Gateway.handle_stream yields text_delta events for text responses."""
        fake_upstream.set_text_response("Hello streaming")

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
            canonical = CanonicalRequest(
                model=CanonicalModelReference(requested_name="test-model"),
                messages=[
                    CanonicalMessage(
                        role="user",
                        content=[CanonicalTextBlock(text="Hello")],
                    )
                ],
                generation=CanonicalGenerationOptions(max_output_tokens=100),
                tool_choice=CanonicalToolChoice.auto(),
            )

            events: list[str] = []
            async for event in gw.handle_stream(canonical, RequestContext()):
                events.append(event.type)

            assert "text_delta" in events, (
                f"Should have text_delta events: {events}"
            )
            assert "message_stop" in events
        finally:
            await gw.close()

    @pytest.mark.asyncio
    async def test_unverified_stream_buffers_until_a_validated_response(
        self, fake_upstream: FakeStreamingUpstreamServer
    ):
        """Schema-v2-style routes use the non-streaming attempt ladder first.

        This proves the model-visible stream is emitted only after the
        buffered response has passed ordinary decode/extraction/validation,
        rather than forwarding speculative textual frames from an unknown
        compatibility path.
        """
        fake_upstream.set_text_response("validated buffered response")
        config = InteropServerConfig(
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
                    ),
                    tool_mode=ToolMode.AUTO,
                    compatibility=CompatibilityConfig(buffer_unverified_streaming=True),
                ),
            },
        )
        gateway = Gateway(config)
        await gateway.startup()
        try:
            request = CanonicalRequest(
                model=CanonicalModelReference(requested_name="test-model"),
                messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hello")])],
                generation=CanonicalGenerationOptions(max_output_tokens=32, stream=True),
            )
            events = [event async for event in gateway.handle_stream(request, RequestContext())]
            assert [event.type for event in events] == ["text_delta", "usage_update", "message_stop"]
            assert events[0].partial == "validated buffered response"
        finally:
            await gateway.close()

    @pytest.mark.asyncio
    async def test_gateway_handle_stream_tool_calls(
        self, fake_upstream: FakeStreamingUpstreamServer
    ):
        """Gateway.handle_stream yields tool_use events for tool call responses."""
        fake_upstream.set_tool_call_response("read_file", {"path": "/tmp/x"})

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
            canonical = CanonicalRequest(
                model=CanonicalModelReference(requested_name="test-model"),
                messages=[
                    CanonicalMessage(
                        role="user",
                        content=[CanonicalTextBlock(text="Read the file")],
                    )
                ],
                tools=[tool],
                generation=CanonicalGenerationOptions(max_output_tokens=100),
                tool_choice=CanonicalToolChoice.auto(),
            )

            events: list[str] = []
            async for event in gw.handle_stream(canonical, RequestContext()):
                events.append(event.type)

            assert "tool_use" in events, (
                f"Should have tool_use events: {events}"
            )
            assert "message_stop" in events
        finally:
            await gw.close()

    @pytest.mark.asyncio
    async def test_gateway_handle_stream_upstream_error_marks_exec_record_failed(self):
        """Upstream 4xx/5xx on a stream must yield an error event + message_stop
        and finalize the exec record as FAILED — never as a successful response.

        Drives ``_handle_stream_send`` directly with a controlled exec_record,
        because ``handle_stream()`` does not expose the record it creates
        internally. The fake transport returns a 500 stream without binding any
        real server (the ``FakeStreamingUpstreamServer`` cannot produce error
        statuses, so we inject a transport here).
        """

        class FakeErrorStream:
            """Minimal ``UpstreamStream`` stand-in returning an error status."""

            def __init__(self, status_code: int, body: str) -> None:
                self.status_code = status_code
                self._body = body

            async def raw_lines(self) -> AsyncIterator[str]:
                for line in self._body.splitlines():
                    yield line

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        class FakeErrorTransport:
            """Transport that always yields an error-status stream."""

            def __init__(self, status_code: int, body: str) -> None:
                self._status_code = status_code
                self._body = body

            @asynccontextmanager
            async def stream(
                self, request: PreparedUpstreamRequest
            ) -> AsyncIterator[FakeErrorStream]:
                yield FakeErrorStream(self._status_code, self._body)

            async def close(self) -> None:
                return None

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
                        base_url="http://127.0.0.1:0",
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )

        route = next(iter(config.routes.values()))
        codec = get_codec(route.upstream.wire_protocol)
        plan = build_invocation_plan(
            tools=[],
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=route.tool_mode,
            model_profile=None,
            repair_policy=None,
        )

        canonical = CanonicalRequest(
            model=CanonicalModelReference(requested_name="test-model"),
            messages=[
                CanonicalMessage(
                    role="user", content=[CanonicalTextBlock(text="Hello")]
                )
            ],
            generation=CanonicalGenerationOptions(max_output_tokens=100),
            tool_choice=CanonicalToolChoice.auto(),
        )

        exec_record = InteropRequestExecution()
        invocation = ResolvedInvocation(
            request_context=RequestContext(),
            original_request=canonical,
            reconciled_request=canonical,
            route=route,
            backend_metadata=None,
            model_profile=None,
            repair_policy=None,
            invocation_plan=plan,
            codec=codec,
            compatibility_key=None,
            evidence_record=None,
            repair_budget=None,
            execution_record=exec_record,
        )

        gw = Gateway(config, transport=FakeErrorTransport(500, "boom"))

        events: list[CanonicalEvent] = []
        async for event in gw._handle_stream_send(invocation, exec_record):
            events.append(event)

        # Exactly an error event followed by message_stop.
        assert len(events) == 2, f"expected error + message_stop, got {events}"
        assert events[0].type == "error"
        assert events[0].error.code == "BACKEND_ERROR"
        assert "500" in events[0].error.message
        assert events[1].type == "message_stop"

        # Critical: the exec record must reflect failure, not success.
        assert exec_record.state == ExecutionState.FAILED
        assert exec_record.response_outcome == "error"

    @pytest.mark.asyncio
    async def test_gateway_handle_stream_upstream_429_maps_to_rate_limited_code(self):
        """A streaming upstream 429 must classify as BACKEND_RATE_LIMITED,
        not the generic BACKEND_ERROR every non-2xx status used to collapse
        to — see the sibling 500 test above, which stays BACKEND_ERROR
        since that's the correct classification for an unclassified 5xx.
        """

        class FakeErrorStream:
            def __init__(self, status_code: int, body: str) -> None:
                self.status_code = status_code
                self._body = body

            async def raw_lines(self) -> AsyncIterator[str]:
                for line in self._body.splitlines():
                    yield line

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *args: object) -> None:
                return None

        class FakeErrorTransport:
            def __init__(self, status_code: int, body: str) -> None:
                self._status_code = status_code
                self._body = body

            @asynccontextmanager
            async def stream(
                self, request: PreparedUpstreamRequest
            ) -> AsyncIterator[FakeErrorStream]:
                yield FakeErrorStream(self._status_code, self._body)

            async def close(self) -> None:
                return None

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
                        base_url="http://127.0.0.1:0",
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )

        route = next(iter(config.routes.values()))
        codec = get_codec(route.upstream.wire_protocol)
        plan = build_invocation_plan(
            tools=[],
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=route.tool_mode,
            model_profile=None,
            repair_policy=None,
        )

        canonical = CanonicalRequest(
            model=CanonicalModelReference(requested_name="test-model"),
            messages=[
                CanonicalMessage(
                    role="user", content=[CanonicalTextBlock(text="Hello")]
                )
            ],
            generation=CanonicalGenerationOptions(max_output_tokens=100),
            tool_choice=CanonicalToolChoice.auto(),
        )

        exec_record = InteropRequestExecution()
        invocation = ResolvedInvocation(
            request_context=RequestContext(),
            original_request=canonical,
            reconciled_request=canonical,
            route=route,
            backend_metadata=None,
            model_profile=None,
            repair_policy=None,
            invocation_plan=plan,
            codec=codec,
            compatibility_key=None,
            evidence_record=None,
            repair_budget=None,
            execution_record=exec_record,
        )

        gw = Gateway(config, transport=FakeErrorTransport(429, "rate limited"))

        events: list[CanonicalEvent] = []
        async for event in gw._handle_stream_send(invocation, exec_record):
            events.append(event)

        assert len(events) == 2
        assert events[0].type == "error"
        assert events[0].error.code == "BACKEND_RATE_LIMITED"
        assert events[1].type == "message_stop"


class TestStreamingArgumentIdentity:
    """Verify argument identity across streaming.

    The most important streaming invariant is:
        arguments observed upstream
            == arguments accumulated canonically
            == arguments validated/repaired
            == arguments emitted to the client

    The existing test_streaming_equivalence tests verify this at the ASGI
    level.  These tests verify it at the unit level for edge cases.
    """

    def test_escaped_braces_json_roundtrip(self):
        """Arguments with escaped braces survive JSON roundtrip."""
        from agent_interop.streaming.coordinator import PendingToolCallAccumulator, ToolStreamKey

        acc = PendingToolCallAccumulator()
        key = ToolStreamKey(0, 0)
        acc.start_call(key, "tc_001")
        acc.feed_name(key, "write_file")
        # Simulate fragmented argument delivery with escaped braces
        acc.feed_arguments(key, '{"content": "{\\"key\\": \\"value\\"}", ')
        acc.feed_arguments(key, '"path": "/tmp/test.json"}')

        completed = acc.complete_call(key)
        assert completed is not None
        assert completed.assembled_name == "write_file"
        # The accumulated arguments must be valid JSON
        parsed = json.loads(completed.assembled_arguments)
        assert parsed["content"] == '{"key": "value"}'
        assert parsed["path"] == "/tmp/test.json"

    def test_unicode_arguments_roundtrip(self):
        """Arguments with Unicode characters survive accumulation."""
        from agent_interop.streaming.coordinator import PendingToolCallAccumulator, ToolStreamKey

        acc = PendingToolCallAccumulator()
        key = ToolStreamKey(0, 0)
        acc.start_call(key, "tc_002")
        acc.feed_name(key, "write_file")
        acc.feed_arguments(key, '{"path": "/tmp/日本語.txt", "content": "Hello 世界 🌍"}')

        completed = acc.complete_call(key)
        assert completed is not None
        parsed = json.loads(completed.assembled_arguments)
        assert parsed["path"] == "/tmp/日本語.txt"
        assert "🌍" in parsed["content"]

    def test_parallel_calls_preserve_identity(self):
        """Multiple parallel calls each preserve their own arguments."""
        from agent_interop.streaming.coordinator import PendingToolCallAccumulator, ToolStreamKey

        acc = PendingToolCallAccumulator()

        # Start two parallel calls
        key0 = ToolStreamKey(0, 0)
        key1 = ToolStreamKey(0, 1)
        acc.start_call(key0, "tc_a")
        acc.start_call(key1, "tc_b")

        # Interleave argument fragments
        acc.feed_name(key0, "read_file")
        acc.feed_arguments(key0, '{"path": "/a.txt"}')
        acc.feed_name(key1, "write_file")
        acc.feed_arguments(key1, '{"path": "/b.txt", "content": "data"}')

        # Complete and verify each call independently
        call_a = acc.complete_call(key0)
        call_b = acc.complete_call(key1)

        assert call_a is not None
        assert call_b is not None

        args_a = json.loads(call_a.assembled_arguments)
        args_b = json.loads(call_b.assembled_arguments)

        assert args_a == {"path": "/a.txt"}
        assert args_b == {"path": "/b.txt", "content": "data"}

    def test_late_id_fragment(self):
        """Tool call ID arriving after name/arguments is preserved."""
        from agent_interop.streaming.coordinator import PendingToolCallAccumulator, ToolStreamKey

        acc = PendingToolCallAccumulator()
        key = ToolStreamKey(0, 0)
        acc.start_call(key, None)  # No ID yet
        acc.feed_name(key, "read_file")
        acc.feed_arguments(key, '{"path": "/x.txt"}')

        # ID arrives late
        call = acc._pending[key]
        call.call_id = "tc_late_001"

        completed = acc.complete_call(key)
        assert completed is not None
        assert completed.call_id == "tc_late_001"
