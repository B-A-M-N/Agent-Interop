"""Tests for the six HTTP-server-boundary bugs fixed in server/app.py.

Covers:
  1. Malformed/oversize body -> protocol-encoded error (not generic Starlette 400)
  2. RequestContext.from_headers() is genuinely used (honors x-session-id)
  3. max_request_bytes enforced at ingress
  4. Unknown auth mode fails closed (middleware + config validation)
  5. count_tokens never performs real inference
  6. Streaming error encoding is protocol-correct
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from asgi_lifespan import LifespanManager

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    ProtocolKind,
)
from agent_interop.config import (
    EvidenceConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    validate_config,
)
from agent_interop.context import RequestContext
from agent_interop.gateway import Gateway
from agent_interop.server.app import create_app
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamResponse

# ─── Helpers ────────────────────────────────────────────────────────────────


def _parse_sse(text: str) -> list[dict[str, Any]]:
    """Parse an SSE string into a list of (event, data) dicts."""
    out: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip("\r")
        if line == "":
            if event_name is not None or data_lines:
                out.append({"event": event_name, "data": "\n".join(data_lines)})
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        elif line.startswith(":"):
            continue
    return out


def _make_config(**overrides: Any) -> InteropServerConfig:
    """Build a minimal InteropServerConfig, with optional overrides."""
    base = {
        "host": "127.0.0.1",
        "port": 0,
        "log_level": "error",
        "probe_on_startup": False,
        "routes": {
            "test-route": ModelRoute(
                id="test-route",
                client_model_aliases=["test-model"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://127.0.0.1:11434",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    }
    base.update(overrides)
    return InteropServerConfig(**base)


# ─── Bug 1: Malformed/oversize body -> protocol-encoded error ──────────────


class TestMalformedBodyProtocolError:
    @pytest.mark.asyncio
    async def test_non_streaming_malformed_json_is_protocol_error(self) -> None:
        """Malformed JSON on a non-streaming request must NOT be a generic 400."""
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    content=b"{not valid json",
                    headers={"Content-Type": "application/json"},
                )
        # Must be a clean JSON response, not a Starlette 422/400 HTML error.
        assert resp.status_code in (400, 422)
        body = resp.json()
        # OpenAI Chat error shape: {"error": {"message": ..., "code": ...}}
        assert "error" in body
        assert body["error"]["message"]
        assert "Invalid JSON" in body["error"]["message"]

    @pytest.mark.asyncio
    async def test_count_tokens_malformed_json_is_protocol_error(self) -> None:
        """Malformed JSON on /v1/messages/count_tokens must NOT be a raw 500.

        count_tokens is the one body-parsing endpoint whose
        ``_read_and_parse_body`` call was not wrapped in ``try/except
        BodyParseError``; a malformed body would raise an uncaught
        BodyParseError and produce a generic unstructured 500 instead of the
        protocol-correct Anthropic error every other endpoint returns.
        """
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/messages/count_tokens",
                    content=b"{not valid json",
                    headers={"Content-Type": "application/json"},
                )
        # Must be a clean JSON response, not an uncaught 500.
        assert resp.status_code == 400
        body = resp.json()
        # Anthropic error shape: {"type": "error", "error": {"type": ..., "message": ...}}
        assert body.get("type") == "error"
        assert "error" in body
        assert "Invalid JSON" in body["error"]["message"]


class TestNonObjectJsonBodyProtocolError:
    """A syntactically valid JSON body that isn't a top-level object (an
    array, string, number, bool, or null) used to reach a bare
    ``body.get(...)`` call downstream and crash with an unhandled
    ``AttributeError`` -> generic 500, instead of the protocol-correct 4xx
    every other malformed-body case already gets."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw_body", [b"[]", b'"just a string"', b"3", b"null", b"true"])
    async def test_openai_chat_non_object_body_is_protocol_error(self, raw_body: bytes) -> None:
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    content=raw_body,
                    headers={"Content-Type": "application/json"},
                )
        assert resp.status_code in (400, 422)
        body = resp.json()
        assert "error" in body
        assert body["error"]["message"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("raw_body", [b"[]", b'"just a string"', b"3"])
    async def test_anthropic_messages_non_object_body_is_protocol_error(self, raw_body: bytes) -> None:
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/messages",
                    content=raw_body,
                    headers={"Content-Type": "application/json"},
                )
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("type") == "error"
        assert "error" in body

    @pytest.mark.asyncio
    async def test_count_tokens_non_object_body_is_protocol_error(self) -> None:
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/messages/count_tokens",
                    content=b"[]",
                    headers={"Content-Type": "application/json"},
                )
        assert resp.status_code == 400
        body = resp.json()
        assert body.get("type") == "error"


class TestManagedLaunchClaudeAlias:
    """create_app_from_env() is what backs the managed `interop run claude`
    launcher (launcher.py -> ManagedGateway.start_gateway), and
    ClaudeCodeIntegration.build_launch() sets CLAUDE_MODEL to
    "claude-interop-{route}" unconditionally (see
    agents/claude_code.py) — the gateway side must register that exact
    alias, or Claude Code's requests for it 404 against the route."""

    @pytest.mark.asyncio
    async def test_client_model_aliases_include_claude_prefixed_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from agent_interop.server.app import create_app_from_env

        # A model name other than the "qwen3-coder" default, to prove the
        # alias is derived from whatever model was actually configured.
        monkeypatch.setenv("INTEROP_MODEL", "deepseek-v4")
        monkeypatch.setenv("INTEROP_BACKEND_URL", "http://127.0.0.1:11434")
        monkeypatch.setenv("INTEROP_BACKEND_TYPE", "ollama")
        monkeypatch.setenv("INTEROP_PORT", "0")
        monkeypatch.delenv("INTEROP_SESSION_CREDENTIAL", raising=False)

        app = create_app_from_env()
        async with LifespanManager(app):
            gw = app.state.gateway
            route = gw.config.routes[gw.config.default_route_id]

            assert "deepseek-v4" in route.client_model_aliases
            assert "claude-interop-deepseek-v4" in route.client_model_aliases


# ─── Bug 2: RequestContext.from_headers is genuinely used ──────────────────


class TestRequestContextFromHeaders:
    @pytest.mark.asyncio
    async def test_x_session_id_is_honored(self) -> None:
        """A request with x-session-id should have that session ID honored.

        We verify by spying on RequestContext.from_headers to confirm it is
        called and that the x-session-id header is present in the headers it
        receives.
        """
        from unittest.mock import patch

        from agent_interop.server import app as app_module

        seen: list[dict[str, str]] = []
        original = app_module.RequestContext.from_headers

        def spy(headers: dict[str, str], **kwargs: Any) -> Any:
            seen.append(dict(headers))
            return original(headers, **kwargs)

        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            with patch.object(
                app_module.RequestContext, "from_headers", side_effect=spy
            ):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 10,
                        },
                        headers={"x-session-id": "sess-custom-123"},
                    )

        assert seen, "RequestContext.from_headers was never called"
        # The last call should carry our x-session-id header.
        assert seen[-1].get("x-session-id") == "sess-custom-123"

    def test_no_session_header_leaves_session_id_empty(self) -> None:
        """MVP-14: without a session header, session_id must stay empty —
        NOT a freshly generated UUID. A synthesized ID would make every
        stateless request create a new one-shot entry in the bounded
        session store, and loop detection can never fire for a session
        that only ever sees one request anyway."""
        from agent_interop.context import RequestContext

        ctx = RequestContext.from_headers({})
        assert ctx.session_id == ""

        # request_id, in contrast, IS still synthesized when absent — it
        # identifies THIS request, not a durable cross-request session.
        assert ctx.request_id != ""


# ─── Bug 3: max_request_bytes enforced at ingress ──────────────────────────


class TestMaxRequestBytesEnforcement:
    @pytest.mark.asyncio
    async def test_oversize_body_rejected_cleanly(self) -> None:
        """A request over max_request_bytes is rejected cleanly, not parsed."""
        config = _make_config(max_request_bytes=100)
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [
                            {
                                "role": "user",
                                "content": (
                                    "Hello world this is a longer message to "
                                    "exceed one hundred bytes easily"
                                ),
                            }
                        ],
                        "max_tokens": 100,
                    },
                )
        # Clean rejection (413), not an unbounded parse / 500.
        assert resp.status_code == 413
        body = resp.json()
        assert "error" in body
        assert "exceeds" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_declared_content_length_over_limit_rejected_before_reading_body(self) -> None:
        """MVP-05: a declared Content-Length over the limit must be rejected
        BEFORE the body is read at all — not after buffering it in full."""
        from agent_interop.server import app as app_module

        config = _make_config(max_request_bytes=100)
        app = create_app(config=config)

        stream_was_consumed = False
        original = app_module._read_and_parse_body

        async def spying_read_and_parse_body(request, cfg):
            nonlocal stream_was_consumed
            # Wrap request.stream() to detect whether any chunk was pulled.
            original_stream = request.stream

            def spying_stream():
                nonlocal stream_was_consumed
                stream_was_consumed = True
                return original_stream()

            request.stream = spying_stream
            return await original(request, cfg)

        app_module._read_and_parse_body = spying_read_and_parse_body
        try:
            async with LifespanManager(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    oversized_content = "x" * 1000
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": oversized_content}],
                            "max_tokens": 100,
                        },
                    )
        finally:
            app_module._read_and_parse_body = original

        assert resp.status_code == 413
        assert stream_was_consumed is False, (
            "declared Content-Length over the limit must short-circuit "
            "before touching request.stream()"
        )


# ─── Bug 4: Unknown auth mode fails closed ──────────────────────────────────


class TestUnknownAuthModeFailsClosed:
    @pytest.mark.asyncio
    async def test_startup_rejects_unknown_mode(self) -> None:
        """A server with a bogus ingress_auth mode fails closed.

        REVISION #4: Gateway.__init__ is now the construction boundary and
        rejects this immediately — stronger than the previous behavior of
        constructing successfully and only failing at startup().
        """
        from agent_interop.gateway import Gateway

        config = _make_config(ingress_auth={"mode": "totally_bogus"})
        with pytest.raises(ValueError, match="Invalid InteropServerConfig"):
            Gateway(config)

        # allow_invalid_config=True is the test-only escape hatch — even
        # then, startup() itself still fails closed (defense in depth).
        gw = Gateway(config, allow_invalid_config=True)
        with pytest.raises(RuntimeError, match="Invalid gateway configuration"):
            await gw.startup()

    def test_validate_config_rejects_unknown_mode(self) -> None:
        """validate_config rejects an unknown ingress_auth.mode at startup."""
        config = _make_config(ingress_auth={"mode": "totally_bogus"})
        issues = validate_config(config)
        assert any("totally_bogus" in issue for issue in issues), (
            f"expected unknown-mode issue, got {issues}"
        )

    def test_validate_config_accepts_known_mode(self) -> None:
        """validate_config must not false-positive on a valid ingress_auth.mode."""
        config = _make_config(
            ingress_auth={"mode": "session_token", "token": "test-token-123"}
        )
        issues = validate_config(config)
        assert not any("ingress_auth.mode" in issue for issue in issues), (
            f"valid session_token mode flagged as invalid: {issues}"
        )


# ─── Bug 5: count_tokens never performs real inference ─────────────────────


class TestCountTokensNoRealInference:
    @pytest.mark.asyncio
    async def test_count_tokens_uses_estimate_not_upstream(self) -> None:
        """count_tokens must never send a real completion request upstream.

        We inject a spy transport that flags if send() is called, and confirm the
        response comes from the estimate path with zero real completion calls.
        """
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            # Force lazy transport creation, then replace it with a spy.
            gw = app.state.gateway
            real_transport = gw.transport  # builds the default transport
            calls: list[Any] = []

            class SpyTransport:
                def __init__(self, real: Any) -> None:
                    self._real = real

                async def send(self, request: Any) -> Any:
                    calls.append(request)
                    return await self._real.send(request)

                async def close(self) -> None:
                    await self._real.close()

            gw._transport = SpyTransport(real_transport)

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/messages/count_tokens",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "Hello"}],
                        "max_tokens": 10,
                    },
                )

            assert resp.status_code == 200
            body = resp.json()
            assert body["confidence"] == "estimated"
            assert body["input_tokens"] > 0
            # Zero real completion requests sent upstream.
            assert calls == [], f"expected no upstream calls, got {calls}"


# ─── Bug 6: Streaming error encoding is protocol-correct ───────────────────


class TestStreamingErrorEncoding:
    @pytest.mark.asyncio
    async def test_malformed_json_streaming_gives_protocol_error_frame(
        self,
    ) -> None:
        """Malformed JSON on a streaming request yields protocol-correct SSE.

        We drive a streaming request whose body is malformed but carries the
        ``stream: true`` marker; the endpoint must return SSE containing the
        protocol-correct error frame shape (Bug 6 Site A), not a bare generic
        data: blob.
        """
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                # Malformed JSON that still advertises stream intent.
                resp = await client.post(
                    "/v1/messages",
                    content=b'{"stream": true, ',
                    headers={"Content-Type": "application/json"},
                )
        # The response should be an SSE stream with a protocol error frame.
        assert "text/event-stream" in resp.headers.get("content-type", "")
        text = resp.text
        frames = _parse_sse(text)
        assert frames, f"expected SSE frames, got: {text!r}"
        # Anthropic protocol: an 'error' event must be present.
        events = [f.get("event") for f in frames]
        assert "error" in events, f"expected 'error' event, got {events}"
        # The error frame must carry the structured error payload.
        error_frames = [f for f in frames if f.get("event") == "error"]
        assert error_frames
        data = json.loads(error_frames[0]["data"])
        assert data["type"] == "error"
        assert data["error"]["message"]

    @pytest.mark.asyncio
    async def test_mid_stream_failure_uses_encoder_terminal(self) -> None:
        """A mid-stream failure must route through the encoder's failure terminal.

        We force a streaming request to a route whose upstream will fail, and
        assert the SSE output contains the protocol-correct error frame and a
        correct failure terminal sequence (not a bare data: blob).
        """
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/messages",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10,
                        "stream": True,
                    },
                )
        assert "text/event-stream" in resp.headers.get("content-type", "")
        text = resp.text
        frames = _parse_sse(text)
        assert frames, f"expected SSE frames, got: {text!r}"
        events = [f.get("event") for f in frames]
        # Must contain an error event (the upstream is unreachable -> failure).
        assert "error" in events, f"expected 'error' event, got {events}"
        # Must contain the failure terminal (message_stop), not a bare [DONE].
        assert "message_stop" in events, (
            f"expected failure terminal message_stop, got {events}"
        )


# ─── Item 0: UnboundLocalError fix in _handle_stream ────────────────────────


class TestStreamEncoderUnboundLocalError:
    @pytest.mark.asyncio
    async def test_create_stream_encoder_raising_yields_clean_error_frame(self) -> None:
        """If create_stream_encoder itself raises (before ``encoder`` is ever
        assigned), the except block must still produce a clean protocol error
        response — never an UnboundLocalError that crashes the generator."""
        from agent_interop.protocols.registry import get_adapter
        from agent_interop.server import app as app_module

        # Adapter whose stream-encoder factory raises ONLY on the first call
        # (the real one). The fallback encoder built in the except block must
        # succeed, so subsequent calls delegate to the real adapter.
        class _BoomAdapter:
            def __init__(self, real: Any) -> None:
                self._real = real
                self._calls = 0

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

            def create_stream_encoder(self, *args: Any, **kwargs: Any) -> Any:
                self._calls += 1
                if self._calls == 1:
                    raise RuntimeError("encoder factory boom")
                return self._real.create_stream_encoder(*args, **kwargs)

        real_adapter = get_adapter(ProtocolKind.OPENAI_CHAT)

        def _boom_get_adapter(protocol: Any) -> Any:
            return _BoomAdapter(real_adapter)

        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            with patch.object(
                app_module, "get_adapter", side_effect=_boom_get_adapter
            ):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "max_tokens": 10,
                            "stream": True,
                        },
                    )
        # Must be a clean SSE error stream, not an unhandled 500/UnboundLocalError.
        assert "text/event-stream" in resp.headers.get("content-type", "")
        frames = _parse_sse(resp.text)
        assert frames, f"expected SSE frames, got: {resp.text!r}"
        # OpenAI Chat streams carry payloads on unnamed data: lines (no event:
        # name). The first frame must be the structured error payload.
        error_frames = [f for f in frames if f.get("data")]
        assert error_frames, f"expected at least one data frame, got {frames}"
        payload = json.loads(error_frames[0]["data"])
        assert "error" in payload, f"expected error payload, got {payload}"
        assert payload["error"]["code"] == "INTERNAL_ERROR"
        assert "encoder factory boom" in payload["error"]["message"]


# ─── Item 1: client credential passthrough headers ──────────────────────────


class TestCredentialPassthroughHeaders:
    @pytest.mark.asyncio
    async def test_passthrough_header_reaches_upstream(self) -> None:
        """A header on the client's forwardable allowlist must be forwarded to
        the upstream transport when the route auth mode is PASSTHROUGH."""
        from agent_interop.auth import HEADER_ALLOWLIST_PASSTHROUGH

        # Pick a real allowlist header (x-api-key) so the test is non-tautological.
        assert "x-api-key" in HEADER_ALLOWLIST_PASSTHROUGH

        route = ModelRoute(
            id="test-route",
            client_model_aliases=["test-model"],
            upstream_model="fake-model",
            upstream=UpstreamConfig(
                kind=UpstreamKind.OPENAI_COMPATIBLE,
                base_url="http://127.0.0.1:11434",
                wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                timeout_seconds=30.0,
                auth={"mode": "passthrough"},
            ),
            tool_mode=ToolMode.AUTO,
            translation_mode=TranslationMode.CANONICAL,
        )
        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            log_level="error",
            probe_on_startup=False,
            routes={"test-route": route},
        )
        app = create_app(config=config)
        async with LifespanManager(app):
            gw = app.state.gateway
            spy = _UpstreamSpyTransport()
            gw._transport = spy

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10,
                    },
                    headers={"x-api-key": "client-key-123"},
                )

        assert spy.send_calls, "expected an upstream send() call"
        upstream_headers = spy.send_calls[0].headers
        assert upstream_headers.get("x-api-key") == "client-key-123", (
            f"expected x-api-key to be forwarded, got headers: {upstream_headers}"
        )


# ─── Item 2: api_key_env consolidation ───────────────────────────────────────


class TestApiKeyEnvConsolidation:
    def test_api_key_env_adds_authorization_for_real_request(self, monkeypatch) -> None:
        """A route with only the legacy api_key_env field set must produce an
        Authorization header for ordinary inference requests — not just probing."""
        monkeypatch.setenv("INTEROP_LEGACY_PROBE_KEY", "legacy-secret-xyz")
        route = ModelRoute(
            id="test-route",
            client_model_aliases=["test-model"],
            upstream_model="fake-model",
            upstream=UpstreamConfig(
                kind=UpstreamKind.OPENAI_COMPATIBLE,
                base_url="http://127.0.0.1:11434",
                wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                api_key_env="INTEROP_LEGACY_PROBE_KEY",
            ),
            tool_mode=ToolMode.AUTO,
            translation_mode=TranslationMode.CANONICAL,
        )
        config = InteropServerConfig(
            probe_on_startup=False, routes={"test-route": route}
        )
        gw = Gateway(config)
        headers = gw._build_upstream_headers(route, codec_headers={})
        assert headers.get("authorization") == "Bearer legacy-secret-xyz", (
            f"expected Bearer <legacy-secret-xyz>, got headers: {headers}"
        )

    def test_invalid_upstream_auth_mode_fails_validation(self) -> None:
        """An invalid upstream auth mode string must fail config validation at
        load time rather than being silently downgraded to NONE."""
        route = ModelRoute(
            id="test-route",
            client_model_aliases=["test-model"],
            upstream_model="fake-model",
            upstream=UpstreamConfig(
                kind=UpstreamKind.OPENAI_COMPATIBLE,
                base_url="http://127.0.0.1:11434",
                wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                auth={"mode": "totally_bogus_mode"},
            ),
            tool_mode=ToolMode.AUTO,
            translation_mode=TranslationMode.CANONICAL,
        )
        config = InteropServerConfig(
            probe_on_startup=False, routes={"test-route": route}
        )
        issues = validate_config(config)
        assert any("totally_bogus_mode" in issue for issue in issues), (
            f"expected invalid-mode issue, got: {issues}"
        )


# ─── Item 4: lifespan cleanup on error ──────────────────────────────────────


class TestLifespanCleanup:
    @pytest.mark.asyncio
    async def test_close_runs_even_when_startup_raises(self) -> None:
        """If Gateway.startup() raises, the lifespan must still call close() so
        the transport (and any evidence store) is never leaked."""
        from agent_interop.server import app as app_module

        closed: list[bool] = []

        class _FailingGateway:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            async def startup(self) -> None:
                raise RuntimeError("startup boom")

            async def close(self) -> None:
                closed.append(True)

        config = _make_config()
        with patch.object(app_module, "Gateway", _FailingGateway):
            app = create_app(config=config)
            try:
                async with LifespanManager(app):
                    pass
            except Exception:
                pass
        assert closed == [True], (
            f"expected close() to be called once despite startup failure, got {closed}"
        )


# ─── Item 5: evidence store opt-in ───────────────────────────────────────────


class TestEvidenceStoreOptIn:
    @pytest.mark.asyncio
    async def test_evidence_store_used_when_enabled(self, tmp_path: Path) -> None:
        """With evidence.enabled=True, a tool-calling request must write a
        record to the store."""
        db_path = str(tmp_path / "evidence.db")
        config = _make_config(evidence=EvidenceConfig(enabled=True, db_path=db_path))
        app = create_app(config=config)
        async with LifespanManager(app):
            gw = app.state.gateway
            assert gw._evidence_store is not None
            gw._transport = _tool_call_transport()
            request = _make_tool_request()
            ctx = RequestContext(client_id="claude_code")
            resp = await gw.handle_request(request, ctx)
            assert resp.error is None

            results = gw._evidence_store.query_results()
            assert len(results) >= 1, (
                "expected at least one evidence record after a tool-calling request"
            )

    @pytest.mark.asyncio
    async def test_no_evidence_store_when_disabled(self, tmp_path: Path) -> None:
        """Without evidence config, no store is ever created and the gateway's
        store attribute stays None (byte-for-byte identical to before)."""
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            gw = app.state.gateway
            assert gw._evidence_store is None


# ─── Item 3: managed-launcher authenticated readiness ───────────────────────


class TestManagedLauncherAuthenticatedReadiness:
    def test_readiness_probe_authenticates_and_gateway_starts(self) -> None:
        """The managed launcher's liveness probe must send the session
        credential, or a healthy gateway process times out because its own
        probe gets rejected with 401. Exercises the REAL start_gateway()
        code path.

        Uses /v1/health/live (process liveness), not /v1/health (backend
        readiness) — the backend is deliberately unreachable here to prove
        that startup does not block on it, so /v1/health would correctly
        report 503 regardless of auth."""
        from agent_interop.launcher import ManagedGateway

        port = _free_port()
        # Point the backend at an unreachable host so probing fails fast
        # (probing is non-blocking — it never prevents startup).
        gw = ManagedGateway(
            model="test-model", port=port, ollama_url="http://127.0.0.1:1"
        )
        session_credential = gw._session_credential
        try:
            # start_gateway() now sends the credential in its liveness poll.
            url = gw.start_gateway(timeout=20.0)
            assert url == f"http://127.0.0.1:{port}"

            # Authenticated liveness succeeds (the launcher's own probe passes).
            r = httpx.get(
                f"{url}/v1/health/live",
                headers={"Authorization": f"Bearer {session_credential}"},
                timeout=5.0,
            )
            assert r.status_code == 200, (
                f"authenticated liveness failed: {r.status_code}"
            )

            # Without the credential, health must be rejected (auth enforced),
            # proving the probe genuinely needs the credential to succeed.
            r2 = httpx.get(f"{url}/v1/health/live", timeout=2.0)
            assert r2.status_code == 401, (
                f"unauthenticated health should be rejected, got {r2.status_code}"
            )

            # /v1/health now reflects backend READINESS, not liveness: this
            # gateway's backend is deliberately unreachable, so it must
            # honestly report not-ready rather than an unconditional "ok".
            r3 = httpx.get(
                f"{url}/v1/health",
                headers={"Authorization": f"Bearer {session_credential}"},
                timeout=5.0,
            )
            assert r3.status_code == 503
            body = r3.json()
            assert body["ready"] is False
        finally:
            gw.cleanup()


# ─── /v1/health no longer reports fabricated placeholder capability data ───


class TestHealthEndpointNoPlaceholders:
    @pytest.mark.asyncio
    async def test_health_response_omits_placeholder_fields(self) -> None:
        """/v1/health used to always report profile=None, level=0,
        level_description="", supports=["multi_route"] — fixed,
        unconditional values with zero relationship to the actual
        configuration. Real capability data belongs at /v1/capabilities."""
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/v1/health")
        body = resp.json()
        for key in ("profile", "level", "level_description", "supports"):
            assert key not in body, f"placeholder key '{key}' should have been removed"
        for key in ("status", "ready", "version", "model", "routes"):
            assert key in body, f"real key '{key}' missing"


# ─── Capabilities endpoint wires in the capability model ────────────────────


class TestCapabilitiesEndpoint:
    @pytest.mark.asyncio
    async def test_capability_model_present_and_structured(self) -> None:
        """The /v1/capabilities response must include a ``capability_model`` key
        with a per-route EffectiveCapabilities structure and a compatibility
        decision.

        Top-level "level"/"description"/"supports" were removed: they were
        permanently-hardcoded placeholders (level=0, supports=["multi_route"])
        regardless of actual configuration, redundant with (and inconsistent
        against) the real per-route data capability_model already carries.
        """
        from agent_interop.capabilities import CapabilityState

        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/v1/capabilities")
        assert resp.status_code == 200
        body = resp.json()

        for key in ("model", "routes"):
            assert key in body, f"existing key '{key}' missing"
        for key in ("level", "description", "supports"):
            assert key not in body, f"placeholder key '{key}' should have been removed"

        # MVP: the endpoint must be self-describing as declared, not verified —
        # every entry here comes from static profile/codec metadata, never a
        # live conformance run.
        assert body["source"] == "declared_profile_metadata"
        assert body["verified"] is False

        assert "capability_model" in body
        cm = body["capability_model"]
        assert "test-route" in cm
        entry = cm["test-route"]

        # Per-route EffectiveCapabilities structure.
        assert "tool_level" in entry
        assert "capabilities" in entry
        assert "compatibility" in entry
        compat = entry["compatibility"]
        assert "status" in compat

        # Cross-check against the Part 1 fix: every DECLARED-state entry must
        # report is_available() == False. A metadata-derived ("declared")
        # capability is an unverified claim and must not be treatable as
        # available — this is exactly the bug fixed in CapabilityState.
        for name, cap in entry["capabilities"].items():
            state = cap["state"]
            assert CapabilityState(state).is_available() is False or state != "declared", (
                f"capability '{name}' is 'declared' but counts as available; "
                "the Part 1 is_available() fix must hold end-to-end"
            )

    @pytest.mark.asyncio
    async def test_capability_model_honours_declared_not_verified(self) -> None:
        """All static-metadata-derived states must be DECLARED, never VERIFIED
        or PROBED — the endpoint must not over-claim conformance it has not
        tested."""
        from agent_interop.capabilities import CapabilityState

        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.get("/v1/capabilities")
        assert resp.status_code == 200
        cm = resp.json()["capability_model"]
        for route_id, entry in cm.items():
            for name, cap in entry["capabilities"].items():
                state = CapabilityState(cap["state"])
                assert state in (CapabilityState.DECLARED, CapabilityState.UNSUPPORTED), (
                    f"route '{route_id}' capability '{name}' has state "
                    f"{state.value!r}; static metadata may only produce "
                    "DECLARED or UNSUPPORTED"
                )


class TestCapabilitiesObservedBlock:
    """REVISION #5/#6: /v1/capabilities carries a sibling "observed" block
    of REAL evidence-store records (never replacing "declared_profile_metadata"),
    reported as a LIST per route — one entry per distinct compatibility key,
    never collapsed — each labeled with capability_source/battery_version."""

    @pytest.mark.asyncio
    async def test_observed_empty_when_evidence_disabled(self) -> None:
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/capabilities")
        body = resp.json()
        assert "observed" in body
        assert body["observed"] == {}

    @pytest.mark.asyncio
    async def test_observed_lists_every_distinct_key_for_the_route(self, tmp_path: Path) -> None:
        from agent_interop.replay.types import CompatibilityKey, CompatibilityResult
        from agent_interop.testing.levels import BATTERY_VERSION

        db_path = str(tmp_path / "evidence.db")
        config = _make_config(evidence=EvidenceConfig(enabled=True, db_path=db_path))
        app = create_app(config=config)
        async with LifespanManager(app):
            gw = app.state.gateway
            assert gw._evidence_store is not None
            key_a = CompatibilityKey(
                client_id="claude_code", model_id="fake-model", backend_kind="openai_compatible",
            )
            key_b = CompatibilityKey(
                client_id="codex", model_id="fake-model", backend_kind="openai_compatible",
            )
            gw._evidence_store.store_result(
                key_a, CompatibilityResult(
                    sample_count=5, tested_at="2026-01-01T00:00:00",
                    battery_version=BATTERY_VERSION,
                ),
            )
            gw._evidence_store.store_result(
                key_b, CompatibilityResult(
                    sample_count=3, tested_at="2026-01-02T00:00:00",
                    battery_version="stale-version-hash",
                ),
            )

            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.get("/v1/capabilities")
        assert resp.status_code == 200
        body = resp.json()
        observed_route = body["observed"]["test-route"]
        assert len(observed_route) == 2
        by_client = {e["client_id"]: e for e in observed_route}
        assert by_client["claude_code"]["capability_source"] == "observed"
        assert by_client["codex"]["capability_source"] == "stale"
        assert body["source"] == "declared_profile_metadata"
        assert body["verified"] is False


# ─── Helpers ─────────────────────────────────────────────────────────────────


class _UpstreamSpyTransport:
    """Captures upstream send() calls and returns a valid OpenAI Chat response."""

    def __init__(self) -> None:
        self.send_calls: list[PreparedUpstreamRequest] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        self.send_calls.append(request)
        body = {
            "id": "spy-chat",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        return UpstreamResponse(
            status_code=200, headers={}, body=json.dumps(body).encode("utf-8")
        )

    async def close(self) -> None:
        pass


def _tool_call_transport() -> Any:
    """Transport returning a single well-formed tool call."""
    body = {
        "id": "fake-chat-response",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tc_fake_001",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    class _T:
        async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
            return UpstreamResponse(
                status_code=200, headers={}, body=json.dumps(body).encode("utf-8")
            )

        async def close(self) -> None:
            pass

    return _T()


_READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)


def _make_tool_request() -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="test-model"),
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalTextBlock(text="Read the file")],
            )
        ],
        tools=[_READ_FILE_TOOL],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )


class _ConfigurableErrorTransport:
    """Transport whose response is configured per-test to drive gateway errors.

    Modes:
      - ``status`` >= 400 with a JSON ``{"error": ...}`` body → backend HTTP error.
      - ``status`` == 200 with a ``body`` that is not valid JSON → invalid upstream
        JSON error.
      - ``status`` == 200 with a tool call to an unregistered tool name → fully
        rejected tool batch error.
    """

    def __init__(self, *, status: int = 502, body: bytes | None = None) -> None:
        self.status = status
        if body is not None:
            self.body = body
        elif status >= 400:
            self.body = b'{"error": "upstream failure"}'
        else:
            self.body = b'not valid json'

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        return UpstreamResponse(
            status_code=self.status, headers={}, body=self.body,
        )

    async def close(self) -> None:
        pass


def _rejected_tool_batch_transport() -> Any:
    """Transport returning a tool call to an unregistered tool (full batch rejection)."""

    class _T:
        async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
            body = {
                "id": "fake-chat-response",
                "object": "chat.completion",
                "created": 0,
                "model": "fake-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "tc_bad_001",
                            "type": "function",
                            # Not in the registered tools list → whole batch rejected.
                            "function": {"name": "nonexistent_tool", "arguments": "{}"},
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            return UpstreamResponse(
                status_code=200, headers={}, body=json.dumps(body).encode("utf-8")
            )

        async def close(self) -> None:
            pass

    return _T()


def _free_port() -> int:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ─── Item 1 (P0.1): non-streaming errors must be non-200 protocol errors ─────


class TestNonStreamingErrorEncoding:
    """Non-streaming gateway errors must surface as non-200 HTTP responses with
    a protocol-shaped error body — never as a 200 success envelope.

    Before the fix, ``_handle_request`` called ``adapter.encode_response(result)``
    unconditionally, ignoring ``result.error``. These tests drive each error
    class through the real ASGI endpoint for all three client protocols.
    """

    @pytest.mark.asyncio
    async def test_backend_http_error_is_not_200(self) -> None:
        """An upstream HTTP error (502) must produce a non-200 protocol error."""
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            app.state.gateway._transport = _ConfigurableErrorTransport(status=502)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10,
                    },
                )
        assert resp.status_code != 200
        assert resp.status_code == 502
        body = resp.json()
        # OpenAI Chat error shape. A 502 from upstream is classified as
        # BACKEND_PROTOCOL_ERROR (not a generic BACKEND_ERROR) so clients
        # can distinguish it from other backend failure classes (auth,
        # rate limit, not-found, timeout) — see errors.classify_http_status.
        assert "error" in body
        assert body["error"]["message"]
        assert body["error"]["code"] == "BACKEND_PROTOCOL_ERROR"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "expected_code"),
        [
            (401, "BACKEND_AUTH_FAILED"),
            (403, "BACKEND_AUTH_FAILED"),
            (404, "MODEL_NOT_FOUND"),
            (429, "BACKEND_RATE_LIMITED"),
            (503, "BACKEND_UNAVAILABLE"),
            (504, "BACKEND_TIMEOUT"),
        ],
    )
    async def test_backend_http_error_code_reflects_status(
        self, status: int, expected_code: str
    ) -> None:
        """Distinct upstream HTTP statuses must map to distinct error codes.

        Before the fix, every non-2xx upstream response — 401, 404, 429,
        503, 504, all of it — was collapsed to a single generic
        "BACKEND_ERROR", so a client (or a human debugging a failed
        launch) couldn't tell an auth failure from a missing model from a
        rate limit without reading raw upstream text.
        """
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            app.state.gateway._transport = _ConfigurableErrorTransport(status=status)
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10,
                    },
                )
        assert resp.status_code != 200
        body = resp.json()
        assert body["error"]["code"] == expected_code

    @pytest.mark.asyncio
    async def test_invalid_upstream_json_is_not_200(self) -> None:
        """A 200 upstream response with a non-JSON body must be a protocol error."""
        config = _make_config()
        app = create_app(config=config)
        async with LifespanManager(app):
            app.state.gateway._transport = _ConfigurableErrorTransport(
                status=200, body=b"<html>not json</html>"
            )
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "test-model",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 10,
                    },
                )
        assert resp.status_code != 200
        body = resp.json()
        assert "error" in body
        assert body["error"]["code"] == "INVALID_UPSTREAM_OUTPUT"

    @pytest.mark.asyncio
    async def test_unsafe_history_is_not_200_all_protocols(self) -> None:
        """Unsafe history (orphan tool result) must be a non-200 protocol error
        for all three client protocols."""
        bodies = {
            "/v1/messages": {
                "model": "test-model",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "hi"},
                    # tool-role message referencing a never-called tool_use_id.
                    {"role": "tool", "tool_use_id": "call_never_made",
                     "content": "result text"},
                ],
            },
            "/v1/chat/completions": {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "call_never_made",
                     "content": "result text"},
                ],
                "max_tokens": 100,
            },
            "/v1/responses": {
                "model": "test-model",
                "input": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "tool_call_id": "call_never_made",
                     "content": "result text"},
                ],
            },
        }
        expected_shape = {
            "/v1/messages": ("type", "error"),            # Anthropic: {"type":"error","error":{...}}
            "/v1/chat/completions": ("error",),           # OpenAI Chat: {"error":{...}}
            "/v1/responses": ("type", "error"),           # OpenAI Responses: {"type":"error","error":{...}}
        }
        for endpoint, payload in bodies.items():
            config = _make_config()
            app = create_app(config=config)
            async with LifespanManager(app):
                # A normal transport; the error comes from history reconciliation,
                # not the upstream.
                app.state.gateway._transport = _UpstreamSpyTransport()
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(endpoint, json=payload)
            assert resp.status_code != 200, f"{endpoint} returned 200"
            assert resp.status_code == 422, (
                f"{endpoint} expected 422, got {resp.status_code}"
            )
            body = resp.json()
            for key in expected_shape[endpoint]:
                assert key in body, f"{endpoint} response missing key {key!r}: {body}"

    @pytest.mark.asyncio
    async def test_rejected_tool_batch_is_not_200_all_protocols(self) -> None:
        """A fully-rejected tool batch must be a non-200 protocol error for all
        three client protocols."""
        endpoint_tools = {
            "/v1/messages": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
            "/v1/chat/completions": [
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
            "/v1/responses": [
                {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                }
            ],
        }
        for endpoint, tools in endpoint_tools.items():
            base: dict[str, Any]
            if endpoint == "/v1/messages":
                base = {
                    "model": "test-model",
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": "do something"}],
                }
            elif endpoint == "/v1/chat/completions":
                base = {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "do something"}],
                    "max_tokens": 100,
                }
            else:
                base = {
                    "model": "test-model",
                    "input": [{"role": "user", "content": "do something"}],
                }
            base["tools"] = tools
            config = _make_config()
            app = create_app(config=config)
            async with LifespanManager(app):
                app.state.gateway._transport = _rejected_tool_batch_transport()
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    resp = await client.post(endpoint, json=base)
            assert resp.status_code != 200, f"{endpoint} returned 200"
            body = resp.json()
            # Each protocol nests the error under "error" (and Anthropic/Responses
            # also under "type":"error"). Assert the message + code are present.
            if endpoint == "/v1/messages":
                assert body.get("type") == "error"
                assert body["error"]["message"]
            elif endpoint == "/v1/chat/completions":
                assert "error" in body
                assert body["error"]["message"]
            else:
                assert body.get("type") == "error"
                assert body["error"]["message"]
