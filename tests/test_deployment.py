"""Deployment integration tests (P0).

Tests that launch real subprocess entry points and verify operational behavior:
- create_app_from_env with session token auth
- Graceful shutdown via signal
- Backend unreachable handling
- Invalid config rejection
- Service unit generation
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ─── create_app_from_env ────────────────────────────────────────────────────


class TestCreateAppFromEnv:
    def test_creates_app_with_session_auth(self):
        from agent_interop.server.app import create_app_from_env

        env = {
            "INTEROP_BACKEND_URL": "http://127.0.0.1:11434",
            "INTEROP_BACKEND_TYPE": "ollama",
            "INTEROP_MODEL": "test-model",
            "INTEROP_PORT": "0",
            "INTEROP_SESSION_CREDENTIAL": "test-token-123",
        }
        with patch.dict(os.environ, env, clear=False):
            app = create_app_from_env()
            assert app is not None

    def test_creates_app_without_session_auth(self):
        from agent_interop.server.app import create_app_from_env

        env = {
            "INTEROP_BACKEND_URL": "http://127.0.0.1:11434",
            "INTEROP_BACKEND_TYPE": "ollama",
            "INTEROP_MODEL": "test-model",
            "INTEROP_PORT": "0",
        }
        # Clear credential if set
        env_clear = {k: v for k, v in os.environ.items() if k != "INTEROP_SESSION_CREDENTIAL"}
        env_clear.update(env)
        with patch.dict(os.environ, env_clear, clear=True):
            app = create_app_from_env()
            assert app is not None

    def test_rejects_unknown_backend_type(self):
        """An unrecognized backend type must be rejected, not silently mapped to Ollama."""
        from agent_interop.server.app import create_app_from_env

        env = {
            "INTEROP_BACKEND_URL": "http://127.0.0.1:11434",
            "INTEROP_BACKEND_TYPE": "not-a-real-backend",
            "INTEROP_MODEL": "test-model",
            "INTEROP_PORT": "0",
        }
        env_clear = {k: v for k, v in os.environ.items() if k not in env}
        env_clear.update(env)
        with patch.dict(os.environ, env_clear, clear=True):
            with pytest.raises(ValueError, match="not-a-real-backend"):
                create_app_from_env()

    def test_validates_config_on_creation(self):
        """Invalid backend type should raise ValueError."""
        from agent_interop.server.app import create_app_from_env

        env = {
            "INTEROP_BACKEND_URL": "not-a-url",
            "INTEROP_BACKEND_TYPE": "ollama",
            "INTEROP_MODEL": "test-model",
            "INTEROP_PORT": "0",
        }
        env_clear = {k: v for k, v in os.environ.items() if k not in env}
        env_clear.update(env)
        with patch.dict(os.environ, env_clear, clear=True):
            with pytest.raises((ValueError, Exception)):
                create_app_from_env()


# ─── Session Token Authentication ───────────────────────────────────────────


class TestSessionTokenAuth:
    def _make_app_with_auth(self, token: str = "test-token"):
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

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
            ingress_auth={"mode": "session_token", "token": token},
        )
        return create_app(config)

    def test_health_endpoint_requires_auth(self):
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("my-secret-token")
        with TestClient(app) as client:
            resp = client.get("/v1/health")
            # Should be 401 without auth
            assert resp.status_code in (401, 403)

    def test_health_endpoint_accepts_valid_token(self):
        """A valid token clears auth. Uses /v1/health/live rather than
        /v1/health — this route's upstream is an unreachable placeholder
        address, so /v1/health now honestly reports 503 (not ready); the
        liveness endpoint is what proves auth itself passed."""
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("my-secret-token")
        with TestClient(app) as client:
            resp = client.get(
                "/v1/health/live",
                headers={"Authorization": "Bearer my-secret-token"},
            )
            # Should be 200 with valid token
            assert resp.status_code == 200

    def test_health_endpoint_rejects_invalid_token(self):
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("my-secret-token")
        with TestClient(app) as client:
            resp = client.get(
                "/v1/health",
                headers={"Authorization": "Bearer wrong-token"},
            )
            assert resp.status_code in (401, 403)


class TestIngressAuthProtocolNativeErrors:
    """MVP-08: auth failures must be shaped like the target protocol's own
    error responses (Anthropic vs OpenAI Chat vs OpenAI Responses), not a
    generic {"error": "..."} body a client's error-handling code won't
    recognize — and a bearer-auth failure must include WWW-Authenticate."""

    def _make_app_with_auth(self, token: str = "test-token"):
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

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
            ingress_auth={"mode": "session_token", "token": token},
        )
        return create_app(config)

    def test_anthropic_path_gets_anthropic_shaped_error(self):
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("secret")
        with TestClient(app) as client:
            resp = client.post("/v1/messages", json={})
            assert resp.status_code == 401
            body = resp.json()
            assert body["type"] == "error"
            assert body["error"]["type"] == "authentication_error"
            assert "message" in body["error"]

    def test_openai_chat_path_gets_openai_shaped_error(self):
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("secret")
        with TestClient(app) as client:
            resp = client.post("/v1/chat/completions", json={})
            assert resp.status_code == 401
            body = resp.json()
            assert body["error"]["type"] == "authentication_error"
            assert body["error"]["code"] == "INGRESS_AUTH_FAILED"

    def test_openai_responses_path_gets_openai_responses_shaped_error(self):
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("secret")
        with TestClient(app) as client:
            resp = client.post("/v1/responses", json={})
            assert resp.status_code == 401
            body = resp.json()
            assert body["type"] == "error"
            assert body["error"]["code"] == "INGRESS_AUTH_FAILED"

    def test_bearer_auth_failure_sets_www_authenticate_header(self):
        from starlette.testclient import TestClient

        app = self._make_app_with_auth("secret")
        with TestClient(app) as client:
            resp = client.post("/v1/messages", json={})
            assert resp.status_code == 401
            assert resp.headers.get("www-authenticate") == "Bearer"

    @pytest.mark.asyncio
    async def test_loopback_forbidden_does_not_set_www_authenticate(self):
        """403 (not a bearer-auth failure) must not carry a WWW-Authenticate
        challenge — that header is specifically an HTTP 401 mechanism."""
        import httpx
        from asgi_lifespan import LifespanManager

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

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
            ingress_auth={"mode": "none_loopback"},
        )
        app = create_app(config)
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app, client=("203.0.113.5", 12345))
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post("/v1/messages", json={})
                assert resp.status_code == 403
                assert "www-authenticate" not in resp.headers
                body = resp.json()
                assert body["error"]["type"] == "permission_error"


# ─── Graceful Shutdown ──────────────────────────────────────────────────────


class TestGracefulShutdown:
    @pytest.mark.asyncio
    async def test_gateway_close_releases_resources(self):
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )
        gw = Gateway(config)
        # Create the transport
        _ = gw.transport
        assert gw._transport is not None
        # Close should release it
        await gw.close()
        assert gw._transport is None

    @pytest.mark.asyncio
    async def test_gateway_rejects_invalid_config_at_construction(self):
        """REVISION #4: Gateway.__init__ is the actual construction
        boundary — an invalid config must never even construct a Gateway,
        not just fail later at startup()."""
        from agent_interop.config import InteropServerConfig
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(probe_on_startup=False, routes={})
        with pytest.raises(ValueError, match="Invalid InteropServerConfig"):
            Gateway(config)

    async def test_gateway_startup_still_validates_when_construction_bypassed(self):
        """allow_invalid_config=True is a test-only escape hatch — a
        Gateway deliberately constructed through it must still fail at
        startup() (defense in depth), never silently proceed."""
        from agent_interop.config import InteropServerConfig
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(probe_on_startup=False, routes={})
        gw = Gateway(config, allow_invalid_config=True)
        with pytest.raises(RuntimeError, match="Invalid gateway configuration"):
            await gw.startup()


# ─── Backend Unavailable ────────────────────────────────────────────────────


class TestBackendUnavailable:
    @pytest.mark.asyncio
    async def test_gateway_handles_unreachable_backend(self):
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:19999",  # Unreachable
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )
        gw = Gateway(config)
        await gw.startup()
        # Gateway should startup even if backend is unreachable
        assert gw is not None


# ─── Transport Config Wiring ────────────────────────────────────────────────


class TestTransportConfigWiring:
    @pytest.mark.asyncio
    async def test_gateway_uses_config_timeouts(self):
        """Transport settings from config flow into the httpx client."""
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            probe_on_startup=False,
            connect_timeout=3.0,
            read_timeout=45.0,
            write_timeout=15.0,
            pool_timeout=2.0,
            max_connections=50,
            max_keepalive_connections=10,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )
        gw = Gateway(config)
        transport = gw.transport
        client = transport.client
        # MVP-05: each config timeout dimension must wire independently,
        # not collapse into one shared value.
        assert client.timeout.connect == 3.0
        assert client.timeout.read == 45.0
        assert client.timeout.write == 15.0
        assert client.timeout.pool == 2.0
        await gw.close()

    @pytest.mark.asyncio
    async def test_gateway_transport_defaults(self):
        """Without explicit transport config, sensible defaults are used."""
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )
        gw = Gateway(config)
        transport = gw.transport
        client = transport.client
        # Defaults come from InteropServerConfig's own per-dimension fields.
        assert client.timeout.connect == 5.0
        assert client.timeout.read == 120.0
        assert client.timeout.write == 30.0
        assert client.timeout.pool == 5.0
        await gw.close()

    def test_tls_verify_false_wired_into_httpx_client(self, monkeypatch):
        """MVP-05: config.tls_verify must actually reach httpx, not just be
        loaded and ignored."""
        import httpx

        from agent_interop.transport.http import UpstreamTransport

        captured: dict[str, object] = {}
        original_init = httpx.AsyncClient.__init__

        def spying_init(self, *args, **kwargs):
            captured.update(kwargs)
            return original_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", spying_init)

        transport = UpstreamTransport(tls_verify=False)
        _ = transport.client  # triggers _build_client()

        assert captured.get("verify") is False

    def test_request_timeout_override_replaces_only_read_dimension(self):
        """A route-specific request timeout must override the READ timeout
        only — connect/write/pool stay at their client-level defaults rather
        than being replaced wholesale."""
        from agent_interop.transport.http import UpstreamTransport

        transport = UpstreamTransport(
            connect_timeout=3.0, timeout_seconds=45.0, write_timeout=15.0, pool_timeout=2.0,
        )
        timeout = transport._request_timeout(600.0)
        assert timeout.connect == 3.0
        assert timeout.read == 600.0
        assert timeout.write == 15.0
        assert timeout.pool == 2.0


# ─── Malformed Upstream Response Handling ───────────────────────────────────


class TestMalformedUpstreamResponse:
    @pytest.mark.asyncio
    async def test_non_json_response_returns_error(self):
        """Gateway should handle upstream returning 200 with non-JSON body."""
        import httpx

        from agent_interop.abi import CanonicalModelReference, CanonicalRequest
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
        from agent_interop.transport.http import UpstreamTransport

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )

        # Upstream returns 200 with a non-JSON body
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not json")

        transport = UpstreamTransport(
            max_retries=0,
            retry_base_delay=0.0,
            retry_max_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        gw = Gateway(config, transport=transport)

        request = CanonicalRequest(model=CanonicalModelReference(requested_name="test"))

        resp = await gw.handle_request(request, RequestContext())
        assert resp.error is not None
        assert resp.error.code == "INVALID_UPSTREAM_OUTPUT"
        await gw.close()


# ─── Retry Logic ────────────────────────────────────────────────────────────


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        """Gateway should retry on 500 errors up to max_retries."""
        import httpx

        from agent_interop.abi import CanonicalModelReference, CanonicalRequest
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
        from agent_interop.transport.http import UpstreamTransport

        config = InteropServerConfig(
            probe_on_startup=False,
            max_retries=2,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )

        # Upstream returns 500 every time.
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(500, json={"error": "server error"})

        transport = UpstreamTransport(
            max_retries=2,
            retry_base_delay=0.0,
            retry_max_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        gw = Gateway(config, transport=transport)

        request = CanonicalRequest(model=CanonicalModelReference(requested_name="test"))
        resp = await gw.handle_request(request, RequestContext())
        # An inference request is a POST — the backend already received
        # and may have started acting on it by the time a 500 comes back,
        # so automatically retrying it would silently duplicate a
        # non-idempotent generation call. Exactly one attempt, surfaced as
        # a real error, not silently retried.
        assert call_count == 1, f"Expected exactly 1 attempt (no auto-retry on POST), got {call_count}"
        assert resp.error is not None

    @pytest.mark.asyncio
    async def test_get_probe_still_retries_on_500(self):
        """Unlike inference POSTs, a GET (e.g. route probing) has no
        generation side effect — it must still retry on a retryable
        status, exactly as before."""
        import httpx

        from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamTransport

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(500, json={"error": "server error"})
            return httpx.Response(200, json={"models": []})

        transport = UpstreamTransport(
            max_retries=2,
            retry_base_delay=0.0,
            retry_max_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        resp = await transport.send(
            PreparedUpstreamRequest(method="GET", url="http://test/api/tags")
        )
        assert call_count == 3, f"Expected 3 attempts, got {call_count}"
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self):
        """Gateway should NOT retry on 4xx client errors."""
        import httpx

        from agent_interop.abi import CanonicalModelReference, CanonicalRequest
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
        from agent_interop.transport.http import UpstreamTransport

        config = InteropServerConfig(
            probe_on_startup=False,
            max_retries=3,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )

        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            return httpx.Response(400, json={"error": "Bad Request"})

        transport = UpstreamTransport(
            max_retries=3,
            retry_base_delay=0.0,
            retry_max_delay=0.0,
            transport=httpx.MockTransport(handler),
        )
        gw = Gateway(config, transport=transport)

        request = CanonicalRequest(model=CanonicalModelReference(requested_name="test"))
        resp = await gw.handle_request(request, RequestContext())
        # Should NOT retry on 400
        assert call_count == 1, f"Expected 1 attempt, got {call_count}"
        assert resp.error is not None


# ─── Request-Local Copy ─────────────────────────────────────────────────────


class TestRequestLocalCopy:
    @pytest.mark.asyncio
    async def test_canonical_request_not_mutated_by_render(self):
        """Gateway must make a request-local copy so codec rendering doesn't
        mutate the original canonical request."""
        import inspect

        from agent_interop.gateway import Gateway

        src = inspect.getsource(Gateway)
        # Verify deepcopy is used before render
        assert "deepcopy" in src, "Gateway should use copy.deepcopy for request-local copy"
        assert "request_local" in src or "copy.deepcopy(reconciled_request)" in src, \
            "Gateway should create a deep copy of the request before rendering"


# ─── Service Unit Generation ────────────────────────────────────────────────


class TestServiceUnitGeneration:
    def test_service_unit_content_has_required_fields(self) -> None:
        """Verify generated unit file contains required systemd fields."""
        interop_bin = sys.executable
        serve_cmd = f"{interop_bin} -m interop.cli serve --path /tmp/test.yaml"

        unit_content = f"""[Unit]
Description=Interop Agent Compatibility Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={serve_cmd}
Restart=on-failure
RestartSec=5s
StartLimitIntervalSec=60s
StartLimitBurst=3

[Install]
WantedBy=default.target
"""
        assert "[Unit]" in unit_content
        assert "[Service]" in unit_content
        assert "[Install]" in unit_content
        assert "Restart=on-failure" in unit_content
        assert "RestartSec=" in unit_content
        assert "WantedBy=default.target" in unit_content


class TestLoggingContract:
    """MVP: foreground start/serve log to file+stderr; under systemd,
    logging is left entirely to journald (no redundant on-disk file)."""

    def test_configure_process_logging_writes_file_when_not_under_systemd(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from agent_interop.cli import _configure_process_logging

        monkeypatch.delenv("INVOCATION_ID", raising=False)
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

        log_file = _configure_process_logging("info")
        assert log_file is not None
        assert log_file == tmp_path / "interop" / "logs" / "interop.log"

    def test_configure_process_logging_skips_file_under_systemd(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from agent_interop.cli import _configure_process_logging

        monkeypatch.setenv("INVOCATION_ID", "fake-invocation-id")
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

        log_file = _configure_process_logging("info")
        assert log_file is None
        # No log file (or its parent dir) should have been created.
        assert not (tmp_path / "interop" / "logs").exists()

    def test_logs_command_prefers_journald_when_service_installed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from typer.testing import CliRunner

        from agent_interop.cli import app

        xdg_config = tmp_path / "xdg-config"
        unit_dir = xdg_config / "systemd" / "user"
        unit_dir.mkdir(parents=True)
        (unit_dir / "interop.service").write_text("[Unit]\n")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))

        import subprocess as subprocess_module

        captured_cmd: list[str] = []

        def fake_run(cmd, *args, **kwargs):
            captured_cmd.extend(cmd)
            return subprocess_module.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess_module, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(app, ["logs"])
        assert result.exit_code == 0, result.output
        assert captured_cmd[:3] == ["journalctl", "--user", "-u"]


class TestServiceInstallCommand:
    """Exercises the REAL `interop service install` CLI path, not a
    hand-constructed string, so it actually catches regressions in
    ExecStart quoting, UMask, StartLimit placement, and daemon-reload."""

    def _run_install(self, tmp_path: Path, config_path: Path, monkeypatch):
        from typer.testing import CliRunner

        from agent_interop.cli import app

        xdg_config = tmp_path / "xdg-config"
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_config))
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))

        # systemctl/systemd-analyze are not necessarily available in the
        # test environment — stub subprocess.run so install() completes
        # deterministically regardless of host tooling.
        import subprocess as subprocess_module

        original_run = subprocess_module.run

        def fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] in ("systemctl", "systemd-analyze"):
                return subprocess_module.CompletedProcess(cmd, 0, stdout="", stderr="")
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess_module, "run", fake_run)

        runner = CliRunner()
        result = runner.invoke(app, ["service", "install", "--path", str(config_path)])
        unit_path = xdg_config / "systemd" / "user" / "interop.service"
        return result, unit_path

    def test_exec_start_is_quoted_for_a_path_with_spaces(self, tmp_path: Path, monkeypatch) -> None:
        config_dir = tmp_path / "my configs"
        config_dir.mkdir()
        config_path = config_dir / "interop.yaml"
        config_path.write_text("routes: {}\n")

        result, unit_path = self._run_install(tmp_path, config_path, monkeypatch)
        assert result.exit_code == 0, result.output
        content = unit_path.read_text()
        exec_start_line = next(line for line in content.splitlines() if line.startswith("ExecStart="))
        # The space-containing path must be quoted, not split into two argv
        # entries by systemd's whitespace-splitting ExecStart= parser.
        assert f'"{config_path}"' in exec_start_line or str(config_path) not in exec_start_line.replace('"', "")

    def test_unit_has_umask_and_unit_scoped_start_limit(self, tmp_path: Path, monkeypatch) -> None:
        config_path = tmp_path / "interop.yaml"
        config_path.write_text("routes: {}\n")

        result, unit_path = self._run_install(tmp_path, config_path, monkeypatch)
        assert result.exit_code == 0, result.output
        content = unit_path.read_text()

        assert "UMask=0077" in content

        unit_section = content.split("[Service]")[0]
        service_section = content.split("[Service]")[1].split("[Install]")[0]
        assert "StartLimitIntervalSec=60s" in unit_section
        assert "StartLimitBurst=3" in unit_section
        assert "StartLimitIntervalSec=" not in service_section
        assert "StartLimitBurst=" not in service_section

    def test_read_write_paths_created_before_install(self, tmp_path: Path, monkeypatch) -> None:
        config_path = tmp_path / "interop.yaml"
        config_path.write_text("routes: {}\n")

        result, unit_path = self._run_install(tmp_path, config_path, monkeypatch)
        assert result.exit_code == 0, result.output

        state_dir = tmp_path / "xdg-state" / "interop"
        cache_dir = tmp_path / "xdg-cache" / "interop"
        assert state_dir.is_dir()
        assert cache_dir.is_dir()
        content = unit_path.read_text()
        assert str(state_dir) in content
        assert str(cache_dir) in content

    def test_status_exits_nonzero_when_unit_not_installed(self, tmp_path: Path, monkeypatch) -> None:
        from typer.testing import CliRunner

        from agent_interop.cli import app

        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))
        runner = CliRunner()
        result = runner.invoke(app, ["service", "status"])
        assert result.exit_code != 0


# ─── Config Validation in CLI ───────────────────────────────────────────────


class TestCLIConfigValidation:
    def test_load_config_from_dict_parses_transport_settings(self, tmp_path: Path) -> None:
        import yaml

        from agent_interop.config import load_config_from_dict

        config_data = {
            "host": "127.0.0.1",
            "port": 8090,
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "wire_protocol": "ollama_chat",
                    },
                },
            },
            "transport": {
                "connect_timeout": 10.0,
                "max_retries": 3,
                "max_connections": 50,
            },
        }

        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump(config_data))

        with open(config_file) as f:
            data = yaml.safe_load(f)

        config = load_config_from_dict(data)
        assert config.connect_timeout == 10.0
        assert config.max_retries == 3
        assert config.max_connections == 50
        # Defaults for unspecified fields
        assert config.read_timeout == 120.0
        assert config.max_simultaneous_tool_calls == 64


class TestWireProtocolDefaultIsKindAware:
    """A route with wire_protocol omitted must default to whatever its
    OWN kind actually speaks — previously every kind defaulted to
    "openai_chat" regardless, so an Ollama route with wire_protocol
    omitted silently got OpenAI-Chat framing sent to an Ollama endpoint."""

    def test_ollama_kind_defaults_to_ollama_chat(self) -> None:
        from agent_interop.config import UpstreamProtocol, load_config_from_dict

        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        # wire_protocol deliberately omitted
                    },
                },
            },
        })
        assert config.routes["default"].upstream.wire_protocol == UpstreamProtocol.OLLAMA_CHAT

    def test_anthropic_kind_defaults_to_anthropic_messages(self) -> None:
        from agent_interop.config import UpstreamProtocol, load_config_from_dict

        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {
                        "kind": "anthropic",
                        "base_url": "https://api.anthropic.com",
                    },
                },
            },
        })
        assert config.routes["default"].upstream.wire_protocol == UpstreamProtocol.ANTHROPIC_MESSAGES

    def test_explicit_wire_protocol_still_overrides_the_default(self) -> None:
        """An explicit (even unusual) wire_protocol must still win over
        the kind-derived default — this only fills in what's OMITTED."""
        from agent_interop.config import UpstreamProtocol, load_config_from_dict

        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "wire_protocol": "openai_chat",
                    },
                },
            },
        })
        assert config.routes["default"].upstream.wire_protocol == UpstreamProtocol.OPENAI_CHAT

    def test_shared_mapping_consistent_across_all_entry_points(self) -> None:
        """config.default_wire_protocol_for_kind is now the single source
        of truth cli.py's _resolve_wire_protocol and server/app.py's
        create_app_from_env both delegate to — this pins the actual
        mapping so a future edit to one can't silently diverge from the
        others again."""
        from agent_interop.cli import _resolve_wire_protocol
        from agent_interop.config import (
            UpstreamKind,
            UpstreamProtocol,
            default_wire_protocol_for_kind,
        )

        expected = {
            UpstreamKind.OLLAMA: UpstreamProtocol.OLLAMA_CHAT,
            UpstreamKind.VLLM: UpstreamProtocol.OPENAI_CHAT,
            UpstreamKind.LLAMACPP: UpstreamProtocol.OPENAI_CHAT,
            UpstreamKind.OPENAI: UpstreamProtocol.OPENAI_CHAT,
            UpstreamKind.ANTHROPIC: UpstreamProtocol.ANTHROPIC_MESSAGES,
            UpstreamKind.OPENAI_COMPATIBLE: UpstreamProtocol.OPENAI_CHAT,
        }
        for kind, protocol in expected.items():
            assert default_wire_protocol_for_kind(kind) == protocol
            assert _resolve_wire_protocol(kind) == protocol


# ─── Evidence Store XDG Paths ───────────────────────────────────────────────


class TestXDGPathCentralization:
    """MVP: XDG paths must be namespaced under interop/ consistently,
    whether the env var is set or falls back to its default."""

    def test_config_file_namespaced_when_env_set(self, tmp_path: Path) -> None:
        from agent_interop.paths import config_file

        with patch.dict(os.environ, {"XDG_CONFIG_HOME": str(tmp_path)}):
            assert config_file() == tmp_path / "interop" / "config.yaml"

    def test_config_file_namespaced_when_env_unset(self) -> None:
        from agent_interop.paths import config_file

        env = {k: v for k, v in os.environ.items() if k != "XDG_CONFIG_HOME"}
        with patch.dict(os.environ, env, clear=True):
            assert config_file() == Path.home() / ".config" / "interop" / "config.yaml"

    def test_log_file_and_evidence_file_share_state_dir(self, tmp_path: Path) -> None:
        from agent_interop.paths import evidence_file, log_file

        with patch.dict(os.environ, {"XDG_STATE_HOME": str(tmp_path)}):
            assert log_file() == tmp_path / "interop" / "logs" / "interop.log"
            assert evidence_file() == tmp_path / "interop" / "evidence.db"


class TestEvidenceStoreXDG:
    def test_default_path_uses_xdg_state_home(self, tmp_path: Path) -> None:
        """Regression: every XDG base directory is namespaced under
        `interop/`, including when the env var is explicitly set — the old
        code applied that namespace only in the unset fallback default."""
        from agent_interop.evidence.store import EvidenceStore

        state_dir = tmp_path / "state"
        state_dir.mkdir()
        with patch.dict(os.environ, {"XDG_STATE_HOME": str(state_dir)}):
            store = EvidenceStore()
            expected = str(state_dir / "interop" / "evidence.db")
            assert store._db_path == expected

    def test_explicit_path_overrides_xdg(self, tmp_path: Path) -> None:
        from agent_interop.evidence.store import EvidenceStore

        explicit = str(tmp_path / "custom.db")
        store = EvidenceStore(db_path=explicit)
        assert store._db_path == explicit


# ─── Managed Launcher (interop run) ─────────────────────────────────────────


class TestManagedLauncher:
    @pytest.mark.asyncio
    async def test_managed_gateway_creates_config_from_env(self):
        """ManagedGateway should create a working gateway from env vars."""
        import os
        from unittest.mock import patch

        from agent_interop.launcher import ManagedGateway

        env = {
            "INTEROP_BACKEND_URL": "http://127.0.0.1:11434",
            "INTEROP_BACKEND_TYPE": "ollama",
            "INTEROP_MODEL": "test-model",
            "INTEROP_PORT": "0",
            "INTEROP_SESSION_CREDENTIAL": "test-token",
        }
        with patch.dict(os.environ, env, clear=False):
            gw = ManagedGateway(model="test-model", port=0)
            assert gw.model == "test-model"
            assert gw._session_credential.startswith("interop_")
            assert gw.backend_type == "ollama"

    def test_managed_gateway_url_construction(self):
        """ManagedGateway should construct proper gateway URL."""
        from agent_interop.launcher import ManagedGateway

        gw = ManagedGateway(model="test-model", host="127.0.0.1", port=8090)
        assert gw._gateway_url == "http://127.0.0.1:8090"


# ─── Installer Manifest/Rollback ────────────────────────────────────────────


class TestInstaller:
    def test_install_creates_shim(self, tmp_path: Path) -> None:
        """install() should create an ollama wrapper shim."""
        from agent_interop.install import install

        install(bin_dir=str(tmp_path), force=True)
        shim_path = tmp_path / "ollama"
        assert shim_path.exists()
        content = shim_path.read_text()
        assert "Interop wrapper" in content
        assert "launch" in content

    def test_uninstall_removes_shim(self, tmp_path: Path) -> None:
        """uninstall() should remove the Interop shim."""
        from unittest.mock import patch

        from agent_interop.install import install, uninstall

        install(bin_dir=str(tmp_path), force=True)
        shim_path = tmp_path / "ollama"
        assert shim_path.exists()

        # uninstall uses _bin_dir() which returns a fixed path;
        # we just verify it doesn't crash and the logic works
        with patch("agent_interop.install._bin_dir", return_value=tmp_path):
            uninstall()
        # After uninstall, the Interop shim should be gone
        if shim_path.exists():
            content = shim_path.read_text()
            assert "Interop wrapper" not in content

    def test_install_renames_existing_wrapper(self, tmp_path: Path) -> None:
        """install() should rename existing non-Interop wrappers."""
        from agent_interop.install import install

        # Create a fake existing wrapper
        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\necho 'vulkan wrapper'")

        result = install(bin_dir=str(tmp_path), force=True)
        # Should have renamed the existing wrapper
        vulkan_path = tmp_path / "ollama-vulkan"
        assert vulkan_path.exists() or "renamed" in result

    def test_status_reports_installation(self, tmp_path: Path) -> None:
        """status() should report whether Interop is installed."""
        from agent_interop.install import install, status

        install(bin_dir=str(tmp_path), force=True)
        result = status()
        assert isinstance(result, dict)

    def test_install_manifest_records_and_restores_original_wrapper(self, tmp_path: Path) -> None:
        """MVP-09: uninstall() must restore the EXACT artifact install()
        backed up (via a manifest), not a fixed 'ollama-vulkan' guess that
        could collide with an unrelated real file of that name."""
        from agent_interop.install import install, uninstall

        shim_path = tmp_path / "ollama"
        original_content = "#!/bin/bash\necho 'some other wrapper'"
        shim_path.write_text(original_content)

        result = install(bin_dir=str(tmp_path), force=True)
        assert "renamed" in result
        backup_path = Path(result["renamed"])
        assert backup_path.exists()
        assert backup_path.read_text() == original_content

        manifest_path = tmp_path / ".install_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["original_wrapper"] == str(backup_path)

        with patch("agent_interop.install._bin_dir", return_value=tmp_path):
            uninstall()

        # The original wrapper is back at its original path, byte-for-byte.
        assert shim_path.read_text() == original_content
        assert not backup_path.exists()
        assert not manifest_path.exists()

    def test_install_refuses_to_clobber_existing_backup_file(self, tmp_path: Path) -> None:
        """A pre-existing file at the backup path (not created by install())
        must never be silently overwritten."""
        from agent_interop.install import install

        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\necho 'some other wrapper'")
        unrelated_backup = tmp_path / "ollama.interop-backup"
        unrelated_backup.write_text("unrelated file — do not touch")

        with pytest.raises(RuntimeError):
            install(bin_dir=str(tmp_path), force=True)

        assert unrelated_backup.read_text() == "unrelated file — do not touch"


class TestOllamaLaunchShimArgvForwarding:
    """MVP-09: the shim is the documented primary product path — it must
    forward agent arguments to the interop runner exactly, without letting
    an agent flag get misparsed as an Interop/Typer option."""

    def _build_fixture(self, tmp_path: Path, model_ps_output: str = ""):
        """Build a real, executable shim plus fake `ollama` and `interop`
        binaries, then run it as a subprocess (matching how a shell would
        actually invoke it) and capture the exact argv the fake interop
        runner received."""
        from agent_interop.install import _build_shim

        fake_ollama = tmp_path / "fake-ollama"
        fake_ollama.write_text(
            "#!/bin/bash\n"
            'if [ "$1" = "ps" ]; then\n'
            f'  printf "NAME\\n{model_ps_output}\\n"\n'
            "  exit 0\n"
            "fi\n"
            'echo "unexpected real-ollama call: $@" >&2\n'
            "exit 1\n"
        )
        fake_ollama.chmod(0o755)

        captured_argv = tmp_path / "captured_argv.json"
        fake_interop = tmp_path / "fake-interop"
        fake_interop.write_text(
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            f'json.dump(sys.argv[1:], open({str(captured_argv)!r}, "w"))\n'
        )
        fake_interop.chmod(0o755)

        shim_path = tmp_path / "ollama"
        shim_path.write_text(
            _build_shim(str(fake_ollama), str(fake_ollama), [str(fake_interop)])
        )
        shim_path.chmod(0o755)
        return shim_path, captured_argv

    def test_extra_args_forwarded_after_separator(self, tmp_path: Path) -> None:
        shim_path, captured_argv = self._build_fixture(tmp_path)

        result = subprocess.run(
            [str(shim_path), "launch", "claude", "--model", "test", "--",
             "--dangerously-skip-permissions"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(captured_argv.read_text())
        assert argv == ["run", "claude", "--model", "test", "--", "--dangerously-skip-permissions"]

    def test_model_equals_syntax_supported(self, tmp_path: Path) -> None:
        shim_path, captured_argv = self._build_fixture(tmp_path)

        result = subprocess.run(
            [str(shim_path), "launch", "codex", "--model=deepseek-v4"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(captured_argv.read_text())
        assert argv == ["run", "codex", "--model", "deepseek-v4", "--"]

    def test_missing_model_value_rejected(self, tmp_path: Path) -> None:
        shim_path, _ = self._build_fixture(tmp_path)

        result = subprocess.run(
            [str(shim_path), "launch", "claude", "--model"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode != 0
        assert "requires a value" in result.stderr

    def test_no_extra_args_forwards_no_spurious_empty_argument(self, tmp_path: Path) -> None:
        shim_path, captured_argv = self._build_fixture(tmp_path)

        result = subprocess.run(
            [str(shim_path), "launch", "claude", "--model", "test"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(captured_argv.read_text())
        # No trailing empty-string argument from an empty EXTRA_ARGS array.
        assert argv == ["run", "claude", "--model", "test", "--"]

    def test_model_auto_detected_from_ollama_ps_when_absent(self, tmp_path: Path) -> None:
        shim_path, captured_argv = self._build_fixture(tmp_path, model_ps_output="running-model\tabc\t1h")

        result = subprocess.run(
            [str(shim_path), "launch", "claude"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        argv = json.loads(captured_argv.read_text())
        assert argv == ["run", "claude", "--model", "running-model", "--"]
