"""Tests for credential passthrough, header normalization, and case-insensitive merge."""

from __future__ import annotations

import pytest

from agent_interop.auth import (
    HOP_BY_HOP_HEADERS,
    UpstreamAuthConfig,
    UpstreamAuthMode,
    build_upstream_headers,
)
from agent_interop.context import RequestContext
from agent_interop.enums import ProtocolKind


class TestBuildUpstreamHeadersCaseInsensitive:
    def test_case_insensitive_merge_dedupes(self):
        # Mixed-case content-type from codec + route must not produce
        # two separate content-type entries.
        route_static = {"content-type": "application/json"}
        client = {"Content-Type": "application/x-ndjson"}
        headers = build_upstream_headers(
            client_headers=client,
            upstream_auth=UpstreamAuthConfig(mode=UpstreamAuthMode.NONE),
            route_upstream_headers=route_static,
        )
        # Only one content-type, route takes precedence over codec.
        ct_keys = [k for k in headers if k.lower() == "content-type"]
        assert len(ct_keys) == 1
        # Whichever value wins, the result is well-formed.
        assert headers[ct_keys[0]] in ("application/json", "application/x-ndjson")

    def test_passthrough_forwards_authorization(self):
        client = {
            "Authorization": "Bearer client-token",
            "x-api-key": "client-key",
        }
        headers = build_upstream_headers(
            client_headers=client,
            upstream_auth=UpstreamAuthConfig(mode=UpstreamAuthMode.PASSTHROUGH),
        )
        assert headers.get("authorization") == "Bearer client-token"
        assert headers.get("x-api-key") == "client-key"

    def test_passthrough_ignores_non_allowlisted(self):
        client = {
            "Authorization": "Bearer x",
            "X-Custom-Secret": "leaked",
        }
        headers = build_upstream_headers(
            client_headers=client,
            upstream_auth=UpstreamAuthConfig(mode=UpstreamAuthMode.PASSTHROUGH),
        )
        assert "x-custom-secret" not in headers
        assert "x-custom-secret" not in {k.lower() for k in headers}

    def test_api_key_env_unset_raises(self, monkeypatch):
        monkeypatch.delenv("INTEROP_TEST_KEY_X", raising=False)
        with pytest.raises(RuntimeError) as exc:
            build_upstream_headers(
                client_headers={},
                upstream_auth=UpstreamAuthConfig(
                    mode=UpstreamAuthMode.API_KEY,
                    env_key="INTEROP_TEST_KEY_X",
                    api_key_header="Authorization",
                ),
            )
        assert "INTEROP_TEST_KEY_X" in str(exc.value)

    def test_api_key_env_set_adds_bearer(self, monkeypatch):
        monkeypatch.setenv("INTEROP_TEST_KEY_Y", "secret-value")
        headers = build_upstream_headers(
            client_headers={},
            upstream_auth=UpstreamAuthConfig(
                mode=UpstreamAuthMode.API_KEY,
                env_key="INTEROP_TEST_KEY_Y",
                api_key_header="Authorization",
            ),
        )
        assert headers["authorization"] == "Bearer secret-value"

    def test_hop_by_hop_stripped(self):
        client = {"Connection": "keep-alive", "authorization": "Bearer x"}
        headers = build_upstream_headers(
            client_headers=client,
            upstream_auth=UpstreamAuthConfig(mode=UpstreamAuthMode.PASSTHROUGH),
        )
        for h in HOP_BY_HOP_HEADERS:
            assert h not in headers
            assert h not in {k.lower() for k in headers}


class TestRequestContextForwardableHeaders:
    def test_separate_diagnostic_and_forwardable(self):
        headers = {
            "Authorization": "Bearer client-token",
            "x-api-key": "client-key",
            "content-type": "application/json",
            "x-custom-allowed": "value",
            "cookie": "session=abc",
        }
        ctx = RequestContext.from_headers(headers, ProtocolKind.ANTHROPIC_MESSAGES)
        # Sanitized must not include credentials
        assert "authorization" not in {k.lower() for k in ctx.sanitized_headers}
        assert "x-api-key" not in {k.lower() for k in ctx.sanitized_headers}
        # Forwardable must include allowlisted credentials
        assert ctx.forwardable_transport_headers.get("authorization") == "Bearer client-token"
        assert ctx.forwardable_transport_headers.get("x-api-key") == "client-key"
        # Cookies must never be forwardable
        assert "cookie" not in {k.lower() for k in ctx.forwardable_transport_headers}
        # Diagnostic headers (cookie-bearing) stripped from forwardable
        assert ctx.forwardable_transport_headers.get("content-type") == "application/json"
        # Non-allowlisted client header is NOT in forwardable
        assert "x-custom-allowed" not in ctx.forwardable_transport_headers

    def test_case_insensitive_lookup(self):
        headers = {"AUTHORIZATION": "Bearer caps"}
        ctx = RequestContext.from_headers(headers, ProtocolKind.ANTHROPIC_MESSAGES)
        assert ctx.forwardable_transport_headers.get("authorization") == "Bearer caps"

    def test_route_id_propagates(self):
        ctx = RequestContext.from_headers(
            {}, ProtocolKind.ANTHROPIC_MESSAGES, route_id="qwen3-route"
        )
        assert ctx.route_id == "qwen3-route"

    def test_claude_code_client_detected_with_version(self):
        ctx = RequestContext.from_headers(
            {
                "x-claude-code-session-id": "sess-1",
                "x-claude-code-client-version": "1.2.3",
            },
            ProtocolKind.ANTHROPIC_MESSAGES,
        )
        assert ctx.client_id == "claude_code"
        assert ctx.client_version == "1.2.3"
