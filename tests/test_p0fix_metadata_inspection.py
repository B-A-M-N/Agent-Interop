"""P0-1/P0-2 regression: metadata-only runtime inspection + qualification default.

Locks in the review invariants:
  * live request traffic NEVER triggers behavioral model generations —
    inspection is metadata-only (or off);
  * ``probe_on_startup`` no longer gates request-time planning metadata
    (health probing and planning metadata are separate concerns);
  * schema-v2 config no longer defaults qualification.bootstrap to
    ``blocking_for_tool_requests`` — blocking first tool requests behind
    synthetic probe generations is opt-in;
  * runtime metadata is warmed concurrently at startup so the user's first
    generation pays no inspection cost.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RuntimeInspectionConfig,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    load_config_from_dict,
)
from agent_interop.context import RequestContext
from agent_interop.transport import UpstreamResponse


class _CountingTransport:
    """Records every request, distinguishing GET (metadata) from POST
    (inference). Returns a minimal valid chat completion for POSTs and a
    minimal models list for GETs."""

    def __init__(self) -> None:
        self.get_urls: list[str] = []
        self.post_urls: list[str] = []

    async def send(self, request):
        if request.method == "GET":
            self.get_urls.append(request.url)
            return UpstreamResponse(
                status_code=200,
                body=json.dumps({"models": [{"name": "fake-model"}]}).encode(),
            )
        self.post_urls.append(request.url)
        return UpstreamResponse(
            status_code=200,
            body=json.dumps({
                "id": "x", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode(),
        )

    async def stream(self, request, raise_on=()):  # pragma: no cover
        raise NotImplementedError()


class _GenerationSentinel(Exception):
    """Raised if a generation endpoint — a real model call — is attempted.

    Ollama's /api/show and /api/ps are POST *metadata* reads (model blob
    digests, loaded-state), so method alone cannot distinguish; the path
    can: /api/chat, /api/generate, /api/embed are the generation endpoints.
    """


_GENERATION_PATHS = ("/api/chat", "/api/generate", "/api/embed", "/v1/chat/completions")


class _NoGenerationTransport(_CountingTransport):
    async def send(self, request):
        if request.method == "POST" and any(p in request.url for p in _GENERATION_PATHS):
            raise _GenerationSentinel(request.url)
        # Metadata POSTs (/api/show, /api/ps) get an empty object — enough
        # for the inspector's optional fields. GETs (version/tags/ps) get
        # their real minimal payloads from the base class.
        if request.method != "POST":
            return await super().send(request)
        return UpstreamResponse(status_code=200, body=b"{}")


def _route() -> ModelRoute:
    return ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
    )


def _config(**inspection: Any) -> InteropServerConfig:
    return InteropServerConfig(
        probe_on_startup=False,
        runtime_inspection=RuntimeInspectionConfig(**inspection),
        routes={"r": _route()},
    )


def _chat_request() -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
    )


# ─── P0-1: request-time inspection is metadata-only ─────────────────────────


def test_inspection_never_generates():
    """A cold chat request with cached_metadata mode issues metadata reads
    BEFORE the single inference POST — and nothing in between. Behavioral
    probes would show up as extra /api/chat calls before the real one."""
    from agent_interop.gateway import Gateway

    transport = _CountingTransport()
    gw = Gateway(_config(mode="cached_metadata"), transport=transport)
    resp = asyncio.run(gw.handle_request(_chat_request(), RequestContext()))
    assert resp.error is None
    assert transport.get_urls, "metadata inspection should have run"
    generation_posts = [
        url for url in transport.post_urls
        if any(p in url for p in _GENERATION_PATHS)
    ]
    # Exactly ONE generation — the user's own. Zero behavioral probes.
    assert len(generation_posts) == 1, transport.post_urls


def test_inspection_off_contacts_nothing_for_metadata():
    from agent_interop.gateway import Gateway

    transport = _CountingTransport()
    gw = Gateway(_config(mode="off"), transport=transport)
    # Directly exercising the inspection helper — a full request would need
    # a streaming-capable transport.
    runtime = asyncio.run(gw._inspect_model_runtime(_route()))
    assert transport.get_urls == [], "mode=off must inspect nothing"
    assert runtime is not None


def test_metadata_warm_on_startup_uses_only_metadata():
    from agent_interop.gateway import Gateway

    transport = _NoGenerationTransport()
    gw = Gateway(_config(mode="cached_metadata"), transport=transport)
    asyncio.run(gw.startup())
    assert transport.get_urls, "startup warm-up should issue metadata reads"
    # After warm-up, a live inspection is served from cache — no more I/O.
    before = list(transport.get_urls)
    asyncio.run(gw._inspect_model_runtime(_route()))
    assert transport.get_urls == before, "warm cache must serve inspection"


def test_probe_on_startup_decoupled_from_request_metadata():
    """``--no-probe`` must not force static capabilities on request traffic;
    health probing and planning metadata are separate knobs."""
    from agent_interop.backends.base import ModelRuntimeCapabilities
    from agent_interop.gateway import Gateway

    transport = _NoGenerationTransport()
    gw = Gateway(_config(mode="cached_metadata"), transport=transport)
    runtime = asyncio.run(gw._inspect_model_runtime(_route()))
    # Static capabilities carry no digests; real metadata inspection does.
    assert isinstance(runtime, ModelRuntimeCapabilities)
    assert transport.get_urls, "probe_on_startup=False must not disable planning metadata"


# ─── P0-2: qualification bootstrap default ──────────────────────────────────


def test_schema_v2_bootstrap_default_is_not_blocking():
    config = load_config_from_dict({
        "schema_version": 2,
        "routes": {
            "r": {
                "client_model_aliases": ["m"],
                "upstream_model": "fake-model",
                "upstream": {
                    "kind": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "wire_protocol": "ollama_chat",
                },
            },
        },
    })
    route = config.routes["r"]
    assert route.qualification.bootstrap != "blocking_for_tool_requests"
    assert route.qualification.bootstrap == "on_demand"


def test_explicit_blocking_bootstrap_is_preserved():
    config = load_config_from_dict({
        "schema_version": 2,
        "routes": {
            "r": {
                "client_model_aliases": ["m"],
                "upstream_model": "fake-model",
                "qualification": {"bootstrap": "blocking"},
                "upstream": {
                    "kind": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "wire_protocol": "ollama_chat",
                },
            },
        },
    })
    assert config.routes["r"].qualification.bootstrap == "blocking"


def test_invalid_bootstrap_value_rejected():
    try:
        load_config_from_dict({
            "routes": {
                "r": {
                    "client_model_aliases": ["m"],
                    "upstream_model": "fake-model",
                    "qualification": {"bootstrap": "bogus"},
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "wire_protocol": "ollama_chat",
                    },
                },
            },
        })
    except ValueError as exc:
        assert "bootstrap" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid bootstrap value must be rejected")


# ─── runtime_inspection config parsing ──────────────────────────────────────


def test_runtime_inspection_config_parses():
    config = load_config_from_dict({
        "runtime_inspection": {"mode": "cached_metadata", "warm_on_startup": False, "ttl_seconds": 60},
        "routes": {
            "r": {
                "client_model_aliases": ["m"],
                "upstream_model": "fake-model",
                "upstream": {
                    "kind": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "wire_protocol": "ollama_chat",
                },
            },
        },
    })
    assert config.runtime_inspection.mode == "cached_metadata"
    assert config.runtime_inspection.warm_on_startup is False
    assert config.runtime_inspection.ttl_seconds == 60


def test_runtime_inspection_behavioral_mode_rejected():
    try:
        load_config_from_dict({
            "runtime_inspection": {"mode": "behavioral"},
            "routes": {
                "r": {
                    "client_model_aliases": ["m"],
                    "upstream_model": "fake-model",
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "wire_protocol": "ollama_chat",
                    },
                },
            },
        })
    except ValueError as exc:
        assert "behavioral" in str(exc) or "cached_metadata" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("behavioral inspection mode must not exist")


def test_validate_config_flags_bad_inspection_mode():
    from agent_interop.config import validate_config

    config = _config(mode="off")
    # Directly mutate to simulate an invalid object (constructor would allow
    # it since dataclasses do not validate).
    object.__setattr__(config.runtime_inspection, "mode", "behavioral")
    issues = validate_config(config)
    assert any("runtime_inspection" in issue for issue in issues)
