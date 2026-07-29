"""Protocol adapter registry — maps requests to matching adapters via matches()."""

from __future__ import annotations

from typing import Any, cast

from agent_interop.abi import ProtocolKind
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from agent_interop.protocols.base import ClientProtocolAdapter
from agent_interop.protocols.openai_chat import OpenAIChatAdapter
from agent_interop.protocols.openai_responses import OpenAIResponsesAdapter

_ADAPTERS: list[ClientProtocolAdapter] = []
_BY_KIND: dict[str, ClientProtocolAdapter] = {}


def _ensure_loaded() -> None:
    if not _ADAPTERS:
        for adapter in [
            AnthropicMessagesAdapter(),
            OpenAIChatAdapter(),
            OpenAIResponsesAdapter(),
        ]:
            _ADAPTERS.append(adapter)
            _BY_KIND[adapter.protocol] = adapter


def detect_adapter(path: str, headers: dict[str, str], body: dict[str, Any]) -> ClientProtocolAdapter | None:
    """Find the first adapter that matches the request."""
    _ensure_loaded()
    for adapter in _ADAPTERS:
        if adapter.matches(path, headers, body):
            return adapter
    return _ADAPTERS[1] if len(_ADAPTERS) > 1 else None  # fallback to chat


def get_adapter(kind: ProtocolKind) -> ClientProtocolAdapter:
    """Get a cached adapter instance for the given protocol kind."""
    _ensure_loaded()
    key = kind.value
    if key not in _BY_KIND:
        raise ValueError(f"Unknown protocol kind: {kind}")
    return _BY_KIND[key]


def detect_protocol(path: str, headers: dict[str, str], body: dict[str, Any]) -> ProtocolKind:
    """Detect the client protocol from request metadata."""
    adapter = detect_adapter(path, headers, body)
    if adapter:
        return cast(ProtocolKind, adapter.protocol)
    return ProtocolKind.OPENAI_CHAT


def list_adapters() -> list[ClientProtocolAdapter]:
    """List all registered adapters."""
    _ensure_loaded()
    return list(_ADAPTERS)