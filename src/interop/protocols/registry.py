"""Protocol adapter registry — maps requests to matching adapters via matches()."""

from __future__ import annotations

from typing import Any

from interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from interop.protocols.openai_chat import OpenAIChatAdapter
from interop.protocols.openai_responses import OpenAIResponsesAdapter
from interop.protocols.base import ClientProtocolAdapter
from interop.types import ProtocolKind


_ADAPTERS: list[ClientProtocolAdapter] = []
_BY_KIND: dict[ProtocolKind, ClientProtocolAdapter] = {}


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
    if kind not in _BY_KIND:
        raise ValueError(f"Unknown protocol kind: {kind}")
    return _BY_KIND[kind]


def detect_protocol(path: str, headers: dict[str, str], body: dict[str, Any]) -> ProtocolKind:
    """Detect the client protocol from request metadata."""
    adapter = detect_adapter(path, headers, body)
    if adapter:
        return adapter.protocol
    return ProtocolKind.OPENAI_CHAT


def list_adapters() -> list[ClientProtocolAdapter]:
    """List all registered adapters."""
    _ensure_loaded()
    return list(_ADAPTERS)