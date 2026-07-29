"""Codec registry — maps UpstreamProtocol enum to ModelCodec instances.

Central registry so the Gateway can look up the correct upstream codec
for a given route's wire_protocol without hardcoding import paths.
"""

from __future__ import annotations

from agent_interop.config import UpstreamProtocol
from agent_interop.upstreams.anthropic import AnthropicCodec
from agent_interop.upstreams.codec import ModelCodec
from agent_interop.upstreams.ollama_chat import OllamaChatCodec
from agent_interop.upstreams.openai_chat import OpenAIChatCodec
from agent_interop.upstreams.openai_responses import OpenAIResponsesCodec

# UpstreamProtocol enum → ModelCodec instance

_CODECS: dict[UpstreamProtocol, ModelCodec] = {}


def _ensure_loaded() -> None:
    if not _CODECS:
        for codec in [
            OpenAIChatCodec(),
            OllamaChatCodec(),
            AnthropicCodec(),
            OpenAIResponsesCodec(),
        ]:
            _CODECS[codec.protocol] = codec


def get_codec(protocol: UpstreamProtocol | str) -> ModelCodec:
    """Get a cached codec instance for the given protocol.

    Args:
        protocol: The UpstreamProtocol enum value or its string value
                  (e.g. UpstreamProtocol.OPENAI_CHAT or "openai_chat").

    Returns:
        The ModelCodec instance.

    Raises:
        ValueError: If no codec is registered for the given protocol.
    """
    _ensure_loaded()
    # Normalize: accept enum or string value
    if isinstance(protocol, str):
        protocol = UpstreamProtocol(protocol)
    if protocol not in _CODECS:
        raise ValueError(
            f"Unknown upstream protocol: '{protocol.value}'. "
            f"Available: {sorted(p.value for p in _CODECS)}"
        )
    return _CODECS[protocol]


def list_codecs() -> list[ModelCodec]:
    """List all registered codecs."""
    _ensure_loaded()
    return list(_CODECS.values())