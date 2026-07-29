"""Embeddable plugin interface for coding agent integration.

This module provides the LocalModelAdapter interface that coding agents
can use to integrate Interop as a runtime plugin. The plugin:

1. Intercepts model requests from the agent
2. Translates them through the gateway
3. Returns responses in the agent's expected format

Uses the route-based InteropServerConfig (not legacy InteropConfig).
"""

from __future__ import annotations

import logging

from agent_interop.abi import CanonicalRequest, CanonicalResponse, ProtocolKind
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.gateway import Gateway

logger = logging.getLogger("agent_interop.plugin")


class LocalModelAdapter:
    """Plugin interface for embedding Interop in a coding agent.

    The adapter wraps the Gateway and provides a high-level API
    that coding agents can call. It handles: request translation,
    tool-call parsing, validation, and response encoding.

    Usage:
        adapter = LocalModelAdapter()
        await adapter.start(config)
        response = await adapter.generate(canonical_request)
        await adapter.close()
    """

    def __init__(self) -> None:
        self.gateway: Gateway | None = None
        self.config: InteropServerConfig | None = None
        self._running = False

    async def start(
        self,
        config: InteropServerConfig | None = None,
        *,
        model: str = "qwen3-coder",
        backend_url: str = "http://127.0.0.1:11434",
        backend_kind: UpstreamKind = UpstreamKind.OLLAMA,
        tool_mode: ToolMode = ToolMode.AUTO,
    ) -> None:
        """Start the adapter and connect to the backend.

        Can accept either a full InteropServerConfig or convenience params.
        """
        if config is None:
            # Build a one-route config from convenience params
            wire_protocol = _resolve_wire_protocol(backend_kind)
            config = InteropServerConfig(
                probe_on_startup=False,
                default_route_id="plugin",
                routes={
                    "plugin": ModelRoute(
                        id="plugin",
                        client_model_aliases=[model],
                        upstream_model=model,
                        upstream=UpstreamConfig(
                            kind=backend_kind,
                            base_url=backend_url,
                            wire_protocol=wire_protocol,
                        ),
                        tool_mode=tool_mode,
                        profile="auto",
                    ),
                },
            )

        self.config = config
        self.gateway = Gateway(config)
        await self.gateway.startup()
        self._running = True
        logger.info("interop plugin started — model=%s", model)

    async def close(self) -> None:
        """Shut down the adapter."""
        self._running = False
        if self.gateway:
            await self.gateway.close()
            self.gateway = None

    @property
    def is_running(self) -> bool:
        return self._running

    async def generate(
        self,
        request: CanonicalRequest,
    ) -> CanonicalResponse:
        """Generate a response from the local model.

        The request is passed through the gateway which handles:
        - Route resolution
        - Profile resolution
        - Upstream rendering
        - Tool call extraction, validation, and repair
        - Response normalization
        """
        if not self.gateway or not self._running:
            raise RuntimeError("adapter not started; call start() first")

        context = RequestContext(client_protocol=ProtocolKind.OPENAI_CHAT)
        return await self.gateway.handle_request(request, context)


def _resolve_wire_protocol(backend_kind: UpstreamKind) -> UpstreamProtocol:
    """Map a backend kind to its default wire protocol."""
    mapping = {
        UpstreamKind.OLLAMA: UpstreamProtocol.OLLAMA_CHAT,
        UpstreamKind.VLLM: UpstreamProtocol.OPENAI_CHAT,
        UpstreamKind.LLAMACPP: UpstreamProtocol.OPENAI_CHAT,
        UpstreamKind.OPENAI: UpstreamProtocol.OPENAI_CHAT,
        UpstreamKind.ANTHROPIC: UpstreamProtocol.ANTHROPIC_MESSAGES,
        UpstreamKind.OPENAI_COMPATIBLE: UpstreamProtocol.OPENAI_CHAT,
    }
    return mapping.get(backend_kind, UpstreamProtocol.OPENAI_CHAT)
