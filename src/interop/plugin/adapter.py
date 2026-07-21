"""Embeddable plugin interface for coding agent integration.

This module provides the LocalModelAdapter interface that coding agents
can use to integrate Interop as a runtime plugin. The plugin:

1. Intercepts model requests from the agent
2. Translates them through the gateway
3. Returns responses in the agent's expected format

It also provides the sidecar adapter that agents too rigid to support
plugins can use — spawn Interop as a local process and point the 
provider transport at it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from interop.gateway import Gateway
from interop.types import (
    AgentMessage,
    BackendKind,
    CanonicalRequest,
    CanonicalTool,
    CanonicalResponse,
    CapabilityLevel,
    ContentBlock,
    InteropConfig,
    ModelProfile,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger("interop.plugin")


@dataclass
class AgentCapabilities:
    """Capabilities that the local model adapter reports to the agent."""

    supports_tools: bool = False
    supports_parallel_tools: bool = False
    supports_images: bool = False
    supports_thinking: bool = False
    max_context_length: int = 4096
    level: CapabilityLevel = CapabilityLevel.L0


class LocalModelAdapter:
    """Plugin interface for embedding Interop in a coding agent.

    The adapter wraps the Gateway and provides a high-level API
    that coding agents can call. It handles: request translation,
    tool-call parsing, validation, and response encoding.

    Usage:
        adapter = LocalModelAdapter()
        await adapter.start(config)
        response = await adapter.generate(canonical_request)
        adapter.stop()
    """

    def __init__(self) -> None:
        self.gateway: Gateway | None = None
        self.config: InteropConfig | None = None
        self._running = False

    async def start(self, config: InteropConfig | None = None) -> None:
        """Start the adapter and connect to the backend."""
        self.config = config or InteropConfig()
        self.gateway = Gateway(self.config)
        await self.gateway.startup()
        self._running = True
        logger.info(
            "interop plugin started — backend=%s model=%s",
            self.config.backend.value,
            self.config.model,
        )

    def stop(self) -> None:
        """Shut down the adapter."""
        self._running = False
        if self.gateway:
            asyncio.create_task(self.gateway.close())
            self.gateway = None

    @property
    def capabilities(self) -> AgentCapabilities:
        """Return the capabilities of the loaded model."""
        if not self.gateway or not self.gateway.get_profile():
            return AgentCapabilities()

        profile = self.gateway.get_profile()
        return AgentCapabilities(
            supports_tools=profile.capabilities.value >= CapabilityLevel.L1.value,
            supports_parallel_tools=profile.parallel_tools,
            supports_images=profile.supports_images,
            supports_thinking=profile.supports_thinking,
            max_context_length=profile.context_length,
            level=profile.capabilities,
        )

    async def generate(
        self,
        request: CanonicalRequest,
    ) -> CanonicalResponse:
        """Generate a response from the local model.

        The request is passed through the gateway which handles:
        - Message rendering for the specific model
        - Backend HTTP call
        - Tool call parsing
        - Validation and repair
        - Response normalization
        """
        if not self.gateway or not self._running:
            raise RuntimeError("adapter not started; call start() first")

        return await self.gateway.handle_request(request, None) or ...  # type: ignore

    def encode_tool_result(self, result: ToolResult) -> AgentMessage:
        """Encode a tool execution result for the conversation history."""
        return AgentMessage(
            role="tool",
            content=result.content,
            tool_call_id=result.call_id,
            name=result.tool_name,
        )


# ─── Sidecar adapter ────────────────────────────────────────────────────────


@dataclass
class SidecarProcess:
    """A managed Interop sidecar process.

    The sidecar is an Interop server process spawned automatically
    when a local model is selected. It handles agents whose plugin
    system cannot intercept model transport but can point at a URL.
    """

    host: str = "127.0.0.1"
    port: int = 8090
    process: subprocess.Popen | None = None
    gateway_url: str = ""
    _started_at: float = 0.0

    def __post_init__(self) -> None:
        self.gateway_url = f"http://{self.host}:{self.port}"

    def start(
        self,
        backend: str = "ollama",
        backend_url: str = "http://127.0.0.1:11434",
        model: str = "qwen3-coder",
        timeout: float = 30.0,
    ) -> None:
        """Start the sidecar process."""
        if self.process and self.process.poll() is None:
            logger.warning("sidecar already running on %s", self.gateway_url)
            return

        cmd = [
            sys.executable, "-m", "interop.cli", "start",
            "--host", self.host,
            "--port", str(self.port),
            "--backend", backend,
            "--backend-url", backend_url,
            "--model", model,
            "--no-probe",
            "--log-level", "info",
        ]

        logger.info("starting interop sidecar: %s", " ".join(cmd))
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._started_at = time.monotonic()

        # Wait for the server to come up
        import httpx
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{self.gateway_url}/v1/health", timeout=2.0)
                if r.status_code == 200:
                    logger.info("sidecar ready at %s", self.gateway_url)
                    return
            except (httpx.RequestError, ConnectionError):
                pass
            time.sleep(0.5)

        raise RuntimeError(
            f"sidecar did not start within {timeout}s — "
            f"check the server logs at {self.gateway_url}"
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the sidecar process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            logger.info("sidecar stopped")

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None