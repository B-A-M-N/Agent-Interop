"""The core gateway engine — orchestrates protocol translation, model calls,
and response conversion.

The Gateway is the central object that ties together:
1. Client protocol adapters (inbound protocol parsing)
2. Model profile system (capability-aware rendering)
3. Backend adapters (outbound HTTP calls)
4. Tool-call parsers (extraction from model output)
5. Validation and repair (quality enforcement)
6. Response encoding (back to client protocol)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import httpx

from interop.backends.registry import get_backend
from interop.model.profiles import get_profile, parse_tool_calls
from interop.model.template import render_messages, render_tools
from interop.protocols.registry import detect_protocol, get_adapter
from interop.repair.validate import RepairReport, repair_tool_calls
from interop.types import (
    BackendEvent,
    BackendKind,
    BackendRequest,
    CanonicalEvent,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalTool,
    CapabilityLevel,
    ContentBlock,
    InteropConfig,
    ModelProfile,
    ProtocolKind,
    RepairAction,
    ServerInfo,
    ToolCall,
    ToolResult,
)

logger = logging.getLogger("interop.gateway")


class Gateway:
    """Core agent compatibility gateway."""

    def __init__(self, config: InteropConfig | None = None) -> None:
        self.config = config or InteropConfig()
        self._backend = get_backend(self.config.backend)
        self._http_client: httpx.AsyncClient | None = None
        self._profile: ModelProfile | None = None
        self._resolved_model: str = ""

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=httpx.Timeout(self.config.backend_timeout or 120.0))
        return self._http_client

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    # ─── Startup / Probe ──────────────────────────────────────────────────

    async def startup(self) -> None:
        """Initialize the gateway, probe the model if configured."""
        logger.info(
            "interop starting — backend=%s url=%s model=%s",
            self.config.backend.value,
            self.config.backend_url,
            self.config.model,
        )

        self._resolved_model = self.config.model
        self._profile = get_profile(self.config.model) or ModelProfile(
            model=self.config.model,
            capabilities=CapabilityLevel.L0,
        )

        if self.config.probe_on_startup:
            await self._probe_model()

    async def _probe_model(self) -> None:
        """Probe the backend to verify connectivity and model existence."""
        try:
            r = await self.http_client.get(
                f"{self.config.backend_url}/api/tags",
                timeout=10.0,
            )
            if r.status_code == 200:
                models = r.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                logger.info("available models (%d): %s", len(models), model_names[:5])
            else:
                logger.warning("backend probe returned %d", r.status_code)
        except Exception as exc:
            logger.warning("backend probe failed: %s", exc)

    # ─── Server info ──────────────────────────────────────────────────────

    def server_info(self) -> ServerInfo:
        level = self._profile.capabilities if self._profile else CapabilityLevel.L0
        level_desc = {
            CapabilityLevel.L0: "Chat only",
            CapabilityLevel.L1: "Forced single-tool calling",
            CapabilityLevel.L2: "Automatic single-tool calling",
            CapabilityLevel.L3: "Parallel and sequential tools",
            CapabilityLevel.L4: "Reliable coding-agent operation",
        }.get(level, "")
        return ServerInfo(
            version="0.1.0",
            model=self._resolved_model,
            profile=(self._profile.model if self._profile else None),
            level=level.value,
            level_description=level_desc,
            supports=list(level.name for level in CapabilityLevel
                         if level.value <= level.value),
        )

    # ─── Non-streaming request ────────────────────────────────────────────

    async def handle_request(
        self,
        canonical: CanonicalRequest,
        protocol: ProtocolKind,
    ) -> CanonicalResponse | dict[str, Any]:
        """Handle a non-streaming request end-to-end."""
        backend_resp = await self._call_backend(canonical, stream=False)
        return self._process_backend_response(backend_resp, canonical)

    # ─── Streaming request ────────────────────────────────────────────────

    async def handle_stream(
        self,
        canonical: CanonicalRequest,
        protocol: ProtocolKind,
    ) -> AsyncIterator[CanonicalEvent]:
        """Handle a streaming request, yielding canonical events."""
        adapter = get_adapter(protocol)
        backend_resp = await self._call_backend(canonical, stream=True)

        buffer = ""
        content_index = 0
        tool_calls_accumulated: list[ToolCall] = []

        async for be in self._stream_events(backend_resp):
            if be.done:
                # Emit accumulated tool calls
                for tc in tool_calls_accumulated:
                    yield CanonicalEvent(
                        type="tool_use",
                        index=content_index,
                        content_block=ContentBlock(type="tool_use", tool_call=tc),
                    )
                yield CanonicalEvent(type="message_stop")
                return

            if be.data is None:
                continue

            # Extract text deltas from chat completion format
            choices = be.data.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                text = delta.get("content", "")
                if text:
                    yield CanonicalEvent(
                        type="text_delta",
                        index=content_index,
                        partial=text,
                    )

                # Tool calls in streaming
                tool_calls_delta = delta.get("tool_calls", [])
                if tool_calls_delta:
                    buffer += json.dumps(tool_calls_delta)

                # When streaming finishes, parse accumulated content
                finish = choices[0].get("finish_reason")
                if finish and finish not in (None, "null"):
                    yield CanonicalEvent(type="message_stop")

            # Also check full content for tool call markers
            content = be.data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content and not choices:
                calls = parse_tool_calls(content, self._profile.tool_dialect if self._profile else "generic")
                for tc in calls:
                    yield CanonicalEvent(
                        type="tool_use",
                        index=content_index,
                        content_block=ContentBlock(type="tool_use", tool_call=tc),
                    )

    # ─── Backend communication ─────────────────────────────────────────────

    async def _call_backend(
        self,
        canonical: CanonicalRequest,
        stream: bool = False,
    ) -> dict[str, Any] | httpx.Response:
        """Build and send the backend request."""
        # Render messages with model template
        profile = self._profile or ModelProfile(model=self.config.model)
        rendered_tools = render_tools(canonical.tools)
        rendered_messages = render_messages(
            canonical.system,
            canonical.messages,
            canonical.tools,
            profile,
        )

        # Build backend request
        backend_req = self._backend.build_request(
            model=self._resolved_model,
            system="",
            messages=rendered_messages,
            tools=rendered_tools if canonical.has_tools() else None,
            tool_choice=canonical.tool_choice,
            max_tokens=canonical.max_tokens,
            temperature=canonical.temperature,
            stream=stream,
        )

        url = f"{self.config.backend_url}/v1/chat/completions"
        headers = {
            **backend_req.headers,
            "Authorization": self._backend_auth(),
        }

        if stream:
            resp = await self.http_client.post(
                url,
                headers=headers,
                json=backend_req.body,
                timeout=httpx.Timeout(300.0),
            )
            return resp
        else:
            resp = await self.http_client.post(
                url,
                headers=headers,
                json=backend_req.body,
                timeout=httpx.Timeout(120.0),
            )
            return resp.json()

    async def _stream_events(
        self, response: httpx.Response
    ) -> AsyncIterator[BackendEvent]:
        """Decode SSE stream from backend."""
        buffer = ""
        async for line in response.aiter_lines():
            be = self._backend.decode_event(line)
            if be.done:
                yield be
                return
            yield be

    # ─── Response processing ───────────────────────────────────────────────

    def _process_backend_response(
        self, body: dict[str, Any], canonical: CanonicalRequest
    ) -> CanonicalResponse:
        """Extract content and tool calls from a backend response."""
        content_blocks: list[ContentBlock] = []
        tool_calls: list[ToolCall] = []

        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content", "")

        # Extract text content
        if text:
            content_blocks.append(ContentBlock(type="text", text=text))

            # Parse tool calls from text (model-native format)
            dialect = self._profile.tool_dialect if self._profile else "generic"
            calls = parse_tool_calls(text, dialect)
            for tc in calls:
                tc.id = tc.id or f"tc_{uuid.uuid4().hex[:12]}"
                tool_calls.append(tc)
                content_blocks.append(ContentBlock(type="tool_use", tool_call=tc))

        # Also check for API-level tool_calls
        for raw_tc in msg.get("tool_calls", []):
            try:
                args = json.loads(raw_tc.get("function", {}).get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            tc = ToolCall(
                id=raw_tc.get("id", f"tc_{uuid.uuid4().hex[:12]}"),
                name=raw_tc.get("function", {}).get("name", ""),
                arguments=args,
                raw=raw_tc.get("function", {}).get("arguments", ""),
            )
            # Deduplicate
            if not any(c.name == tc.name and c.arguments == tc.arguments for c in tool_calls):
                tool_calls.append(tc)
                content_blocks.append(ContentBlock(type="tool_use", tool_call=tc))

        # Validate and repair
        if tool_calls and canonical.tools:
            repair_report = repair_tool_calls(tool_calls, canonical.tools)
            for result in repair_report.repairs:
                if result.repair_action != RepairAction.NONE and not result.fixed:
                    logger.warning(
                        "unreparable tool call: %s — %s",
                        result.call.name, result.error,
                    )

        finish = choice.get("finish_reason", "end_turn")
        stop_map = {
            "stop": "end_turn",
            "tool_calls": "tool_use",
            "length": "max_tokens",
        }
        stop_reason = stop_map.get(finish, finish)

        return CanonicalResponse(
            content=content_blocks,
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=body.get("usage", {}),
            model=body.get("model", ""),
            id=body.get("id", ""),
        )

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _backend_auth(self) -> str:
        if self.config.backend_api_key:
            return f"Bearer {self.config.backend_api_key}"
        return "Bearer not-needed"

    def get_profile(self) -> ModelProfile | None:
        return self._profile

    def get_model(self) -> str:
        return self._resolved_model