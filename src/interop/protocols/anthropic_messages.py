"""Anthropic Messages API adapter (/v1/messages, /v1/messages/count_tokens).

Translates between the Anthropic Messages API and Interop canonical form.
Supports content blocks, tool_use, tool_result, thinking blocks, and streaming.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from interop.protocols.base import ClientProtocolAdapter
from interop.types import (
    AgentMessage,
    CanonicalEvent,
    CanonicalRequest,
    CanonicalTool,
    ContentBlock,
    ProtocolKind,
    ToolCall,
    ToolResult,
    tool_from_anthropic,
    tool_to_anthropic,
)


class AnthropicMessagesAdapter(ClientProtocolAdapter):
    """Translate Anthropic Messages API to/from canonical form."""

    protocol = ProtocolKind.ANTHROPIC_MESSAGES
    id = "anthropic-messages"

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        if path.rstrip("/").endswith("/v1/messages"):
            return True
        if headers.get("anthropic-version") or headers.get("x-api-key", "").startswith("sk-ant-"):
            return True
        return False

    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        system = body.get("system", "")
        if isinstance(system, list):
            # Anthropic allows system as a list of content blocks
            system = " ".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )

        messages: list[AgentMessage] = []
        for msg in body.get("messages", []):
            role = msg["role"]
            raw_content = msg.get("content", "")

            if role == "assistant":
                if isinstance(raw_content, list):
                    blocks: list[ContentBlock] = []
                    tc: list[ToolCall] = []
                    for block in raw_content:
                        btype = block.get("type", "text")
                        if btype == "text":
                            blocks.append(ContentBlock(type="text", text=block.get("text", "")))
                        elif btype == "tool_use":
                            try:
                                inp = block.get("input", {})
                            except Exception:
                                inp = {}
                            tid = block.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
                            tcall = ToolCall(
                                id=tid,
                                name=block.get("name", ""),
                                arguments=dict(inp),
                                dialect=self.protocol,
                            )
                            tc.append(tcall)
                            blocks.append(ContentBlock(type="tool_use", tool_call=tcall))
                        elif btype == "thinking":
                            blocks.append(ContentBlock(
                                type="thinking",
                                text=block.get("thinking", ""),
                                signature=block.get("signature"),
                            ))
                    messages.append(AgentMessage(
                        role="assistant",
                        content=blocks,
                        tool_calls=tc if tc else None,
                    ))
                else:
                    messages.append(AgentMessage(role="assistant", content=str(raw_content)))

            elif role == "user":
                if isinstance(raw_content, list):
                    text_parts: list[str] = []
                    for block in raw_content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_result":
                            # Anthropic tool_result in user messages
                            tlid = block.get("tool_use_id", "")
                            content = block.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(
                                    b.get("text", "") for b in content if b.get("type") == "text"
                                )
                            messages.append(AgentMessage(
                                role="tool",
                                content=str(content),
                                tool_call_id=tlid,
                            ))
                    if text_parts:
                        messages.append(AgentMessage(role="user", content="\n".join(text_parts)))
                else:
                    messages.append(AgentMessage(role="user", content=str(raw_content)))

            elif role == "tool":
                messages.append(AgentMessage(
                    role="tool",
                    content=str(raw_content),
                    tool_call_id=msg.get("tool_use_id", ""),
                ))

        tools = []
        for tspec in body.get("tools", []):
            tools.append(tool_from_anthropic(tspec))

        tc = body.get("tool_choice", {"type": "auto"})
        if isinstance(tc, str):
            tc_map = {"auto": "auto", "any": "required", "tool": "required", "none": "none"}
            tc = tc_map.get(tc, "auto")
        elif isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "tool":
                tc = {"name": tc.get("name", "")}
            elif tc_type in ("any", "auto"):
                tc = "auto"
            elif tc_type == "none":
                tc = "none"

        return CanonicalRequest(
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tc,
            max_tokens=body.get("max_tokens", 4096),
            temperature=body.get("temperature", 0.0),
            stream=body.get("stream", False),
            extra={
                "anthropic_version": body.get("anthropic-version", "2023-06-01"),
                "metadata": body.get("metadata", {}),
            },
        )

    def encode_nonstream_response(
        self, canonical: CanonicalRequest, body: dict[str, Any]
    ) -> dict[str, Any]:
        # body is assumed to be from the backend already in anthropic format
        return {
            "id": body.get("id", f"msg_{uuid.uuid4().hex[:16]}"),
            "type": "message",
            "role": "assistant",
            "content": body.get("content", []),
            "model": body.get("model", canonical.extra.get("model", "unknown")),
            "stop_reason": body.get("stop_reason", "end_turn"),
            "stop_sequence": body.get("stop_sequence"),
            "usage": body.get("usage", {
                "input_tokens": 0,
                "output_tokens": 0,
            }),
        }

    def encode_stream_event(self, event: CanonicalEvent) -> str | None:
        """Encode canonical events as Anthropic SSE events."""
        if event.type == "text_delta":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "text_delta", "text": event.partial},
            })
        if event.type == "text":
            return self._sse("content_block_start", {
                "type": "content_block_start",
                "index": event.index,
                "content_block": {"type": "text", "text": event.partial or ""},
            })
        if event.type == "thinking_delta":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "thinking_delta", "thinking": event.partial},
            })
        if event.type == "thinking":
            return self._sse("content_block_start", {
                "type": "content_block_start",
                "index": event.index,
                "content_block": {
                    "type": "thinking",
                    "thinking": event.partial or "",
                    "signature": event.content_block.signature if event.content_block else None,
                },
            })
        if event.type == "thinking_signature":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {
                    "type": "signature_delta",
                    "signature": event.partial,
                },
            })
        if event.type == "tool_use":
            cb = event.content_block
            if cb and cb.tool_call:
                return self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": event.index,
                    "content_block": {
                        "type": "tool_use",
                        "id": cb.tool_call.id,
                        "name": cb.tool_call.name,
                        "input": cb.tool_call.arguments,
                    },
                })
        if event.type == "tool_use_delta":
            # Anthropic streams partial JSON for tool arguments
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "input_json_delta", "partial_json": event.partial},
            })
        if event.type == "content_block_stop":
            return self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": event.index,
            })
        if event.type == "message_stop":
            return self._sse("message_stop", {"type": "message_stop"})
        return None

    def encode_stream_done(self) -> str:
        return "data: [DONE]\n\n"

    def parse_tool_result(self, body: dict[str, Any]) -> str:
        content = body.get("content", "")
        if isinstance(content, list):
            return " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        return str(content)

    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": body.get("messages", []),
            "system": body.get("system", ""),
            "model": body.get("model", ""),
        }

    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_tokens": backend_body.get("input_tokens", 0),
            "output_tokens": backend_body.get("output_tokens", 0),
        }

    # ── helpers ──────────────────────────────────────────────────────────

    _EVENT_COUNTER = 0

    @classmethod
    def _sse(cls, event_type: str, data: dict[str, Any]) -> str:
        cls._EVENT_COUNTER += 1
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"