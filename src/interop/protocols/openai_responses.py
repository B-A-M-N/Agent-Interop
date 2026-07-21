"""OpenAI Responses API adapter (/v1/responses).

Adopted by Codex as its primary protocol. Extends the chat format with
structured response objects, response-level IDs, and continuation tokens.
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
    tool_from_openai,
)


class OpenAIResponsesAdapter(ClientProtocolAdapter):
    """Translate OpenAI Responses API to/from canonical form."""

    protocol = ProtocolKind.OPENAI_RESPONSES
    id = "openai-responses"

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        if path.rstrip("/").endswith("/v1/responses"):
            return True
        if "input" in body and "tools" in body:
            return True
        return False

    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        messages: list[AgentMessage] = []
        system = body.get("instructions", body.get("system", ""))

        # Previous response continuation
        prev_id = None
        for raw_msg in body.get("input", []):
            role = raw_msg.get("role", "user")

            if role == "system":
                system = raw_msg.get("content", system)
                continue

            content = raw_msg.get("content", "")

            if role == "assistant":
                blocks: list[ContentBlock] = []
                tc: list[ToolCall] = []
                for raw_tc in raw_msg.get("tool_calls") or []:
                    args_raw = raw_tc.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {}
                    tcall = ToolCall(
                        id=raw_tc.get("id", f"tc_{uuid.uuid4().hex[:12]}"),
                        name=raw_tc.get("name", ""),
                        arguments=args,
                        raw=args_raw,
                    )
                    tc.append(tcall)
                    blocks.append(ContentBlock(type="tool_use", tool_call=tcall))
                if content:
                    blocks.insert(0, ContentBlock(type="text", text=content))
                messages.append(AgentMessage(
                    role="assistant",
                    content=blocks if blocks else content,
                    tool_calls=tc if tc else None,
                ))

            elif role == "user":
                messages.append(AgentMessage(role="user", content=content))

            elif role == "tool":
                messages.append(AgentMessage(
                    role="tool",
                    content=content,
                    tool_call_id=raw_msg.get("call_id", raw_msg.get("tool_call_id", "")),
                ))

            elif role == "developer":
                # Developer role in Responses API acts like system
                system = content if not system else system + "\n" + content

            elif role == "previous_response":
                # continuation from a previous response ID
                prev_id = raw_msg.get("previous_response_id")

        tools: list[CanonicalTool] = []
        for tspec in body.get("tools", []) + (body.get("functions", [])):
            tools.append(tool_from_openai(tspec))

        tc = body.get("tool_choice", "auto")

        return CanonicalRequest(
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tc,
            max_tokens=body.get("max_output_tokens", body.get("max_tokens", 4096)),
            temperature=body.get("temperature", 0.0),
            stream=body.get("stream", False),
            previous_response_id=prev_id,
            extra={
                "previous_response_id": prev_id,
                "response_format": body.get("text", {}),
            },
        )

    def encode_nonstream_response(
        self, canonical: CanonicalRequest, body: dict[str, Any]
    ) -> dict[str, Any]:
        # Expects a chat-completion-like body, reshapes into Responses format
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        finish = choice.get("finish_reason", "stop")

        output: list[dict[str, Any]] = []
        if content:
            output.append({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            })

        for raw_tc in msg.get("tool_calls", []):
            output.append({
                "type": "function_call",
                "id": raw_tc.get("id", ""),
                "name": raw_tc.get("function", {}).get("name", ""),
                "arguments": raw_tc.get("function", {}).get("arguments", "{}"),
                "status": "completed",
            })

        status = "completed"
        if finish == "max_tokens":
            status = "incomplete"

        return {
            "id": body.get("id", f"resp_{uuid.uuid4().hex[:16]}"),
            "object": "response",
            "created_at": body.get("created", 0),
            "model": body.get("model", canonical.extra.get("model", "unknown")),
            "status": status,
            "incomplete_details": {"reason": "max_tokens"} if finish == "max_tokens" else None,
            "output": output,
            "usage": {
                "input_tokens": body.get("usage", {}).get("prompt_tokens", 0),
                "output_tokens": body.get("usage", {}).get("completion_tokens", 0),
                "total_tokens": body.get("usage", {}).get("total_tokens", 0),
            },
        }

    def encode_stream_event(self, event: CanonicalEvent) -> str | None:
        """Encode canonical events as OpenAI Responses SSE format.

        Responses uses a specific SSE format with typed events.
        """
        if event.type == "text_delta":
            return self._sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "delta": event.partial,
                "index": event.index,
            })
        if event.type == "text":
            return self._sse("response.output_text.annotated", {
                "type": "response.output_text.annotated",
                "text": event.partial or "",
            })
        if event.type == "tool_use":
            cb = event.content_block
            if cb and cb.tool_call:
                return self._sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "name": cb.tool_call.name,
                    "arguments": json.dumps(cb.tool_call.arguments),
                    "call_id": cb.tool_call.id,
                })
        if event.type == "message_stop":
            return self._sse("response.completed", {
                "type": "response.completed",
            })
        return None

    def parse_tool_result(self, body: dict[str, Any]) -> str:
        # Responses API can carry tool results in output blocks
        output = body.get("output", [])
        for item in output:
            if item.get("type") == "function_call_output":
                return item.get("output", "")
        return body.get("content", "")

    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": body.get("input", []),
            "model": body.get("model", ""),
        }

    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_tokens": backend_body.get("input_tokens", 0),
            "output_tokens": backend_body.get("output_tokens", 0),
        }

    # ── helpers ──────────────────────────────────────────────────────────

    @classmethod
    def _sse(cls, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"