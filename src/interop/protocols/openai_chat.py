"""OpenAI Chat Completions API adapter (/v1/chat/completions)."""

from __future__ import annotations

import json
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
    tool_to_openai,
)


class OpenAIChatAdapter(ClientProtocolAdapter):
    """Translate OpenAI Chat Completions API to/from canonical form."""

    protocol = ProtocolKind.OPENAI_CHAT
    id = "openai-chat"

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        if path.rstrip("/").endswith("/v1/chat/completions"):
            return True
        # Default if nothing else matches
        return False

    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        messages: list[AgentMessage] = []
        system_content = ""

        for msg in body.get("messages", []):
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, list):
                    system_content = " ".join(
                        b.get("text", "") for b in content if b.get("type") == "text"
                    )
                else:
                    system_content = content
                continue

            if role == "assistant":
                blocks: list[ContentBlock] = []
                tc = []

                # Handle tool_calls in the response
                for raw_tc in msg.get("tool_calls") or []:
                    args_raw = raw_tc.get("function", {}).get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {}
                    tool_call = ToolCall(
                        id=raw_tc.get("id", ""),
                        name=raw_tc.get("function", {}).get("name", ""),
                        arguments=args,
                        raw=args_raw,
                    )
                    tc.append(tool_call)
                    blocks.append(ContentBlock(type="tool_use", tool_call=tool_call))

                if content:
                    blocks.insert(0, ContentBlock(type="text", text=content))

                messages.append(AgentMessage(
                    role="assistant",
                    content=blocks if blocks else content,
                    tool_calls=tc if tc else None,
                ))

            elif role == "user":
                if isinstance(content, list):
                    text_parts: list[str] = []
                    for block in content:
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    messages.append(AgentMessage(
                        role="user",
                        content="\n".join(text_parts),
                    ))
                else:
                    messages.append(AgentMessage(role="user", content=content))

            elif role == "tool":
                messages.append(AgentMessage(
                    role="tool",
                    content=msg.get("content", ""),
                    tool_call_id=msg.get("tool_call_id", ""),
                ))

        tools = []
        for tool_spec in body.get("tools") or body.get("functions") or []:
            tools.append(tool_from_openai(tool_spec))

        tc = body.get("tool_choice", "auto")
        if tc == "none":
            tc = "none"
        elif tc == "auto" or tc == "required":
            pass
        elif isinstance(tc, dict) and tc.get("type") == "function":
            tc = {"name": tc.get("function", {}).get("name", "")}

        return CanonicalRequest(
            system=body.get("system", "") or system_content,
            messages=messages,
            tools=tools,
            tool_choice=tc,
            max_tokens=body.get("max_tokens", 4096),
            temperature=body.get("temperature", 0.0),
            stream=body.get("stream", False),
        )

    def encode_nonstream_response(
        self, canonical: CanonicalRequest, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Assumes body is a raw backend response in chat format."""
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content", "")
        finish_reason = choice.get("finish_reason", "stop")
        tool_calls_raw = msg.get("tool_calls", [])

        response = {
            "id": body.get("id", "interop-chat-unknown"),
            "object": "chat.completion",
            "created": body.get("created", 0),
            "model": body.get("model", canonical.extra.get("model", "unknown")),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": finish_reason,
            }],
            "usage": body.get("usage", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }),
        }

        if tool_calls_raw and not canonical.stream:
            response["choices"][0]["message"]["tool_calls"] = tool_calls_raw

        return response

    def encode_stream_event(self, event: CanonicalEvent) -> str | None:
        if event.type == "text_delta":
            data = {"choices": [{"delta": {"content": event.partial}, "index": 0}]}
            return f"data: {json.dumps(data)}\n\n"
        if event.type == "text":
            return None
        if event.type == "tool_use_delta":
            return None  # tool calls in chat are not streamed as partials
        if event.type == "message_stop":
            data = {
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "tool_calls" if event.content_block else "stop",
                }]
            }
            return f"data: {json.dumps(data)}\n\n"
        return None

    def parse_tool_result(self, body: dict[str, Any]) -> str:
        return body.get("content", "")

    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": body.get("messages", []),
            "model": body.get("model", ""),
        }

    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "object": "token_count",
            "input_tokens": backend_body.get("input_tokens", 0) or backend_body.get("prompt_tokens", 0),
            "output_tokens": backend_body.get("output_tokens", 0) or backend_body.get("completion_tokens", 0),
        }