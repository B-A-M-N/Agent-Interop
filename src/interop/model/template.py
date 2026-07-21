"""Chat template rendering — renders messages into the format each model expects."""

from __future__ import annotations

from typing import Any

from interop.types import AgentMessage, CanonicalTool, ModelProfile


def render_messages(
    system: str,
    messages: list[AgentMessage],
    tools: list[CanonicalTool],
    profile: ModelProfile,
) -> list[dict[str, Any]]:
    """Render canonical messages into the chat format expected by this model.

    Applies the model's chat template, wraps tool definitions,
    and returns a list of dicts suitable for the backend request.
    """
    result: list[dict[str, Any]] = []

    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        entry: dict[str, Any] = {"role": msg.role}

        if msg.role == "assistant":
            if isinstance(msg.content, list):
                text_parts: list[str] = []
                for block in msg.content:
                    if block.type == "text" and block.text:
                        text_parts.append(block.text)
                    elif block.type == "thinking" and block.text:
                        text_parts.append(block.text)
                entry["content"] = "\n".join(text_parts)
            else:
                entry["content"] = msg.content

            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": _render_args(tc.arguments),
                        },
                    }
                    for tc in msg.tool_calls
                ]

        elif msg.role == "tool":
            entry["role"] = "tool"
            entry["content"] = msg.content
            if msg.tool_call_id:
                entry["tool_call_id"] = msg.tool_call_id

        elif msg.role == "user":
            entry["content"] = msg.content

        result.append(entry)

    return result


def render_tools(tools: list[CanonicalTool]) -> list[dict[str, Any]]:
    """Render canonical tools to OpenAI function schema for the backend."""
    return [t.to_json_schema() for t in tools]


def _render_args(args: dict[str, Any]) -> str:
    """Render tool arguments as a JSON string."""
    import json
    return json.dumps(args)