"""vLLM backend adapter.

vLLM is the most capable local backend for agent tool calling — it
implements explicit tool-parser plugins, structured decoding, and
native Anthropic Messages API + OpenAI Responses API support.
"""

from __future__ import annotations

import json
from typing import Any

from interop.backends.base import BackendAdapter
from interop.types import BackendEvent, BackendKind, BackendRequest


class VLLMAdapter(BackendAdapter):
    """Adapter for vLLM's OpenAI-compatible endpoint.

    vLLM supports tool calling natively with model-specific parsers.
    We pass tool definitions and let vLLM handle parsing. The adapter
    manages the additional vLLM-specific knobs (guided decoding, tool_parser).
    """

    kind = BackendKind.VLLM

    def default_port(self) -> int:
        return 8000

    def build_request(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict = "auto",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        stream: bool = True,
        guided_decoding: bool | None = None,
        tool_parser: str | None = None,
        **kwargs: Any,
    ) -> BackendRequest:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if system:
            body["messages"] = [{"role": "system", "content": system}] + body["messages"]

        if tools:
            body["tools"] = tools
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "none", "required"):
                pass
            elif isinstance(tool_choice, dict) and "name" in tool_choice:
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            else:
                body["tool_choice"] = tool_choice

        # vLLM-specific: tool_parser for model-native tool call extraction
        if tool_parser:
            body["tool_parser"] = tool_parser

        # guided decoding for structured output
        if guided_decoding:
            body["guided_decoding"] = {"backend": "outlines"}

        return BackendRequest(
            url="",
            headers={"Content-Type": "application/json"},
            body=body,
            stream=stream,
        )

    def decode_event(self, raw: str) -> BackendEvent | list[BackendEvent]:
        line = raw.strip()
        if not line:
            return BackendEvent(done=False)
        if line == "data: [DONE]":
            return BackendEvent(done=True)

        if line.startswith("data: "):
            line = line[6:]

        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return BackendEvent(raw=line, done=False)

        done = False
        choices = data.get("choices", [{}])
        if choices:
            finish = choices[0].get("finish_reason")
            if finish and finish != "null":
                done = True

        return BackendEvent(data=data, done=done)

    def build_count_tokens_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> BackendRequest:
        return BackendRequest(
            url=f"/v1/tokenize",
            headers={"Content-Type": "application/json"},
            body={"model": model, "messages": messages},
            stream=False,
        )