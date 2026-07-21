"""llama.cpp server backend adapter.

llama.cpp exposes an OpenAI-compatible Chat Completions endpoint,
but with its own quirks in the streaming format and tool format.
"""

from __future__ import annotations

import json
from typing import Any

from interop.backends.base import BackendAdapter
from interop.types import BackendEvent, BackendKind, BackendRequest


class LlamacppAdapter(BackendAdapter):
    """Adapter for llama.cpp server's OpenAI-compatible endpoint."""

    kind = BackendKind.LLAMACPP

    def default_port(self) -> int:
        return 8080

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
        **kwargs: Any,
    ) -> BackendRequest:
        body: dict[str, Any] = {
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "n_predict": max_tokens,  # llama.cpp uses n_predict
            "cache_prompt": True,
        }

        if system:
            body["messages"] = [{"role": "system", "content": system}] + body["messages"]

        # llacpp supports OpenAI-format tools
        if tools:
            body["tools"] = tools

        if tool_choice == "none":
            body.pop("tools", None)

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

        done = data.get("stop", False) or data.get("done", False)
        return BackendEvent(data=data, done=done)

    def build_count_tokens_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> BackendRequest:
        return BackendRequest(
            url="",
            headers={"Content-Type": "application/json"},
            body={"messages": messages},
            stream=False,
        )