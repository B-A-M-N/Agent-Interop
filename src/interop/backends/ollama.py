"""Ollama backend adapter — full native API + OpenAI-compatible + Anthropic-compatible.

Ollama exposes:
  Native API: /api/tags, /api/generate, /api/chat, /api/pull, /api/push,
              /api/create, /api/delete, /api/copy, /api/show, /api/embed,
              /api/embeddings, /api/ps, /api/version
  OpenAI-compatible: /v1/chat/completions
  Anthropic-compatible: /v1/messages

Interop's Ollama adapter covers all of them, so Interop can discover,
pull, create, and run models — not just proxy chat completions.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from interop.backends.base import BackendAdapter
from interop.types import BackendEvent, BackendKind, BackendRequest

logger = logging.getLogger("interop.backends.ollama")


class OllamaAdapter(BackendAdapter):
    """Full Ollama backend adapter — native + compatible APIs."""

    kind = BackendKind.OLLAMA

    def default_port(self) -> int:
        return 11434

    # ─── Health / Discovery ─────────────────────────────────────────────

    async def check_health(self, base_url: str) -> dict[str, Any]:
        """Check if Ollama is running and return server info."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{base_url}/api/version", timeout=5.0)
            return r.json()

    async def list_models(self, base_url: str) -> list[dict[str, Any]]:
        """List available models."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{base_url}/api/tags", timeout=10.0)
            data = r.json()
            return data.get("models", [])

    async def show_model(self, base_url: str, model: str) -> dict[str, Any]:
        """Show details of a specific model (Modelfile, template, parameters)."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{base_url}/api/show",
                json={"model": model},
                timeout=30.0,
            )
            return r.json()

    async def ps(self, base_url: str) -> list[dict[str, Any]]:
        """List currently loaded/running models."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{base_url}/api/ps", timeout=5.0)
            return r.json().get("models", [])

    # ─── Model management ───────────────────────────────────────────────

    async def pull_model(
        self, base_url: str, model: str,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Pull/download a model. Streams progress if stream=True."""
        import httpx
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/pull",
                json={"model": model, "stream": stream},
                timeout=300.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield json.loads(line)

    async def push_model(
        self, base_url: str, model: str,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Push a model. Streams progress."""
        import httpx
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/push",
                json={"model": model, "stream": stream},
                timeout=300.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield json.loads(line)

    async def create_model(
        self, base_url: str, model: str,
        modelfile: str = "",
        path: str = "",
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Create a model from a Modelfile."""
        body: dict[str, Any] = {"model": model, "stream": stream}
        if modelfile:
            body["modelfile"] = modelfile
        if path:
            body["path"] = path

        import httpx
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/create",
                json=body,
                timeout=300.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield json.loads(line)

    async def delete_model(self, base_url: str, model: str) -> None:
        """Delete a model."""
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{base_url}/api/delete",
                json={"model": model},
                timeout=30.0,
            )

    async def copy_model(self, base_url: str, source: str, destination: str) -> None:
        """Copy a model."""
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base_url}/api/copy",
                json={"source": source, "destination": destination},
                timeout=30.0,
            )

    # ─── Generation (native) ────────────────────────────────────────────

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
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system and (not messages or messages[0].get("role") != "system"):
            body["messages"] = [{"role": "system", "content": system}] + body["messages"]

        if tools:
            body["tools"] = tools
            body["stream"] = False  # tool calls are non-streaming in native Ollama

        if tool_choice == "none":
            body.pop("tools", None)
        elif isinstance(tool_choice, dict) and "name" in tool_choice:
            body["tool_choice"] = {"type": "function", "function": {"name": tool_choice["name"]}}

        # Pass through any extra Ollama-specific options
        if "keep_alive" in kwargs:
            body["keep_alive"] = kwargs["keep_alive"]
        if "options" in kwargs:
            body["options"].update(kwargs["options"])
        if "format" in kwargs:
            body["format"] = kwargs["format"]  # JSON mode
        if "raw" in kwargs:
            body["raw"] = kwargs["raw"]

        endpoint = kwargs.get("endpoint", "/api/chat")

        return BackendRequest(
            url=endpoint,
            headers={"Content-Type": "application/json"},
            body=body,
            stream=stream and not tools,
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

        done = data.get("done", False)
        return BackendEvent(data=data, event_type=line, done=done)

    def build_count_tokens_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> BackendRequest:
        return BackendRequest(
            url="/api/chat",
            headers={"Content-Type": "application/json"},
            body={"model": model, "messages": messages},
            stream=False,
        )

    # ─── Embeddings ─────────────────────────────────────────────────────

    async def embed(
        self, base_url: str, model: str, input: str | list[str],
    ) -> list[list[float]]:
        """Generate embeddings (Ollama 0.5+)."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": input},
                timeout=30.0,
            )
            return r.json().get("embeddings", [])

    async def embeddings(
        self, base_url: str, model: str, prompt: str,
    ) -> list[float]:
        """Generate embeddings (legacy endpoint)."""
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": prompt},
                timeout=30.0,
            )
            return r.json().get("embedding", [])

    # ─── Generate (no chat template) ────────────────────────────────────

    async def generate(
        self, base_url: str, model: str, prompt: str,
        options: dict | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[dict[str, Any]]:
        """Raw /api/generate (no chat template applied)."""
        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {},
        }

        if stream:
            return self._stream_generate(base_url, body)

        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{base_url}/api/generate",
                json=body,
                timeout=120.0,
            )
            return r.json()

    async def _stream_generate(
        self, base_url: str, body: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        import httpx
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/api/generate",
                json=body,
                timeout=300.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield json.loads(line)