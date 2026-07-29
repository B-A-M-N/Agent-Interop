"""Ollama admin client — model management, discovery, and embeddings.

Salvaged from the dead ``interop.backends.ollama.OllamaAdapter`` (which also
carried generation methods that are now superseded by the codec/transport
layer). This module keeps only the admin / model-management operations that
are genuinely useful but currently unreachable: listing, showing, pulling,
pushing, creating, copying, and deleting models, plus embedding generation.

This is a standalone class — it does NOT depend on
``interop.backends.base.BackendAdapter`` or the v1 compat types
(``BackendEvent`` / ``BackendKind`` / ``BackendRequest``).

Ollama native API endpoints covered:
    /api/tags  /api/show  /api/ps  /api/pull  /api/push  /api/create
    /api/delete  /api/copy  /api/embed  /api/embeddings
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

logger = logging.getLogger("agent_interop.backends.ollama_admin")


class OllamaAdminClient:
    """Stateless Ollama admin client — native model-management API.

    Each method takes a ``base_url`` and preserves the exact HTTP behavior
    (endpoints, timeouts, streaming) of the original adapter.
    """

    # ─── Discovery ──────────────────────────────────────────────────────────

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

    # ─── Model management ───────────────────────────────────────────────────

    async def pull_model(
        self, base_url: str, model: str,
        stream: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        """Pull/download a model. Streams progress if stream=True."""
        async with httpx.AsyncClient() as client, client.stream(
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
        async with httpx.AsyncClient() as client, client.stream(
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

        async with httpx.AsyncClient() as client, client.stream(
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
            await client.request(  # type: ignore[call-arg]
                "DELETE",
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

    # ─── Embeddings ─────────────────────────────────────────────────────────

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
