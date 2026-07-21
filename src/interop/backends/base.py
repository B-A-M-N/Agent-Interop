"""Backend adapters — forward normalized requests to inference backends.

Each backend adapter knows how to construct the correct HTTP request
for its backend, and how to decode streaming events into BackendEvents.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from interop.types import BackendKind, BackendRequest, BackendEvent


class BackendAdapter(ABC):
    """Abstract base for inference backend adapters."""

    kind: BackendKind

    @abstractmethod
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
        """Build a fully resolved HTTP request for this backend."""
        ...

    @abstractmethod
    def decode_event(self, raw: str) -> BackendEvent | list[BackendEvent]:
        """Decode a single raw SSE line into one or more backend events."""
        ...

    @abstractmethod
    def build_count_tokens_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> BackendRequest:
        ...

    def default_port(self) -> int:
        return 11434