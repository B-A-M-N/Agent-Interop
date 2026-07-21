"""Base protocol adapters — transforms between wire protocols and canonical types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from interop.types import (
    BackendRequest,
    CanonicalEvent,
    CanonicalRequest,
    ProtocolKind,
)


class ClientProtocolAdapter(ABC):
    """Adapts an incoming client protocol into canonical request/response."""

    protocol: ProtocolKind
    id: str = ""

    @abstractmethod
    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        """Convert an incoming API request body to canonical form."""
        ...

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        """Return True if this adapter should handle the given request."""
        return False

    @abstractmethod
    def encode_nonstream_response(
        self, canonical: CanonicalRequest, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert a non-streaming backend response body into the client protocol format."""
        ...

    @abstractmethod
    def encode_stream_event(self, event: CanonicalEvent) -> str | None:
        """Convert a canonical stream event into a protocol-specific SSE string.
        Return None if this event should be filtered (not sent to the client)."""
        ...

    def encode_stream_done(self) -> str:
        """Return the SSE 'done' signal for this protocol."""
        return "data: [DONE]\n\n"

    @abstractmethod
    def parse_tool_result(self, body: dict[str, Any]) -> str:
        """Extract tool result content from a tool_result message body."""
        ...

    @abstractmethod
    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        """Convert a count_tokens request for the backend."""
        ...

    @abstractmethod
    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        """Convert a backend token count response to client format."""
        ...