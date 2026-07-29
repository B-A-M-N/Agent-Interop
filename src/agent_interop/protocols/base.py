"""Base protocol adapters — transforms between wire protocols and canonical types."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import (
    CanonicalError,
    CanonicalEvent,
    CanonicalRequest,
    CanonicalResponse,
    new_request_id,
    new_session_id,
    should_forward_provider_metadata,
)
from agent_interop.errors import EncodedErrorResponse


@dataclass
class StreamEncoderState:
    """Per-response state for stateful stream encoders.

    Tracks content block lifecycle, usage, and terminal event state
    so encoders can emit protocol-correct sequences.  ``failure_pending``
    records whether an ``error`` canonical event has been emitted without
    a corresponding non-error terminal; encoders must suppress the
    ordinary success-style terminal when set and instead emit a
    protocol-correct failure terminal.
    """

    response_id: str = ""
    model: str = ""
    content_block_index: int = 0
    block_started: bool = False
    block_stopped: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    terminal_emitted: bool = False
    text_buffer: str = ""
    failure_pending: bool = False
    pending_error: Any = None
    terminal_was_failure: bool = False
    done_emitted: bool = False


class StreamEncoder:
    """Base class for stateful per-response stream encoders.

    Subclasses implement protocol-specific encoding while maintaining
    content block lifecycle state.
    """

    def __init__(self, response_context: dict[str, Any] | None = None) -> None:
        self.state = StreamEncoderState()
        if response_context:
            self.state.response_id = response_context.get("response_id", "")
            self.state.model = response_context.get("model", "")

    def encode(self, event: CanonicalEvent) -> str | None:
        """Encode a canonical event, updating internal state.

        Returns None if the event should be filtered.
        """
        raise NotImplementedError

    def finish(self) -> str | None:
        """Emit any trailing frames after the stream ends.

        Returns None if no trailer is needed.
        """
        return None


class ClientProtocolAdapter(ABC):
    """Adapts an incoming client protocol into canonical request/response."""

    protocol: str = ""
    id: str = ""

    # Metadata kinds this adapter can forward to clients.
    # Only metadata with kind in this set will be forwarded when the
    # forwarding policy is PRESERVE_IF_COMPATIBLE.
    supported_provider_metadata: frozenset[str] = frozenset()

    # Maximum length for client-supplied IDs (prevents abuse)
    _MAX_ID_LENGTH = 128
    # Only allow alphanumeric, hyphens, underscores in client-supplied IDs
    _ID_RE = re.compile(r'^[a-zA-Z0-9_\-]*$')

    def should_forward_provider_metadata(
        self,
        metadata: Any,
    ) -> bool:
        """Shared decision function for provider metadata forwarding.

        Args:
            metadata: The ProviderMetadata instance to check.

        Returns:
            True if the metadata should be forwarded to this adapter's clients.
        """
        return should_forward_provider_metadata(metadata, self.supported_provider_metadata)

    @abstractmethod
    def decode_request(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> CanonicalRequest:
        """Convert an incoming API request body to canonical form."""
        ...

    def _resolve_request_id(self, headers: dict[str, str], body: dict[str, Any]) -> str:
        """Generate or validate request ID at the ingress boundary.

        Priority: recognized header → body field → newly generated.
        Client-supplied IDs are validated for format and length.
        """
        # Check headers
        for header in ("x-request-id", "x-correlation-id"):
            val = headers.get(header, "")
            val = val.strip()
            if val and len(val) <= self._MAX_ID_LENGTH and self._ID_RE.match(val):
                return val

        # Check body
        for key in ("request_id", "id"):
            val = body.get(key, "")
            if isinstance(val, str) and val and len(val) <= self._MAX_ID_LENGTH and self._ID_RE.match(val):
                return val

        return new_request_id()

    def _resolve_session_id(self, headers: dict[str, str]) -> str:
        """Generate or validate session ID at the ingress boundary."""
        for header in ("x-session-id", "x-claude-code-session-id"):
            val = headers.get(header, "")
            val = val.strip()
            if val and len(val) <= self._MAX_ID_LENGTH and self._ID_RE.match(val):
                return val
        return new_session_id()

    @staticmethod
    def require_string(value: Any, field_name: str) -> str:
        """Require a non-empty string, raising ValueError otherwise."""
        if not isinstance(value, str) or not value:
            raise ValueError(f"'{field_name}' must be a non-empty string")
        return value

    @staticmethod
    def optional_string(value: Any) -> str | None:
        """Coerce to string or None, never 'None'."""
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def require_mapping(value: Any, field_name: str) -> dict:
        """Require a dict/mapping, raising ValueError otherwise."""
        if not isinstance(value, dict):
            raise ValueError(f"'{field_name}' must be an object")
        return value

    @staticmethod
    def validate_max_tokens(value: Any, field_name: str = "max_tokens") -> int:
        """Reject a non-positive-integer token budget.

        Silently passing a negative, zero, or non-numeric value through to
        the backend changes request semantics without telling the client
        anything went wrong — the backend either rejects it opaquely or,
        worse, coerces it into something the client never asked for.
        """
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"'{field_name}' must be a positive integer, got {value!r}")
        return value

    @staticmethod
    def validate_temperature(value: Any, field_name: str = "temperature") -> float:
        """Reject a non-numeric or negative temperature, for the same
        reason as ``validate_max_tokens``: a silently-passed-through
        garbage value is a silent semantic change, not a safe default."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{field_name}' must be a number, got {value!r}")
        if value < 0:
            raise ValueError(f"'{field_name}' must be >= 0, got {value!r}")
        return float(value)

    @staticmethod
    def validate_top_p(value: Any, field_name: str = "top_p") -> float | None:
        """Validate an optional top_p (nucleus sampling) value.

        ``None`` (field omitted) means "not requested" and is passed
        through unchanged — top_p, unlike temperature, has no universal
        default the gateway can safely assume on the client's behalf.
        """
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{field_name}' must be a number, got {value!r}")
        if not (0 <= value <= 1):
            raise ValueError(f"'{field_name}' must be between 0 and 1, got {value!r}")
        return float(value)

    @staticmethod
    def validate_stop(value: Any, field_name: str = "stop") -> list[str]:
        """Normalize a stop-sequence field (bare string or list of strings)
        into the canonical ``list[str]`` shape. An absent field yields an
        empty list — never an error, since "no stop sequences" is a
        perfectly ordinary request."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            if not all(isinstance(v, str) for v in value):
                raise ValueError(f"'{field_name}' must be a string or list of strings, got {value!r}")
            return value
        raise ValueError(f"'{field_name}' must be a string or list of strings, got {value!r}")

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        """Return True if this adapter should handle the given request."""
        return False

    @abstractmethod
    def encode_response(
        self,
        response: CanonicalResponse,
    ) -> dict[str, Any]:
        """Convert a canonical response to the client protocol format."""
        ...

    @abstractmethod
    def encode_error(
        self,
        error: CanonicalError,
    ) -> EncodedErrorResponse:
        """Convert a canonical error to the client protocol format."""
        ...

    @abstractmethod
    def encode_event(
        self,
        event: CanonicalEvent,
    ) -> str | None:
        """Convert a canonical stream event into a protocol-specific SSE string.
        Return None if this event should be filtered (not sent to the client)."""

    def create_stream_encoder(
        self,
        response_context: dict[str, Any] | None = None,
    ) -> StreamEncoder:
        """Create a per-response stateful stream encoder.

        Override in subclasses for protocol-specific lifecycle tracking.
        Default uses stateless encode_event.
        """
        return _StatelessAdapterEncoder(self, response_context)

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


class _StatelessAdapterEncoder(StreamEncoder):
    """Wraps a stateless adapter's encode_event for the StreamEncoder interface."""

    def __init__(self, adapter: ClientProtocolAdapter, response_context: dict[str, Any] | None = None) -> None:
        super().__init__(response_context)
        self._adapter = adapter

    def encode(self, event: CanonicalEvent) -> str | None:
        return self._adapter.encode_event(event)