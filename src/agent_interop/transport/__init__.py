"""Transport layer — HTTP client, SSE decoder, NDJSON decoder.

Separates HTTP transport concerns from protocol codecs.
"""

from __future__ import annotations

from agent_interop.transport.http import (
    PreparedUpstreamRequest,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTransport,
)
from agent_interop.transport.ndjson import NDJSONDecoder
from agent_interop.transport.sse import SSEDecoder, SSEFrame

__all__ = [
    "NDJSONDecoder",
    "PreparedUpstreamRequest",
    "SSEDecoder",
    "SSEFrame",
    "UpstreamResponse",
    "UpstreamStream",
    "UpstreamTransport",
]