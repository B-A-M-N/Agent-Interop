"""ModelCodec interface — abstract base for upstream protocol codecs.

Every upstream protocol gets its own codec that owns:

- Endpoint path and required headers
- Request rendering (CanonicalRequest → upstream-native dict)
- Response decoding (upstream-native dict → DecodedModelResponse)
- Streaming decoding (upstream-native chunk → DecodedModelEvent)
- Usage conversion, stop-reason conversion, tool-call extraction

Critical rule: Response decoding must preserve raw_arguments exactly as the
upstream sent them. The codec output is DecodedModelResponse, not
CanonicalResponse. The tool-call list is list[RawToolCallCandidate], not
list[CanonicalToolCallBlock].
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalRequest,
    CanonicalStopReason,
    CanonicalTool,
    CanonicalUsage,
    RawToolCallCandidate,
)
from agent_interop.config import UpstreamProtocol


@dataclass
class DecodedModelResponse:
    """Decoded upstream response — before repair and canonical assembly.

    tool_candidates contains RawToolCallCandidate objects. Malformed JSON
    survives in raw_arguments verbatim. stop_reason is always a
    CanonicalStopReason value — never a provider-native string.
    """

    content: list[CanonicalContentBlock] = field(default_factory=list)
    tool_candidates: list[RawToolCallCandidate] = field(default_factory=list)
    stop_reason: CanonicalStopReason = CanonicalStopReason.END_TURN
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodedTextDelta:
    """A text fragment from the upstream."""

    text: str = ""


@dataclass(frozen=True)
class DecodedToolFragment:
    """A streaming tool-call fragment with full identity.

    Identity is (choice_index, tool_index) to correctly accumulate parallel
    and fragmented calls across choices.
    """

    choice_index: int = 0
    tool_index: int = 0
    call_id_fragment: str = ""
    name_fragment: str = ""
    argument_fragment: str = ""


@dataclass(frozen=True)
class DecodedToolCallComplete:
    """Signals completion of a single tool call.

    Corresponds to provider-level events like Anthropic's content_block_stop
    or OpenAI Responses' function_call_arguments.done.
    """

    choice_index: int = 0
    tool_index: int = 0


@dataclass(frozen=True)
class DecodedToolBatchComplete:
    """Signals that all tool calls for a choice are complete."""

    choice_index: int = 0
    stop_reason: CanonicalStopReason = CanonicalStopReason.END_TURN


@dataclass(frozen=True)
class DecodedUsageUpdate:
    """A usage-only update from the upstream."""

    usage: CanonicalUsage = field(default_factory=CanonicalUsage)


@dataclass(frozen=True)
class DecodedStreamComplete:
    """Signals the end of the stream."""

    stop_reason: CanonicalStopReason = CanonicalStopReason.END_TURN
    usage: CanonicalUsage | None = None


@dataclass(frozen=True)
class DecodedStreamError:
    """A stream-level error."""

    error: str = ""
    stop_reason: CanonicalStopReason = CanonicalStopReason.BACKEND_ERROR


# Discriminated union of all decoded stream events
DecodedStreamEvent = (
    DecodedTextDelta
    | DecodedToolFragment
    | DecodedToolCallComplete
    | DecodedToolBatchComplete
    | DecodedUsageUpdate
    | DecodedStreamComplete
    | DecodedStreamError
)

# Legacy alias for backward compatibility during migration
DecodedModelEvent = DecodedStreamEvent


class StreamFraming(str, Enum):
    """How upstream streaming responses are framed."""

    SSE = "sse"
    NDJSON = "ndjson"


@dataclass(frozen=True)
class CodecCapabilities:
    """Declared capabilities of an upstream codec.

    Used by the gateway to determine what features can be requested
    without probing.
    """

    supports_native_tools: bool = True
    supports_streaming: bool = True
    supports_parallel_tool_calls: bool = False
    supports_vision: bool = False
    supports_system_messages: bool = True
    supports_tool_result_images: bool = False
    supports_n_choices: bool = False
    max_tools: int = 128
    streaming_framing: StreamFraming = StreamFraming.SSE


class ModelCodec(ABC):
    """Abstract interface for upstream protocol codecs."""

    protocol: ClassVar[UpstreamProtocol]
    stream_framing: ClassVar[StreamFraming] = StreamFraming.SSE

    @abstractmethod
    def endpoint_path(self) -> str:
        """Return the upstream API endpoint path (e.g. /api/chat)."""
        ...

    @abstractmethod
    def required_headers(self) -> dict[str, str]:
        """Return headers that must be sent with every request."""
        ...

    @abstractmethod
    def render_request(
        self,
        canonical: CanonicalRequest,
        model_name: str,
        stream: bool = True,
    ) -> dict[str, Any]:
        """Render a canonical request to upstream-native format."""
        ...

    @abstractmethod
    def decode_response(
        self,
        body: dict[str, Any],
        tools: list[CanonicalTool] | None = None,
    ) -> DecodedModelResponse:
        """Decode an upstream non-streaming response.

        Returns DecodedModelResponse with tool_candidates containing
        RawToolCallCandidate objects. raw_arguments must be preserved
        verbatim from the upstream response.
        """
        ...

    @abstractmethod
    def decode_stream_chunk(
        self,
        chunk: dict[str, Any],
    ) -> list[DecodedStreamEvent]:
        """Decode a single streaming chunk into decoded stream events.

        Returns a list of events (multiple events per chunk are possible
        for parallel tool calls, text + tool in same chunk, etc.).
        """
        ...

    @abstractmethod
    def extract_usage(self, body: dict[str, Any]) -> CanonicalUsage:
        """Extract usage from a raw upstream response."""
        ...

    def count_tokens(self, messages: list[dict[str, Any]], model: str) -> CanonicalUsage:
        """Count tokens for messages using the backend's tokenizer.

        Default implementation returns estimated counts. Backends should
        override this to provide accurate token counts.
        """
        import json
        # Rough estimation: ~4 chars per token for English text
        total_chars = sum(len(json.dumps(m)) for m in messages)
        estimated = max(1, total_chars // 4)
        return CanonicalUsage(
            input_tokens=estimated,
            output_tokens=0,
            total_tokens=estimated,
            confidence="estimated",
        )

    def extract_stop_reason(self, body: dict[str, Any]) -> CanonicalStopReason:
        """Extract stop reason from a raw upstream response.

        Override if the upstream uses non-standard finish_reason codes.
        Returns a CanonicalStopReason — never a raw provider string.
        """
        finish = body.get("finish_reason") or body.get("done_reason") or "stop"
        stop_map = {
            "stop": CanonicalStopReason.END_TURN,
            "tool_calls": CanonicalStopReason.TOOL_CALL,
            "length": CanonicalStopReason.MAX_TOKENS,
        }
        return stop_map.get(finish, CanonicalStopReason.END_TURN)

    def is_stream_complete(self, chunk: dict[str, Any]) -> bool:
        """Return True if this chunk signals the end of a stream."""
        return bool(chunk.get("done", False))

    def build_repair_request(
        self,
        original_request: dict[str, Any],
        correction_prompt: str,
    ) -> dict[str, Any]:
        """Build a repair/correction request for hidden regeneration.

        Each codec can override this to construct a protocol-native correction
        request that forces the target tool and uses constrained decoding when
        available. The default implementation appends a user message with the
        correction prompt (Chat Completions style).

        Args:
            original_request: The original upstream request body.
            correction_prompt: The correction prompt to send to the model.

        Returns:
            A new request body suitable for sending to the upstream.
        """
        import copy
        correction_body = copy.deepcopy(original_request)
        correction_body["stream"] = False
        # Remove tool-related top-level keys so the model focuses on correction
        correction_body.pop("tools", None)
        correction_body.pop("tool_choice", None)
        # Append correction as a user message
        if "messages" in correction_body:
            correction_body["messages"] = list(correction_body["messages"]) + [
                {"role": "user", "content": correction_prompt}
            ]
        return correction_body

    def probe_endpoint(self) -> str:
        """Return the endpoint path for probing this upstream.

        Override for protocol-specific probe endpoints.
        """
        return "/"

    def capabilities(self) -> CodecCapabilities:
        """Return the capabilities of this codec.

        Override in subclasses to declare protocol-specific limits.
        Default assumes full OpenAI Chat Completions capability.
        """
        return CodecCapabilities(
            supports_native_tools=True,
            supports_streaming=True,
            supports_system_messages=True,
            max_tools=128,
            streaming_framing=self.stream_framing,
        )

    def backend_constraints(self):
        """Return destination constraints derived from capabilities.

        Subclasses may override to add protocol-specific restrictions
        (e.g. stricter name patterns). The base implementation forwards
        the codec's ``max_tools`` cap so pre-upstream validation can reject
        an over-limit tool count for ANY codec, not only those that override
        this method.
        """
        from agent_interop.request_validation import BackendConstraints

        return BackendConstraints(max_tools=self.capabilities().max_tools)

    async def probe(
        self,
        transport: Any,
        route: Any,
    ) -> ProbeResult:
        """Probe the upstream to verify connectivity and discover capabilities.

        Uses the transport layer with proper auth headers.
        Returns a ProbeResult with reachability, auth, and model info.
        """
        from agent_interop.transport.http import PreparedUpstreamRequest

        base = route.upstream.base_url.rstrip("/")
        url = f"{base}{self.probe_endpoint()}"
        headers = {**self.required_headers()}
        # Include auth if configured
        if hasattr(route.upstream, 'api_key_env') and route.upstream.api_key_env:
            import os
            api_key = os.environ.get(route.upstream.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

        req = PreparedUpstreamRequest(
            method="GET",
            url=url,
            headers=headers,
            body={},
            stream=False,
            timeout_seconds=10.0,
        )
        try:
            resp = await transport.send(req)
            return ProbeResult(
                reachable=resp.status_code < 500,
                authenticated=resp.status_code != 401,
                backend_version="",
                available_models=(),
                capabilities=BackendCapabilities(),
            )
        except Exception:
            return ProbeResult(
                reachable=False,
                authenticated=False,
                backend_version="",
                available_models=(),
                capabilities=BackendCapabilities(),
            )


@dataclass(frozen=True)
class BackendCapabilities:
    """Capabilities reported by or probed from a backend."""

    supports_native_tools: bool = False
    supports_streaming: bool = True
    supports_vision: bool = False
    max_context_tokens: int = 0


@dataclass(frozen=True)
class ProbeResult:
    """Result of probing an upstream backend."""

    reachable: bool = False
    authenticated: bool = False
    backend_version: str = ""
    available_models: tuple[str, ...] = ()
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    diagnostics: tuple[str, ...] = ()


def upstream_extra(body: dict[str, Any], model_key: str = "model") -> dict[str, Any]:
    """Extract upstream provider metadata for forwarding in response extra.

    Pulls commonly-available fields (model name, provider, version) from the
    upstream response body and returns a dict suitable for
    ``DecodedModelResponse.extra``.

    Args:
        body: The raw upstream response body.
        model_key: The key in ``body`` that holds the model identifier
                   (``"model"`` for Ollama/OpenAI, ``"id"`` for Anthropic).

    Returns:
        A dict with keys like ``upstream_model``, ``upstream_provider``,
        ``upstream_id`` — only for fields that are present and non-empty.
    """
    result: dict[str, Any] = {}

    model_val = body.get(model_key)
    if model_val:
        result["upstream_model"] = str(model_val)

    # Many upstreams include an "object" or "type" field identifying the provider
    obj_val = body.get("object")
    if obj_val:
        result["upstream_provider"] = str(obj_val)

    # Some providers include a top-level id (Anthropic, OpenAI Responses)
    id_val = body.get("id")
    if id_val:
        result["upstream_id"] = str(id_val)

    return result
