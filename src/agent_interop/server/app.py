"""FastAPI server for the Interop gateway.

Exposes:
- /v1/messages (Anthropic Messages API)
- /v1/messages/count_tokens
- /v1/chat/completions (OpenAI Chat)
- /v1/responses (OpenAI Responses)
- /v1/health
- /v1/models
- /v1/capabilities

All endpoints normalize incoming requests into canonical form,
process through the gateway, and translate responses back to the
original protocol format.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from agent_interop import __version__
from agent_interop.abi import (
    CanonicalError,
    CanonicalEvent,
    CanonicalResponse,
    CanonicalStopReason,
    ProtocolKind,
)
from agent_interop.capabilities import (
    CapabilityEntry,
    CapabilityState,
    EffectiveCapabilities,
    ToolCapabilityLevel,
    compute_compatibility,
)
from agent_interop.config import InteropServerConfig, validate_config
from agent_interop.context import RequestContext
from agent_interop.errors import InteropErrorCode
from agent_interop.gateway import Gateway
from agent_interop.protocols.base import ClientProtocolAdapter
from agent_interop.protocols.registry import detect_adapter, get_adapter
from agent_interop.upstreams.registry import get_codec

logger = logging.getLogger("agent_interop.server")


class BodyParseError(Exception):
    """Raised when the request body is too large or not valid JSON.

    Carries enough information for the caller to encode a protocol-correct
    error response in either the JSON or SSE shape.
    """

    def __init__(
        self,
        message: str,
        *,
        wants_stream: bool = False,
        code: str = "INVALID_JSON",
        status: int = 400,
    ) -> None:
        super().__init__(message)
        self.wants_stream = wants_stream
        self.code = code
        self.status = status


def _sniff_wants_stream(raw: bytes) -> bool:
    """Best-effort detection of stream intent from a raw (possibly malformed) body.

    Used for size-exceeded and malformed-JSON rejections where the body cannot
    be parsed, so the endpoint cannot inspect ``body.get("stream")`` directly.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return bool(re.search(rb'"stream"\s*:\s*(?:true|\d)', raw))
    if not isinstance(parsed, dict):
        return False
    return bool(parsed.get("stream", False))


async def _read_and_parse_body(
    request: Request,
    config: InteropServerConfig,
) -> dict[str, Any]:
    """Read and parse the request body with a size guard (Bug 1 + Bug 3).

    Enforces ``config.max_request_bytes`` BEFORE the oversized allocation
    happens, not after: a declared ``Content-Length`` over the limit is
    rejected without reading any body bytes, and the body is otherwise read
    incrementally via ``request.stream()`` so a client with no (or a lying)
    Content-Length — e.g. chunked transfer encoding — still can't force an
    unbounded buffer. Raises :class:`BodyParseError` (never a generic
    Starlette 400) on failure so callers can encode a protocol-correct error.
    """
    limit = config.max_request_bytes

    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError:
            declared = None
        if declared is not None and declared > limit:
            raise BodyParseError(
                f"Request body size {declared} exceeds the allowed limit of "
                f"{limit} bytes",
                wants_stream=False,
                code="REQUEST_TOO_LARGE",
                status=413,
            )

    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > limit:
            wants_stream = _sniff_wants_stream(b"".join(chunks))
            raise BodyParseError(
                f"Request body size exceeds the allowed limit of {limit} bytes",
                wants_stream=wants_stream,
                code="REQUEST_TOO_LARGE",
                status=413,
            )
        chunks.append(chunk)
    raw = b"".join(chunks)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        wants_stream = _sniff_wants_stream(raw)
        raise BodyParseError(
            f"Invalid JSON body: {exc}",
            wants_stream=wants_stream,
            code="INVALID_JSON",
            status=400,
        )

    if not isinstance(parsed, dict):
        # json.loads() accepts any top-level JSON value (array, string,
        # number, bool, null) — every endpoint immediately calls
        # `body.get(...)`, so a non-object body reached that as a bare
        # AttributeError (an unhandled 500) instead of a protocol-correct
        # client error.
        wants_stream = _sniff_wants_stream(raw)
        raise BodyParseError(
            f"Request body must be a JSON object, got {type(parsed).__name__}",
            wants_stream=wants_stream,
            code="INVALID_REQUEST",
            status=400,
        )
    return parsed


_STREAM_RESPONSE_ID_PREFIX = {
    ProtocolKind.ANTHROPIC_MESSAGES: "msg",
    ProtocolKind.OPENAI_CHAT: "chatcmpl",
    ProtocolKind.OPENAI_RESPONSES: "resp",
}


def _generate_stream_response_id(protocol: ProtocolKind) -> str:
    """Generate a distinct response ID for one streamed response.

    Must NOT reuse the client's inbound request ID — a response ID
    identifies this particular model response, not the request that
    produced it, and reusing the request ID makes the two impossible to
    tell apart in logs/evidence. Stable across every frame of the stream
    (the encoder is seeded with it once, at construction).
    """
    prefix = _STREAM_RESPONSE_ID_PREFIX.get(protocol, "resp")
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _encode_failure(encoder: Any, error: CanonicalError) -> list[str]:
    """Feed a canonical error through a stream encoder and return SSE frames.

    Emits the protocol-visible error frame, the failure terminal, and the
    terminal sentinel — the same sequences the encoder produces for any
    in-band error, so callers never hand-roll ``data:`` lines (Bug 6).
    """
    frames: list[str] = []
    out = encoder.encode(CanonicalEvent(type="error", error=error))
    if out:
        frames.append(out)
    stop = encoder.encode(CanonicalEvent(
        type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT
    ))
    if stop:
        frames.append(stop)
    trailer = encoder.finish()
    if trailer:
        frames.append(trailer)
    return frames


def _streaming_error_sse(
    protocol: ProtocolKind,
    error: CanonicalError,
) -> StreamingResponse:
    """Build a streaming error response using a minimal stream encoder.

    Used for pre-parse errors (malformed/over-size body on a streaming
    request) where no route or model is resolved yet. The encoder needs
    no resolved model/route — only protocol-level info (Bug 6 Sites A/B).
    """
    adapter = get_adapter(protocol)
    encoder = adapter.create_stream_encoder({})
    frames = _encode_failure(encoder, error)

    async def gen() -> AsyncIterator[str]:
        for frame in frames:
            yield frame

    return StreamingResponse(gen(), media_type="text/event-stream")


def _handle_body_parse_error(
    protocol: ProtocolKind,
    exc: BodyParseError,
) -> JSONResponse | StreamingResponse:
    """Turn a BodyParseError into the right protocol-encoded response."""
    error = CanonicalError(code=exc.code, message=str(exc))
    if exc.wants_stream:
        return _streaming_error_sse(protocol, error)
    adapter = get_adapter(protocol)
    encoded = adapter.encode_error(error)
    return JSONResponse(encoded.body, status_code=exc.status)


def _encode_gateway_result(
    adapter: ClientProtocolAdapter,
    result: CanonicalResponse,
) -> JSONResponse:
    """Encode a gateway result, routing errors through ``encode_error``.

    The gateway returns a ``CanonicalResponse`` whose ``.error`` field is set
    for any non-success outcome (backend failure, unsafe history, rejected
    tool batch, invalid upstream JSON, ...). ``encode_response`` ignores that
    field, so an unconditional ``encode_response`` call would serialize the
    error as an HTTP 200 success envelope. We inspect ``.error`` first and,
    when present, delegate to ``encode_error`` so the client receives a
    protocol-correct non-200 error response (P0.1).
    """
    if result.error is not None:
        encoded = adapter.encode_error(result.error)
        return JSONResponse(
            encoded.body,
            status_code=encoded.status_code,
            headers=encoded.headers or None,
        )
    return JSONResponse(adapter.encode_response(result))


def create_app_from_env() -> FastAPI:
    """Create the FastAPI application from environment variables.

    Used by the managed launcher (``interop run``) to start a Gateway
    configured entirely from environment variables:

    - INTEROP_BACKEND_URL: upstream base URL
    - INTEROP_BACKEND_TYPE: upstream kind (ollama, vllm, etc.)
    - INTEROP_MODEL: model name
    - INTEROP_PORT: port to bind
    - INTEROP_SESSION_CREDENTIAL: session token for ingress auth
    - INTEROP_DEFAULT_ROUTE: optional route ID (defaults to the only route)
    """
    import os

    from agent_interop.config import (
        InteropServerConfig,
        ModelRoute,
        RepairConfig,
        ToolMode,
        TranslationMode,
        UpstreamConfig,
        UpstreamKind,
        validate_config,
    )

    backend_url = os.environ.get("INTEROP_BACKEND_URL", "http://127.0.0.1:11434")
    backend_type = os.environ.get("INTEROP_BACKEND_TYPE", "ollama")
    model = os.environ.get("INTEROP_MODEL", "qwen3-coder")
    port = int(os.environ.get("INTEROP_PORT", "8090"))
    session_credential = os.environ.get("INTEROP_SESSION_CREDENTIAL", "")
    default_route = os.environ.get("INTEROP_DEFAULT_ROUTE", "default")

    # Map backend kind to upstream kind and its default wire protocol —
    # via the single shared mapping (config.default_wire_protocol_for_kind)
    # rather than an independent copy that can drift from it (and from the
    # YAML config loader's own default, which previously didn't vary by
    # kind at all).
    from agent_interop.config import default_wire_protocol_for_kind

    try:
        kind = UpstreamKind(backend_type)
    except ValueError:
        raise ValueError(
            f"Unknown INTEROP_BACKEND_TYPE: {backend_type!r} "
            f"(supported: {', '.join(k.value for k in UpstreamKind)})"
        ) from None
    wire_protocol = default_wire_protocol_for_kind(kind)

    # Build ingress auth from session credential
    ingress_auth: dict[str, str] = {}
    if session_credential:
        ingress_auth = {"mode": "session_token", "token": session_credential}

    config = InteropServerConfig(
        host="127.0.0.1",
        port=port,
        log_level="info",
        probe_on_startup=True,
        default_route_id=default_route,
        routes={
            default_route: ModelRoute(
                id=default_route,
                # Claude Code requires a "claude"/"anthropic"-prefixed model
                # ID for gateway discovery (see agents/claude_code.py) — the
                # raw upstream model name alone never satisfies that, so a
                # claude-prefixed alias must always be registered alongside
                # it for the managed `interop run claude` launch path.
                client_model_aliases=[model, f"claude-interop-{model}"],
                upstream_model=model,
                upstream=UpstreamConfig(
                    kind=kind,
                    base_url=backend_url,
                    wire_protocol=wire_protocol,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
                repair=RepairConfig(),
            ),
        },
        ingress_auth=ingress_auth,
    )

    # Validate config before starting
    issues = validate_config(config)
    if issues:
        raise ValueError(f"Invalid configuration: {'; '.join(issues)}")

    return create_app(config)


def create_app(config: InteropServerConfig | None = None, *, allow_invalid: bool = False) -> FastAPI:
    """Create the FastAPI application with the given route-based config.

    Validates ``server_config`` with ``validate_config`` and raises
    ``ValueError`` on any issue, BEFORE constructing anything else. CLI
    commands (``deploy``, ``check``) already validated before this point,
    but embedding applications or any other direct caller of
    ``create_app``/``Gateway`` bypassed validation entirely — this is the
    actual construction boundary, not just the CLI's own call sites.
    ``allow_invalid=True`` exists strictly for tests that intentionally
    probe invalid-config behavior; production call sites must never pass it.
    """
    from agent_interop.repair.telemetry import RepairTelemetry
    from agent_interop.session import SessionManager

    server_config = config or InteropServerConfig()
    if not allow_invalid:
        issues = validate_config(server_config)
        if issues:
            raise ValueError(
                "Invalid InteropServerConfig:\n" + "\n".join(f"  - {i}" for i in issues)
            )
    _session_manager = SessionManager()
    _telemetry = RepairTelemetry()

    # Opt-in only: construct an EvidenceStore ONLY when explicitly configured.
    # When ``config.evidence`` is None/disabled, no store is ever created and
    # the Gateway gets its safe default (None) — byte-for-byte identical to
    # before. This preserves the "never silently enable persistent state"
    # principle.
    _evidence_store = None
    if server_config.evidence is not None and server_config.evidence.enabled:
        from agent_interop.evidence.store import EvidenceStore
        _evidence_store = EvidenceStore(db_path=server_config.evidence.db_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        g = Gateway(
            server_config,
            session_manager=_session_manager,
            telemetry=_telemetry,
            evidence_store=_evidence_store,
        )
        try:
            # startup() is inside the try so that BOTH a startup failure and
            # a cancellation mid-yield fall through to the finally block —
            # close() always runs, so the transport (and any evidence store)
            # is never leaked.
            await g.startup()
            app.state.gateway = g
            logger.info(
                "interop server ready — %s:%d (routes=%d)",
                server_config.host,
                server_config.port,
                len(server_config.routes),
            )
            yield
        finally:
            await g.close()
            # The app owns the evidence store (the Gateway does not), so the
            # app must close it — in the same finally block so cleanup runs
            # even if startup() raised or the lifespan body is cancelled
            # mid-yield.
            if _evidence_store is not None:
                _evidence_store.close()

    app = FastAPI(
        title="Interop — Agent Compatibility Gateway",
        version=__version__,
        lifespan=lifespan,
    )

    # ─── Ingress Authentication Middleware ─────────────────────────────

    def _auth_error_response(
        request: Request, code: str, message: str, *, bearer_challenge: bool = False,
    ) -> JSONResponse:
        """Encode an ingress auth failure in the SAME shape the target
        protocol's own error responses use, instead of a generic
        {"error": "..."} body a client's error-handling code won't
        recognize. Detected from the request path — matches whichever of
        /v1/messages, /v1/chat/completions, /v1/responses was hit.
        """
        from agent_interop.abi import CanonicalError

        adapter = detect_adapter(request.url.path, dict(request.headers), {})
        if adapter is None:
            adapter = get_adapter(ProtocolKind.OPENAI_CHAT)
        error = CanonicalError(code=code, message=message)
        encoded = adapter.encode_error(error)
        headers = {"WWW-Authenticate": "Bearer"} if bearer_challenge else {}
        return JSONResponse(encoded.body, status_code=encoded.status_code, headers=headers)

    @app.middleware("http")
    async def ingress_auth_middleware(request: Request, call_next):
        """Enforce ingress authentication based on server config.

        Modes:
        - none_loopback: Only allow loopback connections
        - session_token: Require a valid session token
        - static_token: Require a valid static bearer token
        """
        import secrets

        auth_config = server_config.ingress_auth
        mode = auth_config.get("mode", "none_loopback")

        if mode == "none_loopback":
            # Verify the connection is from loopback
            client_host = request.client.host if request.client else ""
            if client_host not in ("127.0.0.1", "::1", "localhost"):
                return _auth_error_response(
                    request, InteropErrorCode.INGRESS_FORBIDDEN,
                    "Only loopback connections are allowed",
                )
        elif mode == "session_token" or mode == "static_token":
            expected_token = auth_config.get("token", "")
            auth_header = request.headers.get("authorization", "")
            # Extract bearer token
            if auth_header.startswith("Bearer "):
                provided_token = auth_header[7:]
            else:
                provided_token = ""

            if not expected_token or not secrets.compare_digest(
                provided_token, expected_token
            ):
                return _auth_error_response(
                    request, InteropErrorCode.INGRESS_AUTH_FAILED,
                    "Invalid or missing bearer token",
                    bearer_challenge=True,
                )
        else:
            # Fail-closed: an unrecognized/typo'd/future mode must never
            # silently allow traffic through with zero authentication.
            return _auth_error_response(
                request, InteropErrorCode.INGRESS_AUTH_FAILED,
                "Unrecognized ingress auth mode",
                bearer_challenge=True,
            )

        response = await call_next(request)
        return response

    def get_gateway(request: Request) -> Gateway:
        return request.app.state.gateway

    # ─── Centralized request handler ─────────────────────────────────────

    async def _handle_request(
        body: dict[str, Any],
        request: Request,
        protocol: ProtocolKind,
    ) -> JSONResponse:
        """Centralized non-streaming request handler.

        Each endpoint delegates here after reading the body via
        ``_read_and_parse_body``. This ensures consistent error handling and
        prevents per-endpoint drift.
        """
        gw = get_gateway(request)
        adapter = get_adapter(protocol)

        try:
            canonical = adapter.decode_request(body, dict(request.headers))
        except Exception as exc:
            encoded = adapter.encode_error(
                CanonicalError(code="INVALID_REQUEST", message=str(exc))
            )
            return JSONResponse(encoded.body, status_code=422)

        # Build RequestContext from request headers (P0.1 contract)
        context = RequestContext.from_headers(
            dict(request.headers), protocol=protocol
        )

        try:
            result = await gw.handle_request(canonical, context)
        except ValueError as exc:
            encoded = adapter.encode_error(
                CanonicalError(code="INVALID_REQUEST", message=str(exc))
            )
            return JSONResponse(encoded.body, status_code=400)
        except Exception as exc:
            logger.exception("Gateway handle_request failed")
            error_encoded = adapter.encode_error(
                CanonicalError(code="INTERNAL_ERROR", message=str(exc))
            )
            return JSONResponse(error_encoded.body, status_code=500)

        return _encode_gateway_result(adapter, result)

    async def _handle_stream(
        body: dict[str, Any],
        request: Request,
        protocol: ProtocolKind,
    ) -> StreamingResponse:
        """Centralized streaming request handler with error handling."""
        gw = get_gateway(request)
        adapter = get_adapter(protocol)

        try:
            canonical = adapter.decode_request(body, dict(request.headers))
        except Exception as exc:
            return _streaming_error_sse(
                protocol,
                CanonicalError(code="INVALID_REQUEST", message=str(exc)),
            )

        async def event_stream() -> AsyncIterator[str]:
            encoder = None
            try:
                # Use stateful stream encoder for protocol-correct event
                # sequences (content block lifecycle, stable IDs, terminal handling)
                encoder = adapter.create_stream_encoder({
                    "response_id": _generate_stream_response_id(protocol),
                    "model": canonical.model.requested_name if canonical.model else "",
                })
                # Anthropic Messages streaming requires message_start as the
                # first SSE event; OpenAI Responses expects an equivalent
                # response.created. The Gateway's event stream starts from
                # content, not from this protocol-level opener, so it is
                # synthesized here once, before any Gateway events.
                start_frame = encoder.encode(CanonicalEvent(type="message_start"))
                if start_frame:
                    yield start_frame
                # Build RequestContext from request headers (P0.1 contract)
                context = RequestContext.from_headers(
                    dict(request.headers), protocol=protocol
                )
                # aclosing() guarantees the Gateway generator is closed (via
                # GeneratorExit) as soon as we stop consuming it — whether we
                # break on message_stop, the client disconnects, or an
                # exception propagates. Without it, breaking early leaves the
                # generator suspended mid-yield, still holding its upstream
                # `async with transport.stream(...)` connection open until
                # garbage collection eventually finalizes it.
                async with aclosing(gw.handle_stream(canonical, context)) as events:
                    async for event in events:
                        sse = encoder.encode(event)
                        if sse:
                            yield sse
                        if event.type == "message_stop":
                            break
                # Emit trailing frames (content block stops, final usage, done signal)
                trailer = encoder.finish()
                if trailer:
                    yield trailer
            except Exception as exc:
                logger.exception("Gateway handle_stream failed")
                # On error, emit a protocol-correct error terminal through the
                # encoder (Bug 6 Site C) — never hand-roll data: lines.
                # If create_stream_encoder itself raised (before ``encoder`` was
                # ever assigned), build a minimal fallback encoder with no
                # resolved model/route so the error is still emitted as a
                # clean protocol frame instead of an ``UnboundLocalError``.
                if encoder is None:
                    encoder = adapter.create_stream_encoder({})
                for frame in _encode_failure(
                    encoder,
                    CanonicalError(code="INTERNAL_ERROR", message=str(exc)),
                ):
                    yield frame

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # ─── Health ───────────────────────────────────────────────────────────
    #
    # Liveness ("is the process alive") and readiness ("can the selected
    # route actually serve a request") are distinct signals. /v1/health used
    # to always answer "ok" regardless of whether the backend was reachable,
    # authenticated, or had the configured model — liveness masquerading as
    # readiness. /v1/health is now an alias for READINESS, not liveness, so
    # existing callers that treat "not ok" as a real problem get an honest
    # signal instead of an unconditional success.

    @app.get("/health/live")
    @app.get("/v1/health/live")
    async def health_live(request: Request):
        return JSONResponse({"status": "alive"})

    async def _readiness_payload(request: Request) -> dict[str, Any]:
        gw = get_gateway(request)
        await gw._probe_routes()
        return gw.readiness()

    @app.get("/health/ready")
    @app.get("/v1/health/ready")
    async def health_ready(request: Request):
        readiness = await _readiness_payload(request)
        return JSONResponse(readiness, status_code=200 if readiness["ready"] else 503)

    @app.get("/v1/health")
    @app.get("/health")
    async def health(request: Request):
        gw = get_gateway(request)
        info = gw.server_info()
        readiness = await _readiness_payload(request)
        return JSONResponse(
            {
                "status": "ok" if readiness["ready"] else "not_ready",
                "ready": readiness["ready"],
                "default_route": readiness["default_route"],
                "routes": readiness["routes"],
                "version": info.version,
                "model": info.model,
                # No "profile"/"level"/"level_description"/"supports" here:
                # these were always hardcoded placeholders (profile=None,
                # level=0, level_description="", supports=["multi_route"])
                # regardless of the actual configuration — see
                # /v1/capabilities for real, per-route capability data
                # instead of a fabricated single number.
            },
            status_code=200 if readiness["ready"] else 503,
        )

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models(request: Request):
        gw = get_gateway(request)
        return JSONResponse({
            "object": "list",
            "data": gw.config.get_models_response(),
        })

    def _build_effective_capabilities(
        model_profile: Any, codec_caps: Any,
    ) -> EffectiveCapabilities:
        """Build an EffectiveCapabilities from a resolved model profile + codec.

        All states are derived purely from static profile/codec metadata (never
        from live conformance results), so every entry is capped at
        ``CapabilityState.DECLARED`` — never ``VERIFIED`` or ``PROBED``. This
        keeps the endpoint honest: a ``DECLARED`` capability is an unverified
        metadata claim, and after ``is_available()`` no longer counts
        ``DECLARED`` as available, clients can rely on ``compatibility_status``
        rather than over-trusting these states.

        Attribute mapping (verified against the real dataclasses):
        - ``supports_tools`` ← ResolvedModelProfile.supports_native_tools
          OR .supports_textual_tools
        - ``supports_auto``  ← ResolvedModelProfile.tool_automatic
        - ``supports_parallel`` ← ResolvedModelProfile.tool_parallel
        - streaming         ← CodecCapabilities.supports_streaming
        - reasoning         ← ResolvedModelProfile.reasoning_supported
        """
        eff = EffectiveCapabilities()

        supports_tools = (
            bool(getattr(model_profile, "supports_native_tools", False))
            or bool(getattr(model_profile, "supports_textual_tools", False))
        )
        supports_auto = bool(getattr(model_profile, "tool_automatic", False))
        supports_parallel = bool(getattr(model_profile, "tool_parallel", False))

        if not supports_tools:
            eff.tool_level = ToolCapabilityLevel.T0
        elif supports_parallel:
            eff.tool_level = ToolCapabilityLevel.T4
        elif supports_auto:
            eff.tool_level = ToolCapabilityLevel.T2
        else:
            # Supports tools but neither auto-selection nor parallel calls:
            # forced/named single-tool calls only.
            eff.tool_level = ToolCapabilityLevel.T1

        eff.automatic_tools = CapabilityEntry(
            "automatic_tools",
            CapabilityState.DECLARED if supports_auto else CapabilityState.UNSUPPORTED,
        )
        eff.forced_tools = CapabilityEntry(
            "forced_tools",
            CapabilityState.DECLARED if supports_tools else CapabilityState.UNSUPPORTED,
        )
        eff.parallel_tools = CapabilityEntry(
            "parallel_tools",
            CapabilityState.DECLARED if supports_parallel else CapabilityState.UNSUPPORTED,
        )
        eff.structured_arguments = CapabilityEntry(
            "structured_arguments",
            CapabilityState.DECLARED if supports_tools else CapabilityState.UNSUPPORTED,
        )
        eff.text_streaming = CapabilityEntry(
            "text_streaming",
            CapabilityState.DECLARED
            if bool(getattr(codec_caps, "supports_streaming", True))
            else CapabilityState.UNSUPPORTED,
        )
        eff.reasoning = CapabilityEntry(
            "reasoning",
            CapabilityState.DECLARED
            if bool(getattr(model_profile, "reasoning_supported", False))
            else CapabilityState.UNSUPPORTED,
        )

        # agent_level, sequential_tools, tool_error_recovery, image_input,
        # tool_result_continuation are NOT derivable from static profile/codec
        # metadata alone (they require actual conformance-test results) — left
        # at their dataclass defaults (UNSUPPORTED / A0) rather than guessing.

        return eff

    def _build_observed_capabilities(gw: Gateway) -> dict[str, list[dict[str, Any]]]:
        """Empirical evidence-store records per route, as a LIST of every
        distinct compatibility key found for that route's upstream_model —
        never collapsed to one record (a CompatibilityKey has many more
        dimensions than route/model alone: client protocol/id, tool
        choice, effective tool mode, streaming, backend identity, profile
        revision — see evidence/key.py). Each entry's capability_source
        distinguishes observed/manually_approved/stale/revoked evidence;
        this block never overrides "declared" below, it's an additional
        sibling a caller can choose to weigh differently.
        """
        if gw._evidence_store is None:
            return {}
        from agent_interop.evidence.store import capability_source

        observed: dict[str, list[dict[str, Any]]] = {}
        for route_id, route in gw.config.routes.items():
            try:
                records = gw._evidence_store.query_results(model_id=route.upstream_model)
            except Exception as exc:
                logger.warning("observed-capabilities lookup failed for route %s: %s", route_id, exc)
                observed[route_id] = []
                continue
            observed[route_id] = [
                {
                    "client_id": key.client_id,
                    "client_protocol": key.client_protocol,
                    "backend_kind": key.backend_kind,
                    "effective_tool_mode": key.effective_tool_mode,
                    "streaming": key.streaming,
                    "profile_id": key.profile_id,
                    "profile_revision": key.profile_revision,
                    "sample_count": result.sample_count,
                    "tool_selection_rate": result.tool_selection_rate,
                    "task_completion_rate": result.task_completion_rate,
                    "tested_at": result.tested_at,
                    "battery_version": result.battery_version,
                    "capability_source": capability_source(result),
                }
                for key, result in records
            ]
        return observed

    @app.get("/v1/capabilities")
    @app.get("/capabilities")
    async def capabilities(request: Request):
        gw = get_gateway(request)
        info = gw.server_info()
        capability_model = {}
        for route_id, route in gw.config.routes.items():
            profile = gw._resolve_profile(route)
            codec_caps = get_codec(route.upstream.wire_protocol).capabilities()
            eff = _build_effective_capabilities(profile, codec_caps)
            decision = compute_compatibility(eff)
            capability_model[route_id] = {
                **eff.as_dict(),
                "compatibility": {
                    "status": decision.status,
                    "missing_capabilities": decision.missing_capabilities,
                    "warnings": decision.warnings,
                    "remediation": decision.remediation,
                },
            }
        observed = _build_observed_capabilities(gw)
        return JSONResponse({
            "model": info.model,
            # No top-level "level"/"supports" here: those were always a
            # single hardcoded placeholder (level=0, supports=["multi_route"])
            # regardless of the real per-route capabilities computed just
            # above — a multi-route deployment has no single meaningful
            # "level" anyway. capability_model already carries the real,
            # per-route tool_level; that's the honest source of truth.
            #
            # This endpoint derives every entry from static profile/codec
            # metadata (see _build_effective_capabilities above) — never
            # from a live conformance run. "declared" is the honest
            # description; only `interop certify` (opt-in, and itself not a
            # source of automated trust in the MVP) can ever produce a
            # VERIFIED capability state.
            "source": "declared_profile_metadata",
            "verified": False,
            "routes": {
                route_id: {
                    "aliases": route.client_model_aliases,
                    "upstream_model": route.upstream_model,
                    "upstream_kind": route.upstream.kind.value,
                    "wire_protocol": route.upstream.wire_protocol.value,
                    "tool_mode": route.tool_mode.value,
                    "profile": route.profile,
                }
                for route_id, route in gw.config.routes.items()
            },
            "capability_model": capability_model,
            # Sibling, never a replacement for "declared_profile_metadata"
            # above: a per-route LIST of every distinct compatibility key
            # this Gateway's evidence store has empirical records for (empty
            # dict if evidence is disabled/unconfigured for this app
            # instance). See _build_observed_capabilities.
            "observed": observed,
        })

    # ─── Anthropic Messages API ───────────────────────────────────────────

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        try:
            body = await _read_and_parse_body(request, server_config)
        except BodyParseError as exc:
            return _handle_body_parse_error(ProtocolKind.ANTHROPIC_MESSAGES, exc)
        if body.get("stream", False):
            return await _handle_stream(body, request, ProtocolKind.ANTHROPIC_MESSAGES)
        return await _handle_request(body, request, ProtocolKind.ANTHROPIC_MESSAGES)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        gw = get_gateway(request)
        try:
            body = await _read_and_parse_body(request, server_config)
        except BodyParseError as exc:
            return _handle_body_parse_error(ProtocolKind.ANTHROPIC_MESSAGES, exc)
        adapter = get_adapter(ProtocolKind.ANTHROPIC_MESSAGES)
        simplified = adapter.count_tokens_request(body)

        # Resolve the route for the request's actual selected model (Bug 5):
        # never send a real completion request upstream just to count tokens.
        requested_model = simplified.get("model", "")
        route = gw.config.get_route_for_model(requested_model) if requested_model else None
        if route is not None:
            codec = get_codec(route.upstream.wire_protocol)
            usage = codec.count_tokens(
                simplified.get("messages", []), route.upstream_model
            )
            return JSONResponse({
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "confidence": "estimated",
            })

        # Strategy 3: Clear unsupported response
        return JSONResponse(
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "confidence": "unsupported",
                "message": "Token counting is not supported by this backend. Use a tokenizer-aware client.",
            },
            status_code=501,
        )

    # ─── OpenAI Chat Completions API ──────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await _read_and_parse_body(request, server_config)
        except BodyParseError as exc:
            return _handle_body_parse_error(ProtocolKind.OPENAI_CHAT, exc)
        if body.get("stream", False):
            return await _handle_stream(body, request, ProtocolKind.OPENAI_CHAT)
        return await _handle_request(body, request, ProtocolKind.OPENAI_CHAT)

    # ─── OpenAI Responses API ─────────────────────────────────────────────

    @app.post("/v1/responses")
    async def responses_api(request: Request):
        try:
            body = await _read_and_parse_body(request, server_config)
        except BodyParseError as exc:
            return _handle_body_parse_error(ProtocolKind.OPENAI_RESPONSES, exc)
        if body.get("stream", False):
            return await _handle_stream(body, request, ProtocolKind.OPENAI_RESPONSES)
        return await _handle_request(body, request, ProtocolKind.OPENAI_RESPONSES)

    return app
