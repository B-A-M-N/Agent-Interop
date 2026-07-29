"""Typed error codes and structured error responses for Interop.

Every error has a machine-readable code, human message, optional
remediation steps, and correlation IDs. Raw backend stack traces
are never returned to the client unless debug mode is enabled.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import CanonicalError

logger = logging.getLogger("agent_interop.errors")


class InteropErrorCode(str):
    """Stable internal error codes.

    Each code documents a specific failure class so clients can
    handle errors programmatically.
    """

    # Configuration errors
    CONFIG_INVALID = "CONFIG_INVALID"
    CLIENT_PROTOCOL_UNSUPPORTED = "CLIENT_PROTOCOL_UNSUPPORTED"
    CLIENT_REQUEST_INVALID = "CLIENT_REQUEST_INVALID"

    # Request-level errors
    INVALID_REQUEST = "INVALID_REQUEST"
    HISTORY_UNSAFE = "HISTORY_UNSAFE"
    TOOL_CHOICE_VIOLATION = "TOOL_CHOICE_VIOLATION"

    # Backend errors
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    BACKEND_AUTH_FAILED = "BACKEND_AUTH_FAILED"
    BACKEND_PROTOCOL_ERROR = "BACKEND_PROTOCOL_ERROR"
    BACKEND_RATE_LIMITED = "BACKEND_RATE_LIMITED"
    BACKEND_TIMEOUT = "BACKEND_TIMEOUT"
    BACKEND_ERROR = "BACKEND_ERROR"

    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_PROFILE_NOT_FOUND = "MODEL_PROFILE_NOT_FOUND"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"

    # Route errors
    ROUTE_NOT_FOUND = "ROUTE_NOT_FOUND"

    # Session errors
    SESSION_INVALID = "SESSION_INVALID"
    SESSION_EXPIRED = "SESSION_EXPIRED"

    # Capability errors
    CAPABILITY_REQUIRED = "CAPABILITY_REQUIRED"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"

    # Tool errors
    TOOL_SCHEMA_UNSUPPORTED = "TOOL_SCHEMA_UNSUPPORTED"
    TOOL_CALL_INVALID = "TOOL_CALL_INVALID"
    TOOL_CALL_REPAIR_FAILED = "TOOL_CALL_REPAIR_FAILED"

    # Stream errors
    STREAM_PROTOCOL_ERROR = "STREAM_PROTOCOL_ERROR"
    STREAM_SIZE_LIMIT = "STREAM_SIZE_LIMIT"
    MALFORMED_MODEL_OUTPUT = "MALFORMED_MODEL_OUTPUT"

    # Repair errors
    REPAIR_BUDGET_EXHAUSTED = "REPAIR_BUDGET_EXHAUSTED"

    # Generation errors
    GENERATION_CANCELLED = "GENERATION_CANCELLED"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    GENERATION_LOOP_DETECTED = "GENERATION_LOOP_DETECTED"

    # Ingress authentication (client → Interop, not Interop → backend)
    INGRESS_AUTH_FAILED = "INGRESS_AUTH_FAILED"
    INGRESS_FORBIDDEN = "INGRESS_FORBIDDEN"

    # Internal
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True)
class ErrorDescriptor:
    """Authoritative error metadata for a single InteropErrorCode.

    Protocol adapters use this to format errors consistently instead of
    each maintaining a separate dictionary.
    """

    code: str
    http_status: int
    retryable: bool
    anthropic_type: str
    openai_type: str


# ─── Centralized error registry ────────────────────────────────────────────
# Single source of truth for HTTP status, retryability, and protocol mappings.

ERROR_REGISTRY: dict[str, ErrorDescriptor] = {
    InteropErrorCode.CONFIG_INVALID: ErrorDescriptor(
        code=InteropErrorCode.CONFIG_INVALID, http_status=500, retryable=False,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.CLIENT_PROTOCOL_UNSUPPORTED: ErrorDescriptor(
        code=InteropErrorCode.CLIENT_PROTOCOL_UNSUPPORTED, http_status=400, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.CLIENT_REQUEST_INVALID: ErrorDescriptor(
        code=InteropErrorCode.CLIENT_REQUEST_INVALID, http_status=400, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.INVALID_REQUEST: ErrorDescriptor(
        code=InteropErrorCode.INVALID_REQUEST, http_status=400, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.HISTORY_UNSAFE: ErrorDescriptor(
        code=InteropErrorCode.HISTORY_UNSAFE, http_status=422, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.TOOL_CHOICE_VIOLATION: ErrorDescriptor(
        code=InteropErrorCode.TOOL_CHOICE_VIOLATION, http_status=422, retryable=True,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.BACKEND_UNAVAILABLE: ErrorDescriptor(
        code=InteropErrorCode.BACKEND_UNAVAILABLE, http_status=503, retryable=True,
        anthropic_type="overloaded_error", openai_type="server_error"),
    InteropErrorCode.BACKEND_AUTH_FAILED: ErrorDescriptor(
        code=InteropErrorCode.BACKEND_AUTH_FAILED, http_status=401, retryable=False,
        anthropic_type="authentication_error", openai_type="authentication_error"),
    InteropErrorCode.BACKEND_PROTOCOL_ERROR: ErrorDescriptor(
        code=InteropErrorCode.BACKEND_PROTOCOL_ERROR, http_status=502, retryable=True,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.BACKEND_RATE_LIMITED: ErrorDescriptor(
        code=InteropErrorCode.BACKEND_RATE_LIMITED, http_status=429, retryable=True,
        anthropic_type="rate_limit_error", openai_type="rate_limit_error"),
    InteropErrorCode.BACKEND_TIMEOUT: ErrorDescriptor(
        code=InteropErrorCode.BACKEND_TIMEOUT, http_status=504, retryable=True,
        anthropic_type="timeout_error", openai_type="timeout_error"),
    InteropErrorCode.BACKEND_ERROR: ErrorDescriptor(
        code=InteropErrorCode.BACKEND_ERROR, http_status=502, retryable=True,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.MODEL_NOT_FOUND: ErrorDescriptor(
        code=InteropErrorCode.MODEL_NOT_FOUND, http_status=404, retryable=False,
        anthropic_type="not_found_error", openai_type="not_found_error"),
    InteropErrorCode.MODEL_PROFILE_NOT_FOUND: ErrorDescriptor(
        code=InteropErrorCode.MODEL_PROFILE_NOT_FOUND, http_status=404, retryable=False,
        anthropic_type="not_found_error", openai_type="not_found_error"),
    InteropErrorCode.MODEL_INCOMPATIBLE: ErrorDescriptor(
        code=InteropErrorCode.MODEL_INCOMPATIBLE, http_status=422, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.ROUTE_NOT_FOUND: ErrorDescriptor(
        code=InteropErrorCode.ROUTE_NOT_FOUND, http_status=404, retryable=False,
        anthropic_type="not_found_error", openai_type="not_found_error"),
    InteropErrorCode.SESSION_INVALID: ErrorDescriptor(
        code=InteropErrorCode.SESSION_INVALID, http_status=401, retryable=False,
        anthropic_type="authentication_error", openai_type="authentication_error"),
    InteropErrorCode.SESSION_EXPIRED: ErrorDescriptor(
        code=InteropErrorCode.SESSION_EXPIRED, http_status=401, retryable=False,
        anthropic_type="authentication_error", openai_type="authentication_error"),
    InteropErrorCode.CAPABILITY_REQUIRED: ErrorDescriptor(
        code=InteropErrorCode.CAPABILITY_REQUIRED, http_status=422, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.CONTEXT_LIMIT_EXCEEDED: ErrorDescriptor(
        code=InteropErrorCode.CONTEXT_LIMIT_EXCEEDED, http_status=400, retryable=False,
        anthropic_type="invalid_request_error", openai_type="context_length_exceeded"),
    InteropErrorCode.TOOL_SCHEMA_UNSUPPORTED: ErrorDescriptor(
        code=InteropErrorCode.TOOL_SCHEMA_UNSUPPORTED, http_status=400, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.TOOL_CALL_INVALID: ErrorDescriptor(
        code=InteropErrorCode.TOOL_CALL_INVALID, http_status=422, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.TOOL_CALL_REPAIR_FAILED: ErrorDescriptor(
        code=InteropErrorCode.TOOL_CALL_REPAIR_FAILED, http_status=422, retryable=True,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.STREAM_PROTOCOL_ERROR: ErrorDescriptor(
        code=InteropErrorCode.STREAM_PROTOCOL_ERROR, http_status=502, retryable=True,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.STREAM_SIZE_LIMIT: ErrorDescriptor(
        code=InteropErrorCode.STREAM_SIZE_LIMIT, http_status=413, retryable=False,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.MALFORMED_MODEL_OUTPUT: ErrorDescriptor(
        code=InteropErrorCode.MALFORMED_MODEL_OUTPUT, http_status=502, retryable=True,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.REPAIR_BUDGET_EXHAUSTED: ErrorDescriptor(
        code=InteropErrorCode.REPAIR_BUDGET_EXHAUSTED, http_status=422, retryable=True,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.GENERATION_CANCELLED: ErrorDescriptor(
        code=InteropErrorCode.GENERATION_CANCELLED, http_status=499, retryable=False,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.GENERATION_TIMEOUT: ErrorDescriptor(
        code=InteropErrorCode.GENERATION_TIMEOUT, http_status=504, retryable=True,
        anthropic_type="timeout_error", openai_type="timeout_error"),
    InteropErrorCode.GENERATION_LOOP_DETECTED: ErrorDescriptor(
        code=InteropErrorCode.GENERATION_LOOP_DETECTED, http_status=422, retryable=False,
        anthropic_type="invalid_request_error", openai_type="invalid_request_error"),
    InteropErrorCode.INTERNAL_ERROR: ErrorDescriptor(
        code=InteropErrorCode.INTERNAL_ERROR, http_status=500, retryable=False,
        anthropic_type="api_error", openai_type="server_error"),
    InteropErrorCode.INGRESS_AUTH_FAILED: ErrorDescriptor(
        code=InteropErrorCode.INGRESS_AUTH_FAILED, http_status=401, retryable=False,
        anthropic_type="authentication_error", openai_type="authentication_error"),
    InteropErrorCode.INGRESS_FORBIDDEN: ErrorDescriptor(
        code=InteropErrorCode.INGRESS_FORBIDDEN, http_status=403, retryable=False,
        anthropic_type="permission_error", openai_type="permission_error"),
}


def get_error_descriptor(code: str) -> ErrorDescriptor:
    """Look up the error descriptor for a given code.

    Returns the INTERNAL_ERROR descriptor for unknown codes.
    """
    return ERROR_REGISTRY.get(code, ERROR_REGISTRY[InteropErrorCode.INTERNAL_ERROR])


# ─── HTTP status → InteropErrorCode mapping (item 77) ──────────────────────


def classify_http_status(status: int, message: str = "") -> str:
    """Map an HTTP status code to an InteropErrorCode.

    Centralized classification so transport, gateway, and adapters
    all agree on what backend failures mean. ``message`` is reserved
    for future content-based classification (e.g., rate-limit
    detection in body text).
    """
    _ = message  # Reserved for content-based classification
    if status == 400:
        return InteropErrorCode.INVALID_REQUEST
    if status == 401:
        return InteropErrorCode.BACKEND_AUTH_FAILED
    if status == 403:
        return InteropErrorCode.BACKEND_AUTH_FAILED
    if status == 404:
        return InteropErrorCode.MODEL_NOT_FOUND
    if status == 408:
        return InteropErrorCode.BACKEND_TIMEOUT
    if status == 429:
        return InteropErrorCode.BACKEND_RATE_LIMITED
    if status == 413:
        return InteropErrorCode.CONTEXT_LIMIT_EXCEEDED
    if status == 502:
        return InteropErrorCode.BACKEND_PROTOCOL_ERROR
    if status == 503:
        return InteropErrorCode.BACKEND_UNAVAILABLE
    if status == 504:
        return InteropErrorCode.BACKEND_TIMEOUT
    if status >= 500:
        return InteropErrorCode.BACKEND_ERROR
    if status >= 400:
        return InteropErrorCode.INVALID_REQUEST
    return InteropErrorCode.BACKEND_ERROR


def parse_retry_after(headers: dict[str, str]) -> float:
    """Extract Retry-After value from response headers.

    Returns delay in seconds, or 0.0 if not present or unparseable.
    Supports both delta-seconds and HTTP-date formats.
    """
    retry_after = headers.get("retry-after", headers.get("Retry-After", ""))
    if not retry_after:
        return 0.0
    try:
        return float(retry_after)
    except ValueError:
        # Could be HTTP-date; conservatively use small fixed delay
        return 1.0


# ─── InteropError exception ─────────────────────────────────────────────────


@dataclass
class InteropError(Exception):
    """Structured error with typed code, message, and remediation."""

    code: str
    message: str
    request_id: str = ""
    session_id: str = ""
    model: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    remediation: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a safe client-facing error response."""
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "request_id": self.request_id,
        }
        if self.details:
            result["details"] = self.details
        if self.remediation:
            result["remediation"] = self.remediation
        if self.model:
            result["model"] = self.model
        return result

    def to_anthropic_error(self) -> dict[str, Any]:
        """Convert to an Anthropic-compatible error response."""
        descriptor = get_error_descriptor(self.code)
        return {
            "type": "error",
            "error": {
                "type": descriptor.anthropic_type,
                "message": self.message,
            },
        }

    def to_openai_error(self) -> dict[str, Any]:
        """Convert to an OpenAI-compatible error response."""
        descriptor = get_error_descriptor(self.code)
        return {
            "error": {
                "message": self.message,
                "type": descriptor.openai_type,
                "code": self.code,
                "param": None,
                "http_status": descriptor.http_status,
            },
        }

    def to_canonical_error(self) -> CanonicalError:
        """Convert to a canonical error representation."""
        descriptor = get_error_descriptor(self.code)
        return CanonicalError(
            code=self.code,
            message=self.message,
            retryable=descriptor.retryable,
            upstream_status=descriptor.http_status,
            request_id=self.request_id,
            details=self.details,
        )


@dataclass(frozen=True)
class EncodedErrorResponse:
    """An error response encoded for a specific protocol."""

    body: dict[str, Any] = field(default_factory=dict)
    status_code: int = 500
    headers: dict[str, str] = field(default_factory=dict)


# ─── Factory helpers ───────────────────────────────────────────────────────


def err_config_invalid(message: str, **kw: Any) -> InteropError:
    return InteropError(code=InteropErrorCode.CONFIG_INVALID, message=message, **kw)


def err_backend_unavailable(backend_url: str, reason: str = "") -> InteropError:
    return InteropError(
        code=InteropErrorCode.BACKEND_UNAVAILABLE,
        message=f"Backend at {backend_url} is not reachable",
        details={"backend_url": backend_url, "reason": reason},
        remediation=["Check that the backend is running", "Verify BACKEND_URL in config"],
    )


def err_route_not_found(model_name: str) -> InteropError:
    return InteropError(
        code=InteropErrorCode.ROUTE_NOT_FOUND,
        message=f"No route configured for model '{model_name}'",
        details={"requested_model": model_name},
        remediation=["Check the model name", "Configure a route for this model"],
    )


def err_session_invalid(session_id: str, reason: str = "") -> InteropError:
    return InteropError(
        code=InteropErrorCode.SESSION_INVALID,
        message=f"Session '{session_id}' is invalid or not found",
        details={"session_id": session_id, "reason": reason} if reason else {"session_id": session_id},
    )


def err_model_incompatible(model: str, required: list[str], verified: list[str]) -> InteropError:
    return InteropError(
        code=InteropErrorCode.MODEL_INCOMPATIBLE,
        message=f"Model {model} cannot satisfy required capabilities",
        model=model,
        details={"required": required, "verified": verified},
        remediation=[
            "Run `interop test {model}` for compatibility details",
            "Select a model rated at least A3",
            "Enable degraded mode for single-operation tasks",
        ],
    )


def err_tool_call_invalid(detail: str, **kw: Any) -> InteropError:
    return InteropError(code=InteropErrorCode.TOOL_CALL_INVALID, message=detail, **kw)


def err_loop_detected(model: str, signal: str) -> InteropError:
    return InteropError(
        code=InteropErrorCode.GENERATION_LOOP_DETECTED,
        message=f"Model output loop detected: {signal}",
        model=model,
        remediation=["Try a different model with higher capability rating",
                     "Reduce tool count or simplify the task"],
    )


def err_backend_timeout(backend_url: str, timeout: float) -> InteropError:
    return InteropError(
        code=InteropErrorCode.BACKEND_TIMEOUT,
        message=f"Backend at {backend_url} timed out after {timeout}s",
        details={"backend_url": backend_url, "timeout_seconds": timeout},
        remediation=["Increase the route timeout", "Check backend load"],
    )


_SENSITIVE_DETAIL_KEYS = frozenset({
    "raw_response", "stack_trace", "traceback", "internal_error",
    "backend_url", "api_key", "password", "token", "secret",
})


def sanitize_error_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize error details for client consumption.

    Strips sensitive fields that must never reach the client (raw backends
    responses, stack traces, credentials). Allows structured corrections
    and diagnostic hints through.
    """
    if not details:
        return {}

    return {
        k: v for k, v in details.items()
        if k.lower() not in _SENSITIVE_DETAIL_KEYS
    }


# Patterns for credential-like content that must never reach a client, even
# when it arrives embedded in a free-text error message (e.g. an upstream
# error body, or str(exc) on a connection failure whose URL carried a
# token). sanitize_error_details() only ever covered the `details` dict —
# error.message flowed to the client completely unredacted.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Authorization: Bearer <token> / Authorization: Basic <token>
    (re.compile(r"(?i)\b(authorization\s*:\s*(?:bearer|basic)\s+)\S+"), r"\1[REDACTED]"),
    # Bearer <token> outside a header line (e.g. embedded in a URL query or body)
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{8,}"), "Bearer [REDACTED]"),
    # x-api-key: <value> and similar single-value API key headers
    (re.compile(r"(?i)\b(x-api-key\s*:\s*)\S+"), r"\1[REDACTED]"),
    # key="value" / key: value style credential fields (api_key, apikey,
    # token, secret, password) in JSON, query strings, or log-style text
    (re.compile(
        r'(?i)\b(api[_-]?key|access[_-]?token|secret|password|passwd)\b'
        r'(["\']?\s*[:=]\s*["\']?)([^\s"\',&]{4,})'
    ), r"\1\2[REDACTED]"),
    # Credentials embedded in a URL: scheme://user:pass@host
    (re.compile(r"(?i)(\b\w+://)[^/\s@]+:[^/\s@]+@"), r"\1[REDACTED]@"),
    # Cookie header values
    (re.compile(r"(?i)\b(cookie\s*:\s*)\S.*"), r"\1[REDACTED]"),
)


def redact_secrets(text: str) -> str:
    """Redact credential-like substrings from free-text error content.

    Applied to CanonicalError.message before it reaches a client — unlike
    `.details`, message is a free-text field that can end up containing
    upstream response fragments or raw exception text, either of which
    might embed an Authorization header, API key, or credentialed URL.
    Best-effort pattern matching, not a guarantee: never a substitute for
    not putting raw upstream/internal content in a client-visible message
    in the first place.
    """
    if not text:
        return text
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def serialize_client_error(
    error: CanonicalError,
    protocol: str,
) -> EncodedErrorResponse:
    """Encode a canonical error for the given protocol, preserving details.

    This is the centralized error serializer that all adapters should use,
    replacing per-adapter encode_error implementations to prevent drift.
    """
    descriptor = get_error_descriptor(error.code)

    details = sanitize_error_details(error.details)
    message = redact_secrets(error.message)

    if protocol == "anthropic":
        body: dict[str, Any] = {
            "type": "error",
            "error": {
                "type": descriptor.anthropic_type,
                "message": message,
            },
        }
    elif protocol == "openai_chat":
        body = {
            "error": {
                "message": message,
                "type": descriptor.openai_type,
                "code": error.code,
                "param": None,
            },
        }
    elif protocol == "openai_responses":
        body = {
            "type": "error",
            "error": {
                "code": error.code,
                "message": message,
            },
        }
    else:
        body = {
            "error": {
                "message": message,
                "code": error.code,
            },
        }

    if details:
        body["details"] = details

    return EncodedErrorResponse(
        body=body,
        status_code=descriptor.http_status,
    )
