"""Typed error codes and structured error responses for Interop.

Every error has a machine-readable code, human message, optional
remediation steps, and correlation IDs. Raw backend stack traces
are never returned to the client unless debug mode is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class InteropErrorCode(str):
    """Stable internal error codes.

    Each code documents a specific failure class so clients can
    handle errors programmatically.
    """

    # Configuration errors
    CONFIG_INVALID = "CONFIG_INVALID"
    CLIENT_PROTOCOL_UNSUPPORTED = "CLIENT_PROTOCOL_UNSUPPORTED"
    CLIENT_REQUEST_INVALID = "CLIENT_REQUEST_INVALID"

    # Backend errors
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    BACKEND_AUTH_FAILED = "BACKEND_AUTH_FAILED"
    BACKEND_PROTOCOL_ERROR = "BACKEND_PROTOCOL_ERROR"

    # Model errors
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_PROFILE_NOT_FOUND = "MODEL_PROFILE_NOT_FOUND"
    MODEL_INCOMPATIBLE = "MODEL_INCOMPATIBLE"

    # Capability errors
    CAPABILITY_REQUIRED = "CAPABILITY_REQUIRED"
    CONTEXT_LIMIT_EXCEEDED = "CONTEXT_LIMIT_EXCEEDED"

    # Tool errors
    TOOL_SCHEMA_UNSUPPORTED = "TOOL_SCHEMA_UNSUPPORTED"
    TOOL_CALL_INVALID = "TOOL_CALL_INVALID"
    TOOL_CALL_REPAIR_FAILED = "TOOL_CALL_REPAIR_FAILED"

    # Stream errors
    STREAM_PROTOCOL_ERROR = "STREAM_PROTOCOL_ERROR"

    # Generation errors
    GENERATION_CANCELLED = "GENERATION_CANCELLED"
    GENERATION_TIMEOUT = "GENERATION_TIMEOUT"
    GENERATION_LOOP_DETECTED = "GENERATION_LOOP_DETECTED"

    # Internal
    INTERNAL_ERROR = "INTERNAL_ERROR"

    # All known codes for validation
    ALL = frozenset({
        CONFIG_INVALID, CLIENT_PROTOCOL_UNSUPPORTED, CLIENT_REQUEST_INVALID,
        BACKEND_UNAVAILABLE, BACKEND_AUTH_FAILED, BACKEND_PROTOCOL_ERROR,
        MODEL_NOT_FOUND, MODEL_PROFILE_NOT_FOUND, MODEL_INCOMPATIBLE,
        CAPABILITY_REQUIRED, CONTEXT_LIMIT_EXCEEDED,
        TOOL_SCHEMA_UNSUPPORTED, TOOL_CALL_INVALID, TOOL_CALL_REPAIR_FAILED,
        STREAM_PROTOCOL_ERROR,
        GENERATION_CANCELLED, GENERATION_TIMEOUT, GENERATION_LOOP_DETECTED,
        INTERNAL_ERROR,
    })


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
        an_type = {
            InteropErrorCode.BACKEND_UNAVAILABLE: "overloaded_error",
            InteropErrorCode.BACKEND_AUTH_FAILED: "authentication_error",
            InteropErrorCode.CONTEXT_LIMIT_EXCEEDED: "invalid_request_error",
            InteropErrorCode.GENERATION_TIMEOUT: "timeout_error",
        }.get(self.code, "api_error")
        return {
            "type": "error",
            "error": {
                "type": an_type,
                "message": self.message,
            },
        }

    def to_openai_error(self) -> dict[str, Any]:
        """Convert to an OpenAI-compatible error response."""
        http_status = {
            InteropErrorCode.BACKEND_UNAVAILABLE: 503,
            InteropErrorCode.BACKEND_AUTH_FAILED: 401,
            InteropErrorCode.MODEL_NOT_FOUND: 404,
            InteropErrorCode.CONTEXT_LIMIT_EXCEEDED: 400,
            InteropErrorCode.CLIENT_REQUEST_INVALID: 400,
            InteropErrorCode.TOOL_CALL_INVALID: 400,
            InteropErrorCode.GENERATION_TIMEOUT: 408,
            InteropErrorCode.GENERATION_LOOP_DETECTED: 422,
        }.get(self.code, 500)
        return {
            "error": {
                "message": self.message,
                "type": "interop_error",
                "code": self.code,
                "param": None,
                "http_status": http_status,
            }
        }


# ─── Factory helpers ───────────────────────────────────────────────────────


def err_config_invalid(message: str, **kw) -> InteropError:
    return InteropError(code=InteropErrorCode.CONFIG_INVALID, message=message, **kw)


def err_backend_unavailable(backend_url: str, reason: str = "") -> InteropError:
    return InteropError(
        code=InteropErrorCode.BACKEND_UNAVAILABLE,
        message=f"Backend at {backend_url} is not reachable",
        details={"backend_url": backend_url, "reason": reason},
        remediation=["Check that the backend is running", "Verify BACKEND_URL in config"],
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


def err_tool_call_invalid(detail: str, **kw) -> InteropError:
    return InteropError(code=InteropErrorCode.TOOL_CALL_INVALID, message=detail, **kw)


def err_loop_detected(model: str, signal: str) -> InteropError:
    return InteropError(
        code=InteropErrorCode.GENERATION_LOOP_DETECTED,
        message=f"Model output loop detected: {signal}",
        model=model,
        remediation=["Try a different model with higher capability rating",
                     "Reduce tool count or simplify the task"],
    )