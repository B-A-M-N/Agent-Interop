"""Capture — record request/response cycles for replay.

Captures a complete request/response cycle, sanitizes sensitive data,
and produces a ReplayCase for deterministic evaluation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTool
from agent_interop.replay.types import (
    CompatibilityKey,
    ReplayCase,
    ReplayInvariant,
)

# Headers that must never be captured
_SENSITIVE_HEADERS = {
    "authorization",
    "x-api-key",
    "cookie",
    "set-cookie",
    "x-session-token",
}


def sanitize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Remove sensitive headers before capture."""
    return {
        k: ("[REDACTED]" if k.lower() in _SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


def sanitize_body(body: Any) -> Any:
    """Recursively remove sensitive content from serializable diagnostics.

    Capture callers pass mappings, ABI dataclasses, headers, and nested lists.
    A shallow pass can leave an API key in a nested tool payload, so every
    branch follows the same field-name rule before persistence.
    """
    if is_dataclass(body) and not isinstance(body, type):
        return sanitize_body(asdict(body))
    if isinstance(body, Mapping):
        return {
            str(key): "[REDACTED]" if str(key).lower() in _SENSITIVE_HEADERS
            or str(key).lower() in ("api_key", "token", "secret", "password")
            else sanitize_body(value)
            for key, value in body.items()
        }
    if isinstance(body, list):
        return [sanitize_body(value) for value in body]
    if isinstance(body, tuple):
        return tuple(sanitize_body(value) for value in body)
    return body


def capture_case(
    *,
    client_protocol: str,
    upstream_protocol: str,
    inbound_request: Mapping[str, Any],
    canonical_request: CanonicalRequest | None = None,
    upstream_request: Mapping[str, Any],
    raw_upstream_response: Mapping[str, Any] | None = None,
    raw_upstream_frames: Sequence[str] = (),
    tool_registry: Sequence[CanonicalTool] = (),
    expected_invariants: Sequence[ReplayInvariant] = (),
    compatibility_key: CompatibilityKey | None = None,
    case_id: str = "",
    diagnostics: Mapping[str, Any] | None = None,
) -> ReplayCase:
    """Capture a request/response cycle as a ReplayCase.

    Sanitizes sensitive data before persistence.
    """
    if not case_id:
        # Deterministic ID from content hash
        content = f"{client_protocol}:{upstream_protocol}:{sorted(inbound_request.items())}"
        case_id = f"case_{hashlib.sha256(content.encode()).hexdigest()[:16]}"

    return ReplayCase(
        format_version="interop.replay.v1",
        case_id=case_id,
        client_protocol=client_protocol,
        upstream_protocol=upstream_protocol,
        compatibility_key=compatibility_key or CompatibilityKey(),
        inbound_request=sanitize_body(inbound_request),
        canonical_request=canonical_request,
        upstream_request=sanitize_body(upstream_request),
        raw_upstream_response=sanitize_body(raw_upstream_response)
        if raw_upstream_response
        else None,
        raw_upstream_frames=tuple(raw_upstream_frames),
        tool_registry=tuple(tool_registry),
        expected_invariants=tuple(expected_invariants),
        diagnostics=sanitize_body(diagnostics or {}),
    )
