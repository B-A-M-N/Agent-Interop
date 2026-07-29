"""Authentication and credential management.

Separates ingress credentials (client → Interop) from upstream
credentials (Interop → backend). Never forwards dummy credentials
upstream. Redacts credentials from logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IngressAuthMode(Enum):
    """How the gateway authenticates incoming client requests."""

    NONE_LOOPBACK = "none_loopback"  # Only allow loopback connections
    SESSION_TOKEN = "session_token"  # Verify a session token
    STATIC_TOKEN = "static_token"    # Verify a static bearer token


class UpstreamAuthMode(Enum):
    """How the gateway authenticates to upstream backends."""

    NONE = "none"                # No auth (local models)
    API_KEY = "api_key"          # Static API key
    PASSTHROUGH = "passthrough"  # Forward client's credential


@dataclass
class IngressAuthConfig:
    """Ingress authentication configuration."""

    mode: IngressAuthMode = IngressAuthMode.NONE_LOOPBACK
    allowed_tokens: list[str] = field(default_factory=list)


@dataclass
class UpstreamAuthConfig:
    """Upstream authentication configuration."""

    mode: UpstreamAuthMode = UpstreamAuthMode.NONE
    api_key: str | None = None
    api_key_header: str = "Authorization"
    env_key: str | None = None  # Environment variable name for the key
    command: list[str] = field(default_factory=list)


# ─── Hop-by-hop headers that must never be forwarded ─────────────────────────

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}

HEADER_ALLOWLIST_PASSTHROUGH = {
    "anthropic-version",
    "anthropic-beta",
    "x-api-key",
    "authorization",
    "x-claude-code-session-id",
    "x-claude-code-agent-id",
    "x-claude-code-parent-agent-id",
    "content-type",
    "accept",
}

SENSITIVE_HEADERS = {
    "authorization",
    "x-api-key",
    "x-session-token",
    "cookie",
    "set-cookie",
}


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive header values for logging."""
    return {
        k: ("[REDACTED]" if k.lower() in SENSITIVE_HEADERS else v)
        for k, v in headers.items()
    }


def build_upstream_headers(
    client_headers: dict[str, str],
    upstream_auth: UpstreamAuthConfig,
    route_upstream_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build headers for the upstream request.

    Rules:
    - Strip hop-by-hop headers.
    - Only forward allowlisted headers if the upstream auth mode is PASSTHROUGH.
    - Otherwise, only set upstream auth if explicitly configured.
    - All header keys are normalized to lowercase so merges are
      case-insensitive (avoiding "content-type" + "Content-Type"
      duplicates).
    """
    # Normalize input keys to lowercase for case-insensitive merge.
    lower_route = (
        {k.lower(): v for k, v in (route_upstream_headers or {}).items()}
        if route_upstream_headers
        else {}
    )
    lower_client = {k.lower(): v for k, v in (client_headers or {}).items()}

    headers: dict[str, str] = {}
    headers.update(lower_route)

    # Auth
    if upstream_auth.mode == UpstreamAuthMode.PASSTHROUGH:
        for h in HEADER_ALLOWLIST_PASSTHROUGH:
            if h in lower_client:
                headers[h] = lower_client[h]
    elif upstream_auth.mode == UpstreamAuthMode.API_KEY:
        key = upstream_auth.api_key
        if upstream_auth.env_key:
            import os
            key = os.getenv(upstream_auth.env_key, key)
        if not key:
            # Fail loudly when env_key was requested but the
            # environment variable is missing.
            import os as _os
            if upstream_auth.env_key and not _os.environ.get(upstream_auth.env_key):
                raise RuntimeError(
                    f"api_key_env={upstream_auth.env_key!r} configured but environment variable is unset"
                )
        if key:
            headers[upstream_auth.api_key_header.lower()] = (
                f"Bearer {key}" if upstream_auth.api_key_header.lower() == "authorization"
                else key
            )

    # Always forward protocol-required metadata headers
    for h in ("anthropic-version", "anthropic-beta", "content-type"):
        if h in lower_client:
            headers[h] = lower_client[h]

    # Strip hop-by-hop
    headers = {k: v for k, v in headers.items() if k not in HOP_BY_HOP_HEADERS}

    if "content-type" not in headers:
        headers["content-type"] = "application/json"

    return headers