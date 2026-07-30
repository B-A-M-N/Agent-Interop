"""RequestContext — per-request context that flows through the system.

Carries request identity, client info, timing, and routing details
through the gateway, repair, telemetry, and error paths.

Two distinct header collections are exposed:

* ``sanitized_headers`` — for diagnostics and logging.  Never contains
  credentials.
* ``forwardable_transport_headers`` — for upstream transport.  May
  contain credentials but must never be logged, exported, or echoed
  back to the client.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field

from agent_interop.auth import HEADER_ALLOWLIST_PASSTHROUGH, HOP_BY_HOP_HEADERS, SENSITIVE_HEADERS
from agent_interop.enums import ProtocolKind


@dataclass(frozen=True)
class RequestContext:
    """Per-request context object."""

    request_id: str = ""
    session_id: str = ""

    client_protocol: ProtocolKind = ProtocolKind.ANTHROPIC_MESSAGES
    client_id: str = ""
    client_version: str = ""

    sanitized_headers: Mapping[str, str] = field(default_factory=dict)
    forwardable_transport_headers: Mapping[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    route_id: str = ""

    @classmethod
    def from_headers(
        cls,
        headers: dict[str, str],
        protocol: ProtocolKind = ProtocolKind.ANTHROPIC_MESSAGES,
        recognized_request_id: str = "",
        route_id: str = "",
    ) -> RequestContext:
        """Build a RequestContext from incoming headers.

        Uses trusted headers (Claude Code session headers, request IDs)
        when available. Otherwise generates fresh IDs.

        The returned context splits incoming headers into:

        * ``sanitized_headers`` — diagnostic only, credentials redacted.
        * ``forwardable_transport_headers`` — a lowercase-normalized
          allowlist that may contain credentials, intended for upstream
          transport.  Callers MUST NOT log or persist this field.
        """
        # Build a lowercased view for stable allowlist lookups.
        lower_headers: dict[str, str] = {k.lower(): v for k, v in headers.items()}

        # Extract request ID from header or generate
        request_id = (
            recognized_request_id
            or lower_headers.get("x-request-id", "")
            or lower_headers.get("x-correlation-id", "")
            or f"req_{uuid.uuid4().hex[:16]}"
        )

        # Extract session ID from a trusted header — do NOT synthesize one
        # when absent. A fabricated per-request session ID would make the
        # session store create a fresh one-shot entry for every stateless
        # request (polluting the bounded LRU store) while being useless for
        # loop detection, which needs repeated requests in the SAME session
        # to observe anything. An empty session_id means "no session
        # tracking for this request" throughout the gateway.
        session_id = (
            lower_headers.get("x-session-id", "")
            or lower_headers.get("x-claude-code-session-id", "")
        )

        # Sanitized headers — never include credentials, useful for logs.
        sanitized = {
            k: v for k, v in headers.items()
            if k.lower() not in SENSITIVE_HEADERS
        }

        # Forwardable transport headers — restricted to the explicit
        # passthrough allowlist, never include hop-by-hop.  May
        # contain credentials (x-api-key, authorization) when the
        # upstream auth mode is PASSTHROUGH.
        forwardable: dict[str, str] = {}
        for allowed in HEADER_ALLOWLIST_PASSTHROUGH:
            val = lower_headers.get(allowed)
            if val and allowed not in HOP_BY_HOP_HEADERS:
                forwardable[allowed] = val

        # Detect client from headers
        client_id = ""
        client_version = ""
        user_agent = lower_headers.get("user-agent", "")

        if "x-claude-code-session-id" in lower_headers:
            client_id = "claude_code"
            client_version = lower_headers.get("x-claude-code-client-version", "")
        elif user_agent.startswith("codex") or "codex-cli" in user_agent:
            client_id = "codex"
        elif "cline" in user_agent:
            client_id = "cline"
        elif "opencode" in user_agent:
            client_id = "opencode"
        elif "continue" in user_agent:
            client_id = "continue"
        elif "aider" in user_agent:
            client_id = "aider"
        elif lower_headers.get("x-interop-client"):
            # Generic self-assertion path for clients with no distinguishing
            # header/User-Agent of their own (e.g. hermes-agent, which wraps
            # the stock OpenAI SDK and sends nothing hermes-specific to a
            # custom/local endpoint — confirmed by reading its source, not
            # assumed). Interop's own launch integration for such a client
            # configures it to send this header via whatever mechanism the
            # client supports (e.g. hermes-agent's `model.default_headers`
            # config field), rather than Interop guessing identity from
            # generic-SDK fingerprints. Trusted the same way the other
            # branches above are: this only ever runs against a route
            # Interop itself launched and pointed at its own gateway.
            client_id = lower_headers["x-interop-client"]

        return cls(
            request_id=request_id,
            session_id=session_id,
            client_protocol=protocol,
            client_id=client_id,
            client_version=client_version,
            sanitized_headers=sanitized,
            forwardable_transport_headers=forwardable,
            route_id=route_id,
        )
