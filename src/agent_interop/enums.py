"""Canonical enums for Interop — ProtocolKind and ToolCallDialect.

These enums are defined here as the authoritative source. Other modules should
import from this file rather than defining their own.
"""

from __future__ import annotations

from enum import Enum


class ProtocolKind(str, Enum):
    """Client-facing protocol variants."""

    ANTHROPIC_MESSAGES = "anthropic_messages"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"


class ToolCallDialect(str, Enum):
    """Known model-native tool-call dialects."""

    HERMES = "hermes"
    OPENAI_NATIVE = "openai"
    MISTRAL = "mistral"
    QWEN = "qwen"
    LLAMA = "llama"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    GENERIC_JSON = "generic"


class ToolAuthority(str, Enum):
    """Who may execute a tool call.

    CLIENT-declared tools are forwarded to the coding client. INTEROP_INTERNAL
    tools are executed inside Interop and never reach the client — generalizing
    the ``controller_delegate_tool`` precedent.
    """

    CLIENT = "client"
    INTEROP_INTERNAL = "interop_internal"


# Reserved internal tool namespace. Client declarations MUST NOT use this
# prefix; inbound client tools that collide are rejected at preparation time.
RESERVED_INTERNAL_TOOL_PREFIX = "__interop_"


__all__ = ["RESERVED_INTERNAL_TOOL_PREFIX", "ProtocolKind", "ToolAuthority", "ToolCallDialect"]