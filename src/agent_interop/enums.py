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


__all__ = ["ProtocolKind", "ToolCallDialect"]