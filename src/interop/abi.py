"""Interop Agent ABI — versioned canonical representation and JSON Schema.

The Agent ABI defines the versioned internal protocol that all adapters
translate to/from. It is independent of any specific client or backend.

All types here match the spec's TypeScript CanonicalRequest/CanonicalResponse
representation, converted to Python dataclasses with JSON Schema generation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Literal

ABI_VERSION = "interop.agent.v1"


# ─── ABI Metadata ──────────────────────────────────────────────────────────


__all__ = [
    "ABI_VERSION",
    # Enums
    "CanonicalStopReason",
    "ContentBlockType",
    "ToolChoiceMode",
    # Dataclasses
    "CanonicalModelReference",
    "CanonicalContentBlock",
    "CanonicalTextBlock",
    "CanonicalReasoningBlock",
    "CanonicalToolCallBlock",
    "CanonicalToolResultBlock",
    "CanonicalMessage",
    "CanonicalTool",
    "CanonicalToolChoice",
    "CanonicalGenerationOptions",
    "RequestedCapabilities",
    "CanonicalUsage",
    "CanonicalRequest",
    "CanonicalResponse",
    # Helpers
    "abi_json_schema",
    "new_request_id",
    "new_session_id",
]


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:16]}"


# ─── Enums ─────────────────────────────────────────────────────────────────


class CanonicalStopReason(str, Enum):
    END_TURN = "end_turn"
    TOOL_CALL = "tool_call"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    BACKEND_ERROR = "backend_error"
    INVALID_OUTPUT = "invalid_output"
    CANCELLED = "cancelled"


class ContentBlockType(str, Enum):
    TEXT = "text"
    REASONING = "reasoning"
    IMAGE = "image"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class ToolChoiceMode(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


# ─── Core dataclasses ──────────────────────────────────────────────────────


@dataclass
class CanonicalModelReference:
    requested_name: str = ""
    resolved_name: str = ""
    backend: str = ""
    profile: str | None = None


@dataclass
class CanonicalTextBlock:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class CanonicalReasoningBlock:
    type: Literal["reasoning"] = "reasoning"
    content: str = ""
    visibility: str = "hidden"  # hidden | summary | exposed
    signature: str | None = None


@dataclass
class CanonicalToolCallBlock:
    type: Literal["tool_call"] = "tool_call"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalToolResultBlock:
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class CanonicalMessage:
    role: Literal["user", "assistant"] = "user"
    content: list[CanonicalTextBlock | CanonicalReasoningBlock | CanonicalToolCallBlock | CanonicalToolResultBlock] = field(default_factory=list)


@dataclass
class CanonicalTool:
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False


@dataclass
class CanonicalToolChoice:
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    name: str = ""
    # Convenience constructors
    @classmethod
    def auto(cls) -> CanonicalToolChoice:
        return cls(mode=ToolChoiceMode.AUTO)
    @classmethod
    def none(cls) -> CanonicalToolChoice:
        return cls(mode=ToolChoiceMode.NONE)
    @classmethod
    def required(cls) -> CanonicalToolChoice:
        return cls(mode=ToolChoiceMode.REQUIRED)
    @classmethod
    def named(cls, tool_name: str) -> CanonicalToolChoice:
        return cls(mode=ToolChoiceMode.NAMED, name=tool_name)


@dataclass
class CanonicalGenerationOptions:
    max_output_tokens: int = 4096
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] = field(default_factory=list)
    stream: bool = True


@dataclass
class RequestedCapabilities:
    tools: bool = False
    parallel_tools: bool = False
    reasoning: bool = False
    images: bool = False
    structured_output: bool = False


@dataclass
class CanonicalUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    confidence: str = "estimated"  # exact | backend-reported | tokenizer-estimated | heuristic


@dataclass
class CanonicalRequest:
    protocol_version: str = ABI_VERSION
    request_id: str = ""
    session_id: str = ""
    model: CanonicalModelReference = field(default_factory=CanonicalModelReference)
    system: list = field(default_factory=list)
    messages: list[CanonicalMessage] = field(default_factory=list)
    tools: list[CanonicalTool] = field(default_factory=list)
    tool_choice: CanonicalToolChoice = field(default_factory=CanonicalToolChoice.auto)
    generation: CanonicalGenerationOptions = field(default_factory=CanonicalGenerationOptions)
    requested_capabilities: RequestedCapabilities = field(default_factory=RequestedCapabilities)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalResponse:
    request_id: str = ""
    model: CanonicalModelReference = field(default_factory=CanonicalModelReference)
    content: list = field(default_factory=list)
    stop_reason: CanonicalStopReason = CanonicalStopReason.END_TURN
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)


# ─── JSON Schema generation ────────────────────────────────────────────────


def abi_json_schema() -> dict[str, Any]:
    """Generate a valid JSON Schema for the Agent ABI CanonicalRequest.

    This is used for schema validation, golden test fixtures, and
    cross-language bindings.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "interop-agent-abi-v1",
        "title": "Interop Agent ABI v1 — CanonicalRequest",
        "description": "Versioned internal protocol for the Interop Agent Compatibility Gateway",
        "type": "object",
        "properties": {
            "protocolVersion": {
                "type": "string",
                "const": ABI_VERSION,
                "description": "Must be 'interop.agent.v1'",
            },
            "requestId": {"type": "string", "pattern": "^req_[a-f0-9]+$"},
            "sessionId": {"type": "string", "pattern": "^sess_[a-f0-9]+$"},
            "model": {
                "type": "object",
                "properties": {
                    "requestedName": {"type": "string"},
                    "resolvedName": {"type": "string"},
                    "backend": {"type": "string"},
                    "profile": {"type": "string"},
                },
                "required": ["requestedName"],
            },
            "system": {
                "type": "array",
                "items": {"$ref": "#/definitions/ContentBlock"},
            },
            "messages": {
                "type": "array",
                "items": {"$ref": "#/definitions/Message"},
            },
            "tools": {
                "type": "array",
                "items": {"$ref": "#/definitions/Tool"},
            },
            "toolChoice": {"$ref": "#/definitions/ToolChoice"},
            "generation": {"$ref": "#/definitions/GenerationOptions"},
            "requestedCapabilities": {"$ref": "#/definitions/RequestedCapabilities"},
            "metadata": {"type": "object"},
        },
        "required": ["protocolVersion", "requestId"],
        "definitions": {
            "Message": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant"]},
                    "content": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/ContentBlock"},
                    },
                },
                "required": ["role"],
            },
            "ContentBlock": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["text", "reasoning", "image", "tool_call", "tool_result"],
                    },
                    "text": {"type": "string"},
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "content": {"type": "string"},
                    "isError": {"type": "boolean"},
                    "visibility": {"type": "string", "enum": ["hidden", "summary", "exposed"]},
                    "signature": {"type": "string"},
                },
                "required": ["type"],
            },
            "Tool": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "inputSchema": {"type": "object"},
                    "strict": {"type": "boolean"},
                },
                "required": ["name"],
            },
            "ToolChoice": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["auto", "none", "required", "named"]},
                    "name": {"type": "string"},
                },
                "required": ["mode"],
            },
            "GenerationOptions": {
                "type": "object",
                "properties": {
                    "maxOutputTokens": {"type": "integer", "minimum": 1},
                    "temperature": {"type": "number"},
                    "topP": {"type": "number"},
                    "stop": {"type": "array", "items": {"type": "string"}},
                    "stream": {"type": "boolean"},
                },
            },
            "RequestedCapabilities": {
                "type": "object",
                "properties": {
                    "tools": {"type": "boolean"},
                    "parallelTools": {"type": "boolean"},
                    "reasoning": {"type": "boolean"},
                    "images": {"type": "boolean"},
                    "structuredOutput": {"type": "boolean"},
                },
            },
        },
    }