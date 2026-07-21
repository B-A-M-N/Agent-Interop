# The canonical internal representation for Interop.
# Every incoming request from any protocol is normalized into these types,
# then model adapters convert them back into the exact format the model expects.

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

__all__ = [
    # Enums
    "CapabilityLevel",
    "ToolCallDialect",
    "ProtocolKind",
    "BackendKind",
    "RepairAction",
    # Core dataclasses
    "InteropConfig",
    "ModelProfile",
    "CanonicalTool",
    "ToolCall",
    "ToolResult",
    "ContentBlock",
    "AgentMessage",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalEvent",
    "ServerInfo",
    "BackendRequest",
    "BackendEvent",
    # Helpers
    "tool_from_openai",
    "tool_from_anthropic",
    "tool_to_openai",
    "tool_to_anthropic",
]


# ─── Capability classification ─────────────────────────────────────────────


class CapabilityLevel(Enum):
    """Agent capability levels — L0 through L4."""

    L0 = "L0"  # Chat only — no tool support
    L1 = "L1"  # Forced single-tool calling (prompt-injected)
    L2 = "L2"  # Automatic single-tool calling
    L3 = "L3"  # Parallel and sequential tool calls
    L4 = "L4"  # Reliable coding-agent operation

    def __lt__(self, other: CapabilityLevel) -> bool:
        levels = ["L0", "L1", "L2", "L3", "L4"]
        return levels.index(self.value) < levels.index(other.value)

    def __le__(self, other: CapabilityLevel) -> bool:
        return self == other or self < other

    def __gt__(self, other: CapabilityLevel) -> bool:
        levels = ["L0", "L1", "L2", "L3", "L4"]
        return levels.index(self.value) > levels.index(other.value)

    def __ge__(self, other: CapabilityLevel) -> bool:
        return self == other or self > other


class ToolCallDialect(Enum):
    """Known model-native tool-call dialects."""

    HERMES = "hermes"             # <tool_call>JSON</tool_call>
    OPENAI_NATIVE = "openai"      # function_call in API
    MISTRAL = "mistral"           # [TOOL_CALLS] or <tool_calls>
    QWEN = "qwen"                 # <tool>\n...\n</tool>
    LLAMA = "llama"               # <|python_tag|> or built-in tool calls
    DEEPSEEK = "deepseek"         # <｜tool▁call▁begin｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜> format
    ANTHROPIC = "anthropic"       # tool_use content blocks
    GENERIC_JSON = "generic"       # heuristic JSON extraction


class ProtocolKind(Enum):
    """Client-facing protocol variants."""

    ANTHROPIC_MESSAGES = "anthropic_messages"   # /v1/messages
    OPENAI_CHAT = "openai_chat"                 # /v1/chat/completions
    OPENAI_RESPONSES = "openai_responses"       # /v1/responses


class BackendKind(Enum):
    """Supported inference backends."""

    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    OPENAI_PROXY = "openai_proxy"  # generic OpenAI-compatible


class RepairAction(Enum):
    """What repair was applied to a tool call."""

    NONE = "none"
    TRIVIAL_JSON = "trivial_json"        # Extra braces, trailing comma
    MISSING_ID = "missing_id"            # Synthesized tool_use id
    SCHEMA_MINOR = "schema_minor"        # Key name typo, wrong type coerce
    CONSTRAINED_REGENERATION = "regenerated"  # Re-ran with guided decoding
    UNREPAIRABLE = "unreparable"         # Deemed unrecoverable


# ─── Core structures ────────────────────────────────────────────────────────


@dataclass
class InteropConfig:
    """Global configuration for an Interop instance."""

    host: str = "127.0.0.1"
    port: int = 8090
    backend: BackendKind = BackendKind.OLLAMA
    backend_url: str = "http://127.0.0.1:11434"
    backend_api_key: str | None = None
    model: str = "qwen3-coder"
    compatibility_profile: str = "auto"
    allow_extra_models: bool = True
    probe_on_startup: bool = True
    log_level: str = "info"


@dataclass
class ModelProfile:
    """Declarative profile describing a model's capabilities and how to talk to it."""

    model: str
    template: str = ""
    tool_parser: str = ""
    reasoning_parser: str = ""
    tool_dialect: ToolCallDialect = ToolCallDialect.GENERIC_JSON
    supports: set[str] = field(default_factory=lambda: {
        "automatic_tools",
        "context_length",
    })
    context_length: int = 8192
    capabilities: CapabilityLevel = CapabilityLevel.L0
    repair_strategies: dict[str, str] = field(default_factory=dict)
    prompt_template: str = ""
    stop_tokens: list[str] = field(default_factory=list)
    parallel_tools: bool = False
    supports_images: bool = False
    supports_thinking: bool = False

    def has(self, feature: str) -> bool:
        return feature in self.supports


@dataclass
class CanonicalTool:
    """Normalized tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    strict: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Return as an OpenAI-style function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        """Return as an Anthropic tool definition."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass
class ToolCall:
    """A tool invocation parsed from model output."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw: str = ""
    dialect: ToolCallDialect = ToolCallDialect.GENERIC_JSON
    repair: RepairAction = RepairAction.NONE

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    call_id: str
    tool_name: str
    content: str
    is_error: bool = False


@dataclass
class ContentBlock:
    """A content block in a message — text, tool_use, tool_result, thinking."""

    type: Literal["text", "tool_use", "tool_result", "thinking"]
    text: str | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    thinking: str | None = None
    signature: str | None = None  # Anthropic thinking signature


@dataclass
class AgentMessage:
    """A single message in the conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentBlock] = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class CanonicalRequest:
    """Normalized request that every adapter produces."""

    system: str = ""
    messages: list[AgentMessage] = field(default_factory=list)
    tools: list[CanonicalTool] = field(default_factory=list)
    tool_choice: str | dict = "auto"
    max_tokens: int = 4096
    temperature: float = 0.0
    stream: bool = True
    capabilities_requested: set[str] = field(default_factory=set)
    conversation_id: str | None = None
    previous_response_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def has_tools(self) -> bool:
        return len(self.tools) > 0


@dataclass
class CanonicalResponse:
    """Normalized response from the model."""

    content: list[ContentBlock] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | stop_sequence
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    id: str = ""

    @property
    def text(self) -> str:
        parts: list[str] = []
        for block in self.content:
            if block.type == "text" and block.text:
                parts.append(block.text)
            if block.type == "thinking" and block.text:
                parts.append(block.text)
        return "\n".join(parts)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


@dataclass
class CanonicalEvent:
    """A streamed event in canonical form."""

    type: Literal[
        "text", "text_delta", "thinking", "thinking_delta",
        "thinking_signature", "tool_use", "tool_use_delta",
        "content_block_start", "content_block_delta",
        "content_block_stop", "message_stop",
    ]
    content_block: ContentBlock | None = None
    index: int = 0
    partial: str = ""


@dataclass
class ServerInfo:
    """Information about the gateway server, returned on health check."""

    version: str = "0.1.0"
    model: str = ""
    profile: str | None = None
    level: str = "L0"
    level_description: str = ""
    supports: list[str] = field(default_factory=list)


@dataclass
class BackendRequest:
    """The final request sent to an inference backend after translation."""

    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    stream: bool = True
    timeout: float = 120.0


@dataclass
class BackendEvent:
    """A streaming event from the backend, partially decoded."""

    raw: str = ""
    data: dict[str, Any] | None = None
    event_type: str = ""
    done: bool = False


# ─── Tool definition helpers ────────────────────────────────────────────────


def tool_from_openai(spec: dict[str, Any]) -> CanonicalTool:
    """Convert an OpenAI function definition to CanonicalTool."""
    fn = spec.get("function", spec)
    return CanonicalTool(
        name=fn["name"],
        description=fn.get("description", ""),
        parameters=fn.get("parameters", {"type": "object", "properties": {}}),
        strict=spec.get("strict", False),
    )


def tool_from_anthropic(spec: dict[str, Any]) -> CanonicalTool:
    """Convert an Anthropic tool definition to CanonicalTool."""
    return CanonicalTool(
        name=spec["name"],
        description=spec.get("description", ""),
        parameters=spec.get("input_schema", {"type": "object", "properties": {}}),
    )


def tool_to_openai(tool: CanonicalTool) -> dict[str, Any]:
    return tool.to_json_schema()


def tool_to_anthropic(tool: CanonicalTool) -> dict[str, Any]:
    return tool.to_anthropic()