# Compatibility facade — re-exports from canonical ABI.
# This module is deprecated; new code should import from agent_interop.abi directly.
#
# The canonical internal representation for Interop lives in:
# - abi.py: semantic request/response/message/event contracts
# - config.py: route/upstream/tool-mode configuration
# - compat.py: legacy request/message/backend classes

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_interop import __version__

# Re-export enums from abi.py
# Re-export dataclasses from abi.py
# Re-export helpers from abi.py
from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalEvent,
    CanonicalGenerationOptions,
    CanonicalImageBlock,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalReasoningBlock,
    CanonicalRefusalBlock,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    CanonicalUnknownBlock,
    CanonicalUsage,
    ContentBlockType,
    ProtocolKind,
    RawToolCallCandidate,
    RepairOutcome,
    RepairStatus,
    RepairStep,
    RequestedCapabilities,
    SchemaIssue,
    ToolCallDecision,
    ToolCallDialect,
    ToolCallProvenance,
    ToolChoiceMode,
    abi_json_schema,
    new_request_id,
    new_session_id,
    new_tool_call_id,
    tool_from_anthropic,
    tool_from_openai,
    tool_to_anthropic,
    tool_to_openai,
)

# ─── Legacy enum (stays in types.py) ──────────────────────────────────────

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


class RepairAction(Enum):
    """What repair was applied to a tool call."""

    NONE = "none"
    TRIVIAL_JSON = "trivial_json"
    MISSING_ID = "missing_id"
    SCHEMA_MINOR = "schema_minor"
    CONSTRAINED_REGENERATION = "regenerated"
    UNREPAIRABLE = "unreparable"


# ─── Legacy dataclasses (for backward compatibility) ───────────────────────

@dataclass
class ContentBlock:
    """v1 backward-compatible ContentBlock."""

    type: str  # "text" | "tool_use" | "tool_result" | "thinking"
    text: str | None = None
    tool_call: Any = None
    tool_result: Any = None
    thinking: str | None = None
    signature: str | None = None


@dataclass
class ModelProfile:
    """Model profile for backward compatibility."""

    model: str
    template: str = ""
    tool_parser: str = ""
    reasoning_parser: str | None = None
    tool_dialect: ToolCallDialect = ToolCallDialect.GENERIC_JSON
    supports: set[str] = field(default_factory=lambda: {"automatic_tools", "context_length"})
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
class InteropConfig:
    """Legacy configuration for backward compatibility."""

    host: str = "127.0.0.1"
    port: int = 8090
    backend: Any = None
    backend_url: str = "http://127.0.0.1:11434"
    backend_api_key: str | None = None
    model: str = "qwen3-coder"
    compatibility_profile: str = "auto"
    allow_extra_models: bool = True
    probe_on_startup: bool = True
    log_level: str = "info"
    backend_timeout: float = 120.0


@dataclass
class ServerInfo:
    """Information about the gateway server, returned on health check.

    Deliberately does NOT carry a "capability level"/"supports" summary —
    those existed here only as permanently-hardcoded placeholder values
    (level=0, supports=["multi_route"]) regardless of actual
    configuration. Real, per-route capability data belongs at
    /v1/capabilities, which derives it honestly from resolved profile +
    codec metadata instead of fabricating a single number.
    """

    version: str = field(default_factory=lambda: __version__)
    model: str = ""
    routes: list[dict[str, Any]] = field(default_factory=list)


# ─── Import config types ───────────────────────────────────────────────────
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairConfig,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)

__all__ = [
    "CanonicalContentBlock",
    "CanonicalEvent",
    "CanonicalGenerationOptions",
    "CanonicalImageBlock",
    "CanonicalMessage",
    # Dataclasses from abi.py
    "CanonicalModelReference",
    "CanonicalReasoningBlock",
    "CanonicalRefusalBlock",
    "CanonicalRequest",
    "CanonicalResponse",
    # Enums from abi.py
    "CanonicalStopReason",
    "CanonicalTextBlock",
    "CanonicalTool",
    "CanonicalToolCallBlock",
    "CanonicalToolChoice",
    "CanonicalToolResultBlock",
    "CanonicalUnknownBlock",
    "CanonicalUsage",
    # Legacy enums
    "CapabilityLevel",
    # Legacy dataclasses
    "ContentBlock",
    "ContentBlockType",
    "InteropConfig",
    "InteropServerConfig",
    "ModelProfile",
    "ModelRoute",
    "ProtocolKind",
    "RawToolCallCandidate",
    "RepairAction",
    "RepairConfig",
    "RepairOutcome",
    "RepairStatus",
    "RepairStep",
    "RequestedCapabilities",
    "SchemaIssue",
    "ToolCallDecision",
    "ToolCallDialect",
    "ToolCallProvenance",
    "ToolChoiceMode",
    # Config types
    "ToolMode",
    "TranslationMode",
    "UpstreamConfig",
    "UpstreamKind",
    "UpstreamProtocol",
    # Helpers from abi.py
    "abi_json_schema",
    "new_request_id",
    "new_session_id",
    "new_tool_call_id",
    "tool_from_anthropic",
    "tool_from_openai",
    "tool_to_anthropic",
    "tool_to_openai",
]