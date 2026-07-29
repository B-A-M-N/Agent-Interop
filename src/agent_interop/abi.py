"""Interop Agent ABI — versioned canonical representation and JSON Schema.

The Agent ABI defines the versioned internal protocol that all adapters
translate to/from. It is independent of any specific client or backend.

All types here match the spec's TypeScript CanonicalRequest/CanonicalResponse
representation, converted to Python dataclasses with JSON Schema generation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from agent_interop.enums import ProtocolKind, ToolCallDialect

ABI_VERSION = "interop.agent.v1"


# ─── Enums re-exported from enums.py for convenience ─────────────────────────

__all__ = [
    "ABI_VERSION",
    # Enums
    "CanonicalContentBlock",
    "CanonicalError",
    "CanonicalEvent",
    "CanonicalGenerationOptions",
    "CanonicalImageBlock",
    "CanonicalMessage",
    "CanonicalModelReference",
    "CanonicalReasoningBlock",
    "CanonicalRefusalBlock",
    "CanonicalRequest",
    "CanonicalResponse",
    "CanonicalStopReason",
    "CanonicalTextBlock",
    "CanonicalTool",
    "CanonicalToolCallBlock",
    "CanonicalToolChoice",
    "CanonicalToolResultBlock",
    "CanonicalUnknownBlock",
    "CanonicalUsage",
    "ContentBlockType",
    "ProtocolKind",
    "RawToolCallCandidate",
    "RepairOutcome",
    "RepairStatus",
    "RepairStep",
    "RequestedCapabilities",
    "SchemaIssue",
    "ToolCallDecision",
    "ToolCallDialect",
    "ToolCallProvenance",
    "ToolChoiceMode",
    # Helpers
    "abi_json_schema",
    "canonical_tool_choice",
    "new_request_id",
    "new_session_id",
    "new_tool_call_id",
    "tool_from_anthropic",
    "tool_from_openai",
    "tool_to_anthropic",
    "tool_to_openai",
]


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def new_session_id() -> str:
    return f"sess_{uuid.uuid4().hex[:16]}"


def new_tool_call_id() -> str:
    return f"tc_{uuid.uuid4().hex[:12]}"


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
    REFUSAL = "refusal"
    UNKNOWN = "unknown"


# ─── Canonical role type ─────────────────────────────────────────────────────

CanonicalRole = Literal[
    "system",
    "developer",
    "user",
    "assistant",
    "tool",
]


class ToolChoiceMode(str, Enum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


class RepairStatus(str, Enum):
    """Status of a tool-call validation or repair operation."""

    VALID_UNCHANGED = "valid_unchanged"
    REPAIRED = "repaired"
    REGENERATED = "regenerated"
    REJECTED = "rejected"


# ─── Repair / Transaction types ────────────────────────────────────────────


@dataclass
class RepairStep:
    """A single step in the repair pipeline."""

    rule: str = ""
    path: str = ""
    message: str = ""
    targeted_path: str = ""
    before: Any = None
    after: Any = None
    semantic_risk: str = "low"
    evidence_source: str = ""
    confidence: float = 1.0


@dataclass
class RepairCursor:
    """Scoped context for a single repair rule invocation.

    Each rule receives exactly one cursor targeting one specific schema
    issue.  The rule proposes at most one mutation on the scoped path.
    The pipeline revalidates the full object after each proposal.
    """

    issue: SchemaIssue = field(default_factory=lambda: SchemaIssue())
    instance_path: list[str | int] = field(default_factory=list)
    schema_path: list[str | int] = field(default_factory=list)
    parent_instance: Any = None
    current_value: Any = None
    parent_schema: dict[str, Any] = field(default_factory=dict)
    target_schema: dict[str, Any] = field(default_factory=dict)
    tool: CanonicalTool | None = None
    client_id: str | None = None


@dataclass
class SchemaIssue:
    """A JSON Schema validation issue."""

    path: list[str | int] = field(default_factory=list)
    keyword: str = ""
    message: str = ""
    expected: str = ""
    actual: str = ""
    absolute_schema_path: list[str | int] = field(default_factory=list)
    branch_identity: str = ""


@dataclass
class RepairOutcome:
    """Result of repairing a single tool call."""

    status: RepairStatus = RepairStatus.VALID_UNCHANGED
    call_name: str = ""
    accepted: dict[str, Any] | None = None
    error: str = ""
    initial_issues: list[SchemaIssue] = field(default_factory=list)
    final_issues: list[SchemaIssue] = field(default_factory=list)
    steps: list[RepairStep] = field(default_factory=list)

    @property
    def is_accepted(self) -> bool:
        return self.status in (RepairStatus.VALID_UNCHANGED, RepairStatus.REPAIRED, RepairStatus.REGENERATED)

    @property
    def was_repaired(self) -> bool:
        return self.status in (RepairStatus.REPAIRED, RepairStatus.REGENERATED)


@dataclass
class ToolCallProvenance:
    """Metadata about how a tool call was extracted/recovered."""

    source: str = "model_output"  # model_output | prompted | streaming | regeneration
    dialect: str = ""
    raw_name: str = ""
    raw_arguments: str = ""


@dataclass(frozen=True)
class RawToolCallCandidate:
    """Raw, unvalidated tool call candidate from model output.

    The candidate should be immutable. Repair produces a decision; it should not
    rewrite the evidence object. raw_arguments must retain the upstream value
    exactly. Do not normalize it in a codec.
    """

    id: str | None = None
    name: str | None = None
    raw_arguments: str | dict[str, Any] | list[Any] | None = None
    source_protocol: str = ""
    source_index: int | None = None
    source_text: str = ""
    raw_name: str | None = None
    choice_index: int = 0  # Index of the choice this call belongs to
    tool_index: int = 0    # Index of the tool within the choice
    provenance: ToolCallProvenance = field(default_factory=ToolCallProvenance)
    execution_nonce: str | None = None
    """Value of a top-level ``interop_call_id`` key, if the source text
    carried one. Used only by the ambiguous whole_message_json/auto
    recovery path (see extraction.py's ExtractorRegistry.extract) to
    verify a recovered candidate against the live per-request nonce
    build_invocation_plan() issued — unrelated to ``id``, which is the
    call's own correlation id."""

    @classmethod
    def from_parsed(
        cls,
        name: str,
        arguments: Any,
        call_id: str = "",
        dialect: str = "",
    ) -> RawToolCallCandidate:
        return cls(
            name=name,
            raw_arguments=arguments,
            id=call_id,
            source_protocol=dialect,
            provenance=ToolCallProvenance(source="model_output", dialect=dialect, raw_name=name),
        )


@dataclass
class ToolCallCorrection:
    """Structured correction data for a rejected tool call.

    Surfaces the exact failure location, observed/expected types, and
    admissible alternatives so that callers can either attempt bounded
    regeneration or return model-readable error details.
    """

    tool_name: str = ""
    candidate_id: str = ""
    issue_path: str = ""
    schema_keyword: str = ""
    observed_type: str = ""
    expected_type: str = ""
    allowed_values: list[str] = field(default_factory=list)
    message: str = ""
    retryable: bool = False


@dataclass
class ToolCallDecision:
    """Final decision about a tool call after the full pipeline."""

    candidate: RawToolCallCandidate = field(default_factory=RawToolCallCandidate)
    outcome: RepairOutcome = field(default_factory=RepairOutcome)
    accepted_block: CanonicalToolCallBlock | None = None
    correction: ToolCallCorrection | None = None

    @property
    def is_accepted(self) -> bool:
        return self.outcome.is_accepted and self.accepted_block is not None

    @property
    def is_rejected(self) -> bool:
        return not self.is_accepted


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
    provider_metadata: ProviderMetadata | None = None


class MetadataForwardingPolicy(str, Enum):
    """Controls how provider metadata is forwarded to the client.

    PRESERVE: Always forward (e.g., encrypted fields needed for replay).
    PRESERVE_IF_COMPATIBLE: Forward only when the destination protocol
        explicitly declares support for this metadata kind.
    DROP: Never forward to the client.
    """

    PRESERVE = "preserve"
    PRESERVE_IF_COMPATIBLE = "preserve_if_compatible"
    DROP = "drop"


@dataclass
class ProviderMetadata:
    """Opaque provider-specific metadata that must survive round-trips.

    Carried on content blocks, tool calls, and messages so that
    provider codecs can preserve replay-critical fields (e.g.
    DeepSeek reasoning_content, Gemini thought_signature) without
    leaking them into unrelated protocols.

    Rules:
    - Only the originating codec may create provider metadata.
    - Only codecs with an explicit mapping may consume it.
    - Unknown metadata is preserved internally but not sprayed
      into unrelated APIs.
    - Sensitive or encrypted fields must never enter ordinary logs.
    """

    origin_protocol: str = ""
    origin_provider: str = ""
    origin_model: str = ""
    metadata_kind: str = ""
    opaque_value: Any = None
    required_for_replay: bool = False
    forwarding_policy: MetadataForwardingPolicy = MetadataForwardingPolicy.PRESERVE_IF_COMPATIBLE


def should_forward_provider_metadata(
    metadata: ProviderMetadata | None,
    destination_supported_kinds: frozenset[str],
) -> bool:
    """Decide whether provider metadata should be forwarded to a destination.

    Args:
        metadata: The provider metadata to check, or None.
        destination_supported_kinds: The frozenset of metadata kinds the
            destination explicitly supports.

    Returns:
        True if the metadata should be forwarded.
    """
    if metadata is None:
        return False

    policy = metadata.forwarding_policy
    if policy == MetadataForwardingPolicy.DROP:
        return False

    if policy == MetadataForwardingPolicy.PRESERVE:
        return True

    # PRESERVE_IF_COMPATIBLE: only if destination supports this kind
    return metadata.metadata_kind in destination_supported_kinds


def should_forward_metadata(
    metadata: ProviderMetadata | None,
    destination: Any,
) -> bool:
    """Legacy wrapper for should_forward_provider_metadata.

    Args:
        metadata: The provider metadata to check, or None.
        destination: The client adapter or codec that would receive the
            metadata. Must have a ``supported_provider_metadata``
            frozenset attribute.

    Returns:
        True if the metadata should be forwarded.
    """
    if metadata is None:
        return False

    policy = metadata.forwarding_policy
    if policy == MetadataForwardingPolicy.DROP:
        return False
    if policy == MetadataForwardingPolicy.PRESERVE:
        return True

    # PRESERVE_IF_COMPATIBLE: check destination capability
    supported = frozenset(
        getattr(destination, "supported_provider_metadata", frozenset())
    )
    return metadata.metadata_kind in supported


@dataclass
class CanonicalToolCallBlock:
    type: Literal["tool_call"] = "tool_call"
    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | dict[str, Any] | list[Any] | None = None
    arguments_validated: bool = True
    provider_metadata: ProviderMetadata | None = None


@dataclass
class CanonicalToolResultBlock:
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str = ""
    content: str | list[CanonicalContentBlock] = ""
    is_error: bool = False
    provider_metadata: ProviderMetadata | None = None


@dataclass
class CanonicalImageBlock:
    type: Literal["image"] = "image"
    source_type: str = ""
    media_type: str = ""
    data: str = ""
    url: str = ""
    detail: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalRefusalBlock:
    type: Literal["refusal"] = "refusal"
    refusal: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalUnknownBlock:
    type: Literal["unknown"] = "unknown"
    source_type: str = ""
    raw: Any = None  # Any JSON value — preserves unknown content losslessly


# Union type for any content block in the ABI
CanonicalContentBlock = (
    CanonicalTextBlock
    | CanonicalReasoningBlock
    | CanonicalImageBlock
    | CanonicalToolCallBlock
    | CanonicalToolResultBlock
    | CanonicalRefusalBlock
    | CanonicalUnknownBlock
)


@dataclass
class CanonicalMessage:
    role: CanonicalRole = "user"
    content: list[CanonicalContentBlock] = field(default_factory=list)
    provider_metadata: ProviderMetadata | None = None


@dataclass
class CanonicalTool:
    name: str = ""
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to OpenAPI-style function tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
            "strict": self.strict,
        }


def canonical_tool_choice(mode: str, name: str = "") -> CanonicalToolChoice:
    """Create a CanonicalToolChoice from a mode string and optional tool name."""
    mode_map = {
        "auto": ToolChoiceMode.AUTO,
        "none": ToolChoiceMode.NONE,
        "required": ToolChoiceMode.REQUIRED,
        "named": ToolChoiceMode.NAMED,
    }
    return CanonicalToolChoice(mode=mode_map.get(mode, ToolChoiceMode.AUTO), name=name)


def tool_to_anthropic(tool: CanonicalTool) -> dict[str, Any]:
    """Convert a CanonicalTool to an Anthropic tool definition."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def tool_to_openai(tool: CanonicalTool) -> dict[str, Any]:
    """Convert a CanonicalTool to an OpenAI function definition."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
        "strict": tool.strict,
    }


def tool_from_openai(spec: dict[str, Any]) -> CanonicalTool:
    """Convert an OpenAI function definition to abi.CanonicalTool."""
    fn = spec.get("function", spec)
    return CanonicalTool(
        name=fn["name"],
        description=fn.get("description", ""),
        input_schema=fn.get("parameters", {"type": "object", "properties": {}}),
        strict=spec.get("strict", False),
    )


def tool_from_anthropic(spec: dict[str, Any]) -> CanonicalTool:
    """Convert an Anthropic tool definition to abi.CanonicalTool."""
    return CanonicalTool(
        name=spec["name"],
        description=spec.get("description", ""),
        input_schema=spec.get("input_schema", {"type": "object", "properties": {}}),
    )


# ─── Remaining dataclasses ─────────────────────────────────────────────────────


@dataclass
class CanonicalToolChoice:
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    name: str = ""

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
    confidence: str = "estimated"


@dataclass
class CanonicalRequest:
    protocol_version: str = ABI_VERSION
    request_id: str = ""
    session_id: str = ""
    model: CanonicalModelReference = field(default_factory=CanonicalModelReference)
    system: list[CanonicalContentBlock] = field(default_factory=list)
    messages: list[CanonicalMessage] = field(default_factory=list)
    tools: list[CanonicalTool] = field(default_factory=list)
    tool_choice: CanonicalToolChoice = field(default_factory=CanonicalToolChoice.auto)
    generation: CanonicalGenerationOptions = field(default_factory=CanonicalGenerationOptions)
    requested_capabilities: RequestedCapabilities = field(default_factory=RequestedCapabilities)
    previous_response_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CanonicalError:
    """Canonical error representation, independent of any provider protocol.

    Carries a stable error code, human message, retry indicator,
    optional upstream HTTP status, and request correlation ID.
    """

    code: str = ""
    message: str = ""
    retryable: bool = False
    upstream_status: int | None = None
    request_id: str = ""
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CanonicalResponse:
    request_id: str = ""
    response_id: str = ""
    model: CanonicalModelReference = field(default_factory=CanonicalModelReference)
    content: list[CanonicalContentBlock] = field(default_factory=list)
    stop_reason: CanonicalStopReason = CanonicalStopReason.END_TURN
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    extra: dict[str, Any] = field(default_factory=dict)
    error: CanonicalError | None = None


@dataclass
class CanonicalEvent:
    """A streamed event in canonical form."""

    type: Literal[
        "text", "text_delta", "thinking", "thinking_delta",
        "thinking_signature", "tool_use", "tool_use_delta",
        "content_block_start", "content_block_delta",
        "content_block_stop", "message_start", "message_stop", "error",
        "usage_update",
    ]
    content_block: CanonicalContentBlock | None = None
    index: int = 0
    partial: str = ""
    error: CanonicalError | None = None
    stop_reason: CanonicalStopReason | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class RepairOperation(str, Enum):
    """Explicit operation type for a repair proposal.

    Using explicit operations prevents ambiguity in the pipeline about
    what change a proposal intends — the pipeline enforces that only
    the declared paths are affected.
    """

    SET = "set"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class RepairProposal:
    """A single, scoped mutation produced by a repair rule.

    Each rule returns zero or one ``RepairProposal`` for a single
    ``RepairCursor``.  The proposal affects exactly one instance path
    and must satisfy the acceptance contract enforced by the pipeline:

    1. The targeted issue disappeared.
    2. No new issue was introduced.
    3. The mutation affected only the declared path.
    4. The complete object now validates better than before.
    5. Ambiguous schema branches were not guessed.

    ``before`` and ``after`` are recorded so the pipeline can build a
    rich audit trail and so the rule can be rejected if the proposal
    violates the contract.
    """

    rule_id: str = ""
    operation: RepairOperation = RepairOperation.SET
    target_path: list[str | int] = field(default_factory=list)
    source_path: list[str | int] | None = None
    before: Any = None
    after: Any = None
    delete: bool = False  # True means remove the key at target_path
    issue_identity: str = ""
    semantic_risk: str = "low"
    evidence_source: str = ""
    confidence: float = 1.0
    message: str = ""


@dataclass
class CanonicalToolCorrection:
    """Structured correction payload for rejected tool calls.

    Emitted as an error event before message_stop so that coding agents
    receive model-actionable feedback about what went wrong, rather than
    a bare INVALID_OUTPUT stop reason with no details.
    """

    request_id: str = ""
    candidate_id: str = ""
    raw_tool_name: str = ""
    canonical_tool_name: str = ""
    issue_paths: list[str] = field(default_factory=list)
    repair_steps_attempted: list[str] = field(default_factory=list)
    correction_instruction: str = ""
    retryable: bool = True


def correction_from_decision(
    decision: ToolCallDecision,
    request_id: str = "",
) -> CanonicalToolCorrection:
    """Build a structured correction from a rejected ToolCallDecision.

    Used by both streaming and nonstreaming rejection paths to give
    coding agents actionable feedback about what went wrong.
    """
    outcome = decision.outcome
    candidate = decision.candidate

    issue_paths: list[str] = []
    if outcome.initial_issues:
        issue_paths = [
            ".".join(str(s) for s in iss.path) or "root"
            for iss in outcome.initial_issues[:5]
        ]

    repair_steps = [step.rule for step in outcome.steps] if outcome.steps else []

    correction_instruction = ""
    if outcome.error:
        correction_instruction = outcome.error
    elif outcome.final_issues:
        correction_instruction = "; ".join(
            iss.message for iss in outcome.final_issues[:3]
        )

    return CanonicalToolCorrection(
        request_id=request_id,
        candidate_id=candidate.id or "",
        raw_tool_name=(candidate.name or "") if hasattr(candidate, "name") else "",
        canonical_tool_name=outcome.call_name or "",
        issue_paths=issue_paths,
        repair_steps_attempted=repair_steps,
        correction_instruction=correction_instruction,
        retryable=True,
    )


# ─── JSON Schema generation ──────────────────────────────────────────────────


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
            "previousResponseId": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
        },
        "required": ["protocolVersion", "requestId"],
        "definitions": {
            "Message": {
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "developer", "user", "assistant", "tool"],
                    },
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
                        "enum": ["text", "reasoning", "image", "tool_call", "tool_result", "refusal", "unknown"],
                    },
                    "text": {"type": "string"},
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "arguments": {"type": "object"},
                    "content": {"type": "string"},
                    "isError": {"type": "boolean"},
                    "visibility": {"type": "string", "enum": ["hidden", "summary", "exposed"]},
                    "signature": {"type": "string"},
                    "sourceType": {"type": "string"},
                    "mediaType": {"type": "string"},
                    "data": {"type": "string"},
                    "url": {"type": "string"},
                    "detail": {"type": ["string", "null"]},
                    "refusal": {"type": "string"},
                    "raw": {"type": "object"},
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