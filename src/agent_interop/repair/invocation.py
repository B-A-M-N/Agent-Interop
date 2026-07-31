"""InvocationPlan — capability resolution and tool presentation strategy.

Determines how tools are presented to the model and how output is parsed,
based on route configuration, profile capability, and backend capability.

Key outputs:
- native_tools_enabled: whether the upstream receives native tool definitions
- prompt_contract: textual tool descriptions (for PROMPTED/TEXTUAL modes)
- parser_id: which parser to use for extracting tool calls from text output
- output_envelope: the envelope format expected from the model
- constrained_output: whether the model output is structurally constrained
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agent_interop.abi import CanonicalTool, CanonicalToolChoice
from agent_interop.config import RepairPolicy, ToolMode

# ── Profile capability levels ──────────────────────────────────────────────


class ProfileCapability(str, Enum):
    """What a model profile advertises about its tool capabilities."""

    STRUCTURED = "structured"  # Native tool support (OpenAI, Anthropic, etc.)
    TEXTUAL_DIALECT = "textual_dialect"  # Hermes, Qwen XML, Mistral JSON, etc.
    CHAT_ONLY = "chat_only"  # L0: no tool support at all


# ── InvocationPlan ─────────────────────────────────────────────────────────


class StreamExtractionMode(str, Enum):
    """How tool calls are extracted from streaming output."""

    NATIVE_FRAGMENTS = "native_fragments"
    BUFFER_TEXTUAL_RESPONSE = "buffer_textual_response"
    INCREMENTAL_ENVELOPE = "incremental_envelope"
    NO_EXTRACTION = "no_extraction"


@dataclass(frozen=True)
class InvocationPlan:
    """Complete per-request semantic contract.

    Constructed once per request by build_invocation_plan().
    Carries the full tool strategy including original choice, contract,
    extractor, repair policy, and streaming strategy.
    """

    effective_tool_mode: ToolMode
    """The resolved tool mode after capability negotiation."""

    original_tool_choice: CanonicalToolChoice
    """The tool choice from the client request, preserved for semantics."""

    native_tools_enabled: bool
    """If True, the upstream receives native tool definitions in the request."""

    upstream_tools: tuple[CanonicalTool, ...]
    """Tools actually sent to the upstream (may be empty for PROMPTED)."""

    validation_tools: tuple[CanonicalTool, ...]
    """Original client tools, always available for validation/repair."""

    prompt_contract: str
    """Textual tool descriptions injected into the system prompt (PROMPTED only)."""

    prompt_contract_digest: str
    """SHA-256 digest of the stable prompt contract prefix (for cache diagnostics)."""

    parser_id: str | None
    """Which extractor to use for extracting tool calls from model output."""

    output_envelope: str | None
    """The envelope format expected from the model."""

    stream_extraction_mode: StreamExtractionMode
    """How tool calls are extracted from streaming output."""

    fallback_strategies: tuple[Any, ...] = ()
    """Ordered ExtractionStrategy entries (interop.model.profiles_v2), tried
    in order when the primary extractor (parser_id) finds no candidates.
    Generalizes what were two independent, hardcoded booleans
    (allow_generic_fallback / allow_whole_message_json) — each carries its
    own conditions (skip_when_native_present, allowed_tool_choice_modes)
    instead of every new fallback shape needing its own profile field and
    gateway call-site change. Empty by default: bare/fenced-JSON extraction
    can turn ordinary JSON (config examples, quoted data) into executable
    tool intent, so only profiles that have proven a given fallback safe
    should list it. Typed ``Any`` (not imported from model.profiles_v2) to
    keep this module's existing duck-typed boundary with the profile
    layer — see _profile_capability_from_model below for the same pattern."""

    constrained_output: bool = False
    """If True, the model output is structurally constrained (JSON mode, grammar)."""

    repair_policy: Any = None
    """The repair policy for this request."""

    tool_names: tuple[str, ...] = ()
    """Names of tools included in this plan, for logging/provenance."""

    source_confidence: float = 0.5
    """Confidence in the model profile resolution (0.0–1.0).
    Low confidence (fallback) → gate risky repairs behind evidence only."""

    codec_capabilities: Any = None
    """Upstream codec capabilities (parallel tools, streaming, etc.)."""

    profile_source: str = "fallback"
    """Which resolution tier produced the model profile."""

    execution_nonce: str | None = None
    """Per-request random marker a genuine tool call must echo (as a
    top-level ``interop_call_id`` field) for the ambiguous whole_message_json
    fallback dialect to be trusted under tool_choice=auto. None for the vast
    majority of requests — only set when a fallback strategy actually needs
    it (see _requires_execution_nonce below), which in practice means a
    project/user-tier profile override explicitly re-enabled "auto" for
    that dialect (builtin profiles can never reach this — profiles_v2.py
    rejects them at load time)."""


# ── Mode resolution ────────────────────────────────────────────────────────


def resolve_tool_mode(
    route_tool_mode: ToolMode,
    profile_capability: ProfileCapability | None = None,
) -> ToolMode:
    """Resolve the effective tool mode from route config × profile capability.

    Mode resolution rules (from PROPOSED_CHANGES.md):

    | Route config | Profile capability          | Resolved mode |
    |--------------|-----------------------------|---------------|
    | AUTO         | STRUCTURED                  | NATIVE        |
    | AUTO         | TEXTUAL_DIALECT             | PROMPTED      |
    | AUTO         | CHAT_ONLY / None            | DISABLED      |
    | NATIVE       | (any)                       | NATIVE        |
    | PROMPTED     | (any)                       | PROMPTED      |
    | TEXTUAL      | (any)                       | TEXTUAL       |
    | DISABLED     | (any)                       | DISABLED      |
    """
    if route_tool_mode == ToolMode.AUTO:
        if profile_capability == ProfileCapability.STRUCTURED:
            return ToolMode.NATIVE
        elif profile_capability == ProfileCapability.TEXTUAL_DIALECT:
            return ToolMode.PROMPTED
        else:
            return ToolMode.DISABLED

    # Explicit modes pass through directly
    return route_tool_mode


def resolve_effective_tool_mode(
    route_mode: ToolMode,
    model_profile: Any = None,
    codec_capabilities: Any = None,
) -> ToolMode:
    """Resolve the final tool mode from route config, profile, AND codec.

    This is the single source of truth for tool-mode negotiation. Call
    it — or let build_invocation_plan call it internally — BEFORE any
    InvocationPlan fields are computed, so nothing downstream ever needs
    to patch an already-built plan when the destination codec turns out
    not to support what the route/profile assumed.
    """
    profile_capability = _profile_capability_from_model(model_profile)
    resolved = resolve_tool_mode(route_mode, profile_capability)
    if (
        resolved == ToolMode.NATIVE
        and codec_capabilities is not None
        and not getattr(codec_capabilities, "supports_native_tools", True)
    ):
        resolved = ToolMode.PROMPTED
    return resolved


# ── Deterministic schema serialization ─────────────────────────────────────


def serialize_tool_schema(schema: dict[str, Any]) -> str:
    """Serialize a JSON schema deterministically (sorted keys, stable JSON).

    This ensures repeated requests produce identical prompt text for
    prompt-cache friendliness.
    """
    return json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_tool_descriptions(
    tools: Sequence[CanonicalTool],
    *,
    include_schemas: bool = True,
) -> str:
    """Build the tool descriptions section for a prompted contract.

    Args:
        tools: Sequence of CanonicalTool objects.
        include_schemas: Whether to include the full JSON schema.

    Returns:
        Formatted tool description string.
    """
    parts: list[str] = []
    for tool in tools:
        # Escape tool name and description to prevent contract injection
        name = _escape_xml_attr(tool.name) if tool.name else "unknown"
        desc = _escape_xml_content(tool.description) if tool.description else ""
        schema = tool.input_schema if tool.input_schema else {}
        line = f"<tool name=\"{name}\">"
        if desc:
            line += f"\n{desc}"
        if include_schemas and schema:
            line += f"\n{serialize_tool_schema(schema)}"
        line += "\n</tool>"
        parts.append(line)
    return "\n\n".join(parts)


def _escape_xml_attr(value: str) -> str:
    """Escape a string for safe use as an XML attribute value."""
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


def _escape_xml_content(value: str) -> str:
    """Escape a string for safe use as XML element content."""
    return value.replace("&", "&amp;").replace("<", "&lt;")


# ── Plan construction ──────────────────────────────────────────────────────


def _profile_capability_from_model(model_profile: Any) -> ProfileCapability | None:
    """Derive ProfileCapability from a ResolvedModelProfile."""
    if model_profile is None:
        return None
    if getattr(model_profile, 'supports_native_tools', False):
        return ProfileCapability.STRUCTURED
    if getattr(model_profile, 'supports_textual_tools', False):
        return ProfileCapability.TEXTUAL_DIALECT
    return ProfileCapability.CHAT_ONLY


def _build_choice_instructions(tool_choice: CanonicalToolChoice, tools: Sequence[CanonicalTool]) -> str:
    """Generate tool-choice-specific prompt instructions."""
    mode = tool_choice.mode.value if hasattr(tool_choice.mode, 'value') else str(tool_choice.mode)

    if mode == "required":
        return "\nYou must emit at least one valid tool call before completing."
    elif mode == "named":
        name = tool_choice.name
        return f"\nYou must call `{name}`. Do not call any other tool."
    elif mode == "none":
        return "\nDo not emit a tool call."
    else:  # auto
        return "\nUse a tool only when it is required to complete the request."


def _requires_execution_nonce(fallback_strategies: Sequence[Any]) -> bool:
    """True when a fallback strategy recovers the ambiguous bare/fenced
    whole-message-JSON dialect under tool_choice=auto — the one shape that
    looks identical whether the model intended a real call or was just
    demonstrating JSON. Only a project/user-tier profile override can ever
    reach this (profiles_v2.py rejects a builtin-tier profile that tries to
    enable "auto" for this dialect at load time); this function decides
    whether the plan needs to embed a live, per-request nonce so the
    extractor can tell a genuine call apart from a copied/habitual example.
    """
    return any(
        getattr(fs, "parser_id", None) == "whole_message_json"
        and "auto" in getattr(fs, "allowed_tool_choice_modes", ())
        for fs in fallback_strategies
    )


def build_invocation_plan(
    tools: Sequence[CanonicalTool] | None,
    tool_choice: CanonicalToolChoice,
    route_mode: ToolMode,
    model_profile: Any = None,
    repair_policy: RepairPolicy | None = None,
    codec_capabilities: Any = None,
    *,
    upstream_tools: Sequence[CanonicalTool] | None = None,
    validation_tools: Sequence[CanonicalTool] | None = None,
) -> InvocationPlan:
    """Build an InvocationPlan given tools, choice, and resolved mode.

    Args:
        tools: Legacy shorthand for both the visible and validation tool set.
        upstream_tools: Tools exposed to the model in this attempt.
        validation_tools: Complete client-declared registry used for validation.
        tool_choice: The original tool choice from the client request.
        route_mode: The route's configured tool mode.
        model_profile: Resolved model profile (provides extractor_id, envelope).
        repair_policy: The repair policy for this request.
        codec_capabilities: Upstream codec capabilities; if provided, a NATIVE
            resolution is vetoed to PROMPTED when the codec lacks native-tool
            support. Passing None makes this a pure route×profile resolution.

    Returns:
        A complete InvocationPlan describing how tools are presented.
    """
    if upstream_tools is None:
        upstream_tools = tools or ()
    if validation_tools is None:
        validation_tools = tools if tools is not None else upstream_tools
    visible_tools = tuple(upstream_tools)
    all_validation_tools = tuple(validation_tools)
    resolved = resolve_effective_tool_mode(route_mode, model_profile, codec_capabilities)

    tool_names = tuple(t.name for t in visible_tools if t.name)

    # Determine extractor and envelope from model profile
    extractor_id = getattr(model_profile, 'parser_id', None) if model_profile else None
    output_envelope = getattr(model_profile, 'output_envelope', None) if model_profile else None
    fallback_strategies = getattr(model_profile, 'fallback_strategies', ()) if model_profile else ()
    contract_template_id = getattr(model_profile, 'contract_template_id', None) if model_profile else None

    # Source confidence gates repair aggressiveness (item 86 integration)
    source_confidence = getattr(model_profile, 'source_confidence', 0.5) if model_profile else 0.5
    profile_source = getattr(model_profile, 'source', 'fallback') if model_profile else 'fallback'

    # Determine stream extraction mode
    if resolved == ToolMode.DISABLED:
        stream_mode = StreamExtractionMode.NO_EXTRACTION
    elif resolved == ToolMode.NATIVE:
        stream_mode = StreamExtractionMode.NATIVE_FRAGMENTS
    elif resolved == ToolMode.PROMPTED:
        stream_mode = StreamExtractionMode.BUFFER_TEXTUAL_RESPONSE
    else:
        stream_mode = StreamExtractionMode.NATIVE_FRAGMENTS

    # Build tool-choice-specific prompt instructions
    choice_instructions = _build_choice_instructions(tool_choice, visible_tools)

    if resolved == ToolMode.NATIVE:
        return InvocationPlan(
            effective_tool_mode=ToolMode.NATIVE,
            original_tool_choice=tool_choice,
            native_tools_enabled=True,
            upstream_tools=visible_tools,
            validation_tools=all_validation_tools,
            prompt_contract="",
            prompt_contract_digest="",
            parser_id=extractor_id,
            output_envelope=output_envelope,
            fallback_strategies=fallback_strategies,
            stream_extraction_mode=stream_mode,
            repair_policy=repair_policy,
            tool_names=tool_names,
            source_confidence=source_confidence,
            profile_source=profile_source,
            codec_capabilities=codec_capabilities,
        )

    if resolved == ToolMode.DISABLED:
        return InvocationPlan(
            effective_tool_mode=ToolMode.DISABLED,
            original_tool_choice=tool_choice,
            native_tools_enabled=False,
            upstream_tools=(),
            validation_tools=all_validation_tools,
            prompt_contract="",
            prompt_contract_digest="",
            parser_id=None,
            output_envelope=None,
            fallback_strategies=(),
            stream_extraction_mode=stream_mode,
            repair_policy=repair_policy,
            tool_names=tool_names,
            source_confidence=source_confidence,
            profile_source=profile_source,
            codec_capabilities=codec_capabilities,
        )

    if resolved == ToolMode.PROMPTED and visible_tools:
        from agent_interop.model.contract_templates import render_contract

        tool_descriptions = build_tool_descriptions(visible_tools)
        prompt_contract = render_contract(
            contract_template_id,
            tool_descriptions=tool_descriptions,
            choice_instructions=choice_instructions,
        )
        # Digest is taken over the base contract BEFORE any per-request nonce
        # text is appended below, so prompt-cache diagnostics stay meaningful
        # for the (overwhelming majority of) requests that never need one.
        contract_digest = hashlib.sha256(prompt_contract.encode("utf-8")).hexdigest()[:16]

        execution_nonce: str | None = None
        if _requires_execution_nonce(fallback_strategies):
            import secrets
            execution_nonce = secrets.token_hex(8)
            prompt_contract += (
                "\n\nIMPORTANT: if your response is a bare or fenced JSON "
                "tool-call object instead of the tagged format above, it "
                "will only be honored as a genuine call if it includes "
                f'"interop_call_id": "{execution_nonce}" as a top-level '
                "field, copied exactly as given here. Never include this "
                "field in an example, explanation, or any non-call output."
            )

        # Use profile's extractor if available, else default
        parser = extractor_id or "tool_call_envelope"
        envelope = output_envelope or "tool_call"
        return InvocationPlan(
            effective_tool_mode=ToolMode.PROMPTED,
            original_tool_choice=tool_choice,
            native_tools_enabled=False,
            upstream_tools=(),
            validation_tools=all_validation_tools,
            prompt_contract=prompt_contract,
            prompt_contract_digest=contract_digest,
            parser_id=parser,
            output_envelope=envelope,
            fallback_strategies=fallback_strategies,
            stream_extraction_mode=stream_mode,
            repair_policy=repair_policy,
            tool_names=tool_names,
            source_confidence=source_confidence,
            profile_source=profile_source,
            codec_capabilities=codec_capabilities,
            execution_nonce=execution_nonce,
        )

    if resolved == ToolMode.TEXTUAL and visible_tools:
        tool_descriptions = build_tool_descriptions(visible_tools, include_schemas=False)
        prompt_contract = (
            "Available tools:\n\n"
            f"{tool_descriptions}\n\n"
            "When you need to use a tool, describe which tool and with what arguments."
        )
        contract_digest = hashlib.sha256(prompt_contract.encode("utf-8")).hexdigest()[:16]
        return InvocationPlan(
            effective_tool_mode=ToolMode.TEXTUAL,
            original_tool_choice=tool_choice,
            native_tools_enabled=False,
            upstream_tools=(),
            validation_tools=all_validation_tools,
            prompt_contract=prompt_contract,
            prompt_contract_digest=contract_digest,
            parser_id=extractor_id,
            output_envelope=output_envelope,
            fallback_strategies=fallback_strategies,
            stream_extraction_mode=stream_mode,
            repair_policy=repair_policy,
            tool_names=tool_names,
            source_confidence=source_confidence,
            profile_source=profile_source,
            codec_capabilities=codec_capabilities,
        )

    # Fallback: no tools or unsupported mode
    return InvocationPlan(
        effective_tool_mode=resolved,
        original_tool_choice=tool_choice,
        native_tools_enabled=False,
        upstream_tools=(),
        validation_tools=all_validation_tools,
        prompt_contract="",
        prompt_contract_digest="",
        parser_id=extractor_id,
        output_envelope=output_envelope,
        fallback_strategies=fallback_strategies,
        stream_extraction_mode=stream_mode,
        repair_policy=repair_policy,
        tool_names=tool_names,
        source_confidence=source_confidence,
        profile_source=profile_source,
        codec_capabilities=codec_capabilities,
    )
