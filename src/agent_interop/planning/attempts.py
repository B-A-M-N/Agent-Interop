"""Compatibility attempt ladder construction."""

from __future__ import annotations

from agent_interop.config import ToolMode
from agent_interop.planning.types import AttemptKind, CompatibilityAttempt, RequestRequirements


def direct_attempts(requirements: RequestRequirements, codec, runtime, behavior) -> tuple[CompatibilityAttempt, ...]:
    attempts: list[CompatibilityAttempt] = []
    native_usable = (
        bool(getattr(codec, "supports_native_tools", False))
        and runtime.accepts_native_tools.is_available()
        and runtime.returns_native_tool_calls.is_available()
        and behavior.native_tools
    )
    if not requirements.tools_present:
        attempts.append(CompatibilityAttempt(AttemptKind.CHAT_ONLY, ToolMode.DISABLED, reason="request_has_no_tools"))
    elif native_usable:
        attempts.append(CompatibilityAttempt(AttemptKind.NATIVE_TOOLS, ToolMode.NATIVE, reason="runtime_and_behavioral_native_tools"))
    return tuple(attempts)


def adapted_attempts(requirements: RequestRequirements, profile, runtime, behavior) -> tuple[CompatibilityAttempt, ...]:
    if not requirements.tools_present:
        return ()
    # A completed bootstrap battery can positively identify a chat-only
    # worker.  Rendering another textual contract would only waste an agent
    # turn; controller mode is the compatible path when available.
    if getattr(behavior, "chat_only", False):
        return ()
    attempts = [CompatibilityAttempt(
        AttemptKind.PROMPTED_TOOLS,
        ToolMode.PROMPTED,
        parser_id=getattr(profile, "parser_id", None) or "tool_call_envelope",
        contract_template_id=getattr(profile, "contract_template_id", None),
        reason="prompted_tool_contract",
    )]
    if runtime.supports_json_schema.is_available() or runtime.supports_grammar.is_available() or runtime.supports_json_mode.is_available():
        attempts.append(CompatibilityAttempt(AttemptKind.CONSTRAINED_JSON, ToolMode.PROMPTED, constrained_output=True, reason="backend_constrained_output"))
    if requirements.automatic_selection_required and behavior.forced_selection:
        attempts.append(CompatibilityAttempt(AttemptKind.FORCED_SELECTION, ToolMode.PROMPTED, reason="forced_selection_after_deterministic_tool_selection"))
    return tuple(attempts)
