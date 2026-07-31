"""Controller safety invariants."""

from __future__ import annotations

from dataclasses import replace

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
    ToolCallProvenance,
)

# This is an Interop-internal control message, never a client tool.  The
# controller can use it when it needs a more focused work product before it
# decides which *client-declared* tool to return.  Keeping the request as a
# typed tool call means native and prompted controller routes share the same
# validation boundary; intercepting it below means the coding client never
# receives an executable synthetic call.
CONTROLLER_DELEGATE_TOOL_NAME = "__interop_request_primary_reasoning"
MAX_PRIMARY_DELEGATION_PROMPT_CHARS = 4096


def controller_delegate_tool() -> CanonicalTool:
    """Return the private controller tool used to request worker refinement."""
    return CanonicalTool(
        name=CONTROLLER_DELEGATE_TOOL_NAME,
        description=(
            "Internal Interop control tool. Request one focused refinement from "
            "the primary worker before deciding the client-visible response."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_PRIMARY_DELEGATION_PROMPT_CHARS,
                    "description": "Specific missing reasoning or code-analysis needed.",
                },
            },
            "required": ["prompt"],
            "additionalProperties": False,
        },
    )


def primary_delegation_prompt(call: CanonicalToolCallBlock) -> str | None:
    """Return a bounded worker prompt only for a valid private call.

    Returning ``None`` for malformed calls keeps the controller control plane
    fail-closed: a broken or spoofed request is never treated as a client tool
    and never silently forwarded to the primary worker.
    """
    if call.name != CONTROLLER_DELEGATE_TOOL_NAME:
        return None
    prompt = call.arguments.get("prompt") if isinstance(call.arguments, dict) else None
    if not isinstance(prompt, str):
        return None
    prompt = prompt.strip()
    if not prompt or len(prompt) > MAX_PRIMARY_DELEGATION_PROMPT_CHARS:
        return None
    return prompt


def mark_controller_provenance(call: CanonicalToolCallBlock) -> CanonicalToolCallBlock:
    """Ensure evidence cannot attribute controller calls to the primary model."""
    return replace(
        call,
        provenance=ToolCallProvenance(source="compatibility_controller", dialect="canonical"),
    )


def missing_controller_result_ids(
    messages: list[CanonicalMessage], pending_tool_call_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Return pending controller calls absent from the current client result."""
    returned = {
        block.tool_call_id
        for message in messages
        for block in message.content
        if isinstance(block, CanonicalToolResultBlock)
    }
    return tuple(call_id for call_id in pending_tool_call_ids if call_id not in returned)
