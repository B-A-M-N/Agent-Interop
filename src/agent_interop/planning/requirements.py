"""Derive a request's contract from payload *and* integration requirements."""

from __future__ import annotations

import json
from typing import Any

from agent_interop.abi import CanonicalRequest, ToolChoiceMode
from agent_interop.context import RequestContext
from agent_interop.context_budget.types import TokenEstimate
from agent_interop.planning.types import RequestRequirements


def _has_tool_results(request: CanonicalRequest) -> bool:
    return any(
        getattr(block, "type", "") == "tool_result"
        for message in request.messages for block in message.content
    )


def _capability(profile: Any, name: str) -> bool:
    return bool(getattr(profile, name, False))


def derive_request_requirements(
    request: CanonicalRequest,
    context: RequestContext,
    client_profile: Any,
    token_estimate: TokenEstimate,
) -> RequestRequirements:
    """Build a request requirement vector without trusting an agent name.

    Tool-result history is an actual continuation requirement, even if the
    last tool choice is auto.  Integration constraints can only add required
    capabilities; they never erase requirements observed in the request.
    """
    requested = request.requested_capabilities
    tool_choice = request.tool_choice
    schema_bytes = len(json.dumps(
        [{"name": tool.name, "schema": tool.input_schema} for tool in request.tools],
        sort_keys=True, default=str,
    ).encode()) if request.tools else 0
    # Client manifests describe capabilities for tool-bearing turns.  A plain
    # chat turn must not be rejected merely because the surrounding agent
    # normally requires tool-result continuation.
    continuation = bool(request.tools) and (
        _has_tool_results(request) or bool(getattr(requested, "tool_result_continuation", False))
    )
    sequential = bool(request.tools) and (
        continuation or bool(getattr(requested, "sequential_tools", False))
    )
    return RequestRequirements(
        client_id=context.client_id,
        client_version=context.client_version,
        client_protocol=context.client_protocol,
        streaming_required=bool(request.generation.stream) or _capability(client_profile, "requires_streaming"),
        tools_present=bool(request.tools),
        tool_choice_mode=tool_choice.mode,
        named_tool=tool_choice.name,
        automatic_selection_required=(tool_choice.mode == ToolChoiceMode.AUTO and bool(request.tools)),
        sequential_tool_use_required=sequential,
        parallel_tool_use_required=bool(request.tools) and (
            bool(requested.parallel_tools) or _capability(client_profile, "requires_parallel_tools")
        ),
        tool_result_continuation_required=continuation or (
            bool(request.tools) and _capability(client_profile, "requires_tool_result_continuation")
        ),
        reasoning_required=bool(requested.reasoning) or _capability(client_profile, "requires_reasoning_blocks"),
        images_required=bool(requested.images),
        structured_output_required=bool(requested.structured_output),
        tool_count=len(request.tools),
        tool_schema_bytes=schema_bytes,
        estimated_input_tokens=token_estimate.input_tokens,
        requested_output_tokens=request.generation.max_output_tokens,
    )
