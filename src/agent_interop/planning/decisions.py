"""Pure compatibility decision helpers."""

from __future__ import annotations

from agent_interop.planning.types import BehavioralCapabilities, RequestRequirements


def missing_behavioral_capabilities(requirements: RequestRequirements, behavior: BehavioralCapabilities) -> tuple[str, ...]:
    missing: list[str] = []
    if requirements.automatic_selection_required and not behavior.automatic_selection:
        missing.append("automatic_tool_selection")
    if requirements.sequential_tool_use_required and not behavior.sequential_tool_use:
        missing.append("sequential_tool_use")
    if requirements.parallel_tool_use_required and not behavior.parallel_tool_use:
        missing.append("parallel_tool_use")
    if requirements.tool_result_continuation_required and not behavior.tool_result_continuation:
        missing.append("tool_result_continuation")
    if requirements.tools_present and requirements.streaming_required and not behavior.streaming:
        missing.append("streaming")
    return tuple(missing)
