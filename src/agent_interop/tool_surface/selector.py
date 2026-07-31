"""Deterministic first-pass tool surface selection."""

from __future__ import annotations

import hashlib
import json

from agent_interop.abi import CanonicalRequest, CanonicalTool, ToolChoiceMode
from agent_interop.config import ToolSurfaceConfig, ToolSurfaceMode
from agent_interop.context_budget import estimate_tool_schema_tokens
from agent_interop.tool_surface.lexical import rank_tools
from agent_interop.tool_surface.types import ToolSurfacePlan


def _request_text(request: CanonicalRequest) -> str:
    fragments: list[str] = []
    for block in [*request.system, *(block for message in request.messages for block in message.content)]:
        text = getattr(block, "text", "")
        if text:
            fragments.append(text)
    return "\n".join(fragments)


def _fingerprint(tools: tuple[CanonicalTool, ...]) -> str:
    value = [{"name": tool.name, "schema": tool.input_schema} for tool in tools]
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ToolSurfacePlanner:
    selector_id = "lexical"
    selector_revision = "1"

    def plan(self, request: CanonicalRequest, config: ToolSurfaceConfig) -> ToolSurfacePlan:
        validation = tuple(request.tools)
        original_tokens = estimate_tool_schema_tokens(validation).input_tokens
        names_allowed = set(config.allow_tools) if config.allow_tools else None
        candidates = tuple(
            tool for tool in validation
            if tool.name not in set(config.deny_tools) and (names_allowed is None or tool.name in names_allowed)
        )
        choice = request.tool_choice
        reason = "transparent"
        if choice.mode == ToolChoiceMode.NONE:
            visible: tuple[CanonicalTool, ...] = ()
            reason = "tool_choice_none"
        elif choice.mode == ToolChoiceMode.NAMED:
            visible = tuple(tool for tool in validation if tool.name == choice.name)
            reason = "named_tool"
        elif config.mode == ToolSurfaceMode.TRANSPARENT:
            visible = candidates
        else:
            ranked = rank_tools(_request_text(request), candidates)
            limit = max(1, config.max_initial_tools)
            visible = tuple(ranked[:limit])
            if choice.mode == ToolChoiceMode.REQUIRED:
                reason = "required_smallest_relevant_set"
            else:
                reason = "top_k_lexical_matches"

        # Schema budget is applied after deterministic rank selection. Named
        # tool requests may exceed it: preserving the explicit contract wins.
        if choice.mode not in (ToolChoiceMode.NAMED, ToolChoiceMode.NONE) and config.max_schema_tokens > 0:
            budgeted: list[CanonicalTool] = []
            for tool in visible:
                if estimate_tool_schema_tokens((*budgeted, tool)).input_tokens > config.max_schema_tokens:
                    continue
                budgeted.append(tool)
            visible = tuple(budgeted)
        visible_tokens = estimate_tool_schema_tokens(visible).input_tokens
        visible_names = {tool.name for tool in visible}
        return ToolSurfacePlan(
            mode=config.mode,
            visible_tools=visible,
            validation_tools=validation,
            withheld_tool_names=tuple(tool.name for tool in validation if tool.name not in visible_names),
            original_schema_tokens=original_tokens,
            visible_schema_tokens=visible_tokens,
            selector_id=self.selector_id,
            selector_revision=self.selector_revision,
            selection_reason=reason,
            fingerprint=_fingerprint(visible),
        )

    @staticmethod
    def replan_with_tool(plan: ToolSurfacePlan, tool_name: str) -> ToolSurfacePlan:
        """Expose exactly one previously withheld declared tool on retry."""
        tool = next((item for item in plan.validation_tools if item.name == tool_name), None)
        if tool is None or tool.name not in plan.withheld_tool_names:
            return plan
        visible = (*plan.visible_tools, tool)
        visible_names = {item.name for item in visible}
        return ToolSurfacePlan(
            mode=plan.mode,
            visible_tools=visible,
            validation_tools=plan.validation_tools,
            withheld_tool_names=tuple(
                item.name for item in plan.validation_tools if item.name not in visible_names
            ),
            original_schema_tokens=plan.original_schema_tokens,
            visible_schema_tokens=estimate_tool_schema_tokens(visible).input_tokens,
            selector_id=plan.selector_id,
            selector_revision=plan.selector_revision,
            selection_reason=f"withheld_tool_requested:{tool_name}",
            fingerprint=_fingerprint(visible),
        )
