"""Deterministic first-pass tool surface selection."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_interop.context_budget.types import RequestCostSnapshot

from agent_interop.abi import (
    CanonicalRequest,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
    ToolChoiceMode,
)
from agent_interop.config import ToolSurfaceConfig, ToolSurfaceMode
from agent_interop.context_budget import estimate_tool_schema_tokens
from agent_interop.tool_surface.lexical import build_tool_terms_cache, rank_tools
from agent_interop.tool_surface.types import ToolSurfacePlan


def _request_text(request: CanonicalRequest) -> str:
    """Build weighted ranking input targeting the CURRENT task.

    P1.1: Instead of combining entire system prompt + every message (which
    makes old unrelated work influence ranking), use weighted components:
    - current/latest user turn: highest
    - unfinished active tool exchange: high
    - recent assistant turn: medium
    - old history/system: very low / excluded
    """
    fragments: list[str] = []
    # Current/latest user turn (highest weight)
    user_turns = [m for m in request.messages if m.role == "user"]
    if user_turns:
        latest_user = user_turns[-1]
        for block in latest_user.content:
            text = getattr(block, "text", "")
            if text:
                fragments.append(text)
    # Recent tool exchange (high weight)
    for msg in request.messages[-3:]:
        for block in msg.content:
            if isinstance(block, (CanonicalToolCallBlock, CanonicalToolResultBlock)):
                fragments.append(str(getattr(block, "name", "") or getattr(block, "content", "")))
    return "\n".join(fragments)


def _fingerprint(tools: tuple[CanonicalTool, ...]) -> str:
    value = [{"name": tool.name, "schema": tool.input_schema} for tool in tools]
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _single_tool_item_bytes(tool: CanonicalTool) -> int:
    """Serialized byte length of one tool's estimator list item.

    Mirrors ``build_request_cost_snapshot``'s per-item form for callers
    that arrive without a cost snapshot (retry replans, controller
    refinement surfaces).
    """
    item = {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
    raw = json.dumps(
        item, default=lambda item: getattr(item, "__dict__", str(item)),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    return len(raw.encode("utf-8"))


class ToolSurfacePlanner:
    selector_id = "lexical"
    selector_revision = "1"

    def __init__(self) -> None:
        # fingerprint -> per-tool (name_terms, desc_terms, name_lower).
        # Single-entry: a gateway plan()s one registry at a time; clearing
        # on registry change bounds memory without TTL bookkeeping.
        self._rank_cache: dict[str, dict[str, tuple[set[str], set[str], str]]] = {}

    def plan(
        self,
        request: CanonicalRequest,
        config: ToolSurfaceConfig,
        *,
        cost_snapshot: "RequestCostSnapshot | None" = None,
    ) -> ToolSurfacePlan:
        """Select the model-visible tool surface.

        ``cost_snapshot`` supplies per-tool byte lengths from the request's
        single serialization pass; without it, per-tool costs are computed
        locally. Either way each tool is priced EXACTLY ONCE — the schema
        budget loop below is O(n), not the historical O(n²) that
        re-serialized the whole accumulating candidate list per tool.
        """
        from agent_interop.context_budget.estimator import price_tool_subset

        validation = tuple(request.tools)
        if cost_snapshot is not None:
            original_tokens = price_tool_subset(cost_snapshot, validation)
        else:
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
            # P1.3: tokenizing the query is per-request work (the query
            # changes every turn); tokenizing the registry is not — cache it
            # per planner instance keyed by the registry fingerprint.
            fingerprint = _fingerprint(candidates)
            registry_cache = self._rank_cache.get(fingerprint)
            if registry_cache is None:
                registry_cache = build_tool_terms_cache(candidates)
                self._rank_cache.clear()
                self._rank_cache[fingerprint] = registry_cache
            ranked = rank_tools(
                _request_text(request), candidates, tool_terms=registry_cache,
            )
            limit = max(1, config.max_initial_tools)
            visible = tuple(ranked[:limit])
            if choice.mode == ToolChoiceMode.REQUIRED:
                reason = "required_smallest_relevant_set"
            else:
                reason = "top_k_lexical_matches"

        # Schema budget is applied after deterministic rank selection. Named
        # tool requests may exceed it: preserving the explicit contract wins.
        # P1.3: price the running set incrementally — each tool's item bytes
        # are added once, so this loop is linear in the number of visible
        # tools (the historical form re-serialized the whole accumulating
        # list per tool: O(n²) serializations of an n-tool surface).
        if choice.mode not in (ToolChoiceMode.NAMED, ToolChoiceMode.NONE) and config.max_schema_tokens > 0:
            budgeted: list[CanonicalTool] = []
            budgeted_bytes = 0
            item_bytes = (
                cost_snapshot.tool_item_bytes if cost_snapshot is not None else None
            )

            def _subset_tokens(count: int, total_bytes: int) -> int:
                if count == 0:
                    return 0
                assembled = total_bytes + (count - 1) + 4  # junctions + brackets + boundary
                return max(1, assembled // 3 + 4)

            for tool in visible:
                if item_bytes is not None:
                    tool_bytes = item_bytes.get(tool.name)
                    if tool_bytes is None:
                        tool_bytes = _single_tool_item_bytes(tool)
                else:
                    tool_bytes = _single_tool_item_bytes(tool)
                next_total = budgeted_bytes + tool_bytes
                if _subset_tokens(len(budgeted) + 1, next_total) > config.max_schema_tokens:
                    continue
                budgeted.append(tool)
                budgeted_bytes = next_total
            visible = tuple(budgeted)
        if cost_snapshot is not None:
            visible_tokens = price_tool_subset(cost_snapshot, visible)
        else:
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
