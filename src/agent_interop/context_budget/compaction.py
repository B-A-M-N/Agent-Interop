"""Safe, deterministic context adaptation.

Interop is not allowed to silently discard a client's current execution
state.  This module therefore performs only the one loss-bounded adaptation
that is safe without another model call: reducing *older*, known pageable
tool results while retaining exact first/last source lines, correlation IDs,
error state, and an integrity digest.  Older conversational turns are never
removed here; controller-mediated summarisation remains an explicit future
execution step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

from agent_interop.abi import CanonicalMessage, CanonicalRequest, CanonicalToolResultBlock
from agent_interop.context_budget.tool_results import ToolResultPolicy, default_tool_result_policy
from agent_interop.context_budget.types import ContextPlan


def is_required_message(message: CanonicalMessage, index: int, last_index: int) -> bool:
    return index == last_index or message.role in {"system", "developer"}


@dataclass(frozen=True)
class ContextAdaptationResult:
    """The exact safe mutation applied to a canonical request."""

    request: CanonicalRequest
    transformations: tuple[str, ...] = ()
    compacted_tool_result_ids: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        return bool(self.transformations)


def _call_names(exchanges: tuple[Any, ...] | list[Any]) -> dict[str, str]:
    return {
        str(exchange.call_id): str(getattr(getattr(exchange, "call", None), "name", ""))
        for exchange in exchanges
        if getattr(exchange, "call_id", "")
    }


def _bounded_lines(content: str, *, max_lines: int = 16) -> str:
    """Retain source lines verbatim with an explicit, stable omission marker."""
    lines = content.splitlines(keepends=True)
    if len(lines) <= max_lines:
        return content
    head_count = max_lines // 2
    tail_count = max_lines - head_count
    omitted = len(lines) - head_count - tail_count
    digest = sha256(content.encode("utf-8", "replace")).hexdigest()[:16]
    marker = f"[interop: compacted {omitted} lines; sha256:{digest}]\n"
    return "".join((*lines[:head_count], marker, *lines[-tail_count:]))


def _structured_reduction(content: str, *, max_items: int = 8, max_depth: int = 4) -> str | None:
    """Reduce an older known-JSON result without producing invalid JSON.

    This policy is intentionally narrower than ordinary text compaction: it
    only activates for a tool explicitly classified as a JSON-query shape and
    keeps scalar values byte-for-byte after JSON decoding.  Large arrays and
    mappings retain deterministic head/tail samples plus an integrity digest
    and omission count, so a later model turn can see that the representation
    is incomplete instead of treating it as a complete query result.
    """
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return None

    changed = False
    digest = sha256(content.encode("utf-8", "replace")).hexdigest()[:16]

    def marker(omitted: int) -> dict[str, Any]:
        return {"__interop_compacted__": {"omitted": omitted, "sha256": digest}}

    def reduce(value: Any, depth: int = 0) -> Any:
        nonlocal changed
        if depth >= max_depth:
            return value
        if isinstance(value, list):
            if len(value) <= max_items:
                return [reduce(item, depth + 1) for item in value]
            changed = True
            head = max_items // 2
            tail = max_items - head
            return [
                *(reduce(item, depth + 1) for item in value[:head]),
                marker(len(value) - head - tail),
                *(reduce(item, depth + 1) for item in value[-tail:]),
            ]
        if isinstance(value, dict):
            keys = sorted(value, key=str)
            if len(keys) <= max_items:
                return {key: reduce(value[key], depth + 1) for key in keys}
            changed = True
            head = max_items // 2
            tail = max_items - head
            kept = (*keys[:head], *keys[-tail:])
            reduced = {key: reduce(value[key], depth + 1) for key in kept}
            marker_key = "__interop_compacted__"
            while marker_key in reduced:
                marker_key += "_"
            reduced[marker_key] = {"omitted": len(keys) - len(kept), "sha256": digest}
            return reduced
        return value

    reduced = reduce(parsed)
    if not changed:
        return None
    return json.dumps(reduced, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def compact_safe_tool_results(
    request: CanonicalRequest,
    *,
    exchanges: tuple[Any, ...] | list[Any],
    plan: ContextPlan,
) -> ContextAdaptationResult:
    """Apply safe result compaction selected by ``ContextPlan``.

    Unknown tools, error output, and current result messages remain
    byte-for-byte intact. Explicit JSON-query tools may use structure-aware
    reduction; other known pageable text tools preserve exact head/tail
    source lines. A model-visible omission marker is intentionally included
    so a later turn cannot mistake a partial result for the complete original.
    """
    if not plan.compaction_required:
        return ContextAdaptationResult(request)
    call_names = _call_names(exchanges)
    candidate_indices = set(plan.compacted_message_indices)
    messages = list(request.messages)
    changed_ids: list[str] = []
    for index, message in enumerate(messages):
        if index not in candidate_indices or message.role != "tool":
            continue
        changed_blocks: list[Any] = []
        message_changed = False
        for block in message.content:
            if not isinstance(block, CanonicalToolResultBlock):
                changed_blocks.append(block)
                continue
            tool_name = call_names.get(block.tool_call_id, "")
            policy = default_tool_result_policy(tool_name)
            if block.is_error or not isinstance(block.content, str):
                changed_blocks.append(block)
                continue
            if policy is ToolResultPolicy.BOUNDED_LINES:
                compacted = _bounded_lines(block.content)
            elif policy is ToolResultPolicy.STRUCTURED_REDUCTION:
                compacted = _structured_reduction(block.content)
            else:
                compacted = None
            if compacted is None:
                changed_blocks.append(block)
                continue
            if compacted == block.content:
                changed_blocks.append(block)
                continue
            changed_blocks.append(replace(block, content=compacted))
            message_changed = True
            changed_ids.append(block.tool_call_id)
        if message_changed:
            messages[index] = replace(message, content=changed_blocks)
    if not changed_ids:
        return ContextAdaptationResult(request)
    return ContextAdaptationResult(
        replace(request, messages=messages),
        transformations=("compact_old_pageable_tool_results",),
        compacted_tool_result_ids=tuple(changed_ids),
    )
