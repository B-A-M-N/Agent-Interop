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


def compact_safe_tool_results(
    request: CanonicalRequest,
    *,
    exchanges: tuple[Any, ...] | list[Any],
    plan: ContextPlan,
) -> ContextAdaptationResult:
    """Apply safe result compaction selected by ``ContextPlan``.

    Unknown tools, error output, structured/list output, and current result
    messages remain byte-for-byte intact.  A model-visible omission marker is
    intentionally included so a later turn cannot mistake a partial result
    for the complete original.
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
            if (
                block.is_error
                or not isinstance(block.content, str)
                or policy is not ToolResultPolicy.BOUNDED_LINES
            ):
                changed_blocks.append(block)
                continue
            compacted = _bounded_lines(block.content)
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
