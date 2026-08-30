"""Safe, deterministic context adaptation.

P0.7: Replace lossy truncation with virtualization by default. ALL large
tool results are virtualized (stored in the ContextStore, replaced with
bounded handles) rather than truncated. This is safer because nothing is
lost — the full result is retained behind a ref.

``stored_refs`` on ``ContextAdaptationResult`` carries the ContextStore refs
that callers must pin for the request lifecycle (so they survive eviction).
``compacted_tool_result_ids`` remains semantic bookkeeping of tool_call_ids
that were virtualized — it is not a storage handle.

Legacy lossy truncation is kept as fallback when no store is available.
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
    compacted_tool_result_ids: tuple[str, ...] = ()  # tool_call_ids (semantic identity)
    stored_refs: tuple[str, ...] = ()  # ContextStore refs, safe to pin

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
    """Reduce an older known-JSON result without producing invalid JSON."""
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


# ─── UTF-8 helpers ─────────────────────────────────────────────────────────


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Truncate *text* to at most *max_bytes* in UTF-8.

    Never splits a multibyte sequence: encodes to UTF-8, trims the byte
    buffer, then decodes with ``errors="ignore"`` so the caller always
    gets a valid string.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


# ─── P0.7 virtualization-based compaction ──────────────────────────────────


def virtualize_tool_results_compaction(
    request: CanonicalRequest,
    *,
    store: Any,
    session_id: str,
    exchanges: tuple[Any, ...] | list[Any],
    plan: ContextPlan,
    policy: Any = None,
) -> ContextAdaptationResult:
    """Apply virtualization-based result compaction (P0.7).

    Replaces the old lossy truncation policy (known tool => maybe truncate,
    unknown tool => verbatim) with: ALL large results => virtualize by default.

    The full result is stored in the ContextStore; the message content becomes
    a head/tail + ref that the model can page through via __interop_read_result.
    This is safer than silent truncation because nothing is lost.
    """
    if policy is None:
        from agent_interop.context_store.policy import VirtualizationPolicy

        policy = VirtualizationPolicy()
    if not plan.compaction_required:
        return ContextAdaptationResult(request)
    candidate_indices = set(plan.compacted_message_indices)
    messages = list(request.messages)
    changed_ids: list[str] = []
    stored_refs: list[str] = []
    transformations: list[str] = []
    for index, message in enumerate(messages):
        if index not in candidate_indices or message.role != "tool":
            continue
        changed_blocks: list[Any] = []
        message_changed = False
        for block in message.content:
            if not isinstance(block, CanonicalToolResultBlock):
                changed_blocks.append(block)
                continue
            if block.is_error or not isinstance(block.content, str):
                # Errors are preserved exactly — they are action-critical
                changed_blocks.append(block)
                continue
            decision = policy.decide(block)
            if not decision.should_virtualize:
                changed_blocks.append(block)
                continue
            # Virtualize: store full content, replace with handle + head/tail
            text = block.content
            entry = store.store(
                session_id=session_id,
                content=text,
                kind="tool_result",
                tool_call_id=block.tool_call_id,
            )
            stored_refs.append(entry.ref)
            lines = text.splitlines(keepends=True)
            head_count = decision.max_inline_lines // 2
            tail_count = decision.max_inline_lines - head_count
            # Determine visible content, ensuring we never exceed max_inline_bytes.
            if len(lines) <= decision.max_inline_lines:
                # Few lines: keep as-is unless the total byte size exceeds the cap.
                visible = "".join(lines)
                if len(visible.encode("utf-8", "replace")) > policy._max_inline_bytes:
                    # Single-line (or few-line) result that exceeds the byte cap.
                    # Use truncate_utf8 to avoid splitting multibyte characters.
                    visible = truncate_utf8(visible, policy._max_inline_bytes)
            else:
                visible = "".join(lines[:head_count] + lines[-tail_count:])
            new_content = (
                f"[Interop result ref: {entry.ref}] Original: {len(text.splitlines())} lines. "
                f"Visible: {decision.max_inline_lines} lines. Full retained (sha256:{entry.sha256[:16]}). "
                f"Use __interop_read_result if more is required.\n{visible}"
            )
            changed_blocks.append(replace(block, content=new_content))
            message_changed = True
            changed_ids.append(block.tool_call_id)
        if message_changed:
            messages[index] = replace(message, content=changed_blocks)
            transformations.append("virtualize_large_tool_results")
    if not changed_ids:
        return ContextAdaptationResult(request)
    return ContextAdaptationResult(
        replace(request, messages=messages),
        transformations=tuple(transformations),
        compacted_tool_result_ids=tuple(changed_ids),
        stored_refs=tuple(stored_refs),
    )


def compact_safe_tool_results(
    request: CanonicalRequest,
    *,
    exchanges: tuple[Any, ...] | list[Any],
    plan: ContextPlan,
    store: Any = None,
    session_id: str = "",
) -> ContextAdaptationResult:
    """Apply safe result compaction selected by ``ContextPlan``.

    When a ContextStore is provided (P0.7), large results are virtualized
    (stored in full, replaced with bounded handles) rather than truncated.
    Falls back to the legacy lossy truncation when no store is available.

    Unknown tools, error output, and current result messages remain
    byte-for-byte intact.
    """
    if store is not None and session_id:
        return virtualize_tool_results_compaction(
            request,
            store=store,
            session_id=session_id,
            exchanges=exchanges,
            plan=plan,
        )
    return _legacy_compact_safe_tool_results(request, exchanges=exchanges, plan=plan)


def _legacy_compact_safe_tool_results(
    request: CanonicalRequest,
    *,
    exchanges: tuple[Any, ...] | list[Any],
    plan: ContextPlan,
) -> ContextAdaptationResult:
    """Legacy lossy truncation (kept for fallback / tests)."""
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
