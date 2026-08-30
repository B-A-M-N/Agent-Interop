"""History projector (P0.15).

Implements bounded history paging: preserves the latest user turn, the most
recent complete tool exchange, and recent N turns within budget. Older closed
history becomes HistoryRef entries that the model can page through via
__interop_recall_history / __interop_search_history.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalMessage,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)


@dataclass(frozen=True)
class HistoryRef:
    """Reference to a stored history fragment."""

    ref: str
    message_range: tuple[int, int]
    summary: str
    kind: str = "history_fragment"


@dataclass(frozen=True)
class HistoryUnit:
    """A non-splitting semantic unit in the message stream.

    *kind* is one of "system" | "user_turn" | "tool_exchange" | "assistant_turn".
    *protected* is True when the unit must stay in the preserved window.
    *start_index* / *end_index* are inclusive message indices.
    """

    start_index: int
    end_index: int
    kind: str
    protected: bool = False


@dataclass(frozen=True)
class HistoryProjectionResult:
    """Result of projecting conversation history."""

    messages: list[CanonicalMessage]
    refs: list[HistoryRef]
    preserved_count: int
    compacted_count: int
    transformations: tuple[str, ...] = ()
    stored_refs: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _message_has_tool_calls(msg: CanonicalMessage) -> bool:
    """Return True when *msg* contains any CanonicalToolCallBlock."""
    return any(isinstance(b, CanonicalToolCallBlock) for b in msg.content)


def _group_messages_into_semantic_units(
    messages: list[CanonicalMessage],
) -> list[HistoryUnit]:
    """Group messages into semantic units that must never be split.

    A tool-exchange unit starts at an assistant message containing
    CanonicalToolCallBlock(s) and extends through the contiguous run of
    tool-role messages that follow it.  P0-57: when that run contains
    results for call IDs the unit does NOT own (an interleaved exchange),
    the unit absorbs them anyway — [call_A][call_B][result_A][result_B]
    pages as ONE unit, because splitting a tool region across units could
    page a call away from its result, which reconciliation rejects as
    unsafe.  A tool message that appears with NO owning exchange anywhere
    in the transcript (its call is not in this history at all) is likewise
    grouped with its neighbouring tool run; reconciliation remains the
    authority on whether the preserved window is safe.

    Returns a list of HistoryUnit objects.
    """
    if not messages:
        return []

    units: list[HistoryUnit] = []
    idx = 0

    while idx < len(messages):
        msg = messages[idx]

        if msg.role in ("system", "developer"):
            # Each system/developer message is its own unit.
            units.append(HistoryUnit(idx, idx, "system", protected=True))
            idx += 1

        elif msg.role == "assistant" and _message_has_tool_calls(msg):
            # Tool-exchange unit: the call-bearing assistant message plus
            # the contiguous tool-result region behind it (possibly
            # containing interleaved exchanges — see docstring).
            start = idx
            idx += 1
            while idx < len(messages) and messages[idx].role == "tool":
                idx += 1
            end = idx - 1
            units.append(HistoryUnit(start, end, "tool_exchange"))

        elif msg.role == "user":
            start = idx
            idx += 1
            while idx < len(messages):
                nxt = messages[idx]
                if nxt.role == "tool":
                    idx += 1
                elif nxt.role == "user":
                    idx += 1
                else:
                    break
            units.append(HistoryUnit(start, idx - 1, "user_turn"))

        else:
            start = idx
            idx += 1
            while idx < len(messages):
                nxt = messages[idx]
                if nxt.role == "tool":
                    idx += 1
                elif nxt.role == "assistant" and not _message_has_tool_calls(nxt):
                    idx += 1
                else:
                    break
            units.append(HistoryUnit(start, idx - 1, "assistant_turn"))

    # P0-57: a unit that contains orphaned results (a tool message whose
    # IDs belong to an exchange OUTSIDE this unit — e.g. the client sent
    # [call_A][result for B separated by a user turn]) must never page
    # independently of that exchange.  Merge any unit carrying results
    # whose IDs are owned by a different call-bearing unit into the unit
    # that owns them (bridging any units between), so a call and its
    # result can never land on opposite sides of the paging boundary.
    owner: dict[str, int] = {}
    for unit_index, unit in enumerate(units):
        for pos in range(unit.start_index, unit.end_index + 1):
            for block in messages[pos].content:
                if isinstance(block, CanonicalToolCallBlock) and block.id:
                    owner[block.id] = unit_index

    merged: list[HistoryUnit] = []
    for unit_index, unit in enumerate(units):
        has_foreign = any(
            isinstance(block, CanonicalToolResultBlock) and block.tool_call_id in owner
            and owner[block.tool_call_id] != unit_index
            for pos in range(unit.start_index, unit.end_index + 1)
            for block in messages[pos].content
        )
        if (
            merged
            and has_foreign
            and owner
            and min(
                (owner[b.tool_call_id] for pos in range(unit.start_index, unit.end_index + 1)
                 for b in messages[pos].content
                 if isinstance(b, CanonicalToolResultBlock) and b.tool_call_id in owner),
                default=unit_index,
            ) < unit_index
        ):
            first = merged.pop()
            merged.append(HistoryUnit(
                first.start_index,
                unit.end_index,
                first.kind if first.kind != "system" else "assistant_turn",
                protected=first.protected or unit.protected,
            ))
        else:
            merged.append(unit)

    return merged


def _make_canonical_block_dict(
    block: CanonicalContentBlock,
) -> dict[str, Any]:
    """Serialize a single block to a deterministic dict."""
    if isinstance(block, CanonicalTextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, CanonicalToolCallBlock):
        return {
            "type": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": block.arguments,
        }
    if isinstance(block, CanonicalToolResultBlock):
        content = block.content
        if isinstance(content, list):
            content = "".join(
                b.text if isinstance(b, CanonicalTextBlock) else str(b)
                for b in content
            )
        return {
            "type": "tool_result",
            "tool_call_id": block.tool_call_id,
            "content": content,
            "is_error": block.is_error,
        }
    return {"type": "raw", "repr": str(block)}


def _canonical_json_for_messages(
    messages: list[CanonicalMessage],
) -> str:
    """Return deterministic canonical JSON for a list of messages."""
    payload = {
        "schema_version": 1,
        "messages": [
            {
                "role": msg.role,
                "content": [
                    _make_canonical_block_dict(b) for b in msg.content
                ],
            }
            for msg in messages
        ],
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _summarize_fragment_units(units: list[HistoryUnit]) -> str:
    """Brief deterministic summary of the unit kinds covered."""
    kinds: dict[str, int] = {}
    for u in units:
        kinds[u.kind] = kinds.get(u.kind, 0) + 1
    parts: list[str] = []
    for k in ("tool_exchange", "user_turn", "assistant_turn", "system"):
        if k in kinds:
            label = k.replace("_", " ")
            count = kinds[k]
            parts.append(f"{count} {label}{'s' if count > 1 else ''}")
    kind_str = ", ".join(parts) if parts else "misc"
    start = units[0].start_index
    end = units[-1].end_index
    return f"messages {start}–{end} ({kind_str})"


# ---------------------------------------------------------------------------
# render_fragment_text -- new public API
# ---------------------------------------------------------------------------


def render_fragment_text(canonical_json: str, max_chars: int) -> str:
    """Render a bounded human-readable transcript from canonical JSON.

    One line per message:

    * ``user: ...``
    * ``assistant -> tool <name>(<args truncated 80 chars>)``
    * ``tool result for <id>: <content truncated>``

    When the rendered text would exceed *max_chars*, truncation happens at a
    message boundary (never mid-message) and the sentinel
    ``[...older messages omitted ...]`` is appended.

    On any parse failure, returns the first *max_chars* characters of the
    raw string.
    """
    try:
        payload = json.loads(canonical_json)
    except (json.JSONDecodeError, TypeError):
        return canonical_json[:max_chars]

    lines: list[str] = []
    total = 0
    # P0-59: truncation is a fact about whether a message was dropped, not
    # an inference from the final string length — a render that exactly
    # fills max_chars truncated nothing, and the sentinel must not lie.
    truncated = False
    for msg in payload.get("messages", []):
        role = msg.get("role", "unknown")
        content_blocks = msg.get("content", [])
        for blk in content_blocks:
            blk_type = blk.get("type", "raw")
            if role in ("user", "system", "developer"):
                text = blk.get("text", "")
                line = f"{role}: {text}"
            elif blk_type == "tool_call":
                name = blk.get("name", "?")
                args = json.dumps(blk.get("arguments", {}))
                args_trunc = args[:80]
                line = f"assistant -> tool {name}({args_trunc})"
            elif blk_type == "tool_result":
                cid = blk.get("tool_call_id", "?")
                txt = blk.get("content", "")
                line = f"tool result for {cid}: {txt}"
            else:
                line = f"{role}: {json.dumps(blk)}"

            if total + len(line) + (1 if lines else 0) > max_chars:
                truncated = True
                break
            lines.append(line)
            total += len(line) + 1
        else:
            continue
        break

    rendered = "\n".join(lines)
    if truncated:
        rendered += (
            "\n[...older messages omitted "
            "— use __interop_read_result on this ref for full content]"
        )
    return rendered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def project_history(
    messages: list[CanonicalMessage],
    *,
    max_recent_turns: int = 6,
    max_total_messages: int = 20,
    store: Any | None = None,
    session_id: str = "",
) -> HistoryProjectionResult:
    """Project conversation history into a bounded model-consumable view.

    Preserves (protected units):

    1. System / developer messages
    2. Latest user turn
    3. Latest complete tool exchange
    4. The most recent *max_recent_turns* non-system units

    Older closed history is stored as canonical JSON fragments with whole-unit
    boundaries.

    *max_total_messages* bounds the PRESERVED window at semantic-unit
    boundaries (P0-58): once the protected units already reach the cap, no
    further unit is preserved — paging takes the OLDEST unprotected units
    first, so the cap can only drop already-paged history.  A cap of 0 or
    less disables the bound.
    """
    if not messages:
        return HistoryProjectionResult(
            messages=[], refs=[], preserved_count=0, compacted_count=0
        )

    transformations: list[str] = []
    units = _group_messages_into_semantic_units(messages)

    if not units:
        return HistoryProjectionResult(
            messages=list(messages),
            refs=[],
            preserved_count=len(messages),
            compacted_count=0,
        )

    # ── Identify protected units ────────────────────────────────────────
    # Mark the latest user-turn unit as protected.
    last_user_unit_idx: int | None = None
    for i in range(len(units) - 1, -1, -1):
        if units[i].kind == "user_turn":
            last_user_unit_idx = i
            break

    # Mark the latest tool-exchange unit as protected.
    last_tool_exc_idx: int | None = None
    for i in range(len(units) - 1, -1, -1):
        if units[i].kind == "tool_exchange":
            last_tool_exc_idx = i
            break

    # Recent non-system units from the end.
    # P0-57: the historical cutoff compared a full-unit index (``i``) with a
    # position in the FILTERED non-system list — with system/developer units
    # interleaved the two index spaces diverge, so the wrong units were
    # protected (too many recent turns when leading units were system, too
    # few when system units appeared mid-history).  Compare within one
    # index space.
    non_system_indices = [
        i for i, u in enumerate(units) if u.kind not in ("system", "developer")
    ]
    recent_indices = set(non_system_indices[-max_recent_turns:]) if non_system_indices else set()

    for i, u in enumerate(units):
        prot = False
        if u.kind in ("system", "developer"):
            prot = True
        elif i == last_user_unit_idx:
            prot = True
        elif i == last_tool_exc_idx:
            prot = True
        elif i in recent_indices:
            prot = True
        units[i] = replace(u, protected=prot)

    # Build a set of indices that belong to protected units.
    # P0-58: max_total_messages applies at UNIT boundaries — mandatory
    # units (system/developer, latest user turn, latest tool exchange) are
    # admitted first, then the remaining recent units fill the cap; a unit
    # that would exceed the cap pages out instead.  A cap of 0 (or less)
    # disables the bound.
    protected_indices: set[int] = set()
    if max_total_messages and max_total_messages > 0:
        latest_user = units[last_user_unit_idx] if last_user_unit_idx is not None else None
        latest_tool_exc = units[last_tool_exc_idx] if last_tool_exc_idx is not None else None
        mandatory_units = [
            u for u in units
            if u.protected and u.kind in ("system", "developer")
        ]
        for special in (
            *mandatory_units,
            *(x for x in (latest_user, latest_tool_exc) if x is not None),
        ):
            protected_indices.update(range(special.start_index, special.end_index + 1))
        remaining = max_total_messages - len(protected_indices)
        for u in units:
            if not u.protected:
                continue
            span = set(range(u.start_index, u.end_index + 1))
            if span <= protected_indices:
                continue
            if len(span) > remaining:
                continue  # would exceed the cap — page this unit instead
            protected_indices.update(span)
            remaining -= len(span)
    else:
        for u in units:
            if u.protected:
                protected_indices.update(range(u.start_index, u.end_index + 1))

    # Build preserved messages from the protected window.
    preserved: list[CanonicalMessage] = [
        messages[i] for i in sorted(protected_indices) if i < len(messages)
    ]

    # ── Page non-protected units ────────────────────────────────────────
    ref_entries: list[HistoryRef] = []
    stored_refs_list: list[str] = []

    frag_units: list[HistoryUnit] = []
    for u in units:
        if u.protected:
            if frag_units:
                _store_fragment(
                    frag_units,
                    messages,
                    store,
                    session_id,
                    ref_entries,
                    stored_refs_list,
                    transformations,
                )
                frag_units = []
        else:
            frag_units.append(u)
    if frag_units:
        _store_fragment(
            frag_units,
            messages,
            store,
            session_id,
            ref_entries,
            stored_refs_list,
            transformations,
        )

    # ── Structural safety check ─────────────────────────────────────────
    try:
        from agent_interop.history.reconcile import reconcile_history
    except ImportError:
        reconcile_result: Any = None
    else:
        reconcile_result = reconcile_history(
            preserved, session_id=session_id, request_id=""
        )
        if not reconcile_result.is_safe:
            raise ValueError("history paging produced unsafe history")

    return HistoryProjectionResult(
        messages=preserved,
        refs=ref_entries,
        preserved_count=len(preserved),
        compacted_count=sum(
            u.end_index - u.start_index + 1
            for u in units
            if not u.protected
        ),
        transformations=tuple(transformations),
        stored_refs=tuple(stored_refs_list),
    )


def _store_fragment(
    frag_units: list[HistoryUnit],
    messages: list[CanonicalMessage],
    store: Any | None,
    session_id: str,
    ref_entries: list[HistoryRef],
    stored_refs_list: list[str],
    transformations: list[str],
) -> None:
    """Store a group of non-protected units as one canonical fragment."""
    start = frag_units[0].start_index
    end = frag_units[-1].end_index
    fragment_messages = messages[start:end + 1]

    if store is not None and session_id:
        canonical_json = _canonical_json_for_messages(fragment_messages)
        entry = store.store(
            session_id=session_id,
            content=canonical_json,
            kind="history_fragment",
            metadata={"start_index": start, "end_index": end},
        )
        summary = _summarize_fragment_units(frag_units)
        ref_entries.append(HistoryRef(
            ref=entry.ref,
            message_range=(start, end),
            summary=summary,
        ))
        stored_refs_list.append(entry.ref)
        transformations.append(
            f"page_history_turns_{start}-{end}"
        )


# ---------------------------------------------------------------------------
# Legacy helpers (kept for tests / back-compat; NOT used for storage)
# ---------------------------------------------------------------------------


def _format_fragment(messages: list[CanonicalMessage]) -> str:
    """Format a list of messages into a stored string.

    DEPRECATED: kept for tests / back-compat only.
    New code must use _canonical_json_for_messages.
    """
    parts = []
    for msg in messages:
        role = msg.role
        content_strs = []
        for block in msg.content:
            if isinstance(block, CanonicalTextBlock):
                content_strs.append(block.text)
            elif isinstance(block, CanonicalToolCallBlock):
                args = (
                    json.dumps(block.arguments)
                    if isinstance(block.arguments, dict)
                    else str(block.arguments)
                )
                content_strs.append(f"[tool_call: {block.name}({args})]")
            elif isinstance(block, CanonicalToolResultBlock):
                content_strs.append(f"[tool_result: {block.content}]")
            else:
                content_strs.append(str(block))
        parts.append(f"[{role}] " + " ".join(content_strs))
    return "\n".join(parts)


def _summarize_fragment(messages: list[CanonicalMessage]) -> str:
    """Create a brief deterministic summary of a message fragment.

    DEPRECATED: kept for back-compat; use _summarize_fragment_units.
    """
    if not messages:
        return "empty"
    tool_calls = sum(
        1
        for msg in messages
        for block in msg.content
        if isinstance(block, CanonicalToolCallBlock)
    )
    return (
        f"turns {messages[0].role}-{messages[-1].role}, "
        f"{len(messages)} msgs, {tool_calls} calls"
    )


def build_history_index_prompt(refs: list[HistoryRef]) -> str:
    """Build a prompt snippet describing available history refs."""
    if not refs:
        return ""
    lines = [
        "Earlier conversation (use __interop_recall_history "
        "or __interop_search_history if needed):"
    ]
    for ref in refs:
        lines.append(
            f"- messages {ref.message_range[0]}–"
            f"{ref.message_range[1]}: "
            f"ref={ref.ref} ({ref.summary})"
        )
    return "\n".join(lines)
