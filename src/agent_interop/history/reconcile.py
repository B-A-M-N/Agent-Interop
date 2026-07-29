"""History reconciliation — repair conversation history before upstream rendering.

Ensures malformed historical tool calls are preserved or safely repaired,
tool call/result IDs are consistent, and ordering is valid.

Critical rule: Never replace malformed historical arguments with {}.
Either repair and mark explicitly, preserve raw for compatible upstream,
or reject with a model-readable correction.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)


@dataclass
class HistoryReconciliationResult:
    """Structured result of reconciling a request's history.

    The gateway must replace ``request.messages`` with ``messages`` and
    surface ``diagnostics`` (without leaking) before building the
    upstream invocation plan.  ``is_safe=False`` indicates the history
    cannot be safely passed to the upstream; the gateway should
    reject the request rather than risk replaying corrupt tool calls.
    """

    messages: list[CanonicalMessage] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    is_safe: bool = True
    unsafe_reasons: list[str] = field(default_factory=list)
    exchanges: list[ToolExchange] = field(default_factory=list)


@dataclass
class _PendingCall:
    """A tool call that has not yet been matched to a result."""
    call_id: str
    msg_idx: int
    block_idx: int
    name: str


@dataclass
class ToolExchange:
    """A single tool call and its corresponding result.

    The canonical call/result pairing view. ``reconcile_history`` builds
    these directly from the pairing data it computes while ingesting the
    message history in order.
    """

    call_id: str
    call_message_index: int
    result_message_index: int | None = None
    call: CanonicalToolCallBlock | None = None
    result: CanonicalToolResultBlock | None = None


def reconcile_history(
    messages: list[CanonicalMessage],
    *,
    session_id: str = "",
    request_id: str = "",
) -> HistoryReconciliationResult:
    """Reconcile conversation history before upstream rendering.

    Sequential, exchange-oriented algorithm:
    - Processes messages in order
    - Tracks pending (unresolved) calls
    - Pairs results with their corresponding calls
    - Marks unsafe for ordering violations, duplicates, orphans

    Operations:
    - Ingest messages into ledger.
    - Validate call/result pairing.
    - Identify malformed historical calls.
    - Preserve raw arguments (never replace with {}).
    - Synthesize missing IDs consistently.

    ``session_id``/``request_id`` seed deterministic ID synthesis (see
    ``synthesize_missing_id``): without them, a missing tool call/result ID
    falls back to a random UUID on every reconciliation pass, so the same
    malformed history produces different synthesized IDs each time —
    breaking replay and any downstream logic that expects a call's ID to be
    stable across retries of the same request.
    """
    actions: list[str] = []
    unsafe_reasons: list[str] = []
    diagnostics: list[str] = []
    repaired: list[CanonicalMessage] = list(messages)

    pending_calls: OrderedDict[str, _PendingCall] = OrderedDict()
    completed_calls: set[str] = set()
    seen_call_ids: set[str] = set()
    call_positions: dict[str, int] = {}  # call_id → msg_idx where it was declared
    seen_result_ids: set[str] = set()
    last_assistant_calls: list[_PendingCall] = []
    exchanges_by_id: dict[str, ToolExchange] = {}  # call_id → exchange (declaration order)

    synthesized_calls = 0
    synthesized_results = 0

    # First pass: synthesize missing call IDs and track calls in order.
    for msg_idx, msg in enumerate(repaired):
        new_blocks: list[Any] = []
        changed = False
        msg_is_assistant = (msg.role == "assistant")

        # Role enforcement: validate block types against message role
        for block_idx, block in enumerate(msg.content):
            if isinstance(block, CanonicalToolCallBlock) and msg.role != "assistant":
                diagnostics.append(
                    f"role_violation:msg={msg_idx}:tool_call_block_in_{msg.role}_message"
                )
                unsafe_reasons.append(
                    f"tool call block in {msg.role} message (msg {msg_idx})"
                )
            if isinstance(block, CanonicalToolResultBlock) and msg.role != "tool":
                diagnostics.append(
                    f"role_violation:msg={msg_idx}:tool_result_block_in_{msg.role}_message"
                )
                unsafe_reasons.append(
                    f"tool result block in {msg.role} message (msg {msg_idx})"
                )

        if msg_is_assistant:
            last_assistant_calls = []

        for block_idx, block in enumerate(msg.content):
            if isinstance(block, CanonicalToolCallBlock):
                call_id = block.id
                if not call_id:
                    call_id = synthesize_missing_id(
                        session_id=session_id,
                        request_id=request_id,
                        message_index=msg_idx,
                        block_index=block_idx,
                        tool_name=block.name,
                    )
                    block = replace(block, id=call_id)
                    actions.append(f"synthesized_id:msg={msg_idx}:name={block.name}")
                    synthesized_calls += 1
                    changed = True
                    new_blocks.append(block)
                else:
                    new_blocks.append(block)

                if call_id in seen_call_ids:
                    diagnostics.append(f"duplicate_call_id:msg={msg_idx}:id={call_id}")
                    unsafe_reasons.append(f"duplicate call ID: {call_id}")
                else:
                    seen_call_ids.add(call_id)
                    call_positions[call_id] = msg_idx
                    pending = _PendingCall(
                        call_id=call_id,
                        msg_idx=msg_idx,
                        block_idx=block_idx,
                        name=block.name,
                    )
                    pending_calls[call_id] = pending
                    # Seed the exchange view with call-side data. The result
                    # fields are filled in during the second pass when a
                    # matching result is found (or left None for unresolved).
                    exchanges_by_id[call_id] = ToolExchange(
                        call_id=call_id,
                        call_message_index=msg_idx,
                        call=block,
                    )
                    if msg_is_assistant:
                        last_assistant_calls.append(pending)
            else:
                new_blocks.append(block)

        if changed:
            repaired[msg_idx] = replace(msg, content=new_blocks)

    # Second pass: match results to calls, synthesize missing result IDs.
    # Reset last_assistant_calls tracking for sequential pairing.
    last_assistant_group: list[_PendingCall] = []

    for msg_idx, msg in enumerate(repaired):
        if msg.role == "assistant":
            # Track the most recent assistant group's pending calls
            last_assistant_group = []
            for block in msg.content:
                if isinstance(block, CanonicalToolCallBlock) and block.id in pending_calls:
                    last_assistant_group.append(pending_calls[block.id])

        new_blocks_2: list[Any] = []
        changed = False
        for block in msg.content:
            if isinstance(block, CanonicalToolResultBlock):
                result_call_id = block.tool_call_id

                if not result_call_id:
                    # Empty result ID — try to pair with preceding assistant group
                    unresolved_in_group = [
                        p for p in last_assistant_group
                        if p.call_id not in completed_calls
                    ]
                    if len(unresolved_in_group) == 1:
                        result_call_id = unresolved_in_group[0].call_id
                        block = replace(block, tool_call_id=result_call_id)
                        actions.append(f"paired_result:msg={msg_idx}:call={result_call_id}")
                        synthesized_results += 1
                        changed = True
                    elif len(unresolved_in_group) > 1:
                        diagnostics.append(f"ambiguous_result:msg={msg_idx}:unresolved={len(unresolved_in_group)}")
                        unsafe_reasons.append(f"ambiguous empty result ID at msg {msg_idx} with {len(unresolved_in_group)} unresolved calls")
                        new_blocks_2.append(block)
                        continue
                    else:
                        # An orphan result already makes this history unsafe
                        # (see unsafe_reasons.append below) — the request
                        # will be rejected before the synthesized ID is ever
                        # rendered upstream. Still synthesize it
                        # deterministically rather than falling back to a
                        # random UUID, so diagnostics/evidence recorded for
                        # this rejection are stable across retries of the
                        # same malformed request.
                        result_call_id = synthesize_missing_id(
                            prefix="orphan",
                            session_id=session_id,
                            request_id=request_id,
                            message_index=msg_idx,
                        )
                        block = replace(block, tool_call_id=result_call_id)
                        actions.append(f"synthesized_orphan_result_id:msg={msg_idx}")
                        synthesized_results += 1
                        changed = True
                        diagnostics.append(f"orphan_result:msg={msg_idx}:id={result_call_id}")
                        unsafe_reasons.append(f"orphan result at msg {msg_idx} with no matching unresolved call")
                        new_blocks_2.append(block)
                        continue

                # Check for result-before-call (ordering violation)
                if result_call_id not in seen_call_ids:
                    diagnostics.append(f"result_before_call:msg={msg_idx}:id={result_call_id}")
                    unsafe_reasons.append(f"result references call {result_call_id} which appears later or never")
                elif call_positions.get(result_call_id, 0) > msg_idx:
                    diagnostics.append(f"result_before_call:msg={msg_idx}:id={result_call_id}")
                    unsafe_reasons.append(f"result references call {result_call_id} which appears at msg {call_positions[result_call_id]}")

                # Check for duplicate results
                if result_call_id in seen_result_ids:
                    diagnostics.append(f"duplicate_result:msg={msg_idx}:id={result_call_id}")
                    unsafe_reasons.append(f"duplicate result for call {result_call_id}")
                else:
                    seen_result_ids.add(result_call_id)

                # Mark as completed
                if result_call_id in pending_calls:
                    completed_calls.add(result_call_id)
                    del pending_calls[result_call_id]
                    # Fill in the result side of the exchange view. `block` is
                    # the (possibly ID-replaced) result block for this call.
                    exchange = exchanges_by_id.get(result_call_id)
                    if exchange is not None:
                        exchange.result = block
                        exchange.result_message_index = msg_idx

                new_blocks_2.append(block)
            else:
                new_blocks_2.append(block)

        if changed:
            repaired[msg_idx] = replace(msg, content=new_blocks_2)

    # Check for unresolved calls (calls without results)
    # Only flag as unsafe if there are subsequent messages after the call
    # (i.e., the conversation continued past the exchange)
    for call_id, pending in pending_calls.items():
        if pending.msg_idx < len(repaired) - 1:
            diagnostics.append(f"unpaired_call:id={call_id}:msg={pending.msg_idx}")
            unsafe_reasons.append(f"call {call_id} has no matching result but conversation continued")

    if synthesized_calls or synthesized_results:
        actions.append(
            f"synthesized_ids: calls={synthesized_calls} results={synthesized_results}"
        )

    # Exchanges are emitted in call-declaration order (the dict preserves
    # first-pass insertion order). Each unresolved call yields an exchange
    # with result/result_message_index left None.
    exchanges = list(exchanges_by_id.values())

    # Historical tool calls that carry raw_arguments but empty parsed
    # arguments (a prior turn's malformed-JSON call, preserved rather than
    # dropped) must be marked arguments_validated=False so codecs re-render
    # them from raw_arguments instead of an empty {}. This was previously a
    # standalone helper (preserve_malformed_arguments) that nothing called.
    repaired = preserve_malformed_arguments(repaired)

    return HistoryReconciliationResult(
        messages=repaired,
        actions=actions,
        diagnostics=diagnostics,
        is_safe=not unsafe_reasons,
        unsafe_reasons=unsafe_reasons,
        exchanges=exchanges,
    )


def preserve_malformed_arguments(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    """Ensure malformed historical tool calls preserve their raw arguments.

    When rendering upstream, if a call has raw_arguments and arguments_validated=False,
    the raw value should be used instead of json.dumps(arguments).

    This function marks messages as repaired when malformed arguments are detected,
    ensuring downstream codecs can resend raw arguments when required.
    """
    from dataclasses import replace

    result = []
    for msg in messages:
        new_blocks = []
        msg_changed = False
        for block in msg.content:
            if isinstance(block, CanonicalToolCallBlock):
                # If the block has raw_arguments but empty arguments,
                # preserve the raw_arguments for upstream rendering
                if block.raw_arguments and not block.arguments:
                    # Mark as repaired so the codec knows to use raw_arguments
                    block = replace(block, arguments_validated=False)
                    msg_changed = True
            new_blocks.append(block)
        if msg_changed:
            msg = replace(msg, content=new_blocks)
        result.append(msg)
    return result


def synthesize_missing_id(
    prefix: str = "tc",
    *,
    session_id: str = "",
    request_id: str = "",
    message_index: int = 0,
    block_index: int = 0,
    tool_name: str = "",
    original_id: str = "",
) -> str:
    """Generate a deterministic missing tool call ID.

    Uses a stable digest of session/request context plus position when
    either is available, falling back to a random UUID only when NEITHER
    is provided (e.g. a caller with no request context at all). A random ID
    here means the same malformed history produces a DIFFERENT synthesized
    ID on every reconciliation pass — breaking replay and any downstream
    logic (evidence, result pairing on retry) that expects a call's
    synthesized ID to be stable for the same request.
    """
    import hashlib

    if session_id or request_id or original_id:
        digest_input = (
            f"{session_id}:{request_id}:{message_index}:{block_index}:"
            f"{tool_name}:{original_id}"
        )
        digest = hashlib.sha256(digest_input.encode()).hexdigest()[:12]
        return f"{prefix}_{digest}"
    # Fallback for when no context is available
    import uuid
    return f"{prefix}_{uuid.uuid4().hex[:12]}"
