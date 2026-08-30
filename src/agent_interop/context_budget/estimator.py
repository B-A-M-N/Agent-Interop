"""Conservative token estimation with no dependency on a specific tokenizer."""

from __future__ import annotations

import json
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTool
from agent_interop.context_budget.types import ContextBreakdown, RequestCostSnapshot, TokenEstimate


def estimate_json_tokens(value: Any) -> TokenEstimate:
    """Estimate tokens conservatively from UTF-8 bytes.

    Four bytes/token is common English prose, but schemas and code often use
    shorter tokens.  The 3-byte divisor and a small fixed boundary charge
    intentionally over-estimate until a real tokenizer is available.
    """
    raw = json.dumps(value, default=lambda item: getattr(item, "__dict__", str(item)),
                     ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return TokenEstimate(input_tokens=max(1, (len(raw.encode("utf-8")) + 2) // 3 + 4), confidence="conservative_estimate")


def estimate_tool_schema_tokens(tools: list[CanonicalTool] | tuple[CanonicalTool, ...]) -> TokenEstimate:
    return estimate_json_tokens([
        {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        for tool in tools
    ]) if tools else TokenEstimate(0, "exact")


def build_request_cost_snapshot(request: CanonicalRequest) -> "RequestCostSnapshot":
    """Serialize the request ONCE and derive every planning cost from it.

    This is the single serialization point for a planning pass. It prices
    the system prompt and message history exactly as
    ``estimate_request_context`` does, prices the full declared tool
    surface, fingerprints the full surface with the same canonical form the
    evidence key uses, and records per-tool byte lengths so downstream
    subset pricing (tool-surface budgeting, dynamic selection) is pure dict
    arithmetic instead of repeated json.dumps over remaining candidates.

    Substitutability note: for any tool subset S, the historical
    ``estimate_tool_schema_tokens(S)`` equals
    ``estimate_json_tokens([items for S])``. With per-item byte lengths the
    subset cost is ``(sum(item_bytes) - 2*len(S) + ... )`` — the JSON list
    payload with its separators — plus the fixed container charge, which is
    exactly what ``price_tool_subset`` computes. Callers get identical
    numbers without the serialization.
    """
    import hashlib

    system = estimate_json_tokens(request.system)
    messages = estimate_json_tokens(request.messages)
    tools = list(request.tools)

    per_tool_tokens: dict[str, int] = {}
    tool_item_bytes: dict[str, int] = {}
    canonical_items: list[dict[str, Any]] = []
    for tool in tools:
        item = {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
        canonical_items.append(item)
        raw = json.dumps(item, default=lambda item: getattr(item, "__dict__", str(item)),
                         ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        item_bytes = len(raw.encode("utf-8"))
        tool_item_bytes[tool.name] = item_bytes
        # Price of this tool as a single-item list — mirrors what
        # estimate_tool_schema_tokens((tool,)) charges: 2 list brackets +
        # the estimator's +2 boundary, then //3 and +4.
        per_tool_tokens[tool.name] = max(1, (item_bytes + 4) // 3 + 4)

    fingerprint = ""
    schema_bytes = 0
    if tools:
        canonical = sorted(
            ({"name": tool.name, "schema": tool.input_schema} for tool in tools),
            key=lambda entry: entry["name"],
        )
        raw = json.dumps(canonical, sort_keys=True, default=str)
        encoded = raw.encode()
        schema_bytes = len(encoded)
        fingerprint = hashlib.sha256(encoded).hexdigest()[:16]

    return RequestCostSnapshot(
        system_tokens=system.input_tokens,
        message_tokens=messages.input_tokens,
        tool_schema_tokens=estimate_json_tokens(canonical_items).input_tokens if canonical_items else 0,
        tool_schema_bytes=schema_bytes,
        tool_schema_fingerprint=fingerprint,
        per_tool_tokens=per_tool_tokens,
        tool_item_bytes=tool_item_bytes,
    )


def price_tool_subset(
    snapshot: "RequestCostSnapshot",
    tools: list[CanonicalTool] | tuple[CanonicalTool, ...],
) -> int:
    """Price a tool subset from the snapshot without serializing again.

    Matches ``estimate_tool_schema_tokens(subset).input_tokens``: the
    estimator serializes ``[{name, description, schema}, ...]`` compactly
    and charges ``(total_bytes + 2) // 3 + 4``. The list's inter-item
    separators cost exactly ``2 * (len-1)`` bytes (compact "," between
    items) and each item's bytes are recorded in the snapshot, so the
    subset total is pure arithmetic.
    """
    item_bytes = snapshot.tool_item_bytes
    if not tools:
        return 0
    total = 0
    for tool in tools:
        # Unknown tool (not in the snapshot — e.g. a private capability
        # added after the snapshot was taken): fall back to a direct
        # serialization of just that item.
        known = item_bytes.get(tool.name)
        if known is None:
            single = build_request_cost_snapshot(CanonicalRequest(tools=[tool]))
            known = single.tool_item_bytes.get(tool.name, 0)
        total += known
    # Assembled bytes = item bytes + one compact "," per junction + the two
    # list brackets; the estimator then charges its fixed +2 boundary.
    # Reproduces estimate_tool_schema_tokens byte-for-byte.
    return max(1, (total + (len(tools) - 1) + 4) // 3 + 4)


def estimate_request_context(
    request: CanonicalRequest,
    *,
    visible_tools: list[CanonicalTool] | tuple[CanonicalTool, ...] | None = None,
    prompted_contract: str = "",
    output_reserve_tokens: int | None = None,
    provider_overhead_tokens: int = 32,
    snapshot: "RequestCostSnapshot | None" = None,
) -> ContextBreakdown:
    """Break down a request's context cost.

    ``snapshot`` supplies pre-serialized system/message costs and per-tool
    byte lengths from ``build_request_cost_snapshot``; when provided, the
    system/message serializations and any subset tool pricing are pure
    arithmetic. Only pass a snapshot built from the SAME request — message
    history edited after the snapshot was taken would be mispriced.
    """
    if snapshot is not None:
        system_tokens = snapshot.system_tokens
        message_tokens = snapshot.message_tokens
        tools_priced = tuple(visible_tools) if visible_tools is not None else tuple(request.tools)
        tool_tokens = price_tool_subset(snapshot, tools_priced)
    else:
        system_tokens = estimate_json_tokens(request.system).input_tokens
        message_tokens = estimate_json_tokens(request.messages).input_tokens
        tools_priced = tuple(visible_tools) if visible_tools is not None else tuple(request.tools)
        tool_tokens = estimate_tool_schema_tokens(tools_priced).input_tokens
    tool_estimate_tokens = tool_tokens
    contract = estimate_json_tokens(prompted_contract) if prompted_contract else TokenEstimate(0, "exact")
    output = output_reserve_tokens if output_reserve_tokens is not None else request.generation.max_output_tokens
    total = system_tokens + message_tokens + tool_estimate_tokens + contract.input_tokens + provider_overhead_tokens + output
    confidence = "conservative_estimate"
    return ContextBreakdown(
        system_tokens=system_tokens,
        message_tokens=message_tokens,
        tool_schema_tokens=tool_estimate_tokens,
        prompted_contract_tokens=contract.input_tokens,
        provider_overhead_tokens=provider_overhead_tokens,
        output_reserve_tokens=output,
        total_required_tokens=total,
        confidence=confidence,
    )
