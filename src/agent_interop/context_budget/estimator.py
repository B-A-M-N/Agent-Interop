"""Conservative token estimation with no dependency on a specific tokenizer."""

from __future__ import annotations

import json
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTool
from agent_interop.context_budget.types import ContextBreakdown, TokenEstimate


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


def estimate_request_context(
    request: CanonicalRequest,
    *,
    visible_tools: list[CanonicalTool] | tuple[CanonicalTool, ...] | None = None,
    prompted_contract: str = "",
    output_reserve_tokens: int | None = None,
    provider_overhead_tokens: int = 32,
) -> ContextBreakdown:
    system = estimate_json_tokens(request.system)
    messages = estimate_json_tokens(request.messages)
    tool_estimate = estimate_tool_schema_tokens(tuple(visible_tools) if visible_tools is not None else request.tools)
    contract = estimate_json_tokens(prompted_contract) if prompted_contract else TokenEstimate(0, "exact")
    output = output_reserve_tokens if output_reserve_tokens is not None else request.generation.max_output_tokens
    total = system.input_tokens + messages.input_tokens + tool_estimate.input_tokens + contract.input_tokens + provider_overhead_tokens + output
    confidence = "conservative_estimate" if "conservative" in {system.confidence, messages.confidence, tool_estimate.confidence, contract.confidence} else "exact"
    return ContextBreakdown(
        system_tokens=system.input_tokens,
        message_tokens=messages.input_tokens,
        tool_schema_tokens=tool_estimate.input_tokens,
        prompted_contract_tokens=contract.input_tokens,
        provider_overhead_tokens=provider_overhead_tokens,
        output_reserve_tokens=output,
        total_required_tokens=total,
        confidence=confidence,
    )
