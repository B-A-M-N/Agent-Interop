"""Deterministic lexical tool selector."""

from __future__ import annotations

import re

from agent_interop.abi import CanonicalTool


def _terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z0-9_]{2,}", value.lower())}


def rank_tools(
    query: str,
    tools: tuple[CanonicalTool, ...],
    *,
    tool_terms: dict[str, tuple[set[str], set[str], str]] | None = None,
) -> list[CanonicalTool]:
    """Rank tools lexically against ``query``.

    ``tool_terms`` optionally supplies pre-tokenized ``(name_terms,
    desc_terms, name_lower)`` per tool name — a registry-level cache. The
    query side (lower-cased once, tokenized once) is hoisted out of the
    loop either way; tokenizing a large tool registry per request was the
    dominant CPU cost of dynamic selection.
    """
    query_lower = query.lower()
    query_terms = _terms(query_lower)
    scored: list[tuple[int, str, CanonicalTool]] = []
    for tool in tools:
        if tool_terms is not None and tool.name in tool_terms:
            name_terms, desc_terms, name_lower = tool_terms[tool.name]
        else:
            name_terms = _terms(tool.name)
            desc_terms = _terms(tool.description)
            name_lower = tool.name.lower()
        exact = 20 if name_lower in query_lower else 0
        score = exact + 4 * len(query_terms & name_terms) + len(query_terms & desc_terms)
        scored.append((score, tool.name, tool))
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))]


def build_tool_terms_cache(
    tools: list[CanonicalTool] | tuple[CanonicalTool, ...],
) -> dict[str, tuple[set[str], set[str], str]]:
    """Pre-tokenize a tool registry for repeated ``rank_tools`` calls."""
    return {
        tool.name: (_terms(tool.name), _terms(tool.description), tool.name.lower())
        for tool in tools
    }
