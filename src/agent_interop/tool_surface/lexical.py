"""Deterministic lexical tool selector."""

from __future__ import annotations

import re

from agent_interop.abi import CanonicalTool


def _terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-zA-Z0-9_]{2,}", value.lower())}


def rank_tools(query: str, tools: tuple[CanonicalTool, ...]) -> list[CanonicalTool]:
    query_terms = _terms(query)
    scored: list[tuple[int, str, CanonicalTool]] = []
    for tool in tools:
        name_terms = _terms(tool.name)
        desc_terms = _terms(tool.description)
        exact = 20 if tool.name.lower() in query.lower() else 0
        score = exact + 4 * len(query_terms & name_terms) + len(query_terms & desc_terms)
        scored.append((score, tool.name, tool))
    return [item[2] for item in sorted(scored, key=lambda item: (-item[0], item[1]))]
