"""Tokenizer resolution seam.

Backends can register an exact counter later; every caller receives a safe
estimate now instead of assuming profile context declarations are enough.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent_interop.context_budget.estimator import estimate_json_tokens
from agent_interop.context_budget.types import TokenEstimate

TokenCounter = Callable[[Any], TokenEstimate]


class TokenizerRegistry:
    def __init__(self) -> None:
        self._counters: dict[str, TokenCounter] = {}

    def register(self, model_family: str, counter: TokenCounter) -> None:
        self._counters[model_family.lower()] = counter

    def estimate(self, value: Any, model_family: str = "") -> TokenEstimate:
        counter = self._counters.get(model_family.lower())
        return counter(value) if counter is not None else estimate_json_tokens(value)

