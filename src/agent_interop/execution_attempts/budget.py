"""Bounded fallback-execution budget."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class AttemptBudget:
    max_upstream_attempts: int = 3
    max_controller_attempts: int = 2
    max_added_latency_ms: int = 30000
    max_total_generated_tokens: int = 8192
    upstream_attempts: int = 0
    controller_attempts: int = 0
    generated_tokens: int = 0
    exhausted_by: str = ""
    started_at: float = 0.0

    def allow(self, use_controller: bool) -> bool:
        if not self.started_at:
            self.started_at = time.monotonic()
        if (time.monotonic() - self.started_at) * 1000 >= self.max_added_latency_ms:
            self.exhausted_by = "max_added_latency_ms"
            return False
        if self.generated_tokens >= self.max_total_generated_tokens:
            self.exhausted_by = "max_total_generated_tokens"
            return False
        if use_controller:
            if self.controller_attempts >= self.max_controller_attempts:
                self.exhausted_by = "max_controller_attempts"
                return False
            self.controller_attempts += 1
            return True
        if self.upstream_attempts >= self.max_upstream_attempts:
            self.exhausted_by = "max_upstream_attempts"
            return False
        self.upstream_attempts += 1
        return True

    def record_generated_tokens(self, count: int) -> None:
        self.generated_tokens += max(0, count)
        if self.generated_tokens > self.max_total_generated_tokens:
            self.exhausted_by = "max_total_generated_tokens"
