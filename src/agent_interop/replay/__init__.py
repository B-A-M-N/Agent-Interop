"""Replay package — capture and replay for deterministic evaluation."""

from agent_interop.replay.capture import capture_case, sanitize_body, sanitize_headers
from agent_interop.replay.compare import PolicyComparison, compare_policies, summarize_comparisons
from agent_interop.replay.runner import replay_all_policies, replay_case
from agent_interop.replay.store import DiagnosticCaseStore
from agent_interop.replay.types import (
    REPAIR_POLICIES,
    CompatibilityKey,
    CompatibilityQuirk,
    CompatibilityResult,
    ReplayCase,
    ReplayInvariant,
    ReplayResult,
)

__all__ = [
    "REPAIR_POLICIES",
    "CompatibilityKey",
    "CompatibilityQuirk",
    "CompatibilityResult",
    "DiagnosticCaseStore",
    "PolicyComparison",
    "ReplayCase",
    "ReplayInvariant",
    "ReplayResult",
    "capture_case",
    "compare_policies",
    "replay_all_policies",
    "replay_case",
    "sanitize_body",
    "sanitize_headers",
    "summarize_comparisons",
]
