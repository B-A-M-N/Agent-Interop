"""Operational attempt-path hint cache (P0-45).

The compatibility ladder re-discovers the same fact on every request: a
given serving tuple fails its native presentation but succeeds prompted.
That rediscovery costs a real backend generation per request.

This module caches the OBSERVED preference — nothing more.  A hint:

* is keyed by the COMPLETE serving tuple (model digest, chat-template
  digest, backend serving config, profile revision, client/protocol,
  tool-surface fingerprint, streaming, tool-choice class) — any change
  invalidates it;
* stores only a preferred AttemptKind, never a capability or permission;
* only REORDERS attempts the planner already deemed permissible.  It can
  never introduce an attempt the plan did not contain, grant a repair
  tier, or flip a mode allowance;
* is short-lived (TTL) and bounded (max entries), and is deliberately NOT
  compatibility evidence — it never reaches the evidence store.
"""

from __future__ import annotations

import hashlib
import time

from agent_interop.planning.types import AttemptKind, CompatibilityAttempt

DEFAULT_HINT_TTL_SECONDS = 900.0
DEFAULT_HINT_MAX_ENTRIES = 256


def attempt_hint_key(
    *,
    model_digest: str,
    template_digest: str,
    serving_config_digest: str,
    profile_revision: str,
    client_protocol: str,
    tool_surface_fingerprint: str,
    streaming: bool,
    tool_choice_class: str,
    prompted_contract_fingerprint: str = "",
) -> str:
    """Stable key over the complete serving tuple.

    Every input that could change how the model presents tools is part of
    the key; a hint computed for one tuple must never surface for another.

    P1.9 (review #19): ``prompted_contract_fingerprint`` is an additional
    dimension — the textual prompt-contract that PROMPTED/TEXTUAL modes
    inject changes how the model surfaces tools, so the same model +
    surface + tool_choice can produce very different hint outcomes
    depending on the contract text. Empty for NATIVE/DISABLED paths.
    """
    payload = "|".join((
        str(model_digest),
        str(template_digest),
        str(serving_config_digest),
        str(profile_revision),
        str(client_protocol),
        str(tool_surface_fingerprint),
        "stream" if streaming else "no-stream",
        str(tool_choice_class),
        str(prompted_contract_fingerprint),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class AttemptHintCache:
    """Short-lived, bounded map from serving tuple → preferred attempt kind."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_HINT_TTL_SECONDS,
        max_entries: int = DEFAULT_HINT_MAX_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> AttemptKind | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        preferred, stored_at = entry
        if (time.monotonic() - stored_at) > self._ttl:
            del self._entries[key]
            return None
        try:
            return AttemptKind(preferred)
        except ValueError:
            del self._entries[key]
            return None

    def record(self, key: str, preferred: AttemptKind) -> None:
        """Record an observed preference (only what was actually accepted)."""
        if not key:
            return
        if len(self._entries) >= self._max and key not in self._entries:
            # Drop the oldest entry (insertion order) — this is an
            # operational optimization, losing one is free.
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        self._entries[key] = (preferred.value, time.monotonic())

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)


def reorder_attempts_by_hint(
    attempts: tuple[CompatibilityAttempt, ...],
    preferred: AttemptKind | None,
) -> tuple[CompatibilityAttempt, ...]:
    """Move the preferred attempt to the FRONT — if (and only if) it is
    already present.

    Reordering never changes the SET of permissible attempts, never adds a
    kind the planner withheld, and never removes a fallback.  The first
    attempt of the original order is the planner's considered default; the
    hint only says "the ladder has already seen that default fail".
    """
    if preferred is None or not attempts:
        return attempts
    for index, attempt in enumerate(attempts):
        if attempt.kind == preferred and index != 0:
            promoted = (attempt,) + attempts[:index] + attempts[index + 1:]
            return promoted
    return attempts
