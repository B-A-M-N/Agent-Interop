"""Stream-safety observation cache (P0-7).

The buffered-stream gate used to trust only the opt-in evidence store:
without a configured store, ``evidence_record`` was always None, so
schema-v2's ``buffer_unverified_streaming=True`` default buffered every
tool-bearing stream forever — a permanent TTFT tax paid for evidence the
deployment never asked to keep.

This module separates the two concerns:

* **Stream-safety observation** (here) — in-process, automatic, per
  serving tuple: after ONE unbuffered streaming turn whose tool batch was
  fully accepted, that tuple is recorded as having streamed safely. Later
  streams of the same tuple start streaming immediately. A rejected batch
  clears the observation — a tuple that just produced an unexecutable
  call goes back to buffer-until-validated.
* **Compatibility evidence** (the evidence store) — durable, explicit,
  certified by operators. Never written automatically.

An observation is NOT evidence: it is an operational optimization in the
same class as ``AttemptHintCache`` — it only decides whether THIS stream
waits for validation, it can never grant a repair tier, flip a tool-mode
allowance, or surface to the client as "verified". It is keyed over the
complete serving tuple (model digest, template digest, serving config,
client/protocol, tool-surface fingerprint, streaming, tool-choice class)
so one model's behavior never unlocks another's.
"""

from __future__ import annotations

import hashlib
import time

DEFAULT_STREAM_SAFETY_TTL_SECONDS = 900.0
DEFAULT_STREAM_SAFETY_MAX_ENTRIES = 256


def stream_safety_key(
    *,
    model_digest: str,
    template_digest: str,
    serving_config_digest: str,
    profile_revision: str,
    client_protocol: str,
    tool_surface_fingerprint: str,
    tool_choice_class: str,
) -> str:
    """Stable key over the serving tuple that produced the observation.

    Streaming is implicit — observations are only recorded for streams.
    The tool-surface fingerprint and tool-choice class are part of the
    key because they shape what the model was asked to emit: a tuple that
    streamed one surface safely says nothing about a different surface.
    """
    payload = "|".join((
        str(model_digest),
        str(template_digest),
        str(serving_config_digest),
        str(profile_revision),
        str(client_protocol),
        str(tool_surface_fingerprint),
        str(tool_choice_class),
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class StreamSafetyCache:
    """Short-lived, bounded map from serving tuple → streams-safely.

    ``observed_at`` is exposed for metrics; the boolean answer is the
    only thing the buffered-stream gate consumes.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_STREAM_SAFETY_TTL_SECONDS,
        max_entries: int = DEFAULT_STREAM_SAFETY_MAX_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._entries: dict[str, float] = {}

    def is_safe(self, key: str) -> bool:
        """True when this tuple recently streamed a fully-accepted turn."""
        return key in self

    def revoke(self, key: str) -> None:
        """Drop the observation — a rejected batch on this tuple."""
        self._entries.pop(key, None)

    def record(self, key: str) -> None:
        """Record one fully-accepted unbuffered streaming turn."""
        if not key:
            return
        if len(self._entries) >= self._max and key not in self._entries:
            # Drop the oldest entry (insertion order) — losing one is free.
            oldest = next(iter(self._entries))
            del self._entries[oldest]
        self._entries[key] = time.monotonic()

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        stored_at = self._entries.get(key)
        if stored_at is None:
            return False
        if (time.monotonic() - stored_at) > self._ttl:
            del self._entries[key]
            return False
        return True

    def observed_at(self, key: str) -> float | None:
        stored_at = self._entries.get(key)
        if stored_at is None:
            return None
        if (time.monotonic() - stored_at) > self._ttl:
            del self._entries[key]
            return None
        return stored_at
