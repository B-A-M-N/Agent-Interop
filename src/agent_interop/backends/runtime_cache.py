"""Bounded cache for backend runtime inspection results."""

from __future__ import annotations

import time
from dataclasses import dataclass

from agent_interop.backends.base import ModelRuntimeCapabilities


@dataclass(frozen=True)
class RuntimeCacheKey:
    backend_url: str
    backend_version: str
    model_name: str
    model_digest: str
    serving_config_digest: str


@dataclass
class _CacheEntry:
    value: ModelRuntimeCapabilities
    expires_at: float


class RuntimeCapabilityCache:
    """TTL cache keyed by the complete served-model identity.

    The digest and serving configuration prevent stale decisions after a tag
    is repointed, a context window changes, or a template is replaced.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._entries: dict[RuntimeCacheKey, _CacheEntry] = {}
        # The immutable full key remains authoritative. This bounded lookup
        # index merely avoids a metadata round trip when the route's tag has
        # already been inspected within its TTL.
        self._route_entries: dict[tuple[str, str], _CacheEntry] = {}

    @staticmethod
    def key_for(value: ModelRuntimeCapabilities, backend_url: str) -> RuntimeCacheKey:
        return RuntimeCacheKey(
            backend_url=backend_url.rstrip("/"),
            backend_version=value.backend_version,
            model_name=value.model_name,
            model_digest=value.model_digest,
            serving_config_digest=value.serving_config_digest,
        )

    def get(self, key: RuntimeCacheKey) -> ModelRuntimeCapabilities | None:
        entry = self._entries.get(key)
        if entry is None or entry.expires_at <= time.monotonic():
            self._entries.pop(key, None)
            return None
        return entry.value

    def put(self, value: ModelRuntimeCapabilities, backend_url: str) -> None:
        entry = _CacheEntry(
            value=value,
            expires_at=time.monotonic() + self.ttl_seconds,
        )
        self._entries[self.key_for(value, backend_url)] = entry
        self._route_entries[(backend_url.rstrip("/"), value.model_name)] = entry

    def get_for_route(self, backend_url: str, model_name: str) -> ModelRuntimeCapabilities | None:
        """Return a fresh route snapshot without confusing it for evidence."""
        key = (backend_url.rstrip("/"), model_name)
        entry = self._route_entries.get(key)
        if entry is None or entry.expires_at <= time.monotonic():
            self._route_entries.pop(key, None)
            return None
        return entry.value

    def clear(self) -> None:
        self._entries.clear()
        self._route_entries.clear()
