"""Bounded controller-session ledger, separate from generic session state."""

from __future__ import annotations

import time
from collections import OrderedDict

from agent_interop.controller.types import ControllerSessionState


class ControllerStateStore:
    def __init__(self, max_entries: int = 512, ttl_seconds: float = 3600.0) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[tuple[str, str, str], tuple[float, ControllerSessionState]] = OrderedDict()

    @staticmethod
    def key(session_id: str, client_id: str, primary_route_id: str) -> tuple[str, str, str]:
        return session_id, client_id, primary_route_id

    def get(self, session_id: str, client_id: str, primary_route_id: str) -> ControllerSessionState | None:
        key = self.key(session_id, client_id, primary_route_id)
        item = self._entries.get(key)
        if item is None:
            return None
        expires_at, state = item
        if expires_at <= time.monotonic():
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return state

    def put(self, state: ControllerSessionState) -> None:
        key = self.key(state.session_id, state.client_id, state.primary_route_id)
        self._entries[key] = (time.monotonic() + self.ttl_seconds, state)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def remove(self, session_id: str, client_id: str, primary_route_id: str) -> None:
        """Discard a completed/failed controller conversation."""
        self._entries.pop(self.key(session_id, client_id, primary_route_id), None)
