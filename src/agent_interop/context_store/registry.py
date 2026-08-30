"""Request-scoped lifecycle for ContextStore refs.

Preparation-time projection refs are pinned once at request start, but refs
created LATER in the same request — most importantly schema-on-demand
exposes via ``__interop_get_tool_schema``, which stores a ``tool_schema``
entry only when the model asks — historically escaped every lifetime
mechanism built around ``ResolvedInvocation.pinned_refs``: they were absent
from request pinning, from the output firewall's leak identity, and from
request cleanup.

The registry is the single authority for "refs this request is allowed to
touch": every ref a projection or an internal executor produces goes
through :meth:`RequestRefRegistry.register`, which pins it in the store for
this request. ``snapshot()`` feeds the output firewall, and ``close()`` in
the request's ``finally`` unpins everything, so dynamically created refs can
no longer outlive the request that minted them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from agent_interop.context_store.store import ContextStore

from agent_interop.errors import ContextStoreError, InteropErrorCode

__all__ = ["RequestRefRegistry"]


class RequestRefRegistry:
    """Pins and tracks every ref one request is allowed to observe.

    A ref registers exactly once per request (idempotent repeats are
    ignored — a model paging the same ref twice must not inflate the
    pin count or the firewall identity).
    """

    def __init__(
        self,
        store: "ContextStore",
        session_id: str,
        request_id: str,
    ) -> None:
        self._store = store
        self._session_id = session_id
        self._request_id = request_id
        self._refs: set[str] = set()
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def request_id(self) -> str:
        return self._request_id

    def register(self, ref: str) -> None:
        """Pin ``ref`` to this request; raise if the store rejects it.

        Empty refs are ignored (executors return ``ref=""`` for
        non-storing outcomes). A ref that cannot be pinned means the entry
        was evicted or is otherwise unavailable — surfacing that here keeps
        the registry's snapshot trustworthy as a firewall identity.
        """
        if self._closed:
            raise ContextStoreError(
                code=InteropErrorCode.CONTEXT_STORE_ERROR,
                message="ref registry is closed for this request",
            )
        ref = str(ref or "")
        if not ref or ref in self._refs:
            return
        if not self._store.pin_refs(
            (ref,), self._request_id, session_id=self._session_id,
        ):
            raise ContextStoreError(
                code=InteropErrorCode.CONTEXT_REF_UNAVAILABLE,
                message=(
                    f"context ref {ref!r} is not available in session "
                    f"{self._session_id!r} and cannot be pinned"
                ),
            )
        self._refs.add(ref)

    def register_all(self, refs: Iterable[str]) -> None:
        for ref in refs:
            self.register(ref)

    def snapshot(self) -> frozenset[str]:
        """Every ref this request has observed — the firewall identity."""
        return frozenset(self._refs)

    def close(self) -> None:
        """Unpin every registered ref. Idempotent; never raises."""
        if self._closed:
            return
        self._closed = True
        try:
            self._store.unpin_refs(
                self._refs, self._request_id, session_id=self._session_id,
            )
        except Exception:  # noqa: BLE001 — cleanup must not mask the outcome
            pass
        self._refs.clear()
