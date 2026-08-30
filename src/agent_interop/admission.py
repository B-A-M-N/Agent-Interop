"""Context capacity state and inference admission control (P0.6/P0.19).

Provides:
1. ContextCapacity — explicit state for context limits (known/configured/unknown)
   with resolution order: observed runtime → route config → ollama_num_ctx → profile → UNKNOWN.
2. InferenceAdmissionController — route/model semaphore to prevent local inference
   from being overwhelmed by client parallelism.

P0.13: Release is exception-safe via context-manager acquisition token.
P0.14: max_queued_generations is actually enforced (counts waiters).
P0.18: queued counter decremented on acquire; CancelledError re-raised;
       BoundedSemaphore; typed acquisition result.
"""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Self


@dataclass(frozen=True)
class ContextCapacity:
    """Explicit context-capacity state.

    Resolution order:
        1. observed runtime allocation (from /api/show or /api/ps)
        2. explicit route context_limit_tokens
        3. explicit ollama_num_ctx
        4. trusted model profile
        5. UNKNOWN (must not mean infinite)
    """

    state: Literal["known", "configured", "unknown"] = "unknown"
    tokens: int | None = None
    source: str = ""

    @property
    def is_unknown(self) -> bool:
        return self.state == "unknown"

    @property
    def effective_limit(self) -> int | None:
        """Return the effective token limit, or None if unknown.

        Unknown does NOT mean infinite — callers must handle None explicitly.
        """
        return self.tokens

    @property
    def safe_limit(self) -> int | None:
        """Return the safe limit (90% of effective), or None if unknown."""
        if self.tokens is None:
            return None
        return int(self.tokens * 0.90)


def resolve_context_capacity(
    *,
    observed_runtime: int = 0,
    route_context_limit: int = 0,
    ollama_num_ctx: int = 0,
    profile_max_context: int = 0,
) -> ContextCapacity:
    """Resolve context capacity from available sources.

    Returns ContextCapacity with state="unknown" if no source provides a limit.
    Unknown does NOT mean infinite — it means the caller must either:
    - Fail with CONTEXT_CAPACITY_UNKNOWN
    - Apply a conservative operator fallback
    """
    if observed_runtime > 0:
        return ContextCapacity(
            state="known",
            tokens=observed_runtime,
            source="observed_runtime",
        )
    if route_context_limit > 0:
        return ContextCapacity(
            state="configured",
            tokens=route_context_limit,
            source="route_config",
        )
    if ollama_num_ctx > 0:
        return ContextCapacity(
            state="configured",
            tokens=ollama_num_ctx,
            source="ollama_num_ctx",
        )
    if profile_max_context > 0:
        return ContextCapacity(
            state="configured",
            tokens=profile_max_context,
            source="model_profile",
        )
    return ContextCapacity(state="unknown", tokens=None, source="none")


# ─── Admission control (P0.18) ──────────────────────────────────────────────


class AdmissionResult(str, Enum):
    """Typed result of an admission attempt (P0.18)."""
    ACQUIRED = "acquired"
    QUEUE_FULL = "queue_full"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass
class AdmissionConfig:
    """Configuration for inference admission control."""

    max_concurrent_generations: int = 1
    max_queued_generations: int = 16
    queue_timeout_seconds: float = 30.0


@dataclass
class _PerKeyState:
    """Mutable per-route admission state (P0.18: BoundedSemaphore)."""

    semaphore: asyncio.BoundedSemaphore
    queued: int = 0
    active: int = 0
    # P1-H: the capacity the semaphore was built with, so a route-level
    # tightening can compare against what is actually provisioned.
    capacity: int = 0


@dataclass
class _GenerationSlot:
    """P0.13: Acquisition token that auto-releases on __aexit__.

    P0-28: the typed acquisition result rides ON the slot so callers can
    distinguish queue-full from timeout instead of every failure collapsing
    into a generic "unavailable".
    """

    controller: InferenceAdmissionController
    key: str
    result: AdmissionResult = AdmissionResult.QUEUE_FULL
    released: bool = False

    @property
    def acquired(self) -> bool:
        return self.result == AdmissionResult.ACQUIRED

    def release_now(self) -> None:
        """Release the slot immediately (idempotent).

        P1-H: a streaming caller finishes reading the upstream transport
        long before it finishes emitting the buffered/validated tail to the
        client; a slow client must not keep a backend generation slot
        pinned. Call this at transport EOF — ``__aexit__`` stays safe
        (this flag is the same one it checks) and every error path still
        releases through the normal context-manager exit.
        """
        if not self.released and self.acquired:
            self.controller.release_by_key(self.key)
            self.released = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.release_now()


class InferenceAdmissionController:
    """Route/model semaphore to prevent local inference from being overwhelmed.

    Keyed by backend URL + served model. Metadata queries don't need
    the same semaphore.

    P0.13: ``generation_slot()`` returns an async context manager that
    guarantees release even if the caller's body raises.

    P0.14: ``max_queued_generations`` counts waiters; exceeding it
    returns ``acquire()`` == False immediately without blocking.

    P0.18: queued counter decremented on successful acquire; CancelledError
    is re-raised (not swallowed); BoundedSemaphore prevents over-release;
    typed result distinguishes queue-full/timeout/cancel.
    """

    def __init__(self, config: AdmissionConfig | None = None) -> None:
        self._config = config or AdmissionConfig()
        self._states: dict[str, _PerKeyState] = {}
        self._lock = asyncio.Lock()
        # P0-15: per-task slot ownership, keyed by admission key. A
        # ContextVar (not a thread-local) so concurrent requests in one
        # event loop each track their own nesting depth, and child tasks
        # spawned by a holder inherit the count naturally.
        # P1-H: default=None with copy-on-write access — a single shared
        # mutable default dict is visible across ALL tasks that never set
        # the var, so one task's slot count leaked into every other task
        # (task A's held slot made task B believe it held one too, letting
        # B bypass the concurrency cap entirely).
        self._task_slots: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
            f"admission_slots_{id(self)}", default=None,
        )

    def _slots(self) -> dict[str, int]:
        """Copy-on-write view of this task's slot counts."""
        slots = self._task_slots.get()
        if slots is None:
            slots = {}
            self._task_slots.set(slots)
        return slots

    def _key(self, backend_url: str, model: str) -> str:
        return f"{backend_url}::{model}"

    async def set_route_capacity(self, backend_url: str, model: str, capacity: int) -> None:
        """Tighten one route's concurrency cap below the configured default.

        P1-H: the effective cap for a key is min(route override, global
        default). Only lowering is permitted — raising would silently
        override the operator's global protection. An existing semaphore
        is swapped only when it is completely idle (no holders, no
        waiters); otherwise the tightened cap applies from the next
        fully-drained cycle, which is the safe order under load.
        """
        if capacity < 1:
            return
        state = await self._get_state(backend_url, model)
        target = min(capacity, self._config.max_concurrent_generations)
        async with self._lock:
            current_capacity = getattr(state, "capacity", self._config.max_concurrent_generations)
            if target >= current_capacity:
                return
            if state.active == 0 and state.queued == 0 and state.semaphore._value == current_capacity:
                state.semaphore = asyncio.BoundedSemaphore(target)
                state.capacity = target

    async def _get_state(self, backend_url: str, model: str) -> _PerKeyState:
        key = self._key(backend_url, model)
        if key not in self._states:
            async with self._lock:
                if key not in self._states:
                    self._states[key] = _PerKeyState(
                        semaphore=asyncio.BoundedSemaphore(
                            self._config.max_concurrent_generations
                        ),
                        capacity=self._config.max_concurrent_generations,
                    )
        return self._states[key]

    async def acquire(
        self,
        backend_url: str,
        model: str,
        *,
        timeout: float | None = None,
    ) -> AdmissionResult:
        """Acquire permission to run a generation.

        Returns an AdmissionResult indicating the outcome.

        P0.14: Checks max_queued_generations BEFORE blocking on the
        semaphore. The queued counter is incremented while waiting and
        decremented on acquire or timeout.

        P0.18: CancelledError is re-raised after cleanup, never swallowed.

        P0-15 (reentrancy): a task that already holds this key's slot — the
        outer request generation running a private continuation or controller
        turn through the same choke point — re-enters without re-queueing.
        Admission bounds CONCURRENT REQUESTS; it must not deadlock a single
        request's own nested generations against the slot it already holds.
        """
        key = self._key(backend_url, model)
        state = await self._get_state(backend_url, model)
        timeout = timeout or self._config.queue_timeout_seconds

        # P1-H: route-aware capacity — operators serve several served models
        # (and quantizations) behind one backend URL with very different
        # concurrency budgets, so the route may tighten (never widen) the
        # configured per-key default.
        slots = self._slots()
        held = slots.get(key, 0)
        if held:
            slots[key] = held + 1
            return AdmissionResult.ACQUIRED

        # P0-29: try an IMMEDIATE acquisition first. With
        # max_queued_generations=0 ("no waiting") this is the only path — a
        # free slot executes, a busy backend rejects fast. A bare yield is
        # required for the semaphore waiter to run, so use a zero-wait
        # round through the event loop rather than wait_for(timeout=0),
        # which times out before the coroutine ever runs.
        acquired_immediately = state.semaphore.locked() is False
        if acquired_immediately:
            await state.semaphore.acquire()
            slots[key] = slots.get(key, 0) + 1
            async with self._lock:
                state.active += 1
            return AdmissionResult.ACQUIRED

        # Slow path: would have to wait. Queue capacity applies before
        # blocking (P0.14): too many already queued for this key -> reject.
        async with self._lock:
            if state.queued >= self._config.max_queued_generations:
                return AdmissionResult.QUEUE_FULL
            state.queued += 1

        try:
            await asyncio.wait_for(state.semaphore.acquire(), timeout=timeout)
            async with self._lock:
                # P0.18: we were queued, now we're active — decrement queued.
                state.queued = max(0, state.queued - 1)
                state.active += 1
            slots = self._slots()
            slots[key] = slots.get(key, 0) + 1
            return AdmissionResult.ACQUIRED
        except asyncio.CancelledError:
            # P0.18: release the queued slot we were holding, then re-raise.
            async with self._lock:
                state.queued = max(0, state.queued - 1)
            raise
        except TimeoutError:
            # Release the queued slot we were holding
            async with self._lock:
                state.queued = max(0, state.queued - 1)
            return AdmissionResult.TIMED_OUT

    def release(self, backend_url: str, model: str) -> None:
        """Release a generation slot."""
        key = self._key(backend_url, model)
        self.release_by_key(key)

    def release_by_key(self, key: str) -> None:
        """Release a generation slot by its internal key.

        P0-15 (reentrancy): nested acquisitions by the same task consumed no
        semaphore capacity, so only the OUTERMOST release (count reaching 0)
        actually frees the slot.
        """
        state = self._states.get(key)
        if state is None:
            return
        slots = self._slots()
        held = slots.get(key, 0)
        if held > 1:
            # Nested holder — just drop one level of the reentrancy count.
            slots[key] = held - 1
            return
        if held == 1:
            slots.pop(key, None)
        # P0.18: BoundedSemaphore raises ValueError on over-release —
        # that's a bug we want to know about, not silently swallow.
        state.semaphore.release()
        # P0.18: these mutations are protected by the invariant that
        # release() is only called when we hold a slot. We decrement
        # under the lock for consistency with acquire().
        # NOTE: this is called from sync context (release()) and async
        # context (__aexit__); the GIL makes individual int ops atomic,
        # but we accept a small race window for the counter display.
        state.active = max(0, state.active - 1)

    @asynccontextmanager
    async def generation_slot(
        self,
        backend_url: str,
        model: str,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[_GenerationSlot]:
        """P0.13: Context-manager acquisition that is exception-safe.

        P0-28: ALWAYS yields a typed slot — ``slot.acquired`` says whether the
        generation may proceed and ``slot.result`` distinguishes QUEUE_FULL
        from TIMED_OUT. The slot releases on __aexit__ only when actually
        acquired, regardless of how the body exits.
        """
        result = await self.acquire(backend_url, model, timeout=timeout)
        key = self._key(backend_url, model)
        slot = _GenerationSlot(controller=self, key=key, result=result)
        try:
            yield slot
        finally:
            if not slot.released and slot.acquired:
                self.release_by_key(key)
                slot.released = True

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass
