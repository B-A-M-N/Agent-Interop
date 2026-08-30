"""Bounded fallback-execution budget."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class GenerationReservation:
    """Handle to headroom pre-allocated by :meth:`AttemptBudget.reserve_generation`.

    The reservation is a two-phase commit: the estimated tokens are added to
    the running totals at reserve time (so concurrent checks cannot
    double-spend headroom), and :meth:`commit` later reconciles them with the
    upstream's ACTUAL usage by REPLACING the estimate — not adding to it, which
    is the double-count the pre-reservation-object code had.  When the
    generation never happens, :meth:`release` returns the reserved headroom.

    A reservation is single-use: commit/release mark it consumed, and repeat
    calls are no-ops (first one wins) so a caller bug cannot double-release
    the same headroom back into the budget.
    """

    budget: AttemptBudget
    estimated_input_tokens: int = 0
    output_reserve_tokens: int = 0
    purpose: str = "worker"
    _consumed: bool = False

    @property
    def consumed(self) -> bool:
        return self._consumed

    def commit(self, *, actual_input_tokens: int = 0, actual_output_tokens: int = 0) -> None:
        """Replace the estimate with the upstream's actual usage.

        Records the real counts against the ceilings (setting ``exhausted_by``
        when a ceiling is crossed) after first returning the estimated
        headroom, so the totals reflect exactly one generation's worth of
        spend: the actuals.
        """
        if self._consumed:
            return
        self._consumed = True
        self.budget.total_input_tokens = max(
            0, self.budget.total_input_tokens - self.estimated_input_tokens,
        )
        self.budget.generated_tokens = max(
            0, self.budget.generated_tokens - self.output_reserve_tokens,
        )
        self.budget.record_generated_tokens(max(0, actual_output_tokens))
        self.budget.record_input_tokens(max(0, actual_input_tokens))
        # P0-62: the generation happened — count it under its purpose.
        self.budget.record_generation(self.purpose)

    def release(self) -> None:
        """Return the reserved headroom — the generation never happened."""
        if self._consumed:
            return
        self._consumed = True
        self.budget.total_input_tokens = max(
            0, self.budget.total_input_tokens - self.estimated_input_tokens,
        )
        self.budget.generated_tokens = max(
            0, self.budget.generated_tokens - self.output_reserve_tokens,
        )

    def commit_estimated(self) -> None:
        """Keep the estimate — the generation MAY have happened.

        For a generation that was dispatched but whose response failed or
        carried no usage, the backend may already have spent the reserved
        resources. Releasing that headroom would let a flood of failing
        generations bypass the token ceilings; recording the estimate is the
        conservative choice. The generation is also counted under its
        purpose, because dispatch happened even though no usage came back.
        """
        if self._consumed:
            return
        self._consumed = True
        self.budget.record_generation(self.purpose)


@dataclass
class AttemptBudget:
    # P0-62: per-purpose generation counts.  Every model generation on this
    # request goes through reserve_generation()/allow() — tagging there
    # gives the release metric "extra model generations" for free.
    generations_by_purpose: dict = field(default_factory=dict)
    max_upstream_attempts: int = 3
    max_controller_attempts: int = 2
    # P0-16: internal (private-retrieval) generations are bounded separately
    # from the compatibility attempt ladder. A request that pages a large
    # result through two __interop_read_result turns must not lose ladder
    # rungs for it — the two budgets measure different things (how many
    # presentations we may try vs. how many internal round-trips one
    # presentation may need).
    max_private_generations: int = 16
    max_added_latency_ms: int = 30000
    max_total_generated_tokens: int = 8192
    max_total_input_tokens: int = 32768
    max_total_model_tokens: int = 65536
    max_total_rendered_bytes: int = 100_000_000
    upstream_attempts: int = 0
    controller_attempts: int = 0
    private_generations: int = 0
    generated_tokens: int = 0
    total_input_tokens: int = 0
    total_rendered_bytes: int = 0
    exhausted_by: str = ""
    started_at: float = 0.0

    @property
    def total_model_tokens(self) -> int:
        """Running total of input tokens plus generated tokens.

        This is the combined model token count — both are needed
        because ``max_total_model_tokens`` gates the overall model
        context window, independent of the per-axis ``input`` /
        ``generated`` limits.
        """
        return self.total_input_tokens + self.generated_tokens

    def allow(self, use_controller: bool) -> bool:
        if not self.started_at:
            self.started_at = time.monotonic()
        if (time.monotonic() - self.started_at) * 1000 >= self.max_added_latency_ms:
            self.exhausted_by = "max_added_latency_ms"
            return False
        if self.generated_tokens >= self.max_total_generated_tokens:
            self.exhausted_by = "max_total_generated_tokens"
            return False
        if self.total_input_tokens >= self.max_total_input_tokens:
            self.exhausted_by = "max_total_input_tokens"
            return False
        if self.total_model_tokens >= self.max_total_model_tokens:
            self.exhausted_by = "max_total_model_tokens"
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

    def allow_private_generation(self) -> bool:
        """P0-16: headroom check for one private-retrieval generation.

        Counts against ``max_private_generations`` and the token/wall-clock
        ceilings, but NEVER against the compatibility ladder's
        ``upstream_attempts``. Returns False (setting ``exhausted_by``) when
        the request has spent its private-loop allowance.
        """
        if not self.started_at:
            self.started_at = time.monotonic()
        if (time.monotonic() - self.started_at) * 1000 >= self.max_added_latency_ms:
            self.exhausted_by = "max_added_latency_ms"
            return False
        if self.generated_tokens >= self.max_total_generated_tokens:
            self.exhausted_by = "max_total_generated_tokens"
            return False
        if self.total_input_tokens >= self.max_total_input_tokens:
            self.exhausted_by = "max_total_input_tokens"
            return False
        if self.total_model_tokens >= self.max_total_model_tokens:
            self.exhausted_by = "max_total_model_tokens"
            return False
        if self.private_generations >= self.max_private_generations:
            self.exhausted_by = "max_private_generations"
            return False
        self.private_generations += 1
        return True

    def reserve_generation(
        self,
        *,
        estimated_input_tokens: int = 0,
        output_reserve_tokens: int = 0,
        purpose: str = "worker",
    ) -> GenerationReservation | None:
        """Pre-check before every actual model generation call.

        Called before public, private-continuation, and controller model
        calls to prevent over-running any budget ceiling.  Starts the
        clock like ``allow()`` does.

        Returns ``None`` (setting ``exhausted_by``) when *any* of these
        conditions would be violated:

        * ``max_added_latency_ms`` — wall-clock time elapsed.
        * ``total_input_tokens + estimated_input_tokens > max_total_input_tokens``
        * ``generated_tokens + output_reserve_tokens > max_total_generated_tokens``
        * ``(total_input_tokens + generated_tokens) + estimated_input_tokens +
           output_reserve_tokens > max_total_model_tokens``

        On success it returns a :class:`GenerationReservation` holding the
        reserved headroom — the estimate is added to the running totals
        immediately so concurrent or sequential checks cannot double-spend
        the same headroom.

        The caller MUST reconcile the reservation exactly once:
        ``reservation.commit(actual_input_tokens=..., actual_output_tokens=...)``
        after the generation, or ``reservation.release()`` when it never
        happened.  ``commit`` replaces the estimate with the actuals;
        ``release`` returns the headroom.
        """
        if not self.started_at:
            self.started_at = time.monotonic()
        # Deadline check (shared with allow()).
        if (time.monotonic() - self.started_at) * 1000 >= self.max_added_latency_ms:
            self.exhausted_by = "max_added_latency_ms"
            return None
        # Per-axis ceilings on what would exist *after* the reservation.
        projected_input = self.total_input_tokens + estimated_input_tokens
        if projected_input > self.max_total_input_tokens:
            self.exhausted_by = "max_total_input_tokens"
            return None
        projected_gen = self.generated_tokens + output_reserve_tokens
        if projected_gen > self.max_total_generated_tokens:
            self.exhausted_by = "max_total_generated_tokens"
            return None
        projected_model = projected_input + projected_gen
        if projected_model > self.max_total_model_tokens:
            self.exhausted_by = "max_total_model_tokens"
            return None
        # Commit the reservation — prevents double-spend of the same
        # headroom by downstream / private continuation calls.
        self.total_input_tokens = projected_input
        self.generated_tokens = projected_gen
        return GenerationReservation(
            budget=self,
            estimated_input_tokens=estimated_input_tokens,
            output_reserve_tokens=output_reserve_tokens,
            purpose=purpose,
        )

    def record_generation(self, purpose: str = "worker") -> None:
        """P0-62: count one model generation under its purpose tag.

        Counted when the generation actually HAPPENS (reservation commit /
        public-attempt dispatch) — a reservation that is released because
        the generation never ran must not inflate the release metric.

        Purposes seen on a healthy request: ``worker`` (the single public
        generation), ``private_continuation`` (only after the model
        explicitly requested private retrieval), ``controller`` /
        ``context_summary`` (only on explicit fallback paths).
        """
        self.generations_by_purpose[purpose] = (
            self.generations_by_purpose.get(purpose, 0) + 1
        )

    def record_generated_tokens(self, count: int) -> None:
        self.generated_tokens += max(0, count)
        if self.generated_tokens > self.max_total_generated_tokens:
            self.exhausted_by = "max_total_generated_tokens"
        # Also trip when the combined model-total ceiling is crossed
        # by a real generation (reconciliation can push us over a
        # reservation that was conservative).
        if self.total_model_tokens > self.max_total_model_tokens:
            self.exhausted_by = "max_total_model_tokens"

    def reserve_input(self, rendered_token_count: int) -> bool:
        if self.total_input_tokens + rendered_token_count > self.max_total_input_tokens:
            self.exhausted_by = "max_total_input_tokens"
            return False
        return True

    def reserve_rendered_bytes(self, rendered_bytes: int) -> bool:
        if self.total_rendered_bytes + rendered_bytes > self.max_total_rendered_bytes:
            self.exhausted_by = "max_total_rendered_bytes"
            return False
        return True

    def record_input_tokens(self, count: int) -> None:
        """Add actual input tokens consumed by a generation.

        This is the *post-call* reconciliation: the real number of
        prompt tokens the model received for this turn.  When the
        running total exceeds ``max_total_input_tokens`` the method
        sets ``exhausted_by="max_total_input_tokens"`` to stop
        compounding — future ``reserve_generation`` / ``allow`` calls
        will reject rather than allowing the budget to be exceeded
        further.
        """
        self.total_input_tokens += max(0, count)
        if self.total_input_tokens > self.max_total_input_tokens:
            self.exhausted_by = "max_total_input_tokens"

    def record_rendered_bytes(self, count: int) -> None:
        self.total_rendered_bytes += max(0, count)

    def record_actual_usage(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Reconcile a real upstream generation with its actual token usage.

        Reservations from :meth:`reserve_generation` are conservative
        spend estimates; this method simply records the real numbers
        without subtracting reservation headroom (the reservation
        was intentionally over-allocated to avoid mid-flight
        rejection).  Calls :meth:`record_generated_tokens` and
        :meth:`record_input_tokens` so that exhaustion flags are
        set if the running totals cross their ceilings.
        """
        self.record_generated_tokens(output_tokens)
        self.record_input_tokens(input_tokens)
        if self.total_model_tokens > self.max_total_model_tokens:
            self.exhausted_by = "max_total_model_tokens"
