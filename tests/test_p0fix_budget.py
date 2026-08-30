"""Regression tests for AttemptBudget review fixes (findings 17-19).

(17) max_total_model_tokens was never enforced.
(18) Private continuation generations were not checked against budgets.
(19) record_input_tokens() never set exhausted_by, allowing compounding.
"""

from __future__ import annotations

import time

from agent_interop.execution_attempts.budget import AttemptBudget


class TestReserveGeneration:
    """reserve_generation must block and set exhausted_by on violations."""

    def test_blocks_when_reservation_crosses_max_total_input_tokens(self):
        budget = AttemptBudget(max_total_input_tokens=100, max_total_model_tokens=1_000_000)
        assert budget.total_input_tokens == 0

        # 60 fits under the 100 limit.
        reservation = budget.reserve_generation(estimated_input_tokens=60, output_reserve_tokens=0)
        assert reservation is not None
        assert budget.total_input_tokens == 60

        # A second request for 50 would push input to 110 > 100 → rejected.
        result = budget.reserve_generation(estimated_input_tokens=50, output_reserve_tokens=0)
        assert result is None
        assert budget.exhausted_by == "max_total_input_tokens"

    def test_successful_reservation_increments_visible_totals(self):
        budget = AttemptBudget(max_total_input_tokens=1000, max_total_generated_tokens=1000)
        reservation = budget.reserve_generation(
            estimated_input_tokens=20,
            output_reserve_tokens=30,
        )
        assert reservation is not None
        assert budget.total_input_tokens == 20
        assert budget.generated_tokens == 30
        assert budget.total_model_tokens == 50
        # commit() REPLACES the estimate with the actuals — the double-count
        # the pre-reservation record_actual_usage path had.
        reservation.commit(actual_input_tokens=18, actual_output_tokens=25)
        assert budget.total_input_tokens == 18
        assert budget.generated_tokens == 25
        assert budget.total_model_tokens == 43


class TestRecordActualUsage:
    """record_actual_usage must set exhausted_by on crossing and
    prevent compounding via subsequent reserve_generation/allow."""

    def test_crossing_max_total_input_tokens_sets_exhausted_by(self):
        budget = AttemptBudget(max_total_input_tokens=100, max_total_model_tokens=1_000_000)
        # Record an actual call that pushes us over the input limit.
        budget.record_actual_usage(input_tokens=120, output_tokens=0)
        assert budget.total_input_tokens == 120
        assert budget.exhausted_by == "max_total_input_tokens"

        # A subsequent reserve_generation must be rejected.
        result = budget.reserve_generation(estimated_input_tokens=1, output_reserve_tokens=0)
        assert result is None
        assert budget.exhausted_by == "max_total_input_tokens"

        # allow() must also be rejected.
        result = budget.allow(use_controller=False)
        assert result is False
        assert budget.exhausted_by == "max_total_input_tokens"

    def test_crossing_max_total_model_tokens_sets_exhausted_by(self):
        budget = AttemptBudget(
            max_total_input_tokens=1000,
            max_total_generated_tokens=1000,
            max_total_model_tokens=100,
        )
        # Record usage that crosses model_tokens ceiling but not
        # per-axis ceilings individually.
        budget.record_actual_usage(input_tokens=60, output_tokens=50)
        assert budget.total_input_tokens == 60
        assert budget.generated_tokens == 50
        assert budget.total_model_tokens == 110
        assert budget.exhausted_by == "max_total_model_tokens"

        # A subsequent reserve_generation must also be rejected.
        result = budget.reserve_generation(
            estimated_input_tokens=1,
            output_reserve_tokens=1,
        )
        assert result is None
        assert budget.exhausted_by == "max_total_model_tokens"


class TestMaxTotalModelTokensEnforcement:
    def test_reserve_generation_fails_at_max_total_model_tokens(self):
        budget = AttemptBudget(
            max_total_input_tokens=50000,
            max_total_generated_tokens=50000,
            max_total_model_tokens=100,
        )
        # First reservation uses 60 model tokens (40 input + 20 gen).
        reservation = budget.reserve_generation(
            estimated_input_tokens=40,
            output_reserve_tokens=20,
        )
        assert reservation is not None
        assert budget.total_model_tokens == 60

        # Another reservation of 50 model tokens (30+20) would make
        # 110 > 100 → rejected.
        result = budget.reserve_generation(
            estimated_input_tokens=30,
            output_reserve_tokens=20,
        )
        assert result is None
        assert budget.exhausted_by == "max_total_model_tokens"

    def test_allow_checks_max_total_model_tokens(self):
        budget = AttemptBudget(
            max_total_input_tokens=50000,
            max_total_generated_tokens=50000,
            max_total_model_tokens=100,
        )
        budget.total_input_tokens = 60
        budget.generated_tokens = 50
        # 110 > 100 → allow() fails.
        assert budget.allow(use_controller=False) is False
        assert budget.exhausted_by == "max_total_model_tokens"


class TestDeadlinePath:
    def test_reserve_generation_rejected_after_max_added_latency_ms(self):
        budget = AttemptBudget(max_added_latency_ms=1000)
        # Prime the clock — first call starts it.
        reservation = budget.reserve_generation(
            estimated_input_tokens=1,
            output_reserve_tokens=0,
        )
        assert reservation is not None
        # Artificially age the clock past the deadline.
        budget.started_at = time.monotonic() - 2
        result = budget.reserve_generation(
            estimated_input_tokens=1,
            output_reserve_tokens=0,
        )
        assert result is None
        assert budget.exhausted_by == "max_added_latency_ms"


class TestAllowBackwardCompatibility:
    """Existing allow() behaviour must not regress: use_controller counts
    still work and upstream_attempts still increments."""

    def test_allow_upstream_attempts_increments(self):
        budget = AttemptBudget(
            max_total_input_tokens=1000,
            max_total_generated_tokens=1000,
            max_total_model_tokens=10000,
        )
        # Three upstream calls should succeed.
        for _ in range(3):
            assert budget.allow(use_controller=False) is True
        # Fourth must be rejected.
        assert budget.allow(use_controller=False) is False
        assert budget.upstream_attempts == 3

    def test_allow_controller_attempts_increments(self):
        budget = AttemptBudget(
            max_controller_attempts=2,
            max_total_input_tokens=1000,
            max_total_generated_tokens=1000,
            max_total_model_tokens=10000,
        )
        assert budget.allow(use_controller=True) is True
        assert budget.controller_attempts == 1
        assert budget.allow(use_controller=True) is True
        assert budget.controller_attempts == 2
        assert budget.allow(use_controller=True) is False
        assert budget.exhausted_by == "max_controller_attempts"


class TestGenerationReservation:
    """Two-phase reservation semantics (review edit-order #6).

    reserve_generation() pre-allocates headroom; commit() REPLACES the
    estimate with the upstream's actual usage; release() returns the
    headroom when the generation never happened.  Neither path may
    double-count the estimate.
    """

    def test_commit_replaces_estimate_with_actuals(self):
        budget = AttemptBudget(
            max_total_input_tokens=10_000,
            max_total_generated_tokens=10_000,
            max_total_model_tokens=1_000_000,
        )
        reservation = budget.reserve_generation(
            estimated_input_tokens=500, output_reserve_tokens=200,
        )
        assert reservation is not None
        assert budget.total_input_tokens == 500
        assert budget.generated_tokens == 200

        reservation.commit(actual_input_tokens=320, actual_output_tokens=90)
        assert budget.total_input_tokens == 320
        assert budget.generated_tokens == 90
        assert budget.total_model_tokens == 410
        assert budget.exhausted_by == ""

    def test_commit_marks_consumed_and_repeats_are_noops(self):
        budget = AttemptBudget()
        reservation = budget.reserve_generation(
            estimated_input_tokens=100, output_reserve_tokens=50,
        )
        assert reservation is not None
        reservation.commit(actual_input_tokens=10, actual_output_tokens=5)
        reservation.commit(actual_input_tokens=999, actual_output_tokens=999)
        assert budget.total_input_tokens == 10
        assert budget.generated_tokens == 5

    def test_release_returns_headroom(self):
        budget = AttemptBudget(max_total_input_tokens=100)
        reservation = budget.reserve_generation(estimated_input_tokens=60)
        assert reservation is not None
        assert budget.total_input_tokens == 60
        reservation.release()
        assert budget.total_input_tokens == 0
        # Headroom is genuinely available again.
        again = budget.reserve_generation(estimated_input_tokens=80)
        assert again is not None

    def test_release_then_commit_is_noop(self):
        budget = AttemptBudget()
        reservation = budget.reserve_generation(estimated_input_tokens=40)
        assert reservation is not None
        reservation.release()
        reservation.commit(actual_input_tokens=7, actual_output_tokens=3)
        assert budget.total_input_tokens == 0
        assert budget.generated_tokens == 0

    def test_commit_crossing_ceiling_sets_exhausted_by(self):
        budget = AttemptBudget(max_total_input_tokens=100)
        reservation = budget.reserve_generation(estimated_input_tokens=50)
        assert reservation is not None
        # Actual usage crossed the ceiling the estimate fit under.
        reservation.commit(actual_input_tokens=120, actual_output_tokens=0)
        assert budget.total_input_tokens == 120
        assert budget.exhausted_by == "max_total_input_tokens"
        # And the budget now refuses further reservations.
        assert budget.reserve_generation(estimated_input_tokens=1) is None
