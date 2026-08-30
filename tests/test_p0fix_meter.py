"""Tests for the full-serving-identity calibration fix (P0.10).

The calibration store is now keyed by (model_digest, chat_template_digest,
wire_protocol) instead of model_digest alone.  A template or protocol change
while the model digest is retained MUST NOT reuse the stale calibration.
"""

from __future__ import annotations

from agent_interop.context_budget.meter import TokenMeter


class TestFullServingIdentityCalibration:
    """Calibration is keyed by (model_digest, chat_template_digest, wire_protocol)."""

    def test_calibrated_identity_returns_confidence_calibrated(self) -> None:
        """Same identity after calibrate => confidence 'calibrated'."""
        meter = TokenMeter()
        body = b"x" * 300
        meter.calibrate("d1", 100, 300, chat_template_digest="tA", wire_protocol="ollama")
        result = meter.estimate(
            None,  # type: ignore[arg-type] — test doesn't use canonical structure
            rendered_body=body,
            model_digest="d1",
            chat_template_digest="tA",
            wire_protocol="ollama",
        )
        assert result.confidence == "calibrated"
        assert result.input_tokens == 100  # 300 bytes / 3.0 bpt

    def test_different_template_returns_confidence_estimated(self) -> None:
        """Same model_digest but different chat_template_digest => no stale reuse."""
        meter = TokenMeter()
        body = b"x" * 300
        meter.calibrate("d1", 100, 300, chat_template_digest="tA", wire_protocol="ollama")
        result = meter.estimate(
            None,  # type: ignore[arg-type]
            rendered_body=body,
            model_digest="d1",
            chat_template_digest="tB",  # different template
            wire_protocol="ollama",
        )
        # Must NOT fall through to the stale digest-only entry
        assert result.confidence == "estimated"
        assert result.source == "conservative_3bytes_per_token"

    def test_different_wire_protocol_returns_confidence_estimated(self) -> None:
        """Same model_digest + same template but different wire_protocol => no stale reuse."""
        meter = TokenMeter()
        body = b"x" * 300
        meter.calibrate("d1", 100, 300, chat_template_digest="tA", wire_protocol="proto1")
        result = meter.estimate(
            None,  # type: ignore[arg-type]
            rendered_body=body,
            model_digest="d1",
            chat_template_digest="tA",
            wire_protocol="proto2",  # different protocol
        )
        assert result.confidence == "estimated"
        assert result.source == "conservative_3bytes_per_token"

    def test_measure_rendered_with_actual_tokens_calibrates_and_returns_exact(self) -> None:
        """measure_rendered with actual_prompt_tokens calibrates under the full triple."""
        meter = TokenMeter()
        body = b"x" * 300
        result = meter.measure_rendered(
            body,
            model_digest="d2",
            chat_template_digest="tX",
            wire_protocol="ollama",
            actual_prompt_tokens=100,
        )
        assert result.confidence == "exact"
        assert result.input_tokens == 100

        # A subsequent call with the exact same identity should be calibrated
        result2 = meter.measure_rendered(
            body,
            model_digest="d2",
            chat_template_digest="tX",
            wire_protocol="ollama",
        )
        assert result2.confidence == "calibrated"
        assert result2.input_tokens == 100  # 300 / 3.0

    def test_legacy_positional_call_still_works(self) -> None:
        """Legacy positional call calibrate("d", 100, 300) then estimate with empty identity."""
        meter = TokenMeter()
        body = b"x" * 300
        # Old-style positional call — defaults to empty strings for new params.
        meter.calibrate("d", 100, 300)
        result = meter.estimate(
            None,  # type: ignore[arg-type]
            rendered_body=body,
            model_digest="d",
            chat_template_digest="",
            wire_protocol="",
        )
        assert result.confidence == "calibrated"
        assert result.input_tokens == 100

    def test_legacy_positional_via_measure_rendered(self) -> None:
        """Legacy positional call calibrate then measure_rendered is calibrated."""
        meter = TokenMeter()
        body = b"x" * 300
        meter.calibrate("d", 100, 300)
        # measure_rendered defaults new keyword-only params to ""
        result = meter.measure_rendered(body, model_digest="d")
        assert result.confidence == "calibrated"
        assert result.input_tokens == 100

    def test_no_fallback_to_digest_only(self) -> None:
        """Bug regression: digest-only key must never be consulted as fallback."""
        meter = TokenMeter()
        body = b"x" * 300
        # Register calibration under empty template/protocol
        meter.calibrate("d1", 100, 300)
        # Look up with non-empty template — should NOT fall back to the empty-template entry.
        result = meter.estimate(
            None,  # type: ignore[arg-type]
            rendered_body=body,
            model_digest="d1",
            chat_template_digest="different",
            wire_protocol="",
        )
        assert result.confidence == "estimated"
        assert result.source == "conservative_3bytes_per_token"
