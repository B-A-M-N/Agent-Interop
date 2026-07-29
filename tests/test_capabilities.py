"""Tests for the capability-state model in interop.capabilities.

The key contract: ``CapabilityState`` must distinguish metadata claims from
verified results. A ``DECLARED`` capability means only that model metadata
claims it — an unverified claim — so it must NOT count as available. Only
``PROBED`` (backend inspection confirms), ``VERIFIED`` (conformance test
passed), and ``USER_FORCED`` (user explicitly enabled) qualify.
"""

from __future__ import annotations

from agent_interop.capabilities import CapabilityState


class TestCapabilityStateIsAvailable:
    def test_declared_is_not_available(self) -> None:
        """DECLARED must NOT be available — it is an unverified metadata claim.

        Regression test for the bug where ``is_available()`` included
        ``DECLARED``, contradicting the module's stated purpose of
        distinguishing metadata claims from verified results.
        """
        assert CapabilityState.DECLARED.is_available() is False

    def test_unsupported_is_not_available(self) -> None:
        assert CapabilityState.UNSUPPORTED.is_available() is False

    def test_verified_is_available(self) -> None:
        assert CapabilityState.VERIFIED.is_available() is True

    def test_probed_is_available(self) -> None:
        assert CapabilityState.PROBED.is_available() is True

    def test_user_forced_is_available(self) -> None:
        assert CapabilityState.USER_FORCED.is_available() is True

    def test_degraded_is_not_available(self) -> None:
        """DEGRADED (available with limitations) is not 'available' per the
        strict definition — it has not passed conformance and is not forced."""
        assert CapabilityState.DEGRADED.is_available() is False
