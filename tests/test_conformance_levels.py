"""Tests for interop.testing.levels — the real test-battery -> L0-L4 mapping.

Wires the actual 12-test battery (testing.runner.get_standard_tests) to a
capability level, replacing the orphaned testing.conformance module (whose
LEVEL_REQUIREMENTS used test names the real runner never produces).
"""

from __future__ import annotations

from agent_interop.testing.levels import (
    BATTERY_VERSION,
    TEST_LEVEL_MAP,
    compute_conformance_level,
)
from agent_interop.testing.runner import (
    ConformanceRunResult,
    get_standard_tests,
    with_repair_disabled,
)
from agent_interop.types import CapabilityLevel


def _result(name: str, *, passed: bool, error_code: str | None = None) -> ConformanceRunResult:
    return ConformanceRunResult(test_name=name, passed=passed, error_code=error_code)


class TestBatteryMapping:
    def test_every_standard_test_name_is_mapped(self) -> None:
        """The real 12-test battery and TEST_LEVEL_MAP must agree exactly —
        an unmapped test silently never contributes to any level."""
        real_names = {t.name for t in get_standard_tests()}
        assert real_names == set(TEST_LEVEL_MAP)

    def test_battery_version_is_stable_and_nonempty(self) -> None:
        assert BATTERY_VERSION
        from agent_interop.testing.levels import _compute_battery_version
        assert _compute_battery_version() == BATTERY_VERSION


class TestComputeLevel:
    def test_all_pass_reaches_l4(self) -> None:
        results = [_result(name, passed=True) for name in TEST_LEVEL_MAP]
        level_result = compute_conformance_level(results)
        assert level_result.level == CapabilityLevel.L4
        assert not level_result.behavioral_failures
        assert not level_result.infra_inconclusive

    def test_no_pass_stays_l0(self) -> None:
        results = [_result(name, passed=False) for name in TEST_LEVEL_MAP]
        level_result = compute_conformance_level(results)
        assert level_result.level == CapabilityLevel.L0
        assert set(level_result.behavioral_failures) == set(TEST_LEVEL_MAP)

    def test_l1_only_passing_reaches_l1(self) -> None:
        l1_names = {n for n, lvl in TEST_LEVEL_MAP.items() if lvl == "L1"}
        results = [
            _result(name, passed=(TEST_LEVEL_MAP[name] == "L1"))
            for name in TEST_LEVEL_MAP
        ]
        level_result = compute_conformance_level(results)
        assert level_result.level == CapabilityLevel.L1
        assert set(level_result.passed_tests) == l1_names

    def test_single_l3_failure_caps_below_l3_even_if_l4_passed(self) -> None:
        """Cumulative requirement: failing ANY L1-L3 test must cap the
        level below L3 even if every L4 test (checked in isolation) would
        otherwise look like it passed — levels are not independent
        per-tier scores."""
        results = []
        for name in TEST_LEVEL_MAP:
            fail_this_one = name == "sequential_calls"  # an L3 test
            results.append(_result(name, passed=not fail_this_one))
        level_result = compute_conformance_level(results)
        assert level_result.level == CapabilityLevel.L2
        assert "sequential_calls" in level_result.behavioral_failures

    def test_infra_failure_reported_separately_not_as_behavioral(self) -> None:
        """A backend timeout on an L3 test must show up in
        infra_inconclusive, not behavioral_failures — it caps the level
        (evidence is incomplete) but must never be presented as 'the model
        failed this test'."""
        results = []
        for name in TEST_LEVEL_MAP:
            if name == "tool_error_recovery":
                results.append(_result(name, passed=False, error_code="BACKEND_TIMEOUT"))
            else:
                results.append(_result(name, passed=True))
        level_result = compute_conformance_level(results)
        assert "tool_error_recovery" in level_result.infra_inconclusive
        assert "tool_error_recovery" not in level_result.behavioral_failures
        # Still capped below L3 — infra failure means incomplete evidence,
        # not a free pass to a level that was never actually confirmed.
        assert level_result.level == CapabilityLevel.L2

    def test_repair_enabled_flag_is_recorded_not_inferred(self) -> None:
        results = [_result(name, passed=True) for name in TEST_LEVEL_MAP]
        level_result = compute_conformance_level(results, repair_enabled=False)
        assert level_result.repair_enabled is False

    def test_unmapped_test_name_ignored_for_level_purposes(self) -> None:
        results = [_result(name, passed=True) for name in TEST_LEVEL_MAP]
        results.append(_result("some_future_test_not_yet_mapped", passed=False))
        level_result = compute_conformance_level(results)
        assert level_result.level == CapabilityLevel.L4
        assert "some_future_test_not_yet_mapped" not in level_result.contributing_tests


class TestWithRepairDisabled:
    def test_produces_a_fully_disabled_repair_policy(self) -> None:
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            RepairConfig,
            RepairPolicy,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )

        config = InteropServerConfig(
            default_route_id="r",
            routes={
                "r": ModelRoute(
                    id="r",
                    client_model_aliases=["m"],
                    upstream_model="m",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA, base_url="http://x",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    repair=RepairConfig(malformed_json="aggressive", max_regenerations=2),
                ),
            },
        )
        disabled_config = with_repair_disabled(config)
        policy = RepairPolicy.from_config(disabled_config.routes["r"].repair)
        assert policy.enabled_tiers == frozenset()
        assert policy.max_regenerations == 0
        # Original config is untouched (with_repair_disabled returns a copy).
        original_policy = RepairPolicy.from_config(config.routes["r"].repair)
        assert original_policy.enabled_tiers != frozenset()
