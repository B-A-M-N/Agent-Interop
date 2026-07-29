"""Maps the real conformance test battery (testing/runner.get_standard_tests)
to L0-L4 capability levels, and computes a level result with explicit
evidence semantics.

testing/conformance.py's CapabilityLevel/LEVEL_REQUIREMENTS predates the
real RealConformanceRunner battery and uses test names
("explicit_tool", "implicit_tool", ...) that don't match the real 12-test
battery's names ("explicit_forced_tool", "automatic_tool_selection", ...) —
that mismatch is exactly why level computation was never wired into
cli.py/`/v1/capabilities` at all. This module defines a mapping against the
REAL battery and is the only place cli.py/server/app.py should ask "what
level did this run achieve".

Level computation here also answers three questions the real battery's
result set forces, that a bare pass/fail count cannot:

1. Battery identity — BATTERY_VERSION changes if the test-name-to-level
   mapping or the standard battery's test names change, so a stored level
   result can be recognized as stale evidence rather than presented as
   current forever.
2. Infra vs. behavioral failure — a backend timeout is not evidence the
   model can't do L3; it's evidence the run was inconclusive. Conflating
   the two would silently under-report a model's real level whenever the
   backend hiccups.
3. Repair-assisted vs. unaided — a level computed with the repair pipeline
   on measures the PIPELINE's assisted level, not the model's own raw
   capability. Both numbers matter for different audiences; neither
   should be presented as "the" level without saying which it is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_interop.types import CapabilityLevel

# ─── Real battery -> level mapping ──────────────────────────────────────────
#
# Matches testing.runner.get_standard_tests()'s 12 real test names.
# Cumulative: achieving L3 requires every L1+L2+L3 test to have passed.

TEST_LEVEL_MAP: dict[str, str] = {
    "explicit_forced_tool": "L1",
    "nested_arguments": "L1",
    "malformed_call_repair": "L1",
    "automatic_tool_selection": "L2",
    "no_tool_request": "L2",
    "sequential_calls": "L3",
    "tool_error_recovery": "L3",
    "tool_result_continuation": "L3",
    "history_round_trip": "L3",
    "parallel_calls": "L4",
    "edit_and_verify": "L4",
    "distinct_ids": "L4",
}

_LEVEL_ORDER = ["L1", "L2", "L3", "L4"]

# Error codes representing an upstream/transport/infrastructure failure —
# the run never got a real behavioral answer out of the model, so a test
# ending in one of these must not count as "the model failed this test".
# See interop/errors.py's InteropErrorCode.
INFRA_ERROR_CODES: frozenset[str] = frozenset({
    "BACKEND_UNAVAILABLE", "BACKEND_AUTH_FAILED", "BACKEND_PROTOCOL_ERROR",
    "BACKEND_RATE_LIMITED", "BACKEND_TIMEOUT", "BACKEND_ERROR", "MODEL_NOT_FOUND",
})


def _cumulative_test_names(level: str) -> frozenset[str]:
    """All test names required for `level`, including every lower level."""
    idx = _LEVEL_ORDER.index(level)
    wanted_levels = set(_LEVEL_ORDER[: idx + 1])
    return frozenset(name for name, lvl in TEST_LEVEL_MAP.items() if lvl in wanted_levels)


def _compute_battery_version() -> str:
    """Hash of the sorted (test_name, level) mapping — changes whenever the
    mapping OR the set of test names it covers changes, so a level result
    computed against an old mapping can be told apart from a current one."""
    canonical = ",".join(f"{name}:{level}" for name, level in sorted(TEST_LEVEL_MAP.items()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


BATTERY_VERSION = _compute_battery_version()


@dataclass(frozen=True)
class ConformanceLevelResult:
    """Structured L0-L4 evidence — never just a bare level enum.

    ``level`` is the highest level for which every cumulative test name is
    in ``passed_tests`` — strictly. A test that was infra-inconclusive does
    NOT count as passed (it caps the level like a failure would), but it is
    reported separately in ``infra_inconclusive`` precisely so a caller can
    tell "the model failed a real test" apart from "the run never got a
    real answer" instead of silently treating an outage as a capability
    verdict.
    """

    level: CapabilityLevel = CapabilityLevel.L0
    battery_version: str = BATTERY_VERSION
    contributing_tests: tuple[str, ...] = ()
    passed_tests: tuple[str, ...] = ()
    behavioral_failures: tuple[str, ...] = ()
    infra_inconclusive: tuple[str, ...] = ()
    repair_enabled: bool | None = None


def classify_result_error(error_code: str | None) -> str:
    """Return "infra" if error_code names an infrastructure failure,
    else "behavioral" (includes the no-error-code case: a criteria/
    tool-choice-violation failure recorded as plain text in .error)."""
    if error_code and error_code in INFRA_ERROR_CODES:
        return "infra"
    return "behavioral"


def compute_conformance_level(
    results: Sequence[Any],  # ConformanceRunResult — duck-typed to avoid an import cycle
    *,
    repair_enabled: bool | None = None,
) -> ConformanceLevelResult:
    """Compute a structured L0-L4 result from a real conformance run.

    ``results`` is the list of ConformanceRunResult objects returned by
    RealConformanceRunner.run_test() for (at minimum) the standard battery.
    Only test names present in TEST_LEVEL_MAP are considered — an unknown
    or extra test name is ignored for level purposes (it may still matter
    to the caller for other reporting).
    """
    contributing = tuple(r.test_name for r in results if r.test_name in TEST_LEVEL_MAP)
    passed = tuple(r.test_name for r in results if r.test_name in TEST_LEVEL_MAP and r.passed)
    infra: list[str] = []
    behavioral: list[str] = []
    for r in results:
        if r.test_name not in TEST_LEVEL_MAP or r.passed:
            continue
        error_code = getattr(r, "error_code", None)
        if classify_result_error(error_code) == "infra":
            infra.append(r.test_name)
        else:
            behavioral.append(r.test_name)

    passed_set = set(passed)
    level = CapabilityLevel.L0
    for lvl_name in reversed(_LEVEL_ORDER):
        required = _cumulative_test_names(lvl_name)
        if required and required.issubset(passed_set):
            level = CapabilityLevel(lvl_name)
            break

    return ConformanceLevelResult(
        level=level,
        battery_version=BATTERY_VERSION,
        contributing_tests=contributing,
        passed_tests=passed,
        behavioral_failures=tuple(behavioral),
        infra_inconclusive=tuple(infra),
        repair_enabled=repair_enabled,
    )
