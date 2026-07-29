#!/usr/bin/env python3
"""Empirical benchmark: does Interop's tool-call repair layer measurably
improve tool-call success against REAL local/cloud models?

Runs the full standard conformance battery (get_standard_tests(), 12 tests
across L1-L4, each checking real behavior — tool names, argument values,
call order, distinct IDs, same-turn parallelism, not just "was some tool
called") against each configured model, twice: once with repair enabled
(the default RepairConfig) and once with repair fully disabled
(malformed_json=reject, field_aliases=disabled, unknown_tool=reject — the
validate-only baseline, zero repair tiers enabled). Every call goes
through the real Gateway, the real protocol/extraction/repair pipeline,
and a real Ollama backend (local or Ollama-cloud) — nothing here is
mocked or simulated.

This exists to answer, with actual data, a question Interop's own README
asserts but had never measured: does the repair layer produce a
"considerable impact on generated tokens and perceived performance," or
is tool-call success roughly the same with or without it?

Usage:
    uv run python scripts/benchmark_repair_impact.py
    uv run python scripts/benchmark_repair_impact.py --models qwen3-coder:latest
    uv run python scripts/benchmark_repair_impact.py --backend-url http://127.0.0.1:11434

Results are written to benchmark_results.json (machine-readable) and
printed as a human-readable report to stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairConfig,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.testing.runner import RealConformanceRunner, get_standard_tests

DEFAULT_MODELS = [
    "qwen3-coder:latest",
    "qwen2.5-coder:7b",
    "gpt-oss:20b-cloud",
]

# A generous repair latency budget for THIS benchmark specifically — not
# a production default. The point here is measuring whether repair CAN
# fix something, not whether it fits inside a tight production SLA on
# whatever the local machine's load happens to be right now; a slow/
# contended machine hitting the default 15s budget mid-repair produces a
# "budget exhausted" failure that's a timing artifact, not a capability
# signal, and would otherwise masquerade as "repair didn't help."
_BENCHMARK_REPAIR_LATENCY_BUDGET_MS = 120_000

# repair ON: the actual default a route gets, latency budget widened (see
# above) so this benchmark measures capability, not today's system load.
REPAIR_ON = RepairConfig(max_added_latency_ms=_BENCHMARK_REPAIR_LATENCY_BUDGET_MS)
# repair OFF: every repair tier disabled — validate-then-reject, no
# recovery attempted at all. This is the true "no repair" baseline, not
# just a milder repair setting.
REPAIR_OFF = RepairConfig(
    malformed_json="reject",
    field_aliases="disabled",
    unknown_tool="reject",
    max_added_latency_ms=_BENCHMARK_REPAIR_LATENCY_BUDGET_MS,
)


# Real infrastructure/transport failure codes (see errors.py's
# classify_http_status / ERROR_REGISTRY) — genuinely "we couldn't get a
# real answer out of the backend", not a behavioral tool-calling result.
# Deliberately does NOT include validation codes like TOOL_CHOICE_VIOLATION
# (e.g. "Required tool not called") — those ARE real capability evidence
# (the model failed to comply), not infrastructure noise. Matching on the
# generic "Gateway error:" string prefix (as an earlier version of this
# classifier did) is wrong for exactly that reason: both failure classes
# share that prefix.
_UPSTREAM_ERROR_CODES = (
    "BACKEND_ERROR",
    "BACKEND_UNAVAILABLE",
    "BACKEND_TIMEOUT",
    "BACKEND_PROTOCOL_ERROR",
    "BACKEND_RATE_LIMITED",
    "BACKEND_AUTH_FAILED",
    "MODEL_NOT_FOUND",
)


def _classify_outcome(
    *, passed: bool, error: str, tool_call_count: int, final_text: str,
) -> str:
    """Distinguish WHY a test failed, not just that it did.

    Collapsing "the model never attempted a call," "it attempted one but
    extraction never recognized it," "a call was recognized but rejected
    by criteria," and "the backend/transport itself failed" into a single
    pass/fail bit makes it impossible to tell repair's effect from
    infrastructure noise — e.g. qwen3-coder:latest's memory-exhaustion
    failures on a 16GB machine are not evidence about tool-calling
    capability at all, and must not be counted as 0/12 capability
    failures the way a real behavioral failure would be.
    """
    if passed:
        return "passed"
    if error.startswith("CRASH:"):
        return "upstream_error"
    if error and any(f"[{code}]" in error for code in _UPSTREAM_ERROR_CODES):
        return "upstream_error"
    if error:
        # Any other response.error (TOOL_CALL_INVALID, TOOL_CHOICE_VIOLATION,
        # REPAIR_BUDGET_EXHAUSTED, ...) comes from the validation/transaction
        # layer, which only runs when there's at least an attempted output
        # to validate — runner.py's error branch returns before
        # result.tool_calls gets populated for this turn, so
        # tool_call_count == 0 here does NOT by itself mean nothing was
        # extracted. Except transaction.py's own "(no candidates)" messages
        # (a required/named choice with literally zero candidates) — that
        # IS an extraction-stage outcome, not a rejected candidate.
        if "no candidates" in error:
            return "extraction_miss" if final_text.strip() else "no_selection"
        return "candidate_rejected"
    if tool_call_count > 0:
        return "candidate_rejected"  # a call was made/extracted, but failed criteria
    if final_text.strip():
        return "extraction_miss"  # model said something; not recognized as a call
    return "no_selection"  # model produced nothing tool-shaped and no text either


@dataclass
class TestOutcome:
    test_name: str
    passed: bool
    error: str
    turns: int
    tool_call_count: int
    duration_seconds: float
    outcome: str = "failed"
    final_text: str = ""


@dataclass
class ModelRunResult:
    model: str
    repair_mode: str  # "repair_on" | "repair_off"
    outcomes: list[TestOutcome]

    @property
    def passed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.passed)

    @property
    def inconclusive_count(self) -> int:
        """upstream_error outcomes — infrastructure failures, not
        behavioral evidence. Excluded from the capability denominator so
        e.g. an out-of-memory backend doesn't masquerade as 0/12
        capability."""
        return sum(1 for o in self.outcomes if o.outcome == "upstream_error")

    @property
    def total(self) -> int:
        """The capability denominator — total tests MINUS inconclusive
        (infrastructure-failure) ones."""
        return sum(1 for o in self.outcomes if o.outcome != "upstream_error")

    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.total else 0.0


def _timeout_for(model: str) -> float:
    """Cloud models run on Ollama's own infra — fast and consistent
    regardless of local hardware. A local model on a memory-constrained
    machine (e.g. a 16GB laptop running an 18GB+ model file) can be
    heavily disk/swap-backed, so a single inference call may legitimately
    take many minutes rather than seconds — give it generous headroom
    rather than let a slow-but-working call get misclassified as a
    timeout failure.
    """
    return 60.0 if model.endswith(":cloud") else 1200.0


async def run_one(
    *, model: str, backend_url: str, repair_config: RepairConfig, repair_mode: str,
    limit_tests: int | None = None,
) -> ModelRunResult:
    config = InteropServerConfig(
        probe_on_startup=False,
        default_route_id="bench",
        routes={
            "bench": ModelRoute(
                id="bench",
                client_model_aliases=[model],
                upstream_model=model,
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url=backend_url,
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    timeout_seconds=_timeout_for(model),
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
                repair=repair_config,
            ),
        },
    )
    route = config.routes["bench"]

    from agent_interop.evidence.store import EvidenceStore
    runner = RealConformanceRunner(config, evidence_store=EvidenceStore(db_path=":memory:"))
    await runner.start()
    outcomes: list[TestOutcome] = []
    tests = get_standard_tests()
    if limit_tests is not None:
        tests = tests[:limit_tests]
    try:
        for test in tests:
            t0 = time.monotonic()
            print(f"  [{model} | {repair_mode}] {test.name}...", end=" ", flush=True)
            try:
                result = await runner.run_test(test, model_name=model, route=route)
            except Exception as exc:  # a hard crash still counts as a failure, not a lost data point
                outcomes.append(TestOutcome(
                    test_name=test.name, passed=False, error=f"CRASH: {exc}",
                    turns=0, tool_call_count=0, duration_seconds=time.monotonic() - t0,
                    outcome="upstream_error", final_text="",
                ))
                print(f"CRASH ({exc})")
                continue
            outcome = _classify_outcome(
                passed=result.passed, error=result.error,
                tool_call_count=len(result.tool_calls), final_text=result.final_text,
            )
            outcomes.append(TestOutcome(
                test_name=test.name,
                passed=result.passed,
                error=result.error,
                turns=result.turns,
                tool_call_count=len(result.tool_calls),
                duration_seconds=time.monotonic() - t0,
                outcome=outcome,
                final_text=result.final_text[:500],
            ))
            print(f"{outcome.upper()} ({result.error or result.final_text[:80] or 'no tool calls'})"
                  if not result.passed else "PASSED")
    finally:
        await runner.close()

    return ModelRunResult(model=model, repair_mode=repair_mode, outcomes=outcomes)


def print_report(results: list[ModelRunResult]) -> None:
    print("\n" + "=" * 78)
    print("REPAIR IMPACT BENCHMARK — summary")
    print("=" * 78)

    by_model: dict[str, dict[str, ModelRunResult]] = {}
    for r in results:
        by_model.setdefault(r.model, {})[r.repair_mode] = r

    print(f"\n{'Model':<28} {'Repair OFF':>16} {'Repair ON':>16} {'Delta':>10}")
    print("-" * 74)
    for model, modes in by_model.items():
        off = modes.get("repair_off")
        on = modes.get("repair_on")
        if off and off.total == 0:
            print(f"{model:<28} {'INCONCLUSIVE':>16} {'':>16} {'':>10}  "
                  f"({off.inconclusive_count} upstream errors, no capability data)")
            continue
        off_rate = f"{off.passed_count}/{off.total} ({off.pass_rate:.0%})" if off else "n/a"
        on_rate = f"{on.passed_count}/{on.total} ({on.pass_rate:.0%})" if on else "n/a"
        delta = f"{(on.pass_rate - off.pass_rate):+.0%}" if off and on and off.total and on.total else "n/a"
        suffix = ""
        if (off and off.inconclusive_count) or (on and on.inconclusive_count):
            n = (off.inconclusive_count if off else 0) + (on.inconclusive_count if on else 0)
            suffix = f"  ({n} upstream errors excluded)"
        print(f"{model:<28} {off_rate:>16} {on_rate:>16} {delta:>10}{suffix}")

    print("\nPer-test breakdown (tests that flip fail→pass WITH repair on):")
    for model, modes in by_model.items():
        off = modes.get("repair_off")
        on = modes.get("repair_on")
        if not off or not on:
            continue
        off_by_name = {o.test_name: o for o in off.outcomes}
        on_by_name = {o.test_name: o for o in on.outcomes}
        flips = [
            name for name in off_by_name
            if not off_by_name[name].passed and on_by_name.get(name, off_by_name[name]).passed
        ]
        regressions = [
            name for name in off_by_name
            if off_by_name[name].passed and not on_by_name.get(name, off_by_name[name]).passed
        ]
        print(f"\n  {model}:")
        if flips:
            print(f"    Repair recovered: {', '.join(flips)}")
        else:
            print("    Repair recovered: (none)")
        if regressions:
            print(f"    Repair REGRESSED (worse with repair on!): {', '.join(regressions)}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--backend-url", default="http://127.0.0.1:11434")
    parser.add_argument("--out", default="benchmark_results.json")
    parser.add_argument("--limit-tests", type=int, default=None, help="Run only the first N tests (smoke testing)")
    args = parser.parse_args()

    results: list[ModelRunResult] = []
    for model in args.models:
        for mode, cfg in (("repair_off", REPAIR_OFF), ("repair_on", REPAIR_ON)):
            print(f"\n--- Running {model} [{mode}] ---")
            result = await run_one(
                model=model, backend_url=args.backend_url, repair_config=cfg, repair_mode=mode,
                limit_tests=args.limit_tests,
            )
            results.append(result)

    print_report(results)

    out_path = Path(args.out)
    out_path.write_text(json.dumps({
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [
            {**asdict(r), "outcomes": [asdict(o) for o in r.outcomes]}
            for r in results
        ],
    }, indent=2))
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
