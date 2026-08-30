"""P2-J regression: fake-backend perf ceilings + generation-count release metrics.

Locks in:
  * generation accounting — a committed reservation counts, a released one
    does not; purposes are distinct; the public worker generation is
    exactly one on a DIRECT happy path;
  * generation_metrics() exposes the P0-62 release shape (total + by
    purpose) on the token-efficiency record;
  * Interop-side CPU ceilings against an instant fake backend (item 64):
    canonical preparation, projection, render, and the full happy-path
    request each stay far below the model-dominated wall clock.  A fake
    backend at ~0ms means any measured time IS Interop.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
)
from agent_interop.config import (
    CompatibilityConfig,
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.context_budget.estimator import build_request_cost_snapshot
from agent_interop.execution import InteropRequestExecution
from agent_interop.execution_attempts import AttemptBudget
from agent_interop.gateway import Gateway
from agent_interop.transport.http import UpstreamResponse

# ─── P0-62: generation accounting ───────────────────────────────────────────


def test_committed_reservation_counts_released_does_not():
    budget = AttemptBudget()
    reservation = budget.reserve_generation(
        estimated_input_tokens=100, output_reserve_tokens=64, purpose="worker",
    )
    reservation.commit(actual_input_tokens=110, actual_output_tokens=50)
    released = budget.reserve_generation(
        estimated_input_tokens=80, output_reserve_tokens=32,
        purpose="private_continuation",
    )
    released.release()  # never generated
    assert budget.generations_by_purpose == {"worker": 1}
    assert budget.record_generation is not None


def test_purposes_are_counted_separately():
    budget = AttemptBudget()
    budget.record_generation("worker")
    budget.record_generation("worker")
    budget.record_generation("private_continuation")
    budget.record_generation("controller")
    assert budget.generations_by_purpose == {
        "worker": 2,
        "private_continuation": 1,
        "controller": 1,
    }


def test_generation_metrics_release_shape():
    record = InteropRequestExecution().token_efficiency
    metrics = record.generation_metrics()
    assert metrics == {
        "model_generation_count": 0,
        "model_generations_by_purpose": {},
    }
    record.update_from_attempt(purpose="worker", output_tokens=10)
    metrics = record.generation_metrics()
    assert metrics["model_generation_count"] == 1
    assert metrics["model_generations_by_purpose"] == {"worker": 1}
    # The breakdown dict carries the release metric.
    assert "model_generation_count" in record.breakdown_dict()


# ─── Item 64: fake-backend Interop-CPU ceilings ──────────────────────────────


class InstantTransport:
    """A transport that answers in ~0ms — any elapsed time is Interop."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.calls = 0

    async def send(self, request):
        self.calls += 1
        return UpstreamResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self._body).encode("utf-8"),
        )

    @asynccontextmanager
    async def stream(self, request):  # pragma: no cover — non-streaming only
        raise NotImplementedError

    async def close(self) -> None:
        pass


def _ollama_body(text: str = "done") -> dict[str, Any]:
    return {
        "model": "fake-model",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "prompt_eval_count": 12,
        "eval_count": 4,
    }


def _gateway(transport: InstantTransport) -> Gateway:
    route = ModelRoute(
        id="local",
        client_model_aliases=["local"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=4),
        context=ContextConfig(output_reserve_tokens=32, context_limit_tokens=32768),
        # direct → adapted ladder; controlled excluded so a controller can
        # never hide behind these happy paths.
        compatibility=CompatibilityConfig(mode="adapted", allow_controlled=False),
    )
    return Gateway(
        InteropServerConfig(
            default_route_id="local",
            routes={"local": route},
            probe_on_startup=False,
        ),
        transport=transport,  # type: ignore[arg-type]
        allow_invalid_config=True,
    )


def _chat_request(text: str = "hello") -> CanonicalRequest:
    from agent_interop.abi import CanonicalGenerationOptions

    return CanonicalRequest(
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text=text)])],
        generation=CanonicalGenerationOptions(stream=False),
    )


def _tool_request() -> CanonicalRequest:
    from agent_interop.abi import CanonicalGenerationOptions, CanonicalToolChoice

    return CanonicalRequest(
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read it")])],
        tools=[CanonicalTool(
            name="read_file", description="read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )],
        # Forced selection: exactly what the bootstrap battery proves
        # (automatic selection is deliberately NOT probe-provable).
        tool_choice=CanonicalToolChoice.named("read_file"),
        generation=CanonicalGenerationOptions(stream=False),
    )


@pytest.mark.asyncio
async def test_happy_path_generation_count_is_exactly_one():
    """P0-62 release gate: a warm DIRECT chat happy path performs exactly
    one model generation — zero extra generations.  (A cold first request
    additionally runs metadata-only runtime inspection; the metric is a
    WARM-path gate, matching the release criterion.)"""
    gateway = _gateway(InstantTransport(_ollama_body("hi there")))
    # Warm: caches runtime inspection, codecs, and first-use state.
    await gateway.handle_request(_chat_request(), RequestContext())
    gateway._transport.calls = 0
    response = await gateway.handle_request(_chat_request(), RequestContext())
    assert response.error is None
    assert gateway._transport.calls == 1, (
        f"warm chat happy path must be ONE model generation, got {gateway._transport.calls}"
    )


@pytest.mark.asyncio
async def test_tool_happy_path_makes_one_upstream_call():
    """A tool-bearing ADAPTED request also costs exactly one upstream
    generation (the review's invariant: no probes, no controller, no
    summary on the happy path).  The fake backend answers with the tool
    call the forced choice demands, so the ladder accepts rung one."""
    from agent_interop.qualification import QualificationRecord, QualificationState
    from agent_interop.qualification.revision import QUALIFICATION_BATTERY_REVISION

    tool_call_body = {
        "model": "fake-model",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "/tmp/x"},
                },
            }],
        },
        "done": True,
        "prompt_eval_count": 20,
        "eval_count": 6,
    }
    gateway = _gateway(InstantTransport(tool_call_body))
    # Pre-qualified worker: forced-tool evidence with the CURRENT battery,
    # so the request needs no behavioral probes.
    gateway.record_qualification(QualificationRecord(
        model_digest="fake-model",
        state=QualificationState.FORCED_TOOL,
        native_forced_tool=True,
        battery_revision=QUALIFICATION_BATTERY_REVISION,
    ))
    await gateway.handle_request(_chat_request(), RequestContext())  # warm
    gateway._transport.calls = 0
    response = await gateway.handle_request(_tool_request(), RequestContext())
    assert response.error is None, response.error
    assert gateway._transport.calls == 1, (
        f"warm tool happy path must be ONE model generation, got {gateway._transport.calls}"
    )


def test_snapshot_build_is_cheap_on_large_history():
    """Item 64: canonical preparation must stay far below model latency.

    Expressed as a RATIO against a raw json.dumps of the same history, not
    an absolute time: the design property under test is "the snapshot pays
    for ONE serialization and derives everything else from it", so the
    correct reference is the cost of that one serialization measured in the
    SAME process under the SAME machine load. An absolute-ms assertion
    measured the wrong thing — on a loaded box (load 14 of 20 cores) the
    identical work measured 8ms and 60ms on different runs, flaking the
    25ms line with zero code change.
    """
    text = "x" * 4096
    messages = [CanonicalMessage(role="user", content=[CanonicalTextBlock(text=text)])
                for _ in range(256)]  # ~1MB of text
    request = CanonicalRequest(messages=messages)

    def _best_ms(fn, n=9) -> float:
        fn()  # warm
        best = float("inf")
        for _ in range(n):
            start = time.perf_counter()
            fn()
            best = min(best, (time.perf_counter() - start) * 1000)
        return best

    # Reference: one raw serialization of the same 1MB message list.
    def _raw_dumps() -> None:
        json.dumps(request.messages, default=lambda item: getattr(item, "__dict__", str(item)),
                   ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    snapshot_ms = _best_ms(lambda: build_request_cost_snapshot(request))
    raw_ms = _best_ms(_raw_dumps)
    ratio = snapshot_ms / max(raw_ms, 0.01)
    # The snapshot does the serialization PLUS fingerprinting and per-tool
    # pricing (small here). It must stay within ~4x the single raw dumps —
    # a second full serialization would show up as ~2x+ raw on its own and
    # blow well past this bound. (This machine: snapshot ≈ 1.5–2.5x raw.)
    assert ratio < 4.0, (
        f"snapshot build is {ratio:.1f}x a single raw serialization "
        f"(snapshot {snapshot_ms:.1f}ms vs raw {raw_ms:.1f}ms) — the "
        "single-pass design regressed"
    )


@pytest.mark.asyncio
async def test_full_chat_request_interop_cpu_bounded():
    """End-to-end Interop CPU against an instant backend stays far below
    the review's p95 ≤ 10–15ms pre-upstream budget for warm chat."""
    gateway = _gateway(InstantTransport(_ollama_body("hi there")))
    # Warm-up (imports, first-use caches, codec registration, runtime
    # inspection cache).
    await gateway.handle_request(_chat_request(), RequestContext())
    gateway._transport.calls = 0
    start = time.perf_counter()
    response = await gateway.handle_request(_chat_request(), RequestContext())
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert response.error is None
    # A fake backend at ~0ms means this is essentially all Interop.  The
    # review's budget is 10–15ms; the hard regression line here is 50ms —
    # anything above it is an architectural leak (extra probes, extra
    # serializations), not measurement noise.
    assert elapsed_ms < 50.0, f"warm chat Interop wall time {elapsed_ms:.1f}ms"
