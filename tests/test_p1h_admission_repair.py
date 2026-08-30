"""P1-H regression: admission correctness + repair validator caching.

Locks in:
  * the admission ContextVar is copy-on-write — one task's held slot must
    never let another task bypass the concurrency cap;
  * per-route capacity may only TIGHTEN the global cap;
  * release_now() releases a streaming slot before the client-facing tail;
  * compiled JSON-schema validators are cached per canonical schema form
    (and invalid schemas are negatively cached);
  * the repair loop stops on a no-progress iteration instead of spinning;
  * CompiledToolRegistry gives the same canonicalization/lookup results as
    per-candidate map rebuilding.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_interop.abi import CanonicalTool
from agent_interop.admission import AdmissionConfig, AdmissionResult, InferenceAdmissionController
from agent_interop.repair.schema import (
    _VALIDATOR_CACHE,
    _cached_validator,
    validate_against_schema,
)
from agent_interop.transaction import CompiledToolRegistry, ToolTransactionContext

# ─── Admission: copy-on-write slot ownership ────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_tasks_cannot_share_slot_state():
    """The historical shared-mutable-default bug: task B must not inherit
    task A's held-slot count, or B acquires without a permit while A holds
    the only one."""
    controller = InferenceAdmissionController(
        AdmissionConfig(max_concurrent_generations=1, max_queued_generations=4),
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        result = await controller.acquire("http://b", "m")
        assert result is AdmissionResult.ACQUIRED
        started.set()
        await release.wait()
        controller.release("http://b", "m")

    holder_task = asyncio.create_task(holder())
    await started.wait()

    # While the holder owns the single slot, another task must NOT believe
    # it already holds a slot (the old shared-default bug returned ACQUIRED
    # via reentrancy without consuming the semaphore).
    got = await controller.acquire("http://b", "m", timeout=0.05)
    assert got is not AdmissionResult.ACQUIRED
    release.set()
    await holder_task


@pytest.mark.asyncio
async def test_slot_counts_are_context_local():
    controller = InferenceAdmissionController()
    assert await controller.acquire("http://x", "m") is AdmissionResult.ACQUIRED
    # Child tasks DO inherit the count (P0-15 nested-generation semantics),
    # but a genuinely independent context starts from zero.
    import contextvars

    empty: contextvars.Context = contextvars.Context()
    key = controller._key("http://x", "m")
    seen = empty.run(lambda: controller._slots().get(key, 0))
    assert seen == 0


# ─── Per-route capacity tightening ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_route_capacity_can_only_tighten():
    controller = InferenceAdmissionController(
        AdmissionConfig(max_concurrent_generations=2),
    )
    await controller.set_route_capacity("http://r", "big", 1)
    state = controller._states[controller._key("http://r", "big")]
    assert state.capacity == 1, "route must tighten 2 -> 1"
    # Raising is refused — only the operator's global config may widen.
    await controller.set_route_capacity("http://r", "big", 8)
    assert state.capacity == 1


@pytest.mark.asyncio
async def test_tightened_route_cap_bounds_concurrency():
    """A DIFFERENT task must be rejected when the tightened cap is full.
    (Same-task re-acquisition is legitimate P0-15 reentrancy.)"""
    controller = InferenceAdmissionController(
        AdmissionConfig(max_concurrent_generations=4, max_queued_generations=4),
    )
    # Swap only happens on a fully idle route, matching the implementation.
    await controller.set_route_capacity("http://t", "m", 1)
    state = controller._states[controller._key("http://t", "m")]
    assert state.capacity == 1

    release = asyncio.Event()

    async def holder():
        await controller.acquire("http://t", "m")
        await release.wait()
        controller.release("http://t", "m")

    holder_task = asyncio.create_task(holder())
    await asyncio.sleep(0)  # let the holder take the slot
    second = await controller.acquire("http://t", "m", timeout=0.05)
    assert second is not AdmissionResult.ACQUIRED, "tightened cap of 1 must reject"
    release.set()
    await holder_task


# ─── release_now() at transport EOF ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_release_now_frees_the_slot_and_is_idempotent():
    controller = InferenceAdmissionController(
        AdmissionConfig(max_concurrent_generations=1),
    )
    async with controller.generation_slot("http://n", "m") as slot:
        assert slot.acquired
        slot.release_now()
        state = controller._states[controller._key("http://n", "m")]
        assert state.active == 0, "slot freed at transport EOF"
        # Idempotent — and the context-manager exit must not over-release.
        slot.release_now()
    # The slot is free for the next acquirer.
    assert await controller.acquire("http://n", "m", timeout=0.05) is AdmissionResult.ACQUIRED


@pytest.mark.asyncio
async def test_unacquired_slot_release_now_is_a_noop():
    """A slot that never got a permit must release_now() as a no-op — no
    ValueError from the BoundedSemaphore, no phantom release."""
    controller = InferenceAdmissionController(
        AdmissionConfig(max_concurrent_generations=1, max_queued_generations=0),
    )
    # Occupy the only slot so a fresh task cannot get a permit.
    started = asyncio.Event()
    release = asyncio.Event()

    async def holder():
        await controller.acquire("http://q", "m")
        started.set()
        await release.wait()
        controller.release("http://q", "m")

    holder_task = asyncio.create_task(holder())
    await started.wait()
    async with controller.generation_slot("http://q", "m", timeout=0.01) as slot:
        assert not slot.acquired
        slot.release_now()  # must not raise or release a permit we never held
    release.set()
    await holder_task


# ─── Schema validator caching ────────────────────────────────────────────────


def _schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "depth": {"type": "integer", "minimum": 0},
        },
        "required": ["path"],
    }


def test_validator_is_compiled_once_per_schema():
    _VALIDATOR_CACHE.clear()
    schema = _schema()
    first = _cached_validator(schema)
    assert first is not None
    assert _cached_validator(schema) is first, "same schema must reuse the validator"
    assert len(_VALIDATOR_CACHE) == 1
    validate_against_schema({"path": 1}, schema)
    validate_against_schema({"path": "/x", "depth": -1}, schema)
    assert len(_VALIDATOR_CACHE) == 1


def test_invalid_schema_is_negatively_cached():
    _VALIDATOR_CACHE.clear()
    bad = {"type": "not-a-type", "properties": "nope"}
    for _ in range(3):
        issues = validate_against_schema({"a": 1}, bad)
        assert issues and issues[0].keyword == "invalid_schema"
    assert len(_VALIDATOR_CACHE) == 1, "invalid schema must be negatively cached"


def test_cached_validation_matches_uncached_results():
    schema = _schema()
    _VALIDATOR_CACHE.clear()
    cold = validate_against_schema({"path": 5}, schema)
    _VALIDATOR_CACHE.clear()
    _cached_validator(schema)  # warm
    warm = validate_against_schema({"path": 5}, schema)
    assert [(i.path, i.keyword, i.message) for i in cold] == [
        (i.path, i.keyword, i.message) for i in warm
    ]


# ─── Repair loop: stop on no progress ────────────────────────────────────────


def test_unfixable_call_terminates_promptly():
    """A call no rule can fix must not spin through 20 validate+rule passes
    on identical state — the loop stops at the first no-progress iteration."""
    tool = CanonicalTool(
        name="t", description="",
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer"}},
            "required": ["n"],
        },
    )
    t0 = time.perf_counter()
    outcome = __import__(
        "agent_interop.repair.pipeline", fromlist=["repair_one"],
    ).repair_one(call_name="t", call_arguments={"n": "forever-string"}, tools=[tool])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert not outcome.is_accepted
    assert elapsed_ms < 100, f"no-progress loop did not stop ({elapsed_ms:.1f}ms)"


def test_fixable_calls_still_repair():
    from agent_interop.config import RepairPolicy, RepairTier
    from agent_interop.repair.pipeline import repair_one

    tool = CanonicalTool(
        name="w", description="",
        input_schema={
            "type": "object",
            "properties": {"n": {"type": "integer", "minimum": 0}},
            "required": ["n"],
        },
    )
    # String→int coercion lives in the COERCIVE tier; the default policy
    # only enables SYNTAX_ONLY + SAFE_SHAPE.
    policy = RepairPolicy(enabled_tiers=frozenset({RepairTier.COERCIVE}))
    outcome = repair_one(
        call_name="w", call_arguments={"n": "42"}, tools=[tool], policy=policy,
    )
    assert outcome.is_accepted, outcome.error
    assert outcome.accepted == {"n": 42}


# ─── CompiledToolRegistry ────────────────────────────────────────────────────


def test_compiled_registry_matches_per_candidate_map():
    tools = [
        CanonicalTool(name=f"tool_{i}", description="d", input_schema={"type": "object"})
        for i in range(10)
    ]
    registry = CompiledToolRegistry.compile(tools)
    for tool in tools:
        assert registry.get(tool.name) is tool
    assert registry.get("missing") is None
    # Identical to the historical per-candidate construction.
    assert registry.by_name == {t.name: t for t in tools}


def test_batch_uses_registry_with_identical_decisions():
    """A batch processed with a pre-compiled registry must produce the same
    decisions as the historical rebuild-per-candidate path."""
    from agent_interop.abi import RawToolCallCandidate
    from agent_interop.transaction import ToolBatchPolicy, process_tool_batch

    tools = [
        CanonicalTool(
            name="read_file", description="",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        ),
    ]
    candidates = [
        RawToolCallCandidate(id="a", name="read_file", raw_arguments={"path": "/x"}),
        RawToolCallCandidate(id="b", name="READ_FILE", raw_arguments={"path": "/y"}),
        RawToolCallCandidate(id="c", name="unknown_tool", raw_arguments={}),
    ]

    async def run(with_registry):
        context = ToolTransactionContext()
        if with_registry:
            from dataclasses import replace

            context = replace(context, registry=CompiledToolRegistry.compile(tools))
        return await process_tool_batch(
            candidates, tools, context=context, policy=ToolBatchPolicy.BEST_EFFORT,
        )

    with_reg = asyncio.run(run(True))
    without_reg = asyncio.run(run(False))
    assert [(d.outcome.status.value, d.outcome.call_name) for d in with_reg.decisions] == [
        (d.outcome.status.value, d.outcome.call_name) for d in without_reg.decisions
    ]
    assert [b.name for b in with_reg.accepted_blocks] == [b.name for b in without_reg.accepted_blocks]
