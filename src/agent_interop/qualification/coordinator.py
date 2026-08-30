"""Gateway-facing qualification cache and bootstrap-probe orchestration.

One coordinator owns the digest-keyed record cache, the per-key probe
locks, and the optional durable store.  Record currency (battery revision
+ served template digest) is enforced on every read, so stale evidence
never reaches planning.

Coupling contract: the coordinator holds no request-scoped state.  The
two hooks that need a live gateway — probe preparation and probe send —
are resolved lazily through the ``gateway`` back-reference on every call
(tests replace ``gateway._execute_bootstrap_probe`` and friends AFTER the
gateway is constructed).
"""

from __future__ import annotations

import asyncio
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    ToolChoiceMode,
)
from agent_interop.config import ToolMode
from agent_interop.execution import InteropRequestExecution

__all__ = ["QualificationCoordinator", "state_meets_controller_level"]


def state_meets_controller_level(state: Any, minimum_level: str) -> bool:
    """Compare a qualification state against a controller level (L1–L4).

    No AUTOMATIC_TOOL state exists: qualification only proves forced
    selection and sequential continuation, never automatic tool choice.
    L2 therefore requires the same evidence as L1.
    """
    from agent_interop.qualification import QualificationState

    required = {
        "L1": QualificationState.FORCED_TOOL,
        "L2": QualificationState.FORCED_TOOL,
        "L3": QualificationState.SEQUENTIAL_AGENT,
        "L4": QualificationState.ADVANCED_AGENT,
    }
    order = {
        QualificationState.UNKNOWN: 0,
        QualificationState.CHAT_ONLY: 1,
        QualificationState.FORCED_TOOL: 2,
        QualificationState.SEQUENTIAL_AGENT: 3,
        QualificationState.ADVANCED_AGENT: 4,
        QualificationState.DEGRADED: 0,
        QualificationState.PROBING: 0,
    }
    candidate = state if state is not None else QualificationState.UNKNOWN
    return order.get(candidate, 0) >= order.get(
        required.get(minimum_level, QualificationState.SEQUENTIAL_AGENT), 4
    )


class QualificationCoordinator:
    """Digest-keyed safe-fact cache plus demand-driven bootstrap probing."""

    def __init__(self, *, gateway: Any, store: Any | None = None) -> None:
        self._gateway = gateway
        self._records: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._store = store

    # ─── Cache identity ──────────────────────────────────────────────────

    @staticmethod
    def key(runtime: Any) -> str:
        # Tags are mutable aliases. Persisting their outcome could apply a
        # previous model's qualification after a tag is repointed.
        return getattr(runtime, "model_digest", "")

    @staticmethod
    def template_digest(invocation: Any) -> str:
        """Digest of the served chat template so cached qualification is
        invalidated when the template (not just the weights) changes."""
        runtime = invocation.runtime_capabilities
        return str(getattr(runtime, "chat_template_digest", "") or "")

    # ─── Record cache ────────────────────────────────────────────────────

    def record(self, record: Any) -> None:
        """Store bounded bootstrap results for future safe planning decisions."""
        key = getattr(record, "model_digest", "")
        if key:
            self._records[key] = record
            if self._store is not None:
                self._store.put(record)

    def record_for_runtime(self, runtime: Any) -> Any | None:
        """Restore only digest-keyed safe facts from the durable cache.

        P0-52: a record is returned only if it was produced by the CURRENT
        battery revision against the CURRENTLY served chat template.  Stale
        evidence proves nothing about the model being served now, so it is
        ignored and dropped from the in-memory cache (the durable copy stays
        until overwritten — it is inert once currency fails).
        """
        key = self.key(runtime)
        # Legacy in-memory callers may explicitly seed a qualification before
        # runtime probing is enabled.  That ephemeral compatibility path is
        # never persisted and therefore cannot survive a mutable tag change.
        if not key and self._store is None:
            key = getattr(runtime, "model_name", "")
        if not key:
            return None
        record = self._records.get(key)
        if record is None and self._store is not None:
            record = self._store.get(key)
            if record is not None:
                self._records[key] = record
        if record is None:
            return None
        from agent_interop.qualification.revision import QUALIFICATION_BATTERY_REVISION
        from agent_interop.qualification.store import record_is_current

        if not record_is_current(
            record,
            battery_revision=QUALIFICATION_BATTERY_REVISION,
            template_digest=getattr(runtime, "chat_template_digest", "") or "",
        ):
            self._records.pop(key, None)
            return None
        return record

    # ─── Planning projection ─────────────────────────────────────────────

    def behavioral_capabilities(self, runtime: Any) -> Any:
        """Return only behavior proven by the synthetic bootstrap battery."""
        from agent_interop.planning import BehavioralCapabilities
        from agent_interop.qualification import QualificationState
        from agent_interop.qualification.state import probe_passed

        record = self.record_for_runtime(runtime)
        if record is None:
            return BehavioralCapabilities()
        # P0-51: ProbeOutcome.FAILED/.UNKNOWN are nonempty strings — truthiness
        # on the enum coerced every probed outcome (even FAILED) to True.
        # Only an explicit PASSED proves capability.
        native = probe_passed(record, "native_forced_tool")
        prompted = probe_passed(record, "prompted_forced_tool")
        continuation = probe_passed(record, "continuation")
        forced = bool(native or prompted)
        sequential = bool(continuation and forced)
        return BehavioralCapabilities(
            # Qualification does not test automatic selection or parallelism.
            native_tools=native,
            prompted_tools=prompted,
            forced_selection=forced,
            sequential_tool_use=sequential,
            tool_result_continuation=sequential,
            streaming=(getattr(record, "state", None) == QualificationState.ADVANCED_AGENT),
            chat_only=(getattr(record, "state", None) == QualificationState.CHAT_ONLY),
            sample_count=1,
        )

    # ─── Demand-driven bootstrap probing ─────────────────────────────────

    def required_probes_for_request(
        self,
        invocation: Any,
        record: Any | None,
    ) -> tuple[str, ...]:
        """Probes this request still needs, given the current record.

        P0-52: currency is enforced inside ``record_for_runtime``
        itself, so a stale record arrives here as None.  P0-53: a partial
        record contributes whatever evidence it has; only genuinely missing
        probes are returned.
        """
        runtime = invocation.runtime_capabilities
        key = self.key(runtime)
        if not key:
            return ()
        if record is None:
            return ("native_forced_tool", "tool_result_continuation")
        need_continuation = any(
            getattr(block, "tool_call_id", "")
            for message in invocation.reconciled_request.messages
            if getattr(message, "role", "") == "tool"
            for block in (message.content or ())
        )
        return record.staged_needed_probes(
            want_native=True, need_continuation=need_continuation,
        )

    async def ensure_bootstrap(self, invocation: Any) -> bool:
        """Qualify an unknown tool model with side-effect-free synthetic calls.

        P0-2: only an explicitly-configured ``blocking`` bootstrap may delay a
        real request behind probes; ``on_demand`` (the default) runs probes
        only when a request needs evidence that is genuinely missing — never
        as a first-request tax.  P0-53: a PARTIAL record no longer short-
        circuits; only probes this request actually demands run.
        """
        from agent_interop.qualification import BootstrapQualifier

        qualification = invocation.route.qualification
        choice = invocation.reconciled_request.tool_choice
        if (
            qualification.bootstrap not in ("blocking", "blocking_for_tool_requests")
            or not invocation.reconciled_request.tools
            or choice.mode not in (ToolChoiceMode.NAMED, ToolChoiceMode.REQUIRED)
        ):
            return False
        key = self.key(invocation.runtime_capabilities)
        if not key or not qualification.cache_by_digest:
            return False
        record = self.record_for_runtime(invocation.runtime_capabilities)
        needed = self.required_probes_for_request(invocation, record)
        if not needed:
            return False
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            record = self.record_for_runtime(invocation.runtime_capabilities)
            needed = self.required_probes_for_request(invocation, record)
            if not needed:
                return False

            async def execute(probe: Any) -> bool:
                # Late-bound: tests replace gateway._execute_bootstrap_probe
                # after construction, so resolve it on the instance per call.
                return await self._gateway._execute_bootstrap_probe(invocation, probe)

            # Review #30/#31: staged demand-driven qualification — run only
            # the probes this request actually needs.  A native pass ends
            # probing (no prompted/no_tool turns); exact_text is not a
            # separate probe, the forced probes carry expected_text.
            # Continuation evidence is probed only when THIS request actually
            # carries tool results (the demand for continuation), not eagerly.
            # P0-53: `existing` is the possibly-PARTIAL current record, so a
            # request that needs evidence the record lacks acquires exactly
            # that evidence instead of silently passing on stale coverage.
            existing = self.record_for_runtime(invocation.runtime_capabilities)
            need_continuation = any(
                getattr(block, "tool_call_id", "")
                for message in invocation.reconciled_request.messages
                if getattr(message, "role", "") == "tool"
                for block in (message.content or ())
            )
            record = await BootstrapQualifier().qualify_demand(
                key,
                execute,
                existing=existing,
                want_native=True,
                need_continuation=need_continuation,
                template_digest=self.template_digest(invocation),
            )
            self.record(record)
            # True only when this call actually generated new evidence — a
            # no-op pass-through must not force the caller to re-prepare.
            return bool(needed)

    async def execute_bootstrap_probe(self, invocation: Any, probe: Any) -> bool:
        """Execute one synthetic qualification probe without client authority.

        Review #22: the probe's ``presentation`` is honored by building an
        ISOLATED route override (``replace(route, tool_mode=...)``) passed
        through explicit resolution — the gateway's ``config.routes`` is
        never mutated, so a probe can never leave the gateway serving in a
        probe's mode.
        Review #22: pass/fail semantics come from the probe contract itself
        (``probe.expected_text``), not name-based if-chains in this module.
        """
        from dataclasses import replace as _replace

        from agent_interop.qualification.probes import SYNTHETIC_TOOL

        tools = [SYNTHETIC_TOOL] if probe.requires_tools else []
        choice = CanonicalToolChoice.required() if probe.requires_tools else CanonicalToolChoice.none()
        messages = [CanonicalMessage(role="user", content=[CanonicalTextBlock(text=probe.prompt)])]
        if probe.name == "tool_result_continuation":
            messages = [
                CanonicalMessage(role="user", content=[CanonicalTextBlock(text="Call the probe tool.")]),
                CanonicalMessage(
                    role="assistant",
                    content=[CanonicalToolCallBlock(id="interop_probe_call", name="interop_probe", arguments={"marker": "done"})],
                ),
                CanonicalMessage(
                    role="tool",
                    content=[CanonicalToolResultBlock(tool_call_id="interop_probe_call", content="marker=done")],
                ),
                CanonicalMessage(role="user", content=[CanonicalTextBlock(text=probe.prompt)]),
            ]
        request = CanonicalRequest(
            model=invocation.reconciled_request.model,
            messages=messages,
            tools=tools,
            tool_choice=choice,
        )

        execution = InteropRequestExecution(context=invocation.request_context)
        route_override = None
        if probe.presentation is not None:
            # Review #22: isolated probe route forces the requested
            # presentation WITHOUT touching the gateway config.  The override
            # lives only for this probe's preparation; controller fallback is
            # disabled so the probe measures the forced mode itself, not a
            # delegated controller turn.
            route_override = _replace(
                invocation.route,
                tool_mode=probe.presentation,
                compatibility=_replace(
                    invocation.route.compatibility,
                    mode="direct",
                    allow_controlled=False,
                ),
            )
        probe_invocation = await self._gateway._prepare_invocation_async(
            request,
            invocation.request_context,
            streaming=False,
            execution=execution,
            route_override=route_override,
        )
        response = await self._gateway._handle_request_send(probe_invocation, execution)
        if response.error is not None:
            return False
        calls = [block for block in response.content if isinstance(block, CanonicalToolCallBlock)]

        # P0-55: exact probe assertions.  A forced-tool probe passes only
        # when the model called interop_probe EXACTLY ONCE with the requested
        # marker — one call with wrong arguments, or two calls with the right
        # one, both prove sloppy surface compliance, not capability.
        if probe.expected_marker:
            if len(calls) != 1:
                return False
            call = calls[0]
            if call.name != SYNTHETIC_TOOL.name:
                return False
            if getattr(call, "arguments", None) != {"marker": probe.expected_marker}:
                return False
            # Presentation probes additionally verify the forced mode actually
            # took effect — otherwise a probe would "pass" because the model
            # called the tool in some other mode entirely.
            if probe.name == "native_forced_tool":
                return probe_invocation.invocation_plan.effective_tool_mode == ToolMode.NATIVE
            if probe.name == "prompted_forced_tool":
                return probe_invocation.invocation_plan.effective_tool_mode in (
                    ToolMode.PROMPTED, ToolMode.TEXTUAL,
                )
            return True

        if probe.expected_text:
            return any(
                probe.expected_text in block.text
                for block in response.content
                if isinstance(block, CanonicalTextBlock)
            ) and not calls
        # Contract probes without an expected_text assert behavioral
        # properties: no_tool / tool_result_continuation must produce NO new
        # tool call (the model answers from the transcript, not the surface).
        return not calls
