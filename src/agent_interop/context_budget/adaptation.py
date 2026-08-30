"""Capacity-adaptation cascade (P0-2/P0-6/P0-48).

Extracted from Gateway's ``_prepare_invocation_async``: the ordered ladder
run when planning says a request does not fit —

    1. deterministic safe tool-result virtualization (when the strategy
       permits it),
    2. deterministic history paging,
    3. controller summarization of old history (lossy, opt-in),
    4. otherwise ``ContextLimitExceededError``.

Every step that reshapes the request re-plans against the exact request
that will be rendered.  Planning alone is not capacity enforcement.

Coupling contract: no request-scoped state.  Re-planning resolves
``gateway._resolve_invocation_plan_and_key_async`` and controller
summarization resolves ``gateway._summarize_old_history_with_controller``
through the ``gateway`` back-reference on every call (tests replace these
on the instance after construction).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTextBlock

__all__ = ["AdaptationState", "run_context_adaptation"]


@dataclass
class AdaptationState:
    """Mutable carrier for the values the adaptation ladder rewrites.

    ``projected_request`` is the P0-4 projection — it starts equal to the
    authoritative request and diverges only through recorded
    transformations below.  The resolved-plan fields mirror the tuple
    ``_resolve_invocation_plan_and_key_async`` returns and are refreshed
    by every re-plan.
    """

    projected_request: CanonicalRequest
    cost_snapshot: Any
    adaptation: Any  # ContextAdaptationResult
    history_page_refs: tuple[str, ...] = ()
    backend_metadata: Any = None
    model_profile: Any = None
    repair_policy: Any = None
    plan: Any = None
    compat_key: Any = None
    runtime_capabilities: Any = None
    behavioral: Any = None
    compatibility_plan: Any = None
    context_plan: Any = None


async def run_context_adaptation(
    gateway: Any,
    state: AdaptationState,
    *,
    route: Any,
    context: Any,
    history_result: Any,
    streaming: bool,
    inspect_runtime: bool,
    allow_controller_summary: bool,
    execution: Any,
) -> None:
    """Run the capacity-adaptation ladder, mutating ``state`` in place.

    Raises ``ContextLimitExceededError`` when the request is still
    oversized after every strategy the route allows.
    """
    from agent_interop.context_budget import (
        ContextLimitExceededError,
        build_request_cost_snapshot,
        compact_safe_tool_results,
    )

    context_plan = state.context_plan
    # Planning alone is not capacity enforcement.  Apply the narrowly
    # safe, deterministic adaptation (only historical pageable tool
    # output), then re-plan against the exact request we will render.
    # Nothing else is discarded or summarized implicitly.
    #
    # P0-2/P0-11: Only virtualize when the context strategy permits
    # it. When virtualization is NOT permitted the adaptation is a
    # NO-OP — legacy lossy truncation must never run from the live
    # path (transparent mode's contract is "reduce nothing, reject
    # if it doesn't fit").
    if context_plan.allow_result_virtualization:
        state.adaptation = compact_safe_tool_results(
            state.projected_request,
            exchanges=history_result.exchanges,
            plan=context_plan,
            store=gateway._context_store,
            session_id=getattr(context, "session_id", "") or "",
        )
    if state.adaptation.changed:
        # P0-4: mutation targets the PROJECTION, never the
        # authoritative request.
        state.projected_request = state.adaptation.request
        execution.record_compatibility_event(
            "context_adaptation:" + ",".join(state.adaptation.transformations)
        )
        # P1-F: the projection diverged, so the snapshot must be
        # rebuilt against it — a stale snapshot would misprice the
        # edited history. This replaces the old cost, never adds a
        # pass on the unchanged-request path.
        state.cost_snapshot = build_request_cost_snapshot(state.projected_request)
        # Planning alone is not capacity enforcement. Re-plan against
        # the exact request we will render — whether or not the
        # adaptation changed anything, the oversized request may have
        # gained an adapted path (or must fail with the strategies we
        # actually attempted).
        await _replan(gateway, state, route=route, context=context,
                      streaming=streaming, inspect_runtime=inspect_runtime)
    # P0-6: deterministic history paging runs BEFORE any model-based
    # summary. A paging pass costs CPU and copies; a controller
    # summary costs an entire model generation and is lossy. Only if
    # the request is STILL oversized after (optional) paging does the
    # controller path get its chance.
    if state.context_plan.compaction_required and route.context.allow_history_compaction:
        from agent_interop.history.projector import (
            build_history_index_prompt,
            project_history,
        )

        history_paging = project_history(
            list(state.projected_request.messages),
            store=gateway._context_store,
            session_id=getattr(context, "session_id", "") or "",
            max_recent_turns=6,
        )
        if history_paging.refs:
            state.projected_request = replace(
                state.projected_request, messages=history_paging.messages,
            )
            # Mirrors the projector's own paging step: paged refs are
            # opaque tokens to the model until the index prompt tells
            # it how to page them back through __interop_read_result.
            index_prompt = build_history_index_prompt(history_paging.refs)
            if index_prompt:
                state.projected_request = replace(
                    state.projected_request,
                    system=[*state.projected_request.system, CanonicalTextBlock(text=index_prompt)],
                )
            state.history_page_refs = tuple(history_paging.stored_refs)
            execution.record_compatibility_event("context_adaptation:history_paged")
            state.cost_snapshot = build_request_cost_snapshot(state.projected_request)
            await _replan(gateway, state, route=route, context=context,
                          streaming=streaming, inspect_runtime=inspect_runtime)
    if state.context_plan.compaction_required and allow_controller_summary:
        summarized_request = await gateway._summarize_old_history_with_controller(
            route=route,
            request=state.projected_request,
            context=context,
            context_plan=state.context_plan,
            inspect_runtime=inspect_runtime,
            # P0-48: the summary is a real model generation on behalf
            # of THIS request — it shares the outer budget and is
            # tagged as a context-summary generation in telemetry
            # rather than vanishing into an unrecorded execution.
            execution=execution,
        )
        if summarized_request is not None:
            # P0-4: the summary reshapes the PROJECTION. The
            # authoritative request keeps the client's original
            # semantics for validation, replay, and audit.
            state.projected_request = summarized_request
            execution.record_compatibility_event("context_adaptation:controller_summary_old_history")
            state.cost_snapshot = build_request_cost_snapshot(state.projected_request)
            await _replan(gateway, state, route=route, context=context,
                          streaming=streaming, inspect_runtime=inspect_runtime)
    if state.context_plan.compaction_required:
        attempted = tuple(dict.fromkeys(
            (*state.context_plan.transformations, *state.adaptation.transformations)
        ))
        execution.record_compatibility_event("context_limit_exceeded")
        raise ContextLimitExceededError(state.context_plan, attempted)


async def _replan(
    gateway: Any,
    state: AdaptationState,
    *,
    route: Any,
    context: Any,
    streaming: bool,
    inspect_runtime: bool,
) -> None:
    """Re-plan against the exact (possibly reshaped) request to render."""
    (
        state.backend_metadata, state.model_profile, state.repair_policy,
        state.plan, state.compat_key, state.runtime_capabilities,
        state.behavioral, state.compatibility_plan, state.context_plan,
    ) = await gateway._resolve_invocation_plan_and_key_async(
        route, state.projected_request, context, streaming,
        inspect_runtime=inspect_runtime,
        cost_snapshot=state.cost_snapshot,
    )
