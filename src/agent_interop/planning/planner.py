"""Request compatibility planner.

The planner intersects client contract, codec transport capability, model
runtime inspection, and observed behavior.  No single source can promote a
model to direct tool mode on its own.
"""

from __future__ import annotations

import threading
from enum import Enum
from typing import Any

from agent_interop.config import ToolMode, ToolSurfaceConfig, ToolSurfaceMode
from agent_interop.context_budget import ContextBudgetPlanner, effective_context_limit
from agent_interop.context_budget.estimator import (
    build_request_cost_snapshot,
    estimate_request_context,
)
from agent_interop.context_budget.types import ContextPlan
from agent_interop.planning.attempts import adapted_attempts, direct_attempts
from agent_interop.planning.decisions import missing_behavioral_capabilities
from agent_interop.planning.requirements import derive_request_requirements
from agent_interop.planning.types import (
    AttemptKind,
    BehavioralCapabilities,
    CompatibilityAttempt,
    CompatibilityPath,
    CompatibilityPlan,
)
from agent_interop.tool_surface import ToolSurfacePlanner


class ContextStrategy(str, Enum):
    """Context-management strategy for a route.

    - TRANSPARENT: No ModelView reduction; reject if it doesn't fit.
    - ADAPTIVE (default): Lossless tool/history virtualization and bounded
      schema projection.
    - STRICT_BOUNDED: Enforce hard context/resource ceiling; no semantic
      summarization unless separately opted in.
    """

    TRANSPARENT = "transparent"
    ADAPTIVE = "adaptive"
    STRICT_BOUNDED = "strict_bounded"


def _resolve_context_strategy(strategy: str | ContextStrategy) -> ContextStrategy:
    """Resolve a raw config string (or enum) to a ContextStrategy.

    Any unrecognized value defaults to ``ContextStrategy.ADAPTIVE``.
    """
    if isinstance(strategy, ContextStrategy):
        return strategy
    normalized = str(strategy).strip().lower()
    if normalized in {"auto", ""}:
        return ContextStrategy.ADAPTIVE
    for member in ContextStrategy:
        if member.value == normalized:
            return member
    return ContextStrategy.ADAPTIVE


def _strategy_defaults(strategy: ContextStrategy) -> dict[str, bool]:
    """Return the default ``allow_*`` flags for a given strategy.

    ``ContextConfig`` may override any of these individually; this only
    supplies the baseline when the operator has not set them explicitly.
    """
    if strategy is ContextStrategy.TRANSPARENT:
        return {
            "allow_tool_reduction": False,
            "allow_result_virtualization": False,
            "allow_history_compaction": False,
            "allow_semantic_summary": False,
            "allow_controller_decomposition": False,
        }
    if strategy is ContextStrategy.STRICT_BOUNDED:
        return {
            "allow_tool_reduction": True,
            "allow_result_virtualization": True,
            "allow_history_compaction": False,
            "allow_semantic_summary": False,
            "allow_controller_decomposition": False,
        }
    # ADAPTIVE
    return {
        "allow_tool_reduction": True,
        "allow_result_virtualization": True,
        "allow_history_compaction": True,
        "allow_semantic_summary": True,
        "allow_controller_decomposition": True,
    }


# P1.8: planner cache lock (per-instance; the planner cache itself lives
# on each RequestCompatibilityPlanner instance).
_PLANNER_CACHE_LOCK = threading.Lock()


class RequestCompatibilityPlanner:
    revision = "1"

    def __init__(self) -> None:
        # P1.8 (review item 15): planner instance cache keyed by the FULL
        # serving tuple — fingerprint of the tool surface, the route id,
        # the behavioral state, and the cost_snapshot fingerprint. Each
        # entry is invalidated when its inputs change (different request
        # shape, different evidence, different surface). Bounded LRU so a
        # burst of distinct requests cannot blow memory; the bound is high
        # enough that warm clients with stable surfaces always hit.
        self._cache: dict[tuple, CompatibilityPlan] = {}
        self._cache_order: list[tuple] = []
        self._cache_max = 256

    def _cache_key(
        self,
        *,
        route,
        requirements,
        codec_capabilities: Any,
        behavioral_capabilities: Any,
        runtime_capabilities: Any = None,
    ) -> tuple | None:
        """Build a cache key from the inputs that actually influence the plan.

        P0-planner-cache: the key is anchored on the NORMALIZED requirement
        vector (``RequestRequirements`` is frozen and hashable) rather than
        an ad-hoc list of source fields. ``derive_request_requirements``
        consumes tool_choice mode/named tool, streaming, requested
        capabilities, tool-result history, and the client profile — keying
        on its OUTPUT means any request whose requirements differ gets a
        different key by construction, and a new source field added to
        derivation is automatically covered instead of silently missing
        from a hand-maintained list.

        Everything else in the key is the plan's remaining direct input:
        route policy (tool mode, surface config, context config,
        compatibility policy, controller config), codec capabilities,
        runtime capacity, and the behavioral-evidence tuple.

        Returns None when a key cannot be built — the caller MUST NOT
        cache in that case.
        """
        try:
            # Route policy fingerprint: every route field the plan reads.
            # Local model names/aliases are content — include them so a
            # repointed route cannot inherit the old target's plan.
            route_key = (
                getattr(route, "id", "") or "",
                str(getattr(route, "upstream_model", "") or ""),
                str(getattr(getattr(route, "tool_mode", None), "value", getattr(route, "tool_mode", ""))),
                repr(getattr(route, "tool_surface", None)),
                repr(getattr(route, "context", None)),
                repr(getattr(route, "compatibility", None)),
                repr(getattr(route, "controller", None)),
            )
            # P0-audit: runtime capacity facts feed runtime_limit directly.
            # A replan (static capabilities) and a live request (inspected
            # capabilities) for the same route must not share an entry.
            runtime_key = (
                getattr(runtime_capabilities, "configured_context_tokens", 0) or 0,
                getattr(runtime_capabilities, "effective_context_tokens", 0) or 0,
                getattr(runtime_capabilities, "architecture_context_tokens", 0) or 0,
            )
            # Behavioral capability tuples: every Boolean and the sample
            # count directly determine the planner's path selection, so
            # two distinct shapes MUST NOT share an entry. Use a tuple of
            # concrete field values rather than just `state` — the bare
            # state value doesn't differentiate UNKNOWN from a fully
            # populated BehavioralCapabilities with the same effective
            # state, which would conflate very different planner inputs.
            behavioral_key = (
                getattr(behavioral_capabilities, "native_tools", None),
                getattr(behavioral_capabilities, "prompted_tools", None),
                getattr(behavioral_capabilities, "forced_selection", None),
                getattr(behavioral_capabilities, "automatic_selection", None),
                getattr(behavioral_capabilities, "sequential_tool_use", None),
                getattr(behavioral_capabilities, "parallel_tool_use", None),
                getattr(behavioral_capabilities, "tool_result_continuation", None),
                getattr(behavioral_capabilities, "streaming", None),
                getattr(behavioral_capabilities, "chat_only", None),
                getattr(behavioral_capabilities, "sample_count", None),
                str(
                    getattr(
                        getattr(behavioral_capabilities, "state", None),
                        "value",
                        "",
                    )
                ),
            )
            return (
                route_key,
                requirements,
                repr(codec_capabilities),
                runtime_key,
                behavioral_key,
            )
        except Exception:
            return None

    async def plan(
        self,
        *,
        request,
        context,
        route,
        client_requirements,
        codec_capabilities,
        runtime_capabilities,
        behavioral_capabilities: BehavioralCapabilities,
        unknown_capacity_policy: str = "reject",
        unknown_capacity_fallback_tokens: int = 8192,
        cost_snapshot: Any | None = None,
    ) -> CompatibilityPlan:
        from agent_interop.context_budget.meter import compute_output_reserve
        from agent_interop.context_budget.types import TokenEstimate

        # P1-F (review item 29): serialize the request ONCE. The snapshot
        # feeds every cost consumer in this pass — requirements derivation,
        # tool-surface pricing, and both context-planner estimates — instead
        # of each re-serializing the system prompt, full history, and tool
        # schemas (a multi-megabyte json.dumps each on large conversations).
        # P1.7 (review #17): trust the caller's snapshot when supplied —
        # the gateway already serialized the request before this planner
        # call.  Discarding the caller snapshot and re-serializing would
        # double the cost and could also misprice (a snapshot taken from
        # the original request is the truth for that request's cost).
        if cost_snapshot is None:
            cost_snapshot = build_request_cost_snapshot(request)
        # P0-planner-cache: requirements are derived BEFORE the cache
        # lookup and the normalized requirement vector anchors the key.
        # The historical form keyed on source fields (tool fingerprint +
        # message size) and could hand a plan computed for one request
        # contract to a different one — an AUTO turn and a REQUIRED turn
        # with the same tools collided, inheriting the wrong attempts and
        # context decisions. Derive-then-key makes cross-request plan
        # contamination structurally impossible: any request whose
        # requirements differ gets a different key by construction.
        token_estimate = estimate_request_context(request, snapshot=cost_snapshot).total_required_tokens
        requirements = derive_request_requirements(
            request, context, client_requirements, TokenEstimate(token_estimate),
            cost_snapshot=cost_snapshot,
        )
        cache_key = self._cache_key(
            route=route,
            requirements=requirements,
            codec_capabilities=codec_capabilities,
            behavioral_capabilities=behavioral_capabilities,
            runtime_capabilities=runtime_capabilities,
        )
        if cache_key is not None:
            with _PLANNER_CACHE_LOCK:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache_order.remove(cache_key)
                    self._cache_order.append(cache_key)
                    return cached
        tool_surface = ToolSurfacePlanner().plan(request, route.tool_surface, cost_snapshot=cost_snapshot)
        # P0-49: architecture_context_tokens is a CEILING on what could be
        # allocated, never itself evidence of allocation.  Usable capacity
        # comes from observed serving / configured serving / operator route
        # cap in that order, then clamps to the architecture ceiling.
        runtime_limit = effective_context_limit(
            configured_limit=runtime_capabilities.configured_context_tokens,
            route_override=route.context.context_limit_tokens,
            observed_effective_limit=runtime_capabilities.effective_context_tokens,
            architecture_ceiling=runtime_capabilities.architecture_context_tokens,
        )
        # P0.19 (review #13/#14): operator policy for unknown capacity. The
        # historical behavior — silently guessing 8K — stays only as the
        # explicit "fallback" policy; "reject" fails a tool-bearing request
        # whose capacity cannot be established.
        if runtime_limit <= 0:
            from agent_interop.context_budget.planner import ContextCapacityUnknownError

            requirements_has_tools = bool(getattr(requirements, "tools_present", False))
            # P0-50: "reject_tool_requests" is the explicit alias for
            # "reject" — both reject TOOL-BEARING requests on unknown
            # capacity; chat-only requests never carried tool-surface
            # overflow risk and proceed on the conservative flagged default.
            rejects_tools = unknown_capacity_policy in ("reject", "reject_tool_requests")
            if unknown_capacity_policy == "fallback":
                runtime_limit = max(1, int(unknown_capacity_fallback_tokens))
            elif rejects_tools and requirements_has_tools:
                raise ContextCapacityUnknownError.unresolvable(
                    model_name=getattr(runtime_capabilities, "model_name", ""),
                    policy_hint=(
                        "Configure resources.unknown_capacity_policy=fallback "
                        "to allow a conservative default."
                    ),
                )
            # reject + chat-only request: defer to the per-strategy 8K
            # conservative default below, with capacity_unknown flagged.
        # P0.11: context-aware output reserve
        output_reserve = compute_output_reserve(
            request,
            runtime_limit,
            mode="auto",
        )
        # If route explicitly sets a reserve, use the smaller of the two
        if route.context.output_reserve_tokens > 0:
            output_reserve = min(output_reserve, route.context.output_reserve_tokens)

        # P1.2: wire ContextConfig policy gates — resolve strategy + flags
        strategy = _resolve_context_strategy(route.context.strategy)
        defaults = _strategy_defaults(strategy)
        allow_tool_reduction = route.context.allow_tool_reduction and defaults["allow_tool_reduction"]
        allow_history_compaction = route.context.allow_history_compaction and defaults["allow_history_compaction"]
        allow_controller_decomposition = route.context.allow_controller_decomposition and defaults["allow_controller_decomposition"]

        missing = list(missing_behavioral_capabilities(requirements, behavioral_capabilities))

        # TRANSPARENT: No ModelView reduction; reject if it doesn't fit.
        # P0.24: Use TRANSPARENT tool surface (no dynamic reduction)
        if strategy is ContextStrategy.TRANSPARENT:
            transparent_surface = ToolSurfacePlanner().plan(
                request, ToolSurfaceConfig(mode=ToolSurfaceMode.TRANSPARENT),
                cost_snapshot=cost_snapshot,
            )
            before = estimate_request_context(
                request,
                visible_tools=transparent_surface.visible_tools,
                output_reserve_tokens=output_reserve,
                snapshot=cost_snapshot,
            )
            safe_limit = int(runtime_limit * 0.90) if runtime_limit else 0
            if safe_limit == 0:
                safe_limit = 8192
                capacity_unknown = True
            else:
                capacity_unknown = False
            fits = before.total_required_tokens <= safe_limit
            context_plan = ContextPlan(
                runtime_limit_tokens=runtime_limit,
                safe_limit_tokens=safe_limit,
                before=before,
                after=before,
                fits_directly=fits,
                compaction_required=not fits,
                selected_strategy=strategy.value,
                preserved_message_indices=tuple(range(len(request.messages))),
                compacted_message_indices=() if fits else tuple(range(len(request.messages))),
                transformations=() if fits else ("context_exceeds_limit",),
                capacity_unknown=capacity_unknown,
                # TRANSPARENT: preserve everything
                allow_tool_reduction=False,
                allow_result_virtualization=False,
                allow_history_paging=False,
                allow_semantic_summary=False,
                allow_controller_decomposition=False,
            )
            if not fits:
                return CompatibilityPlan(
                    path=CompatibilityPath.UNAVAILABLE,
                    requirements=requirements,
                    attempts=(),
                    context_plan=context_plan,
                    tool_surface_plan=transparent_surface,
                    missing_capabilities=tuple(missing),
                    transformations=("context_exceeds_limit",),
                    warnings=("context_adaptation_required",),
                    planner_revision=self.revision,
                )
            tool_surface = transparent_surface
        else:
            # ADAPTIVE or STRICT_BOUNDED: use ContextBudgetPlanner
            context_plan = ContextBudgetPlanner().plan(
                request,
                runtime_limit_tokens=runtime_limit,
                output_reserve_tokens=output_reserve,
                visible_tools=tool_surface.visible_tools,
                original_tools=request.tools,
                cost_snapshot=cost_snapshot,
            )
            # STRICT_BOUNDED: enforce hard ceiling; no semantic summarization
            if strategy is ContextStrategy.STRICT_BOUNDED:
                context_plan = ContextPlan(
                    runtime_limit_tokens=context_plan.runtime_limit_tokens,
                    safe_limit_tokens=context_plan.safe_limit_tokens,
                    before=context_plan.before,
                    after=context_plan.after,
                    fits_directly=context_plan.fits_directly,
                    compaction_required=context_plan.compaction_required,
                    selected_strategy=strategy.value,
                    preserved_message_indices=context_plan.preserved_message_indices,
                    compacted_message_indices=context_plan.compacted_message_indices,
                    transformations=tuple(
                        t for t in context_plan.transformations
                        if t not in {"summarize_old_history_in_controlled_mode", "delegate_through_controller"}
                    ),
                    capacity_unknown=context_plan.capacity_unknown,
                )

        # P0.23: wire strategy flags into ContextPlan
        context_plan = ContextPlan(
            **{
                **context_plan.__dict__,
                "allow_tool_reduction": allow_tool_reduction,
                "allow_result_virtualization": defaults["allow_result_virtualization"],
                "allow_history_paging": allow_history_compaction,
                "allow_semantic_summary": allow_controller_decomposition and defaults["allow_semantic_summary"],
                "allow_controller_decomposition": allow_controller_decomposition,
            }
        )
        # an evidence-derived promotion.  Preserve the existing contract: it
        # exercises the backend's native tool-array validation even before a
        # model has enough evidence to be selected automatically.
        if requirements.tools_present and route.tool_mode == ToolMode.NATIVE:
            direct = (CompatibilityAttempt(
                AttemptKind.NATIVE_TOOLS, ToolMode.NATIVE,
                reason="operator_forced_native_tools",
            ),)
        else:
            direct = direct_attempts(requirements, codec_capabilities, runtime_capabilities, behavioral_capabilities)
        adapted = adapted_attempts(requirements, None, runtime_capabilities, behavioral_capabilities)
        # P0.39: Use resolve_mode() as the single authority for mode→flags
        # mapping. When mode is explicit (not auto), resolve_mode() returns
        # the correct flags. When mode is auto, we still respect the
        # operator's allow_* overrides.
        allow = route.compatibility
        if allow.mode == "auto":
            # auto: respect operator's allow_* overrides
            allow_direct = allow.allow_direct
            allow_adapted = allow.allow_adapted
            allow_controlled = allow.allow_controlled
        else:
            resolved = allow.resolve_mode()
            allow_direct = resolved["allow_direct"]
            allow_adapted = resolved["allow_adapted"]
            allow_controlled = resolved["allow_controlled"]
        controller_attempt = (
            CompatibilityAttempt(
                AttemptKind.CONTROLLER_MEDIATED,
                route.tool_mode,
                use_controller=True,
                reason="fallback_after_direct_or_adapted_attempts",
            )
            if (
                allow_controlled
                and route.controller is not None
                and route.controller.enabled
                and (route.controller.route_id or route.controller.auto_select_route)
            )
            else None
        )
        attempts: tuple[CompatibilityAttempt, ...]
        if direct and (route.tool_mode == ToolMode.NATIVE or not missing) and context_plan.fits_directly and allow_direct:
            attempts = (*direct, *adapted, *((controller_attempt,) if controller_attempt else ()))
            path = CompatibilityPath.DIRECT
        elif adapted and context_plan.fits_directly and allow_adapted:
            attempts = (*adapted, *((controller_attempt,) if controller_attempt else ()))
            path = CompatibilityPath.ADAPTED
        elif controller_attempt is not None:
            attempts = (controller_attempt,)
            path = CompatibilityPath.CONTROLLED
        else:
            attempts = ()
            path = CompatibilityPath.UNAVAILABLE
        transformations = list(context_plan.transformations)
        if tool_surface.withheld_tool_names:
            transformations.append("reduced_tool_surface")
        if path == CompatibilityPath.ADAPTED:
            transformations.append("adapted_tool_protocol")
        if path == CompatibilityPath.CONTROLLED:
            transformations.append("compatibility_controller")
        plan_result = CompatibilityPlan(
            path=path,
            requirements=requirements,
            attempts=attempts,
            context_plan=context_plan,
            tool_surface_plan=tool_surface,
            missing_capabilities=tuple(missing),
            transformations=tuple(transformations),
            warnings=("context_adaptation_required",) if context_plan.compaction_required else (),
            planner_revision=self.revision,
        )
        # P0-planner-cache: store under the SAME key the lookup used — the
        # requirement vector was already derived above, so re-deriving here
        # would both waste the pass and risk a divergent key.
        cache_key = self._cache_key(
            route=route,
            requirements=requirements,
            codec_capabilities=codec_capabilities,
            behavioral_capabilities=behavioral_capabilities,
            runtime_capabilities=runtime_capabilities,
        )
        if cache_key is not None:
            with _PLANNER_CACHE_LOCK:
                cached = self._cache.get(cache_key)
                if cached is not None:
                    self._cache_order.remove(cache_key)
                self._cache[cache_key] = plan_result
                self._cache_order.append(cache_key)
                while len(self._cache_order) > self._cache_max:
                    old_key = self._cache_order.pop(0)
                    self._cache.pop(old_key, None)
        return plan_result