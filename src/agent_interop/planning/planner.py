"""Request compatibility planner.

The planner intersects client contract, codec transport capability, model
runtime inspection, and observed behavior.  No single source can promote a
model to direct tool mode on its own.
"""

from __future__ import annotations

from agent_interop.config import ToolMode
from agent_interop.context_budget import ContextBudgetPlanner, effective_context_limit
from agent_interop.context_budget.estimator import estimate_request_context
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


class RequestCompatibilityPlanner:
    revision = "1"

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
    ) -> CompatibilityPlan:
        token_estimate = estimate_request_context(request).total_required_tokens
        from agent_interop.context_budget.types import TokenEstimate
        requirements = derive_request_requirements(request, context, client_requirements, TokenEstimate(token_estimate))
        tool_surface = ToolSurfacePlanner().plan(request, route.tool_surface)
        runtime_limit = effective_context_limit(
            runtime_capabilities.architecture_context_tokens,
            runtime_capabilities.configured_context_tokens,
            route.context.context_limit_tokens,
            runtime_capabilities.effective_context_tokens,
        )
        context_plan = ContextBudgetPlanner().plan(
            request,
            runtime_limit_tokens=runtime_limit,
            output_reserve_tokens=route.context.output_reserve_tokens,
            visible_tools=tool_surface.visible_tools,
            original_tools=request.tools,
        )
        missing = list(missing_behavioral_capabilities(requirements, behavioral_capabilities))
        # An operator's explicit native mode is a transport instruction, not
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
        allow = route.compatibility
        controller_attempt = (
            CompatibilityAttempt(
                AttemptKind.CONTROLLER_MEDIATED,
                route.tool_mode,
                use_controller=True,
                reason="fallback_after_direct_or_adapted_attempts",
            )
            if (
                allow.allow_controlled
                and route.controller is not None
                and route.controller.enabled
                and (route.controller.route_id or route.controller.auto_select_route)
            )
            else None
        )
        attempts: tuple[CompatibilityAttempt, ...]
        if direct and (route.tool_mode == ToolMode.NATIVE or not missing) and context_plan.fits_directly and allow.allow_direct:
            attempts = (*direct, *adapted, *((controller_attempt,) if controller_attempt else ()))
            path = CompatibilityPath.DIRECT
        elif adapted and context_plan.fits_directly and allow.allow_adapted:
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
        return CompatibilityPlan(
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
