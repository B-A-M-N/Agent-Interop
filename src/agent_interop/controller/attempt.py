"""The controlled (controller-mediated) attempt execution (extracted from Gateway).

One class owns the bounded primary↔controller loop: controller route
selection and qualification, session turn-budget enforcement, work-product
extraction and bounded selection, the controller decision cycle, tool-choice
contract enforcement, provenance labelling, and controller-session state
write-back.

Coupling contract (mirrors GenerationSeam / PrivateContinuationLoop /
StreamEngine): the gateway's machinery — request sending, invocation
preparation, runtime inspection, invocation rebuild — arrives as
constructor callables; this module owns the LOOP POLICY. Holds no
request-scoped state; the controller session ledger is injected (one per
Gateway).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent_interop.abi import (
    CanonicalError,
    CanonicalModelReference,
    CanonicalResponse,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    ToolChoiceMode,
)
from agent_interop.errors import InteropErrorCode
from agent_interop.repair.invocation import ToolMode

__all__ = ["ControllerAttemptExecutor", "controller_invocation_with_delegate_tool"]


def controller_invocation_with_delegate_tool(
    invocation: Any,
    rebuild: Any,
) -> Any:
    """Expose the private delegation tool only on a controller request.

    Tool-surface planning deliberately knows only about client-declared
    tools.  The controller's refinement request is an Interop control
    message, so add it after normal preparation and rebuild the exact
    invocation plan used for rendering/validation.  It never alters the
    outer client request or its validation registry.
    """
    from agent_interop.config import ToolSurfaceConfig, ToolSurfaceMode
    from agent_interop.controller.policy import controller_delegate_tool
    from agent_interop.repair.invocation import build_invocation_plan
    from agent_interop.tool_surface import ToolSurfacePlanner

    delegate_tool = controller_delegate_tool()
    tools = tuple(invocation.reconciled_request.tools)
    if not any(tool.name == delegate_tool.name for tool in tools):
        tools = (*tools, delegate_tool)
    canonical = replace(invocation.reconciled_request, tools=list(tools))
    original_plan = invocation.invocation_plan
    plan = build_invocation_plan(
        tools=None,
        tool_choice=canonical.tool_choice,
        route_mode=original_plan.effective_tool_mode,
        model_profile=invocation.model_profile,
        repair_policy=invocation.repair_policy,
        codec_capabilities=getattr(original_plan, "codec_capabilities", None),
        upstream_tools=tools,
        validation_tools=tools,
        capabilities=invocation.private_capabilities,
    )
    # Review #21: atomic rebuild — model_request/model_view/visible tools
    # are regenerated together with the plan, so the delegate tool is
    # actually rendered for NATIVE controller routes too (previously only
    # invocation_plan was replaced and the tool never reached the wire).
    # This request is already an intentionally reduced controller surface:
    # re-selecting it dynamically could hide a tool that the controller
    # was explicitly permitted to choose and would turn a valid decision
    # into a misleading "withheld" failure.
    surface = ToolSurfacePlanner().plan(
        canonical,
        ToolSurfaceConfig(mode=ToolSurfaceMode.TRANSPARENT),
    )
    return rebuild(
        invocation,
        canonical,
        plan=plan,
        tool_surface_plan=surface,
    )


class ControllerAttemptExecutor:
    """Bounded primary↔controller loop, once per controlled attempt."""

    def __init__(
        self,
        *,
        gateway: Any,
        config: Any,
        controller_state: Any,
        select_controller_route: Any,
        inspect_model_runtime: Any,
        backend_metadata_from_runtime: Any,
        resolve_profile: Any,
        no_private_capabilities: Any,
    ) -> None:
        # ``gateway`` is held for late-bound hook resolution — tests
        # monkeypatch ``gateway._handle_request_send`` /
        # ``_prepare_invocation`` AFTER construction, so those two are
        # resolved per call like the generation seam's lazy transport.
        self._gateway = gateway
        self._config = config
        self._controller_state = controller_state
        self._select_controller_route = select_controller_route
        self._inspect_model_runtime = inspect_model_runtime
        self._backend_metadata_from_runtime = backend_metadata_from_runtime
        self._resolve_profile = resolve_profile
        self._no_private_capabilities = no_private_capabilities

    def _send(self, invocation: Any, exec_record: Any) -> Any:
        return self._gateway._handle_request_send(invocation, exec_record)

    def _rebuild(self, invocation: Any, request: Any, **kw: Any) -> Any:
        return self._gateway._rebuild_invocation_from_request(invocation, request, **kw)

    async def _prepare(self, *args: Any, **kw: Any) -> Any:
        # The loop only needs A prepared invocation for the controller
        # request; preparation resolves through the live gateway so patched
        # test hooks are honored. _prepare_invocation_async internally uses
        # _prepare_invocation, so a patch on the sync method covers both.
        return await self._gateway._prepare_invocation_async(*args, **kw)

    async def execute(
        self,
        invocation: Any,
        exec_record: Any,
    ) -> CanonicalResponse:
        """Use a qualified companion route for the agent/tool protocol.

        The primary route is first asked for a tool-free work product. The
        configured controller receives that work product plus the original
        client request and decides whether to emit canonical tool calls. Tool
        execution remains with the client; controller-generated calls are
        explicitly provenance-labelled before leaving Interop.
        """
        from agent_interop.controller import CompatibilityController
        from agent_interop.controller.policy import (
            mark_controller_provenance,
            missing_controller_result_ids,
        )
        from agent_interop.controller.prompts import CONTROLLER_SYSTEM_PROMPT
        from agent_interop.controller.types import ControllerAction, ControllerSessionState

        # P0-46: controlled mode is a compatibility escape hatch, not a fast
        # path — it multiplies model latency (primary turn + controller turn
        # per cycle).  Make that cost explicit in diagnostics for every
        # generation this path produces.
        exec_record.record_compatibility_event("controlled_path_model_generations")

        effective_controller = invocation.route.controller or self._config.controller
        controller_route = await self._select_controller_route(
            invocation.route,
            effective_controller,
        )
        if controller_route is None:
            from agent_interop.controller.companion import ControllerRegistry

            candidates = ControllerRegistry().candidates(self._config, invocation.route)
            return CanonicalResponse(
                model=CanonicalModelReference(
                    requested_name=invocation.reconciled_request.model.requested_name,
                    resolved_name=invocation.route.upstream_model,
                ),
                error=CanonicalError(
                    code=InteropErrorCode.CONTROLLER_UNAVAILABLE,
                    message=(
                        "No configured controller route has passed the required qualification level"
                        if candidates
                        else "No distinct configured controller route is available"
                    ),
                    details={
                        "path": "controlled",
                        "responsible": "controller",
                        "minimum_controller_level": effective_controller.minimum_controller_level,
                        "next": "run interop qualify for a controller route"
                        if candidates
                        else "configure_controller_route",
                    },
                ),
            )

        # Controller-mediated observations must identify the actual served
        # controller, not merely the primary request's route.  These fields
        # are part of the immutable evidence key and isolate controller
        # upgrades, profile changes, and tag repoints.
        controller_runtime = await self._inspect_model_runtime(controller_route)
        controller_metadata = self._backend_metadata_from_runtime(controller_runtime)
        controller_profile = self._resolve_profile(controller_route, controller_metadata)
        if exec_record.compatibility_key is not None:
            exec_record.compatibility_key = replace(
                exec_record.compatibility_key,
                controller_model_id=controller_route.upstream_model,
                controller_model_digest=controller_runtime.model_digest,
                controller_profile_revision=getattr(controller_profile, "profile_revision", ""),
            )

        session_id = invocation.request_context.session_id
        client_id = invocation.request_context.client_id
        prior_state = self._controller_state.get(session_id, client_id, invocation.route.id)
        if prior_state is not None:
            if prior_state.primary_turn_count >= effective_controller.max_primary_turns:
                self._controller_state.remove(session_id, client_id, invocation.route.id)
                return CanonicalResponse(
                    error=CanonicalError(
                        code=InteropErrorCode.CONTROLLER_LOOP_DETECTED,
                        message="Primary worker exceeded its bounded delegation turn budget",
                        details={
                            "path": "controlled",
                            "responsible": "primary",
                            "next": "abort_session",
                        },
                    ),
                )
            if prior_state.controller_turn_count >= effective_controller.max_controller_turns:
                self._controller_state.remove(session_id, client_id, invocation.route.id)
                return CanonicalResponse(
                    error=CanonicalError(
                        code=InteropErrorCode.CONTROLLER_LOOP_DETECTED,
                        message="Controller exceeded its bounded turn budget",
                        details={
                            "path": "controlled",
                            "responsible": "controller",
                            "next": "abort_session",
                        },
                    ),
                )
            missing_result_ids = missing_controller_result_ids(
                invocation.reconciled_request.messages,
                prior_state.pending_tool_call_ids,
            )
            if missing_result_ids:
                return CanonicalResponse(
                    error=CanonicalError(
                        code=InteropErrorCode.CONTROLLER_LOOP_DETECTED,
                        message="Controller session resumed before every pending tool result arrived",
                        details={
                            "path": "controlled",
                            "responsible": "client",
                            "missing_tool_result_ids": list(missing_result_ids),
                            "next": "return_tool_result",
                        },
                    ),
                )

        # The primary worker may reason/code, but cannot emit tool calls. Its
        # output is untrusted advisory text to the controller.  A controller
        # may request a focused refinement through one private tool; Interop
        # consumes that request and never returns it to the coding client.
        primary_turns = prior_state.primary_turn_count if prior_state else 0
        controller_turns = prior_state.controller_turn_count if prior_state else 0
        work_products: list[str] = []
        controller_context = replace(invocation.request_context, route_id=controller_route.id)

        def loop_error(message: str, *, responsible: str) -> CanonicalResponse:
            self._controller_state.remove(session_id, client_id, invocation.route.id)
            return CanonicalResponse(
                error=CanonicalError(
                    code=InteropErrorCode.CONTROLLER_LOOP_DETECTED,
                    message=message,
                    details={
                        "path": "controlled",
                        "responsible": responsible,
                        "next": "abort_session",
                    },
                ),
            )

        async def ask_primary(refinement_prompt: str = "") -> CanonicalResponse | None:
            nonlocal primary_turns
            if primary_turns >= effective_controller.max_primary_turns:
                return loop_error(
                    "Primary worker exceeded its bounded delegation turn budget",
                    responsible="primary",
                )
            primary_system = list(invocation.reconciled_request.system)
            if refinement_prompt:
                primary_system.append(
                    CanonicalTextBlock(
                        text=(
                            "The compatibility controller needs this focused follow-up. "
                            "Do not claim to execute tools; provide only reasoning or a "
                            f"work product.\n\n{refinement_prompt}"
                        ),
                    )
                )
            primary_request = replace(
                invocation.reconciled_request,
                system=primary_system,
                tools=[],
                tool_choice=CanonicalToolChoice.none(),
            )
            primary_plan = replace(
                invocation.invocation_plan,
                effective_tool_mode=ToolMode.DISABLED,
                upstream_tools=(),
                prompt_contract="",
                parser_id=None,
            )
            # Review #21: atomic rebuild — the worker turn strips tools, so
            # model_request/model_view/visible tools must reflect that too,
            # not just reconciled_request + plan.
            primary_invocation = self._rebuild(
                invocation,
                primary_request,
                plan=primary_plan,
                private_capabilities=self._no_private_capabilities(),
            )
            primary = await self._send(primary_invocation, exec_record)
            if primary.error is not None:
                return primary
            primary_turns += 1
            work_products.append(
                "\n".join(
                    block.text for block in primary.content if isinstance(block, CanonicalTextBlock)
                )
            )
            return None

        refinement_prompt = ""
        while True:
            # A controller delegation consumes a controller turn already.
            # Check that budget before asking the worker for another costly
            # work product; otherwise a saturated controller could still
            # cause one unnecessary primary-model generation.
            if controller_turns >= effective_controller.max_controller_turns:
                return loop_error(
                    "Controller exceeded its bounded turn budget",
                    responsible="controller",
                )
            primary_failure = await ask_primary(refinement_prompt)
            if primary_failure is not None:
                return primary_failure
            refinement_prompt = ""
            # P0-47: newest-first bounded work-product selection.  The
            # controller refines the LATEST primary output; older products
            # are history, and concatenating all of them grows controller
            # latency and context without bound.  Walk newest→oldest keeping
            # what fits the configured cap; the newest product always rides
            # alone even if it alone exceeds the cap (truncating it would
            # hide the very refinement the controller must judge).
            controller_config = getattr(invocation.route, "controller", None)
            work_product_cap = (
                getattr(controller_config, "max_primary_work_product_tokens", 0) or 4000
            )
            kept: list[str] = []
            used = 0
            for product in reversed(work_products):
                cost = (len(product) + 3) // 4
                if kept and used + cost > work_product_cap:
                    break
                kept.append(product)
                used += cost
            kept.reverse()
            first_kept = len(work_products) - len(kept) + 1
            rendered_work_products = "\n\n".join(
                f"[primary turn {index}]\n{product}"
                for index, product in enumerate(kept, start=first_kept)
            )
            controller_system = list(invocation.reconciled_request.system)
            controller_system.append(
                CanonicalTextBlock(
                    text=f"{CONTROLLER_SYSTEM_PROMPT}\n\nPrimary worker work product:\n{rendered_work_products}",
                )
            )
            # The controller gets only the selected client surface plus the
            # private refinement action. A client named-tool requirement must
            # not prevent it from obtaining necessary worker reasoning.
            visible_tools = tuple(
                getattr(
                    invocation.tool_surface_plan,
                    "visible_tools",
                    invocation.reconciled_request.tools,
                )
            )
            controller_request = replace(
                invocation.reconciled_request,
                model=replace(
                    invocation.reconciled_request.model, requested_name=controller_route.id
                ),
                system=controller_system,
                tools=list(visible_tools),
                tool_choice=CanonicalToolChoice.auto(),
            )
            controller_invocation = await self._prepare(
                controller_request,
                controller_context,
                streaming=False,
                execution=exec_record,
            )
            controller_invocation = controller_invocation_with_delegate_tool(
                controller_invocation,
                self._rebuild,
            )
            # The controller is a second model call; keep it visible in the
            # request's efficiency record rather than hiding it behind a single
            # compatibility attempt.
            exec_record.record_attempt(controller_tokens=(len(rendered_work_products) + 3) // 4)
            response = await self._send(controller_invocation, exec_record)
            if response.error is not None:
                return response
            controller_turns += 1

            decision = CompatibilityController().decide(response.content)
            if decision.action == ControllerAction.FAIL:
                diagnostic = (
                    decision.diagnostics[0]
                    if decision.diagnostics
                    else "invalid_controller_decision"
                )
                message = (
                    "Controller mixed a primary-refinement request with client tool calls"
                    if diagnostic == "mixed_private_delegation_and_client_calls"
                    else "Controller emitted an invalid primary-refinement request"
                )
                return loop_error(message, responsible="controller")
            if decision.action == ControllerAction.DELEGATE_PRIMARY:
                exec_record.record_compatibility_event("controller_delegated_primary")
                # The next loop iteration performs exactly one additional
                # worker turn and one controller decision, with both budgets
                # checked before dispatch.
                refinement_prompt = decision.primary_prompt
                continue

            raw_calls = decision.tool_calls

            # The controller used ``auto`` internally so it could request
            # worker refinement even for a client named-tool request. Before
            # its response crosses back to the client boundary, restore and
            # enforce the client's original tool-choice contract.
            outer_choice = invocation.reconciled_request.tool_choice
            call_names = tuple(call.name for call in raw_calls)
            choice_violation = ""
            if outer_choice.mode == ToolChoiceMode.NONE and raw_calls:
                choice_violation = "tool_choice=none forbids controller tool calls"
            elif outer_choice.mode == ToolChoiceMode.REQUIRED and not raw_calls:
                choice_violation = "tool_choice=required requires a controller tool call"
            elif outer_choice.mode == ToolChoiceMode.NAMED and (
                not raw_calls or any(name != outer_choice.name for name in call_names)
            ):
                choice_violation = f"tool_choice=named requires only {outer_choice.name!r}, got {list(call_names)!r}"
            if choice_violation:
                self._controller_state.remove(session_id, client_id, invocation.route.id)
                return CanonicalResponse(
                    error=CanonicalError(
                        code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                        message=choice_violation,
                        details={
                            "path": "controlled",
                            "responsible": "controller",
                            "next": "retry_with_qualified_controller",
                        },
                    ),
                )

            calls = tuple(
                mark_controller_provenance(block)
                if isinstance(block, CanonicalToolCallBlock)
                else block
                for block in response.content
            )
            pending = tuple(
                block.id for block in calls if isinstance(block, CanonicalToolCallBlock)
            )
            self._controller_state.put(
                ControllerSessionState(
                    session_id=controller_context.session_id,
                    route_id=invocation.route.id,
                    client_id=controller_context.client_id,
                    controller_route_id=controller_route.id,
                    primary_route_id=invocation.route.id,
                    phase="awaiting_tool_result" if pending else "final_text",
                    visible_tool_fingerprint=invocation.tool_surface_plan.fingerprint,
                    pending_tool_call_ids=pending,
                    primary_turn_count=primary_turns,
                    controller_turn_count=controller_turns,
                )
            )
            return replace(response, content=list(calls))
