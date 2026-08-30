"""Controller-driven summarization of old conversation history (P0-48).

Extracted from Gateway: one module owns the opt-in, last-resort context
adaptation that replaces planner-approved old turns with a controller
model's summary.  The deterministic history projector
(``history.projector``) runs first and is lossless-by-paging; this module
is the lossy step and is deliberately visible and auditable.

Coupling contract: no request-scoped state.  Everything that needs a live
gateway — controller-route selection, runtime inspection, invocation
preparation, and the send — is resolved lazily through the ``gateway``
back-reference on every call.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTextBlock, CanonicalToolChoice
from agent_interop.execution import InteropRequestExecution

__all__ = [
    "ControllerHistorySummarizer",
    "replace_compacted_history_with_controller_summary",
]


def replace_compacted_history_with_controller_summary(
    request: CanonicalRequest,
    context_plan: Any,
    summary: str,
) -> CanonicalRequest:
    """Replace only planner-approved old turns with an explicit summary.

    The context planner has already excluded system/developer messages,
    the latest user turn, and the current tool exchange from its compacted
    indices.  Keeping this mutation here makes the lossy step visible and
    auditable, instead of allowing a controller response to silently
    overwrite arbitrary canonical history.
    """
    compacted = set(context_plan.compacted_message_indices)
    messages = [
        message for index, message in enumerate(request.messages)
        if index not in compacted
    ]
    system = [*request.system, CanonicalTextBlock(
        text=(
            "Interop controller summary of older conversation history. "
            "It is incomplete; retain and prioritize all unsummarized "
            "system/developer instructions and current tool results.\n\n"
            f"{summary.strip()}"
        ),
    )]
    return replace(request, system=system, messages=messages)


class ControllerHistorySummarizer:
    """Ask a qualified, sufficiently large controller to summarize old turns."""

    def __init__(self, *, gateway: Any) -> None:
        self._gateway = gateway

    async def summarize(
        self,
        *,
        route: Any,
        request: CanonicalRequest,
        context: Any,
        context_plan: Any,
        inspect_runtime: bool,
        execution: InteropRequestExecution | None = None,
    ) -> CanonicalRequest | None:
        """Summarize old turns, or return None to leave the request untouched.

        This is deliberately an opt-in *last* context adaptation.  It runs
        only after deterministic safe tool-result reduction failed, only when
        the controller can hold the original no-tool request itself, and only
        replaces message indices the context planner already marked as old.
        Unknown capacity, an unqualified controller, a controller error, or
        an empty/non-text summary all leave the request untouched.

        P0-48: when the caller passes the OUTER request's execution record,
        the summary generation spends against that request's AttemptBudget
        (it can no longer bypass the token/latency ceilings) and is tagged
        ``purpose="context_summary"`` in the outer telemetry so diagnostics
        show exactly why a second model was consulted.
        """
        gw = self._gateway
        effective_controller = route.controller or gw.config.controller
        if not (
            route.context.allow_controller_decomposition
            and effective_controller.enabled
            and context_plan.compacted_message_indices
        ):
            return None
        controller_route = await gw._select_controller_route(route, effective_controller)
        if controller_route is None:
            return None

        from agent_interop.context_budget import effective_context_limit
        from agent_interop.context_budget.estimator import estimate_request_context

        controller_runtime = (
            await gw._inspect_model_runtime(controller_route)
            if inspect_runtime else gw._static_runtime_capabilities(controller_route)
        )
        # P0-49: architecture maximum is a ceiling, not an allocation — the
        # summary call is gated on real serving capacity only.
        controller_limit = effective_context_limit(
            configured_limit=controller_runtime.configured_context_tokens,
            route_override=controller_route.context.context_limit_tokens,
            observed_effective_limit=controller_runtime.effective_context_tokens,
            architecture_ceiling=controller_runtime.architecture_context_tokens,
        )
        # No observed capacity means no proof that a summary call is safe.
        if not controller_limit:
            return None

        summary_generation = replace(
            request.generation,
            stream=False,
            max_output_tokens=min(512, max(64, request.generation.max_output_tokens)),
        )
        summary_request = replace(
            request,
            model=replace(request.model, requested_name=controller_route.id),
            system=[*request.system, CanonicalTextBlock(
                text=(
                    "Summarize only the older conversation history for a later "
                    "coding-model turn. Preserve file paths, tool-call IDs, error "
                    "outcomes, edits, and unresolved requirements. Do not claim any "
                    "tool was executed. Return concise plain text only."
                ),
            )],
            tools=[],
            tool_choice=CanonicalToolChoice.none(),
            generation=summary_generation,
        )
        summary_required = estimate_request_context(
            summary_request,
            visible_tools=(),
            output_reserve_tokens=summary_generation.max_output_tokens,
        ).total_required_tokens
        if summary_required > int(controller_limit * 0.90):
            return None

        summary_context = replace(context, route_id=controller_route.id)
        summary_execution = InteropRequestExecution(context=summary_context)
        if execution is not None:
            # P0-48: share the outer request's resource budget.  The seam
            # reads attempt_budget off the execution record, so attaching it
            # here routes every accounting hook (rendered bytes, input
            # tokens, output reservations) through the request's ledger.
            summary_execution.attempt_budget = getattr(execution, "attempt_budget", None)
        try:
            # Late-bound: tests replace gateway._prepare_invocation_async /
            # _handle_request_send after construction.
            summary_invocation = await gw._prepare_invocation_async(
                summary_request,
                summary_context,
                streaming=False,
                execution=summary_execution,
                inspect_runtime=inspect_runtime,
                allow_controller_summary=False,
            )
            response = await gw._handle_request_send(summary_invocation, summary_execution)
        except (ValueError, RuntimeError):
            return None
        if response.error is not None:
            return None
        text = "\n".join(
            block.text for block in response.content if isinstance(block, CanonicalTextBlock)
        ).strip()
        if not text:
            return None
        if execution is not None:
            # P0-48: tag the outer telemetry so a summary generation is
            # visible as such (not silently folded into worker accounting).
            usage = getattr(response, "usage", None)
            execution.token_efficiency.update_from_attempt(
                rendered_input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                purpose="context_summary",
                path="controller",
            )
        return replace_compacted_history_with_controller_summary(request, context_plan, text)
