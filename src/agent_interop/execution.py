"""Request-scoped execution coordinator.

Ties together context, route, invocation plan, history diagnostics,
tool decisions, repair budget, and response outcome for a single
gateway request.  Optionally emits a sanitized replay case after
completion.

This is the single coordinator that makes evidence, replay, sessions,
and loop detection actually participate in the live request path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_interop.abi import (
    CanonicalError,
    CanonicalResponse,
    RepairOutcome,
)
from agent_interop.config import ModelRoute
from agent_interop.context import RequestContext
from agent_interop.repair.invocation import InvocationPlan
from agent_interop.repair.pipeline import RepairBudget
from agent_interop.replay.types import CompatibilityKey

logger = logging.getLogger("agent_interop.execution")


class ExecutionState(str, Enum):
    """States for execution lifecycle tracking."""
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ToolDecisionRecord:
    """Record of a single tool-call decision during execution."""

    tool_name: str = ""
    candidate_id: str = ""
    outcome_status: str = ""
    repair_steps: list[str] = field(default_factory=list)
    accepted: bool = False


@dataclass
class RepairSavingsEstimate:
    """Conservative estimate of a repair avoiding a model regeneration.

    These are explicitly estimates: a successful deterministic repair proves
    the call was accepted, not that a model would certainly have retried.
    """

    repaired_without_regeneration: bool = False
    estimated_prompt_tokens_avoided: int = 0
    estimated_tool_schema_tokens_avoided: int = 0
    estimated_latency_avoided_ms: int | None = None


def _estimate_internal_schema_tokens() -> int:
    """Estimate token count for the internal __interop_* tool schemas.

    Uses the actual JSON schema definitions from context_store.tools rather
    than the old ``len(visible_tool_names) * 50`` placeholder.
    """
    import json

    from agent_interop.context_store.tools import all_internal_tools

    total_chars = 0
    for tool in all_internal_tools():
        total_chars += len(tool.name)
        total_chars += len(tool.description)
        total_chars += len(json.dumps(tool.input_schema, default=str))
    return (total_chars + 3) // 4


@dataclass
class AttemptRecord:
    """One completed model generation, tagged by purpose and path (P0.41).

    This is the ledger behind :meth:`TokenEfficiencyRecord.generation_metrics`
    — the P0-62 release metric ("extra model generations" per request).
    """

    purpose: str = "worker"
    path: str = "direct"
    rendered_input_tokens: int = 0
    actual_input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass
class TokenEfficiencyRecord:
    """Per-request accounting for cost per successfully completed task.

    P1.6: fields measure what their names say. The old
    tool_result_tokens_original/visible were entire message-history token
    estimates, not tool-result tokens. Now we report actual breakdown
    fields.
    """
    # Tool schema accounting
    original_tool_schema_tokens: int = 0
    visible_tool_schema_tokens: int = 0
    internal_tool_schema_tokens: int = 0  # P0.4: __interop_ tools added to surface

    # History accounting
    original_history_tokens: int = 0
    model_visible_history_tokens: int = 0

    # Tool-result accounting (P1.6: actual tool-result tokens, not whole history)
    tool_result_tokens_original: int = 0
    tool_result_tokens_visible: int = 0

    # System + contract
    system_tokens: int = 0
    projected_system_tokens: int = 0  # P0.8
    prompted_contract_tokens: int = 0

    # Projection metadata (P0.3/P0.7)
    virtualized_result_count: int = 0
    paged_result_count: int = 0

    # Internal tools (P0.4)
    internal_tool_calls_executed: int = 0

    # Output
    output_reserve_tokens: int = 0

    # Repair accounting
    repair_operations: int = 0
    regenerations_avoided: int = 0
    controller_tokens: int = 0

    # Per-attempt accounting (populated via update_from_attempt)
    rendered_input_tokens: int = 0
    actual_input_tokens: int = 0
    output_tokens: int = 0
    visible_tool_schemas: int = 0
    private_schemas: int = 0
    virtualized_bytes: int = 0
    retrieved_bytes: int = 0
    latency_ms: int = 0

    # Totals
    total_model_input_tokens: int = 0
    total_model_output_tokens: int = 0
    actual_prompt_eval_tokens: int = 0  # P0.10: from backend prompt_eval_count
    total_attempts: int = 0
    task_completed: bool = False

    # Per-generation ledger (P0-62 release metric). A declared field, not a
    # hasattr-grown attribute: the historical `_attempts` dict-list was
    # created lazily on first update_from_attempt and read back through
    # getattr — invisible to tooling and typo-prone.
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def schema_reduction_tokens(self) -> int:
        return max(0, self.original_tool_schema_tokens - self.visible_tool_schema_tokens)

    @property
    def history_reduction_tokens(self) -> int:
        return max(0, self.original_history_tokens - self.model_visible_history_tokens)

    @property
    def total_reduction_tokens(self) -> int:
        return self.schema_reduction_tokens + self.history_reduction_tokens

    @property
    def reduction_ratio(self) -> float:
        original = self.original_tool_schema_tokens + self.original_history_tokens
        if original <= 0:
            return 0.0
        return self.total_reduction_tokens / original

    def update_from_attempt(
        self,
        *,
        rendered_input_tokens: int = 0,
        actual_input_tokens: int = 0,
        output_tokens: int = 0,
        visible_tool_schemas: int = 0,
        private_schemas: int = 0,
        virtualized_bytes: int = 0,
        retrieved_bytes: int = 0,
        controller_tokens: int = 0,
        latency_ms: int = 0,
        purpose: str = "worker",
        path: str = "direct",
    ) -> None:
        """Record actual measurements from a completed model call attempt.

        P0.41: Added purpose and path for tracking different generation types
        (worker, private_continuation, primary, controller, repair).
        """
        self.rendered_input_tokens += rendered_input_tokens
        self.actual_input_tokens += actual_input_tokens
        self.output_tokens += output_tokens
        self.visible_tool_schemas += visible_tool_schemas
        self.private_schemas += private_schemas
        self.virtualized_bytes += virtualized_bytes
        self.retrieved_bytes += retrieved_bytes
        self.controller_tokens += controller_tokens
        self.latency_ms += latency_ms
        self.attempts.append(AttemptRecord(
            purpose=purpose,
            path=path,
            rendered_input_tokens=rendered_input_tokens,
            actual_input_tokens=actual_input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        ))

    def record_virtualization(self, result_count: int, virtualized_bytes: int) -> None:
        """Update virtualization counters after a projection pass."""
        self.virtualized_result_count += result_count
        self.virtualized_bytes += virtualized_bytes

    def record_system_projection(self, projected_tokens: int) -> None:
        """Update projected system tokens after a system-projection pass."""
        self.projected_system_tokens = projected_tokens

    def as_breakdown_dict(self) -> dict[str, Any]:
        return {
            "system_tokens": self.system_tokens,
            "projected_system_tokens": self.projected_system_tokens,
            "ordinary_history_tokens": self.model_visible_history_tokens,
            "tool_result_tokens": self.tool_result_tokens_visible,
            "tool_schema_tokens": self.visible_tool_schema_tokens,
            "internal_tool_schema_tokens": self.internal_tool_schema_tokens,
            "prompt_contract_tokens": self.prompted_contract_tokens,
            "output_reserve_tokens": self.output_reserve_tokens,
            "virtualized_results": self.virtualized_result_count,
            "internal_tool_calls": self.internal_tool_calls_executed,
            "total_model_input_tokens": self.total_model_input_tokens,
            "actual_prompt_eval_tokens": self.actual_prompt_eval_tokens,
            "reduction_ratio": self.reduction_ratio,
        }

    def breakdown_dict(self) -> dict[str, Any]:
        """Return a clean summary for logging.

        Includes per-attempt accounting alongside the standard breakdown.
        """
        base = self.as_breakdown_dict()
        base.update({
            "rendered_input_tokens": self.rendered_input_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "output_tokens": self.output_tokens,
            "visible_tool_schemas": self.visible_tool_schemas,
            "private_schemas": self.private_schemas,
            "virtualized_bytes": self.virtualized_bytes,
            "retrieved_bytes": self.retrieved_bytes,
            "controller_tokens": self.controller_tokens,
            "latency_ms": self.latency_ms,
            "total_attempts": self.total_attempts,
            "task_completed": self.task_completed,
            # P0-62: "extra model generations" is the primary release
            # metric — more important than shaving microseconds.  Every
            # purpose beyond the single worker generation must be
            # explainable (explicit fallback, explicit private retrieval).
            **self.generation_metrics(),
        })
        return base

    def generation_metrics(self) -> dict[str, Any]:
        """P0-62: model-generation counts, total and by purpose.

        Release gate shape:

        * warm chat / DIRECT / ADAPTED happy path → ``total`` == 1 and
          ``by_purpose`` == {"worker": 1} (zero EXTRA generations);
        * behavioral probes during an ordinary request → none recorded
          here at all (probes never touch a request's execution record);
        * controller/context-summary generations appear ONLY when an
          explicit fallback ran;
        * private generations appear ONLY after the model explicitly
          requested private retrieval.
        """
        by_purpose: dict[str, int] = {}
        for attempt in self.attempts:
            by_purpose[attempt.purpose] = by_purpose.get(attempt.purpose, 0) + 1
        return {
            "model_generation_count": len(self.attempts),
            "model_generations_by_purpose": by_purpose,
        }


@dataclass
class InteropRequestExecution:
    """Request-scoped execution coordinator.

    Created once per request at the gateway entry point.  Carries
    mutable state through the request lifecycle and optionally emits
    a sanitized replay case after completion.
    """

    context: RequestContext | None = None
    route: ModelRoute | None = None
    compatibility_key: CompatibilityKey | None = None
    invocation_plan: InvocationPlan | None = None
    history_diagnostics: list[str] = field(default_factory=list)
    raw_frame_evidence: list[dict[str, Any]] = field(default_factory=list)
    parser_diagnostics: list[str] = field(default_factory=list)
    compatibility_events: list[str] = field(default_factory=list)
    tool_decisions: list[ToolDecisionRecord] = field(default_factory=list)
    repair_savings_estimates: list[RepairSavingsEstimate] = field(default_factory=list)
    token_efficiency: TokenEfficiencyRecord = field(default_factory=TokenEfficiencyRecord)
    diagnostic_case_id: str = ""
    repair_budget: RepairBudget | None = None
    attempt_budget: Any | None = None  # P0.18: AttemptBudget for cumulative tracking
    # Request-scoped ContextStore ref lifecycle. Registered refs are pinned
    # for the request and unpinned by registry.close(); snapshot() is the
    # output firewall's leak identity. Present only when the gateway opened
    # a registry for this request.
    ref_registry: Any | None = None
    response_outcome: str = ""  # accepted | rejected | error
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    state: ExecutionState = ExecutionState.ACTIVE

    def record_tool_decision(
        self,
        tool_name: str,
        candidate_id: str,
        outcome: RepairOutcome | None,
        accepted: bool,
    ) -> None:
        """Record a tool-call decision for evidence/replay."""
        record = ToolDecisionRecord(
            tool_name=tool_name,
            candidate_id=candidate_id,
            outcome_status=outcome.status.value if outcome else "unknown",
            # Rule IDs (e.g. "rename_aliased_fields"), not a dataclass repr —
            # this is what feeds `interop repair stats`' per-rule breakdown,
            # so it needs to be a clean, aggregable identifier.
            repair_steps=[s.rule for s in (outcome.steps if outcome and outcome.steps else [])],
            accepted=accepted,
        )
        self.tool_decisions.append(record)
        repair_operations = len(record.repair_steps)
        self.token_efficiency.repair_operations += repair_operations
        if accepted and repair_operations:
            self.token_efficiency.regenerations_avoided += 1
            self.repair_savings_estimates.append(RepairSavingsEstimate(
                repaired_without_regeneration=True,
                estimated_prompt_tokens_avoided=self.token_efficiency.original_history_tokens,
                estimated_tool_schema_tokens_avoided=self.token_efficiency.original_tool_schema_tokens,
            ))

    def configure_token_efficiency(self, invocation: Any) -> None:
        """Seed transparent accounting from the chosen surface/context plan.

        P1.6: populate actual breakdown fields, not whole-history estimates.
        """
        surface = getattr(invocation, "tool_surface_plan", None)
        context = getattr(invocation, "context_plan", None)
        plan = getattr(invocation, "invocation_plan", None)
        model_view = getattr(invocation, "model_view", None)
        if surface is not None:
            self.token_efficiency.original_tool_schema_tokens = surface.original_schema_tokens
            self.token_efficiency.visible_tool_schema_tokens = surface.visible_schema_tokens
        if context is not None:
            self.token_efficiency.original_history_tokens = context.before.message_tokens
            self.token_efficiency.model_visible_history_tokens = context.after.message_tokens
            self.token_efficiency.total_model_input_tokens = context.after.total_required_tokens
            self.token_efficiency.output_reserve_tokens = context.after.output_reserve_tokens
        if plan is not None:
            self.token_efficiency.prompted_contract_tokens = (len(plan.prompt_contract) + 3) // 4
        if model_view is not None:
            self.token_efficiency.internal_tool_schema_tokens = _estimate_internal_schema_tokens()
            self.token_efficiency.virtualized_result_count = 0  # updated during virtualization
            self.token_efficiency.projected_system_tokens = 0  # updated during projection

    def record_attempt(self, *, controller_tokens: int = 0) -> None:
        self.token_efficiency.total_attempts += 1
        self.token_efficiency.controller_tokens += controller_tokens

    def record_response_tokens(self, response: CanonicalResponse) -> None:
        visible = "".join(
            getattr(block, "text", "") + str(getattr(block, "arguments", ""))
            for block in response.content
        )
        self.token_efficiency.total_model_output_tokens += (len(visible) + 3) // 4

    def record_malformed_frame(self, ordinal: int, error: str, raw: str) -> None:
        """Record a malformed stream frame for evidence."""
        self.raw_frame_evidence.append({
            "ordinal": ordinal,
            "error": error,
            "raw": raw[:500],
        })

    def record_parser_diagnostic(self, message: str) -> None:
        """Record a parser/extraction diagnostic."""
        self.parser_diagnostics.append(message)

    def record_compatibility_event(self, event: str) -> None:
        """Record bounded planner/executor events without request content."""
        self.compatibility_events.append(event)

    def record_repair_hint(self, note: str) -> None:
        """Record an internal repair hint (P1.4).

        Stored on the execution record instead of being leaked to the coding
        client via the assistant response. Surfaces to the model through the
        internal recall surface on subsequent turns.
        """
        self.compatibility_events.append("repair_hint:" + note[:200])

    def finalize_response(self, response: CanonicalResponse) -> None:
        """Mark execution as finished with a successful response."""
        if self.state != ExecutionState.ACTIVE:
            return
        self.finished_at = time.monotonic()
        if response.error:
            self.response_outcome = "error"
            self.state = ExecutionState.FAILED
        elif self.tool_decisions and any(not d.accepted for d in self.tool_decisions):
            self.response_outcome = "partial"
            self.state = ExecutionState.SUCCEEDED
        else:
            self.response_outcome = "accepted"
            self.state = ExecutionState.SUCCEEDED
        self.record_response_tokens(response)
        self.token_efficiency.task_completed = self.state == ExecutionState.SUCCEEDED
        self._log_summary()

    def finalize_error(self, error: CanonicalError | Exception | None = None) -> None:
        """Mark execution as finished with an error."""
        if self.state != ExecutionState.ACTIVE:
            return
        self.finished_at = time.monotonic()
        self.response_outcome = "error"
        self.state = ExecutionState.FAILED
        self._log_summary()

    def finalize_cancelled(self) -> None:
        """Mark execution as cancelled."""
        if self.state != ExecutionState.ACTIVE:
            return
        self.finished_at = time.monotonic()
        self.response_outcome = "cancelled"
        self.state = ExecutionState.CANCELLED
        self._log_summary()

    @property
    def is_active(self) -> bool:
        """Check if execution is still active."""
        return self.state == ExecutionState.ACTIVE

    def _log_summary(self) -> None:
        elapsed_ms = ((self.finished_at or time.monotonic()) - self.started_at) * 1000
        logger.info(
            "request %s completed in %.0fms: outcome=%s, tools=%d/%d accepted, "
            "history_issues=%d, malformed_frames=%d, parser_diags=%d",
            self.context.request_id if self.context else "?",
            elapsed_ms,
            self.response_outcome,
            sum(1 for d in self.tool_decisions if d.accepted),
            len(self.tool_decisions),
            len(self.history_diagnostics),
            len(self.raw_frame_evidence),
            len(self.parser_diagnostics),
        )

    def to_sanitized_dict(self) -> dict[str, Any]:
        """Export a sanitized execution summary for replay/evidence.

        Never includes credentials or raw arguments.
        """
        return {
            "request_id": self.context.request_id if self.context else "",
            "session_id": self.context.session_id if self.context else "",
            "client_id": self.context.client_id if self.context else "",
            "route_id": self.route.id if self.route else "",
            "response_outcome": self.response_outcome,
            "diagnostic_case_id": self.diagnostic_case_id,
            "tool_decisions": [
                {
                    "tool_name": d.tool_name,
                    "outcome_status": d.outcome_status,
                    "accepted": d.accepted,
                    "repair_steps_count": len(d.repair_steps),
                }
                for d in self.tool_decisions
            ],
            "history_diagnostics_count": len(self.history_diagnostics),
            "malformed_frames_count": len(self.raw_frame_evidence),
            "parser_diagnostics_count": len(self.parser_diagnostics),
            "elapsed_ms": ((self.finished_at or time.monotonic()) - self.started_at) * 1000,
        }
