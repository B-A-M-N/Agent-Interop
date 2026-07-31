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


@dataclass
class TokenEfficiencyRecord:
    """Per-request accounting for cost per successfully completed task."""

    original_tool_schema_tokens: int = 0
    visible_tool_schema_tokens: int = 0
    original_history_tokens: int = 0
    model_visible_history_tokens: int = 0
    prompted_contract_tokens: int = 0
    tool_result_tokens_original: int = 0
    tool_result_tokens_visible: int = 0
    repair_operations: int = 0
    regenerations_avoided: int = 0
    controller_tokens: int = 0
    total_model_input_tokens: int = 0
    total_model_output_tokens: int = 0
    total_attempts: int = 0
    task_completed: bool = False


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
        """Seed transparent accounting from the chosen surface/context plan."""
        surface = getattr(invocation, "tool_surface_plan", None)
        context = getattr(invocation, "context_plan", None)
        plan = getattr(invocation, "invocation_plan", None)
        if surface is not None:
            self.token_efficiency.original_tool_schema_tokens = surface.original_schema_tokens
            self.token_efficiency.visible_tool_schema_tokens = surface.visible_schema_tokens
        if context is not None:
            self.token_efficiency.original_history_tokens = context.before.message_tokens
            self.token_efficiency.model_visible_history_tokens = context.after.message_tokens
            self.token_efficiency.tool_result_tokens_original = context.before.message_tokens
            self.token_efficiency.tool_result_tokens_visible = context.after.message_tokens
            self.token_efficiency.total_model_input_tokens = context.after.total_required_tokens
        if plan is not None:
            self.token_efficiency.prompted_contract_tokens = (len(plan.prompt_contract) + 3) // 4

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
