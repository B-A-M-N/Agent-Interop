"""Controller decision normalization.

The gateway retains tool execution authority in the coding client; this class
only validates/labels decisions before they become canonical response blocks.
"""

from __future__ import annotations

from agent_interop.controller.policy import mark_controller_provenance
from agent_interop.controller.types import ControllerAction, ControllerDecision


class CompatibilityController:
    def normalize_decision(self, decision: ControllerDecision) -> ControllerDecision:
        if decision.action not in {ControllerAction.TOOL_CALL, ControllerAction.TOOL_BATCH}:
            return decision
        calls = tuple(mark_controller_provenance(call) for call in decision.tool_calls)
        action = ControllerAction.TOOL_CALL if len(calls) == 1 else ControllerAction.TOOL_BATCH
        return ControllerDecision(
            action=action,
            text=decision.text,
            tool_calls=calls,
            primary_prompt=decision.primary_prompt,
            diagnostics=decision.diagnostics,
        )
