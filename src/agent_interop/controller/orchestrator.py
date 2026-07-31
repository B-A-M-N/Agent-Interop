"""Controller decision normalization.

The gateway retains tool execution authority in the coding client; this class
only validates/labels decisions before they become canonical response blocks.
"""

from __future__ import annotations

from collections.abc import Sequence

from agent_interop.abi import CanonicalContentBlock, CanonicalTextBlock, CanonicalToolCallBlock
from agent_interop.controller.policy import (
    CONTROLLER_DELEGATE_TOOL_NAME,
    mark_controller_provenance,
    primary_delegation_prompt,
)
from agent_interop.controller.types import ControllerAction, ControllerDecision


class CompatibilityController:
    def decide(self, content: Sequence[CanonicalContentBlock]) -> ControllerDecision:
        """Classify a controller response before it reaches the client.

        The private delegation call is control flow, not a client-visible
        tool.  A controller cannot mix it with real client calls: that would
        make it unclear whether the client is authorised to act before the
        worker refinement is available, so it is an explicit failure.
        """
        calls = tuple(block for block in content if isinstance(block, CanonicalToolCallBlock))
        private_calls = tuple(
            call for call in calls if call.name == CONTROLLER_DELEGATE_TOOL_NAME
        )
        if private_calls:
            if len(private_calls) != 1 or len(calls) != 1:
                return ControllerDecision(
                    action=ControllerAction.FAIL,
                    diagnostics=("mixed_private_delegation_and_client_calls",),
                )
            prompt = primary_delegation_prompt(private_calls[0])
            if prompt is None:
                return ControllerDecision(
                    action=ControllerAction.FAIL,
                    diagnostics=("invalid_private_delegation",),
                )
            return ControllerDecision(
                action=ControllerAction.DELEGATE_PRIMARY,
                primary_prompt=prompt,
            )
        if calls:
            action = ControllerAction.TOOL_CALL if len(calls) == 1 else ControllerAction.TOOL_BATCH
            return self.normalize_decision(ControllerDecision(action=action, tool_calls=calls))
        return ControllerDecision(
            action=ControllerAction.FINAL_TEXT,
            text="\n".join(
                block.text for block in content if isinstance(block, CanonicalTextBlock)
            ),
        )

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
