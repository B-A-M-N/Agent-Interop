"""Gateway-owned repair-generation adapter (P0-wire-regeneration).

Connects the transaction service's ``regenerate_fn`` contract to the ONE
model-generation seam.  The hidden repair generation is a REAL generation:
it reserves against the request's AttemptBudget, passes through admission
control, records telemetry, and shares every ceiling an ordinary worker
turn pays — under purpose ``"tool_repair"`` so per-purpose accounting
shows exactly why a second model call happened.

Coupling contract: no request-scoped state; ``make_regenerate_fn`` closes
over the one invocation and execution record it serves, and resolves the
seam through the ``gateway`` back-reference on every call (tests replace
gateway internals after construction).

The correction travels as the seam's ``attempt_request`` — the seam was
built to render whichever request it is handed and never inspects tools —
so the wire shape is always protocol-correct through the codec's own
``render_request``, never a hand-rolled body.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalToolChoice,
)

logger = logging.getLogger("agent_interop.repair.adapter")

__all__ = ["RepairGenerationAdapter", "make_regenerate_fn"]


class RepairGenerationAdapter:
    """Dispatch one constrained correction generation through the seam."""

    def __init__(self, *, gateway: Any) -> None:
        self._gateway = gateway

    def build_repair_request(
        self,
        invocation: Any,
        *,
        correction_prompt: str,
    ) -> CanonicalRequest:
        """The original model-facing surface minus tools, plus the prompt.

        Tools are stripped deliberately: the correction asks for a plain
        corrected argument object as TEXT, and re-offering the tool array
        invites the model to answer with a second tool call instead.
        Streaming stays off — the transaction service is a synchronous
        repair step, not a user-visible stream.
        """
        canonical = invocation.reconciled_request
        return CanonicalRequest(
            model=CanonicalModelReference(
                requested_name=canonical.model.requested_name,
            ),
            system=list(canonical.system),
            messages=[
                *canonical.messages,
                CanonicalMessage(
                    role="user",
                    content=[CanonicalTextBlock(text=correction_prompt)],
                ),
            ],
            tools=[],
            tool_choice=CanonicalToolChoice.none(),
            # Bounded: a corrected argument object is small.  The reserve is
            # a ceiling on runaway repair output, not a target.
            generation=CanonicalGenerationOptions(
                max_output_tokens=min(
                    1024,
                    max(256, canonical.generation.max_output_tokens),
                ),
                stream=False,
            ),
        )

    async def generate(
        self,
        invocation: Any,
        exec_record: Any,
        *,
        correction_prompt: str,
    ) -> str:
        """Run one hidden repair generation for ``invocation``'s request.

        Dispatches through the same seam as every other generation —
        budget reservation, admission control, the exact-context gate, and
        telemetry all apply — accounted under ``purpose="tool_repair"``.

        Returns the model's text answer (empty on any failure).  An empty
        string is the orchestrator's "no usable correction" signal, so a
        failed repair generation degrades to deterministic-repair-only
        without ever surfacing an error to the client.
        """
        repair_request = self.build_repair_request(
            invocation, correction_prompt=correction_prompt,
        )
        response, _ = await self._gateway._send_one_model_step(
            invocation,
            exec_record,
            purpose="tool_repair",
            attempt_request=repair_request,
        )
        if response.error is not None:
            logger.warning(
                "tool_repair generation failed: %s", response.error.code,
            )
            return ""
        return "\n".join(
            block.text
            for block in response.content
            if isinstance(block, CanonicalTextBlock)
        )


def make_regenerate_fn(
    gateway: Any,
    invocation: Any,
    exec_record: Any,
) -> Any:
    """Build the request-bound ``regenerate_fn`` for one tool batch.

    Returns None when the request's repair policy disallows regeneration —
    the transaction service treats a None callback as "deterministic
    repair only" and never attempts a hidden generation, so AUTO-mode
    requests (``max_regenerations=0`` by default) never pay a second
    model call.
    """
    policy = getattr(invocation, "repair_policy", None)
    if policy is None or policy.max_regenerations <= 0:
        return None

    adapter = RepairGenerationAdapter(gateway=gateway)

    async def regenerate(correction_prompt: str) -> str:
        return await adapter.generate(
            invocation,
            exec_record,
            correction_prompt=correction_prompt,
        )

    return regenerate
