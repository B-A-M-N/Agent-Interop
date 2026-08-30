"""Private tool continuation loop and output firewall (P0.4/P0.5).

This module provides:
1. A bounded internal-tool loop that continues the model turn after
   Interop-internal retrieval calls (__interop_read_result, etc.).
2. An output firewall that guarantees no internal content leaks to the
   coding client across non-streaming, streaming, and controller paths.

Identity contract
------------------
The gateway builds an ``InternalIdentity`` per request from the private
loop's actual call IDs, the ContextStore projection refs exposed to the
request, and the set of internal tool names enabled on the model surface.
When an ``InternalIdentity`` is supplied the firewall enforces structural
rules against it — identity checks are authoritative and text pattern
checks are skipped.  When identity is ``None`` the legacy text-PATTERN
checks run exactly as before (backwards-compatible).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalError,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    CanonicalUsage,
)
from agent_interop.enums import RESERVED_INTERNAL_TOOL_PREFIX, ToolAuthority
from agent_interop.errors import InteropErrorCode
from agent_interop.upstreams.codec import DecodedModelResponse

# Pattern that appears in virtualized content markers
_REFS_PATTERN = re.compile(r"\[Interop (result ref|search):")

#: P0-17: bound on private continuation round-trips per presentation.
MAX_INTERNAL_TOOL_LOOP_DEPTH = 8


@dataclass(frozen=True)
class InternalIdentity:
    """Request-scoped identity for private-tool enforcement.

    Built by the gateway from the private loop's actual call IDs, the
    ContextStore refs visible to this request, and the internal tool
    names enabled on the model surface.
    """

    call_ids: frozenset[str] = frozenset()
    refs: frozenset[str] = frozenset()
    tool_names: frozenset[str] = frozenset()


def classify_tool_authority(name: str) -> ToolAuthority:
    """Classify a tool call by execution authority."""
    if name.startswith(RESERVED_INTERNAL_TOOL_PREFIX):
        return ToolAuthority.INTEROP_INTERNAL
    return ToolAuthority.CLIENT


def assert_no_internal_leakage(
    response: CanonicalResponse,
    identity: InternalIdentity | None = None,
) -> None:
    """Fail closed if any internal content reaches the public response.

    Identity mode (identity is not None):
        Enforce structurally against the request-scoped identity.
        A model quoting the Interop marker syntax in prose is NOT a leak.

    Legacy mode (identity is None):
        Fall back to text-PATTERN checks for backwards compatibility.
    """
    if identity is not None:
        # Identity mode: authoritative structural checks
        for block in response.content:
            if isinstance(block, CanonicalToolCallBlock):
                if block.name in identity.tool_names or block.name.startswith(
                    RESERVED_INTERNAL_TOOL_PREFIX
                ):
                    raise ValueError(
                        f"Output firewall: internal tool call {block.name!r} "
                        "would reach the coding client"
                    )
            if isinstance(block, CanonicalTextBlock):
                for ref in identity.refs:
                    if ref in block.text:
                        raise ValueError("Output firewall: internal ref leaked in text block")
            if isinstance(block, CanonicalToolResultBlock):
                if block.tool_call_id in identity.call_ids:
                    raise ValueError(
                        "Output firewall: private call result reached the coding client"
                    )
                text = block.content if isinstance(block.content, str) else str(block.content)
                for ref in identity.refs:
                    if ref in text:
                        raise ValueError("Output firewall: internal ref leaked in tool result")
    else:
        # Legacy mode: text-PATTERN checks (backwards-compatible)
        for block in response.content:
            if isinstance(block, CanonicalToolCallBlock):
                if block.name.startswith(RESERVED_INTERNAL_TOOL_PREFIX):
                    raise ValueError(
                        f"Output firewall: internal tool call {block.name!r} "
                        "would reach the coding client"
                    )
            if isinstance(block, CanonicalTextBlock):
                if _REFS_PATTERN.search(block.text):
                    raise ValueError(
                        "Output firewall: internal retrieval ref would reach the coding client"
                    )
            if isinstance(block, CanonicalToolResultBlock):
                text = block.content if isinstance(block.content, str) else str(block.content)
                if text.startswith(("[Interop internal error", "[Interop result ref")):
                    raise ValueError(
                        "Output firewall: internal tool result would reach the coding client"
                    )


def filter_public_blocks(
    blocks: list[CanonicalContentBlock],
    identity: InternalIdentity | None = None,
) -> list[CanonicalContentBlock]:
    """Filter response content to only include public (client-visible) blocks.

    Identity mode (identity is not None):
        Drop tool calls by name/authority, drop tool results whose
        ``tool_call_id`` is in ``identity.call_ids``, and drop text
        blocks that contain any ``identity.refs``.

    Legacy mode (identity is None):
        Drop by text-PATTERN checks, matching the original behaviour exactly.
    """
    public: list[CanonicalContentBlock] = []
    if identity is not None:
        for block in blocks:
            if isinstance(block, CanonicalToolCallBlock):
                if block.name in identity.tool_names or block.name.startswith(
                    RESERVED_INTERNAL_TOOL_PREFIX
                ):
                    continue
            if isinstance(block, CanonicalToolResultBlock):
                if block.tool_call_id in identity.call_ids:
                    continue
                text = block.content if isinstance(block.content, str) else str(block.content)
                if any(ref in text for ref in identity.refs):
                    continue
            if isinstance(block, CanonicalTextBlock):
                if any(ref in block.text for ref in identity.refs):
                    continue
            public.append(block)
    else:
        for block in blocks:
            if isinstance(block, CanonicalToolCallBlock):
                if classify_tool_authority(block.name) == ToolAuthority.INTEROP_INTERNAL:
                    continue
            if isinstance(block, CanonicalToolResultBlock):
                text = block.content if isinstance(block.content, str) else str(block.content)
                if text.startswith(("[Interop internal error", "[Interop result ref")):
                    continue
            public.append(block)
    return public


def partition_blocks_by_authority(
    blocks: list[CanonicalContentBlock],
) -> tuple[list[CanonicalToolCallBlock], list[CanonicalToolCallBlock]]:
    """Partition tool call blocks by authority.

    Returns (client_calls, internal_calls).
    """
    client: list[CanonicalToolCallBlock] = []
    internal: list[CanonicalToolCallBlock] = []
    for block in blocks:
        if not isinstance(block, CanonicalToolCallBlock):
            continue
        if classify_tool_authority(block.name) == ToolAuthority.INTEROP_INTERNAL:
            internal.append(block)
        else:
            client.append(block)
    return client, internal


def assert_model_request_invariant(invocation: Any) -> None:
    """Hard invariant: if any model-visible content mentions an Interop ref,
    the rendered upstream request MUST contain the corresponding retrieval tool.

    Inspects both CanonicalTextBlock.text and CanonicalToolResultBlock.content
    (virtualized refs live inside tool-result blocks, not just text blocks).
    """
    model_request = getattr(invocation, "model_request", None)
    if model_request is None:
        return

    has_ref = False
    for msg in model_request.messages:
        for block in msg.content:
            if isinstance(block, CanonicalTextBlock) and _REFS_PATTERN.search(block.text):
                has_ref = True
                break
            if isinstance(block, CanonicalToolResultBlock):
                content_str = (
                    block.content if isinstance(block.content, str) else str(block.content)
                )
                if _REFS_PATTERN.search(content_str):
                    has_ref = True
                    break
        if has_ref:
            break

    if not has_ref:
        return

    # If a ref is visible, __interop_read_result MUST be in the tool surface
    tool_names = {t.name for t in (model_request.tools or [])}
    if "__interop_read_result" not in tool_names:
        raise ValueError(
            "Invariant violation: model-visible content references an "
            "Interop ref but __interop_read_result is not in the tool surface"
        )


def build_private_continuation_messages(
    original_messages: list[Any],
    assistant_calls: list[CanonicalToolCallBlock],
    tool_results: list[CanonicalToolResultBlock],
) -> list[Any]:
    """Append assistant calls + tool results to the message list for
    a private continuation turn.

    This creates the messages needed to continue the model turn after
    internal tool execution.
    """
    new_messages = list(original_messages)

    # Assistant turn with tool calls
    new_messages.append(
        CanonicalMessage(
            role="assistant",
            content=list(assistant_calls),
        )
    )

    # Tool results
    if tool_results:
        new_messages.append(
            CanonicalMessage(
                role="tool",
                content=list(tool_results),
            )
        )

    return new_messages


def parse_private_arguments(candidate: Any, tool: Any) -> tuple[dict[str, Any], str | None]:
    """P0-20: strict argument parse + JSON Schema validation for a private call.

    Returns (arguments, None) or ({}, error_message). NO repair, NO
    aliasing, NO coercion beyond JSON itself: the private control plane
    is Interop-authored and the model must emit exactly the documented
    shape — a malformed call becomes an is_error result the model can
    read and correct, never a silent `{}` execution.

    Uses the same cached :class:`Draft202012Validator` pipeline as the
    public repair path so private calls honor $ref/$defs/oneOf/anyOf/
    enum/const/additionalProperties/numeric-limits.
    """
    from agent_interop.repair.schema import validate_against_schema

    raw = candidate.raw_arguments
    if isinstance(raw, str):
        try:
            arguments = json.loads(raw)
        except json.JSONDecodeError as exc:
            return {}, f"arguments are not valid JSON: {exc}"
    elif isinstance(raw, dict):
        arguments = dict(raw)
    else:
        return {}, "arguments must be a JSON object"
    if not isinstance(arguments, dict):
        return {}, "arguments must be a JSON object"
    # Full draft-2020-12 validation — types, ranges, enums, $refs, etc.
    # A bare object schema with no constraints validates as always; a
    # malformed schema returns [SchemaIssue(...)] (not []) so the model
    # sees a "schema invalid" error rather than a silent pass.
    schema = getattr(tool, "input_schema", None) or {}
    issues = validate_against_schema(arguments, schema)
    if issues:
        return {}, f"argument schema violation: {issues[0].message}"
    return arguments, None


def response_to_decoded(response: CanonicalResponse) -> DecodedModelResponse:
    """Adapt a canonical response back into the decoded-response shape the
    extraction path consumes (private continuations return canonical).

    P0-17: codec-native tool candidates ride along — without them the
    next private-loop iteration's extraction would see an empty turn and
    silently terminate the loop early.
    """
    candidates = getattr(response, "tool_candidates", None) or ()
    return DecodedModelResponse(
        content=list(response.content),
        stop_reason=response.stop_reason,
        usage=response.usage,
        tool_candidates=list(candidates),
        extra={"response_id": response.response_id} if response.response_id else {},
    )


class PrivateContinuationLoop:
    """Bounded internal-tool loop: continue the model turn privately.

    The model's turn is not finished when it requests internal retrieval:
    this loop executes the enabled internal tools against the ContextStore,
    feeds the results back as a private continuation turn, and only the
    model's NEXT turn can produce the public response. Internal call IDs
    and results never cross the client boundary (identity-based firewall).

    One instance per Gateway (holds no request-scoped state). Each
    :meth:`run` call is one request's loop.
    """

    def __init__(
        self,
        *,
        send_step: Callable[..., Awaitable[tuple[CanonicalResponse, bytes]]],
        internal_executor: Any,
        extract_candidates: Callable[[DecodedModelResponse, Any], list[Any]],
        enabled_internal_tools: Callable[[Any], dict[str, Any]],
        request_identity: Callable[[Any, set[str]], Any],
    ) -> None:
        """Collaborators are the Gateway's own bound methods — this class
        owns the LOOP, not the machinery around it (extraction, execution
        authority, identity) which stays with the Gateway."""
        self._send_step = send_step
        self._internal_executor = internal_executor
        self._extract_candidates = extract_candidates
        self._enabled_internal_tools = enabled_internal_tools
        self._request_identity = request_identity

    async def run(
        self,
        invocation: Any,
        exec_record: Any,
        *,
        internal_candidates: list[Any],
        decoded: DecodedModelResponse,
        budget: Any | None,
    ) -> CanonicalResponse | DecodedModelResponse:
        """Consume internal calls privately, then continue the model turn.

        Returns a terminal ``CanonicalResponse`` on loop/budget failure, or
        the firewalled final decoded turn for the caller's ordinary
        transaction pipeline.
        """
        from agent_interop.context_store.executor import InternalExecutionContext

        canonical = invocation.reconciled_request
        route = invocation.route
        enabled = self._enabled_internal_tools(invocation)
        if not enabled:
            # Nothing is executable — internal calls fall through to the
            # ordinary transaction layer, which rejects unknown tools.
            return CanonicalResponse()

        session_id = getattr(invocation.request_context, "session_id", "") or ""
        context = InternalExecutionContext(
            session_id=session_id,
            authorized_tools={t.name: t for t in invocation.reconciled_request.tools},
            withheld_tools=frozenset(
                getattr(invocation.tool_surface_plan, "withheld_tool_names", ()) or (),
            ),
        )

        identity_call_ids: set[str] = set()
        # P0-17: ONE non-recursive loop. Each iteration sends exactly one
        # model step through ``send_step`` (decode only, no nested
        # private-loop dispatch), so the depth bound below counts actual
        # round-trips and cannot be circumvented by re-entry.
        loop_generations = 0
        pending_internal = list(internal_candidates)
        current_decoded = decoded

        while pending_internal:
            loop_generations += 1
            if loop_generations > MAX_INTERNAL_TOOL_LOOP_DEPTH:
                break
            results: list[CanonicalToolResultBlock] = []
            assistant_calls: list[CanonicalToolCallBlock] = []

            for candidate in pending_internal:
                # P0-19: normalize missing IDs ONCE — the SAME synthetic id
                # goes into the identity set, the assistant call block, and
                # the tool result block, so the firewall's leak check and the
                # transcript can never disagree about which turn was private.
                # A blank provider ID ("") is treated as missing.
                call_id = candidate.id or f"interop_no_id_{loop_generations}_{len(results)}"
                identity_call_ids.add(call_id)
                if candidate.name not in enabled:
                    # Fail closed on a non-enabled internal tool.
                    results.append(
                        CanonicalToolResultBlock(
                            tool_call_id=call_id,
                            content=(
                                f"[Interop internal error for {candidate.name}]: "
                                "tool is not enabled for this request"
                            ),
                            is_error=True,
                        )
                    )
                    continue
                # P0-20: STRICT argument validation on the private control
                # plane — no repair, no coercion, no `{}` fallback. A
                # malformed internal call is an is_error result fed back to
                # the model, never silently repaired into a read.
                arguments, parse_error = parse_private_arguments(
                    candidate,
                    enabled[candidate.name],
                )
                if parse_error is not None:
                    results.append(
                        CanonicalToolResultBlock(
                            tool_call_id=call_id,
                            content=f"[Interop internal error for {candidate.name}]: {parse_error}",
                            is_error=True,
                        )
                    )
                    assistant_calls.append(
                        CanonicalToolCallBlock(
                            id=call_id,
                            name=candidate.name,
                            arguments={},
                        )
                    )
                    continue
                outcome = self._internal_executor.execute(
                    candidate.name,
                    arguments,
                    session_id,
                    context=context,
                )
                # P0: a ref the executor just minted (schema-on-demand pages
                # large schemas through the store) joins the request registry
                # immediately — it becomes pinnable, firewalled, and cleaned
                # up with the request instead of leaking past it.
                if outcome.ref and not outcome.is_error:
                    registry = getattr(exec_record, "ref_registry", None)
                    if registry is not None:
                        registry.register(outcome.ref)
                results.append(
                    CanonicalToolResultBlock(
                        tool_call_id=call_id,
                        content=outcome.to_model_string(),
                        is_error=outcome.is_error,
                    )
                )
                assistant_calls.append(
                    CanonicalToolCallBlock(
                        id=call_id,
                        name=candidate.name,
                        arguments=arguments,
                    )
                )

            base_request = invocation.model_request
            if base_request is None:
                return self._terminal_error(
                    canonical,
                    route,
                    InteropErrorCode.INTERNAL_ERROR,
                    "private continuation requires a resolved model_request",
                )
            continuation_messages = build_private_continuation_messages(
                base_request.messages,
                assistant_calls,
                results,
            )
            continuation_request = replace(
                base_request,
                messages=continuation_messages,
                tool_choice=CanonicalToolChoice.auto(),
            )
            continuation_invocation = replace(
                invocation,
                model_request=continuation_request,
                reconciled_request=continuation_request,
            )

            # P0-16: private generations spend against their OWN allowance —
            # never against the compatibility ladder's upstream_attempts —
            # plus the same token/latency ceilings as any other generation.
            # P0-audit: the allowance GATE lives here; the token reservation
            # itself happens exactly once inside the generation seam.
            # Pre-allocating a second reservation here orphaned it (never
            # committed/released), permanently inflating the budget with a
            # phantom estimate until every later rung failed as
            # budget-exhausted.
            if budget is not None and not budget.allow_private_generation():
                return self._terminal_error(
                    canonical,
                    route,
                    InteropErrorCode.INTERNAL_TOOL_LOOP_EXHAUSTED,
                    f"Private continuation stopped: budget exhausted ({budget.exhausted_by})",
                    details={"budget_exhausted_by": budget.exhausted_by},
                )

            # P0-17: single model step — decode only, never re-enters the
            # private-loop dispatch (which lives in the gateway's send path).
            response, _ = await self._send_step(continuation_invocation, exec_record)
            if response.error is not None:
                return self._terminal_error(
                    canonical,
                    route,
                    response.error.code,
                    f"Private continuation generation failed: {response.error.message}",
                )

            next_candidates = self._extract_candidates(
                response_to_decoded(response),
                invocation,
            )
            pending_internal = [c for c in next_candidates if c.name in enabled]
            if not pending_internal:
                current_decoded = response_to_decoded(response)

        if pending_internal:
            # Bounded loop cap exceeded — fail closed without leaking the
            # partially-retrieved internal state.
            exec_record.record_compatibility_event("internal_tool_loop_exhausted")
            return self._terminal_error(
                canonical,
                route,
                InteropErrorCode.INTERNAL_TOOL_LOOP_EXHAUSTED,
                (
                    "Model exceeded the bounded internal-tool loop depth "
                    f"({MAX_INTERNAL_TOOL_LOOP_DEPTH})"
                ),
                details={"max_internal_tool_loop_depth": MAX_INTERNAL_TOOL_LOOP_DEPTH},
            )

        identity = self._request_identity(invocation, identity_call_ids)
        public_blocks = filter_public_blocks(list(current_decoded.content), identity=identity)
        # P0-17: the final turn's codec-native PUBLIC candidates ride along —
        # under native protocols tool calls live in `tool_candidates`, not in
        # `content`, so dropping them here would make the caller see an empty
        # final turn and silently drop a legitimate public tool call.
        public_candidates = [
            c
            for c in getattr(current_decoded, "tool_candidates", ()) or ()
            if c.name not in enabled
        ]
        checked = CanonicalResponse(
            content=public_blocks,
            stop_reason=current_decoded.stop_reason,
            usage=current_decoded.usage,
            model=CanonicalModelReference(
                requested_name=canonical.model.requested_name,
                resolved_name=route.upstream_model,
            ),
            request_id=canonical.request_id,
        )
        assert_no_internal_leakage(checked, identity=identity)
        # Hand the firewalled final turn content back to the caller, which
        # re-derives public candidates from it and continues through the
        # ordinary transaction pipeline (validation/repair/assembly) — the
        # private loop NEVER bypasses client-authority checks.
        return DecodedModelResponse(
            content=list(checked.content),
            stop_reason=checked.stop_reason,
            usage=checked.usage,
            tool_candidates=list(public_candidates),
            extra={"response_id": checked.response_id} if checked.response_id else {},
        )

    @staticmethod
    def _terminal_error(
        canonical: Any,
        route: Any,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> CanonicalResponse:
        return CanonicalResponse(
            content=[],
            stop_reason=CanonicalStopReason.END_TURN,
            usage=CanonicalUsage(),
            model=CanonicalModelReference(
                requested_name=canonical.model.requested_name,
                resolved_name=route.upstream_model,
            ),
            error=CanonicalError(code=code, message=message, details=details or {}),
        )
