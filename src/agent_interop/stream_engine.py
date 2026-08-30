"""The streaming request engine (extracted from Gateway).

One class owns the stream frame loop and everything downstream of "the
prepared invocation is handed to streaming": admission slot, transport
stream, frame decoding, tool-fragment accumulation, end-of-turn batch
decision, event emission, and the stream tail's budget reconciliation.

The Gateway keeps the ORCHESTRATION around streaming (preparation,
qualification, ref-registry pinning, buffered-vs-direct dispatch) and this
engine owns the STREAM itself. The render → serialize-once → exact-context
gate prologue is NOT duplicated here — it is the generation seam's
``prepare(stream=True)``, injected as ``prepare_generation``.

Coupling contract (mirrors GenerationSeam / PrivateContinuationLoop):

* takes the per-request state it reads/writes as explicit arguments
  (``invocation``, ``exec_record``);
* owns NO request-scoped state — one instance per Gateway is safe for
  concurrent streams;
* the gateway's policy hooks (transaction context, session/decision
  recording, evidence write-back, internal-tool authority, private loop)
  arrive as constructor callables so this module stays mechanism, not
  policy.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalError,
    CanonicalEvent,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalUsage,
    RawToolCallCandidate,
)
from agent_interop.errors import InteropErrorCode, classify_http_status
from agent_interop.repair.invocation import StreamExtractionMode
from agent_interop.streaming.coordinator import (
    PendingToolCall,
    StreamCoordinator,
    StreamLimits,
    ToolCallLimitExceeded,
    ToolStreamKey,
)
from agent_interop.transaction import ToolBatchPolicy, process_tool_batch
from agent_interop.transport.http import PreparedUpstreamRequest
from agent_interop.transport.ndjson import MalformedNDJSONLine
from agent_interop.upstreams.codec import (
    DecodedModelResponse,
    DecodedStreamComplete,
    DecodedStreamError,
    DecodedTextDelta,
    DecodedToolBatchComplete,
    DecodedToolCallComplete,
    DecodedToolFragment,
    DecodedUsageUpdate,
)

logger = logging.getLogger("agent_interop.stream_engine")

__all__ = ["StreamEngine"]


class StreamEngine:
    """Frame loop → accumulate → atomic batch → events, once per stream."""

    def __init__(
        self,
        *,
        config: Any,
        admission_controller: Any,
        transport_provider: Callable[[], Any],
        prepare_generation: Callable[..., Awaitable[Any]],
        context_limit_error: Callable[[Any, str], Any],
        build_upstream_headers: Callable[..., dict[str, str]],
        disabled_tool_choice_conflict: Callable[[Any], Any],
        extract_tool_candidates: Callable[[Any, Any], list[Any]],
        dedup_tool_candidates: Callable[[list[Any], list[Any]], list[Any]],
        enabled_internal_tools: Callable[[Any], dict[str, Any]],
        private_loop: Any,
        build_transaction_context: Callable[[Any, Any], Any],
        record_repairs_to_session: Callable[[Any, Any], None],
        record_tool_decisions: Callable[[Any, Any], None],
        record_evidence_observation: Callable[[Any, Any], None],
        build_batch_rejection_error: Callable[[Any, str], Any],
        record_stream_safety: Callable[[Any, bool], None] | None = None,
    ) -> None:
        self._config = config
        self._admission_controller = admission_controller
        # Resolved per dispatch — the gateway's transport honors test
        # injection via a lazy property, so it must never be captured here.
        self._transport_provider = transport_provider
        self._prepare_generation = prepare_generation
        self._context_limit_error = context_limit_error
        self._build_upstream_headers = build_upstream_headers
        self._disabled_tool_choice_conflict = disabled_tool_choice_conflict
        self._extract_tool_candidates = extract_tool_candidates
        self._dedup_tool_candidates = dedup_tool_candidates
        self._enabled_internal_tools = enabled_internal_tools
        self._private_loop = private_loop
        self._build_transaction_context = build_transaction_context
        self._record_repairs_to_session = record_repairs_to_session
        self._record_tool_decisions = record_tool_decisions
        self._record_evidence_observation = record_evidence_observation
        self._build_batch_rejection_error = build_batch_rejection_error
        # P0-7: optional stream-safety recorder — (invocation, accepted).
        # The gateway injects a closure that records/revokes the serving
        # tuple's observation; None keeps the cache inert for direct
        # engine constructions in tests.
        self._record_stream_safety = record_stream_safety

    # ─── Entry ────────────────────────────────────────────────────────────

    async def run_send_stream(
        self,
        invocation: Any,
        exec_record: Any,
    ) -> AsyncIterator[Any]:
        """Send a prepared invocation upstream and stream decoded events."""
        # Unsafe history — finalize BEFORE yielding the terminal event. The
        # server stops consuming this generator as soon as it sees
        # message_stop, so any bookkeeping placed after that yield may never
        # run through the real ASGI path.
        if invocation.invocation_plan is None or invocation.codec is None:
            err = CanonicalError(
                code=InteropErrorCode.HISTORY_UNSAFE,
                message="History reconciliation detected unsafe history",
            )
            exec_record.finalize_error(err)
            yield CanonicalEvent(type="error", error=err)
            yield CanonicalEvent(type="message_stop")
            return

        route = invocation.route
        plan = invocation.invocation_plan
        codec = invocation.codec
        canonical = invocation.reconciled_request

        choice_conflict = self._disabled_tool_choice_conflict(plan)
        if choice_conflict is not None:
            exec_record.finalize_error(choice_conflict)
            yield CanonicalEvent(type="error", error=choice_conflict)
            yield CanonicalEvent(type="message_stop")
            return

        # P0-1/P0-10/P0-26: the SAME render → serialize-once → exact-context
        # gate as every other generation — the generation seam owns it.
        prepared = await self._prepare_generation(invocation, stream=True)
        if isinstance(prepared, CanonicalResponse):
            exec_record.finalize_error(prepared.error)
            yield CanonicalEvent(type="error", error=prepared.error)
            yield CanonicalEvent(type="message_stop")
            return
        _request_local, rendered_bytes, _meter, _reserve, rendered = prepared

        budget = getattr(exec_record, "attempt_budget", None)
        reservation = None
        if budget is not None:
            if not budget.reserve_rendered_bytes(len(rendered_bytes)) or not budget.reserve_input(
                _meter.input_tokens,
            ):
                error = self._context_limit_error(invocation, "max_total_rendered_bytes")
                exec_record.finalize_error(error.error)
                yield CanonicalEvent(type="error", error=error.error)
                yield CanonicalEvent(type="message_stop")
                return
            # P0: stream generations spend against the request's reservation
            # ledger; commit/reconcile happens at terminal EOF below.
            reservation = budget.reserve_generation(
                estimated_input_tokens=_meter.input_tokens,
                output_reserve_tokens=_reserve,
                purpose="worker",
            )
            if reservation is None:
                error = self._context_limit_error(
                    invocation,
                    f"generation budget exhausted ({budget.exhausted_by})",
                )
                exec_record.finalize_error(error.error)
                yield CanonicalEvent(type="error", error=error.error)
                yield CanonicalEvent(type="message_stop")
                return

        # Build typed upstream request
        upstream_request = PreparedUpstreamRequest(
            method="POST",
            url=f"{route.upstream.base_url}{codec.endpoint_path()}",
            headers=self._build_upstream_headers(
                route,
                client_headers=dict(invocation.request_context.forwardable_transport_headers),
                codec_headers=codec.required_headers(),
            ),
            # P0-6: body kept for diagnostics only; transport sends
            # serialized_body verbatim.
            body=rendered,
            stream=True,
            timeout_seconds=route.upstream.timeout_seconds,
            serialized_body=rendered_bytes,
        )

        coordinator = StreamCoordinator(
            route.upstream.wire_protocol,
            limits=StreamLimits(
                max_accumulated_arg_bytes=self._config.max_tool_argument_bytes,
                max_simultaneous_tool_calls=self._config.max_simultaneous_tool_calls,
            ),
        )

        # P0-15: stream dispatch passes through the SAME admission
        # architecture as the non-streaming send path — every generation
        # (public, private continuation, controller, probe) holds a slot for
        # the duration of its transport. P0-28: the typed admission result
        # is preserved — queue-full and timeout are distinct canonical
        # errors, and ALL decoding stays inside the transport context so the
        # stream object can never outlive its context manager.
        async with self._admission_controller.generation_slot(
            route.upstream.base_url,
            route.upstream_model,
        ) as slot:
            if not getattr(slot, "acquired", False):
                result_kind = getattr(getattr(slot, "result", None), "value", "unavailable")
                canonical_error = CanonicalError(
                    code=(
                        InteropErrorCode.BACKEND_UNAVAILABLE
                        if result_kind == "queue_full"
                        else InteropErrorCode.BACKEND_TIMEOUT
                    ),
                    message=(
                        "Backend admission control: too many queued generations"
                        if result_kind == "queue_full"
                        else "Backend admission control: generation slot timed out"
                    ),
                    details={"admission": result_kind, "retryable": True},
                )
                # P0: never opened the transport — release reservation.
                if reservation is not None:
                    reservation.release()
                exec_record.finalize_error(canonical_error)
                yield CanonicalEvent(type="error", error=canonical_error)
                yield CanonicalEvent(type="message_stop")
                return

            async with self._transport_provider().stream(upstream_request) as stream:
                if stream.status_code >= 400:
                    # Read a short excerpt of the error body for the message
                    excerpt = ""
                    try:
                        parts: list[str] = []
                        total = 0
                        async for raw in stream.raw_lines():
                            parts.append(raw)
                            total += len(raw)
                            if total >= 200 or len(parts) >= 5:
                                break
                        excerpt = "".join(parts)[:200]
                    except Exception:
                        excerpt = ""
                    error_msg = f"Upstream returned {stream.status_code}"
                    if excerpt:
                        error_msg += f": {excerpt}"
                    canonical_error = CanonicalError(
                        code=classify_http_status(stream.status_code), message=error_msg
                    )
                    # P0: dispatched to backend without usage — keep the
                    # estimate (conservative), never release.
                    if reservation is not None:
                        reservation.commit_estimated()
                        budget_ = getattr(exec_record, "attempt_budget", None)
                        if budget_ is not None:
                            budget_.record_rendered_bytes(len(rendered_bytes))
                    exec_record.finalize_error(canonical_error)
                    yield CanonicalEvent(type="error", error=canonical_error)
                    yield CanonicalEvent(type="message_stop")
                    return

                # The invocation plan decides how tool calls are extracted. For
                # PROMPTED-mode local models stream_extraction_mode is
                # BUFFER_TEXTUAL_RESPONSE: raw <tool_call>...</tool_call> envelopes
                # must be buffered until the response is complete, then run through
                # the SAME textual-extraction machinery as the non-streaming path —
                # never streamed straight through as text. Any other mode uses the
                # native-fragments path (text streams through immediately).
                mode = plan.stream_extraction_mode
                buffered_text_parts: list[str] = []
                final_stop_reason: CanonicalStopReason | None = None
                final_usage: CanonicalUsage | None = None
                malformed_frame_count = 0

                # Decode stream frames through codec
                async for frame_data, raw_frame_text in self._iter_frame_data(
                    stream, codec.stream_framing
                ):
                    if frame_data is None:
                        # Malformed frame. Record a bounded, sanitized diagnostic
                        # regardless of outcome so evidence/replay can see how
                        # often a backend emits unparseable frames.
                        malformed_frame_count += 1
                        exec_record.record_malformed_frame(
                            malformed_frame_count,
                            "unparseable_frame",
                            raw_frame_text,
                        )
                        if coordinator.has_pending_tool_calls:
                            # Fail the tool batch and terminate
                            coordinator.tool_accumulator.fail_all_pending("malformed_frame")
                            err = CanonicalError(
                                code="MALFORMED_FRAME",
                                message="Malformed frame with open tool state",
                            )
                            # GAP 5 FIX — a malformed frame with open tool state is a
                            # terminal error; the record must be finalized as FAILED
                            # rather than left permanently ACTIVE. Finalize BEFORE
                            # yielding message_stop — the server stops consuming
                            # this generator as soon as it sees that event.
                            exec_record.finalize_error(err)
                            yield CanonicalEvent(type="error", error=err)
                            yield CanonicalEvent(type="message_stop")
                            return
                        if malformed_frame_count > self._config.max_malformed_stream_frames:
                            # No open tool state, but the backend has now sent
                            # more malformed frames than the configured threshold.
                            # Without a bound, a backend that never resends valid
                            # frames could be silently `continue`d forever.
                            err = CanonicalError(
                                code="MALFORMED_FRAME",
                                message=(
                                    f"Too many malformed stream frames "
                                    f"({malformed_frame_count} > "
                                    f"{self._config.max_malformed_stream_frames})"
                                ),
                            )
                            exec_record.finalize_error(err)
                            yield CanonicalEvent(type="error", error=err)
                            yield CanonicalEvent(type="message_stop")
                            return
                        continue

                    # GAP 2 FIX — always decode the frame FIRST, even when it is
                    # the terminal frame. The terminal frame often carries the final
                    # tool fragments, usage, and/or stop reason; checking
                    # is_stream_complete first and returning early silently dropped
                    # all of that. Decode now, then consult is_stream_complete below.
                    is_complete = codec.is_stream_complete(frame_data)
                    decoded_events: list[Any] = codec.decode_stream_chunk(frame_data)

                    stream_error: CanonicalError | None = None
                    for decoded_event in decoded_events:
                        # Handle text deltas. In BUFFER_TEXTUAL_RESPONSE mode the
                        # model emits raw tool envelopes as text, so we must buffer
                        # and extract later — never yield the envelope literally.
                        if isinstance(decoded_event, DecodedTextDelta):
                            if mode == StreamExtractionMode.BUFFER_TEXTUAL_RESPONSE:
                                buffered_text_parts.append(decoded_event.text)
                            else:
                                yield CanonicalEvent(
                                    type="text_delta",
                                    index=0,
                                    partial=decoded_event.text,
                                )

                        # Handle tool batch completion — complete all pending calls
                        elif isinstance(decoded_event, DecodedToolBatchComplete):
                            coordinator.tool_accumulator.complete_all_pending()

                        # Handle tool fragments - accumulate via the accumulator's
                        # feed methods so size limits are actually enforced.
                        elif isinstance(decoded_event, DecodedToolFragment):
                            key = ToolStreamKey(
                                choice_index=decoded_event.choice_index,
                                tool_index=decoded_event.tool_index,
                            )
                            coordinator.tool_accumulator.start_call(
                                key, decoded_event.call_id_fragment or None
                            )
                            if decoded_event.name_fragment:
                                coordinator.tool_accumulator.feed_name(
                                    key, decoded_event.name_fragment
                                )
                            if decoded_event.argument_fragment:
                                # GAP 3 FIX — route argument fragments through
                                # feed_arguments so max_accumulated_arg_bytes is
                                # enforced. The old direct-append path bypassed the
                                # limit entirely. A limit breach is a terminal error.
                                try:
                                    coordinator.tool_accumulator.feed_arguments(
                                        key, decoded_event.argument_fragment
                                    )
                                except ToolCallLimitExceeded as exc:
                                    err = CanonicalError(
                                        code="TOOL_CALL_LIMIT_EXCEEDED",
                                        message=str(exc),
                                    )
                                    exec_record.finalize_error(err)
                                    yield CanonicalEvent(type="error", error=err)
                                    yield CanonicalEvent(type="message_stop")
                                    return

                        # Handle per-call completion
                        elif isinstance(decoded_event, DecodedToolCallComplete):
                            key = ToolStreamKey(
                                choice_index=decoded_event.choice_index,
                                tool_index=decoded_event.tool_index,
                            )
                            coordinator.tool_accumulator.complete_call(key)

                        # GAP 2 — capture usage updates from terminal frames instead
                        # of silently dropping them.
                        elif isinstance(decoded_event, DecodedUsageUpdate):
                            final_usage = decoded_event.usage

                        # Capture the stream's stop reason and any trailing usage.
                        elif isinstance(decoded_event, DecodedStreamComplete):
                            final_stop_reason = decoded_event.stop_reason
                            if decoded_event.usage is not None:
                                final_usage = decoded_event.usage

                        # Surface a stream-level error as a terminal error event.
                        elif isinstance(decoded_event, DecodedStreamError):
                            stream_error = CanonicalError(
                                code="BACKEND_STREAM_ERROR",
                                message=decoded_event.error,
                            )

                    if stream_error is not None:
                        exec_record.finalize_error(stream_error)
                        yield CanonicalEvent(type="error", error=stream_error)
                        yield CanonicalEvent(type="message_stop")
                        return

                    # GAP 4 FIX — the mid-loop drain_completed() + immediate
                    # _process_completed_stream_tools(...) call that used to live
                    # here is DELETED. Completed tool calls now accumulate in the
                    # coordinator until end-of-turn (below), so the whole turn's
                    # tool calls are validated as ONE atomic batch instead of being
                    # split across per-drain decisions.

                    if is_complete:
                        break

                # ── End-of-turn (shared by terminal-frame break AND natural loop-end) ──
                # P1-H: the upstream transport is exhausted — the backend is no
                # longer doing work on our behalf, so release the admission slot
                # NOW instead of holding it through tool-batch processing and the
                # (client-backpressured) event-yield tail. A slow client must not
                # pin a backend generation slot. The context-manager exit below
                # is idempotent, so error paths that already returned still
                # released exactly once.
                slot.release_now()
                # Native fragments are accumulated during the loop regardless of the
                # plan's extraction mode (the codec emits them from the wire stream,
                # which does not depend on the plan). So always finish any still-
                # pending calls and drain the WHOLE turn's completed calls. In BUFFER
                # mode a prompted model emits its tool envelopes as text deltas rather
                # than native fragments — but a model can also hallucinate native-style
                # tool_calls deltas even when PROMPTED mode stripped tool schemas from
                # the request, so the two sources are NOT guaranteed mutually exclusive.
                # To honour the turn-level atomicity guarantee, BOTH candidate sources
                # are merged into ONE deduped list (via _dedup_tool_candidates) and
                # decided as a single atomic batch. In non-BUFFER mode textual
                # candidates are empty, so this collapses to the single native batch
                # exactly as before.
                coordinator.tool_accumulator.complete_all_pending()
                remaining = coordinator.tool_accumulator.drain_completed()

                if mode == StreamExtractionMode.BUFFER_TEXTUAL_RESPONSE:
                    # GAP 1 FIX — run the buffered raw text through the SAME
                    # textual-extraction machinery as the non-streaming path. The
                    # envelopes are consumed here so the literal <tool_call> text
                    # never leaks to the client as a text delta. Native fragments (if
                    # any) are merged with the textual candidates below so the WHOLE
                    # turn is decided as ONE atomic batch instead of two.
                    native_candidates = self._pending_to_candidates(remaining, invocation)
                    textual_candidates: list[Any] = []
                    remaining_content: list[CanonicalContentBlock] = []
                    buffered_text = "".join(buffered_text_parts)
                    if buffered_text:
                        decoded = DecodedModelResponse(
                            content=[CanonicalTextBlock(text=buffered_text)],
                        )
                        textual_candidates = self._extract_tool_candidates(decoded, invocation)
                        remaining_content = list(decoded.content)

                    merged = self._dedup_tool_candidates(native_candidates, textual_candidates)
                    # P0.4: internal retrieval calls mean the model's turn is not
                    # finished — run the private continuation (via the non-streaming
                    # send path) and let its firewalled FINAL turn become the only
                    # emission source. The first turn's buffered envelope text and
                    # candidates are discarded: they are pre-retrieval state.
                    enabled_internal = self._enabled_internal_tools(invocation)
                    internal_candidates = [c for c in merged if c.name in enabled_internal]
                    if internal_candidates:
                        continued = await self._private_loop.run(
                            invocation,
                            exec_record,
                            internal_candidates=internal_candidates,
                            decoded=DecodedModelResponse(
                                content=[CanonicalTextBlock(text=buffered_text)]
                                if buffered_text
                                else [],
                            ),
                            budget=getattr(exec_record, "attempt_budget", None),
                        )
                        if isinstance(continued, CanonicalResponse):
                            # Terminal: loop/budget failure — already firewalled.
                            if continued.error is not None:
                                exec_record.finalize_error(continued.error)
                                yield CanonicalEvent(type="error", error=continued.error)
                                yield CanonicalEvent(
                                    type="message_stop",
                                    stop_reason=CanonicalStopReason.INVALID_OUTPUT,
                                )
                                return
                        else:
                            # The firewalled final turn replaces every emission
                            # source: text, candidates, and usage.
                            buffered_text = "".join(
                                block.text
                                for block in continued.content
                                if isinstance(block, CanonicalTextBlock)
                            )
                            remaining_content = [
                                block
                                for block in continued.content
                                if isinstance(block, CanonicalTextBlock)
                            ]
                            merged = self._extract_tool_candidates(continued, invocation)
                            if continued.usage is not None:
                                final_usage = continued.usage
                    merged = [c for c in merged if c.name not in enabled_internal]
                    if merged:
                        transaction_context = self._build_transaction_context(invocation, canonical)
                        batch_decision = await process_tool_batch(
                            merged,
                            canonical.tools,
                            context=transaction_context,
                            policy=ToolBatchPolicy(invocation.repair_policy.batch_policy),
                        )
                        self._record_repairs_to_session(batch_decision, invocation.request_context)
                        # Record per-call decisions onto the shared execution record
                        # unconditionally — the in-memory record is always populated
                        # (so finalize_response's outcome classification sees the
                        # decisions), matching the non-streaming path and the
                        # non-BUFFER streaming path. Evidence-store write-back
                        # remains a separate, opt-in step.
                        self._record_tool_decisions(batch_decision, invocation.execution_record)
                        # Emit the remaining (non-envelope) text, then the accepted
                        # tool calls, exactly as the non-streaming path assembles them.
                        # Plain text is emitted even on rejection — _assemble_response
                        # keeps content text blocks alongside a set .error.
                        for block in remaining_content:
                            if isinstance(block, CanonicalTextBlock) and block.text:
                                yield CanonicalEvent(type="text_delta", index=0, partial=block.text)
                        # Emit the decided batch: either the accepted tool_use blocks,
                        # or — on a fully-rejected batch — a structured error +
                        # INVALID_OUTPUT message_stop (mirrors _assemble_response's
                        # non-streaming handling). The shared helper also finalizes the
                        # record as failed and marks the coordinator's turn rejected so
                        # the caller skips its generic end-of-turn tail.
                        async for event in self._emit_batch_decision_events(
                            batch_decision,
                            canonical.request_id,
                            coordinator,
                            invocation.execution_record,
                        ):
                            yield event
                        # P0-7: a fully-accepted unbuffered tool batch is the
                        # one observation that unlocks later streams of this
                        # tuple; a rejected batch revokes it. Runs on BOTH
                        # buffered and direct paths — a buffered path that
                        # just accepted a batch is the same model behavior.
                        if self._record_stream_safety is not None:
                            self._record_stream_safety(
                                invocation,
                                (
                                    batch_decision.is_accepted
                                    and bool(batch_decision.accepted_blocks)
                                ),
                            )
                    else:
                        # No candidates extracted but there is plain text — yield it.
                        if buffered_text:
                            yield CanonicalEvent(type="text_delta", index=0, partial=buffered_text)
                else:
                    # Non-BUFFER mode: native fragments only (text already streamed as
                    # it arrived). Decide them as one atomic batch via the single-source
                    # path — byte-for-byte identical to the pre-fix behaviour.
                    if remaining:
                        async for event in self._process_completed_stream_tools(
                            remaining,
                            invocation,
                            coordinator,
                        ):
                            yield event

                # A fully-rejected batch already emitted its terminal error +
                # message_stop(INVALID_OUTPUT) and finalized the record as failed via
                # the shared helper. Skip the generic end-of-turn tail (stop-reason
                # computation, a second message_stop, evidence write-back, and
                # finalize_response) — mirroring the non-streaming path's
                # `result.error is None` gate on evidence write-back.
                if coordinator.turn_rejected:
                    return

                # GAP 6 FIX — read the public property instead of the private attr.
                #
                # coordinator.has_emitted_tool_calls MUST win over whatever the
                # backend's terminal frame reported. Found via a real live-client
                # run: Ollama (gpt-oss:20b-cloud) streams the tool_calls fragment
                # in a non-terminal chunk, then closes with a `done:true` frame
                # whose own done_reason is "stop" (mapped to END_TURN) and no
                # tool_calls of its own — decode_stream_chunk has no cross-chunk
                # state, so it can only see that one frame and reports END_TURN
                # as final_stop_reason, silently overriding the correct TOOL_CALL
                # signal `coordinator` already recorded from the earlier chunk. A
                # response containing an emitted tool_use block is a protocol
                # invariant that must report stop_reason=tool_use regardless of
                # what the backend's last frame claimed — this mirrors the
                # equivalent guard already applied on the non-streaming path.
                stop_reason = (
                    CanonicalStopReason.TOOL_CALL
                    if coordinator.has_emitted_tool_calls
                    else (final_stop_reason or CanonicalStopReason.END_TURN)
                )

                # Live evidence write-back at clean stream end: only when tools were
                # offered and an evidence store was injected. This shared end-of-turn
                # path is reached by BOTH the terminal-frame break and the natural
                # loop-end, so write-back fires on both success exits.
                #
                # IMPORTANT: this tail runs BEFORE the terminal message_stop is
                # yielded. The server stops consuming this generator as soon as it
                # sees message_stop, so bookkeeping placed after that yield may
                # never execute through the real ASGI path (only direct-generator
                # tests would see it run). If finalization itself fails, surface
                # it as a genuine terminal error instead of silently completing.
                try:
                    # P0: reconcile the generation reservation at the SAME tail
                    # that finalizes the record — actual usage from the backend's
                    # final frame replaces the estimate, accounting for rendered
                    # bytes lands here too. Streams MUST commit before message_stop
                    # so per-purpose generation counts reflect what actually ran.
                    if reservation is not None:
                        if final_usage is not None:
                            reservation.commit(
                                actual_input_tokens=int(
                                    getattr(final_usage, "input_tokens", 0) or 0,
                                ),
                                actual_output_tokens=int(
                                    getattr(final_usage, "output_tokens", 0) or 0,
                                ),
                            )
                        else:
                            reservation.commit_estimated()
                        budget_ = getattr(exec_record, "attempt_budget", None)
                        if budget_ is not None:
                            budget_.record_rendered_bytes(len(rendered_bytes))
                    # Live evidence write-back: only when tools were offered
                    # (the injected hook is the gateway's opt-in recorder; the
                    # tools guard lives here because it is stream-tail policy).
                    if canonical.tools:
                        self._record_evidence_observation(invocation, exec_record)

                    # GAP 5 FIX — the terminal-frame completion path (the most common
                    # one for OpenAI/Ollama streams) used to return WITHOUT finalizing
                    # the record. Unifying both exits into this shared path means a
                    # normal completion now finalizes as SUCCEEDED.
                    exec_record.finalize_response(
                        CanonicalResponse(usage=final_usage or CanonicalUsage())
                    )
                except Exception as exc:
                    logger.warning(
                        "stream finalization failed before the terminal event",
                        exc_info=True,
                    )
                    err = CanonicalError(code="STREAM_ERROR", message=str(exc))
                    exec_record.finalize_error(err)
                    yield CanonicalEvent(type="error", error=err)
                    yield CanonicalEvent(
                        type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT
                    )
                    return

                if final_usage is not None:
                    yield CanonicalEvent(
                        type="usage_update",
                        input_tokens=final_usage.input_tokens,
                        output_tokens=final_usage.output_tokens,
                    )

                yield CanonicalEvent(type="message_stop", stop_reason=stop_reason)

    # ─── Frame machinery ──────────────────────────────────────────────────

    async def _emit_batch_decision_events(
        self,
        batch_decision: Any,
        request_id: str,
        coordinator: StreamCoordinator,
        exec_record: Any,
    ) -> AsyncIterator[CanonicalEvent]:
        """Emit canonical events for a decided tool batch.

        Fully rejected atomic batch (no calls accepted): emits an ``error``
        event with the structured rejection (mirrors ``_assemble_response``'s
        non-streaming handling), then ``message_stop(INVALID_OUTPUT)``,
        finalizes the execution record as failed, and marks the coordinator's
        turn as rejected so the caller skips its own generic end-of-turn
        handling (stop-reason computation, second message_stop, and evidence
        write-back). Accepted / partially-accepted batches emit each accepted
        tool_use block, unchanged from prior behavior.
        """
        if not batch_decision.is_accepted and not batch_decision.accepted_blocks:
            rejection_error = self._build_batch_rejection_error(batch_decision, request_id)
            exec_record.finalize_error(rejection_error)
            coordinator.mark_turn_rejected()
            yield CanonicalEvent(type="error", error=rejection_error)
            yield CanonicalEvent(
                type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT
            )
            return

        for accepted_block in batch_decision.accepted_blocks:
            coordinator.mark_tool_calls_emitted()
            yield CanonicalEvent(type="tool_use", index=0, content_block=accepted_block)

    def _pending_to_candidates(
        self,
        completed: list[PendingToolCall],
        invocation: Any,
    ) -> list[Any]:
        """Convert drained ``PendingToolCall``s to ``RawToolCallCandidate``s.

        Shared by the native-only streaming path and the combined
        native+textual BUFFER path so both build candidates from the same
        wire fragments.
        """
        route = invocation.route
        candidates: list[Any] = []
        for c in completed:
            if c.completed and c.name_fragments:
                name = "".join(c.name_fragments)
                args_str = "".join(c.argument_fragments)
                candidates.append(
                    RawToolCallCandidate(
                        id=c.call_id,
                        name=name,
                        raw_arguments=args_str,
                        source_protocol=route.upstream.wire_protocol.value,
                        source_index=c.tool_index,
                        choice_index=c.choice_index,
                        tool_index=c.tool_index,
                    )
                )
        return candidates

    async def _process_completed_stream_tools(
        self,
        completed: list[PendingToolCall],
        invocation: Any,
        coordinator: StreamCoordinator,
    ) -> AsyncIterator[CanonicalEvent]:
        """Process completed native tool calls through the transaction service.

        Used by the non-BUFFER streaming path where native fragments are the
        only candidate source. The combined native+textual BUFFER path builds
        its candidate list separately (via ``_pending_to_candidates``) but runs
        the same single atomic batch below the end-of-turn merge.
        """
        candidates = self._pending_to_candidates(completed, invocation)
        if not candidates:
            return

        canonical = invocation.reconciled_request
        request_context = invocation.request_context

        # P0.4: internal retrieval calls mean the model's turn is not
        # finished — run the private continuation (via the non-streaming send
        # path) and let its firewalled FINAL turn become the only emission
        # source. This is the non-BUFFER (native fragments) mirror of the
        # BUFFER-mode interception in the BUFFER branch above.
        enabled_internal = self._enabled_internal_tools(invocation)
        internal_candidates = [c for c in candidates if c.name in enabled_internal]
        if internal_candidates:
            continued = await self._private_loop.run(
                invocation,
                invocation.execution_record,
                internal_candidates=internal_candidates,
                decoded=DecodedModelResponse(),
                budget=getattr(invocation.execution_record, "attempt_budget", None),
            )
            if isinstance(continued, CanonicalResponse):
                # Terminal: loop/budget failure — already firewalled.
                err = continued.error or CanonicalError(
                    code="INTERNAL_TOOL_LOOP_EXHAUSTED",
                    message="Private continuation failed",
                )
                invocation.execution_record.finalize_error(err)
                coordinator.mark_turn_rejected()
                yield CanonicalEvent(type="error", error=err)
                yield CanonicalEvent(
                    type="message_stop",
                    stop_reason=CanonicalStopReason.INVALID_OUTPUT,
                )
                return
            for block in continued.content:
                if isinstance(block, CanonicalTextBlock) and block.text:
                    yield CanonicalEvent(type="text_delta", index=0, partial=block.text)
                elif isinstance(block, CanonicalToolCallBlock):
                    coordinator.mark_tool_calls_emitted()
                    yield CanonicalEvent(type="tool_use", index=0, content_block=block)
            # P0-17: native-protocol tool calls live in tool_candidates, not
            # in content — a candidates-only final turn must still surface
            # its PUBLIC calls as tool_use events.
            for candidate in getattr(continued, "tool_candidates", ()) or ():
                coordinator.mark_tool_calls_emitted()
                yield CanonicalEvent(
                    type="tool_use",
                    index=0,
                    content_block=CanonicalToolCallBlock(
                        id=candidate.id or "",
                        name=candidate.name,
                        arguments=(
                            json.loads(candidate.raw_arguments)
                            if isinstance(candidate.raw_arguments, str) and candidate.raw_arguments
                            else dict(candidate.raw_arguments or {})
                        ),
                    ),
                )
            emitted_calls = any(
                isinstance(b, CanonicalToolCallBlock) for b in continued.content
            ) or bool(getattr(continued, "tool_candidates", None))
            if not emitted_calls:
                # Turn completed with plain text after retrieval — mark the
                # turn rejected so the caller skips its generic tool-batch tail.
                coordinator.mark_turn_rejected()
            return

        # Run through the transaction service. Uses the shared helper so the
        # streaming path gets the confidence-gated repair policy, the
        # request-scoped budget (shared across batches), telemetry, and the
        # compatibility key — identical to the non-streaming path.
        transaction_context = self._build_transaction_context(invocation, canonical)
        batch_decision = await process_tool_batch(
            candidates,
            canonical.tools,
            context=transaction_context,
            policy=ToolBatchPolicy(invocation.repair_policy.batch_policy),
        )

        # Record repairs into session state for loop detection
        self._record_repairs_to_session(batch_decision, request_context)

        # Record per-call decisions onto the shared execution record. The
        # in-memory record is always populated; evidence-store write-back is a
        # separate, opt-in step.
        self._record_tool_decisions(batch_decision, invocation.execution_record)

        # Emit the decided batch: either the accepted tool_use blocks, or — on a
        # fully-rejected batch — a structured error + INVALID_OUTPUT message_stop
        # (mirrors _assemble_response's non-streaming handling). The shared
        # helper also finalizes the record as failed and marks the coordinator's
        # turn rejected so the caller skips its generic end-of-turn tail.
        async for event in self._emit_batch_decision_events(
            batch_decision,
            canonical.request_id,
            coordinator,
            invocation.execution_record,
        ):
            yield event
        # P0-7: mirror the BUFFER path — the batch outcome updates the
        # serving tuple's stream-safety observation either way.
        if self._record_stream_safety is not None:
            self._record_stream_safety(
                invocation,
                (
                    batch_decision.is_accepted
                    and bool(batch_decision.accepted_blocks)
                ),
            )

    async def _iter_frame_data(
        self,
        stream: Any,  # UpstreamStream
        framing: Any,  # StreamFraming
    ) -> AsyncIterator[tuple[dict[str, Any] | None, str]]:
        """Normalize SSE or NDJSON frames into the shape the downstream
        streaming loop expects.

        Yields ``(frame, raw_text)`` per wire frame. ``frame`` is a ``dict``
        for a successfully parsed frame, or ``None`` for a malformed/
        unparseable one. ``raw_text`` is the frame's original text (bounded
        by the caller before use) so malformed frames can be recorded as
        bounded diagnostics instead of being dropped with no trace.
        """
        from agent_interop.upstreams.codec import StreamFraming

        if framing == StreamFraming.NDJSON:
            async for item in stream.ndjson_events():
                if isinstance(item, MalformedNDJSONLine):
                    yield None, item.line
                else:
                    # NDJSON yields already-parsed frames (dicts); yield them
                    # directly. Non-dict items are treated as unparseable.
                    yield (item if isinstance(item, dict) else None), str(item)
            return

        # Default: SSE framing
        async for frame in stream.sse_events():
            data = frame.data
            if not data or not data.strip():
                # Skip empty/whitespace-only frames (e.g. keep-alives)
                continue
            stripped = data.strip()
            if stripped == "[DONE]":
                yield {"done": True}, stripped
                continue
            try:
                yield json.loads(stripped), stripped
            except json.JSONDecodeError:
                yield None, stripped
