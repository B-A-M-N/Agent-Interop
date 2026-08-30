"""The single model-generation seam (P0-12/P0-17).

One class owns everything that must be identical for every model
generation a request makes — public attempt, private continuation,
controller turn, context summary:

    render → serialize once → exact-context gate → budget reservation →
    admission slot → transport → usage reconciliation

It exists because the historical code had three parallel send paths
(public, private, streaming) that each re-implemented parts of this
pipeline and drifted: private generations bypassed admission control,
budgets were double-reserved in two places, and the exact-context gate
ran only on one of the three paths. With this seam the drift is
structurally impossible: a caller either goes through
:meth:`GenerationSeam.run_step` or it is not a generation.

Reservation lifecycle (the accounting invariant, P0-62):

* :meth:`dispatch` reserves headroom, then reconciles FAILED generations
  itself (release when nothing was sent, keep the estimate after a
  dispatched failure — the backend may have spent it);
* on SUCCESS the reservation is returned still OPEN, because actual
  token usage only exists after the codec decodes the wire answer;
* the caller commits it through :meth:`finalize_reservation` with the
  decoded actuals (run_step does this internally; the gateway's public
  path does it right after its own decode). A success reservation is
  therefore never left dangling — dispatch, decode-failure, and
  decode-success are the only three exits, and all three reconcile.

Coupling contract (deliberately narrow):

* takes the per-request state it reads/writes as explicit arguments
  (``invocation``, ``exec_record``) instead of owning request state;
* owns NO request-scoped state of its own — one instance per Gateway is
  safe for concurrent requests;
* delegates rendered-request presentation to callables supplied at
  construction (``apply_plan``/``build_headers``) so the codec/plan/auth
  logic stays with the Gateway's attempt machinery and this module stays
  presentation-agnostic.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from agent_interop.abi import (
    CanonicalError,
    CanonicalModelReference,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalUsage,
)
from agent_interop.errors import InteropErrorCode, classify_http_status

logger = logging.getLogger("agent_interop.generation_seam")

__all__ = ["GenerationSeam"]


def generation_output_reserve(request_local: Any) -> int:
    """Output-token headroom charged against a generation's budget."""
    return min(
        1024,
        max(64, getattr(getattr(request_local, "generation", None), "max_output_tokens", 0) or 512),
    )


class GenerationSeam:
    """Render → gate → reserve → admit → transport → reconcile, once.

    See the module docstring for why this is a class and what it must
    never grow into (no extraction, no transactions, no projection).
    """

    def __init__(
        self,
        *,
        gateway: Any,
        admission_controller: Any,
        token_meter: Any,
        apply_invocation_plan: Callable[[dict[str, Any], Any, Any], dict[str, Any]],
        build_upstream_headers: Callable[..., dict[str, str]],
    ) -> None:
        # ``gateway`` is held for its ``transport`` property only — that
        # property builds the default transport lazily and honors test
        # injection (``gw._transport = fake``), so the seam must resolve it
        # per dispatch, never capture it at construction time.
        self._gateway = gateway
        self._admission_controller = admission_controller
        self._token_meter = token_meter
        self._apply_invocation_plan = apply_invocation_plan
        self._build_upstream_headers = build_upstream_headers

    @property
    def _transport(self) -> Any:
        return self._gateway.transport

    # ─── Preparation ──────────────────────────────────────────────────────

    async def prepare(
        self,
        invocation: Any,
        *,
        stream: bool,
        attempt_request: Any | None = None,
        context_limit_error: Callable[[Any, str], CanonicalResponse],
    ) -> tuple[Any, bytes, Any, int, Any] | CanonicalResponse:
        """Render, serialize once, and run the exact-context gate.

        Returns ``(request_local, rendered_bytes, meter_count,
        output_reserve, rendered)`` on success, or a terminal
        :class:`CanonicalResponse` error from the gate.

        ``attempt_request`` — the request already narrowed for the current
        ladder attempt (tools stripped for PROMPTED/TEXTUAL/DISABLED); the
        seam renders whichever one it is given and never inspects tools.
        ``context_limit_error`` — the gateway's bounded error builder, so
        CONTEXT_LIMIT_EXCEEDED details stay in one place.
        """
        route = invocation.route
        plan = invocation.invocation_plan
        codec = invocation.codec
        canonical = invocation.reconciled_request

        request_local = attempt_request if attempt_request is not None else canonical
        rendered = codec.render_request(request_local, route.upstream_model, stream=stream)
        rendered = self._apply_invocation_plan(rendered, plan, route)
        rendered_bytes = json.dumps(
            rendered,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8", "replace")

        # Unconditional exact-context gate — the EXACT body about to be sent,
        # for every generation kind (P0-12).
        runtime = invocation.runtime_capabilities
        meter_count = self._token_meter.measure_rendered(
            rendered_bytes,
            model_digest=getattr(runtime, "model_digest", ""),
            chat_template_digest=getattr(runtime, "chat_template_digest", ""),
            wire_protocol=route.upstream.wire_protocol.value,
        )
        safe_limit = getattr(invocation.model_view, "safe_context_limit", 0) or 0
        output_reserve = generation_output_reserve(request_local)
        if safe_limit and meter_count.input_tokens + output_reserve > safe_limit:
            return context_limit_error(
                invocation,
                f"rendered_input={meter_count.input_tokens}+reserve>{safe_limit}",
            )
        return request_local, rendered_bytes, meter_count, output_reserve, rendered

    # ─── Dispatch ─────────────────────────────────────────────────────────

    async def dispatch(
        self,
        invocation: Any,
        exec_record: Any,
        *,
        prepared: tuple[Any, bytes, Any, int, Any],
        purpose: str,
        context_limit_error: Callable[[Any, str], CanonicalResponse],
    ) -> tuple[Any, bytes, Any | None]:
        """Own admission, transport, and error-path budget reconciliation.

        Reservation of budget headroom, rendered-byte accounting, an
        admission slot, the transport call, and reconciliation of FAILED
        dispatches (release when nothing was sent, keep the estimate after
        a dispatched failure). On SUCCESS the reservation is returned still
        OPEN — actual token usage only exists after codec decoding, so the
        caller commits it through :meth:`finalize_reservation`
        (:meth:`run_step` does this internally).

        Returns ``(response, rendered_bytes, open_reservation)``. On success
        the response is the raw upstream response (callers decode it through
        their codec) and the reservation may still be open; failures come
        back as ``CanonicalResponse`` with ``.error`` set — always terminal,
        never retried here, and always fully reconciled.
        """
        _request_local, rendered_bytes, meter_count, output_reserve, _ = prepared
        route = invocation.route
        canonical = invocation.reconciled_request
        budget = getattr(exec_record, "attempt_budget", None)

        # Cumulative request-resource accounting. reserve_generation() also
        # bounds latency and token ceilings; commit() is the ONLY place a
        # completed generation is counted (record_generation), so a queued-
        # out request never reports a generation that did not happen.
        # THE invariant this seam exists to enforce: exactly ONE reservation
        # per generation, owned HERE — callers gate allowances but never
        # pre-reserve (the historical double-reservation bug).
        reservation = None
        if budget is not None:
            if not budget.reserve_rendered_bytes(len(rendered_bytes)):
                return (
                    context_limit_error(invocation, "max_total_rendered_bytes"),
                    rendered_bytes,
                    None,
                )
            if not budget.reserve_input(meter_count.input_tokens):
                return (
                    context_limit_error(invocation, "max_total_input_tokens"),
                    rendered_bytes,
                    None,
                )
            reservation = budget.reserve_generation(
                estimated_input_tokens=meter_count.input_tokens,
                output_reserve_tokens=output_reserve,
                purpose=purpose,
            )
            if reservation is None:
                return (
                    context_limit_error(
                        invocation,
                        f"generation budget exhausted ({budget.exhausted_by})",
                    ),
                    rendered_bytes,
                    None,
                )

        upstream_request = self._build_upstream_request(invocation, route, rendered_bytes)

        dispatched = False
        try:
            async with self._admission_controller.generation_slot(
                route.upstream.base_url,
                route.upstream_model,
            ) as slot:
                if not getattr(slot, "acquired", False):
                    result_kind = getattr(getattr(slot, "result", None), "value", "unavailable")
                    if reservation is not None:
                        reservation.release()
                    return (
                        CanonicalResponse(
                            content=[],
                            stop_reason=CanonicalStopReason.END_TURN,
                            usage=CanonicalUsage(),
                            model=CanonicalModelReference(
                                requested_name=canonical.model.requested_name,
                                resolved_name=route.upstream_model,
                            ),
                            error=CanonicalError(
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
                            ),
                        ),
                        rendered_bytes,
                        None,
                    )
                dispatched = True
                response = await self._transport.send(upstream_request)
        except Exception as exc:  # transport-layer failures map to terminal errors
            response = self._transport_error(invocation, route, exc)

        # Error paths reconcile HERE (release if nothing was sent, keep the
        # estimate after a dispatched failure). Success paths return the
        # reservation OPEN: actual usage appears only after codec decode, so
        # committing here would strand every generation on the estimate.
        if isinstance(response, CanonicalResponse) or response.is_error():
            self._reconcile_failed_reservation(
                budget,
                reservation,
                rendered_bytes,
                dispatched,
            )
            return response, rendered_bytes, None
        return response, rendered_bytes, reservation

    # ─── One-step convenience (decode included) ───────────────────────────

    async def run_step(
        self,
        invocation: Any,
        exec_record: Any,
        *,
        purpose: str,
        context_limit_error: Callable[[Any, str], CanonicalResponse],
        attempt_request: Any | None = None,
    ) -> tuple[CanonicalResponse, bytes]:
        """prepare → dispatch → decode into ``CanonicalResponse``.

        The private continuation loop's unit of work: one generation with
        no extraction and no transaction pipeline, returning a uniform
        ``CanonicalResponse`` in both error and success cases so the loop's
        downstream decoding has one shape.
        """
        prepared = await self.prepare(
            invocation,
            stream=False,
            attempt_request=attempt_request,
            context_limit_error=context_limit_error,
        )
        if isinstance(prepared, CanonicalResponse):
            return prepared, b""
        upstream, rendered_bytes, reservation = await self.dispatch(
            invocation,
            exec_record,
            prepared=prepared,
            purpose=purpose,
            context_limit_error=context_limit_error,
        )
        if isinstance(upstream, CanonicalResponse):
            return upstream, rendered_bytes
        decoded, decode_error = self._decode_upstream(invocation, upstream)
        if decode_error is not None:
            # Dispatched but the wire answer was unusable (HTTP error status
            # or non-JSON body): same conservative rule as any dispatched
            # failure — keep the estimate, the backend may have spent it.
            if reservation is not None:
                reservation.commit_estimated()
                reservation.budget.record_rendered_bytes(len(rendered_bytes))
            return decode_error, rendered_bytes
        # The generation happened and its REAL usage is now known: replace
        # the reservation's estimate with actuals (single commit per
        # generation — the invariant this seam exists for).
        if reservation is not None:
            self.finalize_reservation(decoded, reservation)
            reservation.budget.record_rendered_bytes(len(rendered_bytes))
        return self._decoded_to_step(invocation, decoded), rendered_bytes

    # ─── Internals ────────────────────────────────────────────────────────

    def _build_upstream_request(
        self,
        invocation: Any,
        route: Any,
        rendered_bytes: bytes,
    ) -> Any:
        from agent_interop.transport.http import PreparedUpstreamRequest

        return PreparedUpstreamRequest(
            method="POST",
            url=f"{route.upstream.base_url}{invocation.codec.endpoint_path()}",
            headers=self._build_upstream_headers(
                route,
                client_headers=dict(invocation.request_context.forwardable_transport_headers),
                codec_headers=invocation.codec.required_headers(),
            ),
            body=json.loads(rendered_bytes.decode("utf-8", "replace")),
            stream=False,
            timeout_seconds=route.upstream.timeout_seconds,
            serialized_body=rendered_bytes,
        )

    def _transport_error(
        self,
        invocation: Any,
        route: Any,
        exc: Exception,
    ) -> Any:
        """Map a transport exception to a terminal response or re-raise.

        Only ``UpstreamResponseTooLargeError`` converts (the one case the
        historical code handled); every other transport exception
        propagates raw so callers' existing exception handling — including
        cancellation — stays intact.
        """
        from agent_interop.transport.http import UpstreamResponseTooLargeError

        if not isinstance(exc, UpstreamResponseTooLargeError):
            raise exc
        return CanonicalResponse(
            content=[],
            stop_reason=CanonicalStopReason.INVALID_OUTPUT,
            usage=CanonicalUsage(),
            model=CanonicalModelReference(
                requested_name=invocation.reconciled_request.model.requested_name,
                resolved_name=route.upstream_model,
            ),
            error=CanonicalError(
                code=InteropErrorCode.STREAM_SIZE_LIMIT,
                message=str(exc),
            ),
        )

    @staticmethod
    def _reconcile_failed_reservation(
        budget: Any,
        reservation: Any | None,
        rendered_bytes: bytes,
        dispatched: bool,
    ) -> None:
        """Reconcile a FAILED generation: dispatched → keep the estimate
        (the backend saw the wire and may have spent the tokens); not
        dispatched → release (nothing was sent)."""
        if reservation is not None:
            if dispatched:
                reservation.commit_estimated()
            else:
                reservation.release()
        budget.record_rendered_bytes(len(rendered_bytes))

    def finalize_reservation(
        self,
        decoded: Any,
        reservation: Any | None,
    ) -> None:
        """Commit an OPEN success reservation with post-decode actuals.

        Called after codec decoding, when the backend's real usage numbers
        exist. A missing reservation (no budget / already-failed dispatch)
        is a no-op; missing usage keeps the estimate, matching the stream
        tail's ``final_usage is None`` branch.
        """
        if reservation is None:
            return
        usage = getattr(decoded, "usage", None)
        if usage is None:
            reservation.commit_estimated()
            return
        reservation.commit(
            actual_input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            actual_output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        )

    def _decode_upstream(
        self,
        invocation: Any,
        upstream: Any,
    ) -> tuple[Any | None, CanonicalResponse | None]:
        """Decode the raw upstream response; ``(decoded, None)`` or
        ``(None, terminal_error)``. Never reconciles anything."""
        model_ref = CanonicalModelReference(
            requested_name=invocation.reconciled_request.model.requested_name,
            resolved_name=invocation.route.upstream_model,
        )
        if upstream.is_error():
            return None, CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=model_ref,
                error=CanonicalError(
                    code=classify_http_status(upstream.status_code),
                    message=(
                        f"Upstream returned {upstream.status_code}: "
                        f"{upstream.body[:500].decode('utf-8', errors='replace')}"
                    ),
                ),
            )
        try:
            return invocation.codec.decode_response(upstream.json()), None
        except (json.JSONDecodeError, ValueError):
            return None, CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=model_ref,
                error=CanonicalError(
                    code="INVALID_UPSTREAM_OUTPUT",
                    message=(
                        f"Upstream returned non-JSON response (status={upstream.status_code})"
                    ),
                ),
            )

    def _decoded_to_step(
        self,
        invocation: Any,
        decoded: Any,
    ) -> CanonicalResponse:
        """Project a decoded model response into a step ``CanonicalResponse``."""
        return CanonicalResponse(
            content=list(decoded.content),
            stop_reason=decoded.stop_reason,
            usage=getattr(decoded, "usage", None) or CanonicalUsage(),
            model=CanonicalModelReference(
                requested_name=invocation.reconciled_request.model.requested_name,
                resolved_name=invocation.route.upstream_model,
            ),
            tool_candidates=list(getattr(decoded, "tool_candidates", ()) or ()),
        )
