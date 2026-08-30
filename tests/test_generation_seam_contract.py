"""GenerationSeam contract tests (P1.13 extraction).

The seam is the single owner of: render+serialize-once, the exact-context
gate, budget reservation (exactly ONE per generation), admission, and
reservation reconciliation. These tests pin the contract so future
refactors cannot reintroduce the historical bug class (parallel send
paths that each reserved/bypassed admission).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.context_store.store import ContextStore
from agent_interop.execution import InteropRequestExecution
from agent_interop.execution_attempts import AttemptBudget
from agent_interop.gateway import Gateway, ResolvedInvocation
from agent_interop.generation_seam import GenerationSeam, generation_output_reserve


def _route() -> ModelRoute:
    return ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        context=ContextConfig(context_limit_tokens=2000, output_reserve_tokens=500),
    )


def _request() -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=64),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
    )


class _RecordingAdmission:
    def __init__(self) -> None:
        self.calls = 0

    def generation_slot(self, base_url: str, model: str):  # noqa: ANN201
        self.calls += 1

        class _Slot:
            acquired = True

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

        return _Slot()


class _UsageTransport:
    """Returns one canned successful upstream response, recording bodies."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.bodies: list[bytes] = []

    async def send(self, request):  # noqa: ANN001, ANN201
        self.bodies.append(request.serialized_body)

        class _Resp:
            status_code = 200

            def __init__(self, outer: "_UsageTransport") -> None:
                self._outer = outer

            def is_error(self) -> bool:
                return False

            def json(self) -> dict[str, Any]:
                return self._outer.payload

        return _Resp(self)


class SimpleNamespace:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def _seam_gateway(transport: Any, admission: Any) -> tuple[Gateway, GenerationSeam]:
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    seam = GenerationSeam(
        gateway=gw,
        admission_controller=admission,
        token_meter=gw._token_meter,
        apply_invocation_plan=lambda rendered, plan, route: rendered,
        build_upstream_headers=lambda route, **kw: {},
    )
    return gw, seam


def _invocation(gw: Gateway) -> ResolvedInvocation:
    from agent_interop.config import ToolMode as _TM
    from agent_interop.upstreams.registry import get_codec

    route = _route()
    request = _request()
    minimal_plan = SimpleNamespace(effective_tool_mode=_TM.NATIVE)
    return ResolvedInvocation(
        request_context=RequestContext(session_id="s-seam"),
        original_request=request,
        reconciled_request=request,
        route=route,
        backend_metadata=None,
        model_profile=None,
        repair_policy=None,
        invocation_plan=minimal_plan,
        codec=get_codec(route.upstream.wire_protocol),
        compatibility_key=None,
        evidence_record=None,
        repair_budget=None,
        execution_record=InteropRequestExecution(),
        runtime_capabilities=gw._static_runtime_capabilities(route),
        pinned_refs=(),
        authoritative_request=request,
    )


def test_generation_output_reserve_bounds():
    assert generation_output_reserve(SimpleNamespace(generation=None)) == 512
    assert generation_output_reserve(
        SimpleNamespace(generation=SimpleNamespace(max_output_tokens=10**9)),
    ) == 1024
    assert generation_output_reserve(
        SimpleNamespace(generation=SimpleNamespace(max_output_tokens=10)),
    ) == 64


def test_run_step_commits_actual_usage_once():
    """Exactly ONE reservation per generation, committed with actuals.

    The totals reflect EXACTLY ONE generation's real spend: the reservation's
    estimate is folded into the running totals at reserve time, and commit
    REPLACES that estimate with the backend's actuals. ``reserve_input`` is a
    ceiling check only — it never writes the ledger — so a double
    reservation (the historical bug) would have left the input total at 33
    (12 + 21-estimate leftover) instead of the correct 12.
    """
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }
    transport = _UsageTransport(payload)
    admission = _RecordingAdmission()
    gw, seam = _seam_gateway(transport, admission)
    invocation = _invocation(gw)
    exec_record = InteropRequestExecution()
    budget = AttemptBudget()
    exec_record.attempt_budget = budget

    response, rendered = asyncio.run(seam.run_step(
        invocation, exec_record, purpose="private_continuation",
        context_limit_error=lambda inv, reason: pytest.fail(
            "context gate should not fire on a tiny request"
        ),
    ))

    assert response.error is None
    assert rendered  # bytes were returned
    assert admission.calls == 1, "exactly one admission slot per generation"
    # Commit REPLACED the reservation's estimate with actuals — the totals
    # are the actuals alone. A double reservation would leave input at 33
    # (12 actual + 21 stranded estimate); a stranded output estimate would
    # leave generated at 69 (5 + 64).
    assert budget.total_input_tokens == 12
    assert budget.generated_tokens == 5
    assert budget.generations_by_purpose == {"private_continuation": 1}
    assert len(transport.bodies) == 1


def test_run_step_serializes_body_once_and_matches_sent_bytes():
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    transport = _UsageTransport(payload)
    gw, seam = _seam_gateway(transport, _RecordingAdmission())
    invocation = _invocation(gw)

    response, rendered = asyncio.run(seam.run_step(
        invocation, InteropRequestExecution(), purpose="worker",
        context_limit_error=lambda inv, reason: pytest.fail("gate must not fire"),
    ))
    assert response.error is None
    # The returned byte count IS the exact bytes that hit the transport.
    assert rendered == transport.bodies[0]
    assert json.loads(rendered.decode())["model"] == "fake-model"


def test_context_gate_rejects_oversized_render_before_dispatch():
    class _NeverTransport:
        calls = 0

        async def send(self, request):  # noqa: ANN001
            self.calls += 1
            raise AssertionError("transport must not be reached")

    def _limit_error(inv: Any, reason: str):  # noqa: ANN001
        from agent_interop.abi import CanonicalError, CanonicalResponse

        return CanonicalResponse(
            model=CanonicalModelReference(requested_name="m"),
            error=CanonicalError(code="CONTEXT_LIMIT_EXCEEDED", message=reason),
        )

    class _TinyView:
        safe_context_limit = 1  # forces the gate

    transport = _NeverTransport()
    gw, seam = _seam_gateway(transport, _RecordingAdmission())
    from dataclasses import replace as _dc_replace

    invocation = _dc_replace(_invocation(gw), model_view=_TinyView())

    response, rendered = asyncio.run(seam.run_step(
        invocation, InteropRequestExecution(), purpose="worker",
        context_limit_error=_limit_error,
    ))
    assert response.error is not None
    assert response.error.code == "CONTEXT_LIMIT_EXCEEDED"
    assert rendered == b""
    assert transport.calls == 0


def test_no_budget_still_gates_but_skips_reservation():
    """A request without an AttemptBudget skips reservation but still goes
    through admission and the context gate."""
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
    }
    transport = _UsageTransport(payload)
    admission = _RecordingAdmission()
    gw, seam = _seam_gateway(transport, admission)
    invocation = _invocation(gw)
    exec_record = InteropRequestExecution()  # no attempt_budget attached

    response, _ = asyncio.run(seam.run_step(
        invocation, exec_record, purpose="worker",
        context_limit_error=lambda inv, reason: pytest.fail("gate must not fire"),
    ))
    assert response.error is None
    assert admission.calls == 1


def test_post_construction_transport_injection_is_honored():
    """The gateway's own seam uses the lazy transport property — injecting a
    test transport AFTER construction must be honored."""
    gw = Gateway(InteropServerConfig(
        probe_on_startup=False, log_level="error", routes={"r": _route()},
    ))
    store = ContextStore()
    gw._context_store = store
    payload = {
        "choices": [{"message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }
    transport = _UsageTransport(payload)
    gw._transport = transport  # post-construction injection

    invocation = _invocation(gw)
    response, rendered = asyncio.run(gw._send_one_model_step(
        invocation, InteropRequestExecution(),
    ))
    assert response.error is None
    assert len(transport.bodies) == 1
