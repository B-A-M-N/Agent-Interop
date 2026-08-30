"""P0-1 acceptance: hidden tool-repair generation is REAL and wired.

The review's acceptance criteria, end to end through the real Gateway
machinery (real GenerationSeam, real admission, real budget reservation,
real transaction pipeline — only the transport is scripted):

1. A REQUIRED read_file request whose tool call is missing a required
   argument triggers exactly ONE hidden repair generation.
2. Per-purpose accounting reports worker = 1, tool_repair = 1.
3. The repair request goes out WITHOUT the tool array (no second tool
   call invited) and its answer is re-validated, never trusted.
4. AUTO mode under the default policy (max_regenerations=0) adds no
   second call; AUTO with regeneration enabled still refuses (the
   tool-choice gate), because AUTO must not silently spend a generation.
5. The malformed call is never client-visible: either the corrected call
   or a rejection — never the raw broken arguments.
"""

from __future__ import annotations

import json
from typing import Any

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    RepairStatus,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    RepairPolicy,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.execution import InteropRequestExecution
from agent_interop.execution_attempts import AttemptBudget
from agent_interop.gateway import Gateway, ResolvedInvocation
from agent_interop.repair.pipeline import RepairBudget
from agent_interop.transaction import process_tool_batch
from agent_interop.transport.http import (
    PreparedUpstreamRequest,
    UpstreamResponse,
    UpstreamTransport,
)


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
        context=ContextConfig(context_limit_tokens=4000, output_reserve_tokens=500),
    )


TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "encoding": {"type": "string"},
        },
        "required": ["path"],
    },
)


def _completion(content: Any, tool_calls: list[dict] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "resp",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
    }


class _ScriptedTransport(UpstreamTransport):
    """Serves canned upstream responses in order, recording sent bodies."""

    def __init__(self, payloads: list[dict[str, Any]]) -> None:
        super().__init__()
        self._payloads = payloads
        self.bodies: list[dict[str, Any]] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        assert request.serialized_body is not None
        self.bodies.append(json.loads(request.serialized_body))
        payload = self._payloads.pop(0)
        return UpstreamResponse(
            status_code=200,
            body=json.dumps(payload).encode("utf-8"),
        )


class _Plan:
    effective_tool_mode = ToolMode.NATIVE


def _invocation(
    gw: Gateway,
    request: CanonicalRequest,
    policy: RepairPolicy,
) -> ResolvedInvocation:
    from agent_interop.upstreams.registry import get_codec

    route = _route()
    return ResolvedInvocation(
        request_context=RequestContext(session_id="s-repair", client_id=""),
        original_request=request,
        reconciled_request=request,
        route=route,
        backend_metadata=None,
        model_profile=None,
        repair_policy=policy,
        invocation_plan=_Plan(),
        codec=get_codec(route.upstream.wire_protocol),
        compatibility_key=None,
        evidence_record=None,
        repair_budget=RepairBudget(),
        execution_record=InteropRequestExecution(attempt_budget=AttemptBudget()),
        runtime_capabilities=gw._static_runtime_capabilities(route),
        pinned_refs=(),
        authoritative_request=request,
    )


def _required_request() -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        tool_choice=CanonicalToolChoice.required(),
        generation=CanonicalGenerationOptions(max_output_tokens=256, stream=False),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read it")]),
        ],
    )


def _auto_request() -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        tool_choice=CanonicalToolChoice.auto(),
        generation=CanonicalGenerationOptions(max_output_tokens=256, stream=False),
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read it")]),
        ],
    )


async def test_required_mode_repair_generates_and_accounts():
    transport = _ScriptedTransport([
        # Worker generation: malformed call — required "path" missing.
        _completion(None, tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"encoding": "utf-8"}'},
        }]),
        # Hidden repair generation: corrected call as plain JSON text.
        _completion(json.dumps({
            "name": "read_file",
            "arguments": {"path": "/tmp/x", "encoding": "utf-8"},
        })),
    ])
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    policy = RepairPolicy(max_regenerations=1)
    invocation = _invocation(gw, _required_request(), policy)
    exec_record = invocation.execution_record

    # 1 hidden generation is available to the transaction service.
    context = gw._build_transaction_context(
        invocation, invocation.reconciled_request,
    )
    assert context.regenerate_fn is not None

    # Worker generation through the ONE seam — same path every request pays.
    step, _ = await gw._send_one_model_step(
        invocation, exec_record, purpose="worker",
    )
    assert step.error is None
    assert len(step.tool_candidates) == 1

    decision_batch = await process_tool_batch(
        list(step.tool_candidates), [TOOL], context=context,
    )

    # The corrected call is accepted; the malformed one never surfaces.
    assert decision_batch.is_accepted
    assert decision_batch.rejected_count == 0
    assert len(decision_batch.accepted_blocks) == 1
    block = decision_batch.accepted_blocks[0]
    assert block.name == "read_file"
    assert block.arguments == {"path": "/tmp/x", "encoding": "utf-8"}
    assert decision_batch.decisions[0].outcome.status is RepairStatus.REGENERATED

    # Exactly two upstream bodies: the worker call and ONE hidden repair.
    assert len(transport.bodies) == 2
    repair_body = transport.bodies[1]
    # The repair surface carries no tools — a corrected argument object is
    # requested as text, not a second tool call.
    assert "tools" not in repair_body
    assert repair_body["messages"][-1]["role"] == "user"

    # Per-purpose accounting: worker = 1, tool_repair = 1 — the repair is a
    # real generation in every budget, not an invisible side call.
    purpose_counts = exec_record.attempt_budget.generations_by_purpose
    assert purpose_counts.get("worker") == 1
    assert purpose_counts.get("tool_repair") == 1

    # The request-scoped repair budget saw exactly the one attempt.
    assert invocation.repair_budget.regeneration_attempts == 1


async def test_default_policy_auto_adds_no_hidden_call():
    """AUTO + default policy (max_regenerations=0): no callback is even
    attached, so no request can silently pay a second generation."""
    transport = _ScriptedTransport([
        _completion(None, tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"encoding": "utf-8"}'},
        }]),
    ])
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    invocation = _invocation(gw, _required_request(), RepairPolicy())
    assert invocation.repair_policy.max_regenerations == 0

    context = gw._build_transaction_context(
        invocation, invocation.reconciled_request,
    )
    assert context.regenerate_fn is None

    step, _ = await gw._send_one_model_step(
        invocation, invocation.execution_record, purpose="worker",
    )
    decision_batch = await process_tool_batch(
        list(step.tool_candidates), [TOOL], context=context,
    )

    # Malformed call rejected — never client-visible as an accepted block.
    assert not decision_batch.is_accepted
    assert decision_batch.accepted_blocks == []
    # One generation total; no hidden call was smuggled in.
    assert len(transport.bodies) == 1
    assert invocation.execution_record.attempt_budget.generations_by_purpose == {
        "worker": 1,
    }


async def test_auto_with_regeneration_enabled_still_refers_to_choice_gate():
    """AUTO never silently regenerates even when the request explicitly
    enabled it — hidden repair is reserved for REQUIRED/NAMED, where the
    tool call was mandatory, not merely permitted."""
    transport = _ScriptedTransport([
        _completion(None, tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"encoding": "utf-8"}'},
        }]),
    ])
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    invocation = _invocation(gw, _auto_request(), RepairPolicy(max_regenerations=1))
    context = gw._build_transaction_context(
        invocation, invocation.reconciled_request,
    )
    # The callback exists, but the tool choice is AUTO...
    assert context.regenerate_fn is not None
    from agent_interop.abi import ToolChoiceMode

    assert context.tool_choice.mode is ToolChoiceMode.AUTO

    step, _ = await gw._send_one_model_step(
        invocation, invocation.execution_record, purpose="worker",
    )
    decision_batch = await process_tool_batch(
        list(step.tool_candidates), [TOOL], context=context,
    )

    assert not decision_batch.is_accepted
    assert decision_batch.accepted_blocks == []
    assert len(transport.bodies) == 1
    assert invocation.execution_record.attempt_budget.generations_by_purpose == {
        "worker": 1,
    }


async def test_failed_repair_generation_degrades_to_rejection():
    """A repair generation that fails (transport error payload) returns
    empty text; the orchestrator treats that as no-correction and the
    malformed call is rejected — nothing broken reaches the client."""
    transport = _ScriptedTransport([
        _completion(None, tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"encoding": "utf-8"}'},
        }]),
        # Repair answer is unparseable prose — no usable correction.
        _completion("I cannot help with that."),
    ])
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    invocation = _invocation(gw, _required_request(), RepairPolicy(max_regenerations=1))
    context = gw._build_transaction_context(
        invocation, invocation.reconciled_request,
    )
    exec_record = invocation.execution_record

    step, _ = await gw._send_one_model_step(
        invocation, exec_record, purpose="worker",
    )
    decision_batch = await process_tool_batch(
        list(step.tool_candidates), [TOOL], context=context,
    )

    assert not decision_batch.is_accepted
    assert decision_batch.accepted_blocks == []
    # The hidden generation still happened and is still accounted.
    assert len(transport.bodies) == 2
    purpose_counts = exec_record.attempt_budget.generations_by_purpose
    assert purpose_counts.get("worker") == 1
    assert purpose_counts.get("tool_repair") == 1


async def test_regenerated_dict_reaches_pipeline_without_round_trip():
    """P0-5: the correction dict is passed to repair_one as a dict.

    The orchestrator already holds parsed arguments; serializing them back
    to a JSON string only to re-parse them in the pipeline is a redundant
    round trip. repair_one accepts a dict and skips the parse stage for
    it, so the fast path is a pure call-site change. The round trip is
    observable: a dict pass-through means the pipeline sees the exact
    object the orchestrator validated for shape.
    """
    import agent_interop.transaction as txn_module

    correction = {
        "name": "read_file",
        "arguments": {"path": "/tmp/x", "encoding": "utf-8"},
    }
    seen_arguments: list[object] = []
    real_repair_one = txn_module.repair_one

    def spy_repair_one(*args: object, **kwargs: object) -> object:
        seen_arguments.append(kwargs.get("call_arguments"))
        return real_repair_one(*args, **kwargs)  # type: ignore[arg-type]

    transport = _ScriptedTransport([
        _completion(None, tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"encoding": "utf-8"}'},
        }]),
        _completion(json.dumps(correction)),
    ])
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    invocation = _invocation(gw, _required_request(), RepairPolicy(max_regenerations=1))
    context = gw._build_transaction_context(
        invocation, invocation.reconciled_request,
    )

    original_repair_one = txn_module.repair_one
    txn_module.repair_one = spy_repair_one  # type: ignore[assignment]
    try:
        step, _ = await gw._send_one_model_step(
            invocation, invocation.execution_record, purpose="worker",
        )
        decision_batch = await process_tool_batch(
            list(step.tool_candidates), [TOOL], context=context,
        )
    finally:
        txn_module.repair_one = original_repair_one  # type: ignore[assignment]

    assert decision_batch.is_accepted
    assert len(seen_arguments) == 2, (
        "expected worker validation + regeneration validation"
    )
    # First call is the worker's raw string; the regeneration's call — the
    # one P0-5 changes — must arrive as the dict, not a re-serialized string.
    worker_arguments, regenerated_arguments = seen_arguments
    assert isinstance(worker_arguments, str)
    assert isinstance(regenerated_arguments, dict)
    assert regenerated_arguments == {"path": "/tmp/x", "encoding": "utf-8"}


async def test_regenerated_output_still_fully_validated_not_privileged():
    """P0-5 corollary: skipping the re-serialize does NOT privilege the
    regenerated output. A correction whose arguments violate the schema
    still fails schema validation in the pipeline and degrades to the
    original rejection."""
    transport = _ScriptedTransport([
        _completion(None, tool_calls=[{
            "id": "call_1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"encoding": "utf-8"}'},
        }]),
        # "Corrected" call is STILL missing the required "path".
        _completion(json.dumps({
            "name": "read_file",
            "arguments": {"encoding": "utf-16"},
        })),
    ])
    gw = Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": _route()},
        ),
        transport=transport,
    )
    invocation = _invocation(gw, _required_request(), RepairPolicy(max_regenerations=1))
    context = gw._build_transaction_context(
        invocation, invocation.reconciled_request,
    )

    step, _ = await gw._send_one_model_step(
        invocation, invocation.execution_record, purpose="worker",
    )
    decision_batch = await process_tool_batch(
        list(step.tool_candidates), [TOOL], context=context,
    )

    assert not decision_batch.is_accepted
    assert decision_batch.accepted_blocks == []
    assert len(transport.bodies) == 2
