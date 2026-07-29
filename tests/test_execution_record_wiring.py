"""Tests for the unified execution-record lifecycle across the streaming and
non-streaming gateway paths.

These guard against the duplicate-``InteropRequestExecution`` bug (Bug 1) and
the divergent streaming tool-transaction context (Bug 2) fixed in the same
change. Before the fix:

* ``_prepare_invocation`` built its *own* execution record, so parser
  diagnostics recorded by ``_extract_tool_candidates`` landed on an object
  that was never the one ``handle_request``/``handle_stream`` finalized.
* The streaming path rebuilt the ``ToolTransactionContext`` inline, skipping
  the confidence gate, resetting the ``RepairBudget`` per batch, and dropping
  ``telemetry``/``compatibility_key``.

The fix routes everything through the single record built at the public entry
point and a shared ``_build_transaction_context`` helper.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    CanonicalUsage,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairConfig,
    RepairPolicy,
    RepairTier,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.execution import ExecutionState, InteropRequestExecution
from agent_interop.gateway import Gateway, ResolvedInvocation
from agent_interop.repair.pipeline import RepairBudget
from agent_interop.streaming.coordinator import PendingToolCall, StreamCoordinator, ToolStreamKey
from agent_interop.transport.http import PreparedUpstreamRequest
from agent_interop.upstreams.codec import DecodedModelResponse

TEST_TOOL = CanonicalTool(
    name="test_tool",
    description="A test tool",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
)


def _make_gateway() -> Gateway:
    """Gateway whose sole route resolves to the qwen-coder-ollama profile,
    which selects the ``tool_call_envelope`` parser (prompted mode)."""
    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "qwen": ModelRoute(
                id="qwen",
                client_model_aliases=["qwen2.5-coder"],
                upstream_model="qwen2.5-coder",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    return Gateway(config=config)


def _qwen_route(gw: Gateway) -> ModelRoute:
    return next(iter(gw.config.routes.values()))


async def _collect(agen: Any) -> list[Any]:
    return [item async for item in agen]


# ─── Bug 1: diagnostics must land on the record that gets finalized ────────


class TestDiagnosticOnFinalizedRecord:
    @pytest.mark.asyncio
    async def test_parser_diagnostic_lands_on_finalized_record(self):
        """The execution record that ``handle_request`` finalizes must be the
        SAME object that ``_extract_tool_candidates`` writes parser diagnostics
        into. Before the fix, ``_prepare_invocation`` built a separate record,
        so diagnostics were recorded on an object nobody ever inspected.

        We replicate the ``handle_request`` flow with a record we control
        (``handle_request`` does not expose its internal record), then assert
        the diagnostic and the finalization state both ended up on it.
        """
        gw = _make_gateway()
        canonical = CanonicalRequest(
            model=CanonicalModelReference(requested_name="qwen2.5-coder"),
            messages=[
                CanonicalMessage(
                    role="user",
                    content=[CanonicalTextBlock(text="Do something")],
                )
            ],
            tools=[TEST_TOOL],
            generation=CanonicalGenerationOptions(max_output_tokens=100),
            tool_choice=CanonicalToolChoice.auto(),
        )

        # The single record built at the entry point — this is the object the
        # public handler finalizes.
        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            canonical, RequestContext(), streaming=False, execution=record,
        )

        # Core invariant of the fix: the invocation carries OUR record, not a
        # freshly-constructed one.
        assert invocation.execution_record is record, (
            "prepare_invocation must use the caller-supplied execution record"
        )

        # A truncated <tool_call> envelope forces the textual extractor to emit
        # an error diagnostic (verified separately in test_extraction_wiring).
        decoded = DecodedModelResponse(
            content=[
                CanonicalTextBlock(
                    text='<tool_call>{"name":"test_tool", "argum'
                )
            ],
        )
        gw._extract_tool_candidates(decoded, invocation)

        # The diagnostic must be on the record that will be finalized.
        assert len(record.parser_diagnostics) == 1, (
            f"expected one parser diagnostic on the shared record, got "
            f"{record.parser_diagnostics!r}"
        )
        assert "error" in record.parser_diagnostics[0].lower()

        # The record is also populated with the resolved route/plan/key/budget.
        assert record.route is not None
        assert record.invocation_plan is not None
        assert record.repair_budget is not None

        # Now finalize the record exactly as handle_request does on success.
        result = CanonicalResponse(
            content=[],
            stop_reason=CanonicalStopReason.END_TURN,
            usage=CanonicalUsage(),
            model=CanonicalModelReference(
                requested_name="qwen2.5-coder",
                resolved_name="qwen2.5-coder",
            ),
        )
        record.finalize_response(result)

        # The SAME object carries both the diagnostic and the terminal state.
        assert record.state is ExecutionState.SUCCEEDED
        assert record.response_outcome == "accepted"
        assert len(record.parser_diagnostics) == 1


# ─── Bug 2 / (3): streaming uses the confidence-gated repair policy ─────────


class TestStreamingConfidenceGatedRepairPolicy:
    @pytest.mark.asyncio
    async def test_streaming_uses_invocation_repair_policy(self):
        """The streaming tool transaction must use ``invocation.repair_policy``
        — the confidence-gated policy built in ``_prepare_invocation`` — not a
        freshly-recomputed ``RepairPolicy.from_config(route.repair)`` that
        skips the confidence gate.

        Proven by capturing the ``ToolTransactionContext`` passed down the
        pipeline and checking object identity with the invocation's policy.
        """
        gw = _make_gateway()
        route = _qwen_route(gw)

        # A route whose raw repair config enables COERCIVE + REGENERATION, so a
        # naive ``from_config`` would yield a policy with 4 tiers. The gated
        # policy strips the risky tiers (simulating a low-confidence profile).
        route.repair = RepairConfig(
            max_regenerations=1, malformed_json="aggressive",
        )
        ungated = RepairPolicy.from_config(route.repair)
        gated = RepairPolicy(
            enabled_tiers=frozenset({RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE}),
            max_regenerations=1,
        )
        assert ungated.enabled_tiers != gated.enabled_tiers, (
            "test precondition: gating must change the enabled tiers"
        )

        canonical = CanonicalRequest(
            model=CanonicalModelReference(requested_name="qwen2.5-coder"),
            messages=[
                CanonicalMessage(
                    role="user",
                    content=[CanonicalTextBlock(text="x")],
                )
            ],
            tools=[TEST_TOOL],
            generation=CanonicalGenerationOptions(max_output_tokens=100),
            tool_choice=CanonicalToolChoice.auto(),
        )
        invocation = ResolvedInvocation(
            request_context=RequestContext(),
            original_request=canonical,
            reconciled_request=canonical,
            route=route,
            backend_metadata=None,
            model_profile=None,
            repair_policy=gated,
            invocation_plan=None,
            codec=None,
            compatibility_key=None,
            evidence_record=None,
            repair_budget=RepairBudget(),
            execution_record=InteropRequestExecution(),
        )

        coordinator = StreamCoordinator(route.upstream.wire_protocol)
        call = PendingToolCall(
            key=ToolStreamKey(choice_index=0, tool_index=0),
            call_id="tc_0",
            name_fragments=["test_tool"],
            argument_fragments=['{"key": "value"}'],
            completed=True,
        )

        # Capture the ToolTransactionContext handed to process_tool_batch.
        captured: dict[str, Any] = {}

        async def fake_process_tool_batch(candidates, tools, *, context=None, policy=None):
            captured["context"] = context
            # Return an accepted decision so the generator yields tool_use.
            from agent_interop.transaction import ToolBatchDecision
            return ToolBatchDecision(is_accepted=True)

        with patch("agent_interop.gateway.process_tool_batch", side_effect=fake_process_tool_batch):
            await _collect(
                gw._process_completed_stream_tools([call], invocation, coordinator)
            )

        ctx = captured.get("context")
        assert ctx is not None, "process_tool_batch was not invoked"
        # Identity check: the streaming path used the invocation's (gated)
        # policy, not a freshly-recomputed ungated one.
        assert ctx.repair_policy is gated, (
            "streaming must use the confidence-gated invocation.repair_policy"
        )
        assert ctx.repair_policy is not ungated
        assert ctx.repair_policy.enabled_tiers == {RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE}
        # And the request_id is derived correctly (not the old str(id(None)) "").
        assert ctx.request_id == invocation.request_context.request_id


# ─── Bug 2 / (1): streaming shares one budget across batches ────────────────


class TestStreamingBudgetSharing:
    @pytest.mark.asyncio
    async def test_repair_budget_accumulates_across_batches(self):
        """All completed-tool-call batches in a streaming request must share a
        single ``RepairBudget``. Before the fix, the streaming path created a
        fresh ``RepairBudget()`` per batch, so exhaustion/regeneration limits
        never accumulated across the request.

        We drive two batches through ``_process_completed_stream_tools`` with
        the same invocation and assert the shared budget's repair-operation
        counter accumulates rather than resetting.
        """
        gw = _make_gateway()
        route = _qwen_route(gw)

        # Enable syntax recovery so a trailing-comma arg triggers a repair
        # operation (which increments the shared budget counter).
        repair_policy = RepairPolicy.from_config(RepairConfig(malformed_json="safe"))
        assert RepairTier.SYNTAX_ONLY in repair_policy.enabled_tiers

        shared_budget = RepairBudget()
        canonical = CanonicalRequest(
            model=CanonicalModelReference(requested_name="qwen2.5-coder"),
            messages=[
                CanonicalMessage(
                    role="user",
                    content=[CanonicalTextBlock(text="x")],
                )
            ],
            tools=[TEST_TOOL],
            generation=CanonicalGenerationOptions(max_output_tokens=100),
            tool_choice=CanonicalToolChoice.auto(),
        )
        invocation = ResolvedInvocation(
            request_context=RequestContext(),
            original_request=canonical,
            reconciled_request=canonical,
            route=route,
            backend_metadata=None,
            model_profile=None,
            repair_policy=repair_policy,
            invocation_plan=None,
            codec=None,
            compatibility_key=None,
            evidence_record=None,
            repair_budget=shared_budget,
            execution_record=InteropRequestExecution(),
        )

        coordinator = StreamCoordinator(route.upstream.wire_protocol)

        def make_call(call_id: str) -> PendingToolCall:
            return PendingToolCall(
                key=ToolStreamKey(choice_index=0, tool_index=0),
                call_id=call_id,
                name_fragments=["test_tool"],
                # Trailing comma forces syntax recovery → one repair operation.
                argument_fragments=['{"key": "value",}'],
                completed=True,
            )

        # Batch 1.
        await _collect(gw._process_completed_stream_tools([make_call("tc_1")], invocation, coordinator))
        assert shared_budget.repair_operations == 1, (
            f"after batch 1 expected 1 repair op, got {shared_budget.repair_operations}"
        )

        # Batch 2 — same invocation, same budget. Must accumulate to 2, not reset.
        await _collect(gw._process_completed_stream_tools([make_call("tc_2")], invocation, coordinator))
        assert shared_budget.repair_operations == 2, (
            f"after batch 2 expected 2 accumulated repair ops, got "
            f"{shared_budget.repair_operations}"
        )


# ─── Helpers for items 2-5 ─────────────────────────────────────────────────


class _RecordingExec(InteropRequestExecution):
    """Subclass that records every instance constructed.

    Used to capture the execution record that ``handle_request``/``handle_stream``
    build internally (they do not expose it). Swap in via ``_record_execs``.
    """

    instances: list[InteropRequestExecution] = []

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        _RecordingExec.instances.append(self)


@contextlib.contextmanager
def _record_execs():
    """Context manager that patches the gateway's InteropRequestExecution with
    ``_RecordingExec`` and yields the list of captured instances."""
    import agent_interop.gateway as gateway_module

    _RecordingExec.instances = []
    with patch.object(gateway_module, "InteropRequestExecution", _RecordingExec):
        yield _RecordingExec.instances


def _make_text_canonical(model: str = "qwen2.5-coder") -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name=model),
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalTextBlock(text="Do something")],
            )
        ],
        tools=[TEST_TOOL],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )


class _FrozenToolsTransport:
    """Transport returning a single well-formed tool call for TEST_TOOL.

    Emits an Ollama-format non-streaming response (the qwen test route uses the
    ``OLLAMA_CHAT`` wire protocol, whose codec reads ``body.message.tool_calls``).
    """

    async def send(self, request: Any) -> Any:
        from agent_interop.transport.http import UpstreamResponse
        body = {
            "model": "qwen2.5-coder",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {"name": "test_tool", "arguments": {"key": "v"}},
                }],
            },
            "done_reason": "tool_calls",
            "done": True,
            "prompt_eval_count": 5,
            "eval_count": 3,
        }
        return UpstreamResponse(
            status_code=200, headers={}, body=json.dumps(body).encode("utf-8")
        )

    async def close(self) -> None:
        pass


# ─── Item 2: execution record is constructed with request context ───────────


class TestExecutionContextWired:
    @pytest.mark.asyncio
    async def test_handle_request_passes_context(self) -> None:
        """``handle_request`` must build its record with ``context=`` set."""
        gw = _make_gateway()
        canonical = _make_text_canonical()
        ctx = RequestContext(client_id="claude_code")
        with _record_execs() as instances:
            await gw.handle_request(canonical, ctx)
        assert instances, "expected an InteropRequestExecution to be constructed"
        record = instances[-1]
        assert record.context is not None, "record.context must be non-None"
        assert record.context is ctx, "record.context must be the supplied context"

    @pytest.mark.asyncio
    async def test_handle_stream_passes_context(self) -> None:
        """``handle_stream`` must build its record with ``context=`` set."""
        gw = _make_gateway()
        canonical = _make_text_canonical()
        ctx = RequestContext(client_id="codex")
        with _record_execs() as instances:
            # Drive the generator to completion with a transport that returns a
            # text response (no tool calls) so the record is finalized.
            gw._transport = _FrozenToolsTransport()
            await _collect(gw.handle_stream(canonical, ctx))
        assert instances, "expected an InteropRequestExecution to be constructed"
        record = instances[-1]
        assert record.context is not None, "record.context must be non-None"
        assert record.context is ctx, "record.context must be the supplied context"


# ─── Item 3: tool_decisions populated without an evidence store ─────────────


class TestToolDecisionsWithoutEvidenceStore:
    @pytest.mark.asyncio
    async def test_tool_decisions_populated_no_evidence_store(self) -> None:
        """With no evidence store configured, a tool-calling request must still
        populate the in-memory ``execution.tool_decisions`` record. Before the
        fix, ``_record_tool_decisions`` early-returned when the store was None,
        so the record's ``tool_decisions`` was always empty on the default path.
        """
        gw = _make_gateway()
        assert gw._evidence_store is None, (
            "test precondition: default gateway has no evidence store"
        )
        canonical = _make_text_canonical()
        with _record_execs() as instances:
            gw._transport = _FrozenToolsTransport()
            result = await gw.handle_request(canonical, RequestContext())
        assert result.error is None, f"unexpected error: {result.error}"
        assert instances, "expected an InteropRequestExecution to be constructed"
        record = instances[-1]
        assert len(record.tool_decisions) >= 1, (
            f"expected tool_decisions to be populated, got {record.tool_decisions!r}"
        )
        assert record.tool_decisions[0].tool_name == "test_tool"


# ─── Item 4: _log_summary fires exactly once per terminal outcome ───────────


class TestLogSummaryOncePerRequest:
    @pytest.mark.asyncio
    async def test_log_summary_once_on_success(self) -> None:
        gw = _make_gateway()
        canonical = _make_text_canonical()
        calls = []
        original = InteropRequestExecution._log_summary

        def counting(self):
            calls.append(1)
            return original(self)

        with patch.object(InteropRequestExecution, "_log_summary", counting):
            gw._transport = _FrozenToolsTransport()
            await gw.handle_request(canonical, RequestContext())
        assert len(calls) == 1, f"expected exactly 1 summary on success, got {len(calls)}"

    @pytest.mark.asyncio
    async def test_log_summary_once_on_error(self) -> None:
        gw = _make_gateway()
        canonical = _make_text_canonical()
        calls = []
        original = InteropRequestExecution._log_summary

        def counting(self):
            calls.append(1)
            return original(self)

        # Force an error from the send path.
        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("synthetic send failure")

        with patch.object(InteropRequestExecution, "_log_summary", counting), \
                patch.object(gw, "_handle_request_send", side_effect=boom):
            with pytest.raises(RuntimeError, match="synthetic send failure"):
                await gw.handle_request(canonical, RequestContext())
        assert len(calls) == 1, f"expected exactly 1 summary on error, got {len(calls)}"

    @pytest.mark.asyncio
    async def test_log_summary_once_on_cancel(self) -> None:
        gw = _make_gateway()
        canonical = _make_text_canonical()
        calls = []
        original = InteropRequestExecution._log_summary

        def counting(self):
            calls.append(1)
            return original(self)

        async def cancelled(*args: Any, **kwargs: Any) -> Any:
            raise asyncio.CancelledError()

        with patch.object(InteropRequestExecution, "_log_summary", counting), \
                patch.object(gw, "_handle_request_send", side_effect=cancelled):
            with pytest.raises(asyncio.CancelledError):
                await gw.handle_request(canonical, RequestContext())
        assert len(calls) == 1, f"expected exactly 1 summary on cancel, got {len(calls)}"

    @pytest.mark.asyncio
    async def test_log_summary_once_on_stream_error(self) -> None:
        """Streaming entry point: a raised exception in the send path summarizes
        exactly once. ``handle_stream`` converts the internal exception into
        protocol error events (it does not re-raise), so we assert the summary
        fired once and an ``error`` event was yielded."""
        gw = _make_gateway()
        canonical = _make_text_canonical()
        calls = []
        original = InteropRequestExecution._log_summary

        def counting(self):
            calls.append(1)
            return original(self)

        # The mock must raise synchronously when ``handle_stream`` calls
        # ``self._handle_stream_send(...)`` (``async for ... in`` it). Use a
        # Mock whose side_effect is the exception — passing the exception as
        # ``patch.object``'s third arg would assign it to ``new`` instead.
        fake_send = Mock(side_effect=RuntimeError("synthetic stream failure"))
        with patch.object(InteropRequestExecution, "_log_summary", counting), \
                patch.object(gw, "_handle_stream_send", fake_send):
            events = await _collect(gw.handle_stream(canonical, RequestContext()))
        assert len(calls) == 1, (
            f"expected exactly 1 summary on stream error, got {len(calls)}"
        )
        types = [e.type for e in events]
        assert "error" in types, f"expected an error event, got {types}"
        assert "message_stop" in types, f"expected a message_stop terminal, got {types}"

    @pytest.mark.asyncio
    async def test_log_summary_once_on_stream_success(self) -> None:
        """Streaming entry point, SUCCESS path: ``_log_summary`` fires exactly
        once. Uses an OpenAI Chat route (native tools, SSE framing) and a fake
        SSE transport that returns a plain text completion."""
        from agent_interop.transport.sse import SSEFrame

        calls = []
        original = InteropRequestExecution._log_summary

        def counting(self):
            calls.append(1)
            return original(self)

        # OpenAI Chat route so the upstream codec uses native tools + SSE framing.
        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            log_level="error",
            probe_on_startup=False,
            routes={
                "openai": ModelRoute(
                    id="openai",
                    client_model_aliases=["test-model"],
                    upstream_model="fake-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OPENAI_COMPATIBLE,
                        base_url="http://127.0.0.1:0",
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )
        gw = Gateway(config=config)
        canonical = _make_text_canonical(model="test-model")

        # Two SSE frames: a text delta, then the terminal [DONE].
        delta = json.dumps({
            "id": "x",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hello"}}],
        })
        done = "[DONE]"

        @asynccontextmanager
        async def fake_stream(request: PreparedUpstreamRequest) -> Any:
            class _SseStream:
                status_code = 200

                async def sse_events(self_inner: Any) -> Any:
                    yield SSEFrame(data=delta)
                    yield SSEFrame(data=done)

                async def __aenter__(self_inner: Any) -> Any:
                    return self_inner

                async def __aexit__(self_inner: Any, *args: object) -> None:
                    return None

            yield _SseStream()

        class _SseTransport:
            stream = staticmethod(fake_stream)

            async def close(self_inner: Any) -> None:
                pass

        with patch.object(InteropRequestExecution, "_log_summary", counting):
            gw._transport = _SseTransport()
            events = await _collect(gw.handle_stream(canonical, RequestContext()))
        assert len(calls) == 1, (
            f"expected exactly 1 summary on stream success, got {len(calls)}"
        )
        types = [e.type for e in events]
        assert "text_delta" in types, f"expected a text_delta, got {types}"
        assert "message_stop" in types, f"expected a message_stop terminal, got {types}"


# ─── Item 5: asyncio.CancelledError handled, record finalized CANCELLED ─────


class TestCancellationFinalizesRecord:
    @pytest.mark.asyncio
    async def test_handle_request_cancel_finalizes_cancelled(self) -> None:
        """Cancelling ``handle_request`` mid-flight must finalize the record as
        CANCELLED and re-raise the ``CancelledError`` (never swallow it)."""
        gw = _make_gateway()
        canonical = _make_text_canonical()

        async def cancelled(*args: Any, **kwargs: Any) -> Any:
            raise asyncio.CancelledError()

        with _record_execs() as instances, \
                patch.object(gw, "_handle_request_send", side_effect=cancelled):
            with pytest.raises(asyncio.CancelledError):
                await gw.handle_request(canonical, RequestContext())
        assert instances, "expected an InteropRequestExecution to be constructed"
        record = instances[-1]
        assert record.state is ExecutionState.CANCELLED, (
            f"expected CANCELLED state, got {record.state}"
        )
        assert record.response_outcome == "cancelled"

    @pytest.mark.asyncio
    async def test_handle_stream_cancel_finalizes_cancelled(self) -> None:
        """Cancelling ``handle_stream`` mid-flight must finalize the record as
        CANCELLED and re-raise, without yielding further frames."""
        gw = _make_gateway()
        canonical = _make_text_canonical()

        # The mock must raise synchronously when ``handle_stream`` calls
        # ``self._handle_stream_send(...)`` so the CancelledError propagates to
        # ``handle_stream``'s ``except asyncio.CancelledError`` clause.
        fake_send = Mock(side_effect=asyncio.CancelledError())
        with _record_execs() as instances, \
                patch.object(gw, "_handle_stream_send", fake_send):
            with pytest.raises(asyncio.CancelledError):
                await _collect(gw.handle_stream(canonical, RequestContext()))
        assert instances, "expected an InteropRequestExecution to be constructed"
        record = instances[-1]
        assert record.state is ExecutionState.CANCELLED, (
            f"expected CANCELLED state, got {record.state}"
        )
        assert record.response_outcome == "cancelled"
