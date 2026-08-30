"""P0-7: stream-safety observations separate qualification from evidence.

Without a configured (opt-in) evidence store, the buffered-stream gate
used to buffer every tool-bearing stream forever — schema-v2's
``buffer_unverified_streaming=True`` default could never be satisfied by
``evidence_record``, which only exists when the operator configured the
store. The fix: one fully-accepted unbuffered streaming turn on a serving
tuple records an in-process observation that unlocks later streams of the
SAME tuple; a rejected batch revokes it. The evidence-store path is
unchanged and remains the durable, operator-certified channel.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairPolicy,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.gateway import Gateway
from agent_interop.planning.stream_safety import (
    StreamSafetyCache,
    stream_safety_key,
)

ROUTE = ModelRoute(
    id="r",
    client_model_aliases=["m"],
    upstream_model="fake-model",
    upstream=UpstreamConfig(
        kind=UpstreamKind.OPENAI_COMPATIBLE,
        base_url="http://127.0.0.1:1",
        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
    ),
)


def _gateway() -> Gateway:
    return Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": ROUTE},
        ),
    )


def _tool_invocation(path: str = "direct", effective_mode: Any = ToolMode.NATIVE) -> Any:
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        tools=[CanonicalTool(name="t", description="d", input_schema={"type": "object"})],
        tool_choice=CanonicalToolChoice.auto(),
    )
    return SimpleNamespace(
        authoritative_request=request,
        reconciled_request=request,
        route=SimpleNamespace(compatibility=SimpleNamespace(buffer_unverified_streaming=True)),
        private_capabilities=None,
        invocation_plan=SimpleNamespace(effective_tool_mode=effective_mode),
        compatibility_plan=SimpleNamespace(path=SimpleNamespace(value=path)),
        evidence_record=None,
    )


# ─── Cache unit behavior ─────────────────────────────────────────────────────


def test_cache_roundtrip_and_revoke():
    key = stream_safety_key(
        model_digest="d", template_digest="t", serving_config_digest="s",
        profile_revision="1", client_protocol="c/anthropic",
        tool_surface_fingerprint="f", tool_choice_class="auto",
    )
    cache = StreamSafetyCache()
    assert not cache.is_safe(key)
    cache.record(key)
    assert cache.is_safe(key)
    cache.revoke(key)
    assert not cache.is_safe(key)


def test_key_discriminates_tuples():
    def k(**over: Any) -> str:
        base: dict[str, Any] = dict(
            model_digest="d", template_digest="t", serving_config_digest="s",
            profile_revision="1", client_protocol="c/proto",
            tool_surface_fingerprint="f", tool_choice_class="auto",
        )
        base.update(over)
        return stream_safety_key(**base)

    assert k() == k()
    assert k(model_digest="other") != k()
    assert k(tool_surface_fingerprint="other") != k()
    assert k(tool_choice_class="required") != k()


def test_empty_key_never_records():
    cache = StreamSafetyCache()
    cache.record("")
    assert len(cache) == 0


# ─── Gate integration ────────────────────────────────────────────────────────


def test_unobserved_tuple_still_buffers():
    gw = _gateway()
    assert gw._requires_buffered_stream_validation(_tool_invocation()) is True


def test_observed_tuple_streams_immediately():
    gw = _gateway()
    invocation = _tool_invocation()
    gw._record_stream_safety_observation(invocation, accepted=True)
    assert gw._requires_buffered_stream_validation(invocation) is False


def test_rejection_revokes_observation():
    gw = _gateway()
    invocation = _tool_invocation()
    gw._record_stream_safety_observation(invocation, accepted=True)
    assert gw._requires_buffered_stream_validation(invocation) is False
    gw._record_stream_safety_observation(invocation, accepted=False)
    assert gw._requires_buffered_stream_validation(invocation) is True


def test_observation_is_tuple_scoped():
    gw = _gateway()
    unlocked = _tool_invocation()
    other_surface = _tool_invocation()
    # A different tool surface → different fingerprint → different tuple.
    other_surface.reconciled_request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        tools=[
            CanonicalTool(name="t", description="d", input_schema={"type": "object"}),
            CanonicalTool(name="u", description="d2", input_schema={"type": "object"}),
        ],
        tool_choice=CanonicalToolChoice.auto(),
    )
    gw._record_stream_safety_observation(unlocked, accepted=True)
    assert gw._requires_buffered_stream_validation(unlocked) is False
    assert gw._requires_buffered_stream_validation(other_surface) is True


def test_private_capabilities_ignore_observation():
    """P0-25 stands: the firewall needs buffered final turns regardless of
    observed tool-call validity."""
    gw = _gateway()
    from agent_interop.projection.types import PrivateCapabilityPlan

    invocation = _tool_invocation()
    invocation.private_capabilities = PrivateCapabilityPlan(read_result=True)
    gw._record_stream_safety_observation(invocation, accepted=True)
    assert gw._requires_buffered_stream_validation(invocation) is True


def test_flag_off_ignores_observation():
    gw = _gateway()
    invocation = _tool_invocation()
    invocation.route = SimpleNamespace(
        compatibility=SimpleNamespace(buffer_unverified_streaming=False),
    )
    gw._record_stream_safety_observation(invocation, accepted=True)
    assert gw._requires_buffered_stream_validation(invocation) is False


def test_broken_invocation_does_not_crash_recorder():
    gw = _gateway()
    bad = SimpleNamespace(reconciled_request=None)
    # Must not raise — a defensive recorder never breaks the stream.
    gw._record_stream_safety_observation(bad, accepted=True)


# ─── Engine hook wiring ──────────────────────────────────────────────────────


def _tool_call_chunk(arguments: str) -> dict[str, Any]:
    """One chat.completion.chunk carrying a complete native tool call."""
    return {
        "id": "r1", "object": "chat.completion.chunk", "model": "fake-model",
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": 0, "id": "c1", "type": "function",
                "function": {"name": "t", "arguments": arguments},
            }]},
            "finish_reason": None,
        }],
    }


def _done_chunk(finish_reason: str) -> dict[str, Any]:
    return {
        "id": "r1", "object": "chat.completion.chunk", "model": "fake-model",
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


class _StreamTransport:
    """Serves one SSE stream whose data lines come from ``payloads``.

    Mirrors the gateway's contract: ``stream(...)`` returns an async
    context manager yielding an object with ``status_code`` and
    ``sse_events()``.
    """

    def __init__(self, data_lines: list[str]) -> None:
        self._data_lines = data_lines

    def stream(self, request: Any):  # noqa: ANN201
        from contextlib import asynccontextmanager

        outer = self

        @asynccontextmanager
        async def _ctx():
            yield _SseStream(outer._data_lines)

        return _ctx()


class _SseStream:
    def __init__(self, data_lines: list[str]) -> None:
        self.status_code = 200
        self._data_lines = data_lines

    async def sse_events(self):  # noqa: ANN201
        from agent_interop.transport.sse import SSEFrame

        for line in self._data_lines:
            yield SSEFrame(data=line)

    async def raw_lines(self):  # noqa: ANN201
        return
        yield  # pragma: no cover

    async def __aenter__(self):  # noqa: ANN201
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False


def _stream_setup(payloads: list[dict[str, Any]]) -> Gateway:
    data_lines = [json.dumps(payload) for payload in payloads]
    data_lines.append("[DONE]")
    return Gateway(
        InteropServerConfig(
            probe_on_startup=False, log_level="error", routes={"r": ROUTE},
        ),
        transport=_StreamTransport(data_lines),
    )


def _stream_invocation(gw: Gateway) -> Any:
    from agent_interop.abi import CanonicalGenerationOptions
    from agent_interop.context import RequestContext
    from agent_interop.execution import InteropRequestExecution
    from agent_interop.gateway import ResolvedInvocation
    from agent_interop.repair.invocation import build_invocation_plan
    from agent_interop.upstreams.registry import get_codec

    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="m"),
        generation=CanonicalGenerationOptions(max_output_tokens=64, stream=True),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hi")])],
        tools=[CanonicalTool(
            name="t", description="d",
            input_schema={"type": "object", "properties": {"p": {"type": "string"}},
                          "required": ["p"]},
        )],
        tool_choice=CanonicalToolChoice.required(),
    )
    plan = build_invocation_plan(
        tools=None,
        tool_choice=request.tool_choice,
        route_mode=ToolMode.NATIVE,
        model_profile=None,
        repair_policy=None,
        codec_capabilities=get_codec(UpstreamProtocol.OPENAI_CHAT).capabilities(),
        upstream_tools=list(request.tools),
        validation_tools=list(request.tools),
    )
    return ResolvedInvocation(
        request_context=RequestContext(session_id="s-ss", client_id=""),
        original_request=request,
        reconciled_request=request,
        route=ROUTE,
        backend_metadata=None,
        model_profile=None,
        repair_policy=RepairPolicy(),
        invocation_plan=plan,
        codec=get_codec(UpstreamProtocol.OPENAI_CHAT),
        compatibility_key=None,
        evidence_record=None,
        repair_budget=None,
        execution_record=InteropRequestExecution(),
        runtime_capabilities=gw._static_runtime_capabilities(ROUTE),
        pinned_refs=(),
        authoritative_request=request,
    )


async def test_accepted_stream_records_observation_end_to_end():
    gw = _stream_setup([
        _tool_call_chunk(json.dumps({"p": "x"})),
        _done_chunk("tool_calls"),
    ])
    invocation = _stream_invocation(gw)
    key = gw._stream_safety_key(invocation)
    assert not gw._stream_safety.is_safe(key)

    events = [
        event
        async for event in gw._stream_engine.run_send_stream(
            invocation, invocation.execution_record,
        )
    ]
    assert any(e.type == "tool_use" for e in events)
    assert gw._stream_safety.is_safe(key)


async def test_rejected_stream_revokes_observation_end_to_end():
    gw = _stream_setup([
        _tool_call_chunk('{"wrong": true}'),
        _done_chunk("tool_calls"),
    ])
    invocation = _stream_invocation(gw)
    key = gw._stream_safety_key(invocation)
    gw._stream_safety.record(key)

    events = [
        event
        async for event in gw._stream_engine.run_send_stream(
            invocation, invocation.execution_record,
        )
    ]
    assert any(e.type == "error" for e in events)
    assert not gw._stream_safety.is_safe(key)
