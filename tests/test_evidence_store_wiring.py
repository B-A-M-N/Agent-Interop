"""Tests wiring the EvidenceStore into live Gateway decisions.

These prove the evidence feature (lookup, verified-pack gating, and live
write-back) actually participates in the live request path — today it is only
used by CLI/testing code. Every test that exercises the feature injects an
in-memory ``EvidenceStore(db_path=":memory:")``; none touch the real default
store.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Self

import pytest

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
    FieldAliasPolicy,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.evidence.store import EvidenceStore
from agent_interop.execution import InteropRequestExecution, ToolDecisionRecord
from agent_interop.gateway import Gateway
from agent_interop.replay.types import (
    CompatibilityKey,
    CompatibilityResult,
)
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamResponse

READ_FILE_TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
)

# Matches Claude Code's REAL "Read" tool (name + canonical "file_path"
# field) — see compatibility_packs/claude_code. Kept separate from
# READ_FILE_TOOL above (an arbitrary, internally-consistent fixture used
# by every other test in this file) so only the pack-specific tests below
# depend on the pack's actual real-world tool name and field direction.
CLAUDE_REAL_READ_TOOL = CanonicalTool(
    name="Read",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
)


def _make_gateway(
    store: EvidenceStore | None = None,
) -> Gateway:
    """Gateway routing to an OPENAI_CHAT upstream whose responses are supplied
    by a fake transport (see ``_FakeTransport``)."""
    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "test-route": ModelRoute(
                id="test-route",
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
    return Gateway(config=config, evidence_store=store)


def _make_request(client_id: str = "claude_code") -> CanonicalRequest:
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="test-model"),
        messages=[
            CanonicalMessage(
                role="user",
                content=[CanonicalTextBlock(text="Read the file")],
            )
        ],
        tools=[READ_FILE_TOOL],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )


def _tool_call_body(
    name: str = "read_file", arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = arguments or {"path": "/tmp/x"}
    return {
        "id": "fake-chat-response",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tc_fake_001",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(args),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _text_body(text: str = "Hello") -> dict[str, Any]:
    return {
        "id": "fake-chat-response",
        "object": "chat.completion",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _FakeTransport:
    """Returns a pre-seeded OpenAI Chat-formatted body for every send()."""

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.send_calls: list[PreparedUpstreamRequest] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        self.send_calls.append(request)
        return UpstreamResponse(
            status_code=200,
            headers={},
            body=json.dumps(self._body).encode("utf-8"),
        )

    async def stream(self, request: PreparedUpstreamRequest):
        raise AssertionError("non-streaming tests must not stream")

    async def close(self) -> None:
        pass


def _verified_result(**overrides: Any) -> CompatibilityResult:
    defaults: dict[str, Any] = {
        "tested_at": "2026-07-24T12:00:00+00:00",
        "sample_count": 50,
        "tool_selection_rate": 0.9,
        "valid_call_rate_before_repair": 0.7,
        "valid_call_rate_after_repair": 0.95,
        "task_completion_rate": 0.85,
        "deterministic_repair_rate": 0.25,
        "regeneration_rate": 0.0,
        "rejection_rate": 0.05,
        "streaming_equivalent": True,
        "history_round_trip_valid": True,
        "verified_capabilities": frozenset({"native"}),
        "known_quirks": (),
        "created_at": "2026-07-01T00:00:00+00:00",
        "last_verified_at": "2026-07-24T12:00:00+00:00",
        "manually_verified": True,
        "revoked": False,
        "revocation_reason": "",
    }
    defaults.update(overrides)
    return CompatibilityResult(**defaults)


# ═══════════════════════════════════════════════════════════════════════
# Part 1 regression: the AttributeError fix in _prepare_invocation
# ═══════════════════════════════════════════════════════════════════════


class TestCompatibilityKeyBugFix:
    def test_nonempty_client_id_does_not_crash(self):
        """Passing a RequestContext with a non-empty client_id through
        _prepare_invocation must not raise, and the resulting
        compatibility_key must carry that client_id.

        Before the fix, ``request_context=getattr(context, 'client_id', '')``
        passed the string client_id itself to build_compatibility_key, which
        then crashed with ``AttributeError: 'str' object has no attribute
        'client_id'`` whenever client_id was non-empty."""
        gw = _make_gateway()
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code", client_version="2.1.0")
        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=record,
        )
        assert invocation.compatibility_key is not None
        assert invocation.compatibility_key.client_id == "claude_code"
        # client_version should also flow through now (was lost before the fix).
        assert invocation.compatibility_key.client_version == "2.1.0"

    def test_empty_client_id_still_produces_valid_key(self):
        """An empty client_id must not crash either (the old getattr fallback
        returned '' for this case)."""
        gw = _make_gateway()
        canonical = _make_request()
        ctx = RequestContext()
        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=record,
        )
        assert invocation.compatibility_key is not None
        assert invocation.compatibility_key.client_id == ""


# ═══════════════════════════════════════════════════════════════════════
# Part 2B: evidence lookup in _prepare_invocation
# ═══════════════════════════════════════════════════════════════════════


class TestEvidenceLookup:
    def test_no_store_means_evidence_record_none(self):
        """Without an injected store, evidence_record is always None and no
        lookup is performed — the default path is unchanged."""
        gw = _make_gateway(store=None)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=record,
        )
        assert invocation.evidence_record is None

    def test_verified_record_is_picked_up(self):
        """A manually-verified, non-revoked, non-stale record with sufficient
        sample count IS picked up as evidence_record."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        # Discover the exact key the gateway computes for this request, then
        # seed the store with a verified record for that tuple.
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        key = probe.compatibility_key
        assert key is not None
        store.store_result(key, _verified_result())

        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=record,
        )
        assert invocation.evidence_record is not None
        assert invocation.evidence_record.manually_verified is True
        assert invocation.evidence_record.sample_count == 50

    def test_unverified_record_is_not_picked_up(self):
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result(manually_verified=False))
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert invocation.evidence_record is None

    def test_revoked_record_is_not_picked_up(self):
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result(revoked=True, revocation_reason="bad"))
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert invocation.evidence_record is None

    def test_stale_record_is_not_picked_up(self):
        """A record older than passes_expiry_hours (default 720h) is stale and
        must NOT be picked up."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(
            probe.compatibility_key,
            _verified_result(
                tested_at="2020-01-01T00:00:00+00:00",
                last_verified_at="2020-01-01T00:00:00+00:00",
            ),
        )
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert invocation.evidence_record is None

    def test_low_sample_count_record_is_not_picked_up(self):
        """A verified record below MIN_EVIDENCE_SAMPLE_COUNT (5) is too thin to
        trust and must NOT be picked up."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result(sample_count=2))
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert invocation.evidence_record is None


# ═══════════════════════════════════════════════════════════════════════
# Part 2C: verified evidence reaches the transaction context
# ═══════════════════════════════════════════════════════════════════════


class TestCompatibilityVerifiedPropagation:
    def test_verified_evidence_yields_compatibility_verified_true(self):
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result())

        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert invocation.evidence_record is not None
        tx_ctx = gw._build_transaction_context(invocation, canonical)
        assert tx_ctx.compatibility_verified is True

    def test_no_evidence_yields_compatibility_verified_false(self):
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert invocation.evidence_record is None
        tx_ctx = gw._build_transaction_context(invocation, canonical)
        assert tx_ctx.compatibility_verified is False


# ═══════════════════════════════════════════════════════════════════════
# Part 2C: compatibility-pack gating on verified evidence
# ═══════════════════════════════════════════════════════════════════════


class TestPackGatingOnResolvedIdentity:
    """Registered compatibility packs (maintainer-authored, static — see
    repair/aliases.py's module docstring) gate on a properly RESOLVED
    client identity, not on compatibility_verified. That flag is reserved
    for a hypothetical future dynamic/learned alias source; see
    tests/test_alias_policy.py for the full behavior matrix."""

    def test_pack_not_applied_without_a_compatibility_key_at_all(self):
        from agent_interop.repair.aliases import get_aliases_for_tool

        result = get_aliases_for_tool(
            "Read", CLAUDE_REAL_READ_TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        # Only schema x-aliases should be present, never pack aliases like
        # "path".
        assert "path" not in result.get("file_path", [])

    def test_pack_applied_with_resolved_identity_and_no_verification_flag(self):
        """The actual gate: a sufficiently-populated key (client_id plus
        at least one other real dimension) is enough — verification is
        never passed here at all."""
        from agent_interop.repair.aliases import get_aliases_for_tool
        from agent_interop.replay.types import CompatibilityKey

        key = CompatibilityKey(client_id="claude_code", model_id="test-model")
        result = get_aliases_for_tool(
            "Read", CLAUDE_REAL_READ_TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "path" in result.get("file_path", [])


# ═══════════════════════════════════════════════════════════════════════
# Part 2D: live write-back (read-modify-write, never naive overwrite)
# ═══════════════════════════════════════════════════════════════════════


class TestLiveWriteBack:
    @pytest.mark.asyncio
    async def test_write_back_merges_across_two_requests(self):
        """The single most important test: driving two requests (same route /
        profile / model, tools offered) through the live gateway with an
        in-memory store must MERGE the second observation into the first —
        sample_count goes 1->2 and rates reflect a weighted average — rather
        than resetting to sample_count=1."""
        store = EvidenceStore(db_path=":memory:")
        body = _tool_call_body("read_file", {"path": "/tmp/x"})

        # First request: seed the store with a fat verified record so we can
        # observe the merge math against a known prior.
        gw = _make_gateway(store=store)
        gw._transport = _FakeTransport(body)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        prior = _verified_result(
            sample_count=10,
            tool_selection_rate=1.0,
            valid_call_rate_before_repair=0.6,
            valid_call_rate_after_repair=1.0,
            rejection_rate=0.0,
        )
        store.store_result(probe.compatibility_key, prior)

        resp1 = await gw.handle_request(canonical, ctx)
        assert resp1.error is None

        after_first = store.get_result(probe.compatibility_key)
        assert after_first is not None
        # One live request merged into sample_count=10 -> 11.
        assert after_first.sample_count == 11, (
            f"after first request expected sample_count 11, got {after_first.sample_count}"
        )
        # The live request produced an accepted call (valid_after_repair=1.0),
        # so the weighted average of valid_call_rate_after_repair should rise:
        # (1.0*10 + 1.0)/11 == 1.0 here since both are 1.0.
        assert after_first.valid_call_rate_after_repair == pytest.approx(1.0)

        # Second request: same route/profile/model -> same key. sample_count
        # must go 11 -> 12 (merge), NOT reset to 1.
        gw2 = _make_gateway(store=store)
        gw2._transport = _FakeTransport(body)
        resp2 = await gw2.handle_request(canonical, ctx)
        assert resp2.error is None

        after_second = store.get_result(probe.compatibility_key)
        assert after_second is not None
        assert after_second.sample_count == 12, (
            f"after second request expected sample_count 12 (merge), "
            f"got {after_second.sample_count} — write-back is overwriting, not merging"
        )

    @pytest.mark.asyncio
    async def test_write_back_preserves_manually_verified(self):
        """A pre-existing record with manually_verified=True must stay
        manually_verified=True after a live write-back updates its
        sample_count / rates — write-back never auto-verifies AND never
        de-verifies."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeTransport(_tool_call_body("read_file", {"path": "/tmp/x"}))
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result(manually_verified=True))

        await gw.handle_request(canonical, ctx)

        result = store.get_result(probe.compatibility_key)
        assert result is not None
        assert result.manually_verified is True, (
            "write-back must not flip manually_verified to False"
        )
        # P0.3: live write-back must NOT refresh the certification clock.
        # last_verified_at must remain exactly what the seed record had.
        assert result.last_verified_at == "2026-07-24T12:00:00+00:00", (
            "write-back must not modify last_verified_at (certification clock); "
            f"got {result.last_verified_at!r}"
        )

    @pytest.mark.asyncio
    async def test_no_write_back_when_no_tools_offered(self):
        """Plain conversational requests (no tools) must not trigger a
        write-back — there is nothing meaningful to observe about tool-calling
        compatibility."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeTransport(_text_body("Hi"))
        canonical = _make_request()
        canonical.tools = []  # no tools offered
        ctx = RequestContext(client_id="claude_code")

        await gw.handle_request(canonical, ctx)

        # Nothing should have been written for this key.
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert store.get_result(probe.compatibility_key) is None

    @pytest.mark.asyncio
    async def test_no_write_back_on_backend_error(self):
        """When the request ends in a backend/transport error, no write-back
        must occur — an upstream 500 tells you nothing about model tool-calling
        behavior."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeTransport(_tool_call_body("read_file", {"path": "/tmp/x"}))
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        # Force a backend error by making send() raise.
        async def _boom(req):
            raise RuntimeError("upstream 500")

        gw._transport.send = _boom
        with pytest.raises(RuntimeError, match="upstream 500"):
            await gw.handle_request(canonical, ctx)

        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        assert store.get_result(probe.compatibility_key) is None


class _FakeSseStream:
    """Minimal ``UpstreamStream`` stand-in yielding pre-seeded SSE data lines.

    Only the surface the gateway's SSE path touches is implemented:
    ``status_code`` and ``sse_events()`` (``_iter_frame_data`` reads
    ``frame.data`` for the SSE framing). Each element of ``data_lines`` is
    the literal text of a single ``data:`` payload.
    """

    def __init__(self, data_lines: list[str], status_code: int = 200) -> None:
        self.status_code = status_code
        self._data_lines = data_lines

    async def sse_events(self):
        from agent_interop.transport.sse import SSEFrame

        for line in self._data_lines:
            yield SSEFrame(data=line)

    async def raw_lines(self):
        return
        yield  # pragma: no cover

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _FakeSseTransport:
    """Transport that yields a pre-seeded SSE stream for every request.

    ``data_lines`` are the literal ``data:`` payloads (already-encoded
    strings). Use ``json.dumps(...)`` for well-formed frames and a
    non-JSON string to emulate a malformed frame.
    """

    def __init__(self, data_lines: list[str], status_code: int = 200) -> None:
        self._data_lines = data_lines
        self._status_code = status_code

    @asynccontextmanager
    async def stream(
        self, request: PreparedUpstreamRequest,
    ) -> AsyncIterator[_FakeSseStream]:
        yield _FakeSseStream(self._data_lines, self._status_code)

    async def close(self) -> None:
        return None


# A normal OpenAI-style streamed TEXT completion: a non-terminal text delta
# followed by a terminal frame carrying finish_reason="stop". The terminal
# frame is what triggers the ``is_stream_complete`` branch — the NORMAL
# completion path for most codecs.
NORMAL_TEXT_STREAM = [
    json.dumps({"choices": [{
        "delta": {"content": "Hello"},
        "index": 0,
        "finish_reason": None,
    }]}),
    json.dumps({"choices": [{
        "delta": {},
        "index": 0,
        "finish_reason": "stop",
    }]}),
]

# A stream that ends WITHOUT ever sending a terminal frame — the upstream
# just closes the connection after a text delta. The ``async for`` over
# frames exhausts naturally, exercising the post-loop path.
TEXT_STREAM_NO_TERMINATOR = [
    json.dumps({"choices": [{
        "delta": {"content": "Hello"},
        "index": 0,
        "finish_reason": None,
    }]}),
]


# ═══════════════════════════════════════════════════════════════════════
# Part 2E: live write-back on the streaming path
# ═══════════════════════════════════════════════════════════════════════


class TestLiveWriteBackStreaming:
    """Live write-back must fire on BOTH successful streaming completion paths.

    The gateway has two success exits inside ``_handle_stream_send``:

    A. The *normal* path — ``codec.is_stream_complete(frame_data)`` returns
       ``True`` on a terminal frame (``finish_reason`` / ``[DONE]``) mid-loop.
       This is how OpenAI-style streams end. Before the fix, this branch
       returned WITHOUT writing back.
    B. The *natural loop-end* path — the ``async for`` over frames exhausts
       because the upstream closed the connection without ever sending a
       terminal frame. This branch already wrote back.

    Evidence write-back is gated identically on both: store configured AND
    ``canonical.tools`` non-empty.
    """

    @pytest.mark.asyncio
    async def test_write_back_fires_on_normal_completion_path(self):
        """THE regression test for the branch-placement bug.

        Seed an in-memory store with sample_count=50 for the exact key a
        streaming request with tools resolves to, then drive a NORMAL
        OpenAI-style completion (text chunk + a ``finish_reason:"stop"``
        terminal frame — NOT an abrupt disconnect) through ``handle_stream``.
        The seeded record's sample_count must advance 50 -> 51.

        Before the fix this asserted 50 (write-back silently never fired),
        because the ``is_stream_complete`` branch returned early, before the
        write-back call that only lived at the natural-loop-end.
        """
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeSseTransport(NORMAL_TEXT_STREAM)

        canonical = _make_request()  # offers READ_FILE_TOOL
        ctx = RequestContext(client_id="claude_code")

        # Discover the exact key the gateway computes for a STREAMING request
        # with tools (the key incorporates streaming=True), then seed it.
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=True, execution=InteropRequestExecution(),
        )
        key = probe.compatibility_key
        assert key is not None
        store.store_result(key, _verified_result())  # sample_count=50

        # Drive the normal streaming completion end-to-end.
        events: list[str] = []
        async for event in gw.handle_stream(canonical, ctx):
            events.append(event.type)

        # A clean completion: text_delta + message_stop, no error.
        assert "error" not in events, f"unexpected error in events: {events}"
        assert "text_delta" in events
        assert "message_stop" in events

        # THE assertion: write-back fired on the normal path -> 50 -> 51.
        after = store.get_result(key)
        assert after is not None
        assert after.sample_count == 51, (
            f"normal-completion write-back did not fire: expected sample_count"
            f" 51, got {after.sample_count}"
        )

    @pytest.mark.asyncio
    async def test_write_back_fires_at_natural_loop_end(self):
        """The abnormal path — upstream closes WITHOUT ever sending a terminal
        frame. The ``async for`` over frames exhausts naturally and the
        post-loop write-back fires. This path worked before the fix and must
        keep working (50 -> 51)."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeSseTransport(TEXT_STREAM_NO_TERMINATOR)

        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        probe = gw._prepare_invocation(
            canonical, ctx, streaming=True, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result())

        events: list[str] = []
        async for event in gw.handle_stream(canonical, ctx):
            events.append(event.type)

        assert "error" not in events
        assert "message_stop" in events

        after = store.get_result(probe.compatibility_key)
        assert after is not None
        assert after.sample_count == 51, (
            f"natural-loop-end write-back regressed: expected 51, "
            f"got {after.sample_count}"
        )

    @pytest.mark.asyncio
    async def test_no_write_back_when_no_tools_offered_streaming(self):
        """Stream guard: tools offered is required. A streaming completion
        with no tools must not write back."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeSseTransport(NORMAL_TEXT_STREAM)

        canonical = _make_request()
        canonical.tools = []  # no tools offered
        ctx = RequestContext(client_id="claude_code")

        async for _ in gw.handle_stream(canonical, ctx):
            pass

        probe = gw._prepare_invocation(
            canonical, ctx, streaming=True, execution=InteropRequestExecution(),
        )
        assert store.get_result(probe.compatibility_key) is None

    @pytest.mark.asyncio
    async def test_no_write_back_on_upstream_error_streaming(self):
        """Error path: upstream returns status >= 400 on the stream. No
        write-back must occur (an HTTP error tells you nothing about tool-
        calling behavior)."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        gw._transport = _FakeSseTransport([], status_code=500)

        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        events: list[str] = []
        async for event in gw.handle_stream(canonical, ctx):
            events.append(event.type)

        assert "error" in events
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=True, execution=InteropRequestExecution(),
        )
        assert store.get_result(probe.compatibility_key) is None

    @pytest.mark.asyncio
    async def test_no_write_back_on_malformed_frame_streaming(self):
        """Error path: a malformed frame with open tool state aborts the
        stream with an error and must NOT write back. We open a pending tool
        call with a tool fragment, then send a non-JSON ``data:`` line which
        ``_iter_frame_data`` yields as ``None`` (malformed). The gateway
        detects pending tool state, fails the batch, and returns an error
        BEFORE any write-back call on either success path."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        # Frame 1: a tool-call fragment (opens a pending call for read_file).
        # Frame 2: a non-JSON payload -> _iter_frame_data yields None.
        data_lines = [
            json.dumps({"choices": [{
                "delta": {"tool_calls": [{
                    "index": 0,
                    "id": "tc_fake_001",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": ""},
                }]},
                "index": 0,
                "finish_reason": None,
            }]}),
            "this is not valid json {{{",
        ]
        gw._transport = _FakeSseTransport(data_lines)

        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        events: list[str] = []
        async for event in gw.handle_stream(canonical, ctx):
            events.append(event.type)

        # The malformed frame aborts with an error; no write-back occurs.
        assert "error" in events, f"expected an error event, got {events}"
        probe = gw._prepare_invocation(
            canonical, ctx, streaming=True, execution=InteropRequestExecution(),
        )
        assert store.get_result(probe.compatibility_key) is None, (
            "write-back must NOT fire after a malformed-frame error"
        )


# ═══════════════════════════════════════════════════════════════════════
# Part 2A: opt-in only — no store means no evidence behavior at all
# ═══════════════════════════════════════════════════════════════════════


class TestOptInOnly:
    @pytest.mark.asyncio
    async def test_gateway_without_store_behaves_identically(self):
        """A Gateway constructed with NO evidence store must behave identically
        to before this change: no lookups, no writes, evidence_record always
        None, compatibility_verified always False. We drive a real request
        through with a fake transport and assert the response is unaffected."""
        store = EvidenceStore(db_path=":memory:")
        # Seed the store with a verified record so a store-using gateway WOULD
        # pick it up — proving the record is real and reachable.
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")
        probe_gw = _make_gateway(store=store)
        probe = probe_gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        store.store_result(probe.compatibility_key, _verified_result())

        # Now a gateway with NO store.
        gw = _make_gateway(store=None)
        gw._transport = _FakeTransport(_tool_call_body("read_file", {"path": "/tmp/x"}))

        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=record,
        )
        assert invocation.evidence_record is None
        assert invocation.compatibility_key is not None  # key still computed
        tx_ctx = gw._build_transaction_context(invocation, canonical)
        assert tx_ctx.compatibility_verified is False

        resp = await gw.handle_request(canonical, ctx)
        assert resp.error is None
        # And the store was never written by the no-store gateway.
        assert store.get_result(probe.compatibility_key).sample_count == 50  # untouched


# ═══════════════════════════════════════════════════════════════════════
# P0.3 / P1.11 regression tests — counter-based rate aggregation and
# the certification-clock fix.
# ═══════════════════════════════════════════════════════════════════════


def _decisions(statuses: list[str], accepted_flags: list[bool] | None = None) -> list[ToolDecisionRecord]:
    """Build a list of ToolDecisionRecord from outcome-status strings.

    ``accepted_flags`` defaults to True for every decision when omitted."""
    if accepted_flags is None:
        accepted_flags = [True] * len(statuses)
    assert len(accepted_flags) == len(statuses)
    return [
        ToolDecisionRecord(
            tool_name="read_file",
            candidate_id=f"tc_{i:03d}",
            outcome_status=status,
            accepted=acc,
        )
        for i, (status, acc) in enumerate(zip(statuses, accepted_flags))
    ]


def _observe(
    gw: Gateway,
    store: EvidenceStore,
    key: CompatibilityKey,
    decisions: list[ToolDecisionRecord],
) -> None:
    """Fire one live observation directly into the inner write-back path."""
    execution = InteropRequestExecution()
    execution.tool_decisions = decisions
    # Any prepared invocation with the right key suffices; the inner method
    # only reads invocation.compatibility_key.
    invocation = gw._prepare_invocation(
        _make_request(), RequestContext(client_id="claude_code"),
        streaming=False, execution=InteropRequestExecution(),
    )
    # Force the key under test (the probe's key is structurally identical in
    # the single-route tests, but pinning it makes intent explicit).
    object.__setattr__(invocation, "compatibility_key", key)
    gw._record_evidence_observation_inner(invocation, execution, store)


class TestCertificationClockNotResetByLiveTraffic:
    """P0.3 core regression: live traffic must not refresh the staleness clock."""

    @pytest.mark.asyncio
    async def test_live_traffic_does_not_reset_staleness_clock(self):
        """A manually-verified record with an OLD last_verified_at is stale.
        Driving several live observations against the same key must leave the
        record stale afterward — i.e. the certification clock is untouched."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        canonical = _make_request()
        ctx = RequestContext(client_id="claude_code")

        probe = gw._prepare_invocation(
            canonical, ctx, streaming=False, execution=InteropRequestExecution(),
        )
        key = probe.compatibility_key
        assert key is not None

        # Seed a manually-verified record whose certification timestamp is
        # well beyond the default 720h expiry window.
        old_cert = "2020-01-01T00:00:00+00:00"
        store.store_result(
            key,
            CompatibilityResult(
                manually_verified=True,
                last_verified_at=old_cert,
                tested_at="2020-01-01T00:00:00+00:00",
                created_at="2020-01-01T00:00:00+00:00",
                sample_count=50,
                tool_selection_rate=0.9,
                valid_call_rate_before_repair=0.7,
                valid_call_rate_after_repair=0.95,
                deterministic_repair_rate=0.25,
                regeneration_rate=0.0,
                rejection_rate=0.05,
            ),
        )

        # Sanity: the seeded record is stale right now.
        assert store.is_stale(key) is True

        # Drive several live observations (each with an accepted tool call).
        for _ in range(5):
            _observe(gw, store, key, _decisions([RepairStatus.VALID_UNCHANGED.value]))

        after = store.get_result(key)
        assert after is not None
        # The certification timestamp is untouched ...
        assert after.last_verified_at == old_cert, (
            f"live traffic moved last_verified_at from {old_cert!r} to "
            f"{after.last_verified_at!r}"
        )
        # ... so the record is STILL stale — the core P0.3 guarantee.
        assert store.is_stale(key) is True, (
            "live traffic must not reset the certification clock; the record "
            "should remain stale"
        )
        # sample_count advanced (write-back still happened), proving the
        # observations were recorded, just not as re-certifications.
        assert after.sample_count == 55


class TestPerCallRateWeighting:
    """P1.11: rates are per-tool-call, not per-request."""

    @pytest.mark.asyncio
    async def test_imbalanced_request_sizes_are_weighted_per_call(self):
        """One request with 1 accepted decision and one request with 10
        decisions (7 accepted) must yield a valid_call_rate_after_repair of
        (1 + 7) / (1 + 10) = 8/11, NOT the per-request mean of the two
        request-level rates (1.0 + 0.7) / 2 = 0.85."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        probe = gw._prepare_invocation(
            _make_request(), RequestContext(client_id="claude_code"),
            streaming=False, execution=InteropRequestExecution(),
        )
        key = probe.compatibility_key
        assert key is not None

        # Request 1: 1 decision, accepted (rate 1.0 at request level).
        _observe(gw, store, key, _decisions([RepairStatus.VALID_UNCHANGED.value]))
        # Request 2: 10 decisions, 7 accepted + 3 rejected (rate 0.7 at
        # request level). The 3 rejected ones are marked REPAIRED with
        # accepted=False.
        statuses = (
            [RepairStatus.VALID_UNCHANGED.value] * 7
            + [RepairStatus.REPAIRED.value] * 3
        )
        accepted = [True] * 7 + [False] * 3
        _observe(gw, store, key, _decisions(statuses, accepted))

        result = store.get_result(key)
        assert result is not None
        # Per-call weighting: 8 accepted out of 11 candidates.
        assert result.valid_call_rate_after_repair == pytest.approx(8 / 11), (
            f"expected per-call 8/11, got {result.valid_call_rate_after_repair}"
        )
        # Counters must reflect the 11 total candidates, not 2 requests.
        assert result.candidate_count == 11
        assert result.accepted_count == 8
        assert result.sample_count == 2
        # The per-request naive mean would be 0.85 — make sure we did NOT get it.
        assert result.valid_call_rate_after_repair != pytest.approx(0.85)


class TestPreV4MigrationSeeding:
    """Pre-v4 records (rates but no counters) must be seeded, not discarded."""

    @pytest.mark.asyncio
    async def test_pre_v4_record_counters_are_seeded_from_rates(self):
        """A record written before this fix has sample_count > 0 but
        candidate_count == 0. One live observation must seed the counters
        from the stored rates and then accumulate — the history must not be
        silently discarded."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        probe = gw._prepare_invocation(
            _make_request(), RequestContext(client_id="claude_code"),
            streaming=False, execution=InteropRequestExecution(),
        )
        key = probe.compatibility_key
        assert key is not None

        # Pre-v4 record: rates present, counters all zero (the default).
        store.store_result(
            key,
            CompatibilityResult(
                sample_count=10,
                tool_selection_rate=0.9,
                valid_call_rate_before_repair=0.6,
                valid_call_rate_after_repair=0.8,
                deterministic_repair_rate=0.2,
                regeneration_rate=0.1,
                rejection_rate=0.2,
            ),
        )
        pre = store.get_result(key)
        assert pre is not None
        assert pre.candidate_count == 0  # the pre-v4 shape

        # One live observation: a single accepted, valid-unchanged call.
        _observe(gw, store, key, _decisions([RepairStatus.VALID_UNCHANGED.value]))

        result = store.get_result(key)
        assert result is not None
        # sample_count advanced by one request.
        assert result.sample_count == 11
        # Counters were seeded from rates (10 candidates) then +1 observed
        # candidate => 11. Crucially NOT 1 (which would mean seeding failed).
        assert result.candidate_count == 11, (
            f"expected seeded candidate_count 11, got {result.candidate_count}"
        )
        # accepted seeded from round(0.8*10)=8, plus 1 observed => 9.
        assert result.accepted_count == 9
        # valid_unchanged seeded from round(0.6*10)=6, plus 1 observed => 7.
        assert result.valid_unchanged_count == 7
        # Rate re-derived from the merged counters: 9/11.
        assert result.valid_call_rate_after_repair == pytest.approx(9 / 11)

    @pytest.mark.asyncio
    async def test_no_false_positive_seeding_on_no_selection_then_tool_call(self):
        """A genuine v4 record whose first observation was a no-selection
        request (text-only reply, n==0) must NOT be mistaken for a pre-v4
        record and re-seeded. Only the real tool-call observation should
        contribute candidates.

        Reproduces the false-positive: after one no-selection observation the
        record legitimately has candidate_count==0 and sample_count==1. Before
        the fix, the guard ``candidate_count == 0 and sample_count > 0`` fired
        on that record and treated the no-selection request as a phantom
        candidate."""
        store = EvidenceStore(db_path=":memory:")
        gw = _make_gateway(store=store)
        probe = gw._prepare_invocation(
            _make_request(), RequestContext(client_id="claude_code"),
            streaming=False, execution=InteropRequestExecution(),
        )
        key = probe.compatibility_key
        assert key is not None

        # Observation 1: a real no-selection request (text, no tool call).
        # Under correct v4 logic this yields sample_count=1, candidate_count=0,
        # no_selection_request_count=1, all rates 0.0.
        _observe(gw, store, key, _decisions([]))

        after_no_sel = store.get_result(key)
        assert after_no_sel is not None
        assert after_no_sel.sample_count == 1
        assert after_no_sel.candidate_count == 0
        assert after_no_sel.no_selection_request_count == 1

        # Observation 2: one accepted, valid-unchanged tool call.
        _observe(gw, store, key, _decisions([RepairStatus.VALID_UNCHANGED.value]))

        result = store.get_result(key)
        assert result is not None
        # Two requests observed in total.
        assert result.sample_count == 2
        # Exactly one candidate — the real tool call. NOT 2 (which would mean
        # the no-selection request was wrongly seeded as a phantom candidate).
        assert result.candidate_count == 1, (
            f"expected candidate_count 1, got {result.candidate_count} "
            "(false-positive seeding on a real v4 no-selection record)"
        )
        # One accepted candidate out of one candidate total => rate 1.0.
        assert result.accepted_count == 1
        assert result.valid_call_rate_after_repair == pytest.approx(1.0)
