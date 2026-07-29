"""Gateway-level integration tests proving architectural features participate in live paths.

Tests:
- Nested repair through full gateway flow
- Provider metadata preserved through Gateway._assemble_response
- Multi-route probing
- Evidence store round-trip
- Recursive diff enforcement
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pytest

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalReasoningBlock,
    CanonicalRequest,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    ProviderMetadata,
    RawToolCallCandidate,
    RepairOutcome,
    RepairStatus,
    RepairStep,
    SchemaIssue,
    ToolCallCorrection,
    ToolCallDecision,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairConfig,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.errors import InteropErrorCode
from agent_interop.gateway import Gateway
from agent_interop.repair.paths import (
    _MISSING,
    delete_at_path,
    diff_paths,
    get_at_path,
    set_at_path,
)
from agent_interop.transaction import ToolBatchDecision
from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamResponse
from agent_interop.upstreams.codec import DecodedModelResponse

# ─── Path helpers ──────────────────────────────────────────────────────────


class TestPathHelpers:
    def test_get_at_path_nested(self):
        obj = {"a": {"b": {"c": 42}}}
        assert get_at_path(obj, ["a", "b", "c"]) == 42

    def test_get_at_path_array(self):
        obj = {"items": [10, 20, 30]}
        assert get_at_path(obj, ["items", 1]) == 20

    def test_get_at_path_missing(self):
        obj = {"a": 1}
        assert get_at_path(obj, ["b"]) is _MISSING

    def test_set_at_path_nested(self):
        obj = {"a": {"b": "old"}}
        result = set_at_path(obj, ["a", "b"], "new")
        assert result == {"a": {"b": "new"}}
        assert obj == {"a": {"b": "old"}}  # immutable

    def test_set_at_path_creates_intermediate(self):
        """Intermediate containers must NOT be silently fabricated (item 51).

        When the path traverses a non-existent key, the operation is a no-op
        rather than inventing {"a": {"b": "val"}} out of nothing.
        """
        obj = {"x": 1}
        result = set_at_path(obj, ["a", "b"], "val")
        # "a" does not exist — should not fabricate it
        assert result == {"x": 1}

    def test_set_at_path_existing_intermediate(self):
        """When the intermediate container exists, the set should succeed."""
        obj = {"a": {"c": 1}}
        result = set_at_path(obj, ["a", "b"], "val")
        assert result["a"]["b"] == "val"
        assert result["a"]["c"] == 1  # existing key preserved

    def test_delete_at_path(self):
        obj = {"a": 1, "b": 2}
        result = delete_at_path(obj, ["a"])
        assert result == {"b": 2}
        assert obj == {"a": 1, "b": 2}  # immutable

    def test_delete_at_path_nested(self):
        obj = {"config": {"x": 1, "y": 2}}
        result = delete_at_path(obj, ["config", "x"])
        assert result == {"config": {"y": 2}}

    def test_diff_paths_simple(self):
        before = {"a": 1, "b": 2}
        after = {"a": 1, "b": 3}
        assert diff_paths(before, after) == {("b",)}

    def test_diff_paths_nested(self):
        before = {"config": {"x": 1, "y": 2}}
        after = {"config": {"x": 1, "y": 99}}
        assert diff_paths(before, after) == {("config", "y")}

    def test_diff_paths_added_key(self):
        before = {"a": 1}
        after = {"a": 1, "b": 2}
        assert diff_paths(before, after) == {("b",)}

    def test_diff_paths_type_change(self):
        before = {"a": "hello"}
        after = {"a": ["hello"]}
        changed = diff_paths(before, after)
        assert ("a",) in changed


# ─── Provider metadata round-trip ──────────────────────────────────────────


class TestProviderMetadataRoundTrip:
    def test_reasoning_block_preserves_metadata_through_assembly(self):
        """Decoded blocks are preserved directly — metadata is not stripped."""
        from agent_interop.upstreams.openai_chat import OpenAIChatCodec
        codec = OpenAIChatCodec()

        body = {
            "choices": [{
                "message": {
                    "content": "The answer.",
                    "reasoning_content": "Let me think...",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }
        decoded = codec.decode_response(body)

        # Verify metadata is attached
        reasoning_block = decoded.content[0]
        assert isinstance(reasoning_block, CanonicalReasoningBlock)
        assert reasoning_block.provider_metadata is not None
        assert reasoning_block.provider_metadata.origin_protocol == "openai_chat"
        assert reasoning_block.provider_metadata.metadata_kind == "reasoning_content"
        assert reasoning_block.provider_metadata.forwarding_policy == "preserve_if_compatible"

    def test_forwarding_policy_drop_suppresses_rendering(self):
        from agent_interop.upstreams.openai_chat import _render_message

        msg = CanonicalMessage(
            role="assistant",
            content=[
                CanonicalReasoningBlock(
                    content="secret thoughts",
                    provider_metadata=ProviderMetadata(
                        origin_protocol="internal",
                        metadata_kind="reasoning_content",
                        forwarding_policy="drop",
                    ),
                ),
                CanonicalTextBlock(text="Hello"),
            ],
        )
        rendered = _render_message(msg)
        assert "reasoning_content" not in rendered
        assert rendered["content"] == "Hello"

    def test_forwarding_policy_preserve_renders(self):
        from agent_interop.upstreams.openai_chat import _render_message

        msg = CanonicalMessage(
            role="assistant",
            content=[
                CanonicalReasoningBlock(
                    content="thinking...",
                    provider_metadata=ProviderMetadata(
                        origin_protocol="openai_chat",
                        metadata_kind="reasoning_content",
                        forwarding_policy="preserve_if_compatible",
                    ),
                ),
                CanonicalTextBlock(text="Result"),
            ],
        )
        rendered = _render_message(msg)
        assert rendered["reasoning_content"] == "thinking..."
        assert rendered["content"] == "Result"


# ─── MVP-04: readiness reflects real backend/model state ──────────────────


def _gateway_with_mocked_probe(handler):
    """Build a single-route Gateway whose transport is backed by an
    httpx.MockTransport, so _probe_routes() exercises the real probe →
    readiness() pipeline against a controlled fake response."""
    import httpx

    from agent_interop.transport.http import UpstreamTransport

    config = InteropServerConfig(
        probe_on_startup=False,
        default_route_id="default",
        routes={
            "default": ModelRoute(
                id="default",
                client_model_aliases=["test-model"],
                upstream_model="test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:11434",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
            ),
        },
    )
    transport = UpstreamTransport(max_retries=0)
    transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Gateway(config, transport=transport)


class TestReadinessReflectsBackendState:
    @pytest.mark.asyncio
    async def test_ready_when_model_present(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        readiness = gw.readiness()

        assert readiness["ready"] is True
        entry = readiness["routes"]["default"]
        assert entry["reachable"] is True
        assert entry["model_present"] is True
        await gw.close()

    @pytest.mark.asyncio
    async def test_not_ready_when_model_missing(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "some-other-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        readiness = gw.readiness()

        assert readiness["ready"] is False
        entry = readiness["routes"]["default"]
        assert entry["reachable"] is True
        assert entry["model_present"] is False
        assert "test-model" in entry["reason"]
        await gw.close()

    @pytest.mark.asyncio
    async def test_not_ready_when_backend_unreachable(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        readiness = gw.readiness()

        assert readiness["ready"] is False
        entry = readiness["routes"]["default"]
        assert entry["reachable"] is False
        assert "unreachable" in entry["reason"]
        await gw.close()

    @pytest.mark.asyncio
    async def test_not_ready_when_unauthenticated_401(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "unauthorized"})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        readiness = gw.readiness()

        assert readiness["ready"] is False
        entry = readiness["routes"]["default"]
        assert entry["reachable"] is True
        assert entry["authenticated"] is False
        await gw.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [403, 404, 429, 500, 503])
    async def test_not_ready_on_error_status(self, status: int):
        """A reachable backend returning a non-200/401 error status (403,
        404, 429, 500, 503) must not be reported ready — reachability alone
        used to be conflated with authenticated readiness."""
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(status)

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        readiness = gw.readiness()

        assert readiness["ready"] is False
        entry = readiness["routes"]["default"]
        assert entry["reachable"] is True
        assert entry["authenticated"] is False
        await gw.close()

    def test_unprobed_route_reports_not_ready_not_false_ok(self):
        """A route with no probe on record must never look silently healthy."""
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            default_route_id="default",
            routes={
                "default": ModelRoute(
                    id="default",
                    client_model_aliases=["test-model"],
                    upstream_model="test-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        gw = Gateway(config)
        readiness = gw.readiness()
        assert readiness["ready"] is False
        assert readiness["routes"]["default"]["reason"] == "not probed"

    @pytest.mark.asyncio
    async def test_model_present_uses_tag_aware_matching_not_exact_string(self):
        """Re-audit P1#13: model_present previously did an exact `in`
        check — a configured 'test-model' against a backend reporting
        'test-model:latest' was reported MISSING even though it's the
        same model. Must use the same tag-aware normalizer as the
        launcher (interop.model_names)."""
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "test-model:latest"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        readiness = gw.readiness()
        assert readiness["routes"]["default"]["model_present"] is True
        await gw.close()

    @pytest.mark.asyncio
    async def test_profile_resolved_reflects_a_real_match_not_hardcoded_true(self):
        """Re-audit P1#13: profile_resolved was hardcoded True the moment
        codec resolution succeeded, regardless of whether the model
        actually matched any real profile — every route always reported
        resolved. A model with no builtin/explicit profile match (falls
        back to the conservative generic-fallback tier) must report
        profile_resolved=False, with profile_source recorded for
        diagnostics."""
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes()
        entry = gw.readiness()["routes"]["default"]
        # "test-model" matches no real builtin profile pattern.
        assert entry["profile_source"] == "fallback"
        assert entry["profile_resolved"] is False
        await gw.close()


class TestProbeRoutesCaching:
    """Re-audit P1#13: /health/ready and /health called _probe_routes() on
    EVERY request, which cleared and fully re-probed every route
    (sequentially, up to a 10s timeout each) with no caching at all."""

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_serves_cached_snapshot(self):
        import httpx

        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes(ttl=60.0)
        await gw._probe_routes(ttl=60.0)
        assert call_count["n"] == 1, "second call within ttl must not re-probe the backend"
        await gw.close()

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self):
        import httpx

        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes(ttl=60.0)
        await gw._probe_routes(ttl=60.0, force=True)
        assert call_count["n"] == 2
        await gw.close()

    @pytest.mark.asyncio
    async def test_stale_snapshot_triggers_real_reprobe(self):
        import httpx

        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await gw._probe_routes(ttl=0.0)
        await gw._probe_routes(ttl=0.0)
        assert call_count["n"] == 2, "ttl=0 must always re-probe"
        await gw.close()

    @pytest.mark.asyncio
    async def test_concurrent_callers_do_not_each_trigger_a_full_reprobe(self):
        """Two health-check requests racing while the snapshot is stale
        must not each independently probe every backend — the lock
        collapses them into a single real probe pass."""
        import asyncio

        import httpx

        call_count = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            call_count["n"] += 1
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})

        gw = _gateway_with_mocked_probe(handler)
        await asyncio.gather(gw._probe_routes(), gw._probe_routes())
        assert call_count["n"] == 1
        await gw.close()


# ─── Multi-route probing ───────────────────────────────────────────────────


class TestMultiRouteProbing:
    @pytest.mark.asyncio
    async def test_gateway_probes_all_routes(self):
        """Gateway should attempt to probe every configured route and record results."""
        config = InteropServerConfig(
            probe_on_startup=True,
            routes={
                "route_a": ModelRoute(
                    id="route_a",
                    client_model_aliases=["model_a"],
                    upstream_model="model_a",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://localhost:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
                "route_b": ModelRoute(
                    id="route_b",
                    client_model_aliases=["model_b"],
                    upstream_model="model_b",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.VLLM,
                        base_url="http://localhost:11435",
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        gw = Gateway(config)
        await gw.startup()

        # Both routes should have probe results recorded
        # even if they fail (since no backends are running)
        assert len(gw._probe_results) == 2, (
            f"Expected 2 probe results, got {len(gw._probe_results)}"
        )
        probed_ids = set(gw._probe_results.keys())
        assert "route_a" in probed_ids
        assert "route_b" in probed_ids

        await gw.close()

    @pytest.mark.asyncio
    async def test_gateway_startup_no_probe(self):
        """probe_on_startup=False should skip probing entirely."""
        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://localhost:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        gw = Gateway(config)
        await gw.startup()
        assert len(gw._probe_results) == 0
        await gw.close()


class TestSessionManagerIntegration:
    """Tests that the session manager receives repair data from the gateway."""

    def test_record_repairs_to_session_records_repaired(self):
        """_record_repairs_to_session should record repaired tools."""
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )
        from agent_interop.gateway import Gateway
        from agent_interop.session import SessionManager
        from agent_interop.transaction import ToolBatchDecision

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://localhost:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        mgr = SessionManager()
        gw = Gateway(config, session_manager=mgr)
        mgr.begin_request("sess-1")

        # Build a batch decision with one repaired tool
        outcome = RepairOutcome(
            status=RepairStatus.REPAIRED,
            call_name="read_file",
            initial_issues=[],
            final_issues=[],
        )
        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(name="read_file", raw_arguments="{}"),
            outcome=outcome,
        )
        batch = ToolBatchDecision(decisions=[decision])

        # Create a fake request context with session_id
        class FakeContext:
            session_id = "sess-1"

        gw._record_repairs_to_session(batch, FakeContext())

        state = mgr.get("sess-1")
        assert state is not None
        assert state.repair_count == 1
        assert "read_file" in state.recent_repairs

    def test_record_repairs_to_session_records_rejected(self):
        """_record_repairs_to_session should record rejected tools."""
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )
        from agent_interop.gateway import Gateway
        from agent_interop.session import SessionManager
        from agent_interop.transaction import ToolBatchDecision

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://localhost:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        mgr = SessionManager()
        gw = Gateway(config, session_manager=mgr)
        mgr.begin_request("sess-2")

        # Build a batch decision with one rejected tool
        outcome = RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name="bad_tool",
            initial_issues=[],
            final_issues=[],
        )
        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(name="bad_tool", raw_arguments="{}"),
            outcome=outcome,
        )
        batch = ToolBatchDecision(decisions=[decision])

        class FakeContext:
            session_id = "sess-2"

        gw._record_repairs_to_session(batch, FakeContext())

        state = mgr.get("sess-2")
        assert state is not None
        assert state.repair_count == 1
        assert state.recent_repairs["bad_tool"].status == "rejected"

    def test_record_repairs_no_session_id_is_noop(self):
        """_record_repairs_to_session should be safe with no session_id."""
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )
        from agent_interop.gateway import Gateway
        from agent_interop.session import SessionManager
        from agent_interop.transaction import ToolBatchDecision

        config = InteropServerConfig(
            probe_on_startup=False,
            routes={
                "test": ModelRoute(
                    id="test",
                    client_model_aliases=["test"],
                    upstream_model="test",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://localhost:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                ),
            },
        )
        mgr = SessionManager()
        gw = Gateway(config, session_manager=mgr)

        outcome = RepairOutcome(status=RepairStatus.REPAIRED, call_name="t")
        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(name="t", raw_arguments="{}"),
            outcome=outcome,
        )
        batch = ToolBatchDecision(decisions=[decision])

        class FakeContextNoSession:
            pass

        # Should not raise, just be a no-op
        gw._record_repairs_to_session(batch, FakeContextNoSession())
        assert mgr.active_count == 0


# ─── Recursive diff enforcement ────────────────────────────────────────────


class TestDiffEnforcement:
    def test_proposal_only_changes_target(self):
        """Verify the pipeline rejects out-of-scope mutations."""
        from agent_interop.config import RepairPolicy, RepairTier
        from agent_interop.repair.pipeline import repair_one

        tool = CanonicalTool(
            name="test",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "string"}},
                    "b": {"type": "array", "items": {"type": "string"}},
                    "c": {"type": "string"},
                },
                "required": ["a", "b", "c"],
            },
        )
        policy = RepairPolicy(enabled_tiers=frozenset({RepairTier.SAFE_SHAPE, RepairTier.COERCIVE}))
        out = repair_one(
            "test",
            {"a": '["x"]', "b": '["y"]', "c": "hello"},
            [tool],
            policy=policy,
        )
        assert out.is_accepted
        # Each step only changed one field
        for step in out.steps:
            assert step.path in ("a", "b")
        # c was never touched
        assert out.accepted["c"] == "hello"


# ─── Fake transport (lightweight — no real HTTP server) ──────────────────────


class FakeTransport:
    """An in-memory ``UpstreamTransport`` that returns a canned JSON body.

    Used to drive ``Gateway.handle_request`` without spinning up an HTTP server.
    """

    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body
        self.sent: list[PreparedUpstreamRequest] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        self.sent.append(request)
        return UpstreamResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body=json.dumps(self._body).encode("utf-8"),
        )

    @asynccontextmanager
    async def stream(self, request: PreparedUpstreamRequest):
        raise NotImplementedError("FakeTransport.send is for non-streaming only")

    async def close(self) -> None:
        pass


# ─── Tool-batch rejection assembly (Bug 4) and request/response ID (Bug 5) ──


class TestAssembleResponseRejection:
    """Bug 4: ``_assemble_response`` must not crash on a full batch rejection and
    must emit the right structured error + ``INVALID_OUTPUT`` stop reason."""

    @staticmethod
    def _rejected_decision(
        name: str = "read_file",
        error: str = "missing required field 'path'",
        with_correction: bool = False,
        with_repair_steps: bool = False,
    ) -> ToolCallDecision:
        outcome = RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name=name,
            accepted=None,
            error=error,
            initial_issues=[SchemaIssue(
                path=["path"], keyword="required",
                message=error, expected="string", actual="absent",
            )],
            final_issues=[SchemaIssue(
                path=["path"], keyword="required",
                message=error, expected="string", actual="absent",
            )],
            steps=[RepairStep(rule="fix_required", path="path")] if with_repair_steps else [],
        )
        correction = None
        if with_correction:
            correction = ToolCallCorrection(
                tool_name=name,
                candidate_id=f"tc_{name}",
                issue_path="path",
                schema_keyword="required",
                observed_type="absent",
                expected_type="string",
                message=error,
                retryable=True,
            )
        return ToolCallDecision(
            candidate=RawToolCallCandidate(id=f"tc_{name}", name=name, raw_arguments="{}"),
            outcome=outcome,
            accepted_block=None,
            correction=correction,
        )

    def _assemble(self, batch_decision: ToolBatchDecision, decoded: DecodedModelResponse) -> Any:
        config = InteropServerConfig(probe_on_startup=False, routes={
            "r": ModelRoute(
                id="r",
                client_model_aliases=["m"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://localhost:11434",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
            ),
        })
        gw = Gateway(config)
        canonical = CanonicalRequest(request_id="client-req-123")
        route = config.routes["r"]
        return gw._assemble_response(decoded, batch_decision, canonical, route)

    def test_full_rejection_per_call_does_not_crash(self) -> None:
        """All calls rejected (per-call) → no exception, structured error."""
        batch = ToolBatchDecision(
            decisions=[self._rejected_decision("read_file"), self._rejected_decision("get_weather")],
            is_accepted=False,
            accepted_blocks=[],
            rejected_count=2,
        )
        decoded = DecodedModelResponse(stop_reason=CanonicalStopReason.TOOL_CALL)
        resp = self._assemble(batch, decoded)
        assert resp.error is not None
        assert resp.error.code in (
            InteropErrorCode.TOOL_CALL_INVALID,
            InteropErrorCode.TOOL_CALL_REPAIR_FAILED,
        )
        # A fully rejected batch must NEVER report TOOL_CALL stop reason.
        assert resp.stop_reason == CanonicalStopReason.INVALID_OUTPUT
        assert resp.stop_reason != CanonicalStopReason.TOOL_CALL

    def test_full_rejection_per_call_details_are_structured(self) -> None:
        """Per-call rejection populates ``error.details`` with per-call info."""
        decision = self._rejected_decision("read_file", with_correction=True)
        batch = ToolBatchDecision(
            decisions=[decision],
            is_accepted=False,
            accepted_blocks=[],
            rejected_count=1,
        )
        decoded = DecodedModelResponse()
        resp = self._assemble(batch, decoded)
        assert resp.error is not None
        assert isinstance(resp.error.details, dict)
        assert "rejected_calls" in resp.error.details
        assert len(resp.error.details["rejected_calls"]) >= 1
        entry = resp.error.details["rejected_calls"][0]
        assert entry["tool_name"] == "read_file"
        assert entry["schema_keyword"] == "required"

    def test_repair_attempted_failed_uses_repair_failed_code(self) -> None:
        """When a repair was attempted but still failed, code is REPAIR_FAILED."""
        decision = self._rejected_decision("read_file", with_repair_steps=True)
        batch = ToolBatchDecision(
            decisions=[decision],
            is_accepted=False,
            accepted_blocks=[],
            rejected_count=1,
        )
        decoded = DecodedModelResponse()
        resp = self._assemble(batch, decoded)
        assert resp.error is not None
        assert resp.error.code == InteropErrorCode.TOOL_CALL_REPAIR_FAILED
        assert resp.error.retryable is True

    def test_choice_error_uses_tool_choice_violation_code(self) -> None:
        """A tool-choice policy violation (choice_error set) gets its own code."""
        batch = ToolBatchDecision(
            is_accepted=False,
            accepted_blocks=[],
            choice_error="Tool calls not permitted (choice=none): read_file",
        )
        decoded = DecodedModelResponse(stop_reason=CanonicalStopReason.TOOL_CALL)
        resp = self._assemble(batch, decoded)
        assert resp.error is not None
        assert resp.error.code == InteropErrorCode.TOOL_CHOICE_VIOLATION
        assert resp.error.retryable is True
        assert resp.stop_reason == CanonicalStopReason.INVALID_OUTPUT

    def test_upstream_tool_call_stop_reason_is_overridden_on_full_rejection(self) -> None:
        """Even if the upstream signaled TOOL_CALL, full rejection forces
        INVALID_OUTPUT."""
        batch = ToolBatchDecision(
            decisions=[self._rejected_decision("read_file")],
            is_accepted=False,
            accepted_blocks=[],
            rejected_count=1,
        )
        # Upstream itself said tool_calls — but the batch was fully rejected.
        decoded = DecodedModelResponse(stop_reason=CanonicalStopReason.TOOL_CALL)
        resp = self._assemble(batch, decoded)
        assert resp.stop_reason == CanonicalStopReason.INVALID_OUTPUT


class TestRequestResponseIdSeparation:
    """Bug 5: ``request_id`` (client) and ``response_id`` (provider) must not be
    conflated."""

    def test_request_id_and_response_id_are_separate_fields(self) -> None:
        config = InteropServerConfig(probe_on_startup=False, routes={
            "r": ModelRoute(
                id="r",
                client_model_aliases=["m"],
                upstream_model="fake-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://localhost:11434",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
            ),
        })
        gw = Gateway(config)
        canonical = CanonicalRequest(request_id="client-req-ABC")
        route = config.routes["r"]
        # Upstream carries a distinct provider response id in ``extra``.
        decoded = DecodedModelResponse(extra={"response_id": "provider-resp-XYZ"})
        batch = ToolBatchDecision(is_accepted=True, accepted_blocks=[])
        resp = gw._assemble_response(decoded, batch, canonical, route)
        assert resp.request_id == "client-req-ABC"
        assert resp.response_id == "provider-resp-XYZ"
        # The two values differ — proving they are not the same field.
        assert resp.request_id != resp.response_id


# ─── Batch policy read from route config (Bug 2) ───────────────────────────


class TestBatchPolicyFromConfig:
    """Bug 2: ``repair.batch_policy`` on the route must actually be read and
    passed through to ``process_tool_batch`` (not hardcoded to ATOMIC)."""

    def _make_config(self, batch_policy: str) -> InteropServerConfig:
        return InteropServerConfig(
            probe_on_startup=False,
            routes={
                "r": ModelRoute(
                    id="r",
                    client_model_aliases=["m"],
                    upstream_model="fake-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OPENAI_COMPATIBLE,
                        base_url="http://localhost:11434",
                        wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    repair=RepairConfig(batch_policy=batch_policy),
                ),
            },
        )

    def _upstream_body(self) -> dict[str, Any]:
        """Two tool calls: one valid (read_file), one invalid (undeclared)."""
        return {
            "id": "fake-chat-response",
            "object": "chat.completion",
            "created": 0,
            "model": "fake-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "tc_1", "type": "function", "function": {
                            "name": "read_file", "arguments": '{"path": "/tmp/x"}',
                        }},
                        {"id": "tc_2", "type": "function", "function": {
                            "name": "nonexistent_tool", "arguments": "{}",
                        }},
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def _make_canonical(self) -> CanonicalRequest:
        return CanonicalRequest(
            request_id="client-req-1",
            model=CanonicalModelReference(requested_name="m"),
            tools=[CanonicalTool(
                name="read_file",
                description="Read a file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )],
        )

    async def test_best_effort_returns_accepted_subset(self) -> None:
        """With ``batch_policy=best_effort``, a mixed batch returns the accepted
        subset instead of rejecting everything."""
        config = self._make_config("best_effort")
        gw = Gateway(config, transport=FakeTransport(self._upstream_body()))
        canonical = self._make_canonical()
        resp = await gw.handle_request(canonical, RequestContext())
        # The valid read_file call should be present in the assembled content.
        tool_blocks = [c for c in resp.content if getattr(c, "type", "") == "tool_call"]
        assert len(tool_blocks) >= 1, f"Expected accepted subset, got content: {resp.content}"
        assert any(getattr(b, "name", "") == "read_file" for b in tool_blocks)
        # best_effort emits the accepted subset → not a hard rejection.
        assert resp.error is None

    async def test_atomic_rejects_entire_batch(self) -> None:
        """With ``batch_policy=atomic`` (the default), the same mixed batch is
        fully rejected — proving the config value drives the decision."""
        config = self._make_config("atomic")
        gw = Gateway(config, transport=FakeTransport(self._upstream_body()))
        canonical = self._make_canonical()
        resp = await gw.handle_request(canonical, RequestContext())
        tool_blocks = [c for c in resp.content if getattr(c, "type", "") == "tool_call"]
        assert len(tool_blocks) == 0, (
            "ATOMIC policy should reject the entire batch when any call is invalid"
        )
        assert resp.stop_reason == CanonicalStopReason.INVALID_OUTPUT
