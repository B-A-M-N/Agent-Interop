"""Focused P0 tests for request compatibility planning."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_interop.abi import (
    CanonicalError,
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
)
from agent_interop.agents.manifests import load_builtin_descriptor
from agent_interop.backends.base import ModelRuntimeCapabilities
from agent_interop.backends.runtime_cache import RuntimeCapabilityCache
from agent_interop.capabilities import CapabilityState
from agent_interop.config import (
    CompatibilityConfig,
    ContextConfig,
    ControllerConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    ToolSurfaceMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    load_config_from_dict,
)
from agent_interop.context import RequestContext
from agent_interop.context_budget import ContextBudgetPlanner, effective_context_limit
from agent_interop.context_budget.compaction import compact_safe_tool_results
from agent_interop.context_budget.types import TokenEstimate
from agent_interop.controller import CompatibilityController, ControllerAction, ControllerDecision
from agent_interop.evidence import has_confident_capability
from agent_interop.execution import InteropRequestExecution
from agent_interop.execution_attempts import CompatibilityAttemptExecutor
from agent_interop.execution_attempts.budget import AttemptBudget
from agent_interop.gateway import Gateway, ResolvedInvocation
from agent_interop.history import reconcile_history
from agent_interop.planning import (
    BehavioralCapabilities,
    CompatibilityPath,
    RequestCompatibilityPlanner,
    derive_request_requirements,
)
from agent_interop.planning.types import AttemptKind, CompatibilityAttempt
from agent_interop.qualification import BootstrapQualifier, QualificationRecord, QualificationState
from agent_interop.qualification.store import QualificationStore
from agent_interop.repair.invocation import build_invocation_plan
from agent_interop.replay.capture import sanitize_body
from agent_interop.replay.store import DiagnosticCaseStore
from agent_interop.replay.types import CompatibilityKey, ReplayCase
from agent_interop.tool_surface import ToolSurfacePlanner
from agent_interop.upstreams.codec import CodecCapabilities


def _tool(name: str, description: str) -> CanonicalTool:
    return CanonicalTool(name=name, description=description, input_schema={"type": "object"})


def _route(**kwargs) -> ModelRoute:
    return ModelRoute(
        id="local",
        client_model_aliases=["local"],
        upstream_model="unknown:latest",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
        tool_surface=ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=1),
        context=ContextConfig(output_reserve_tokens=32),
        compatibility=CompatibilityConfig(**kwargs),
    )


def test_tool_result_history_requires_continuation_even_when_auto() -> None:
    request = CanonicalRequest(
        messages=[CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="call_1", content="ok")])],
        tools=[_tool("read_file", "read a file")],
        tool_choice=CanonicalToolChoice.auto(),
    )
    requirements = derive_request_requirements(request, RequestContext(), object(), TokenEstimate(12))
    assert requirements.tool_result_continuation_required
    assert requirements.sequential_tool_use_required


def test_schema_v2_enables_buffered_unverified_streaming_by_default() -> None:
    config = load_config_from_dict({
        "schema_version": 2,
        "routes": {
            "local": {
                "aliases": ["local"],
                "upstream_model": "local-model",
                "upstream": {"kind": "ollama", "wire_protocol": "ollama_chat"},
            },
        },
    })
    assert config.routes["local"].compatibility.buffer_unverified_streaming


def test_named_tool_is_never_hidden_by_surface_reduction() -> None:
    named = _tool("edit_file", "edit source file")
    request = CanonicalRequest(
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read docs")])],
        tools=[_tool("read_file", "read source file"), named],
        tool_choice=CanonicalToolChoice.named("edit_file"),
    )
    plan = ToolSurfacePlanner().plan(request, ToolSurfaceConfig(mode=ToolSurfaceMode.DYNAMIC, max_initial_tools=1))
    assert [tool.name for tool in plan.visible_tools] == ["edit_file"]
    assert [tool.name for tool in plan.validation_tools] == ["read_file", "edit_file"]


def test_effective_context_limit_cannot_exceed_runtime() -> None:
    assert effective_context_limit(32768, 16384, 65536) == 16384


def test_planner_honors_route_context_override_as_a_hard_ceiling() -> None:
    route = _route()
    route.context = ContextConfig(context_limit_tokens=1024, output_reserve_tokens=16)
    request = CanonicalRequest(
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="hello")])],
    )
    plan = asyncio.run(RequestCompatibilityPlanner().plan(
        request=request,
        context=RequestContext(),
        route=route,
        client_requirements=object(),
        codec_capabilities=CodecCapabilities(),
        runtime_capabilities=ModelRuntimeCapabilities(
            backend_kind=UpstreamKind.OLLAMA,
            architecture_context_tokens=32768,
            configured_context_tokens=8192,
            effective_context_tokens=8192,
        ),
        behavioral_capabilities=BehavioralCapabilities(),
    ))
    assert plan.context_plan.runtime_limit_tokens == 1024


def test_direct_path_requires_codec_runtime_and_behavioral_evidence() -> None:
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="local"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read file")])],
        tools=[_tool("read_file", "read a source file")],
        generation=CanonicalGenerationOptions(stream=False, max_output_tokens=16),
    )
    runtime = ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OLLAMA,
        model_name="unknown:latest",
        effective_context_tokens=8192,
        accepts_native_tools=CapabilityState.PROBED,
        returns_native_tool_calls=CapabilityState.PROBED,
    )
    planner = RequestCompatibilityPlanner()
    direct = asyncio.run(planner.plan(
        request=request, context=RequestContext(), route=_route(), client_requirements=object(),
        codec_capabilities=CodecCapabilities(supports_native_tools=True), runtime_capabilities=runtime,
        behavioral_capabilities=BehavioralCapabilities(native_tools=True, automatic_selection=True),
    ))
    assert direct.path == CompatibilityPath.DIRECT

    no_behavior = asyncio.run(planner.plan(
        request=request, context=RequestContext(), route=_route(), client_requirements=object(),
        codec_capabilities=CodecCapabilities(supports_native_tools=True), runtime_capabilities=runtime,
        behavioral_capabilities=BehavioralCapabilities(),
    ))
    assert no_behavior.path == CompatibilityPath.ADAPTED


def test_adapted_ladder_keeps_controller_as_final_bounded_fallback() -> None:
    route = _route()
    route.controller = ControllerConfig(route_id="controller")
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="local"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read file")])],
        tools=[_tool("read_file", "read")],
    )
    plan = asyncio.run(RequestCompatibilityPlanner().plan(
        request=request,
        context=RequestContext(),
        route=route,
        client_requirements=object(),
        codec_capabilities=CodecCapabilities(supports_native_tools=True),
        runtime_capabilities=ModelRuntimeCapabilities(backend_kind=UpstreamKind.OLLAMA, effective_context_tokens=8192),
        behavioral_capabilities=BehavioralCapabilities(),
    ))
    assert plan.path == CompatibilityPath.ADAPTED
    assert plan.attempts[-1].kind == AttemptKind.CONTROLLER_MEDIATED


def test_qualified_chat_only_model_selects_controller_path() -> None:
    route = _route()
    route.controller = ControllerConfig(route_id="controller")
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="local"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read file")])],
        tools=[_tool("read_file", "read")],
    )
    plan = asyncio.run(RequestCompatibilityPlanner().plan(
        request=request,
        context=RequestContext(),
        route=route,
        client_requirements=object(),
        codec_capabilities=CodecCapabilities(supports_native_tools=True),
        runtime_capabilities=ModelRuntimeCapabilities(backend_kind=UpstreamKind.OLLAMA, effective_context_tokens=8192),
        behavioral_capabilities=BehavioralCapabilities(chat_only=True),
    ))
    assert plan.path == CompatibilityPath.CONTROLLED
    assert plan.attempts[0].kind == AttemptKind.CONTROLLER_MEDIATED


def test_auto_controller_selection_uses_a_verified_distinct_route() -> None:
    primary = _route()
    first = _route()
    first.id, first.client_model_aliases, first.upstream_model = "first", ["first"], "first-model"
    verified = _route()
    verified.id, verified.client_model_aliases, verified.upstream_model = "verified", ["verified"], "verified-model"
    gateway = Gateway(InteropServerConfig(
        default_route_id="local",
        routes={"local": primary, "first": first, "verified": verified},
        controller=ControllerConfig(auto_select_route=True, require_verified=True),
        probe_on_startup=False,
    ))
    gateway.record_qualification(QualificationRecord(
        model_digest="verified-model",
        state=QualificationState.SEQUENTIAL_AGENT,
        prompted_forced_tool=True,
        continuation=True,
    ))
    selected = asyncio.run(gateway._select_controller_route(
        primary, gateway.config.controller,
    ))
    assert selected is verified


def test_controller_evidence_key_isolated_from_primary_tuple() -> None:
    primary = CompatibilityKey(compatibility_path="controlled", controller_model_id="controller-route")
    selected = CompatibilityKey(
        compatibility_path="controlled",
        controller_model_id="controller-model",
        controller_model_digest="sha256:controller",
        controller_profile_revision="2",
    )
    execution = InteropRequestExecution(compatibility_key=selected)
    invocation = SimpleNamespace(compatibility_key=primary)
    assert Gateway._selected_evidence_key(invocation, execution) == selected


def test_context_plan_preserves_current_tool_result_and_latest_message() -> None:
    request = CanonicalRequest(
        messages=[
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="old " * 100)]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="x", content="current")]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="latest")]),
        ],
        generation=CanonicalGenerationOptions(max_output_tokens=64),
    )
    plan = ContextBudgetPlanner().plan(request, runtime_limit_tokens=100)
    assert plan.compaction_required
    assert 1 in plan.preserved_message_indices
    assert 2 in plan.preserved_message_indices


def test_context_plan_accounts_for_tool_surface_reduction_before_adaptation() -> None:
    request = CanonicalRequest(
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read")])],
        tools=[_tool(f"tool_{index}", "x" * 100) for index in range(8)],
        generation=CanonicalGenerationOptions(max_output_tokens=16),
    )
    plan = ContextBudgetPlanner().plan(
        request,
        runtime_limit_tokens=1000,
        original_tools=request.tools,
        visible_tools=request.tools[:1],
        output_reserve_tokens=16,
    )
    assert plan.before.tool_schema_tokens > plan.after.tool_schema_tokens
    assert "reduce_tool_surface" in plan.transformations


def test_context_compaction_only_reduces_old_pageable_results() -> None:
    old_call = CanonicalToolCallBlock(id="old", name="read_file", arguments={"path": "old.py"})
    current_call = CanonicalToolCallBlock(id="current", name="run_tests", arguments={"command": "pytest"})
    old_output = "".join(f"old line {index}\n" for index in range(80))
    current_output = "current failure detail\n" * 20
    request = CanonicalRequest(messages=[
        CanonicalMessage(role="assistant", content=[old_call]),
        CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="old", content=old_output)]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="continue")]),
        CanonicalMessage(role="assistant", content=[current_call]),
        CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="current", content=current_output, is_error=True)]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="fix the failure")]),
    ])
    history = reconcile_history(request.messages)
    plan = ContextBudgetPlanner().plan(request, runtime_limit_tokens=200)
    assert plan.compaction_required
    adapted = compact_safe_tool_results(request, exchanges=history.exchanges, plan=plan)
    assert adapted.changed
    old_result = adapted.request.messages[1].content[0]
    current_result = adapted.request.messages[4].content[0]
    assert isinstance(old_result, CanonicalToolResultBlock)
    assert isinstance(current_result, CanonicalToolResultBlock)
    assert "[interop: compacted" in old_result.content
    assert old_result.content.startswith("old line 0\n")
    assert current_result.content == current_output


def test_context_compaction_keeps_unknown_tool_output_verbatim() -> None:
    call = CanonicalToolCallBlock(id="shell", name="shell", arguments={"command": "x"})
    request = CanonicalRequest(messages=[
        CanonicalMessage(role="assistant", content=[call]),
        CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="shell", content="x\n" * 100)]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="new turn")]),
        CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(id="latest", name="read_file")]),
        CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="latest", content="current")]),
        CanonicalMessage(role="user", content=[CanonicalTextBlock(text="continue")]),
    ])
    plan = ContextBudgetPlanner().plan(request, runtime_limit_tokens=200)
    adapted = compact_safe_tool_results(request, exchanges=reconcile_history(request.messages).exchanges, plan=plan)
    assert not adapted.changed
    assert adapted.request.messages[1].content[0].content == "x\n" * 100


def test_gateway_replans_after_safe_context_adaptation() -> None:
    route = _route()
    route.context = ContextConfig(output_reserve_tokens=16)
    gateway = Gateway(InteropServerConfig(default_route_id="local", routes={"local": route}))

    async def inspect(_route):
        return ModelRuntimeCapabilities(
            backend_kind=UpstreamKind.OLLAMA,
            model_name="unknown:latest",
            effective_context_tokens=800,
        )

    gateway._inspect_model_runtime = inspect  # type: ignore[method-assign]
    old = "".join(f"line {index}\n" for index in range(180))
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="local"),
        messages=[
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(id="old", name="read_file")]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="old", content=old)]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="continue")]),
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(id="current", name="read_file")]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="current", content="current")]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="make a change")]),
        ],
        tools=[_tool("read_file", "read")],
    )
    invocation = asyncio.run(gateway._prepare_invocation_async(
        request, RequestContext(), False, InteropRequestExecution(),
    ))
    assert invocation.context_plan.fits_directly
    assert "[interop: compacted" in invocation.reconciled_request.messages[1].content[0].content


def test_gateway_returns_structured_error_when_no_compatibility_path_exists() -> None:
    route = _route(allow_direct=False, allow_adapted=False, allow_controlled=False)
    gateway = Gateway(InteropServerConfig(
        default_route_id="local", routes={"local": route}, probe_on_startup=False,
    ))
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="local"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read")])],
        tools=[_tool("read_file", "read")],
    )
    response = asyncio.run(gateway.handle_request(request, RequestContext()))
    assert response.error is not None
    assert response.error.code == "REQUEST_PLAN_UNAVAILABLE"
    assert response.error.details["path"] == "unavailable"


def test_controller_calls_have_controller_provenance() -> None:
    call = CanonicalToolCallBlock(
        id="call_1", name="read_file", arguments={"path": "a.py"}
    )
    decision = CompatibilityController().normalize_decision(
        ControllerDecision(action=ControllerAction.TOOL_CALL, tool_calls=(call,))
    )
    assert decision.tool_calls[0].provenance.source == "compatibility_controller"


def test_bootstrap_qualifier_uses_only_bounded_synthetic_probes() -> None:
    async def execute(probe) -> bool:
        assert probe.name in {
            "exact_text", "native_forced_tool", "prompted_forced_tool", "no_tool", "tool_result_continuation",
        }
        return True

    record = asyncio.run(BootstrapQualifier().qualify("sha256:abc", execute))
    assert record.state == QualificationState.SEQUENTIAL_AGENT


def test_qualification_store_restores_digest_scoped_safe_facts(tmp_path) -> None:
    store = QualificationStore(tmp_path / "qualification.json")
    record = QualificationRecord(
        model_digest="sha256:qualified",
        state=QualificationState.SEQUENTIAL_AGENT,
        native_forced_tool=True,
        continuation=True,
    )
    store.put(record)
    restored = QualificationStore(tmp_path / "qualification.json").get("sha256:qualified")
    assert restored == record


def test_gateway_uses_only_bootstrap_proven_qualification_capabilities() -> None:
    gateway = Gateway(InteropServerConfig(default_route_id="local", routes={"local": _route()}))
    gateway.record_qualification(QualificationRecord(
        model_digest="sha256:qualified",
        state=QualificationState.SEQUENTIAL_AGENT,
        prompted_forced_tool=True,
        continuation=True,
    ))
    capabilities = gateway._behavioral_capabilities(ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OLLAMA,
        model_name="unknown:latest",
        model_digest="sha256:qualified",
    ))
    assert capabilities.prompted_tools
    assert capabilities.forced_selection
    assert capabilities.tool_result_continuation
    assert not capabilities.automatic_selection
    assert not capabilities.parallel_tool_use


def test_schema_v2_loads_qualification_and_diagnostics_controls() -> None:
    config = load_config_from_dict({
        "schema_version": 2,
        "default_route": "local",
        "diagnostics": {"capture": "failures", "content_mode": "metadata_only"},
        "routes": {
            "local": {
                "aliases": ["local"],
                "upstream_model": "unknown:latest",
                "upstream": {"kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                "qualification": {"bootstrap": "blocking_for_tool_requests", "cache_by_digest": True},
            },
        },
    })
    assert config.routes["local"].qualification.bootstrap == "blocking_for_tool_requests"
    assert config.routes["local"].qualification.cache_by_digest
    assert config.diagnostics.capture == "failures"


def test_capability_promotion_requires_a_minimum_sample_base() -> None:
    assert not has_confident_capability(1, 1)
    assert has_confident_capability(5, 5, threshold=0.5)


def test_blocking_bootstrap_qualification_is_limited_to_required_or_named_tools() -> None:
    route = _route()
    route.qualification.bootstrap = "blocking_for_tool_requests"
    gateway = Gateway(InteropServerConfig(default_route_id="local", routes={"local": route}))
    probes: list[str] = []

    async def fake_probe(_invocation, probe):
        probes.append(probe.name)
        return True

    gateway._execute_bootstrap_probe = fake_probe  # type: ignore[method-assign]
    runtime = ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OLLAMA,
        model_name="unknown:latest",
        model_digest="sha256:unknown",
    )
    required = SimpleNamespace(
        route=route,
        runtime_capabilities=runtime,
        reconciled_request=CanonicalRequest(
            tools=[_tool("read_file", "read")], tool_choice=CanonicalToolChoice.required(),
        ),
    )
    assert asyncio.run(gateway._ensure_bootstrap_qualification(required))
    assert len(probes) == 5
    automatic = SimpleNamespace(
        route=route,
        runtime_capabilities=ModelRuntimeCapabilities(backend_kind=UpstreamKind.OLLAMA, model_name="new"),
        reconciled_request=CanonicalRequest(
            tools=[_tool("read_file", "read")], tool_choice=CanonicalToolChoice.auto(),
        ),
    )
    assert not asyncio.run(gateway._ensure_bootstrap_qualification(automatic))


def test_failed_live_request_captures_metadata_only_diagnostic_case() -> None:
    route = _route()
    config = InteropServerConfig(
        default_route_id="local",
        routes={"local": route},
        probe_on_startup=False,
    )
    config.diagnostics.capture = "failures"
    gateway = Gateway(config)

    async def fail(_invocation, _execution):
        return CanonicalResponse(error=CanonicalError(code="BACKEND_ERROR", message="synthetic failure"))

    gateway._execute_compatibility_attempt = fail  # type: ignore[method-assign]
    response = asyncio.run(gateway.handle_request(CanonicalRequest(
        model=CanonicalModelReference(requested_name="local"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="secret prompt")])],
        tools=[_tool("read_file", "read")],
        tool_choice=CanonicalToolChoice.required(),
    ), RequestContext()))
    assert response.error is not None
    case_ids = gateway._diagnostic_cases.list_ids()
    assert len(case_ids) == 1
    case = gateway.diagnostic_case(case_ids[0])
    assert case is not None
    assert case.canonical_request is None
    assert case.diagnostics["response"]["error_code"] == "BACKEND_ERROR"
    assert "secret prompt" not in str(case.inbound_request)


def test_controller_qualification_requires_sequential_agent_level() -> None:
    route = _route()
    controller = _route()
    controller.id = "controller"
    controller.client_model_aliases = ["controller"]
    controller.upstream_model = "controller-model"
    gateway = Gateway(InteropServerConfig(
        default_route_id="local",
        routes={"local": route, "controller": controller},
        probe_on_startup=False,
    ))
    assert not asyncio.run(gateway._controller_route_is_qualified(controller, "L3"))
    gateway.record_qualification(QualificationRecord(
        model_digest="controller-model",
        state=QualificationState.FORCED_TOOL,
        prompted_forced_tool=True,
    ))
    assert not asyncio.run(gateway._controller_route_is_qualified(controller, "L3"))
    gateway.record_qualification(QualificationRecord(
        model_digest="controller-model",
        state=QualificationState.SEQUENTIAL_AGENT,
        prompted_forced_tool=True,
        continuation=True,
    ))
    assert asyncio.run(gateway._controller_route_is_qualified(controller, "L3"))


def test_runtime_cache_reuses_a_fresh_route_snapshot() -> None:
    cache = RuntimeCapabilityCache(ttl_seconds=60)
    runtime = ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OLLAMA,
        model_name="model:latest",
        model_digest="sha256:one",
    )
    cache.put(runtime, "http://127.0.0.1:11434/")
    assert cache.get_for_route("http://127.0.0.1:11434", "model:latest") == runtime


def test_diagnostic_case_store_persists_sanitized_case_with_lru_retention(tmp_path) -> None:
    store = DiagnosticCaseStore(retention_count=1, directory=tmp_path)
    first = ReplayCase(case_id="first", inbound_request={"authorization": "secret"})
    second = ReplayCase(case_id="second", inbound_request={"safe": "value"})
    store.put(first)
    store.put(second)
    assert not (tmp_path / "first.json").exists()
    loaded = DiagnosticCaseStore(retention_count=1, directory=tmp_path).get("second")
    assert loaded is not None
    assert loaded.inbound_request == {"safe": "value"}


def test_diagnostic_case_store_bounds_oversized_case() -> None:
    store = DiagnosticCaseStore(max_case_bytes=1024)
    case = ReplayCase(case_id="large", diagnostics={"payload": "x" * 5000})
    store.put(case)
    retained = store.get("large")
    assert retained is not None
    assert retained.diagnostics["capture_truncated"]


def test_bundled_client_manifest_exposes_continuation_contract() -> None:
    descriptor = load_builtin_descriptor("claude_code")
    assert descriptor is not None
    assert descriptor.required_capabilities.requires_tool_result_continuation


def test_replay_sanitization_recurses_through_nested_payloads() -> None:
    assert sanitize_body({"outer": [{"authorization": "secret"}], "safe": "ok"}) == {
        "outer": [{"authorization": "[REDACTED]"}], "safe": "ok"
    }


def test_attempt_ladder_retries_required_tool_but_not_auto() -> None:
    class Invocation:
        def __init__(self, choice):
            self.compatibility_plan = type("Plan", (), {"attempts": (
                CompatibilityAttempt(AttemptKind.PROMPTED_TOOLS, ToolMode.PROMPTED),
                CompatibilityAttempt(AttemptKind.CONSTRAINED_JSON, ToolMode.PROMPTED),
            )})()
            self.reconciled_request = type("Request", (), {"tool_choice": choice})()

    calls = 0

    async def send(invocation):
        nonlocal calls
        calls += 1
        return CanonicalResponse()

    required = Invocation(CanonicalToolChoice.required())
    asyncio.run(CompatibilityAttemptExecutor().execute(required, build_invocation=lambda i, _: i, execute_attempt=send))
    assert calls == 2

    calls = 0
    automatic = Invocation(CanonicalToolChoice.auto())
    asyncio.run(CompatibilityAttemptExecutor().execute(automatic, build_invocation=lambda i, _: i, execute_attempt=send))
    assert calls == 1


def test_attempt_ladder_replans_one_withheld_tool_once() -> None:
    class Invocation:
        compatibility_plan = type("Plan", (), {"attempts": (
            CompatibilityAttempt(AttemptKind.PROMPTED_TOOLS, ToolMode.PROMPTED),
        )})()
        reconciled_request = type("Request", (), {"tool_choice": CanonicalToolChoice.required()})()

    sends = 0
    replans: list[str] = []

    async def send(invocation):
        nonlocal sends
        sends += 1
        if sends == 1:
            return CanonicalResponse(error=CanonicalError(details={"withheld_tool_requested": "edit_file"}))
        return CanonicalResponse(content=[CanonicalToolCallBlock(name="edit_file")])

    def replan(invocation, tool_name):
        replans.append(tool_name)
        return invocation

    asyncio.run(CompatibilityAttemptExecutor().execute(
        Invocation(), build_invocation=lambda i, _: i, execute_attempt=send, replan_withheld_tool=replan,
    ))
    assert replans == ["edit_file"]
    assert sends == 2


def test_attempt_ladder_returns_structured_error_when_token_budget_is_exhausted() -> None:
    class Invocation:
        compatibility_plan = type("Plan", (), {"attempts": (
            CompatibilityAttempt(AttemptKind.PROMPTED_TOOLS, ToolMode.PROMPTED),
            CompatibilityAttempt(AttemptKind.CONSTRAINED_JSON, ToolMode.PROMPTED),
        )})()
        reconciled_request = type("Request", (), {"tool_choice": CanonicalToolChoice.required()})()

    async def send(_invocation):
        return CanonicalResponse(content=[CanonicalTextBlock(text="x" * 100)])

    result = asyncio.run(CompatibilityAttemptExecutor(AttemptBudget(max_total_generated_tokens=1)).execute(
        Invocation(), build_invocation=lambda invocation, _: invocation, execute_attempt=send,
    ))
    assert result.error is not None
    assert result.error.code == "ATTEMPT_BUDGET_EXHAUSTED"


def test_controlled_execution_labels_controller_tool_calls() -> None:
    primary = _route(allow_adapted=False, allow_direct=False, allow_controlled=True)
    primary.id = "primary"
    primary.client_model_aliases = ["primary"]
    primary.controller = ControllerConfig(route_id="controller")
    controller = _route(allow_adapted=True, allow_direct=False, allow_controlled=False)
    controller.id = "controller"
    controller.client_model_aliases = ["controller"]
    gateway = Gateway(InteropServerConfig(default_route_id="primary", routes={"primary": primary, "controller": controller}))
    context = RequestContext(session_id="s1", client_id="test")
    request = CanonicalRequest(
        model=CanonicalModelReference(requested_name="primary"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read a file")])],
        tools=[_tool("read_file", "read a file")],
        tool_choice=CanonicalToolChoice.required(),
    )
    base_plan = build_invocation_plan([], CanonicalToolChoice.none(), ToolMode.DISABLED)
    invocation = ResolvedInvocation(
        context, request, request, primary, object(), object(), object(), base_plan, object(), object(), None, None,
        InteropRequestExecution(context=context),
        tool_surface_plan=type("Surface", (), {"fingerprint": "tools"})(),
    )
    sent = 0

    async def fake_send(inv, record):
        nonlocal sent
        sent += 1
        if sent == 1:
            return CanonicalResponse(content=[CanonicalTextBlock(text="use read_file")])
        return CanonicalResponse(content=[CanonicalToolCallBlock(id="c1", name="read_file", arguments={})])

    async def fake_prepare(*args, **kwargs):
        return invocation

    gateway._handle_request_send = fake_send  # type: ignore[method-assign]
    gateway._prepare_invocation = fake_prepare  # type: ignore[method-assign]
    response = asyncio.run(gateway._execute_controller_attempt(invocation, invocation.execution_record))
    assert response.content[0].provenance.source == "compatibility_controller"
