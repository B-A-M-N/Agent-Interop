"""Acceptance matrix coverage for scripted model behaviours."""

import asyncio

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
)
from agent_interop.config import (
    CompatibilityConfig,
    ContextConfig,
    ControllerConfig,
    ModelRoute,
    ToolMode,
    ToolSurfaceConfig,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.planning import CompatibilityPath, RequestCompatibilityPlanner
from agent_interop.testing.scripted_fixtures import scripted_model_fixtures
from agent_interop.upstreams.codec import CodecCapabilities


def test_scripted_fixture_catalog_covers_required_model_behaviors() -> None:
    fixtures = {fixture.name: fixture for fixture in scripted_model_fixtures()}
    assert set(fixtures) == {
        "native_tool", "prompted_envelope", "bare_json", "forced_only",
        "automatic_selection_failure", "chat_only", "malformed_arguments",
        "duplicate_id", "low_context", "streaming_text_only",
        "continuation_failure", "looping",
    }
    assert fixtures["native_tool"].expected_path is CompatibilityPath.DIRECT
    assert fixtures["chat_only"].expected_path is CompatibilityPath.CONTROLLED
    assert fixtures["malformed_arguments"].expected_path is CompatibilityPath.ADAPTED


def _route() -> ModelRoute:
    return ModelRoute(
        id="primary",
        client_model_aliases=["primary"],
        upstream_model="scripted",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://scripted.test",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
        ),
        tool_mode=ToolMode.AUTO,
        context=ContextConfig(output_reserve_tokens=16),
        tool_surface=ToolSurfaceConfig(max_initial_tools=1),
        compatibility=CompatibilityConfig(),
        controller=ControllerConfig(auto_select_route=True),
    )


def _request(name: str) -> CanonicalRequest:
    tool = CanonicalTool(name="read_file", description="Read source", input_schema={"type": "object"})
    messages = [CanonicalMessage(role="user", content=[CanonicalTextBlock(text="Read /tmp/a.py")])]
    if name in {"continuation_failure", "looping"}:
        messages = [
            CanonicalMessage(role="assistant", content=[CanonicalToolCallBlock(id="prior", name="read_file")]),
            CanonicalMessage(role="tool", content=[CanonicalToolResultBlock(tool_call_id="prior", content="old")]),
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text="Continue")]),
        ]
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="primary"),
        messages=messages,
        tools=[] if name == "streaming_text_only" else [tool],
        generation=CanonicalGenerationOptions(stream=name == "streaming_text_only", max_output_tokens=16),
    )


def test_scripted_fixtures_prove_the_expected_compatibility_paths() -> None:
    planner = RequestCompatibilityPlanner()
    for fixture in scripted_model_fixtures():
        plan = asyncio.run(planner.plan(
            request=_request(fixture.name),
            context=RequestContext(),
            route=_route(),
            client_requirements=object(),
            codec_capabilities=CodecCapabilities(supports_native_tools=True),
            runtime_capabilities=fixture.runtime,
            behavioral_capabilities=fixture.behavior,
        ))
        assert plan.path is fixture.expected_path, fixture.name
