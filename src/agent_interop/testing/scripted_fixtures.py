"""Deterministic model behaviours used by compatibility-path acceptance tests.

These fixtures are deliberately model- and backend-free.  They model the
failure shapes Interop must handle, so a regression in planning, extraction,
repair, or controller handoff can be reproduced without claiming that a real
vendor model has those exact characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_interop.backends.base import ModelRuntimeCapabilities
from agent_interop.capabilities import CapabilityState
from agent_interop.config import UpstreamKind
from agent_interop.planning import BehavioralCapabilities, CompatibilityPath
from agent_interop.testing.fake_upstream import FakeResponseTemplate, make_text, make_tool_call


@dataclass(frozen=True)
class ScriptedModelFixture:
    """One deterministic model capability/failure profile."""

    name: str
    expected_path: CompatibilityPath
    runtime: ModelRuntimeCapabilities
    behavior: BehavioralCapabilities
    response: FakeResponseTemplate
    notes: str


def _runtime(*, native: bool = False, context: int = 8192, streaming: bool = True) -> ModelRuntimeCapabilities:
    capability = CapabilityState.PROBED if native else CapabilityState.UNSUPPORTED
    return ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OLLAMA,
        model_name="scripted",
        model_digest="sha256:scripted",
        effective_context_tokens=context,
        accepts_native_tools=capability,
        returns_native_tool_calls=capability,
        supports_streaming=CapabilityState.PROBED if streaming else CapabilityState.UNSUPPORTED,
    )


def scripted_model_fixtures() -> tuple[ScriptedModelFixture, ...]:
    """Return the required acceptance matrix's twelve model behaviours."""
    native_behavior = BehavioralCapabilities(native_tools=True, automatic_selection=True, streaming=True)
    adapted_behavior = BehavioralCapabilities(automatic_selection=True)
    forced_behavior = BehavioralCapabilities(forced_selection=True)
    chat = BehavioralCapabilities(chat_only=True)
    return (
        ScriptedModelFixture(
            "native_tool", CompatibilityPath.DIRECT, _runtime(native=True), native_behavior,
            make_tool_call("read_file", {"path": "/tmp/a.py"}), "Backend-native structured call.",
        ),
        ScriptedModelFixture(
            "prompted_envelope", CompatibilityPath.ADAPTED, _runtime(), adapted_behavior,
            make_text('<tool_call>{"name":"read_file","arguments":{"path":"/tmp/a.py"}}</tool_call>'),
            "Prompted envelope dialect.",
        ),
        ScriptedModelFixture(
            "bare_json", CompatibilityPath.ADAPTED, _runtime(), adapted_behavior,
            make_text('{"name":"read_file","arguments":{"path":"/tmp/a.py"}}'),
            "Bare JSON tool envelope.",
        ),
        ScriptedModelFixture(
            "forced_only", CompatibilityPath.ADAPTED, _runtime(), forced_behavior,
            make_tool_call("read_file", {"path": "/tmp/a.py"}), "Needs named/required selection.",
        ),
        ScriptedModelFixture(
            "automatic_selection_failure", CompatibilityPath.CONTROLLED, _runtime(), chat,
            make_text("I cannot choose a tool automatically."), "Controller owns selection.",
        ),
        ScriptedModelFixture(
            "chat_only", CompatibilityPath.CONTROLLED, _runtime(), chat,
            make_text("I can provide a work product but cannot invoke tools."), "Primary worker only.",
        ),
        ScriptedModelFixture(
            "malformed_arguments", CompatibilityPath.ADAPTED, _runtime(), adapted_behavior,
            make_text('<tool_call>{"name":"read_file","arguments":"{\\"path\\":\\"/tmp/a.py\\"}"}</tool_call>'),
            "Stringified arguments exercise bounded repair.",
        ),
        ScriptedModelFixture(
            "duplicate_id", CompatibilityPath.ADAPTED, _runtime(), adapted_behavior,
            FakeResponseTemplate(tool_calls=[
                {"id": "duplicate", "function": {"name": "read_file", "arguments": {"path": "/tmp/a.py"}}},
                {"id": "duplicate", "function": {"name": "read_file", "arguments": {"path": "/tmp/b.py"}}},
            ], finish_reason="tool_calls"),
            "Duplicate IDs exercise transaction normalization.",
        ),
        ScriptedModelFixture(
            "low_context", CompatibilityPath.ADAPTED, _runtime(context=512), adapted_behavior,
            make_tool_call("read_file", {"path": "/tmp/a.py"}), "Requires reduced visible tool surface.",
        ),
        ScriptedModelFixture(
            "streaming_text_only", CompatibilityPath.DIRECT, _runtime(streaming=True),
            BehavioralCapabilities(streaming=True), make_text("streamed text"), "No tool contract required.",
        ),
        ScriptedModelFixture(
            "continuation_failure", CompatibilityPath.CONTROLLED, _runtime(), chat,
            make_text("I cannot continue after a tool result."), "Controller resumes the tool loop.",
        ),
        ScriptedModelFixture(
            "looping", CompatibilityPath.CONTROLLED, _runtime(), chat,
            make_tool_call("read_file", {"path": "/tmp/a.py"}), "Controller loop budget must terminate repeats.",
        ),
    )
