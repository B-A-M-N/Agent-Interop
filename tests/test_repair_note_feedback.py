"""Repair-note feedback: a compact, structured text block telling the
model what the repair pipeline actually changed on this turn — included
only when a repair was applied, never for calls that were already valid.

Rides along in the SAME assistant turn as the (now-corrected) tool call,
so it survives into conversation history like any other assistant text
and reaches the model again on a later turn without any separate
state-tracking mechanism.
"""

from __future__ import annotations

import json

import pytest

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.gateway import Gateway
from agent_interop.transport.http import UpstreamResponse, UpstreamTransport

_TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string", "x-aliases": ["file_path"]},
        },
        "required": ["path"],
    },
)


def _config() -> InteropServerConfig:
    return InteropServerConfig(
        probe_on_startup=False,
        default_route_id="r",
        routes={
            "r": ModelRoute(
                id="r",
                client_model_aliases=["test-model"],
                upstream_model="test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://x",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )


class _FakeTransport(UpstreamTransport):
    def __init__(self, model_text: str) -> None:
        self._model_text = model_text

    async def close(self) -> None:
        pass

    async def send(self, request):
        body = {
            "model": "test-model",
            "message": {"role": "assistant", "content": self._model_text},
            "done": True,
            "done_reason": "stop",
        }
        return UpstreamResponse(status_code=200, headers={}, body=json.dumps(body).encode())


async def _run(model_text: str):
    from agent_interop.model.profiles_v2 import ModelProfile, ProfileIndex, ToolBehaviorProfile
    from agent_interop.model.registry import ModelProfileRegistry

    profile = ModelProfile(
        id="test-repair-note",
        tool_behavior=ToolBehaviorProfile(
            presentation_mode="prompted",
            extractor_id="tool_call_envelope",
            output_envelope="tool_call",
        ),
    )
    index = ProfileIndex()
    index.add_profile(profile, {"match": {"model_patterns": [".*"]}})
    registry = ModelProfileRegistry(profiles=index)

    gw = Gateway(_config(), transport=_FakeTransport(model_text), profile_registry=registry)
    req = CanonicalRequest(
        model=CanonicalModelReference(requested_name="test-model"),
        messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read it")])],
        tools=[_TOOL],
        tool_choice=CanonicalToolChoice.auto(),
    )
    resp = await gw.handle_request(req, RequestContext())
    await gw.close()
    return resp


class TestRepairNoteFeedback:
    @pytest.mark.asyncio
    async def test_note_present_when_alias_repair_applied(self) -> None:
        model_text = '<tool_call>{"name":"read_file","arguments":{"file_path":"/tmp/x"}}</tool_call>'
        resp = await _run(model_text)

        assert resp.error is None
        tool_calls = [b for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
        assert len(tool_calls) == 1
        assert tool_calls[0].arguments == {"path": "/tmp/x"}

        notes = [
            b for b in resp.content
            if isinstance(b, CanonicalTextBlock) and b.text.startswith("[Interop]")
        ]
        assert len(notes) == 1
        assert "file_path" in notes[0].text
        assert "path" in notes[0].text

    @pytest.mark.asyncio
    async def test_no_note_when_call_was_already_valid(self) -> None:
        """The negative case matters as much as the positive one: a call
        that needed no repair must not carry any [Interop] note at all."""
        model_text = '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>'
        resp = await _run(model_text)

        assert resp.error is None
        tool_calls = [b for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
        assert len(tool_calls) == 1

        notes = [
            b for b in resp.content
            if isinstance(b, CanonicalTextBlock) and b.text.startswith("[Interop]")
        ]
        assert notes == []

    @pytest.mark.asyncio
    async def test_no_note_when_no_tool_call_at_all(self) -> None:
        resp = await _run("Just some plain text, no tool call here.")

        assert resp.error is None
        notes = [
            b for b in resp.content
            if isinstance(b, CanonicalTextBlock) and b.text.startswith("[Interop]")
        ]
        assert notes == []


class TestBuildRepairNoteUnit:
    """Direct unit coverage of Gateway._build_repair_note, independent of
    the full request pipeline."""

    def test_empty_decisions_yields_none(self) -> None:
        assert Gateway._build_repair_note([]) is None

    def test_valid_unchanged_yields_none(self) -> None:
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )

        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            outcome=RepairOutcome(status=RepairStatus.VALID_UNCHANGED, call_name="read_file"),
        )
        assert Gateway._build_repair_note([decision]) is None

    def test_repaired_with_steps_yields_compact_note(self) -> None:
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            RepairStep,
            ToolCallDecision,
        )

        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            outcome=RepairOutcome(
                status=RepairStatus.REPAIRED,
                call_name="read_file",
                steps=[RepairStep(rule="rename_aliased_fields", message="Renamed `file_path` -> `path`")],
            ),
        )
        note = Gateway._build_repair_note([decision])
        assert note is not None
        assert note.startswith("[Interop]")
        assert "read_file" in note
        assert "file_path" in note and "path" in note

    def test_repaired_with_no_steps_yields_none(self) -> None:
        """A REPAIRED status with an empty steps list (shouldn't happen in
        practice, but the helper must not crash or emit a blank note)."""
        from agent_interop.abi import (
            RawToolCallCandidate,
            RepairOutcome,
            RepairStatus,
            ToolCallDecision,
        )

        decision = ToolCallDecision(
            candidate=RawToolCallCandidate(id="1", name="read_file", raw_arguments="{}"),
            outcome=RepairOutcome(status=RepairStatus.REPAIRED, call_name="read_file", steps=[]),
        )
        assert Gateway._build_repair_note([decision]) is None
