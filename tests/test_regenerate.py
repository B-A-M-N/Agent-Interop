"""Tests for hidden constrained regeneration."""
import json

from agent_interop.abi import CanonicalTool, SchemaIssue
from agent_interop.repair.regenerate import (
    RegenerationOrchestrator,
    build_correction_request,
)

SAMPLE_TOOL = CanonicalTool(
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


SAMPLE_ISSUES = [
    SchemaIssue(
        path=("path",),
        keyword="required",
        message="missing required property: path",
        expected="present",
        actual="missing",
    ),
]


def test_build_correction_request():
    prompt = build_correction_request(
        tool_name="read_file",
        raw_arguments={"encoding": "utf-8"},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
    )
    assert "read_file" in prompt
    assert "missing required property: path" in prompt
    assert '"path"' in prompt


def test_correction_request_truncation():
    large_args = {"data": "x" * 5000}
    prompt = build_correction_request(
        tool_name="read_file",
        raw_arguments=large_args,
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
    )
    assert "read_file" in prompt
    # Should not exceed reasonable size
    assert len(prompt) < 20000


async def _fake_regenerate_success(prompt: str) -> str:
    return json.dumps({"name": "read_file", "arguments": {"path": "/tmp/x", "encoding": "utf-8"}})


async def _fake_regenerate_empty(prompt: str) -> str:
    return ""


async def _fake_regenerate_bad_shape(prompt: str) -> str:
    return json.dumps({"foo": "bar"})


async def _fake_regenerate_wrong_tool(prompt: str) -> str:
    return json.dumps({"name": "write_file", "arguments": {"path": "/tmp/x"}})


async def test_regeneration_success():
    orch = RegenerationOrchestrator()
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=_fake_regenerate_success,
    )
    assert result is not None
    assert result["name"] == "read_file"
    assert result["arguments"]["path"] == "/tmp/x"


async def test_regeneration_empty_response():
    orch = RegenerationOrchestrator()
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=_fake_regenerate_empty,
    )
    assert result is None


async def test_regeneration_bad_shape():
    orch = RegenerationOrchestrator()
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=_fake_regenerate_bad_shape,
    )
    assert result is None


async def test_regeneration_wrong_tool():
    orch = RegenerationOrchestrator()
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=_fake_regenerate_wrong_tool,
    )
    assert result is None


async def test_regeneration_single_shot():
    """P0-repair-budget: one attempt() call = exactly ONE generation.
    Repetition moved to the caller (transaction budget); the orchestrator
    never loops internally, so a bad answer costs one dispatch, not N."""
    calls: list[str] = []

    async def always_bad(_prompt: str) -> str:
        calls.append(_prompt)
        return "not json at all"

    orch = RegenerationOrchestrator()
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=always_bad,
    )
    assert result is None
    assert len(calls) == 1, f"expected exactly one dispatch, got {len(calls)}"
    assert orch.attempts == 1


async def test_regeneration_oversized_correction_rejected_before_dispatch():
    """max_input_bytes is a real ceiling: an oversized correction prompt is
    rejected BEFORE the generation, not after the model paid for it."""
    calls: list[str] = []

    async def spy(_prompt: str) -> str:
        calls.append(_prompt)
        return "{}"

    from agent_interop.abi import CanonicalTool

    big_schema_tool = CanonicalTool(
        name="read_file",
        description="read",
        # ~100KB schema — far beyond a 1KB ceiling
        input_schema={"type": "object", "properties": {
            f"field_{i}": {"type": "string", "description": "x" * 100}
            for i in range(500)
        }},
    )
    orch = RegenerationOrchestrator(max_input_bytes=1024)
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=big_schema_tool,
        regenerate_fn=spy,
    )
    assert result is None
    assert calls == [], "oversized correction must not reach the model"


async def test_regeneration_deadline_enforced():
    """An already-exceeded deadline aborts before dispatch."""
    import time as _time

    calls: list[str] = []

    async def spy(_prompt: str) -> str:
        calls.append(_prompt)
        return "{}"

    orch = RegenerationOrchestrator(max_latency_ms=0)
    orch.start_time = _time.monotonic() - 1  # deadline already blown
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=spy,
    )
    assert result is None
    assert calls == []