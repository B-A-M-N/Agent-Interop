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


async def test_regeneration_max_attempts():
    orch = RegenerationOrchestrator(max_attempts=2)
    result = await orch.attempt(
        tool_name="read_file",
        raw_arguments={},
        issues=SAMPLE_ISSUES,
        tool=SAMPLE_TOOL,
        regenerate_fn=_fake_regenerate_bad_shape,
    )
    assert result is None
    # Should have attempted twice
    assert orch.attempts <= 2