"""P0-20 (review #10): private argument JSON Schema validation.

The strict private-argument parser must reject malformed values, not just
missing keys.  The pre-fix code only checked ``required``; type, range,
enum, and additionalProperties were silently accepted and the internal
tool got a `{}` or wrong-type payload — defeating the purpose of the
private control plane.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent_interop.abi import CanonicalTool
from agent_interop.gateway import Gateway
from agent_interop.private_loop import parse_private_arguments


def _candidate(name: str, raw: object) -> SimpleNamespace:
    return SimpleNamespace(id="call_x", name=name, raw_arguments=raw)


def test_missing_required_argument_still_rejected():
    tool = CanonicalTool(
        name="read_result",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    )
    args, err = parse_private_arguments(
        _candidate("read_result", "{}"), tool,
    )
    assert args == {}
    assert "missing required" in err.lower()


def test_wrong_type_argument_rejected():
    tool = CanonicalTool(
        name="read_result",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    )
    args, err = parse_private_arguments(
        _candidate("read_result", json_dumps({"ref": 123})), tool,
    )
    assert args == {}
    assert "schema violation" in err.lower() or "string" in err.lower()


def test_enum_violation_rejected():
    tool = CanonicalTool(
        name="search_history",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"scope": {"enum": ["session", "global"]}},
            "required": ["scope"],
        },
    )
    args, err = parse_private_arguments(
        _candidate("search_history", json_dumps({"scope": "everywhere"})), tool,
    )
    assert args == {}
    assert "schema violation" in err.lower() or "enum" in err.lower()


def test_valid_arguments_pass():
    tool = CanonicalTool(
        name="read_result",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
        },
    )
    args, err = parse_private_arguments(
        _candidate("read_result", json_dumps({"ref": "abc"})), tool,
    )
    assert args == {"ref": "abc"}
    assert err is None


def test_additional_properties_violation_rejected():
    tool = CanonicalTool(
        name="read_result",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"ref": {"type": "string"}},
            "required": ["ref"],
            "additionalProperties": False,
        },
    )
    args, err = parse_private_arguments(
        _candidate("read_result", json_dumps({"ref": "abc", "extra": 1})), tool,
    )
    assert args == {}
    assert "schema violation" in err.lower() or "additional" in err.lower()


def test_bare_object_schema_always_valid():
    """A bare {"type":"object"} contract — Interop-authored tools that
    take any payload — must not reject valid arguments with a spurious
    'schema invalid' error (P1-H negative-cache behavior)."""
    tool = CanonicalTool(
        name="anything",
        description="d",
        input_schema={"type": "object"},
    )
    args, err = parse_private_arguments(
        _candidate("anything", json_dumps({"x": 1, "y": [1, 2]})), tool,
    )
    assert args == {"x": 1, "y": [1, 2]}
    assert err is None


def json_dumps(obj: dict) -> str:
    import json
    return json.dumps(obj)
