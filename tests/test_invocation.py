"""Phase 3 gate: InvocationPlan, capability resolution, and mode resolution tests.

Validates:
1. Mode resolution rules from the config/capability table
2. Prompted mode produces a reproducible prompt contract
3. Deterministic serialization for prompt-cache friendliness
4. NATIVE mode disables prompt injection
5. DISABLED mode produces empty contract
6. TEXTUAL mode produces description-only contract
7. Edge cases: no tools, empty tools, AUTO without capability
"""

from __future__ import annotations

from agent_interop.abi import CanonicalTool, CanonicalToolChoice
from agent_interop.config import ToolMode
from agent_interop.repair.invocation import (
    InvocationPlan,
    ProfileCapability,
    build_invocation_plan,
    build_tool_descriptions,
    resolve_tool_mode,
    serialize_tool_schema,
)

# ── Sample tools ──────────────────────────────────────────────────────────

SAMPLE_TOOLS = [
    CanonicalTool(
        name="get_weather",
        description="Get the current weather for a city",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    ),
    CanonicalTool(
        name="read_file",
        description="Read the contents of a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
            },
            "required": ["path"],
        },
    ),
]

SAMPLE_TOOL_NAMES = ["get_weather", "read_file"]


# ── Mode resolution tests ────────────────────────────────────────────────


class TestModeResolution:
    """Verify the mode resolution table from PROPOSED_CHANGES.md."""

    def test_auto_structured_resolves_to_native(self) -> None:
        assert resolve_tool_mode(ToolMode.AUTO, ProfileCapability.STRUCTURED) == ToolMode.NATIVE

    def test_auto_textual_dialect_resolves_to_prompted(self) -> None:
        assert resolve_tool_mode(ToolMode.AUTO, ProfileCapability.TEXTUAL_DIALECT) == ToolMode.PROMPTED

    def test_auto_chat_only_resolves_to_disabled(self) -> None:
        assert resolve_tool_mode(ToolMode.AUTO, ProfileCapability.CHAT_ONLY) == ToolMode.DISABLED

    def test_auto_none_capability_resolves_to_disabled(self) -> None:
        assert resolve_tool_mode(ToolMode.AUTO) == ToolMode.DISABLED

    def test_native_any_capability_resolves_to_native(self) -> None:
        for cap in ProfileCapability:
            assert resolve_tool_mode(ToolMode.NATIVE, cap) == ToolMode.NATIVE

    def test_prompted_any_capability_resolves_to_prompted(self) -> None:
        for cap in ProfileCapability:
            assert resolve_tool_mode(ToolMode.PROMPTED, cap) == ToolMode.PROMPTED

    def test_textual_any_capability_resolves_to_textual(self) -> None:
        for cap in ProfileCapability:
            assert resolve_tool_mode(ToolMode.TEXTUAL, cap) == ToolMode.TEXTUAL

    def test_disabled_any_capability_resolves_to_disabled(self) -> None:
        for cap in ProfileCapability:
            assert resolve_tool_mode(ToolMode.DISABLED, cap) == ToolMode.DISABLED


# ── InvocationPlan construction tests ──────────────────────────────────────


class TestInvocationPlanConstruction:
    """Verify build_invocation_plan produces correct plans."""

    def test_native_mode_plan(self) -> None:
        plan = build_invocation_plan(
            tools=SAMPLE_TOOLS,
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.NATIVE,
            model_profile=None,
        )
        assert isinstance(plan, InvocationPlan)
        assert plan.effective_tool_mode == ToolMode.NATIVE
        assert plan.native_tools_enabled is True
        assert plan.prompt_contract == ""
        assert plan.parser_id is None
        assert plan.output_envelope is None
        assert plan.constrained_output is False

    def test_prompted_mode_contract(self) -> None:
        """Prompted mode produces a reproducible prompt contract with
        tool schemas, envelope, and no native tool_choice."""
        plan = build_invocation_plan(
            tools=SAMPLE_TOOLS,
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.PROMPTED,
            model_profile=None,
        )
        assert plan.native_tools_enabled is False
        assert plan.output_envelope == "tool_call"
        assert plan.parser_id == "tool_call_envelope"
        assert plan.constrained_output is False
        assert len(plan.prompt_contract) > 0
        # Must contain tool names
        assert "get_weather" in plan.prompt_contract
        assert "read_file" in plan.prompt_contract
        # Must contain the envelope instruction
        assert "<tool_call>" in plan.prompt_contract

    def test_prompted_determinism(self) -> None:
        """Verify determinism: same input produces identical prompt_contract."""
        plan1 = build_invocation_plan(
            tools=SAMPLE_TOOLS, tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.PROMPTED, model_profile=None,
        )
        plan2 = build_invocation_plan(
            tools=SAMPLE_TOOLS, tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.PROMPTED, model_profile=None,
        )
        assert plan1.prompt_contract == plan2.prompt_contract

    def test_disabled_mode_plan(self) -> None:
        plan = build_invocation_plan(
            tools=SAMPLE_TOOLS, tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.DISABLED, model_profile=None,
        )
        assert plan.effective_tool_mode == ToolMode.DISABLED
        assert plan.native_tools_enabled is False
        assert plan.prompt_contract == ""
        assert plan.parser_id is None

    def test_textual_mode_plan(self) -> None:
        plan = build_invocation_plan(
            tools=SAMPLE_TOOLS, tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.TEXTUAL, model_profile=None,
        )
        assert plan.effective_tool_mode == ToolMode.TEXTUAL
        assert plan.native_tools_enabled is False
        assert plan.parser_id is None
        assert plan.output_envelope is None
        # Should have description-only (no schema JSON)
        assert "get_weather" in plan.prompt_contract
        assert "read_file" in plan.prompt_contract

    def test_no_tools_native(self) -> None:
        plan = build_invocation_plan(
            tools=[], tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.NATIVE, model_profile=None,
        )
        assert plan.native_tools_enabled is True
        assert plan.effective_tool_mode == ToolMode.NATIVE

    def test_no_tools_prompted(self) -> None:
        plan = build_invocation_plan(
            tools=[], tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.PROMPTED, model_profile=None,
        )
        assert plan.native_tools_enabled is False
        # Empty prompted contract since there are no tools
        assert plan.prompt_contract == ""

    def test_plan_has_tool_names(self) -> None:
        plan = build_invocation_plan(
            tools=SAMPLE_TOOLS, tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.NATIVE, model_profile=None,
        )
        assert list(plan.tool_names) == SAMPLE_TOOL_NAMES


# ── Deterministic serialization ────────────────────────────────────────────


class TestDeterministicSerialization:
    """Verify tool schemas serialize deterministically."""

    def test_sorted_keys(self) -> None:
        schema = {"b": 2, "a": 1, "c": {"z": 26, "y": 25}}
        result = serialize_tool_schema(schema)
        assert result == '{"a":1,"b":2,"c":{"y":25,"z":26}}'

    def test_determinism(self) -> None:
        schema = {"type": "object", "properties": {"city": {"type": "string"}}}
        assert serialize_tool_schema(schema) == serialize_tool_schema(schema)

    def test_no_whitespace(self) -> None:
        """Stable JSON with no extra whitespace for prompt cache."""
        schema = {"type": "object"}
        result = serialize_tool_schema(schema)
        assert " " not in result


# ── Tool descriptions formatting ──────────────────────────────────────────


class TestToolDescriptions:
    """Verify build_tool_descriptions formatting."""

    def test_includes_schema_by_default(self) -> None:
        result = build_tool_descriptions(SAMPLE_TOOLS)
        assert "<tool name=" in result
        assert '"city"' in result  # schema content
        assert '"path"' in result

    def test_without_schemas(self) -> None:
        result = build_tool_descriptions(SAMPLE_TOOLS, include_schemas=False)
        assert "<tool name=" in result
        assert '"city"' not in result  # no schema content
        assert "Get the current weather" in result

    def test_single_tool(self) -> None:
        result = build_tool_descriptions([SAMPLE_TOOLS[0]])
        assert result.count("<tool") == 1
        assert "get_weather" in result

    def test_empty_tools(self) -> None:
        result = build_tool_descriptions([])
        assert result == ""


# ── Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_auto_requires_capability(self) -> None:
        """AUTO mode with no capability should resolve to DISABLED (not crash)."""
        mode = resolve_tool_mode(ToolMode.AUTO)
        assert mode == ToolMode.DISABLED

    def test_tools_with_duplicate_names(self) -> None:
        """Duplicate names are preserved in tool_names list."""
        dup_tools = [SAMPLE_TOOLS[0], SAMPLE_TOOLS[0]]
        plan = build_invocation_plan(tools=dup_tools, tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.NATIVE)
        assert len(plan.tool_names) == 2

    def test_prompted_no_tools_empty_contract(self) -> None:
        """Without tools, prompted mode produces empty contract."""
        plan = build_invocation_plan(tools=[], tool_choice=CanonicalToolChoice.auto(), route_mode=ToolMode.PROMPTED)
        assert plan.prompt_contract == ""