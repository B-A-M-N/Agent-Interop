"""Tests for cursor-scoped repair rules.

Verifies that:
- Rules only propose mutations at the cursor's targeted path.
- The pipeline passes cursor and context correctly.
- Rename alias rule uses cursor scope.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalTool, RepairCursor, SchemaIssue
from agent_interop.config import FieldAliasPolicy, RepairPolicy, RepairTier
from agent_interop.repair.pipeline import repair_one

TOOLS = [
    CanonicalTool(
        name="read_file",
        description="Read a file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "x-aliases": ["p", "file_path"]},
                "offset": {"type": "integer"},
                "extra": {"type": "string"},
            },
            "required": ["path"],
        },
    ),
]

POLICY = RepairPolicy(
    enabled_tiers=frozenset({RepairTier.SAFE_SHAPE, RepairTier.COERCIVE}),
    field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
)


class TestCursorScopedRename:
    def test_cursor_scopes_alias_rename_to_targeted_field(self):
        from agent_interop.repair.rules import RepairRuleContext, rename_aliased_fields
        from agent_interop.replay.types import CompatibilityKey

        tool = TOOLS[0]
        args = {"file_path": "/tmp/x", "extra": "value"}
        # Cursor targets 'extra' field — rename should NOT fire for file_path→path
        cursor = RepairCursor(
            issue=SchemaIssue(path=["extra"], keyword="type"),
            instance_path=["extra"],
            schema_path=["properties", "extra"],
            parent_instance=args,
            current_value="value",
            parent_schema=tool.input_schema,
            target_schema=tool.input_schema["properties"]["extra"],
            tool=tool,
            client_id="claude_code",
        )
        ctx = RepairRuleContext(
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
            compatibility_key=CompatibilityKey(client_id="claude_code"),
        )
        proposal = rename_aliased_fields(args, cursor, ctx)
        # Should not fire because cursor targets 'extra', not 'path'
        assert proposal is None

    def test_cursor_for_path_field_renames_alias(self):
        from agent_interop.repair.rules import RepairRuleContext, rename_aliased_fields
        from agent_interop.replay.types import CompatibilityKey

        tool = TOOLS[0]
        args = {"p": "/tmp/x"}
        cursor = RepairCursor(
            issue=SchemaIssue(path=[], keyword="required", expected="['path']"),
            instance_path=[],
            schema_path=[],
            parent_instance=args,
            current_value=args,
            parent_schema=tool.input_schema,
            target_schema=tool.input_schema,
            tool=tool,
            client_id="claude_code",
        )
        ctx = RepairRuleContext(
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
            compatibility_key=CompatibilityKey(client_id="claude_code"),
        )
        proposal = rename_aliased_fields(args, cursor, ctx)
        assert proposal is not None
        assert proposal.target_path == ["path"]
        assert proposal.after == "/tmp/x"
        assert proposal.rule_id == "rename_aliased_fields"


class TestCursorAwarePipeline:
    def test_pipeline_routes_cursor_to_rule(self):
        from agent_interop.replay.types import CompatibilityKey
        compat_key = CompatibilityKey(client_id="claude_code", model_id="test-model")
        out = repair_one(
            "read_file", {"file_path": "/tmp/x"},
            TOOLS, policy=POLICY, client_id="claude_code",
            compatibility_key=compat_key,
        )
        assert out.is_accepted
        assert out.accepted == {"path": "/tmp/x"}


class TestNestedRepair:
    def test_nested_stringified_array(self):
        """Repair should work on nested fields, not just root level."""
        tool = CanonicalTool(
            name="config_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "config": {
                        "type": "object",
                        "properties": {
                            "files": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["files"],
                    },
                },
                "required": ["config"],
            },
        )
        policy = RepairPolicy(enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}))
        out = repair_one(
            "config_tool",
            {"config": {"files": '["a.txt", "b.txt"]'}},
            [tool],
            policy=policy,
        )
        assert out.is_accepted
        assert out.accepted == {"config": {"files": ["a.txt", "b.txt"]}}

    def test_nested_coercion(self):
        tool = CanonicalTool(
            name="nested_tool",
            input_schema={
                "type": "object",
                "properties": {
                    "settings": {
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                        },
                        "required": ["count"],
                    },
                },
                "required": ["settings"],
            },
        )
        policy = RepairPolicy(enabled_tiers=frozenset({RepairTier.SAFE_SHAPE, RepairTier.COERCIVE}))
        out = repair_one(
            "nested_tool",
            {"settings": {"count": "42"}},
            [tool],
            policy=policy,
        )
        assert out.is_accepted
        assert out.accepted == {"settings": {"count": 42}}

    def test_out_of_scope_mutation_rejected(self):
        """A rule must not modify fields outside its cursor path."""
        before = {"a": "[\"x\"]", "b": "[\"y\"]"}
        # Both are stringified arrays, but repair should fix one at a time
        tool = CanonicalTool(
            name="multi",
            input_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "array", "items": {"type": "string"}},
                    "b": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["a", "b"],
            },
        )
        policy = RepairPolicy(enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}))
        out = repair_one("multi", dict(before), [tool], policy=policy)
        assert out.is_accepted
        # Both should be repaired but via separate iterations
        assert out.accepted == {"a": ["x"], "b": ["y"]}
        assert len(out.steps) == 2
        # Each step targeted a different field
        step_paths = {s.path for s in out.steps}
        assert "a" in step_paths
        assert "b" in step_paths
