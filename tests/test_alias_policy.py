"""Tests for RepairPolicy.field_alias_policy enforcement.

The policy must actually control which alias sources are consulted:

* DISABLED       — no aliasing at all
* SCHEMA_ONLY    — only x-aliases declared on the tool schema
* COMPATIBILITY_PACK — schema aliases + compatibility-pack entries,
  requiring a verified compatibility key.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalTool
from agent_interop.config import FieldAliasPolicy, RepairPolicy, RepairTier
from agent_interop.repair.aliases import get_aliases_for_tool
from agent_interop.repair.pipeline import repair_one
from agent_interop.replay.types import CompatibilityKey

TOOL = CanonicalTool(
    name="read_file",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path",
                "x-aliases": ["p"],
            },
        },
        "required": ["path"],
    },
)


class TestFieldAliasPolicy:
    def test_disabled_returns_empty(self):
        result = get_aliases_for_tool(
            "read_file", TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.DISABLED,
        )
        assert result == {}

    def test_schema_only_includes_x_aliases(self):
        result = get_aliases_for_tool(
            "read_file", TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.SCHEMA_ONLY,
        )
        assert "p" in result.get("path", [])

    def test_compatibility_pack_requires_compatibility_key(self):
        result = get_aliases_for_tool(
            "read_file", TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "p" in result.get("path", [])

    def test_compatibility_pack_with_key_activates_pack(self):
        key = CompatibilityKey(client_id="claude_code")
        result = get_aliases_for_tool(
            "read_file", TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "p" in result.get("path", [])


class TestPolicyFlowsThroughPipeline:
    def test_disabled_policy_blocks_alias_repair(self):
        policy = RepairPolicy(
            enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}),
            field_alias_policy=FieldAliasPolicy.DISABLED,
        )
        outcome = repair_one(
            "read_file", {"file_path": "/tmp/x"},
            [TOOL], policy=policy, client_id="claude_code",
        )
        assert not outcome.is_accepted

    def test_compatibility_pack_policy_allows_alias_repair(self):
        policy = RepairPolicy(
            enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}),
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        from agent_interop.replay.types import CompatibilityKey
        compat_key = CompatibilityKey(client_id="claude_code", model_id="test-model")
        outcome = repair_one(
            "read_file", {"file_path": "/tmp/x"},
            [TOOL], policy=policy, client_id="claude_code", compatibility_key=compat_key,
            compatibility_verified=True,
        )
        assert outcome.is_accepted
        assert outcome.accepted == {"path": "/tmp/x"}
