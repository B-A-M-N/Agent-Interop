"""Tests for RepairPolicy.field_alias_policy enforcement.

The policy must actually control which alias sources are consulted:

* DISABLED       — no aliasing at all
* SCHEMA_ONLY    — only x-aliases declared on the tool schema
* COMPATIBILITY_PACK — schema aliases + REGISTERED compatibility-pack
  entries. Registered packs are maintainer-authored and static (see
  repair/aliases.py's module docstring) — they activate on a properly
  RESOLVED client identity (a populated CompatibilityKey naming client_id
  plus at least one other real dimension), NOT on compatibility_verified,
  which is reserved for a hypothetical future dynamic/learned alias
  source. This intentionally does NOT require per-tuple manual
  verification for a table that's identical on every installation.
"""

from __future__ import annotations

from agent_interop.abi import CanonicalTool
from agent_interop.config import FieldAliasPolicy, RepairPolicy, RepairTier
from agent_interop.repair.aliases import get_aliases_for_tool
from agent_interop.repair.pipeline import repair_one
from agent_interop.replay.types import CompatibilityKey

TOOL = CanonicalTool(
    name="Read",
    description="Read a file",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path",
                "x-aliases": ["p"],
            },
        },
        "required": ["file_path"],
    },
)

# No x-aliases declared anywhere on this schema — any alias found for it
# must have come from the compatibility pack, not the schema tier. Mirrors
# compatibility_packs/claude_code's real "Edit" entry (old_string has
# aliases including "old_str"), so this cleanly isolates pack-sourced
# aliasing from schema-declared aliasing. Tool name and canonical
# "file_path" field match Claude Code's actual real "Edit" tool schema —
# see compatibility_packs/claude_code, corrected after a live acceptance
# run against the real claude binary proved the old snake_case names
# ("edit_file") never matched anything Claude Code actually sends.
EDIT_TOOL = CanonicalTool(
    name="Edit",
    description="Edit a file",
    input_schema={
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
)


class TestFieldAliasPolicy:
    def test_disabled_returns_empty(self):
        result = get_aliases_for_tool(
            "Read", TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.DISABLED,
        )
        assert result == {}

    def test_schema_only_includes_x_aliases(self):
        result = get_aliases_for_tool(
            "Read", TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.SCHEMA_ONLY,
        )
        assert "p" in result.get("file_path", [])

    def test_schema_only_never_includes_pack_aliases(self):
        """SCHEMA_ONLY must never reach the compatibility pack, no matter
        how well-populated the compatibility key is."""
        key = CompatibilityKey(client_id="claude_code", model_id="claude-opus")
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.SCHEMA_ONLY,
        )
        assert result == {}


class TestRegisteredPackActivatesWithoutVerification:
    """The actual behavior change: a registered pack (claude_code) must
    activate for a properly-resolved identity WITHOUT compatibility_verified
    — that flag is not even passed in most of these tests, and the one
    that does pass it explicitly sets it False to prove it's ignored."""

    def test_no_compatibility_key_yields_no_pack_aliases(self):
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="claude_code",
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert result == {}

    def test_bare_client_id_key_with_no_other_dimension_yields_no_pack_aliases(self):
        """A CompatibilityKey with ONLY client_id set is not 'sufficiently
        populated' — this is the 'more than a loose client_id string
        match' requirement, not a verification requirement."""
        key = CompatibilityKey(client_id="claude_code")
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert result == {}

    def test_sufficiently_populated_key_activates_pack_without_verified_flag(self):
        """The core behavior change: compatibility_verified is never
        passed here at all (defaults to False) and the pack still fires,
        because it's a registered, static, maintainer-authored table."""
        key = CompatibilityKey(client_id="claude_code", model_id="claude-opus")
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "old_str" in result.get("old_string", [])

    def test_explicit_compatibility_verified_false_does_not_block_pack(self):
        key = CompatibilityKey(client_id="claude_code", model_id="claude-opus")
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
            compatibility_verified=False,
        )
        assert "old_str" in result.get("old_string", [])

    def test_unregistered_client_yields_no_pack_aliases_even_with_populated_key(self):
        """A well-populated key naming a client that has no registered
        pack module must still yield nothing — registration in code is a
        hard requirement, not something a caller can spoof via a
        plausible-looking key."""
        key = CompatibilityKey(client_id="totally-unknown-client", model_id="some-model")
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="totally-unknown-client",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert result == {}

    def test_backend_dimension_alone_is_sufficient(self):
        """Any one of model/profile/backend alongside client_id counts as
        'sufficiently populated' — not specifically model_id."""
        key = CompatibilityKey(client_id="claude_code", backend_kind="ollama")
        result = get_aliases_for_tool(
            "Edit", EDIT_TOOL.input_schema,
            client_id="claude_code",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "old_str" in result.get("old_string", [])


class TestPolicyFlowsThroughPipeline:
    def test_disabled_policy_blocks_alias_repair(self):
        policy = RepairPolicy(
            enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}),
            field_alias_policy=FieldAliasPolicy.DISABLED,
        )
        outcome = repair_one(
            "Read", {"path": "/tmp/x"},
            [TOOL], policy=policy, client_id="claude_code",
        )
        assert not outcome.is_accepted

    def test_compatibility_pack_policy_allows_alias_repair_without_verification(self):
        """End-to-end proof through the real pipeline: a registered pack
        alias (edit_file.old_string <- old_str, NOT declared on the
        tool's own schema) repairs a call with a sufficiently-populated
        key and compatibility_verified never set to True."""
        policy = RepairPolicy(
            enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}),
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        compat_key = CompatibilityKey(client_id="claude_code", model_id="test-model")
        outcome = repair_one(
            "Edit",
            {"file_path": "/tmp/x", "old_str": "foo", "new_string": "bar"},
            [EDIT_TOOL], policy=policy, client_id="claude_code", compatibility_key=compat_key,
        )
        assert outcome.is_accepted
        assert outcome.accepted == {"file_path": "/tmp/x", "old_string": "foo", "new_string": "bar"}

    def test_compatibility_pack_policy_still_blocks_without_sufficient_key(self):
        """Without a sufficiently-populated compatibility key, the pack
        must not fire even though the policy allows it in principle."""
        policy = RepairPolicy(
            enabled_tiers=frozenset({RepairTier.SAFE_SHAPE}),
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        outcome = repair_one(
            "Edit",
            {"file_path": "/tmp/x", "old_str": "foo", "new_string": "bar"},
            [EDIT_TOOL], policy=policy, client_id="claude_code",
        )
        assert not outcome.is_accepted
