"""P0.5 + P0.6 regression:

P0.5 (wrapper normalization)
  ``{"tool": "X", "params": {...}}`` must normalize the same way
  ``{"name": "X", "params": {...}}`` does — the ``{"tool": ...}`` branch
  previously missed the ``params`` key.

P0.5.2 / P0.6 (client tool-name aliases)
  A small model emitting a generic tool name (``read_file``) against a client
  that declared ``Read`` must be recovered via the resolved compatibility pack,
  and ONLY when exactly one declared tool matches. The pack identity is a
  strong (client_id, version, fingerprint) tuple — a bare client_id claiming
  to be claude_code does not inherit aliases without a matching version.
"""

from __future__ import annotations


from agent_interop.abi import CanonicalTool
from agent_interop.compatibility_packs import (
    ClientCompatibilityPack,
    get_tool_name_alias_map,
    register_pack,
)
from agent_interop.extraction import _normalize_name_and_args_from_json
from agent_interop.repair.pipeline import canonicalize_tool_name


def _read_tool() -> CanonicalTool:
    return CanonicalTool(
        name="Read",
        description="Read a file",
        input_schema={
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    )


class TestWrapperParamsNormalization:
    def test_name_params_branch(self):
        name, args = _normalize_name_and_args_from_json(
            '{"name": "Read", "params": {"file_path": "x"}}'
        )
        assert name == "Read"
        assert args == {"file_path": "x"}

    def test_tool_params_branch(self):
        # Previously this returned the whole raw payload as arguments because
        # the {"tool": ...} branch omitted "params".
        name, args = _normalize_name_and_args_from_json(
            '{"tool": "Read", "params": {"file_path": "README.md"}}'
        )
        assert name == "Read"
        assert args == {"file_path": "README.md"}

    def test_tool_args_branch(self):
        name, args = _normalize_name_and_args_from_json(
            '{"tool": "Read", "arguments": {"file_path": "y"}}'
        )
        assert name == "Read"
        assert args == {"file_path": "y"}


class TestClientToolNameAliases:
    def test_read_file_resolves_to_read(self):
        tools = [_read_tool()]
        alias_map = get_tool_name_alias_map(
            "claude_code",
            client_version="2.1.220",
            tool_schema_fingerprint=None,
            declared_tool_names=tuple(t.name for t in tools),
        )
        assert alias_map.get("read_file") == "Read"
        # canonicalize via the map
        assert canonicalize_tool_name("read_file", tools, tool_aliases=alias_map) == "Read"

    def test_no_match_without_declared_tool(self):
        # Client declared "Write" but not "Read" — read_file must NOT resolve
        # to anything, so we never invent a tool the client didn't expose.
        tools = [CanonicalTool(name="Write", description="", input_schema={})]
        alias_map = get_tool_name_alias_map(
            "claude_code",
            client_version="2.1.220",
            declared_tool_names=tuple(t.name for t in tools),
        )
        assert "read_file" not in alias_map
        assert canonicalize_tool_name("read_file", tools, tool_aliases=alias_map) is None

    def test_version_mismatch_blocks_aliases(self):
        # A claude_code 3.x with a different tool schema must not inherit 2.x
        # aliases when a fingerprint or version mismatch occurs.
        tools = [_read_tool()]
        alias_map = get_tool_name_alias_map(
            "claude_code",
            client_version="3.0.0",
            tool_schema_fingerprint=None,
            declared_tool_names=tuple(t.name for t in tools),
        )
        # 2.x prefix constraint does not satisfy 3.0.0
        assert "read_file" not in alias_map

    def test_unknown_client_no_aliases(self):
        tools = [_read_tool()]
        alias_map = get_tool_name_alias_map(
            "some_other_agent",
            client_version="1.0.0",
            declared_tool_names=tuple(t.name for t in tools),
        )
        assert alias_map == {}

    def test_ambiguous_alias_dropped(self):
        # Two declared tools both claim the alias "run" → must be dropped.
        pack = ClientCompatibilityPack(
            client_id="test_client",
            version_constraint="*",
            tool_name_aliases={
                "Bash": ("run",),
                "Runner": ("run",),
            },
        )
        register_pack(pack)
        tools = [
            CanonicalTool(name="Bash", description="", input_schema={}),
            CanonicalTool(name="Runner", description="", input_schema={}),
        ]
        alias_map = get_tool_name_alias_map(
            "test_client",
            declared_tool_names=tuple(t.name for t in tools),
        )
        assert "run" not in alias_map
