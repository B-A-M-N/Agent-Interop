"""hermes-agent integration tests.

hermes-agent has no distinguishing header/User-Agent of its own when
pointed at a custom/local endpoint (confirmed by reading its source, see
agents/hermes_agent.py's module docstring) — its config schema does
support model.default_headers on the OpenAI wire, which this integration
uses to self-assert identity via X-Interop-Client, and HERMES_HOME
(a real, respected env var in hermes-agent's own code) to point it at a
session-scoped config without touching the user's real one.
"""

from __future__ import annotations

import shutil

import yaml  # type: ignore[import-untyped]

from agent_interop.abi import ProtocolKind
from agent_interop.agents.base import AgentLaunchContext
from agent_interop.agents.hermes_agent import (
    HERMES_INTEROP_CLIENT_HEADER,
    HERMES_INTEROP_CLIENT_ID,
    HermesAgentIntegration,
    _generate_temp_hermes_home,
)
from agent_interop.agents.registry import get_agent_integration


def _ctx(**overrides) -> AgentLaunchContext:
    defaults = {
        "route": "test-route",
        "gateway_url": "http://127.0.0.1:8090",
        "model_name": "qwen3-coder",
        "session_credential": "test-cred-123",
        "extra_args": (),
    }
    defaults.update(overrides)
    return AgentLaunchContext(**defaults)


class TestRegistration:
    def test_registered_under_hermes_agent_id(self):
        integration = get_agent_integration("hermes-agent")
        assert integration is not None
        assert isinstance(integration, HermesAgentIntegration)
        assert integration.id == "hermes-agent"


class TestTempHomeGeneration:
    def test_config_round_trips(self):
        path = _generate_temp_hermes_home(
            gateway_url="http://127.0.0.1:8090/",
            model_name="qwen3-coder:latest",
            api_key="test-cred-123",
        )
        try:
            assert path is not None
            data = yaml.safe_load((path / "config.yaml").read_text())
            model = data["model"]
            assert model["default"] == "qwen3-coder:latest"
            assert model["provider"] == "custom"
            # Trailing slash stripped and /v1 appended — the OpenAI SDK
            # hermes-agent wraps appends "/chat/completions" directly to
            # base_url rather than inserting "/v1" itself, and the real
            # gateway route is /v1/chat/completions. Confirmed live: without
            # the /v1 segment, hermes-agent got a 404 posting to the bare
            # path.
            assert model["base_url"] == "http://127.0.0.1:8090/v1"
            assert model["api_key"] == "test-cred-123"
            assert model["default_headers"][HERMES_INTEROP_CLIENT_HEADER] == HERMES_INTEROP_CLIENT_ID
        finally:
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)

    def test_special_characters_in_model_name_do_not_break_yaml(self):
        """model_name is ultimately derived from the --model CLI argument
        (arbitrary user-supplied text) — yaml.safe_dump must escape it
        correctly rather than needing hand-rolled escaping."""
        evil_model = 'x"\ndefault_headers:\n  evil: "yes"\n'
        path = _generate_temp_hermes_home(
            gateway_url="http://127.0.0.1:8090",
            model_name=evil_model,
            api_key="test-cred-123",
        )
        try:
            assert path is not None
            data = yaml.safe_load((path / "config.yaml").read_text())
            assert data["model"]["default"] == evil_model
            # The injected "evil" header must not have escaped into the
            # real default_headers block.
            assert "evil" not in data["model"]["default_headers"]
        finally:
            if path is not None:
                shutil.rmtree(path, ignore_errors=True)


class TestBuildLaunch:
    def test_sets_hermes_home_env_var(self):
        integration = HermesAgentIntegration()
        spec = integration.build_launch(_ctx())
        try:
            assert spec.readiness == "ready"
            assert spec.protocol == ProtocolKind.OPENAI_CHAT
            assert "HERMES_HOME" in spec.env
            config_path = f"{spec.env['HERMES_HOME']}/config.yaml"
            with open(config_path) as f:
                data = yaml.safe_load(f)
            assert data["model"]["default"] == "qwen3-coder"
            assert data["model"]["base_url"] == "http://127.0.0.1:8090/v1"
        finally:
            if spec.cleanup:
                spec.cleanup()

    def test_command_is_hermes_plus_extra_args(self):
        integration = HermesAgentIntegration()
        spec = integration.build_launch(_ctx(extra_args=("-z", "do the thing")))
        try:
            assert spec.command == ["hermes", "-z", "do the thing"]
        finally:
            if spec.cleanup:
                spec.cleanup()

    def test_no_extra_args_yields_bare_command(self):
        integration = HermesAgentIntegration()
        spec = integration.build_launch(_ctx())
        try:
            assert spec.command == ["hermes"]
        finally:
            if spec.cleanup:
                spec.cleanup()

    def test_cleanup_removes_temp_home(self):
        integration = HermesAgentIntegration()
        spec = integration.build_launch(_ctx())
        assert spec.env["HERMES_HOME"]
        from pathlib import Path
        temp_home = Path(spec.env["HERMES_HOME"])
        assert temp_home.exists()
        assert spec.cleanup is not None
        spec.cleanup()
        assert not temp_home.exists()

    def test_session_credential_never_written_to_env(self):
        """The credential goes into the (temp, session-scoped) config
        file's api_key field, not a process env var another process on
        the same machine could read via /proc."""
        integration = HermesAgentIntegration()
        spec = integration.build_launch(_ctx(session_credential="super-secret-token"))
        try:
            assert "super-secret-token" not in spec.env.values()
        finally:
            if spec.cleanup:
                spec.cleanup()


class TestCompatibilityPackAliases:
    """The pack's canonical field must be "path" (hermes-agent's own real
    field name, confirmed by reading tools/file_tools.py) with "file_path"
    and friends as aliases — the same words as Claude Code's pack, but the
    OPPOSITE canonical/alias direction, since the two clients' real tools
    disagree on which name is native."""

    def test_read_file_aliases(self):
        from agent_interop.config import FieldAliasPolicy
        from agent_interop.repair.aliases import get_aliases_for_tool
        from agent_interop.replay.types import CompatibilityKey

        key = CompatibilityKey(client_id="hermes_agent", model_id="test-model")
        result = get_aliases_for_tool(
            "read_file", {"type": "object", "properties": {"path": {"type": "string"}}},
            client_id="hermes_agent",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "file_path" in result.get("path", [])

    def test_patch_old_new_string_aliases(self):
        from agent_interop.config import FieldAliasPolicy
        from agent_interop.repair.aliases import get_aliases_for_tool
        from agent_interop.replay.types import CompatibilityKey

        key = CompatibilityKey(client_id="hermes_agent", model_id="test-model")
        result = get_aliases_for_tool(
            "patch", {"type": "object", "properties": {}},
            client_id="hermes_agent",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "old_str" in result.get("old_string", [])
        assert "new_str" in result.get("new_string", [])

    def test_terminal_command_alias(self):
        from agent_interop.config import FieldAliasPolicy
        from agent_interop.repair.aliases import get_aliases_for_tool
        from agent_interop.replay.types import CompatibilityKey

        key = CompatibilityKey(client_id="hermes_agent", model_id="test-model")
        result = get_aliases_for_tool(
            "terminal", {"type": "object", "properties": {}},
            client_id="hermes_agent",
            compatibility_key=key,
            field_alias_policy=FieldAliasPolicy.COMPATIBILITY_PACK,
        )
        assert "cmd" in result.get("command", [])


class TestDefaultPolicyIsCompatibilityPack:
    def test_repair_config_default_is_compatibility_pack(self):
        from agent_interop.config import RepairConfig
        assert RepairConfig().field_aliases == "compatibility_pack"
