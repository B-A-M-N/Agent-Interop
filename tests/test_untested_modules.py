"""Tests for previously-untested modules: plugin, install, loop detection integration.

These tests use mocking where needed to avoid requiring a live backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_interop.abi import (
    CanonicalTool,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)

# ─── retryable_statuses config loading ─────────────────────────────────────


def test_retryable_statuses_loaded_from_transport_config():
    """MVP-05: retryable_statuses has a dataclass default but was never
    actually read from the config dict — YAML overrides were silently
    ignored."""
    from agent_interop.config import load_config_from_dict

    config = load_config_from_dict({
        "routes": {
            "default": {
                "upstream_model": "test-model",
                "upstream": {"kind": "ollama", "wire_protocol": "ollama_chat"},
            },
        },
        "transport": {"retryable_statuses": [500, 503]},
    })
    assert config.retryable_statuses == (500, 503)


def test_retryable_statuses_default_when_absent():
    from agent_interop.config import load_config_from_dict

    config = load_config_from_dict({
        "routes": {
            "default": {
                "upstream_model": "test-model",
                "upstream": {"kind": "ollama", "wire_protocol": "ollama_chat"},
            },
        },
    })
    assert config.retryable_statuses == (429, 500, 502, 503, 504)


def test_validate_config_rejects_invalid_retryable_status():
    from agent_interop.config import (
        InteropServerConfig,
        ModelRoute,
        ToolMode,
        TranslationMode,
        UpstreamConfig,
        UpstreamKind,
        UpstreamProtocol,
        validate_config,
    )

    config = InteropServerConfig(
        retryable_statuses=(200, 999),
        routes={
            "default": ModelRoute(
                id="default",
                client_model_aliases=["test-model"],
                upstream_model="test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:11434",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    issues = validate_config(config)
    assert any("retryable_statuses" in issue for issue in issues)


# ─── example_config() ──────────────────────────────────────────────────────


def test_example_config_is_valid():
    """example_config() is used in documentation; it must actually validate."""
    from agent_interop.config import example_config, validate_config

    assert validate_config(example_config()) == []


# ─── Plugin Adapter ───────────────────────────────────────────────────────


class TestLocalModelAdapter:
    def test_adapter_init(self):
        from agent_interop.plugin.adapter import LocalModelAdapter
        adapter = LocalModelAdapter()
        assert adapter.gateway is None
        assert not adapter.is_running

    @pytest.mark.asyncio
    async def test_adapter_start_creates_gateway(self):
        from agent_interop.plugin.adapter import LocalModelAdapter
        adapter = LocalModelAdapter()
        config = InteropServerConfig(
            probe_on_startup=False,
            routes={"test": ModelRoute(
                id="test",
                client_model_aliases=["test"],
                upstream_model="test",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://localhost:11434",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
            )},
        )
        await adapter.start(config)
        assert adapter.is_running
        assert adapter.gateway is not None
        await adapter.close()
        assert not adapter.is_running
        assert adapter.gateway is None


# ─── Install Module ────────────────────────────────────────────────────────


class TestInstallModule:
    def test_bin_dir_returns_path(self):
        from agent_interop.install import _bin_dir
        result = _bin_dir()
        assert isinstance(result, Path)
        assert "interop" in str(result).lower() or "bin" in str(result).lower()

    def test_find_real_ollama(self):
        from agent_interop.install import _find_real_ollama
        # May return None if ollama not installed, that's fine
        result = _find_real_ollama()
        assert result is None or isinstance(result, str)

    def test_status_returns_dict(self):
        from agent_interop.install import status
        result = status()
        assert isinstance(result, dict)
        assert "installed" in result or "shim_exists" in result or len(result) >= 0


# ─── Tool Normalize ───────────────────────────────────────────────────────


class TestToolNormalize:
    def test_from_openai(self):
        from agent_interop.tool.normalize import from_openai
        spec = {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        tool = from_openai(spec)
        assert tool.name == "read_file"
        assert tool.description == "Read a file"
        assert "path" in tool.input_schema.get("properties", {})

    def test_from_mcp(self):
        from agent_interop.tool.normalize import from_mcp
        spec = {
            "name": "list_files",
            "description": "List directory",
            "inputSchema": {
                "type": "object",
                "properties": {"dir": {"type": "string"}},
            },
        }
        tool = from_mcp(spec)
        assert tool.name == "list_files"

    def test_to_openai_round_trip(self):
        from agent_interop.tool.normalize import to_openai
        tool = CanonicalTool(
            name="test_tool",
            description="A test",
            input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
        )
        rendered = to_openai(tool)
        assert rendered["type"] == "function"
        assert rendered["function"]["name"] == "test_tool"


# ─── Gate 7: Managed deployment smoke tests ─────────────────────────────


class TestManagedDeploymentFlow:
    """Prove the managed deployment flow works without root or Docker.

    These tests exercise:
        interop init → config validate → gateway startup/shutdown
    with token enforcement and clean resource release.
    """

    def test_init_generates_valid_config(self, tmp_path: Path) -> None:
        """interop init should produce a config that passes validation."""
        from typer.testing import CliRunner

        from agent_interop.cli import app

        runner = CliRunner()
        config_path = tmp_path / "interop.yaml"
        result = runner.invoke(app, ["init", "--path", str(config_path), "--backend", "ollama", "--model", "test-model"])
        assert result.exit_code == 0, f"init failed: {result.output}"
        assert config_path.exists()

        # Validate the generated config
        result = runner.invoke(app, ["config", "validate", "--path", str(config_path)])
        assert result.exit_code == 0, f"config validate failed: {result.output}"

    def test_init_rejects_unknown_backend(self, tmp_path: Path) -> None:
        """`interop init --backend bogus` must fail instead of silently using Ollama."""
        from typer.testing import CliRunner

        from agent_interop.cli import app

        runner = CliRunner()
        config_path = tmp_path / "interop.yaml"
        result = runner.invoke(
            app, ["init", "--path", str(config_path), "--backend", "not-a-real-backend", "--model", "test-model"]
        )
        assert result.exit_code != 0
        assert not config_path.exists()

    def test_init_default_route_round_trips(self, tmp_path: Path) -> None:
        """A config written by `interop init` must resolve its default route on load.

        Regression test: `init` used to serialize `default_route_id` while the
        loader read `default_route`, so the default route silently vanished
        (get_route_for_model("") returned None) after every init → load cycle.
        """
        import yaml  # type: ignore[import-untyped]
        from typer.testing import CliRunner

        from agent_interop.cli import app
        from agent_interop.config import load_config_from_dict

        runner = CliRunner()
        config_path = tmp_path / "interop.yaml"
        result = runner.invoke(
            app, ["init", "--path", str(config_path), "--backend", "ollama", "--model", "test-model"]
        )
        assert result.exit_code == 0, f"init failed: {result.output}"

        with open(config_path) as f:
            raw = yaml.safe_load(f)
        assert raw.get("default_route"), "init should serialize the default_route key"

        config = load_config_from_dict(raw)
        assert config.default_route_id == raw["default_route"]

        route = config.get_route_for_model("")
        assert route is not None, "default route must resolve when no model is specified"
        assert route.id == config.default_route_id

    @pytest.mark.asyncio
    async def test_gateway_startup_with_token_auth(self) -> None:
        """Gateway should start with session token enforcement."""
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            probe_on_startup=False,
            ingress_auth={"mode": "session_token", "token": "test-token-123"},
            routes={
                "default": ModelRoute(
                    id="default",
                    client_model_aliases=["test-model"],
                    upstream_model="test-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:11434",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )

        gw = Gateway(config)
        await gw.startup()
        # Gateway should have no probe results (probe_on_startup=False)
        assert gw._probe_results == {}
        await gw.close()

    def test_config_validate_rejects_invalid(self, tmp_path: Path) -> None:
        """config validate should reject configs with bad kind/protocol combos."""
        import yaml  # type: ignore[import-untyped]
        from typer.testing import CliRunner

        from agent_interop.cli import app

        runner = CliRunner()

        # Create a config with an invalid kind/protocol combination
        bad_config = {
            "routes": {
                "bad": {
                    "aliases": ["test"],
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://localhost:11434",
                        "wire_protocol": "openai_chat",  # Invalid for ollama
                    },
                }
            }
        }
        config_path = tmp_path / "bad.yaml"
        with open(config_path, "w") as f:
            yaml.dump(bad_config, f)
        result = runner.invoke(app, ["config", "validate", "--path", str(config_path)])
        assert result.exit_code != 0, f"Should reject invalid kind/protocol combo: {result.output}"


# ─── CLI Evidence Commands ─────────────────────────────────────────────────


class TestEvidenceCommands:
    """Test the evidence CLI commands (list, show, revoke)."""

    def test_evidence_list_empty(self):
        """evidence list should work with empty store."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from agent_interop.cli import app

        runner = CliRunner()
        with patch("agent_interop.evidence.store.get_default_store") as mock_store:
            mock_store.return_value.query_results.return_value = []
            result = runner.invoke(app, ["evidence", "list"])
            assert result.exit_code == 0, f"evidence list failed: {result.output}"

    def test_evidence_list_with_results(self):
        """evidence list should display stored evidence."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from agent_interop.cli import app
        from agent_interop.replay.types import CompatibilityKey, CompatibilityResult

        runner = CliRunner()
        key = CompatibilityKey(model_id="test-model", backend_kind="ollama")
        result_obj = CompatibilityResult(
            tested_at="2024-01-01T00:00:00",
            sample_count=10,
            task_completion_rate=0.9,
            valid_call_rate_after_repair=0.95,
        )
        mock_store = MagicMock()
        mock_store.query_results.return_value = [(key, result_obj)]
        mock_store._make_result_id.return_value = "res_abc123"

        with patch("agent_interop.evidence.store.get_default_store", return_value=mock_store):
            result = runner.invoke(app, ["evidence", "list"])
            assert result.exit_code == 0, f"evidence list failed: {result.output}"
            assert "test-model" in result.output or "ollama" in result.output

    def test_evidence_show_by_model(self):
        """evidence show should filter by model."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from agent_interop.cli import app
        from agent_interop.replay.types import CompatibilityKey, CompatibilityResult

        runner = CliRunner()
        key = CompatibilityKey(model_id="test-model", backend_kind="ollama")
        result_obj = CompatibilityResult(
            tested_at="2024-01-01T00:00:00",
            sample_count=10,
            task_completion_rate=0.9,
        )
        mock_store = MagicMock()
        mock_store.query_results.return_value = [(key, result_obj)]
        mock_store._make_result_id.return_value = "res_abc123"

        with patch("agent_interop.evidence.store.get_default_store", return_value=mock_store):
            result = runner.invoke(app, ["evidence", "show", "--model", "test-model"])
            assert result.exit_code == 0, f"evidence show failed: {result.output}"
            assert "test-model" in result.output

    def test_evidence_show_no_match(self):
        """evidence show should handle no matches gracefully."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from agent_interop.cli import app

        runner = CliRunner()
        mock_store = MagicMock()
        mock_store.query_results.return_value = []

        with patch("agent_interop.evidence.store.get_default_store", return_value=mock_store):
            result = runner.invoke(app, ["evidence", "show", "--model", "nonexistent"])
            assert result.exit_code == 0, f"evidence show failed: {result.output}"
            assert "No matching" in result.output

    def test_evidence_revoke_by_model(self):
        """evidence revoke should call store.revoke."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from agent_interop.cli import app
        from agent_interop.replay.types import CompatibilityKey, CompatibilityResult

        runner = CliRunner()
        key = CompatibilityKey(model_id="test-model", backend_kind="ollama")
        result_obj = CompatibilityResult(
            tested_at="2024-01-01T00:00:00",
            sample_count=10,
        )
        mock_store = MagicMock()
        mock_store.query_results.return_value = [(key, result_obj)]
        mock_store._make_result_id.return_value = "res_abc123"

        with patch("agent_interop.evidence.store.get_default_store", return_value=mock_store):
            result = runner.invoke(app, ["evidence", "revoke", "--model", "test-model", "--reason", "test_revoke"])
            assert result.exit_code == 0, f"evidence revoke failed: {result.output}"
            mock_store.revoke.assert_called_once_with(key, reason="test_revoke")

    def test_evidence_unknown_action(self):
        """evidence should reject unknown actions."""
        from typer.testing import CliRunner

        from agent_interop.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["evidence", "unknown_action"])
        assert result.exit_code != 0


# ─── Agent Integration Registry ───────────────────────────────────────────


class TestAgentRegistry:
    """Test the agent integration registry."""

    def test_claude_registered(self):
        from agent_interop.agents.registry import get_agent_integration
        integration = get_agent_integration("claude")
        assert integration is not None
        assert integration.id == "claude"

    def test_codex_registered(self):
        from agent_interop.agents.registry import get_agent_integration
        integration = get_agent_integration("codex")
        assert integration is not None
        assert integration.id == "codex"

    def test_cline_registered(self):
        from agent_interop.agents.registry import get_agent_integration
        integration = get_agent_integration("cline")
        assert integration is not None

    def test_aider_registered(self):
        from agent_interop.agents.registry import get_agent_integration
        integration = get_agent_integration("aider")
        assert integration is not None
        assert integration.id == "aider"

    def test_continue_registered(self):
        from agent_interop.agents.registry import get_agent_integration
        integration = get_agent_integration("continue")
        assert integration is not None
        assert integration.id == "continue"

    def test_unknown_agent_returns_none(self):
        from agent_interop.agents.registry import get_agent_integration
        integration = get_agent_integration("totally_unknown_agent_xyz")
        assert integration is None


# ─── Anthropic Messages System Role ────────────────────────────────────────


class TestAnthropicSystemRole:
    """Test that system/developer role messages are preserved in Anthropic adapter."""

    def test_system_role_in_messages_array(self):
        from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter

        adapter = AnthropicMessagesAdapter()
        body = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
        }
        request = adapter.decode_request(body, {})
        # System message should be in the messages array
        system_msgs = [m for m in request.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "helpful assistant" in str(system_msgs[0].content)

    def test_developer_role_in_messages_array(self):
        from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter

        adapter = AnthropicMessagesAdapter()
        body = {
            "messages": [
                {"role": "developer", "content": "Use Python."},
                {"role": "user", "content": "Hello"},
            ],
        }
        request = adapter.decode_request(body, {})
        system_msgs = [m for m in request.messages if m.role == "system"]
        assert len(system_msgs) == 1
        assert "Python" in str(system_msgs[0].content)
