"""Per-client launch-contract tests for the named GenericOpenAI agents.

These tests exercise the *real* registered integrations (looked up via
``get_agent_integration``), not hand-constructed instances, so they lock in
the actual launch contract each client advertises:

    - Cline, OpenCode, Aider, Continue, Qwen Code all emit
      OPENAI_BASE_URL / OPENAI_API_KEY
    - Cline additionally emits CLINE_API_PROVIDER / CLINE_MODEL
    - OPENAI_MODEL / LLM_MODEL are gated on a non-empty model_name
    - CLINE_MODEL is emitted (as "" when model_name is empty) regardless
"""

from __future__ import annotations

import pytest

from agent_interop.abi import ProtocolKind
from agent_interop.agents.base import AgentInstallation, AgentLaunchContext
from agent_interop.agents.crush import CrushIntegration
from agent_interop.agents.registry import get_agent_integration

# The named clients backed by GenericOpenAICompatibleIntegration.
CLIENTS = ["cline", "opencode", "aider", "continue", "qwen-code"]

# The binary each client launches as. Usually equals the client ID, but
# qwen-code ships as the "qwen" executable.
CLIENT_BINARY = {
    "cline": "cline",
    "opencode": "opencode",
    "aider": "aider",
    "continue": "continue",
    "qwen-code": "qwen",
}


def _ctx(
    gateway_url: str = "http://127.0.0.1:8080",
    model_name: str = "test-model",
    extra_args: tuple[str, ...] = (),
) -> AgentLaunchContext:
    """Build a realistic launch context for the tests below."""
    return AgentLaunchContext(
        route="test-route",
        gateway_url=gateway_url,
        model_name=model_name,
        session_credential="test-cred-123",
        extra_args=extra_args,
    )


# ─── Shared launch contract (steps 1-2) ────────────────────────────────────


@pytest.mark.parametrize("client", CLIENTS)
def test_build_launch_env_and_protocol(client: str) -> None:
    """Every named client emits the core OpenAI env vars and chat protocol."""
    integration = get_agent_integration(client)
    assert integration is not None  # registered

    spec = integration.build_launch(_ctx())

    # Core env vars (URL join with no trailing slash on the base).
    assert spec.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert spec.env["OPENAI_API_KEY"] == "test-cred-123"

    # Model env vars are emitted when model_name is set.
    assert spec.env["OPENAI_MODEL"] == "test-model"
    assert spec.env["LLM_MODEL"] == "test-model"

    # Command is just the binary when there are no extra args.
    assert spec.command == [CLIENT_BINARY[client]]

    # Protocol is always OpenAI chat.
    assert spec.protocol == ProtocolKind.OPENAI_CHAT


@pytest.mark.parametrize("client", CLIENTS)
def test_build_launch_trailing_slash_produces_single_v1(client: str) -> None:
    """A trailing slash on gateway_url must not yield a double-slash ``//v1``."""
    integration = get_agent_integration(client)
    assert integration is not None

    spec = integration.build_launch(_ctx(gateway_url="http://127.0.0.1:8080/"))

    # Exactly one "/v1" suffix — never "//v1".
    assert spec.env["OPENAI_BASE_URL"] == "http://127.0.0.1:8080/v1"
    assert "//v1" not in spec.env["OPENAI_BASE_URL"]


@pytest.mark.parametrize("client", CLIENTS)
def test_build_launch_extra_args(client: str) -> None:
    """extra_args are appended to the binary in order."""
    integration = get_agent_integration(client)
    assert integration is not None

    extra = ("--flag", "value")
    spec = integration.build_launch(_ctx(extra_args=extra))

    assert spec.command == [CLIENT_BINARY[client], "--flag", "value"]


# ─── Cline-specific env (step 3) ───────────────────────────────────────────


def test_cline_has_provider_model_env_others_do_not() -> None:
    """Cline emits CLINE_API_PROVIDER / CLINE_MODEL; the other three do not."""
    cline = get_agent_integration("cline")
    assert cline is not None
    cline_spec = cline.build_launch(_ctx())

    assert cline_spec.env["CLINE_API_PROVIDER"] == "openai"
    assert cline_spec.env["CLINE_MODEL"] == "test-model"

    # These keys must NOT leak into the other named clients.
    for other in ("opencode", "aider", "continue"):
        integration = get_agent_integration(other)
        assert integration is not None
        spec = integration.build_launch(_ctx())
        assert "CLINE_API_PROVIDER" not in spec.env, f"{other} leaked CLINE_API_PROVIDER"
        assert "CLINE_MODEL" not in spec.env, f"{other} leaked CLINE_MODEL"


# ─── Empty model_name edge case (step 4) ────────────────────────────────────


@pytest.mark.parametrize("client", CLIENTS)
def test_empty_model_name_omits_openai_model_llm_model(client: str) -> None:
    """An empty model_name suppresses OPENAI_MODEL / LLM_MODEL for all clients."""
    integration = get_agent_integration(client)
    assert integration is not None

    spec = integration.build_launch(_ctx(model_name=""))

    assert "OPENAI_MODEL" not in spec.env
    assert "LLM_MODEL" not in spec.env


def test_empty_model_name_still_emits_cline_model_as_empty() -> None:
    """Cline's CLINE_MODEL is set unconditionally (via ``model_name or ""``).

    This locks in the real asymmetry: CLINE_MODEL is present (as "") even when
    model_name is empty, whereas OPENAI_MODEL/LLM_MODEL are gated and absent.
    """
    cline = get_agent_integration("cline")
    assert cline is not None

    spec = cline.build_launch(_ctx(model_name=""))

    assert "CLINE_MODEL" in spec.env
    assert spec.env["CLINE_MODEL"] == ""
    # Sanity: the gated keys are still absent for cline too.
    assert "OPENAI_MODEL" not in spec.env
    assert "LLM_MODEL" not in spec.env


# ─── discover() smoke test (step 5) ────────────────────────────────────────


def test_discover_returns_agent_installation() -> None:
    """discover() returns an AgentInstallation without raising in any env."""
    integration = get_agent_integration("opencode")
    assert integration is not None

    installed = integration.discover()

    assert isinstance(installed, AgentInstallation)
    # found must be a bool; path a str. We do NOT assert a specific found value
    # because it depends on whether the binary is on the test env's PATH.
    assert isinstance(installed.found, bool)
    assert isinstance(installed.path, str)


# ─── Claude Code (CLAUDE_MODEL must be gateway-discoverable) ───────────────


class TestClaudeCodeLaunchContract:
    """Claude Code only routes through a custom gateway when the model ID
    starts with "claude" or "anthropic" (see agents/claude_code.py's module
    docstring, quoting Claude Code's own gateway-discovery requirement).

    CLAUDE_MODEL must therefore always be a claude-prefixed alias derived
    from the route, never the raw upstream/route model name verbatim — a
    raw name like "qwen3-coder" doesn't satisfy Claude Code's own
    discovery check, silently breaking managed `interop run claude`
    launches.
    """

    def _ctx(self, route: str = "qwen3-coder", model_name: str = "qwen3-coder") -> AgentLaunchContext:
        return AgentLaunchContext(
            route=route,
            gateway_url="http://127.0.0.1:8090",
            model_name=model_name,
            session_credential="test-cred-123",
        )

    def test_claude_model_is_claude_prefixed_not_raw_model_name(self) -> None:
        integration = get_agent_integration("claude")
        assert integration is not None

        spec = integration.build_launch(self._ctx(route="qwen3-coder", model_name="qwen3-coder"))

        assert spec.env["CLAUDE_MODEL"].startswith(("claude", "anthropic"))
        assert spec.env["CLAUDE_MODEL"] != "qwen3-coder"

    def test_claude_model_derived_from_route(self) -> None:
        integration = get_agent_integration("claude")
        assert integration is not None

        spec = integration.build_launch(self._ctx(route="deepseek-local", model_name="deepseek-r1:70b"))

        assert spec.env["CLAUDE_MODEL"] == "claude-interop-deepseek-local"

    def test_model_passed_as_explicit_cli_flag_not_just_env_var(self) -> None:
        """Confirmed via a real end-to-end run: the installed Claude Code
        CLI does not honor CLAUDE_MODEL — a launch with only the env var
        set fell back to the operator's own persisted default model and
        got a 400 from the gateway for an unknown model. `--model` is the
        flag the CLI actually reads, so it must be in the command argv,
        not just the environment."""
        integration = get_agent_integration("claude")
        assert integration is not None

        spec = integration.build_launch(self._ctx(route="qwen3-coder", model_name="qwen3-coder"))

        assert spec.command is not None
        assert "--model" in spec.command
        idx = spec.command.index("--model")
        assert spec.command[idx + 1] == "claude-interop-qwen3-coder"
        assert spec.command[idx + 1] == spec.env["CLAUDE_MODEL"]

    def test_explicit_cli_model_flag_precedes_user_extra_args(self) -> None:
        """Our injected --model must appear before whatever the caller's
        own extra_args contribute, and those args must survive unchanged
        after it — argument placement AND preservation in one check."""
        integration = get_agent_integration("claude")
        assert integration is not None

        ctx = AgentLaunchContext(
            route="qwen3-coder",
            gateway_url="http://127.0.0.1:8090",
            model_name="qwen3-coder",
            session_credential="test-cred-123",
            extra_args=("--print", "hello"),
        )
        spec = integration.build_launch(ctx)

        assert spec.command is not None
        assert spec.command[:3] == ["claude", "--model", "claude-interop-qwen3-coder"]
        assert spec.command[3:] == ["--print", "hello"]

    def test_no_duplicate_model_flag_when_user_supplies_their_own(self) -> None:
        """If the caller's extra_args already names --model, Interop must
        NOT also inject its own — argv must never carry --model twice.
        The user's explicit choice wins outright (not merely 'appears
        later'), since a duplicate flag's precedence isn't guaranteed
        across every possible CLI parser."""
        integration = get_agent_integration("claude")
        assert integration is not None

        ctx = AgentLaunchContext(
            route="qwen3-coder",
            gateway_url="http://127.0.0.1:8090",
            model_name="qwen3-coder",
            session_credential="test-cred-123",
            extra_args=("--model", "opus", "--print", "hello"),
        )
        spec = integration.build_launch(ctx)

        assert spec.command is not None
        assert spec.command.count("--model") == 1
        assert spec.command == ["claude", "--model", "opus", "--print", "hello"]

    def test_env_var_still_set_even_when_user_overrides_via_cli(self) -> None:
        """CLAUDE_MODEL in the environment is harmless to keep set to
        Interop's own alias even when the user's CLI flag wins — nothing
        currently reads it, and if a future CLI version ever does, this
        preserves the intended fallback rather than leaving it unset."""
        integration = get_agent_integration("claude")
        assert integration is not None

        ctx = AgentLaunchContext(
            route="qwen3-coder",
            gateway_url="http://127.0.0.1:8090",
            model_name="qwen3-coder",
            session_credential="test-cred-123",
            extra_args=("--model", "opus"),
        )
        spec = integration.build_launch(ctx)

        assert spec.env["CLAUDE_MODEL"] == "claude-interop-qwen3-coder"


# ─── Crush (configuration_required) ─────────────────────────────────────────


class TestCrushLaunchContract:
    """Crush cannot be fully auto-launched (no base_url env override yet).

    It surfaces a configuration_required LaunchSpec with the exact
    crush.json provider block the user must add themselves.
    """

    def test_registration_and_discover(self) -> None:
        """get_agent_integration('crush') returns a CrushIntegration."""
        integration = get_agent_integration("crush")
        assert integration is not None
        assert isinstance(integration, CrushIntegration)

        # discover() must not raise regardless of test environment.
        installed = integration.discover()
        assert isinstance(installed, AgentInstallation)
        assert isinstance(installed.found, bool)
        assert isinstance(installed.path, str)

    def test_build_launch_is_configuration_required(self) -> None:
        """readiness == 'configuration_required' and no auto-launch."""
        integration = get_agent_integration("crush")
        assert integration is not None

        spec = integration.build_launch(_ctx())

        assert spec.readiness == "configuration_required"
        assert spec.command is None
        assert spec.protocol == ProtocolKind.OPENAI_CHAT

    def test_config_instructions_contain_base_url_no_trailing_slash(self) -> None:
        """Instructions contain the correct base_url with a single /v1 suffix."""
        integration = get_agent_integration("crush")
        assert integration is not None

        spec = integration.build_launch(_ctx(gateway_url="http://127.0.0.1:8080"))
        body = "\n".join(spec.config_instructions)

        assert spec.config_instructions
        assert '"base_url": "http://127.0.0.1:8080/v1"' in body
        assert "//v1" not in body

    def test_config_instructions_handle_trailing_slash(self) -> None:
        """A trailing slash on gateway_url must not yield a double-slash //v1."""
        integration = get_agent_integration("crush")
        assert integration is not None

        spec = integration.build_launch(_ctx(gateway_url="http://127.0.0.1:8080/"))
        body = "\n".join(spec.config_instructions)

        assert '"base_url": "http://127.0.0.1:8080/v1"' in body
        assert "//v1" not in body

    def test_config_instructions_mention_session_credential(self) -> None:
        """Instructions mention the actual session_credential value to export."""
        integration = get_agent_integration("crush")
        assert integration is not None

        spec = integration.build_launch(_ctx())
        body = "\n".join(spec.config_instructions)

        # The concrete credential the user should export must appear.
        assert "test-cred-123" in body
        assert "OPENAI_API_KEY" in body


# ─── ManagedGateway orchestration: configuration_required must not kill the
#     gateway it just printed credentials for ──────────────────────────────


class TestManagedGatewayConfigurationRequiredFlow:
    """Regression coverage for a real bug: launch_agent() used to
    ``raise SystemExit(0)`` right after printing the crush.json block and
    session credential. Since ManagedGateway.cleanup() is registered via
    atexit as soon as the gateway subprocess starts, that SystemExit tore
    the gateway down the instant the CLI process exited — the credential
    and base_url the user was told to use were already dead by the time
    they could act on them. launch_agent() must return None (not raise)
    for a configuration_required integration, and wait() must block on the
    gateway process (not return instantly) when there's no agent process
    to wait on, so the gateway survives until the user is done and hits
    Ctrl+C themselves.
    """

    def test_launch_agent_returns_none_instead_of_exiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agent_interop.agents.crush import CrushIntegration
        from agent_interop.launcher import ManagedGateway

        monkeypatch.setattr(
            CrushIntegration, "discover", lambda self: AgentInstallation(found=True, path="/usr/bin/crush")
        )

        gw = ManagedGateway(model="test-model", port=0)
        gw._gateway_url = "http://127.0.0.1:8090"
        gw._session_credential = "test-cred-xyz"

        result = gw.launch_agent("crush")

        assert result is None
        assert gw._agent_process is None

    def test_wait_blocks_on_gateway_process_when_no_agent_process(self) -> None:
        from agent_interop.launcher import ManagedGateway

        gw = ManagedGateway(model="test-model", port=0)
        assert gw._agent_process is None

        class _FakeProcess:
            def __init__(self) -> None:
                self.waited = False

            def wait(self) -> int:
                self.waited = True
                return 0

        fake_gateway_process = _FakeProcess()
        gw._gateway_process = fake_gateway_process  # type: ignore[assignment]

        exit_code = gw.wait()

        assert exit_code == 0
        assert fake_gateway_process.waited is True

    def test_terminate_process_tree_kills_grandchild(self, tmp_path) -> None:
        """The real regression this guards against: an agent process that
        spawns its own child (a shell, an MCP server, a pty helper) used
        to leave that child running as an orphan after cleanup(), because
        proc.terminate() only ever reaches the direct child. Launching
        with start_new_session=True + killpg on cleanup must reach the
        whole process group.
        """
        import os
        import signal
        import subprocess
        import time
        from pathlib import Path

        from agent_interop.launcher import ManagedGateway

        pid_file = tmp_path / "child.pid"
        proc = subprocess.Popen(
            ["sh", "-c", f"sleep 60 & echo $! > {pid_file}; wait"],
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert pid_file.exists(), "grandchild never wrote its pid"
            child_pid = int(pid_file.read_text().strip())

            os.kill(child_pid, 0)  # sanity: grandchild alive, no exception

            ManagedGateway._terminate_process_tree(proc)

            assert proc.poll() is not None, "parent shell was not reaped"

            # A "vanished" check right after killpg races: the grandchild
            # gets reparented to init/a subreaper the instant the shell
            # (its original parent) exits, and reaping by its NEW parent
            # is an independent, asynchronously-scheduled kernel event —
            # it can lag behind SIGTERM delivery by a nontrivial amount.
            # Until reaped, the PID is a zombie: os.kill(pid, 0) still
            # succeeds (no ProcessLookupError) even though the process is
            # functionally dead. Poll with a bounded wait, and accept
            # either "gone" or "zombie" as proof killpg reached it —
            # reaping an orphan once it's been reparented away from our
            # direct child is not something Interop controls or promises.
            def _grandchild_terminated() -> bool:
                try:
                    os.kill(child_pid, 0)
                except ProcessLookupError:
                    return True
                try:
                    state = Path(f"/proc/{child_pid}/stat").read_text()
                    # Format: "pid (comm) STATE ...". comm may contain
                    # spaces/parens, so split on the LAST ')'.
                    after_comm = state.rsplit(")", 1)[-1].split()
                    return bool(after_comm) and after_comm[0] == "Z"
                except (OSError, IndexError):
                    # /proc unavailable or process vanished mid-read —
                    # either way it's not a live, non-zombie process.
                    return True

            deadline = time.monotonic() + 5
            while not _grandchild_terminated() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert _grandchild_terminated(), (
                f"grandchild pid {child_pid} still running (not gone, not a zombie) "
                "after killpg + bounded wait"
            )
        finally:
            if proc.poll() is None:
                proc.kill()
            try:
                if pid_file.exists():
                    os.kill(int(pid_file.read_text().strip()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


# ─── Codex (TOML injection via unescaped f-string interpolation) ──────────


class TestCodexTomlConfigGeneration:
    """profile_name and model_name are ultimately derived from the
    ``--model`` CLI argument — fully user-supplied, arbitrary text — and
    used to be interpolated directly into TOML string literals and a
    table header with an unescaped f-string. A value containing a `"`,
    backslash, or newline could break out of the string literal or (as a
    table-header segment) inject an entirely separate TOML table.
    """

    def test_normal_config_round_trips(self) -> None:
        import tomllib

        from agent_interop.agents.codex import _generate_temp_codex_config

        path = _generate_temp_codex_config(
            gateway_url="http://127.0.0.1:8090",
            profile_name="interop-qwen3-coder",
            model_name="qwen3-coder:latest",
        )
        try:
            assert path is not None
            data = tomllib.loads(path.read_text())
            assert data["profiles"]["interop-qwen3-coder"]["model"] == "qwen3-coder:latest"
            assert data["model_providers"]["interop-local"]["base_url"] == "http://127.0.0.1:8090/v1"
        finally:
            if path is not None:
                import shutil
                shutil.rmtree(path.parent.parent, ignore_errors=True)

    def test_quote_and_newline_in_model_name_cannot_inject_a_table(self) -> None:
        import tomllib

        from agent_interop.agents.codex import _generate_temp_codex_config

        evil_model = 'x"\n[profiles.evil]\nmodel_provider = "other"\n'
        path = _generate_temp_codex_config(
            gateway_url="http://127.0.0.1:8090", profile_name="interop-test", model_name=evil_model,
        )
        try:
            assert path is not None
            data = tomllib.loads(path.read_text())  # must not raise TOMLDecodeError
            assert data["profiles"]["interop-test"]["model"] == evil_model
            assert "evil" not in data["profiles"]
        finally:
            if path is not None:
                import shutil
                shutil.rmtree(path.parent.parent, ignore_errors=True)

    def test_quote_and_newline_in_profile_name_cannot_inject_a_table(self) -> None:
        import tomllib

        from agent_interop.agents.codex import _generate_temp_codex_config

        evil_profile = 'x"\n[profiles.evil2]\nmodel_provider = "other"\n'
        path = _generate_temp_codex_config(
            gateway_url="http://127.0.0.1:8090", profile_name=evil_profile, model_name="model-x",
        )
        try:
            assert path is not None
            data = tomllib.loads(path.read_text())  # must not raise TOMLDecodeError
            assert evil_profile in data["profiles"]
            assert "evil2" not in data["profiles"]
        finally:
            if path is not None:
                import shutil
                shutil.rmtree(path.parent.parent, ignore_errors=True)
