"""Codex integration — provider configuration and launcher.

Codex supports custom model providers via config.toml:
  https://github.com/openai/codex

Key requirements:
  - Custom provider ID (e.g., interop-local)
  - Responses wire protocol
  - env_key for authentication
  - Profile for model selection
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from agent_interop.abi import ProtocolKind
from agent_interop.agents.base import (
    AgentInstallation,
    AgentIntegration,
    AgentLaunchContext,
    LaunchSpec,
)

CODEX_PROVIDER_ID = "interop-local"


class CodexIntegration(AgentIntegration):
    """Integration for OpenAI Codex."""

    id = "codex"

    def discover(self) -> AgentInstallation:
        path = shutil.which("codex")
        if path:
            return AgentInstallation(found=True, path=path)
        # Check common locations
        for loc in [
            os.path.expanduser("~/.codex/codex"),
            "/usr/local/bin/codex",
        ]:
            if os.path.isfile(loc):
                return AgentInstallation(found=True, path=loc)
        return AgentInstallation(found=False)

    def build_launch(
        self,
        context: AgentLaunchContext,
    ) -> LaunchSpec:
        # Codex profile name
        profile_name = f"interop-{context.route}" if context.route else "interop-local"

        env: dict[str, str] = {
            "INTEROP_SESSION_TOKEN": context.session_credential,
        }

        # Generate a temporary Codex config with the real gateway URL
        config_lines: list[str] = []
        temp_config = _generate_temp_codex_config(
            gateway_url=context.gateway_url,
            profile_name=profile_name,
            model_name=context.model_name or context.route or "default",
        )

        if not temp_config:
            # Configuration failed - cannot launch
            return LaunchSpec(
                command=None,
                env=env,
                config_instructions=[
                    "Failed to generate temporary Codex configuration.",
                    "Manual configuration required.",
                ],
                readiness="configuration_required",
                protocol=ProtocolKind.OPENAI_RESPONSES,
            )

        env["CODEX_CONFIG_DIR"] = str(temp_config.parent)
        config_lines.append(f"Temporary config at: {temp_config}")

        # Build command
        cmd = ["codex", "-p", profile_name]
        if context.extra_args:
            cmd.extend(context.extra_args)

        # Build cleanup to remove temp config dir
        temp_dir = temp_config.parent.parent  # config_dir/.codex/config.toml → config_dir
        cleanup_fn = _make_cleanup_fn(str(temp_dir))

        return LaunchSpec(
            command=cmd,
            env=env,
            config_instructions=config_lines,
            readiness="ready",
            protocol=ProtocolKind.OPENAI_RESPONSES,
            cleanup=cleanup_fn,
        )


def _make_cleanup_fn(temp_dir: str) -> Callable[[], None]:
    """Create a cleanup function that removes the temp directory."""
    import shutil
    from pathlib import Path

    def _cleanup() -> None:
        try:
            p = Path(temp_dir)
            if p.exists():
                shutil.rmtree(p)
        except Exception:
            pass

    return _cleanup


def _toml_string(value: str) -> str:
    """Render ``value`` as a quoted TOML basic string literal.

    ``profile_name``/``model_name`` ultimately come from the ``--model``
    CLI argument (fully user-supplied, arbitrary text) and ``gateway_url``
    is caller-controlled too — interpolating them into TOML with an
    unescaped f-string lets a value containing a `"`, backslash, or
    newline break out of the string literal (or, for a table header,
    inject an arbitrary additional TOML table). Escaping here, and always
    quoting table-header segments through this same function, closes that
    off regardless of what the value contains.
    """
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _generate_temp_codex_config(
    gateway_url: str,
    profile_name: str,
    model_name: str,
) -> Path | None:
    """Generate a temporary isolated Codex config for this session.

    Creates a temporary directory and writes the provider/profile
    configuration with the actual gateway URL. Returns the config
    file path, or None on failure.
    """
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="interop-codex-"))
        config_dir = temp_dir / ".codex"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "config.toml"

        config_lines = [
            f"[model_providers.{CODEX_PROVIDER_ID}]",
            'name = "Interop Local Gateway"',
            f"base_url = {_toml_string(gateway_url.rstrip('/') + '/v1')}",
            'env_key = "INTEROP_SESSION_TOKEN"',
            'wire_api = "responses"',
            "",
            f"[profiles.{_toml_string(profile_name)}]",
            f'model_provider = "{CODEX_PROVIDER_ID}"',
            f"model = {_toml_string(model_name)}",
            "",
        ]
        config_path.write_text("\n".join(config_lines) + "\n")
        return config_path
    except Exception:
        return None