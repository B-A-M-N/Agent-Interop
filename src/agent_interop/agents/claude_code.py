"""Claude Code integration — launcher and gateway configuration.

Claude Code officially supports an LLM gateway exposing Anthropic Messages:
  https://code.claude.com/docs/en/llm-gateway

Key requirements:
  - /v1/messages endpoint with anthropic-version and anthropic-beta headers
  - Model IDs starting with "claude" or "anthropic" for gateway discovery
  - Session and agent attribution headers are preserved
"""

from __future__ import annotations

import shutil

from agent_interop.abi import ProtocolKind
from agent_interop.agents.base import (
    AgentInstallation,
    AgentIntegration,
    AgentLaunchContext,
    LaunchSpec,
)


class ClaudeCodeIntegration(AgentIntegration):
    """Integration for Claude Code (claude CLI)."""

    id = "claude"

    def discover(self) -> AgentInstallation:
        path = shutil.which("claude")
        if path:
            return AgentInstallation(found=True, path=path)
        return AgentInstallation(found=False)

    def build_launch(
        self,
        context: AgentLaunchContext,
    ) -> LaunchSpec:
        """Build launch spec to run Claude Code through Interop."""
        env = {
            "ANTHROPIC_BASE_URL": context.gateway_url.rstrip("/"),
            "ANTHROPIC_AUTH_TOKEN": context.session_credential,
            "ANTHROPIC_API_KEY": context.session_credential,
            "ANTHROPIC_TOOLS_ENABLED": "true",
        }

        # Claude Code requires model IDs starting with "claude" or
        # "anthropic" for gateway discovery (see module docstring) — the
        # raw upstream/route model name (e.g. "qwen3-coder") never
        # satisfies that on its own, so CLAUDE_MODEL must always be a
        # claude-prefixed alias rather than context.model_name verbatim.
        # The gateway registers this same alias for the route (see
        # server/app.py:create_app_from_env).
        env["CLAUDE_MODEL"] = f"claude-interop-{context.route}"

        # Build CLI command
        cmd = ["claude"]
        if context.extra_args:
            cmd.extend(context.extra_args)

        config_lines: list[str] = []

        if context.route:
            config_lines.append(f"Interop route: {context.route}")

        return LaunchSpec(
            command=cmd,
            env=env,
            protocol=ProtocolKind.ANTHROPIC_MESSAGES,
            config_instructions=config_lines,
        )