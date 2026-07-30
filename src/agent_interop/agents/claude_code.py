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
        # satisfies that on its own, so the model alias must always be a
        # claude-prefixed alias rather than context.model_name verbatim.
        # The gateway registers this same alias for the route (see
        # server/app.py:create_app_from_env).
        model_alias = f"claude-interop-{context.route}"
        env["CLAUDE_MODEL"] = model_alias

        # Confirmed via a real end-to-end run (not just reading the docs):
        # the installed Claude Code CLI does NOT pick up CLAUDE_MODEL — a
        # session launched with only the env var set fell back to the
        # operator's own persisted default model and got a 400 from the
        # gateway ("Unknown model: 'claude-opus-5'"). The `--model` CLI
        # flag is what the CLI actually honors, so it's passed explicitly
        # here; the env var is kept too in case a future/older CLI version
        # does read it, but the CLI flag is what makes this work today.
        #
        # If the caller's own extra_args already specifies --model, trust
        # their explicit choice instead of injecting ours — otherwise
        # argv would carry two --model flags, and depending on the CLI's
        # own parsing order that's either a silent last-wins surprise or
        # an outright parse error, neither of which is what "the user
        # explicitly supplied a model" should do.
        user_supplied_model = context.extra_args and "--model" in context.extra_args
        cmd = ["claude"] if user_supplied_model else ["claude", "--model", model_alias]
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