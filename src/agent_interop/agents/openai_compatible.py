"""Generic OpenAI-compatible agent integration.

Any agent that supports the standard OpenAI Chat Completions or Responses
API can be integrated via environment variables alone — no config files
needed.
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


class GenericOpenAICompatibleIntegration(AgentIntegration):
    """Integration for any agent that supports OpenAI-compatible endpoints.

    Works with: Cline, OpenCode, Continue.dev, and any agent that accepts
    OPENAI_BASE_URL / OPENAI_API_KEY env vars.
    """

    id = "opencode"

    def __init__(self, agent_id: str = "opencode", binary: str = "opencode") -> None:
        self.id = agent_id
        self._binary = binary

    def discover(self) -> AgentInstallation:
        path = shutil.which(self._binary)
        if path:
            return AgentInstallation(found=True, path=path)
        return AgentInstallation(found=False)

    def build_launch(
        self,
        context: AgentLaunchContext,
    ) -> LaunchSpec:
        """Build launch spec using OpenAI-compatible environment variables."""
        env = {
            "OPENAI_BASE_URL": f"{context.gateway_url.rstrip('/')}/v1",
            "OPENAI_API_KEY": context.session_credential,
        }

        # Some agents use different env var names
        if self._binary == "cline":
            env["CLINE_API_PROVIDER"] = "openai"
            env["CLINE_MODEL"] = context.model_name or ""

        if context.model_name:
            env["OPENAI_MODEL"] = context.model_name
            env["LLM_MODEL"] = context.model_name

        cmd = [self._binary]
        if context.extra_args:
            cmd.extend(context.extra_args)

        return LaunchSpec(
            command=cmd,
            env=env,
            protocol=ProtocolKind.OPENAI_CHAT,
        )