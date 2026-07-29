"""Crush integration — launcher and gateway configuration.

Crush (charmbracelet/crush) supports OpenAI- and Anthropic-compatible
providers via its ``crush.json`` config file, but (as of this writing)
has no environment-variable override for a provider's ``base_url`` — only
``OPENAI_API_KEY``/``ANTHROPIC_API_KEY`` are read directly from the
environment for auth (see https://github.com/charmbracelet/crush/issues/2017,
an open, unimplemented feature request for ``ANTHROPIC_BASE_URL`` support).

So this integration cannot fully automate a launch the way the generic
OpenAI-compatible integrations do — it surfaces the exact ``crush.json``
provider block the user needs to add, rather than writing config files
to disk on the user's behalf. Crush's own README explicitly warns that
``crush.json`` is trusted, shell-expanded configuration that should be
reviewed before use, so auto-writing it would be inappropriate here.
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


class CrushIntegration(AgentIntegration):
    """Integration for Crush (crush CLI)."""

    id = "crush"

    def discover(self) -> AgentInstallation:
        path = shutil.which("crush")
        if path:
            return AgentInstallation(found=True, path=path)
        return AgentInstallation(found=False)

    def build_launch(
        self,
        context: AgentLaunchContext,
    ) -> LaunchSpec:
        """Surface the crush.json provider block Crush needs.

        No env-var override for ``base_url`` exists upstream yet, so this
        cannot be a fully automated launch — it returns config instructions
        instead.
        """
        base_url = f"{context.gateway_url.rstrip('/')}/v1"

        instructions = [
            "Crush has no environment-variable override for a provider's",
            "base_url yet (see charmbracelet/crush#2017), so add this",
            "provider block to your crush.json yourself:",
            "",
            "{",
            '  "providers": {',
            '    "interop": {',
            '      "type": "openai-compat",',
            f'      "base_url": "{base_url}",',
            '      "api_key": "$OPENAI_API_KEY"',
            "    }",
            "  }",
            "}",
            "",
            f"export OPENAI_API_KEY={context.session_credential}",
            'Then select the "interop" provider in Crush.',
        ]

        return LaunchSpec(
            command=None,
            env={"OPENAI_API_KEY": context.session_credential},
            config_instructions=instructions,
            readiness="configuration_required",
            protocol=ProtocolKind.OPENAI_CHAT,
        )
