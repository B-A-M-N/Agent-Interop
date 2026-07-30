"""hermes-agent integration — provider configuration and launcher.

hermes-agent (https://github.com/NousResearch/hermes-agent — a coding/
research agent, distinct from the Hermes model family) wraps the stock
OpenAI Python SDK for OpenAI-wire providers, selected via a `provider:
"custom"` entry in `~/.hermes/config.yaml` pointing `base_url` at any
OpenAI-compatible endpoint. Config location is controlled by the
HERMES_HOME env var (confirmed by reading hermes-agent's own source —
`hermes_cli/config.py`, `hermes_cli/env_loader.py`, `hermes_cli/main.py`
all read it via `os.environ.get("HERMES_HOME", ...)`), so a session-scoped
config can be pointed at without ever touching the user's real one.

Unlike Claude Code, hermes-agent sends no distinguishing header/User-Agent
to a custom/local endpoint (it clears default_headers and falls through to
the generic OpenAI SDK's own UA) — confirmed by reading
`run_agent.py`'s client-construction path, not assumed. Its config schema
does support `model.default_headers`, applied on the OpenAI wire, which
this integration uses to send `X-Interop-Client: hermes_agent` — the
generic self-assertion header `RequestContext.from_headers` checks for
any client with no fingerprint of its own.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agent_interop.abi import ProtocolKind
from agent_interop.agents.base import (
    AgentInstallation,
    AgentIntegration,
    AgentLaunchContext,
    LaunchSpec,
)

HERMES_INTEROP_CLIENT_HEADER = "X-Interop-Client"
HERMES_INTEROP_CLIENT_ID = "hermes_agent"


class HermesAgentIntegration(AgentIntegration):
    """Integration for hermes-agent."""

    id = "hermes-agent"

    def discover(self) -> AgentInstallation:
        path = shutil.which("hermes")
        if path:
            return AgentInstallation(found=True, path=path)
        return AgentInstallation(found=False)

    def build_launch(
        self,
        context: AgentLaunchContext,
    ) -> LaunchSpec:
        temp_home = _generate_temp_hermes_home(
            gateway_url=context.gateway_url,
            model_name=context.model_name or context.route or "default",
            api_key=context.session_credential,
        )

        if temp_home is None:
            return LaunchSpec(
                command=None,
                env={},
                config_instructions=[
                    "Failed to generate temporary hermes-agent configuration.",
                    "Manual configuration required.",
                ],
                readiness="configuration_required",
                protocol=ProtocolKind.OPENAI_CHAT,
            )

        env = {"HERMES_HOME": str(temp_home)}

        cmd = ["hermes"]
        if context.extra_args:
            cmd.extend(context.extra_args)

        return LaunchSpec(
            command=cmd,
            env=env,
            config_instructions=[f"Temporary hermes-agent home at: {temp_home}"],
            readiness="ready",
            protocol=ProtocolKind.OPENAI_CHAT,
            cleanup=_make_cleanup_fn(str(temp_home)),
        )


def _make_cleanup_fn(temp_dir: str) -> Callable[[], None]:
    """Create a cleanup function that removes the temp HERMES_HOME dir."""
    def _cleanup() -> None:
        try:
            p = Path(temp_dir)
            if p.exists():
                shutil.rmtree(p)
        except Exception:
            pass
    return _cleanup


def _generate_temp_hermes_home(
    gateway_url: str,
    model_name: str,
    api_key: str,
) -> Path | None:
    """Generate a temporary, isolated HERMES_HOME with the real gateway URL.

    Writes ~/.hermes/config.yaml (per hermes-agent's own layout, at
    get_hermes_home() / "config.yaml") under a fresh temp directory rather
    than the user's real one, so nothing about their actual hermes-agent
    setup is touched. Returns the temp home directory path, or None on
    failure.
    """
    import yaml  # type: ignore[import-untyped]

    try:
        temp_home = Path(tempfile.mkdtemp(prefix="interop-hermes-"))
        config: dict[str, Any] = {
            "model": {
                "default": model_name,
                "provider": "custom",
                # hermes-agent wraps the stock OpenAI SDK, which appends
                # "/chat/completions" directly to base_url rather than
                # inserting "/v1" itself — the real gateway route is
                # /v1/chat/completions (see server/app.py), so the "/v1"
                # segment must be included here explicitly. Same convention
                # already used by GenericOpenAICompatibleIntegration for
                # OPENAI_BASE_URL. Confirmed live: without it, hermes-agent
                # posted to plain /chat/completions and got a 404.
                "base_url": f"{gateway_url.rstrip('/')}/v1",
                "api_key": api_key,
                "default_headers": {
                    HERMES_INTEROP_CLIENT_HEADER: HERMES_INTEROP_CLIENT_ID,
                },
            },
        }
        (temp_home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return temp_home
    except Exception:
        return None
