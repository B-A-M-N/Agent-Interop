"""Agent integration registry — maps agent IDs to their integration classes."""

from __future__ import annotations

from agent_interop.agents.base import AgentIntegration
from agent_interop.agents.claude_code import ClaudeCodeIntegration
from agent_interop.agents.codex import CodexIntegration
from agent_interop.agents.crush import CrushIntegration
from agent_interop.agents.hermes_agent import HermesAgentIntegration
from agent_interop.agents.openai_compatible import GenericOpenAICompatibleIntegration

_REGISTRY: dict[str, AgentIntegration] = {}


def register(integration: AgentIntegration) -> None:
    _REGISTRY[integration.id] = integration


def get_agent_integration(agent_id: str) -> AgentIntegration | None:
    return _REGISTRY.get(agent_id)


# Register all built-in integrations
register(ClaudeCodeIntegration())
register(CodexIntegration())
register(GenericOpenAICompatibleIntegration("cline", "cline"))
register(GenericOpenAICompatibleIntegration("opencode", "opencode"))
register(GenericOpenAICompatibleIntegration("aider", "aider"))
register(GenericOpenAICompatibleIntegration("continue", "continue"))
register(GenericOpenAICompatibleIntegration("qwen-code", "qwen"))
register(CrushIntegration())
register(HermesAgentIntegration())


__all__ = [
    "AgentIntegration",
    "get_agent_integration",
    "register",
]