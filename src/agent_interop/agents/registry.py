"""Agent integration registry — maps agent IDs to their integration classes."""

from __future__ import annotations

from agent_interop.agents.base import AgentDescriptor, AgentIntegration
from agent_interop.agents.claude_code import ClaudeCodeIntegration
from agent_interop.agents.codex import CodexIntegration
from agent_interop.agents.crush import CrushIntegration
from agent_interop.agents.hermes_agent import HermesAgentIntegration
from agent_interop.agents.manifests import load_external_integrations
from agent_interop.agents.openai_compatible import GenericOpenAICompatibleIntegration

_REGISTRY: dict[str, AgentIntegration] = {}
_CANONICAL: dict[str, AgentIntegration] = {}


def register(integration: AgentIntegration) -> None:
    descriptor = integration.descriptor
    canonical = descriptor.canonical_id or integration.id
    if canonical in _CANONICAL:
        raise ValueError(f"Agent integration '{canonical}' is already registered")
    aliases = {canonical, integration.id, *descriptor.aliases}
    collisions = [alias for alias in aliases if alias in _REGISTRY]
    if collisions:
        raise ValueError(f"Agent integration aliases already registered: {', '.join(sorted(collisions))}")
    _CANONICAL[canonical] = integration
    for alias in aliases:
        _REGISTRY[alias] = integration


def get_agent_integration(agent_id: str) -> AgentIntegration | None:
    return _REGISTRY.get(agent_id)


def list_agent_integrations() -> list[AgentIntegration]:
    """List canonical integrations once, regardless of aliases."""
    return list(_CANONICAL.values())


def get_agent_descriptor(agent_id: str) -> AgentDescriptor | None:
    integration = get_agent_integration(agent_id)
    return integration.descriptor if integration is not None else None


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


def register_external_manifests() -> list[AgentIntegration]:
    """Register project/user manifest integrations, rejecting alias collisions."""
    integrations = load_external_integrations()
    for integration in integrations:
        register(integration)
    return integrations


# Declarative external integrations are discovered with the registry.  A
# collision is intentionally an error: silently choosing one agent contract
# would make launcher behavior dependent on import order.
register_external_manifests()


__all__ = [
    "AgentIntegration",
    "get_agent_descriptor",
    "get_agent_integration",
    "list_agent_integrations",
    "register",
    "register_external_manifests",
]
