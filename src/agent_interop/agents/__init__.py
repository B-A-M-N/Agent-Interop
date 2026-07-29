"""Agent integration packages — optional launch wrappers for coding agents."""

from agent_interop.agents.base import AgentInstallation, AgentIntegration, LaunchSpec
from agent_interop.agents.registry import get_agent_integration, register

__all__ = [
    "AgentInstallation",
    "AgentIntegration",
    "LaunchSpec",
    "get_agent_integration",
    "register",
]