"""Generic agent integration interface.

Each agent defines:
- How to discover if it's installed
- How to build a launch command (env vars, args)
- How to verify the gateway is compatible

The gateway itself does NOT contain agent-specific launch logic.
Agent integrations are optional extras.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from agent_interop.abi import ProtocolKind


@dataclass(frozen=True)
class AgentLaunchContext:
    """Typed context for building an agent launch.

    Prevents adding more positional parameters every time agent
    configuration grows.
    """

    route: str
    gateway_url: str
    model_name: str
    session_credential: str
    extra_args: tuple[str, ...] = ()


@dataclass
class LaunchSpec:
    """Specification for launching an agent with Interop."""

    command: list[str] | None = None
    env: dict[str, str] = field(default_factory=dict)
    config_instructions: list[str] = field(default_factory=list)
    readiness: Literal["ready", "configuration_required", "unsupported"] = "ready"
    protocol: ProtocolKind | None = None
    cleanup: Callable[[], None] | None = None


@dataclass
class AgentInstallation:
    """Information about an installed agent."""

    found: bool = False
    path: str = ""
    version: str | None = None


class AgentIntegration(ABC):
    """Abstract interface for a coding-agent integration."""

    id: str

    @abstractmethod
    def discover(self) -> AgentInstallation:
        """Check if this agent is installed and find its path."""
        ...

    @abstractmethod
    def build_launch(
        self,
        context: AgentLaunchContext,
    ) -> LaunchSpec:
        """Build the launch spec for this agent.

        Args:
            context: Typed context with route, gateway_url, model_name,
                session_credential, and extra_args.

        Returns:
            A LaunchSpec with command, env, and config instructions.
        """
        ...