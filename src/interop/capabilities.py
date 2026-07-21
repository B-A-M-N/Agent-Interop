"""Capability model — tool levels (T0-T5) and agent levels (A0-A5) with states.

Every capability has an explicit state track:
    unsupported → declared → probed → verified or degraded → user-forced

Interop must distinguish metadata claims from verified results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─── Capability States ──────────────────────────────────────────────────────


class CapabilityState(str, Enum):
    """The verification state of a capability.

    States progress left-to-right:
    unsupported → declared → probed → verified or degraded → user-forced
    """
    UNSUPPORTED = "unsupported"
    DECLARED = "declared"           # Model metadata claims it
    PROBED = "probed"               # Backend inspection confirms
    VERIFIED = "verified"           # Conformance test passed
    DEGRADED = "degraded"           # Available with limitations
    USER_FORCED = "user-forced"     # User explicitly enabled despite checks

    def is_available(self) -> bool:
        """True if this capability can be used."""
        return self in (CapabilityState.VERIFIED, CapabilityState.PROBED,
                        CapabilityState.DECLARED, CapabilityState.USER_FORCED)

    def is_verified(self) -> bool:
        return self == CapabilityState.VERIFIED


# ─── Tool Capability Levels ─────────────────────────────────────────────────


class ToolCapabilityLevel(str, Enum):
    """Tool-use capability: T0 (none) through T5 (reliable parallel + recovery)."""
    T0 = "T0"   # No tool support
    T1 = "T1"   # Named/forced single-tool calls only
    T2 = "T2"   # Automatic single-tool selection
    T3 = "T3"   # Sequential multi-turn tool use
    T4 = "T4"   # Multiple tool calls in one response
    T5 = "T5"   # Reliable parallel and sequential tool use with error recovery

    _ORDER = ["T0", "T1", "T2", "T3", "T4", "T5"]

    def __lt__(self, other: ToolCapabilityLevel) -> bool:
        return self._ORDER.index(self.value) < self._ORDER.index(other.value)

    def __le__(self, other: ToolCapabilityLevel) -> bool:
        return self == other or self < other

    def __gt__(self, other: ToolCapabilityLevel) -> bool:
        return self._ORDER.index(self.value) > self._ORDER.index(other.value)

    def __ge__(self, other: ToolCapabilityLevel) -> bool:
        return self == other or self > other


# ─── Agent Capability Levels ────────────────────────────────────────────────


class AgentCapabilityLevel(str, Enum):
    """Coding-agent capability: A0 (chat only) through A5 (full conformance)."""
    A0 = "A0"   # Chat only
    A1 = "A1"   # Can explain code but cannot safely operate tools
    A2 = "A2"   # Can perform one forced repository operation
    A3 = "A3"   # Can complete short read-edit-verify tasks
    A4 = "A4"   # Can complete multi-step tasks with recoverable failures
    A5 = "A5"   # Passes the complete Interop coding-agent conformance suite

    _ORDER = ["A0", "A1", "A2", "A3", "A4", "A5"]

    def __lt__(self, other: AgentCapabilityLevel) -> bool:
        return self._ORDER.index(self.value) < self._ORDER.index(other.value)

    def __le__(self, other: AgentCapabilityLevel) -> bool:
        return self == other or self < other

    def __gt__(self, other: AgentCapabilityLevel) -> bool:
        return self._ORDER.index(self.value) > self._ORDER.index(other.value)

    def __ge__(self, other: AgentCapabilityLevel) -> bool:
        return self == other or self > other


# ─── Minimum requirements ────────────────────────────────────────────────────

MINIMUM_AGENT_LEVEL = AgentCapabilityLevel.A3
MINIMUM_TOOL_LEVEL = ToolCapabilityLevel.T3


# ─── Capability Map ─────────────────────────────────────────────────────────


@dataclass
class CapabilityEntry:
    """A single capability with its current state."""

    name: str
    state: CapabilityState = CapabilityState.UNSUPPORTED
    details: str = ""

    @property
    def available(self) -> bool:
        return self.state.is_available()


@dataclass
class EffectiveCapabilities:
    """The resolved capability set for a model/backend pair."""

    tool_level: ToolCapabilityLevel = ToolCapabilityLevel.T0
    agent_level: AgentCapabilityLevel = AgentCapabilityLevel.A0

    # Individual capability states
    text_streaming: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("text_streaming"))
    automatic_tools: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("automatic_tools"))
    forced_tools: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("forced_tools"))
    sequential_tools: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("sequential_tools"))
    parallel_tools: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("parallel_tools"))
    tool_error_recovery: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("tool_error_recovery"))
    structured_arguments: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("structured_arguments"))
    reasoning: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("reasoning"))
    image_input: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("image_input"))
    tool_result_continuation: CapabilityEntry = field(default_factory=lambda: CapabilityEntry("tool_result_continuation"))

    def meets_minimum(self) -> bool:
        return (self.tool_level >= MINIMUM_TOOL_LEVEL and
                self.agent_level >= MINIMUM_AGENT_LEVEL)

    def as_dict(self) -> dict[str, dict[str, str]]:
        return {
            "tool_level": {"value": self.tool_level.value},
            "agent_level": {"value": self.agent_level.value},
            "capabilities": {
                entry.name: {"state": entry.state.value, "details": entry.details}
                for entry in [
                    self.text_streaming, self.automatic_tools, self.forced_tools,
                    self.sequential_tools, self.parallel_tools, self.tool_error_recovery,
                    self.structured_arguments, self.reasoning, self.image_input,
                    self.tool_result_continuation,
                ]
            },
        }


# ─── Compatibility Decision ─────────────────────────────────────────────────


@dataclass
class CompatibilityDecision:
    """Result of capability negotiation at session startup."""

    status: str  # compatible | degraded | incompatible
    capabilities: EffectiveCapabilities = field(default_factory=EffectiveCapabilities)
    disabled_features: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing_capabilities: list[str] = field(default_factory=list)
    remediation: list[str] = field(default_factory=list)


def compute_compatibility(
    effective: EffectiveCapabilities,
    minimum_agent: AgentCapabilityLevel = MINIMUM_AGENT_LEVEL,
    minimum_tool: ToolCapabilityLevel = MINIMUM_TOOL_LEVEL,
) -> CompatibilityDecision:
    """Compare effective capabilities against minimum requirements."""
    decision = CompatibilityDecision(capabilities=effective)

    if effective.meets_minimum():
        decision.status = "compatible"
        return decision

    # Check what's missing
    if effective.tool_level < minimum_tool:
        decision.missing_capabilities.append(f"tool_level >= {minimum_tool.value}")
        decision.remediation.append(
            f"Current tool level {effective.tool_level.value} below minimum {minimum_tool.value}"
        )

    if effective.agent_level < minimum_agent:
        decision.missing_capabilities.append(f"agent_level >= {minimum_agent.value}")
        decision.remediation.append(f"Select A{minimum_agent.value}+ model or enable degraded mode")

    # Check degraded features
    warnings = []
    if effective.parallel_tools.state == CapabilityState.DEGRADED:
        warnings.append("Parallel tools degraded — falling back to sequential")
    if effective.reasoning.state == CapabilityState.DEGRADED:
        warnings.append("Reasoning degraded — suppressing exposed reasoning")

    if warnings:
        decision.warnings = warnings
        if not decision.missing_capabilities:
            decision.status = "degraded"
        else:
            decision.status = "incompatible"
    elif decision.missing_capabilities:
        decision.status = "incompatible"

    return decision