"""Evidence facets used by conservative compatibility planning.

They remain separate from the legacy aggregate ``CompatibilityResult`` so
transport acceptance, model behavior, and adaptation success cannot be
accidentally treated as interchangeable proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransportEvidence:
    accepted_parameters: frozenset[str] = field(default_factory=frozenset)
    rejected_parameters: frozenset[str] = field(default_factory=frozenset)
    streaming_valid: bool = False


@dataclass(frozen=True)
class BehavioralEvidence:
    forced_tool_rate: float = 0.0
    automatic_tool_rate: float = 0.0
    no_tool_compliance_rate: float = 0.0
    continuation_rate: float = 0.0
    sequential_rate: float = 0.0
    parallel_rate: float = 0.0
    task_completion_rate: float = 0.0
    sample_count: int = 0


@dataclass(frozen=True)
class AdaptationEvidence:
    direct_success_rate: float = 0.0
    prompted_success_rate: float = 0.0
    constrained_success_rate: float = 0.0
    controller_success_rate: float = 0.0
