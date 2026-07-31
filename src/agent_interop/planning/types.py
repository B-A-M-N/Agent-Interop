"""Compatibility-planning data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_interop.abi import ToolChoiceMode
from agent_interop.config import ToolMode
from agent_interop.context_budget.types import ContextPlan
from agent_interop.enums import ProtocolKind
from agent_interop.tool_surface.types import ToolSurfacePlan


class CompatibilityPath(str, Enum):
    DIRECT = "direct"
    ADAPTED = "adapted"
    CONTROLLED = "controlled"
    UNAVAILABLE = "unavailable"


class AttemptKind(str, Enum):
    NATIVE_TOOLS = "native_tools"
    PROMPTED_TOOLS = "prompted_tools"
    CONSTRAINED_JSON = "constrained_json"
    FORCED_SELECTION = "forced_selection"
    CONTROLLER_MEDIATED = "controller_mediated"
    CHAT_ONLY = "chat_only"


@dataclass(frozen=True)
class RequestRequirements:
    client_id: str = ""
    client_version: str = ""
    client_protocol: ProtocolKind = ProtocolKind.ANTHROPIC_MESSAGES
    streaming_required: bool = False
    tools_present: bool = False
    tool_choice_mode: ToolChoiceMode = ToolChoiceMode.AUTO
    named_tool: str = ""
    automatic_selection_required: bool = False
    sequential_tool_use_required: bool = False
    parallel_tool_use_required: bool = False
    tool_result_continuation_required: bool = False
    reasoning_required: bool = False
    images_required: bool = False
    structured_output_required: bool = False
    tool_count: int = 0
    tool_schema_bytes: int = 0
    estimated_input_tokens: int = 0
    requested_output_tokens: int = 0


@dataclass(frozen=True)
class CompatibilityAttempt:
    kind: AttemptKind
    tool_mode: ToolMode
    parser_id: str | None = None
    contract_template_id: str | None = None
    constrained_output: bool = False
    use_controller: bool = False
    reason: str = ""


@dataclass(frozen=True)
class BehavioralCapabilities:
    """Conservative observed model behaviour, separate from wire support."""

    native_tools: bool = False
    prompted_tools: bool = False
    forced_selection: bool = False
    automatic_selection: bool = False
    sequential_tool_use: bool = False
    parallel_tool_use: bool = False
    tool_result_continuation: bool = False
    streaming: bool = False
    chat_only: bool = False
    sample_count: int = 0


@dataclass(frozen=True)
class CompatibilityPlan:
    path: CompatibilityPath
    requirements: RequestRequirements
    attempts: tuple[CompatibilityAttempt, ...]
    context_plan: ContextPlan
    tool_surface_plan: ToolSurfacePlan
    missing_capabilities: tuple[str, ...] = ()
    transformations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    planner_revision: str = "1"
