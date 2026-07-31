"""Tool visibility planning contracts."""

from __future__ import annotations

from dataclasses import dataclass

from agent_interop.abi import CanonicalTool
from agent_interop.config import ToolSurfaceMode


@dataclass(frozen=True)
class ToolSurfacePlan:
    mode: ToolSurfaceMode
    visible_tools: tuple[CanonicalTool, ...]
    validation_tools: tuple[CanonicalTool, ...]
    withheld_tool_names: tuple[str, ...]
    original_schema_tokens: int
    visible_schema_tokens: int
    selector_id: str
    selector_revision: str
    selection_reason: str
    fingerprint: str

