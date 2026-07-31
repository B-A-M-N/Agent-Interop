"""Registry seam for future deterministic/semantic selectors."""

from __future__ import annotations

from agent_interop.tool_surface.selector import ToolSurfacePlanner

_SELECTORS = {"lexical": ToolSurfacePlanner}


def get_selector(name: str = "lexical") -> ToolSurfacePlanner:
    return _SELECTORS.get(name, ToolSurfacePlanner)()
