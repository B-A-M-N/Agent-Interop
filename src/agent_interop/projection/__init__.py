"""System prompt projection (P0.8) and model-projection aggregation (P1.6).

Projection owns system projection, history paging, tool-surface + private-capability
selection, withheld-tool index, and ModelView; the gateway only orchestrates ordering
(adapt before plan, project after plan) and owns pinning/unpinning of referenced_refs.
"""

from agent_interop.projection.planner import (
    ProjectionResult,
    ModelProjector,
    build_internal_tool_surface,
    rebuild_invocation_atomic,
)
from agent_interop.projection.system import (
    SystemProjectionResult,
    project_system,
)
from agent_interop.projection.types import (
    PrivateCapabilityPlan,
)

__all__ = [
    "PrivateCapabilityPlan",
    "ProjectionResult",
    "ModelProjector",
    "SystemProjectionResult",
    "build_internal_tool_surface",
    "project_system",
    "rebuild_invocation_atomic",
]
