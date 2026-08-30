"""Model-view contract (P0.1).

This module defines ``ModelView`` — the bounded, model-specific projection of
the authoritative client state that the inference model actually consumes.

Why this exists
----------------
The local inference model must NOT own the entire client context. Interop keeps
the *authoritative* request (complete system instructions, full client tool
registry, every tool result, all history, actual client tool choice,
provenance) and exposes only a *model view* to the model. Rendering always uses
``ResolvedInvocation.model_request`` (the bounded view); validation,
reconciliation, and client-response assembly continue to use
``ResolvedInvocation.authoritative_request``. This separation is what prevents
context reduction from silently becoming state destruction.

``ModelView`` is the declarative description of what projection was applied to
produce ``model_request`` from ``authoritative_request``. Later P0 steps
(P0.3 result virtualization, P0.7 paging, P0.8 system projection) populate it
with concrete figures; for P0.1 it records the tool-surface narrowing that the
send paths already enforce and marks the view as bounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelView:
    """Declarative description of the bounded model view.

    P0.40: Enhanced with actual transformation tracking and context limits.
    """

    authorized_tool_count: int = 0
    visible_tool_count: int = 0
    visible_tool_names: tuple[str, ...] = field(default_factory=tuple)
    system_projected: bool = False
    results_virtualized: bool = False
    history_paged: bool = False
    max_context_tokens: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)
    # P0.40: New fields for accurate telemetry
    visible_client_tool_names: tuple[str, ...] = field(default_factory=tuple)
    visible_private_tool_names: tuple[str, ...] = field(default_factory=tuple)
    virtualized_refs_count: int = 0
    history_ref_count: int = 0
    effective_context_limit: int = 0
    safe_context_limit: int = 0

    @property
    def tool_reduction_ratio(self) -> float:
        """Fraction of the authorized tool set that is hidden from the model.

        0.0 means no reduction; 1.0 means every authorized tool is hidden.
        """
        if self.authorized_tool_count <= 0:
            return 0.0
        hidden = self.authorized_tool_count - self.visible_tool_count
        return hidden / self.authorized_tool_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "authorized_tool_count": self.authorized_tool_count,
            "visible_tool_count": self.visible_tool_count,
            "visible_tool_names": list(self.visible_tool_names),
            "visible_client_tool_names": list(self.visible_client_tool_names),
            "visible_private_tool_names": list(self.visible_private_tool_names),
            "system_projected": self.system_projected,
            "results_virtualized": self.results_virtualized,
            "history_paged": self.history_paged,
            "virtualized_refs_count": self.virtualized_refs_count,
            "history_ref_count": self.history_ref_count,
            "max_context_tokens": self.max_context_tokens,
            "effective_context_limit": self.effective_context_limit,
            "safe_context_limit": self.safe_context_limit,
            "tool_reduction_ratio": self.tool_reduction_ratio,
            "notes": list(self.notes),
        }
