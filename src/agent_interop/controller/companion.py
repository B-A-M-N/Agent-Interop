"""Controller route selection based on explicit configuration/evidence."""

from __future__ import annotations

from agent_interop.config import InteropServerConfig, ModelRoute


class ControllerRegistry:
    def candidates(self, config: InteropServerConfig, primary_route: ModelRoute) -> tuple[ModelRoute, ...]:
        """Return configured controller candidates in deterministic priority.

        Qualification remains asynchronous and is intentionally evaluated by
        the gateway.  Keeping this registry side-effect free makes it safe to
        ask for candidates during planning and diagnostics.
        """
        override = primary_route.controller.route_id if primary_route.controller else ""
        if not override:
            override = config.controller.route_id
        candidates: list[ModelRoute] = []
        explicit = config.routes.get(override) if override else None
        if explicit is not None and explicit.id != primary_route.id:
            candidates.append(explicit)
        for route_id in sorted(config.routes):
            route = config.routes[route_id]
            if route.id != primary_route.id and route not in candidates:
                candidates.append(route)
        return tuple(candidates)

    def select(self, config: InteropServerConfig, primary_route: ModelRoute) -> ModelRoute | None:
        """Compatibility helper for callers that only need explicit priority."""
        candidates = self.candidates(config, primary_route)
        return candidates[0] if candidates else None
