"""Controller route selection based on explicit configuration/evidence."""

from __future__ import annotations

from agent_interop.config import InteropServerConfig, ModelRoute


class ControllerRegistry:
    def select(self, config: InteropServerConfig, primary_route: ModelRoute) -> ModelRoute | None:
        override = primary_route.controller.route_id if primary_route.controller else ""
        if not override:
            override = config.controller.route_id
        return config.routes.get(override) if override else None
