"""Embeddable compatibility runtime.

This is the SDK-facing counterpart to the HTTP gateway.  It deliberately
delegates planning, inspection, qualification, and generation to one Gateway
instance so an embedded caller gets the same compatibility decision as a
proxy caller.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    ProtocolKind,
)
from agent_interop.config import InteropServerConfig
from agent_interop.context import RequestContext
from agent_interop.execution import InteropRequestExecution
from agent_interop.gateway import Gateway
from agent_interop.plugin.adapter import LocalModelAdapter
from agent_interop.qualification import BootstrapQualifier, QualificationRecord
from agent_interop.qualification.probes import SYNTHETIC_TOOL
from agent_interop.replay.runner import replay_all_policies
from agent_interop.replay.types import ReplayCase


@dataclass(frozen=True)
class RuntimePlanResult:
    """Normalized, inspection-safe result of planning one request."""

    compatibility_plan: Any
    invocation_plan: Any
    runtime_capabilities: Any
    compatibility_key: Any
    context_plan: Any
    tool_surface_plan: Any


class InteropRuntime(LocalModelAdapter):
    """Full embeddable Interop API.

    ``LocalModelAdapter`` remains a small backwards-compatible facade; new
    integrations should use this class when they need compatibility planning
    and diagnostics in-process.
    """

    async def start(self, config: InteropServerConfig | None = None, **kwargs: Any) -> None:
        await super().start(config, **kwargs)

    def _require_gateway(self) -> Gateway:
        if self.gateway is None or not self.is_running:
            raise RuntimeError("runtime not started; call start() first")
        return self.gateway

    @staticmethod
    def _context(context: RequestContext | None = None) -> RequestContext:
        return context or RequestContext(client_protocol=ProtocolKind.OPENAI_CHAT)

    async def inspect_model(self, model: str = "") -> Any:
        gateway = self._require_gateway()
        route = gateway._resolve_route(CanonicalRequest(
            model=CanonicalModelReference(requested_name=model),
        ))
        return await gateway._inspect_model_runtime(route)

    async def plan(
        self, request: CanonicalRequest, context: RequestContext | None = None,
    ) -> RuntimePlanResult:
        gateway = self._require_gateway()
        execution = InteropRequestExecution(context=self._context(context))
        invocation = await gateway._prepare_invocation_async(
            request, execution.context, streaming=request.generation.stream, execution=execution,
        )
        return RuntimePlanResult(
            compatibility_plan=invocation.compatibility_plan,
            invocation_plan=invocation.invocation_plan,
            runtime_capabilities=invocation.runtime_capabilities,
            compatibility_key=invocation.compatibility_key,
            context_plan=invocation.context_plan,
            tool_surface_plan=invocation.tool_surface_plan,
        )

    async def stream(
        self, request: CanonicalRequest, context: RequestContext | None = None,
    ) -> AsyncIterator[Any]:
        gateway = self._require_gateway()
        async for event in gateway.handle_stream(request, self._context(context)):
            yield event

    async def qualify(
        self, model: str = "", context: RequestContext | None = None,
    ) -> QualificationRecord:
        """Run only the bounded synthetic bootstrap battery.

        The probe tool is declarative and never executes locally.  The backend
        merely emits a request for it, so qualification cannot touch files,
        shell, or client-owned tools.
        """
        gateway = self._require_gateway()
        ctx = self._context(context)
        resolved_model = model or (self.config.default_route_id if self.config else "")
        runtime = await self.inspect_model(resolved_model)

        async def execute(probe: Any) -> bool:
            choice = (
                CanonicalToolChoice.required() if probe.requires_tools
                else CanonicalToolChoice.none()
            )
            request = CanonicalRequest(
                model=CanonicalModelReference(requested_name=resolved_model),
                messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text=probe.prompt)])],
                tools=[SYNTHETIC_TOOL] if probe.requires_tools else [],
                tool_choice=choice,
            )
            response = await gateway.handle_request(request, ctx)
            if response.error is not None:
                return False
            calls = [block for block in response.content if isinstance(block, CanonicalToolCallBlock)]
            if probe.name == "no_tool":
                return not calls
            if probe.requires_tools and probe.name != "tool_result_continuation":
                return bool(calls)
            return True

        digest = runtime.model_digest or runtime.model_name or resolved_model
        record = await BootstrapQualifier().qualify(digest, execute)
        gateway.record_qualification(record)
        return record

    async def explain(
        self, request: CanonicalRequest, context: RequestContext | None = None,
    ) -> RuntimePlanResult:
        """Alias for ``plan`` that emphasizes this method sends no inference."""
        return await self.plan(request, context)

    async def replay(self, case: ReplayCase) -> dict[str, Any]:
        results = await replay_all_policies(case)
        return {name: result for name, result in results.items()}
