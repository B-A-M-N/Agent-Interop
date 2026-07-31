"""Runtime inspector registry."""

from __future__ import annotations

from agent_interop.backends.base import BackendInspector, ModelRuntimeCapabilities
from agent_interop.backends.llamacpp_inspector import LlamaCppInspector
from agent_interop.backends.ollama_inspector import OllamaInspector
from agent_interop.backends.openai_compatible_inspector import OpenAICompatibleInspector
from agent_interop.backends.vllm_inspector import VLLMInspector
from agent_interop.config import ModelRoute, UpstreamKind
from agent_interop.transport.http import UpstreamTransport


class StaticInspector:
    """Conservative fallback for backends without an inspector yet.

    This deliberately reports no model tool capability: a compatible wire
    protocol is not proof of behavioral model support.
    """

    async def inspect(self, route: ModelRoute, transport: UpstreamTransport) -> ModelRuntimeCapabilities:
        del transport
        return ModelRuntimeCapabilities(backend_kind=route.upstream.kind, model_name=route.upstream_model)


_INSPECTORS: dict[UpstreamKind, BackendInspector] = {
    UpstreamKind.OLLAMA: OllamaInspector(),
    UpstreamKind.VLLM: VLLMInspector(),
    UpstreamKind.LLAMACPP: LlamaCppInspector(),
    UpstreamKind.OPENAI_COMPATIBLE: OpenAICompatibleInspector(),
    UpstreamKind.OPENAI: OpenAICompatibleInspector(),
}
_FALLBACK = StaticInspector()


def get_backend_inspector(kind: UpstreamKind) -> BackendInspector:
    return _INSPECTORS.get(kind, _FALLBACK)


def register_backend_inspector(kind: UpstreamKind, inspector: BackendInspector) -> None:
    _INSPECTORS[kind] = inspector
