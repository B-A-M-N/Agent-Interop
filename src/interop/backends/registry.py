"""Backend adapter registry."""

from __future__ import annotations

from interop.backends.ollama import OllamaAdapter
from interop.backends.llamacpp import LlamacppAdapter
from interop.backends.vllm import VLLMAdapter
from interop.types import BackendKind


_REGISTRY: dict[BackendKind, object] = {}
_CLS_MAP: dict[BackendKind, type] = {
    BackendKind.OLLAMA: OllamaAdapter,
    BackendKind.LLAMACPP: LlamacppAdapter,
    BackendKind.VLLM: VLLMAdapter,
}


def get_backend(kind: BackendKind):
    """Get a cached backend adapter instance."""
    if kind not in _REGISTRY:
        cls = _CLS_MAP.get(kind)
        if cls is None:
            raise ValueError(f"Unknown backend kind: {kind}")
        _REGISTRY[kind] = cls()
    return _REGISTRY[kind]