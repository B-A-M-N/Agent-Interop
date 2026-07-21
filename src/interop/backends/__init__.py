"""Backend adapters for local inference servers."""

from interop.backends.base import BackendAdapter
from interop.backends.ollama import OllamaAdapter
from interop.backends.llamacpp import LlamacppAdapter
from interop.backends.vllm import VLLMAdapter

__all__ = ["BackendAdapter", "OllamaAdapter", "LlamacppAdapter", "VLLMAdapter"]