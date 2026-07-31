"""Runtime facts reported by an inference backend.

Codec capability describes a wire format; it deliberately does *not* imply
that the selected model can use that format.  Inspectors collect the latter
facts and retain their verification state independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_interop.capabilities import CapabilityState
from agent_interop.config import ModelRoute, UpstreamKind
from agent_interop.transport.http import UpstreamTransport


@dataclass(frozen=True)
class ModelRuntimeCapabilities:
    """Observed or declared runtime capabilities for one served model."""

    backend_kind: UpstreamKind
    backend_version: str = ""
    model_name: str = ""
    model_digest: str = ""
    architecture: str = ""
    quantization: str = ""
    parameter_count: str = ""
    architecture_context_tokens: int = 0
    configured_context_tokens: int = 0
    effective_context_tokens: int = 0
    chat_template: str = ""
    chat_template_digest: str = ""
    accepts_native_tools: CapabilityState = CapabilityState.UNSUPPORTED
    returns_native_tool_calls: CapabilityState = CapabilityState.UNSUPPORTED
    accepts_named_tool_choice: CapabilityState = CapabilityState.UNSUPPORTED
    accepts_required_tool_choice: CapabilityState = CapabilityState.UNSUPPORTED
    accepts_parallel_tool_flag: CapabilityState = CapabilityState.UNSUPPORTED
    supports_json_schema: CapabilityState = CapabilityState.UNSUPPORTED
    supports_json_mode: CapabilityState = CapabilityState.UNSUPPORTED
    supports_grammar: CapabilityState = CapabilityState.UNSUPPORTED
    supports_streaming: CapabilityState = CapabilityState.UNSUPPORTED
    supports_images: CapabilityState = CapabilityState.UNSUPPORTED
    serving_config_digest: str = ""
    probed_at: str = ""


class BackendInspector(Protocol):
    """Backend-specific, side-effect-free runtime inspector."""

    async def inspect(
        self,
        route: ModelRoute,
        transport: UpstreamTransport,
    ) -> ModelRuntimeCapabilities:
        ...
