"""Backend runtime inspection and admin clients.

``ollama_admin.OllamaAdminClient`` lives in this package but is
intentionally NOT exported here: no production CLI feature or gateway
path currently uses it. Import it directly
(``from agent_interop.backends.ollama_admin import OllamaAdminClient``) if you
are building on it, but it should not be treated as a supported public
surface until something in the tested request/CLI path actually wires
it in.
"""

from agent_interop.backends.base import BackendInspector, ModelRuntimeCapabilities
from agent_interop.backends.registry import get_backend_inspector, register_backend_inspector

__all__ = [
    "BackendInspector",
    "ModelRuntimeCapabilities",
    "get_backend_inspector",
    "register_backend_inspector",
]
