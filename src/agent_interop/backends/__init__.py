"""Backend clients for local inference servers.

``ollama_admin.OllamaAdminClient`` lives in this package but is
intentionally NOT exported here: no production CLI feature or gateway
path currently uses it. Import it directly
(``from agent_interop.backends.ollama_admin import OllamaAdminClient``) if you
are building on it, but it should not be treated as a supported public
surface until something in the tested request/CLI path actually wires
it in.
"""

__all__: list[str] = []
