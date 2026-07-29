"""Compatibility packs — agent-specific and model-specific mappings.

Alias sources (priority order):
1. Schema-declared x-interop-aliases
2. Exact client/version compatibility pack
3. Exact tool fingerprint compatibility entry
4. User route override
5. Minimal global casing normalization

Agent-specific tool name mappings live here, NOT in the universal repair core.

Packs are registered explicitly rather than imported dynamically from
arbitrary client strings — this prevents injection via client-supplied IDs.
"""

from __future__ import annotations

# Registered compatibility packs keyed by client_id
_PACKS: dict[str, dict[str, dict[str, list[str]]]] = {}


def register_pack(client_id: str, aliases: dict[str, dict[str, list[str]]]) -> None:
    """Register a compatibility pack for a known client."""
    _PACKS[client_id] = aliases


def get_pack_aliases(client_id: str, tool_name: str) -> dict[str, list[str]]:
    """Get compatibility-pack aliases for a specific client and tool."""
    # Try registered packs first
    pack = _PACKS.get(client_id)
    if pack is not None:
        return pack.get(tool_name, {})

    # Fallback: try lazy import of known pack modules
    # Only safe client_ids are attempted (validated against known identifiers)
    known_packs = {"claude_code", "codex", "cline", "opencode"}
    if client_id in known_packs:
        return _lazy_load_pack(client_id, tool_name)

    return {}


def _lazy_load_pack(client_id: str, tool_name: str) -> dict[str, list[str]]:
    """Lazily import a known compatibility pack module."""
    try:
        module = __import__(
            f"agent_interop.compatibility_packs.{client_id}",
            fromlist=["ALIASES"],
        )
        aliases = getattr(module, "ALIASES", {})
        if isinstance(aliases, dict):
            _PACKS[client_id] = aliases
            return aliases.get(tool_name, {})
    except ImportError:
        pass
    return {}
