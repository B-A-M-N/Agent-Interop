"""Compatibility packs — agent-specific tool-name and field-alias mappings.

Alias sources (priority order):
1. Schema-declared x-interop-aliases (field level)
2. Exact client/version/fingerprint compatibility pack (tool-name + field)
3. User route override
4. Minimal global casing normalization

Two distinct kinds of alias live here:

* **Field aliases** (``ALIASES``) — property-level rename candidates, consumed by
  ``repair/aliases.py`` via :func:`get_pack_aliases`. These recover a call where
  the model used the wrong *argument* field name.
* **Tool-name aliases** (``TOOL_NAME_ALIASES``) — semantic tool-identity
  mappings (e.g. ``read_file`` → ``Read``, ``shell`` → ``Bash``), consumed by
  :func:`get_tool_name_alias_map`. These recover a call where the model used the
  wrong *tool* name but the right intent.

Both are curated, maintainer-authored, and reviewed once at write time — never
learned from traffic. Activation is gated by a resolved compatibility identity,
not a bare ``client_id`` string alone (see ``_is_identity_resolved``).

Pack identity is stronger than ``client_id``: a pack is selected only when the
resolved client id, version constraint, and tool-schema fingerprint all line up.
A client merely *claiming* to be ``claude_code`` does not inherit aliases meant
for a specific Claude Code tool-schema revision — this prevents a different
agent or a future Claude Code revision with renamed tools from silently
absorbing aliases that no longer apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Registered compatibility packs keyed by client_id. Each entry is a
# ``ClientCompatibilityPack`` describing version constraints and the alias
# tables. Populated by :func:`register_pack` or lazy import of a
# ``compatibility_packs/<client_id>`` module.
_PACKS: dict[str, ClientCompatibilityPack] = {}


def _version_satisfies(constraint: str, version: str | None) -> bool:
    """Loose semver-prefix check: ``constraint`` is a ``major.minor`` or
    ``major`` prefix the resolved ``version`` must start with. An empty or
    ``*`` constraint matches anything.

    When no version is resolved (``version is None``) we cannot *disprove* the
    constraint, so we return True — a missing version simply means we don't
    tighten the match, it must not block a pack that would otherwise apply.
    The exactness gate for packs is the tool-schema fingerprint, not the
    version. A *conflicting* resolved version still rejects.
    """
    if not constraint or constraint in ("*", "any"):
        return True
    if version is None:
        return True
    # Normalize: take leading numeric.dot segments from the version.
    m = re.match(r"(\d+(?:\.\d+)?)", version)
    v_prefix = m.group(1) if m else version
    return v_prefix.startswith(constraint) or constraint.startswith(v_prefix)


@dataclass(frozen=True)
class ClientCompatibilityPack:
    """A curated, static compatibility pack for one client.

    Identity is the tuple (client_id, version_constraint,
    tool_schema_fingerprint). Only when all three resolve does the pack apply.
    """

    client_id: str
    version_constraint: str = "*"
    tool_schema_fingerprint: str | None = None
    # {canonical_tool_name: (alias, ...)} — semantic tool-name recovery.
    tool_name_aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # {canonical_tool_name: {canonical_field: [alias, ...]}} — field-level
    # recovery, keyed by tool name (consumed via pack.field_aliases[tool_name]).
    field_aliases: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name_aliases", dict(self.tool_name_aliases or {})
        )
        object.__setattr__(self, "field_aliases", dict(self.field_aliases or {}))

    def matches(self, version: str | None, fingerprint: str | None) -> bool:
        if not _version_satisfies(self.version_constraint, version):
            return False
        if self.tool_schema_fingerprint and fingerprint:
            # A declared fingerprint must match the resolved one exactly.
            return self.tool_schema_fingerprint == fingerprint
        # No fingerprint declared → version check alone suffices.
        return True


def register_pack(pack: ClientCompatibilityPack) -> None:
    """Register a curated compatibility pack (strong identity)."""
    _PACKS[pack.client_id] = pack


def _lazy_load_pack(client_id: str) -> ClientCompatibilityPack | None:
    """Lazily import a known compatibility pack module.

    Only safe, allowlisted client_ids are attempted. The module must expose a
    ``BUILD_PACK`` ``ClientCompatibilityPack`` (preferred) or a legacy
    ``ALIASES`` dict (field aliases only).
    """
    known_packs = {"claude_code", "codex", "cline", "opencode", "hermes_agent"}
    if client_id not in known_packs:
        return None
    try:
        module = __import__(
            f"agent_interop.compatibility_packs.{client_id}",
            fromlist=["BUILD_PACK", "ALIASES"],
        )
    except ImportError:
        return None
    build = getattr(module, "BUILD_PACK", None)
    if isinstance(build, ClientCompatibilityPack):
        _PACKS[client_id] = build
        return build
    aliases = getattr(module, "ALIASES", {})
    if isinstance(aliases, dict) and aliases:
        pack = ClientCompatibilityPack(client_id=client_id, field_aliases=aliases)
        _PACKS[client_id] = pack
        return pack
    return None


def _resolve_pack(
    client_id: str | None,
    client_version: str | None = None,
    tool_schema_fingerprint: str | None = None,
) -> ClientCompatibilityPack | None:
    """Resolve a pack by strong identity, never a bare client_id.

    A pack applies only if its version constraint AND (when declared) schema
    fingerprint match the resolved identity. An unrecognized or non-matching
    client resolves to no pack at all.
    """
    if not client_id:
        return None
    pack = _PACKS.get(client_id) or _lazy_load_pack(client_id)
    if pack is None:
        return None
    if not pack.matches(client_version, tool_schema_fingerprint):
        return None
    return pack


# ─── Field-alias API (consumed by repair/aliases.py) ────────────────────────


def get_pack_aliases(client_id: str, tool_name: str) -> dict[str, list[str]]:
    """Field-level aliases for a tool under a resolved client identity.

    Delegates to :func:`_resolve_pack` with no version/fingerprint (the
    field-alias path is keyed by client_id + tool name). Returns ``{}`` when
    no pack resolves — never raises for an unknown client.
    """
    pack = _resolve_pack(client_id)
    if pack is None:
        return {}
    return dict(pack.field_aliases.get(tool_name, {}))


# ─── Tool-name-alias API (consumed by repair/pipeline.canonicalize_tool_name) ─


def get_tool_name_alias_map(
    client_id: str | None,
    client_version: str | None = None,
    tool_schema_fingerprint: str | None = None,
    declared_tool_names: tuple[str, ...] = (),
) -> dict[str, str]:
    """Return ``{alias_lower: canonical_tool_name}`` for semantic tool recovery.

    Only aliases that resolve to exactly ONE declared client tool are emitted,
    so a malformed model call ``read_file`` maps to ``Read`` only when the
    client actually declared exactly one tool matching ``Read``. Ambiguous
    aliases (more than one candidate) are dropped — never guessed.

    Identity is resolved by strong tuple, not bare client_id.
    """
    pack = _resolve_pack(client_id, client_version, tool_schema_fingerprint)
    if pack is None:
        return {}
    declared = set(declared_tool_names)
    result: dict[str, str] = {}
    for canonical, aliases in pack.tool_name_aliases.items():
        if canonical not in declared:
            continue
        for alias in aliases:
            al = alias.lower()
            if al in result and result[al] != canonical:
                # Ambiguous — two canonical tools both claim this alias. Drop.
                del result[al]
            else:
                result[al] = canonical
    return result
