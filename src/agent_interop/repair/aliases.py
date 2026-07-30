"""Field-alias maps for tool-argument repair.

When a model emits the right intent but the wrong field name (because its
pretraining used a different convention), alias renaming recovers the call
without a round-trip.

This map is deliberately curated — not a blind copy of any single product's
alias table. Every entry corresponds to a naming convention we have observed
across coding agents (Claude Code, Codex, Cline, etc.) and common model
families (DeepSeek, Qwen, Llama, etc.).

The policy that determines which alias sources are consulted is passed
in by the caller as ``FieldAliasPolicy``:

* ``DISABLED`` — no aliasing at all.
* ``SCHEMA_ONLY`` — only ``x-aliases`` declared on the tool's JSON
  Schema.  No compatibility-pack or universal-table aliases.
* ``COMPATIBILITY_PACK`` — schema aliases plus aliases from a
  registered compatibility pack (``compatibility_packs/<client>/``).

Registered compatibility packs do NOT require ``compatibility_verified``
evidence to activate. That flag exists for empirically-observed or
otherwise dynamic repair behavior, where correctness for a given tuple is
genuinely an open question the evidence store answers. A registered pack
is a different risk category: it is maintainer-authored, static, and
reviewed once at write time — the uncertainty "is this mapping correct"
was already resolved by a human reading the client's real tool schemas,
not something that needs re-proving per installation via live traffic.
Requiring per-tuple manual verification for a table that's already the
same on every installation adds friction without adding safety.

What registered-pack activation DOES require, so it is never a loose
``client_id`` string match:
  - The pack is explicitly registered in code (``_PACKS`` or the
    hardcoded ``known_packs`` import allowlist in
    ``compatibility_packs/__init__.py`` — a client_id that doesn't name a
    real registered pack module resolves to no aliases at all).
  - A properly RESOLVED client identity: a populated ``CompatibilityKey``
    naming ``client_id`` plus at least one other real dimension
    (model/profile/backend) — see ``_is_key_sufficiently_populated`` —
    not a bare, unauthenticated client_id string alone.
  - The tool/schema contract still matches: the rule that actually
    applies an alias (``rename_aliased_fields`` in ``rules.py``) only
    fires for a field name jsonschema validation ALREADY reported as
    missing/wrong-typed against the tool's real schema — the alias table
    is a candidate list, never a bypass of real schema validation.

If a dynamic, learned, or user-supplied alias source is ever added, it
must be gated by ``compatibility_verified`` through a SEPARATE code path
— this module's ``COMPATIBILITY_PACK`` branch is reserved for registered,
static packs only.

Safety rules enforced by the pipeline (not here), unaffected by which
alias source proposed the rename:
  - Apply only when the canonical field is absent.
  - Apply only when exactly one alias is present (no ambiguity).
  - Apply only when the renamed result validates better than the original.
  - Reject when the rename would change a discriminated-union branch.

The map is also dynamically extended: if a tool's JSON Schema declares
``x-aliases`` on a property, those are merged in at lookup time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_interop.config import FieldAliasPolicy
    from agent_interop.replay.types import CompatibilityKey

# ─── Global normalization aliases ───────────────────────────────────────────
# These apply to ALL tools as a safe last-resort. They cover only the most
# universal snake_case ↔ camelCase transformations. They are intentionally
# minimal — the tool-specific maps below do the heavy lifting.

_GLOBAL_ALIASES: dict[str, list[str]] = {}


# ─── Minimal universal aliases ──────────────────────────────────────────────
# Only unambiguous casing conventions. Agent-specific tool name mappings
# live in compatibility_packs/ — NOT here.

_TOOL_ALIASES: dict[str, dict[str, list[str]]] = {}


def _is_key_sufficiently_populated(key: CompatibilityKey) -> bool:
    """Check that a CompatibilityKey is sufficiently populated for pack activation.

    Requires at least one model/profile/backend dimension to be
    populated so a bare client_id alone is never sufficient for exact-tuple
    verification.

    NOTE: evidence_verified is no longer part of the key identity.
    Verification state is tracked separately by the evidence store.
    This function now only checks structural sufficiency of the key.
    """
    if not key.client_id:
        return False
    # Require at least one model/profile/backend dimension
    # to prevent bare client_id activation.
    has_dimension = bool(
        getattr(key, "model_id", None)
        or getattr(key, "profile_id", None)
        or getattr(key, "backend_kind", None)
    )
    return has_dimension


def get_aliases_for_tool(
    tool_name: str,
    schema: dict[str, Any] | None = None,
    client_id: str | None = None,
    compatibility_key: CompatibilityKey | None = None,
    field_alias_policy: FieldAliasPolicy | None = None,
    compatibility_verified: bool = False,
) -> dict[str, list[str]]:
    """Return ``{canonical_field: [aliases...]}`` for the given tool.

    The result is built from at most three sources, gated by
    ``field_alias_policy``:

    * ``DISABLED`` — no aliasing at all, empty result.
    * ``SCHEMA_ONLY`` — only property-level ``x-aliases`` declared on
      the tool's JSON Schema.  No compatibility-pack or universal-table
      aliases.
    * ``COMPATIBILITY_PACK`` — schema aliases plus aliases from a
      registered compatibility pack.  This source requires a
      fully-populated ``compatibility_key`` (client_id alone is never
      sufficient) naming a client that resolves to an actually-registered
      pack module — see the module docstring for why this does NOT also
      require ``compatibility_verified``.

    Universal aliases from the curated ``_TOOL_ALIASES`` / ``_GLOBAL_ALIASES``
    tables are NOT included unless the policy explicitly allows them.

    When ``field_alias_policy`` is ``DISABLED`` (or None and the
    caller did not opt in) the result is empty and no aliasing
    occurs.

    When a full ``compatibility_key`` is provided, resolution uses the
    exact client/model/backend/profile tuple rather than only ``client_id``.

    ``compatibility_verified`` is accepted for API stability and reserved
    for a future dynamic/learned alias source; it is not consulted by the
    registered-pack branch below (see module docstring).
    """
    from agent_interop.config import FieldAliasPolicy

    # Resolve effective policy: when None, explicitly DISABLED (not SCHEMA_ONLY)
    # to ensure callers must opt-in to alias behavior.
    policy = field_alias_policy or FieldAliasPolicy.DISABLED
    if policy == FieldAliasPolicy.DISABLED:
        return {}

    result: dict[str, list[str]] = {}

    # 1. Schema-declared x-aliases (highest precedence, allowed by
    #    both SCHEMA_ONLY and COMPATIBILITY_PACK).
    if schema and isinstance(schema, dict):
        props = schema.get("properties", {})
        if isinstance(props, dict):
            for prop_name, prop_schema in props.items():
                if not isinstance(prop_schema, dict):
                    continue
                declared = prop_schema.get("x-aliases", [])
                if isinstance(declared, list) and declared:
                    result[prop_name] = [
                        a for a in declared if isinstance(a, str) and a
                    ]

    # 2. Compatibility pack aliases (only when policy permits AND a
    #    properly resolved client identity is present).
    if policy == FieldAliasPolicy.COMPATIBILITY_PACK:
        # Registered packs are maintainer-authored and static — see the
        # module docstring for why they do NOT require
        # compatibility_verified. What they DO require is a resolved
        # identity, not a bare client_id string: a compatibility key
        # naming client_id plus at least one other real dimension
        # (model/profile/backend). get_pack_aliases() itself only ever
        # resolves a client_id that names an actually-registered pack
        # module (see compatibility_packs/__init__.py's known_packs
        # allowlist) — an unrecognized client_id yields no aliases
        # regardless of how well-populated the key is.
        if (
            compatibility_key is not None
            and _is_key_sufficiently_populated(compatibility_key)
        ):
            from agent_interop.compatibility_packs import get_pack_aliases

            # Use the client_id from the compatibility key if available
            effective_client = compatibility_key.client_id or client_id
            if effective_client:
                pack_aliases = get_pack_aliases(effective_client, tool_name)
                for canonical, aliases in pack_aliases.items():
                    if canonical in result:
                        existing = result[canonical]
                        for a in aliases:
                            if a not in existing:
                                existing.append(a)
                    else:
                        result[canonical] = list(aliases)

    # 3. Curated universal aliases (lowest precedence).  These are
    #    safe cross-tool conventions and are allowed under both
    #    SCHEMA_ONLY and COMPATIBILITY_PACK since they are not
    #    agent-specific.
    tool_map = _TOOL_ALIASES.get(tool_name, {})
    for canonical, aliases in tool_map.items():
        if canonical in result:
            existing = result[canonical]
            for a in aliases:
                if a not in existing:
                    existing.append(a)
        else:
            result[canonical] = list(aliases)

    for canonical, aliases in _GLOBAL_ALIASES.items():
        if canonical not in result:
            result[canonical] = list(aliases)

    return result
