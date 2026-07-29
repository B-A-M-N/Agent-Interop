"""Deterministic, ordered repair rules for tool-call arguments.

Each rule returns ``RepairProposal | None``. Rules never mutate the input
dict — the pipeline applies proposals and verifies the contract.

Each rule declares which schema keywords it handles via ``handled_keywords``.
A rule only fires when the cursor's target issue matches a keyword it handles.

Rule ordering is critical:
  1. rename_aliased_fields (required, type)
  2. drop_null_optional (type)
  3. drop_empty_placeholder (type)
  4. parse_stringified_arrays (type)
  5. parse_stringified_object (type)
  6. wrap_bare_string_as_array (type)
  7. coerce_scalar_types (type)

All nested-field access uses get_at_path / set_at_path / delete_at_path (item 45).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_interop.abi import RepairCursor, RepairOperation, RepairProposal
from agent_interop.config import FieldAliasPolicy, RepairTier
from agent_interop.repair.aliases import get_aliases_for_tool
from agent_interop.repair.paths import _MISSING, get_at_path

if TYPE_CHECKING:
    from agent_interop.replay.types import CompatibilityKey


@dataclass(frozen=True)
class RepairRuleContext:
    """Context passed to repair rules."""

    client_id: str | None = None
    request_id: str = ""
    field_alias_policy: FieldAliasPolicy | None = None
    compatibility_key: CompatibilityKey | None = None
    compatibility_verified: bool = False


def _rule_tier(tier: str):
    """Decorator to assign a repair tier to a rule function."""
    def decorator(fn):
        fn.tier = tier
        return fn
    return decorator


def _handles(*keywords: str):
    """Decorator to declare which schema keywords a rule handles."""
    def decorator(fn):
        fn.handled_keywords = frozenset(keywords)
        return fn
    return decorator


def _would_change_union_branch(
    args: dict[str, Any],
    alias: str,
    target: str,
    cursor: RepairCursor,
) -> bool:
    """Check if renaming alias→target would change a discriminated-union branch.

    When the schema has oneOf/anyOf with a discriminator property, moving a field
    from one name to another could shift which union variant is selected. This is
    only a concern when the field being moved IS the discriminator or influences
    the discriminator selection.
    """
    tool = cursor.tool
    if tool is None:
        return False

    schema = tool.input_schema
    if not schema:
        return False

    # Check for discriminator at the current level
    discriminator = schema.get("discriminator", {})
    if isinstance(discriminator, dict) and discriminator.get("propertyName"):
        prop_name = discriminator["propertyName"]
        # If we're renaming a field that matches the discriminator, it's unsafe
        if alias == prop_name or target == prop_name:
            return True

    # Check oneOf/anyOf branches for discriminator-like behavior
    for keyword in ("oneOf", "anyOf"):
        branches = schema.get(keyword, [])
        if not isinstance(branches, list):
            continue
        for branch in branches:
            if isinstance(branch, dict):
                branch_disc = branch.get("discriminator", {})
                if isinstance(branch_disc, dict) and branch_disc.get("propertyName"):
                    prop_name = branch_disc["propertyName"]
                    if alias == prop_name or target == prop_name:
                        return True

    return False


# ─── Rule 1: Rename aliased fields ──────────────────────────────────────────


@_rule_tier(RepairTier.SAFE_SHAPE)
@_handles("required", "type")
def rename_aliased_fields(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Rename fields the model emitted with non-canonical names.

    Fires on ``required`` or ``type`` errors: if the canonical field is absent
    but one of its aliases is present (and is the only candidate), propose a rename.
    """
    tool = cursor.tool
    if tool is None:
        return None

    alias_map = get_aliases_for_tool(
        tool.name,
        tool.input_schema,
        client_id=context.client_id,
        compatibility_key=context.compatibility_key,
        field_alias_policy=context.field_alias_policy,
        compatibility_verified=context.compatibility_verified,
    )
    if not alias_map:
        return None

    # Determine the target field from the cursor
    target: str | None = None
    parent_path: list[str | int] = []

    if cursor.instance_path:
        # For nested issues, get the parent path and the final field name
        if len(cursor.instance_path) > 1:
            parent_path = list(cursor.instance_path[:-1])
            target = str(cursor.instance_path[-1])
        else:
            parent_path = []
            target = str(cursor.instance_path[0])
    elif cursor.issue and cursor.issue.keyword == "required":
        # For required errors at root, instance_path may be empty
        expected = cursor.issue.expected
        if isinstance(expected, list) and expected and isinstance(expected[0], str):
            target = expected[0]
        elif isinstance(expected, str) and expected:
            cleaned = expected.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                inner = cleaned[1:-1].strip()
                target = inner.strip("'\"")
            elif cleaned.startswith(("'", '"')) and cleaned.endswith(("'", '"')):
                target = cleaned[1:-1]
            else:
                target = cleaned.split(",")[0].strip("[']\" ")
        parent_path = []  # Root-level required field

    if not target or target not in alias_map:
        return None

    # Collision check: if canonical field already exists AND alias also exists
    # with different values, the rename is ambiguous — reject it.
    if target in args and args[target] is not None:
        # Both canonical and alias present — only safe if values are identical
        aliases = alias_map[target]
        present_aliases = [a for a in aliases if a in args and args[a] is not None]
        if present_aliases:
            alias = present_aliases[0]
            if args[target] != args[alias]:
                # Ambiguous: canonical and alias have different values.
                # Cannot safely determine model intent — reject repair.
                return None
            # Values are identical — safe to drop the alias (duplicate).
            # But this is a delete, not a rename. Fall through to normal path.
        return None

    aliases = alias_map[target]
    present_aliases = [a for a in aliases if a in args and args[a] is not None]
    if len(present_aliases) != 1:
        return None

    alias = present_aliases[0]
    alias_value = args[alias]

    # Discriminated-union safety: if the parent has a "type" discriminator field,
    # verify the rename doesn't change which union branch is selected.
    if _would_change_union_branch(args, alias, target, cursor):
        return None

    # Build exact source and target paths
    target_path = parent_path + [target]
    source_path = parent_path + [alias]

    return RepairProposal(
        rule_id="rename_aliased_fields",
        operation=RepairOperation.MOVE,
        target_path=target_path,
        source_path=source_path,
        before=alias_value,
        after=alias_value,
        message=f"Renamed `{alias}` → `{target}`",
        issue_identity=f"required:{target}" if cursor.issue and cursor.issue.keyword == "required" else f"type:{target}",
    )


# ─── Rule 2: Drop null/None from optional fields ────────────────────────────


@_rule_tier(RepairTier.SAFE_SHAPE)
@_handles("type")
def drop_null_optional(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Remove ``null`` values from optional fields.

    Uses get_at_path to support nested paths (item 45).
    """
    tool = cursor.tool
    if tool is None:
        return None

    target_path = list(cursor.instance_path)
    if not target_path:
        return None

    # For root-level issues, the path may be empty (jsonschema reports
    # required errors at parent path). Use the issue's expected field.
    if len(target_path) == 1:
        target_key = str(target_path[0])
        # Check it's a direct key of args (root-level)
        if target_key not in args or args[target_key] is not None:
            # Maybe it's nested — try get_at_path
            value = get_at_path(args, target_path)
            if value is _MISSING or value is not None:
                return None
    else:
        # Nested path — use get_at_path
        value = get_at_path(args, target_path)
        if value is _MISSING or value is not None:
            return None

    # Determine the field name for schema lookup
    # For nested paths, resolve the parent schema
    root_schema = tool.input_schema
    required = set(root_schema.get("required", []))

    # For root-level single-segment paths
    if len(target_path) == 1:
        target_key = str(target_path[0])
        if target_key in required:
            return None

        props = root_schema.get("properties", {})
        prop_schema = props.get(target_key)
        if isinstance(prop_schema, dict):
            ptype = prop_schema.get("type")
            types = ptype if isinstance(ptype, list) else [ptype]
            if "null" in types or prop_schema.get("nullable") is True:
                return None
            if prop_schema.get("enum") and None in prop_schema["enum"]:
                return None
    else:
        # For nested paths, check if the parent has the field as required
        # Walk the schema to find the property schema
        parent_path = target_path[:-1]
        parent_schema = root_schema
        for segment in parent_path:
            if isinstance(parent_schema, dict):
                props = parent_schema.get("properties", {})
                if isinstance(props, dict) and str(segment) in props:
                    parent_schema = props[str(segment)]
                else:
                    # Can't determine schema — be conservative, allow drop
                    break

        if isinstance(parent_schema, dict):
            parent_required = set(parent_schema.get("required", []))
            last_key = str(target_path[-1])
            if last_key in parent_required:
                return None

    return RepairProposal(
        rule_id="drop_null_optional",
        target_path=target_path,
        before=None,
        after=None,
        delete=True,
        message=f"Dropped null optional field: {'.'.join(str(s) for s in target_path)}",
        issue_identity=f"type:{'.'.join(str(s) for s in target_path)}",
    )


# ─── Rule 3: Drop empty-object placeholder ──────────────────────────────────


@_rule_tier(RepairTier.SAFE_SHAPE)
@_handles("type")
def drop_empty_placeholder(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Remove ``{}`` where an array is expected."""
    tool = cursor.tool
    if tool is None:
        return None

    target_path = list(cursor.instance_path)
    if not target_path:
        return None

    target_key = str(target_path[0]) if len(target_path) == 1 else None
    if target_key is None:
        return None

    required = set(tool.input_schema.get("required", []))
    if target_key in required:
        return None

    value = get_at_path(args, target_path)
    if value is _MISSING or not (isinstance(value, dict) and len(value) == 0):
        return None

    props = tool.input_schema.get("properties", {})
    prop_schema = props.get(target_key)
    if not isinstance(prop_schema, dict):
        return None
    expected = prop_schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if "array" not in types:
        return None

    return RepairProposal(
        rule_id="drop_empty_placeholder",
        target_path=target_path,
        before={},
        after=None,
        delete=True,
        message=f"Dropped empty `{{}}` placeholder for array field: {target_key}",
        issue_identity=f"type:{target_key}",
    )


# ─── Rule 4: Parse stringified arrays ──────────────────────────────────────


@_rule_tier(RepairTier.SAFE_SHAPE)
@_handles("type")
def parse_json_stringified_array(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Parse JSON-encoded strings that should be arrays."""
    target_path = list(cursor.instance_path)
    if not target_path:
        return None

    value = get_at_path(args, target_path)
    if value is _MISSING or not isinstance(value, str):
        return None

    # Get schema for this path
    schema = cursor.target_schema
    if not isinstance(schema, dict):
        return None
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if "array" not in types:
        return None

    trimmed = value.strip()
    if not (trimmed.startswith("[") and trimmed.endswith("]")):
        return None

    try:
        parsed = json.loads(trimmed)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, list):
        return None

    return RepairProposal(
        rule_id="parse_json_stringified_array",
        target_path=target_path,
        before=value,
        after=parsed,
        message="Parsed JSON string into array",
        issue_identity=f"type:{'.'.join(str(s) for s in target_path)}",
    )


# ─── Rule 4b: Parse stringified objects ─────────────────────────────────────


@_rule_tier(RepairTier.SAFE_SHAPE)
@_handles("type")
def parse_json_stringified_object(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Parse JSON-encoded strings that should be objects."""
    target_path = list(cursor.instance_path)
    if not target_path:
        return None

    value = get_at_path(args, target_path)
    if value is _MISSING or not isinstance(value, str):
        return None

    schema = cursor.target_schema
    if not isinstance(schema, dict):
        return None
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if "object" not in types:
        return None

    trimmed = value.strip()
    if not (trimmed.startswith("{") and trimmed.endswith("}")):
        return None

    try:
        parsed = json.loads(trimmed)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None

    return RepairProposal(
        rule_id="parse_json_stringified_object",
        target_path=target_path,
        before=value,
        after=parsed,
        message="Parsed JSON string into object",
        issue_identity=f"type:{'.'.join(str(s) for s in target_path)}",
    )


# ─── Rule 5: Wrap bare string as array ──────────────────────────────────────


@_rule_tier(RepairTier.COERCIVE)
@_handles("type")
def wrap_bare_string_as_array(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Wrap a bare string into a single-element array where schema expects array."""
    target_path = list(cursor.instance_path)
    if not target_path:
        return None

    value = get_at_path(args, target_path)
    if value is _MISSING or not isinstance(value, str):
        return None

    schema = cursor.target_schema
    if not isinstance(schema, dict):
        return None
    expected = schema.get("type")
    types = expected if isinstance(expected, list) else [expected]
    if "array" not in types:
        return None

    item_schema = schema.get("items", {})
    item_type = item_schema.get("type", "string") if isinstance(item_schema, dict) else "string"
    if item_type not in ("string", None):
        return None

    return RepairProposal(
        rule_id="wrap_bare_string_as_array",
        target_path=target_path,
        before=value,
        after=[value],
        message="Wrapped bare string in single-element array",
        issue_identity=f"type:{'.'.join(str(s) for s in target_path)}",
    )


# ─── Rule 6: Coerce scalar types ────────────────────────────────────────────


_TRUE_STRINGS = {"true", "1", "yes"}
_FALSE_STRINGS = {"false", "0", "no"}


@_rule_tier(RepairTier.COERCIVE)
@_handles("type")
def coerce_scalar_types(
    args: dict[str, Any],
    cursor: RepairCursor,
    context: RepairRuleContext,
) -> RepairProposal | None:
    """Coerce scalars between compatible types: string↔number↔boolean."""
    target_path = list(cursor.instance_path)
    if not target_path:
        return None

    value = get_at_path(args, target_path)
    if value is _MISSING:
        return None

    schema = cursor.target_schema
    if not isinstance(schema, dict):
        return None
    expected_types = schema.get("type")
    types = expected_types if isinstance(expected_types, list) else [expected_types]

    coerced = _coerce_value(value, types)
    if coerced is _MISSING:
        return None

    return RepairProposal(
        rule_id="coerce_scalar_types",
        target_path=target_path,
        before=value,
        after=coerced,
        message=f"Coerced {type(value).__name__} → {type(coerced).__name__}",
        issue_identity=f"type:{'.'.join(str(s) for s in target_path)}",
    )


def _coerce_value(value: Any, types: list[Any]) -> Any:
    """Attempt to coerce value to match one of the expected types.

    Returns _MISSING if no coercion is possible.
    """
    if "string" in types and not isinstance(value, str):
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)

    if "integer" in types and isinstance(value, str):
        try:
            int_val = int(value)
            if str(int_val) == value.strip():
                return int_val
        except (ValueError, TypeError):
            pass

    if "integer" in types and isinstance(value, float) and value.is_integer():
        return int(value)

    if "number" in types and isinstance(value, str):
        try:
            import math
            num_val = float(value)
            if not math.isfinite(num_val):
                return _MISSING
            if _is_clean_number(value):
                return num_val
        except (ValueError, TypeError):
            pass

    if "boolean" in types and isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_STRINGS:
            return True
        if lowered in _FALSE_STRINGS:
            return False

    return _MISSING


def _is_clean_number(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


# ─── Rule registry (ORDER MATTERS) ──────────────────────────────────────────


REPAIR_RULES = [
    rename_aliased_fields,
    drop_null_optional,
    drop_empty_placeholder,
    parse_json_stringified_array,
    parse_json_stringified_object,
    wrap_bare_string_as_array,
    coerce_scalar_types,
]
