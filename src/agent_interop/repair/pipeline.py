"""The validate-then-repair pipeline — heart of Interop's tool-call repair.

Architecture:

    raw tool-call candidate
        │
        ▼
    [Step 1] tool-name canonicalization  ── fail → REJECT
        │
        ▼
    [Step 2] argument JSON recovery      ── fail → REJECT
        │
        ▼
    [Step 3] JSON Schema validation (jsonschema, draft 2020-12)
        │
        ├─ no issues → VALID_UNCHANGED (valid inputs are never touched)
        │
        ▼ issues exist
    [Step 4] run ordered repair rules
        │
        ▼
    [Step 5] re-validate repaired args
        │
        ├─ issues reduced → REPAIRED
        └─ no improvement → REJECT

Key properties:
  - Valid inputs are NEVER mutated.
  - Rejected calls are NEVER emitted as executable.
  - The pipeline is pure: returns RepairOutcome, never mutates input.
  - Issues are processed in stable deterministic order.
  - Repair proposals are scoped to the declared target path.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import CanonicalTool, RepairCursor, RepairOperation, RepairProposal
from agent_interop.repair.parse import parse_tool_args
from agent_interop.repair.rules import REPAIR_RULES
from agent_interop.repair.schema import validate_against_schema
from agent_interop.repair.types import RepairOutcome, RepairStatus, RepairStep, SchemaIssue
from agent_interop.replay.types import CompatibilityKey

logger = logging.getLogger("agent_interop.repair")


@dataclass
class RepairBudget:
    """Response-scoped budget for tool-call repair operations.

    Shared across all candidates in a batch so that latency, input bytes,
    and regeneration attempts are tracked collectively rather than per-call.
    """

    started_at: float = field(default_factory=time.monotonic)
    input_bytes_processed: int = 0
    regeneration_attempts: int = 0
    repair_operations: int = 0

    def add_input_bytes(self, n: int) -> None:
        self.input_bytes_processed += n

    def add_regeneration(self) -> None:
        self.regeneration_attempts += 1

    def add_repair_operation(self) -> None:
        self.repair_operations += 1

    def remaining_latency_ms(self, max_ms: int = 15000) -> float:
        """Return the remaining latency budget in milliseconds.

        Returns 0.0 if the budget is exhausted.
        """
        elapsed = (time.monotonic() - self.started_at) * 1000
        remaining = max_ms - elapsed
        return max(0.0, remaining)

    def is_deterministic_exhausted(self, max_ms: int = 15000, max_input: int = 65536, max_operations: int = 100) -> bool:
        """Check if the deterministic repair budget (latency, bytes, operations) is exhausted.

        This does NOT include the regeneration budget. Regeneration is only
        checked immediately before a regeneration attempt via can_regenerate().
        This ensures that disabling regeneration (max_regenerations=0) does
        NOT disable validation or deterministic repair.
        """
        if self.remaining_latency_ms(max_ms) <= 0:
            return True
        if self.input_bytes_processed > max_input:
            return True
        return self.repair_operations >= max_operations

    def can_regenerate(self, max_regen: int = 1) -> bool:
        """Check whether a regeneration attempt is still permitted.

        Called immediately before a regeneration attempt starts.
        Use add_regeneration() when the attempt begins.
        """
        return self.regeneration_attempts < max_regen


# ─── Stable issue ordering (item 42) ──────────────────────────────────────────


def _sort_issues(issues: list[SchemaIssue]) -> list[SchemaIssue]:
    """Sort issues into stable deterministic order for repair.

    Ordering priority:
    1. Required-field errors first (they enable alias-rename and other rules)
    2. Type errors second
    3. Path depth (shallow first — fix roots before leaves)
    4. Lexicographic path for determinism
    """
    def sort_key(issue: SchemaIssue) -> tuple[int, int, tuple]:
        keyword_priority = 0 if issue.keyword == "required" else 1 if issue.keyword == "type" else 2
        path_tuple = tuple(
            (0, s) if isinstance(s, str) else (1, s)
            for s in issue.path
        )
        return (keyword_priority, len(issue.path), path_tuple)

    return sorted(issues, key=sort_key)


# ─── Path containment check (item 41) ────────────────────────────────────────


def _path_is_under(path: tuple[str | int, ...], prefix: tuple[str | int, ...]) -> bool:
    """Check if a changed path is at or under the allowed prefix.

    A path is "under" a prefix if it equals the prefix or extends it.
    This prevents a proposal targeting `a.b` from also mutating `a.c`
    or `a` itself.
    """
    if not prefix:
        return True
    if len(path) < len(prefix):
        return False  # Shorter path can't be "under" a longer prefix
    return path[:len(prefix)] == prefix


# ─── Schema resolution helpers ───────────────────────────────────────────────


def _resolve_ref(schema: dict[str, Any], ref: str, root: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local JSON Schema $ref to its target.

    Supports #/$defs/Name and #/definitions/Name style references.
    """
    if not ref.startswith("#/"):
        return {}
    parts = ref.lstrip("#/").split("/")
    node: Any = root
    for part in parts:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return {}
    return node if isinstance(node, dict) else {}


def _resolve_schema_node(schema: dict[str, Any], root: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve $ref and collect union branches from a schema node.

    Returns the fully resolved schema with union branches expanded.
    """
    if root is None:
        root = schema
    # Follow $ref
    if "$ref" in schema:
        resolved = _resolve_ref(root, schema["$ref"], root)
        if resolved:
            return resolved
    return schema


def _get_union_branches(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract candidate branches from oneOf/anyOf/allOf."""
    branches: list[dict[str, Any]] = []
    for key in ("oneOf", "anyOf"):
        if key in schema and isinstance(schema[key], list):
            branches.extend([b for b in schema[key] if isinstance(b, dict)])
    # allOf: merge all branches into a single composite schema
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and all_of:
        merged: dict[str, Any] = {}
        for branch in all_of:
            if isinstance(branch, dict):
                merged.update(branch)
        if merged:
            branches.append(merged)
    return branches


def _get_conditional_schema(
    schema: dict[str, Any],
    instance: Any,
) -> dict[str, Any] | None:
    """Resolve if/then/else to the applicable branch.

    If the instance validates against the 'if' schema, returns 'then'.
    Otherwise returns 'else' if present.
    """
    if "if" not in schema:
        return None
    import jsonschema
    try:
        jsonschema.validate(instance, schema["if"])
        if "then" in schema and isinstance(schema["then"], dict):
            return schema["then"]
    except jsonschema.ValidationError:
        if "else" in schema and isinstance(schema["else"], dict):
            return schema["else"]
    except Exception:
        pass
    return None


def _build_repair_cursor(
    issue: SchemaIssue,
    args: dict[str, Any],
    schema: dict[str, Any],
    tool: CanonicalTool | None,
    client_id: str | None,
) -> RepairCursor:
    """Build a RepairCursor scoped to a specific schema issue's path.

    Walks the instance and schema trees to find the parent container
    and current value at the issue's instance path. Supports:
    - Local $ref resolution
    - oneOf/anyOf/allOf branch expansion
    - if/then/else conditional schemas
    - Array item schemas
    """
    from agent_interop.abi import RepairCursor

    # Resolve the root schema (follow top-level $ref)
    root_schema = _resolve_schema_node(schema, schema)

    # Handle if/then/else: select the applicable conditional branch
    conditional = _get_conditional_schema(root_schema, args)
    if conditional is not None:
        root_schema = {**root_schema, **conditional}

    instance_path = list(issue.path)
    parent_instance: Any = args
    current_value: Any = args

    # Walk the instance tree to find the value at the issue path
    for segment in instance_path:
        parent_instance = current_value
        if isinstance(current_value, dict) and str(segment) in current_value:
            current_value = current_value[str(segment)]
        elif isinstance(current_value, list) and isinstance(segment, int) and 0 <= segment < len(current_value):
            current_value = current_value[segment]
        else:
            # Path not fully traversable — cursor points to the deepest reachable node
            break

    # Walk the schema tree to find the target schema at the issue path
    target_schema: Any = root_schema
    parent_schema: Any = root_schema
    for segment in instance_path:
        if isinstance(target_schema, dict):
            # Resolve any $ref at this level
            resolved = _resolve_schema_node(target_schema, root_schema)
            if resolved is not target_schema:
                target_schema = resolved
                parent_schema = resolved
            if "properties" in target_schema and str(segment) in target_schema["properties"]:
                parent_schema = target_schema
                target_schema = target_schema["properties"][str(segment)]
            elif "items" in target_schema:
                parent_schema = target_schema
                target_schema = target_schema["items"]
                # Handle array items that use oneOf/anyOf
                if not isinstance(target_schema, dict) or target_schema == {}:
                    # Try union branches for item schema
                    for _discard in ("oneOf", "anyOf"):
                        branches = _get_union_branches(parent_schema)
                        if branches:
                            target_schema = branches[0]
                            break
            else:
                # Try union branches for this path segment
                branches = _get_union_branches(target_schema)
                if branches:
                    # Pick the first branch that has the segment
                    for branch in branches:
                        if isinstance(branch, dict) and "properties" in branch and str(segment) in branch["properties"]:
                            parent_schema = branch
                            target_schema = branch["properties"][str(segment)]
                            break
                    else:
                        break
                else:
                    break
        else:
            break

    return RepairCursor(
        issue=issue,
        instance_path=instance_path,
        schema_path=list(issue.absolute_schema_path) if issue.absolute_schema_path else instance_path,
        parent_instance=parent_instance,
        current_value=current_value,
        parent_schema=parent_schema if isinstance(parent_schema, dict) else {},
        target_schema=target_schema if isinstance(target_schema, dict) else {},
        tool=tool,
        client_id=client_id,
    )


# ─── Tool name canonicalization ──────────────────────────────────────────────


def canonicalize_tool_name(
    name: str,
    tools: list[CanonicalTool],
    _policy: Any = None,
) -> str | None:
    """Canonicalize a tool name, returning the exact registered name.

    Safe normalizations (only if they produce exactly one unique match):
      - case (Read_File → read_file)
      - hyphen/underscore (list-files → list_files)
      - namespace strip (mcp__server__read_file → read_file)

    When _policy is provided and its unknown_tool_policy is REJECT,
    only exact match is performed — all normalizations are skipped.

    Returns None if no unique match exists.
    """
    if not name:
        return None

    tool_map = {t.name: t for t in tools}

    # Exact match — always works regardless of policy
    if name in tool_map:
        return name

    # Check if normalization is allowed by policy
    if _policy is not None and hasattr(_policy, 'unknown_tool_policy'):
        from agent_interop.config import UnknownToolPolicy
        if _policy.unknown_tool_policy == UnknownToolPolicy.REJECT:
            return None

    # Case-insensitive exact match
    name_lower = name.lower()
    case_matches = [t for t in tool_map if t.lower() == name_lower]
    if len(case_matches) == 1:
        return case_matches[0]

    # Hyphen/underscore normalization
    normalized = name_lower.replace("-", "_")
    norm_matches = [t for t in tool_map if t.lower().replace("-", "_") == normalized]
    if len(norm_matches) == 1:
        return norm_matches[0]

    # Namespace strip: mcp__server__tool → tool (MCP convention)
    if "__" in name:
        short = name.rsplit("__", 1)[-1]
        ns_matches = [t for t in tool_map if t == short or t.lower() == short.lower()]
        if len(ns_matches) == 1:
            return ns_matches[0]

    return None


def _parse_args_exact(raw_args: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Parse arguments using exact JSON parsing only — no syntax recovery.

    Returns the parsed dict, or None if parsing fails.
    """
    import json

    if raw_args is None:
        return None
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str):
        return None

    text = raw_args.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return None


# ─── Repair notes extraction (item 44) ──────────────────────────────────────


def extract_repair_notes(outcome: RepairOutcome) -> list[str]:
    """Extract human-readable repair notes from a RepairOutcome.

    Returns a list of structured notes describing what was repaired,
    what was rejected, and what remains unresolved. Useful for telemetry
    and model-readable error feedback.
    """
    notes: list[str] = []

    if outcome.status == RepairStatus.VALID_UNCHANGED:
        return notes

    if outcome.status == RepairStatus.REPAIRED and outcome.steps:
        for step in outcome.steps:
            notes.append(f"repaired: {step.rule} at {step.path} — {step.message}")
        return notes

    if outcome.status == RepairStatus.REJECTED:
        if outcome.error:
            notes.append(f"rejected: {outcome.error}")
        for issue in outcome.final_issues[:5]:
            path = ".".join(str(p) for p in issue.path) if issue.path else "root"
            notes.append(f"unresolved: {path} — {issue.message}")
        if outcome.steps:
            notes.append(f"attempted: {', '.join(s.rule for s in outcome.steps)}")
        return notes

    return notes


# ─── Main pipeline ───────────────────────────────────────────────────────────


def repair_one(
    call_name: str,
    call_arguments: dict[str, Any] | str | None,
    tools: list[CanonicalTool],
    policy: Any = None,
    client_id: str | None = None,
    telemetry: Any = None,
    request_id: str = "",
    budget: RepairBudget | None = None,
    compatibility_key: CompatibilityKey | None = None,
    compatibility_verified: bool = False,
) -> RepairOutcome:
    """Run a single tool-call through the validate-then-repair pipeline.

    Args:
        call_name: The raw emitted tool name.
        call_arguments: The raw arguments (JSON string, or dict).
        tools: The list of registered canonical tools.
        policy: RepairPolicy controlling which repair tiers are enabled.
        client_id: Client identifier for compatibility-pack aliases.
        telemetry: Optional telemetry sink.
        request_id: Optional request ID for telemetry.
        budget: Response-scoped repair budget. If None, an unlimited budget
            is created (safe for single-call contexts).
        compatibility_key: Exact client/model/backend/profile tuple for
            compatibility pack activation. Required for exact-tuple verification.

    Returns:
        A RepairOutcome. If REJECTED, accepted is None.
    """
    # Import here to avoid circular import
    from agent_interop.config import RepairPolicy, RepairTier

    if policy is None:
        policy = RepairPolicy()
    if budget is None:
        budget = RepairBudget()
    tool_map = {t.name: t for t in tools}

    max_input = policy.max_input_bytes if hasattr(policy, 'max_input_bytes') else 65536

    # ── Step 1: Tool-name canonicalization ────────────────────────────
    canonical_name = canonicalize_tool_name(call_name, tools, _policy=policy)
    if canonical_name is None:
        if telemetry and request_id:
            telemetry.emit_rejected(request_id, call_name, paths=["tool_name"])
        return RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name=call_name,
            accepted=None,
            error=f"Tool '{call_name}' not found",
        )
    tool = tool_map[canonical_name]

    if telemetry and request_id:
        telemetry.emit_tool_candidate(request_id, canonical_name, parser="repair_pipeline", raw_name=call_name)

    # ── Step 2: Argument JSON recovery ────────────────────────────────
    raw_args = call_arguments
    if raw_args is None or (isinstance(raw_args, dict) and not raw_args):
        raw_args = {}

    # Track syntax-recovery steps for provenance
    parse_steps: list[RepairStep] = []

    if not isinstance(raw_args, dict):
        syntax_recovery_enabled = (
            hasattr(policy, 'enabled_tiers')
            and RepairTier.SYNTAX_ONLY in policy.enabled_tiers
        )

        if syntax_recovery_enabled:
            # Full syntax recovery path: trailing commas, truncated JSON, etc.
            # Pass max_input_bytes from RepairPolicy into parse_tool_args.
            recovery = parse_tool_args(
                raw_args if isinstance(raw_args, str) else None,
                max_bytes=max_input,
            )
            if recovery.value is None:
                return RepairOutcome(
                    status=RepairStatus.REJECTED,
                    call_name=canonical_name,
                    accepted=None,
                    error=f"Arguments could not be parsed: {type(raw_args).__name__}",
                )
            raw_args = recovery.value
            budget.add_repair_operation()
            if recovery.status == "recovered" and recovery.steps:
                for step_name in recovery.steps:
                    parse_steps.append(RepairStep(
                        rule=step_name,
                        path="",
                        message=f"Syntax recovery: {step_name}",
                    ))
        else:
            # Exact parsing only — no syntax recovery
            parsed = _parse_args_exact(raw_args if isinstance(raw_args, str) else None)
            if parsed is None:
                return RepairOutcome(
                    status=RepairStatus.REJECTED,
                    call_name=canonical_name,
                    accepted=None,
                    error="Arguments could not be parsed (syntax recovery disabled)",
                )
            raw_args = parsed

    # Track input bytes processed (string length as proxy for bytes)
    if isinstance(raw_args, str):
        budget.add_input_bytes(len(raw_args.encode("utf-8")))
    elif isinstance(raw_args, dict):
        budget.add_input_bytes(len(str(raw_args).encode("utf-8")))

    # Check deterministic budget after parsing (latency, bytes, operations).
    # Regeneration budget is NOT checked here — it only gates regeneration
    # attempts, not validation or deterministic repair.
    if budget.is_deterministic_exhausted(
        max_ms=policy.max_added_latency_ms,
        max_input=max_input,
    ):
        return RepairOutcome(
            status=RepairStatus.REJECTED,
            call_name=canonical_name,
            accepted=None,
            error="Repair budget exhausted (latency/bytes/operations) before validation",
        )

    args = copy.deepcopy(raw_args)

    # ── Step 3: Validate against JSON Schema ──────────────────────────
    schema = tool.input_schema
    initial_issues = _sort_issues(validate_against_schema(args, schema))

    if not initial_issues:
        # If syntax recovery occurred, report REPAIRED even if schema-valid
        if parse_steps:
            return RepairOutcome(
                status=RepairStatus.REPAIRED,
                call_name=canonical_name,
                accepted=args,
                steps=parse_steps,
            )
        if telemetry and request_id:
            telemetry.emit_tool_candidate(request_id, canonical_name, parser="repair_pipeline", raw_name=call_name)
        return RepairOutcome(
            status=RepairStatus.VALID_UNCHANGED,
            call_name=canonical_name,
            accepted=args,
        )

    # ── Step 4: Proposal-based repair loop ────────────────────────────────
    # For each issue: build cursor → invoke rules → accept at most one
    # proposal → apply via path helpers → verify with recursive diff →
    # revalidate → repeat.
    enabled_tiers = policy.enabled_tiers if hasattr(policy, 'enabled_tiers') else None
    all_steps: list[RepairStep] = list(parse_steps)
    current = args
    from agent_interop.repair.paths import diff_paths
    from agent_interop.repair.rules import RepairRuleContext

    rule_context = RepairRuleContext(
        client_id=client_id,
        field_alias_policy=getattr(policy, "field_alias_policy", None),
        compatibility_key=compatibility_key,
        compatibility_verified=compatibility_verified,
    )

    # Track which paths have already been successfully repaired to prevent
    # later rules from undoing earlier fixes (item 46: strict collision handling).
    # We track the proposal's target_path (the actual mutation) rather than the
    # issue's path, because jsonschema reports required errors at the parent
    # path — multiple missing-property errors share the same empty path.
    repaired_paths: set[tuple[str | int, ...]] = set()

    max_iterations = 20
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        issues = _sort_issues(validate_against_schema(current, schema))
        if not issues:
            break

        # Pick the first unresolved issue. An issue is "resolved" if its
        # path has been directly repaired OR if the corresponding field
        # now exists and validates (item 46: collision handling).
        target_issue = None
        for issue in issues:
            issue_path = tuple(issue.path)
            # For required errors at root, the path is empty — we can't
            # use it as a unique identifier. Instead check if the field
            # has been repaired by looking at the proposal target.
            if issue_path and issue_path in repaired_paths:
                continue
            target_issue = issue
            break
        if target_issue is None:
            # All remaining issues have been attempted — stop
            break

        cursor = _build_repair_cursor(target_issue, current, schema, tool, client_id)

        fixed = False
        for rule_fn in REPAIR_RULES:
            rule_tier = getattr(rule_fn, 'tier', None)
            if enabled_tiers is not None and rule_tier is not None:
                if rule_tier not in enabled_tiers:
                    continue

            handled = getattr(rule_fn, 'handled_keywords', None)
            if handled is not None and target_issue.keyword not in handled:
                continue

            # Invoke rule with the new proposal interface
            proposal = rule_fn(current, cursor, rule_context)
            if proposal is None:
                continue

            # Apply the proposal based on operation type
            candidate = _apply_proposal(current, proposal)
            if candidate is None:
                continue

            # Verify with recursive diff: only the declared paths should change
            changed = diff_paths(current, candidate)
            allowed_paths = {tuple(proposal.target_path)}
            if proposal.source_path is not None:
                allowed_paths.add(tuple(proposal.source_path))

            illegal_changes = {
                p for p in changed
                if not any(_path_is_under(p, allowed) for allowed in allowed_paths)
            }

            if illegal_changes:
                logger.debug(
                    "rule %s proposal modified paths outside target %s: %s — rejecting",
                    proposal.rule_id, proposal.target_path, illegal_changes,
                )
                continue

            # Revalidate
            new_issues = validate_against_schema(candidate, schema)
            if len(new_issues) < len(issues):
                old_required = {tuple(i.path) for i in issues if i.keyword == 'required'}
                new_required = {tuple(i.path) for i in new_issues if i.keyword == 'required'}
                old_type = {tuple(i.path) for i in issues if i.keyword == 'type'}
                new_type = {tuple(i.path) for i in new_issues if i.keyword == 'type'}

                if not (new_required - old_required) and not (new_type - old_type):
                    current = candidate
                    # Track the target path of the successful proposal (item 46)
                    repaired_paths.add(tuple(proposal.target_path))
                    if tuple(target_issue.path):
                        repaired_paths.add(tuple(target_issue.path))
                    all_steps.append(RepairStep(
                        rule=proposal.rule_id,
                        path=".".join(str(s) for s in proposal.target_path),
                        message=proposal.message,
                    ))
                    fixed = True
                    budget.add_repair_operation()
                    break

        if not fixed:
            pass  # Issue remains in next iteration's list but won't re-trigger same rule

    # ── Step 5: Final validation ─────────────────────────────────────────
    final_issues = validate_against_schema(current, schema)

    if not final_issues:
        if telemetry and request_id:
            telemetry.emit_repaired(
                request_id, canonical_name,
                rules=[s.rule for s in all_steps],
                paths=[str(s.path) for s in all_steps],
            )
        return RepairOutcome(
            status=RepairStatus.REPAIRED,
            call_name=canonical_name,
            accepted=current,
            initial_issues=initial_issues,
            final_issues=final_issues,
            steps=all_steps,
        )

    if telemetry and request_id:
        telemetry.emit_rejected(
            request_id, canonical_name,
            paths=[str(s.path) for s in final_issues[:5]],
        )

    return RepairOutcome(
        status=RepairStatus.REJECTED,
        call_name=canonical_name,
        accepted=None,
        initial_issues=initial_issues,
        final_issues=final_issues,
        steps=all_steps,
        error=_format_issues(canonical_name, final_issues),
    )


def _apply_proposal(
    current: dict[str, Any],
    proposal: RepairProposal,
) -> dict[str, Any] | None:
    """Apply a repair proposal immutably based on its operation type.

    Returns the modified copy, or None if the proposal cannot be applied
    (e.g., source path missing or value mismatch for a MOVE).
    """
    from agent_interop.repair.paths import _MISSING, delete_at_path, get_at_path, set_at_path

    if proposal.operation is RepairOperation.MOVE:
        if proposal.source_path is None:
            return None
        source_value = get_at_path(current, proposal.source_path)
        if source_value is _MISSING:
            return None
        if source_value != proposal.after:
            return None

        candidate = set_at_path(current, proposal.target_path, source_value)
        candidate = delete_at_path(candidate, proposal.source_path)
        return candidate

    if proposal.delete:
        return delete_at_path(current, proposal.target_path)

    return set_at_path(current, proposal.target_path, proposal.after)


def _format_issues(name: str, issues: list[SchemaIssue]) -> str:
    parts = [f"{name}: {len(issues)} issue(s)"]
    for iss in issues[:5]:
        path = ".".join(str(p) for p in iss.path) if iss.path else "root"
        parts.append(f"  - {path}: {iss.message}")
    return "\n".join(parts)


# ─── Backward-compat deprecated batch wrapper (item 43) ─────────────────────

__all__ = ["RepairBudget", "RepairProposal", "extract_repair_notes", "repair_one"]


def repair_tool_calls_v2(
    calls: list[dict[str, Any]],
    tools: list[CanonicalTool],
    telemetry: Any = None,
    request_id: str = "",
    policy: Any = None,
    budget: RepairBudget | None = None,
) -> tuple[list[tuple[dict[str, Any], RepairOutcome]], list[RepairOutcome]]:
    """Process a batch of tool calls through the pipeline.

    .. deprecated::
        Use ``transaction.process_tool_batch()`` instead. This function
        exists only for backward compatibility during the v2 migration.
        It will be removed in the next major version.

    Each call dict should have ``name`` and ``arguments`` (or ``raw_arguments``).

    Returns ``(accepted_with_outcomes, rejected_outcomes)``.
    Accepted calls have their ``name`` and ``arguments`` updated.
    """
    import warnings
    warnings.warn(
        "repair_tool_calls_v2 is deprecated. Use transaction.process_tool_batch() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    accepted: list[tuple[dict[str, Any], RepairOutcome]] = []
    rejected: list[RepairOutcome] = []

    for call in calls:
        name = call.get("name", "")
        args = call.get("arguments", call.get("raw_arguments"))

        outcome = repair_one(
            name, args, tools,
            telemetry=telemetry, request_id=request_id,
            policy=policy, budget=budget,
        )

        if outcome.is_accepted and outcome.accepted is not None:
            accepted.append((
                {"name": outcome.call_name, "arguments": outcome.accepted, **{k: v for k, v in call.items() if k not in ("name", "arguments")}},
                outcome,
            ))
        else:
            rejected.append(outcome)

    return accepted, rejected
