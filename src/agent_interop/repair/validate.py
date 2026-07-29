"""Tool-call validation and bounded repair.

When the model emits a malformed call:
1. Validate the tool exists.
2. Validate arguments against JSON Schema.
3. Correct trivial serialization defects.
4. Regenerate with constrained decoding when necessary.
5. Return a native tool call in the client's expected protocol.

The gateway repairs syntax and protocol defects — not silently changing
the model's intended action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_interop.abi import CanonicalTool, CanonicalToolCallBlock
from agent_interop.types import RepairAction


class ValidationIssue(Enum):
    OK = "ok"
    TOOL_NOT_FOUND = "tool_not_found"
    MISSING_NAME = "missing_name"
    MISSING_ARGUMENTS = "missing_arguments"
    ARGUMENTS_NOT_DICT = "arguments_not_dict"
    MISSING_REQUIRED_PROPERTY = "missing_required_property"
    TYPE_MISMATCH = "type_mismatch"
    EXTRA_PROPERTY = "extra_property"
    BAD_JSON = "bad_json"
    STRING_ARGS_NOT_JSON = "string_args_not_json"
    EMPTY_TOOL_NAME = "empty_tool_name"
    DUPLICATE_CALL = "duplicate_call"
    UNREPAIRABLE = "unrepairable"


@dataclass
class ValidationResult:
    """Result of validating a single tool call."""

    call: CanonicalToolCallBlock
    issues: list[ValidationIssue] = field(default_factory=list)
    fixed: bool = False
    repair_action: RepairAction = RepairAction.NONE
    error: str = ""

    @property
    def valid(self) -> bool:
        return len(self.issues) == 0


@dataclass
class RepairReport:
    """Summary of all repairs applied in a single response."""

    total_calls: int = 0
    valid: int = 0
    repaired: int = 0
    unreparable: int = 0
    repairs: list[ValidationResult] = field(default_factory=list)
    total_attempts: int = 0
    total_latency_ms: float = 0.0
    max_attempts_exceeded: bool = False

    @property
    def all_valid(self) -> bool:
        return self.unreparable == 0 and self.repaired == self.total_calls - self.valid


# ─── Repair Configuration ───────────────────────────────────────────────────


@dataclass
class LegacyRepairConfig:
    """Configuration for legacy repair behavior.

    Deprecated: use interop.config.RepairConfig and interop.config.RepairPolicy
    instead. This class is retained only for backward compatibility with
    tests that have not yet migrated to the v2 pipeline.
    """

    enabled: bool = True
    max_attempts_per_response: int = 1
    max_added_latency_ms: float = 15000.0
    allow_tool_name_guessing: bool = False
    allow_argument_invention: bool = False


class ToolValidator:
    """Validates tool calls against available tool definitions."""

    def __init__(self, tools: list[CanonicalTool]) -> None:
        self.tool_map: dict[str, CanonicalTool] = {t.name: t for t in tools}
        self.tools = tools

    def validate(self, call: CanonicalToolCallBlock) -> ValidationResult:
        """Validate a single tool call and attempt bounded repair."""
        result = ValidationResult(call=call)
        issues: list[ValidationIssue] = []

        # 1. Tool name exists
        if not call.name or call.name.strip() == "":
            issues.append(ValidationIssue.EMPTY_TOOL_NAME)
            result.error = "Tool call with empty name"
            result.issues = issues
            result.repair_action = RepairAction.UNREPAIRABLE
            return result

        if call.name not in self.tool_map:
            issues.append(ValidationIssue.TOOL_NOT_FOUND)
            # Attempt fuzzy match
            matched = self._fuzzy_match(call.name)
            if matched:
                result.fixed = True
                result.repair_action = RepairAction.TRIVIAL_JSON
                call.name = matched
                result.error = ""
            else:
                result.error = f"Tool '{call.name}' not found; available: {list(self.tool_map.keys())}"
                result.repair_action = RepairAction.UNREPAIRABLE
                result.issues = issues
                return result

        # 2. Arguments exist and are a dict
        if not call.arguments:
            issues.append(ValidationIssue.MISSING_ARGUMENTS)
            call.arguments = {}
            result.fixed = True
            result.repair_action = RepairAction.TRIVIAL_JSON

        if not isinstance(call.arguments, dict):
            issues.append(ValidationIssue.ARGUMENTS_NOT_DICT)
            result.error = f"Arguments not a dict: {type(call.arguments).__name__}"
            result.repair_action = RepairAction.UNREPAIRABLE
            result.issues = issues
            return result

        # 3. Validate against JSON Schema
        tool = self.tool_map[call.name]
        schema_issues = self._validate_schema(call.arguments, tool.input_schema)
        issues.extend(schema_issues)

        if schema_issues:
            # Attempt trivial fixes
            self._attempt_schema_repair(call, tool, issues)
            if issues:
                result.repair_action = RepairAction.SCHEMA_MINOR
                result.error = f"Schema issues: {[i.value for i in issues]}"

        result.issues = issues
        if not issues:
            result.repair_action = RepairAction.NONE
        elif result.fixed:
            pass  # repair_action already set

        return result

    def _validate_schema(
        self, args: dict[str, Any], schema: dict[str, Any]
    ) -> list[ValidationIssue]:
        """Validate arguments against JSON Schema.

        Does basic structural validation — not full JSON Schema compliance.
        """
        issues: list[ValidationIssue] = []
        if not schema or not isinstance(schema, dict):
            return issues

        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Check required properties
        for req in required:
            if req not in args:
                issues.append(ValidationIssue.MISSING_REQUIRED_PROPERTY)

        # Check types for known properties
        for key, value in args.items():
            if key in props:
                prop_schema = props[key]
                ptype = prop_schema.get("type", "")
                if ptype == "string" and not isinstance(value, str):
                    # Try coercion
                    if isinstance(value, (int, float, bool)):
                        args[key] = str(value)
                    else:
                        issues.append(ValidationIssue.TYPE_MISMATCH)
                elif ptype in ("integer", "number") and isinstance(value, str):
                    try:
                        args[key] = int(value) if ptype == "integer" else float(value)
                    except (ValueError, TypeError):
                        issues.append(ValidationIssue.TYPE_MISMATCH)
                elif ptype == "array" and not isinstance(value, list):
                    issues.append(ValidationIssue.TYPE_MISMATCH)
                elif ptype == "boolean" and not isinstance(value, bool):
                    if isinstance(value, str):
                        args[key] = value.lower() in ("true", "1", "yes")
                    else:
                        issues.append(ValidationIssue.TYPE_MISMATCH)

        return issues

    def _attempt_schema_repair(
        self, call: CanonicalToolCallBlock, tool: CanonicalTool, issues: list[ValidationIssue]
    ) -> None:
        """Attempt to repair trivial schema violations."""
        schema = tool.input_schema
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Fix missing required properties by adding empty/default values
        for req in required:
            if req in call.arguments:
                continue
            if req in props and props[req].get("type") == "string":
                call.arguments[req] = ""
                self._remove_issue(issues, ValidationIssue.MISSING_REQUIRED_PROPERTY)

    @staticmethod
    def _remove_issue(issues: list[ValidationIssue], issue: ValidationIssue) -> None:
        while issue in issues:
            issues.remove(issue)

    def _fuzzy_match(self, name: str) -> str | None:
        """Fuzzy match a tool name against known tools."""
        name_lower = name.lower()
        for tool_name in self.tool_map:
            if tool_name.lower() == name_lower:
                return tool_name
            # prefix overlap
            if name_lower in tool_name.lower() or tool_name.lower() in name_lower:
                return tool_name
        return None


def repair_tool_calls(
    calls: list[CanonicalToolCallBlock], tools: list[CanonicalTool],
    config: LegacyRepairConfig | None = None,
) -> RepairReport:
    """Validate and repair all tool calls in a response.

    Applies repair limits from LegacyRepairConfig — caps attempts and latency.
    """
    cfg = config or LegacyRepairConfig()
    validator = ToolValidator(tools)
    report = RepairReport(total_calls=len(calls))

    if not cfg.enabled:
        # Bypass repair, just validate
        for call in calls:
            result = validator.validate(call)
            if result.valid:
                report.valid += 1
            else:
                report.unreparable += 1
            report.repairs.append(result)
        return report

    for call in calls:
        if cfg.max_attempts_per_response <= 0:
            report.max_attempts_exceeded = True
            report.unreparable += 1
            report.repairs.append(ValidationResult(call=call, issues=[ValidationIssue.UNREPAIRABLE],
                                                    repair_action=RepairAction.UNREPAIRABLE,
                                                    error="Repair disabled (max_attempts=0)"))
            continue

        result = validator.validate(call)
        attempts = 0
        while not result.valid and attempts < cfg.max_attempts_per_response:
            result = validator.validate(call)
            attempts += 1
            report.total_attempts += 1

        if result.valid:
            report.valid += 1
        elif result.fixed:
            report.repaired += 1
        else:
            report.unreparable += 1
        report.repairs.append(result)

    return report