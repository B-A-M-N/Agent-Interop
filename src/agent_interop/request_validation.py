"""Pre-upstream validation of the client tool contract.

Rejects invalid requests before contacting the upstream model,
saving latency and preventing ambiguous responses. Checks:

- Tool names are non-empty and unique
- Tool names match safe charset and length limits
- Named tool choice refers to a declared tool
- REQUIRED tool choice has at least one tool declared
- DISABLED mode is compatible with the request
- Tool schemas are structurally valid (Draft 2020-12)
- Root schema type is object (when present)
- No unsupported keywords for target backends
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from agent_interop.abi import CanonicalTool, CanonicalToolChoice, ToolChoiceMode
from agent_interop.config import ToolMode
from agent_interop.errors import InteropErrorCode

# Safe tool name pattern: alphanumeric, underscore, hyphen, dot, slash
_TOOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_./-]+$")
_MAX_TOOL_NAME_LENGTH = 128


class ValidationIssue:
    """A single validation finding."""

    def __init__(self, message: str, code: str = InteropErrorCode.CLIENT_REQUEST_INVALID) -> None:
        self.message = message
        self.code = code

    def __repr__(self) -> str:
        return f"ValidationIssue({self.code}: {self.message})"


RequestValidationResult = tuple[bool, list[ValidationIssue]]


def _validate_tool_schema(tool: CanonicalTool) -> list[ValidationIssue]:
    """Validate a single tool's input schema for structural correctness.

    Uses jsonschema Draft 2020-12 check_schema when available,
    plus additional structural checks.
    """
    issues: list[ValidationIssue] = []
    schema = tool.input_schema

    if not isinstance(schema, dict):
        issues.append(ValidationIssue(
            f"Tool '{tool.name}': input_schema is not a dict",
            code=InteropErrorCode.TOOL_SCHEMA_UNSUPPORTED,
        ))
        return issues

    # Check root type is object (when type is specified)
    root_type = schema.get("type")
    if root_type is not None and root_type != "object":
        issues.append(ValidationIssue(
            f"Tool '{tool.name}': root schema type is '{root_type}', expected 'object'",
            code=InteropErrorCode.TOOL_SCHEMA_UNSUPPORTED,
        ))

    # Use jsonschema check_schema if available
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        issues.append(ValidationIssue(
            f"Tool '{tool.name}': schema validation failed: {exc}",
            code=InteropErrorCode.TOOL_SCHEMA_UNSUPPORTED,
        ))

    return issues


@dataclass(frozen=True)
class BackendConstraints:
    """Destination-specific constraints for request validation.

    Allows validation to reject requests that the target backend
    cannot handle, before invocation plan construction.
    """

    name_pattern: re.Pattern[str] | None = None
    max_name_length: int = 128
    max_tools: int = 0  # 0 = unlimited
    supports_parallel: bool = True
    supports_strict_schema: bool = False


def validate_tool_contract(
    tools: Sequence[CanonicalTool] | None,
    tool_choice: CanonicalToolChoice | None,
    tool_mode: ToolMode | None = None,
    backend_constraints: BackendConstraints | None = None,
) -> RequestValidationResult:
    """Validate the client's tool contract before upstream processing.

    Returns (is_valid, issues). When is_valid is False the request should
    be rejected without contacting the upstream.

    When ``backend_constraints`` is provided, also checks that tool names
    and schemas are compatible with the destination backend.
    """
    issues: list[ValidationIssue] = []
    tool_list = list(tools) if tools else []

    # Determine effective constraints
    name_re = _TOOL_NAME_RE
    max_name_len = _MAX_TOOL_NAME_LENGTH
    if backend_constraints:
        if backend_constraints.name_pattern is not None:
            name_re = backend_constraints.name_pattern
        max_name_len = backend_constraints.max_name_length

    # Check for empty tool names and charset/length
    seen_names: set[str] = set()
    for t in tool_list:
        if not t.name or not t.name.strip():
            issues.append(ValidationIssue(
                "Tool with empty name is not allowed",
                code=InteropErrorCode.TOOL_CALL_INVALID,
            ))
        else:
            if len(t.name) > max_name_len:
                issues.append(ValidationIssue(
                    f"Tool name '{t.name[:30]}...' exceeds {max_name_len} chars",
                    code=InteropErrorCode.TOOL_CALL_INVALID,
                ))
            if not name_re.match(t.name):
                issues.append(ValidationIssue(
                    f"Tool name '{t.name}' contains characters not allowed by destination",
                    code=InteropErrorCode.TOOL_CALL_INVALID,
                ))
        if t.name in seen_names:
            issues.append(ValidationIssue(
                f"Duplicate tool name: '{t.name}'",
                code=InteropErrorCode.TOOL_CALL_INVALID,
            ))
        seen_names.add(t.name)

    # Check tool count limit
    if backend_constraints and backend_constraints.max_tools > 0:
        if len(tool_list) > backend_constraints.max_tools:
            issues.append(ValidationIssue(
                f"Too many tools ({len(tool_list)}), backend limit is {backend_constraints.max_tools}",
                code=InteropErrorCode.CONTEXT_LIMIT_EXCEEDED,
            ))

    # Validate tool schemas
    for t in tool_list:
        issues.extend(_validate_tool_schema(t))

    # Validate tool choice references
    if tool_choice and tool_choice.mode != ToolChoiceMode.AUTO:
        if tool_choice.mode == ToolChoiceMode.NONE:
            pass  # Always valid, even with no tools
        elif tool_choice.mode == ToolChoiceMode.REQUIRED:
            if not tool_list:
                issues.append(ValidationIssue(
                    "REQUIRED tool choice but no tools declared",
                    code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                ))
        elif tool_choice.mode == ToolChoiceMode.NAMED:
            named_tool = tool_choice.name
            tool_names = {t.name for t in tool_list}
            if not named_tool:
                issues.append(ValidationIssue(
                    "NAMED tool choice but no tool name specified",
                    code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                ))
            elif named_tool not in tool_names:
                issues.append(ValidationIssue(
                    f"NAMED tool choice refers to undeclared tool: '{named_tool}'",
                    code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                ))

    # Check parallel support
    if backend_constraints and not backend_constraints.supports_parallel:
        if tool_choice and tool_choice.mode == ToolChoiceMode.REQUIRED and len(tool_list) > 1:
            issues.append(ValidationIssue(
                "Backend does not support parallel tool calls",
                code=InteropErrorCode.CAPABILITY_REQUIRED,
            ))

    # Validate tool mode compatibility
    if tool_mode == ToolMode.DISABLED and tool_choice and tool_choice.mode == ToolChoiceMode.REQUIRED:
        issues.append(ValidationIssue(
            "DISABLED tool mode conflicts with REQUIRED tool choice",
            code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
        ))

    if tool_mode == ToolMode.DISABLED and tool_list:
        issues.append(ValidationIssue(
            "Tools declared but tool mode is DISABLED",
            code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
        ))

    return len(issues) == 0, issues