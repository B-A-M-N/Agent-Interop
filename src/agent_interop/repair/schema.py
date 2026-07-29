"""Pure JSON Schema validation for tool arguments.

Uses the ``jsonschema`` library for full draft-2020-12 compliance so that
$ref, $defs, oneOf, anyOf, enum, const, additionalProperties, nested
objects/arrays, and numeric limits are all honored — rather than the
hand-rolled partial check in the old validator.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from agent_interop.repair.types import SchemaIssue


def validate_against_schema(
    instance: Any,
    schema: dict[str, Any],
) -> list[SchemaIssue]:
    """Validate ``instance`` against ``schema``.

    Returns a list of :class:`SchemaIssue`. An empty list means the instance
    is valid. Each issue carries the JSON-path where the error occurred so
    that repair rules can target it precisely.
    """
    if not schema or not isinstance(schema, dict):
        return []

    # Reject empty schemas that carry no contract.
    if not schema.get("properties") and not schema.get("required") and schema.get("type", "object") == "object":
        # A bare {"type": "object"} with no constraints — everything is valid.
        return []

    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
    except Exception as exc:
        # Invalid schema — return a schema issue so the caller knows validation
        # could not be performed. Do NOT return [] (which means "valid").
        return [SchemaIssue(
            path=[],
            keyword="invalid_schema",
            message=f"Tool schema is not valid JSON Schema: {exc}",
        )]

    issues: list[SchemaIssue] = []
    try:
        errors_iter = validator.iter_errors(instance)
        for error in errors_iter:
            path = tuple(error.absolute_path)
            keyword = error.validator
            expected = ""
            actual = ""

            # Capture the schema-side path for nested/branched schema navigation
            schema_path = list(error.absolute_schema_path) if error.absolute_schema_path else []

            # Extract expected/actual from the error where possible.
            if error.validator_value is not None:
                expected = _truncate(str(error.validator_value))
            if error.instance is not None:
                actual = _truncate(_type_name(error.instance))

            message = error.message

            # Rewrite the most common jsonschema messages into a compact,
            # model-readable form. We keep keyword so repair rules can branch.
            if keyword == "required":
                # error.message: "'old_string' is a required property"
                # Extract the specific missing property from the message.
                missing = error.validator_value
                if isinstance(missing, list) and missing:
                    # jsonschema reports one error per missing property.
                    # The message tells us which one: "'prop' is a required property"
                    import re
                    match = re.search(r"'([^']+)' is a required property", error.message)
                    if match:
                        missing_prop = match.group(1)
                        message = f"missing required property: {missing_prop}"
                        expected = f"['{missing_prop}']"
                    else:
                        # Fallback: join all (legacy behavior)
                        message = f"missing required property: {', '.join(missing)}"
                        expected = str(missing)
                else:
                    message = f"missing required property: {missing}"
                    expected = str(missing)
            elif keyword == "type":
                # error.message: "123 is not of type 'string'"
                message = f"expected {error.validator_value}, got {actual}"

            issues.append(SchemaIssue(
                path=list(path),
                keyword=keyword or "unknown",
                message=message,
                expected=expected,
                actual=actual,
                absolute_schema_path=schema_path,
            ))
    except Exception as exc:
        # Runtime schema resolution errors ($ref, etc.) — fail closed
        return [SchemaIssue(
            path=[],
            keyword="unresolvable_schema",
            message=f"Schema validation could not complete: {exc}",
        )]

    return issues


def _type_name(value: Any) -> str:
    """Map a Python value to the JSON Schema type name it would satisfy."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _truncate(s: str, limit: int = 80) -> str:
    return s if len(s) <= limit else s[: limit - 1] + "…"
