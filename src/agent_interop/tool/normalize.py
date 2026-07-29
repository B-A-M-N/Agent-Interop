"""Tool normalization — convert various tool schema formats into a canonical JSON Schema.

Ingests tools from:
- MCP tool definitions
- OpenAI function schemas
- Anthropic tool schemas
- Native agent tool definitions

Schema ingestion is LOSSLESS by default. Backend-specific schema reduction
returns a SchemaProjectionResult with dropped features and warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import CanonicalTool


@dataclass(frozen=True)
class SchemaProjectionResult:
    """Result of projecting a schema for a specific backend.

    Contains the projected schema, the list of dropped features, and
    warnings about potential semantic changes.
    """

    projected_schema: dict[str, Any] = field(default_factory=dict)
    dropped_features: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

# ─── Known tool schemas from major formats ─────────────────────────────────


def from_mcp(tool: dict[str, Any]) -> CanonicalTool:
    """Convert an MCP tool definition to CanonicalTool.

    MCP tool format:
    {
        "name": "tool-name",
        "description": "...",
        "inputSchema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
    """
    name = tool.get("name", "")
    description = tool.get("description", "")
    schema = tool.get("inputSchema", tool.get("input_schema", {
        "type": "object",
        "properties": {},
    }))

    return CanonicalTool(
        name=name,
        description=description,
        input_schema=_normalize_schema(schema),
    )


def from_openai(spec: dict[str, Any]) -> CanonicalTool:
    """Convert an OpenAI function spec.

    OpenAPI format:
    {
        "type": "function",
        "function": {
            "name": "...",
            "description": "...",
            "parameters": {...}
        }
    }
    or standalone:
    {
        "name": "...",
        "description": "...",
        "parameters": {...}
    }
    """
    fn = spec.get("function", spec)

    return CanonicalTool(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        input_schema=_normalize_schema(fn.get("parameters", {
            "type": "object",
            "properties": {},
        })),
        strict=spec.get("strict", False),
    )


def from_anthropic(spec: dict[str, Any]) -> CanonicalTool:
    """Convert an Anthropic tool spec.

    Anthropic format:
    {
        "name": "...",
        "description": "...",
        "input_schema": {...}
    }
    """
    return CanonicalTool(
        name=spec.get("name", ""),
        description=spec.get("description", ""),
        input_schema=_normalize_schema(spec.get("input_schema", {
            "type": "object",
            "properties": {},
        })),
    )


def from_any(spec: dict[str, Any]) -> CanonicalTool:
    """Auto-detect tool format and convert."""
    # Check OpenAI format first (has "function" key)
    if "function" in spec and isinstance(spec["function"], dict):
        return from_openai(spec)
    # Then check MCP format (has "inputSchema")
    if "inputSchema" in spec:
        return from_mcp(spec)
    # Then check Anthropic format (has "input_schema" at top level)
    if "input_schema" in spec:
        return from_anthropic(spec)
    # Fallback — assume OpenAI-like
    return from_openai(spec)


# ─── Schema normalization ────────────────────────────────────────────────────


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema to a LOSSLESS canonical form.

    Preserves ALL JSON Schema keywords:
    - $ref, $defs, $schema
    - oneOf, anyOf, allOf, not
    - unevaluatedProperties, patternProperties
    - dependentRequired, dependentSchemas
    - annotations (title, description, default, examples)
    - nested constraints (minLength, pattern, etc.)

    Only ensures the root has type: "object" if not specified.
    """
    if not schema or not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    # Start with a complete copy to preserve all keywords
    normalized: dict[str, Any] = dict(schema)

    # Ensure root type is object (only if not already specified)
    if "type" not in normalized:
        normalized["type"] = "object"

    # Ensure properties exists
    if "properties" not in normalized:
        normalized["properties"] = {}

    return normalized


def to_strict_schema(tool: CanonicalTool) -> dict[str, Any]:
    """Convert to a strict JSON Schema suitable for constrained decoding.

    Ensures all properties have explicit types and no empty schemas.
    """
    schema = dict(tool.input_schema)
    props = schema.get("properties", {})
    for prop in props.values():
        if isinstance(prop, dict):
            if "type" not in prop:
                prop["type"] = "string"
            if prop.get("type") == "any":
                prop["type"] = "string"
            # Ensure enum is present when it should be
            if "enum" not in prop and prop.get("type") == "string":
                pass  # not all strings need enum

    return schema


def to_mcp(tool: CanonicalTool) -> dict[str, Any]:
    """Convert CanonicalTool to MCP format."""
    return {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    }


def project_schema_for_backend(
    tool: CanonicalTool,
    backend_kind: str,
) -> SchemaProjectionResult:
    """Project a canonical schema for a specific backend.

    Returns the projected schema plus dropped features and warnings.
    This is the ONLY place that should drop schema features.
    """
    schema = dict(tool.input_schema)
    dropped: list[str] = []
    warnings: list[str] = []

    if backend_kind in ("ollama", "llamacpp"):
        # These backends may not support advanced JSON Schema features
        for key in ("$ref", "$defs", "oneOf", "anyOf", "allOf",
                     "unevaluatedProperties", "patternProperties",
                     "dependentRequired", "dependentSchemas"):
            if key in schema:
                del schema[key]
                dropped.append(key)

        if dropped:
            dropped_str = ", ".join(dropped)
            warnings.append(
                f"Backend '{backend_kind}' dropped {len(dropped)} "
                f"schema features: {dropped_str}"
            )

    return SchemaProjectionResult(
        projected_schema=schema,
        dropped_features=tuple(dropped),
        warnings=tuple(warnings),
    )


def to_openai(tool: CanonicalTool) -> dict[str, Any]:
    """Convert to OpenAI function schema."""
    return tool.to_json_schema()


def to_anthropic(tool: CanonicalTool) -> dict[str, Any]:
    """Convert to Anthropic tool schema."""
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }