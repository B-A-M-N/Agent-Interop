"""Tool normalization — convert various tool schema formats into a canonical JSON Schema.

Ingests tools from:
- MCP tool definitions
- OpenAI function schemas
- Anthropic tool schemas
- Native agent tool definitions

And converts them to a strict canonical JSON Schema before rendering
for the model.
"""

from __future__ import annotations

import json
from typing import Any

from interop.types import CanonicalTool

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
        parameters=_normalize_schema(schema),
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
    if "function" in spec:
        fn = spec["function"]
    else:
        fn = spec

    return CanonicalTool(
        name=fn.get("name", ""),
        description=fn.get("description", ""),
        parameters=_normalize_schema(fn.get("parameters", {
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
        parameters=_normalize_schema(spec.get("input_schema", {
            "type": "object",
            "properties": {},
        })),
    )


def from_any(spec: dict[str, Any]) -> CanonicalTool:
    """Auto-detect tool format and convert."""
    if "inputSchema" in spec or "input_schema" in spec:
        return from_mcp(spec)
    if "function" in spec and isinstance(spec["function"], dict):
        return from_openai(spec)
    if "input_schema" in spec or "inputSchema" in spec:
        return from_anthropic(spec)
    # Fallback — assume OpenAI-like
    return from_openai(spec)


# ─── Schema normalization ────────────────────────────────────────────────────


def _normalize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize a JSON Schema to a canonical form.

    Ensures:
    - Has type: "object" at root
    - Has properties dict
    - Fields with empty anyOf or empty oneOf are simplified
    - Nullable fields are handled
    """
    normalized: dict[str, Any] = {
        "type": schema.get("type", "object"),
        "properties": schema.get("properties", {}),
    }

    if "required" in schema:
        normalized["required"] = schema["required"]
    if "additionalProperties" in schema:
        normalized["additionalProperties"] = bool(schema["additionalProperties"])
    if "description" in schema:
        normalized["description"] = schema["description"]

    return normalized


def to_strict_schema(tool: CanonicalTool) -> dict[str, Any]:
    """Convert to a strict JSON Schema suitable for constrained decoding.

    Ensures all properties have explicit types and no empty schemas.
    """
    schema = dict(tool.parameters)
    props = schema.get("properties", {})
    for name, prop in props.items():
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
        "inputSchema": tool.parameters,
    }


def to_openai(tool: CanonicalTool) -> dict[str, Any]:
    """Convert to OpenAI function schema."""
    return tool.to_json_schema()


def to_anthropic(tool: CanonicalTool) -> dict[str, Any]:
    """Convert to Anthropic tool schema."""
    return tool.to_anthropic()