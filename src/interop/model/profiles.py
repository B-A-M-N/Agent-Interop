"""Model profile registry — declarative model definitions and capability probes.

Each supported model gets a profile that specifies:
- The chat template/style to use
- The tool-call parser for extracting tool calls from model output
- The reasoning parser for extracting thinking blocks
- Known capabilities and limitations
- Repair strategies for malformed output
"""

from __future__ import annotations

import json
import re
from typing import Any

from interop.types import (
    CapabilityLevel,
    ModelProfile,
    ToolCall,
    ToolCallDialect,
    RepairAction,
)


# ─── Built-in profiles ──────────────────────────────────────────────────────

BUILTIN_PROFILES: dict[str, ModelProfile] = {}


def _register(p: ModelProfile) -> ModelProfile:
    BUILTIN_PROFILES[p.model] = p
    return p


# Qwen family
_register(ModelProfile(
    model="qwen3-coder",
    template="qwen3-tool",
    tool_parser="qwen3_coder",
    reasoning_parser="qwen3",
    tool_dialect=ToolCallDialect.QWEN,
    context_length=131072,
    capabilities=CapabilityLevel.L3,
    parallel_tools=True,
    repair_strategies={
        "malformed_tool_call": "constrained_regeneration",
        "missing_tool_id": "synthesize",
    },
))

_register(ModelProfile(
    model="qwen3-235b-a100b",
    template="qwen3-tool",
    tool_parser="qwen3",
    reasoning_parser="qwen3",
    tool_dialect=ToolCallDialect.QWEN,
    context_length=32768,
    capabilities=CapabilityLevel.L3,
    parallel_tools=True,
))

_register(ModelProfile(
    model="qwen2.5-coder",
    template="qwen2.5",
    tool_parser="qwen2.5_coder",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.QWEN,
    context_length=131072,
    capabilities=CapabilityLevel.L2,
    parallel_tools=False,
))

_register(ModelProfile(
    model="qwen2.5-7b-instruct",
    template="qwen2.5",
    tool_parser="qwen2.5",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.QWEN,
    context_length=32768,
    capabilities=CapabilityLevel.L2,
    parallel_tools=False,
))


# Mistral family
_register(ModelProfile(
    model="mistral-small-2509",
    template="mistral-v3",
    tool_parser="mistral",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.MISTRAL,
    context_length=32768,
    capabilities=CapabilityLevel.L2,
    parallel_tools=True,
    repair_strategies={
        "missing_tool_id": "synthesize",
    },
))

_register(ModelProfile(
    model="mistral-7b",
    template="mistral",
    tool_parser="mistral",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.MISTRAL,
    context_length=8192,
    capabilities=CapabilityLevel.L1,
    parallel_tools=False,
))


# Llama family
_register(ModelProfile(
    model="llama-4-scout",
    template="llama-4",
    tool_parser="llama",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.LLAMA,
    context_length=262144,
    capabilities=CapabilityLevel.L3,
    parallel_tools=True,
    supports_images=True,
))

_register(ModelProfile(
    model="llama-3.3-70b-instruct",
    template="llama-3.1",
    tool_parser="llama",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.LLAMA,
    context_length=131072,
    capabilities=CapabilityLevel.L2,
    parallel_tools=False,
))

_register(ModelProfile(
    model="llama-3.1-8b-instruct",
    template="llama-3.1",
    tool_parser="llama",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.LLAMA,
    context_length=131072,
    capabilities=CapabilityLevel.L2,
    parallel_tools=False,
))


# Hermes family
_register(ModelProfile(
    model="hermes-3-llama-3.1-405b",
    template="hermes",
    tool_parser="hermes",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.HERMES,
    context_length=131072,
    capabilities=CapabilityLevel.L4,
    parallel_tools=True,
    repair_strategies={
        "malformed_tool_call": "reparse",
    },
))

_register(ModelProfile(
    model="hermes-3-llama-3.1-8b",
    template="hermes",
    tool_parser="hermes",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.HERMES,
    context_length=8192,
    capabilities=CapabilityLevel.L2,
    parallel_tools=True,
))


# DeepSeek
_register(ModelProfile(
    model="deepseek-v4-0324",
    template="deepseek-v4",
    tool_parser="deepseek",
    reasoning_parser="deepseek",
    tool_dialect=ToolCallDialect.DEEPSEEK,
    context_length=131072,
    capabilities=CapabilityLevel.L4,
    parallel_tools=True,
    supports_thinking=True,
))

_register(ModelProfile(
    model="deepseek-r1",
    template="deepseek-r1",
    tool_parser="deepseek",
    reasoning_parser="deepseek",
    tool_dialect=ToolCallDialect.DEEPSEEK,
    context_length=16384,
    capabilities=CapabilityLevel.L3,
    parallel_tools=True,
    supports_thinking=True,
))


# Generic fallback profiles
_register(ModelProfile(
    model="gpt-4o-mini-compat",
    template="openai",
    tool_parser="openai",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.OPENAI_NATIVE,
    context_length=128000,
    capabilities=CapabilityLevel.L3,
    parallel_tools=True,
))

_register(ModelProfile(
    model="generic-fallback",
    template="generic-json",
    tool_parser="generic",
    reasoning_parser=None,
    tool_dialect=ToolCallDialect.GENERIC_JSON,
    context_length=4096,
    capabilities=CapabilityLevel.L1,
    parallel_tools=False,
    repair_strategies={
        "malformed_tool_call": "reparse",
    },
))


# ─── Profile lookup ─────────────────────────────────────────────────────────

def get_profile(model_name: str) -> ModelProfile | None:
    """Look up a profile by model name, supporting fuzzy matching."""
    model_lower = model_name.lower().strip()

    # Exact match first
    if model_lower in BUILTIN_PROFILES:
        return BUILTIN_PROFILES[model_lower]

    # Try prefix matching (e.g. "qwen3-coder:latest")
    for key, profile in BUILTIN_PROFILES.items():
        if model_lower.startswith(key):
            return profile

    # Try partial match (e.g. "llama-3.1-8b" matches "llama-3.1-8b-instruct")
    for key, profile in BUILTIN_PROFILES.items():
        if key in model_lower or model_lower in key:
            return profile

    return None


def list_profiles() -> dict[str, ModelProfile]:
    """Return all registered profiles."""
    return dict(BUILTIN_PROFILES)


# ─── Tool call parsers ──────────────────────────────────────────────────────

_HERMES_TOOL_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>",
    re.DOTALL,
)

_QWEN_TOOL_RE = re.compile(
    r"<tool>\n(.*?)\n</tool>",
    re.DOTALL,
)

_DEEPSEEK_TOOL_RE = re.compile(
    r"\u0014(.*?)\u0014",
    re.DOTALL,
)

_MISTRAL_TOOL_RE = re.compile(
    r"\[TOOL_CALLS\](.*?)(?:\n|$)",
    re.DOTALL,
)

# Match any JSON object that looks like a tool call — must have name/function/tool key
_GENERIC_JSON_RE = re.compile(
    r'\{(?:[^{}]|(?:\{[^{}]*\}))*"(?:name|function|tool)"\s*:\s*"[^"]*"(?:[^{}]|(?:\{[^{}]*\}))*\}|\{"function"\s*:\s*\{[^}]+\}(?:[^{}]|(?:\{[^{}]*\}))*\}',
    re.DOTALL,
)


def parse_tool_calls(text: str, dialect: str | ToolCallDialect) -> list[ToolCall]:
    """Extract tool calls from raw model output text.

    Returns a list of ToolCall objects. Each call gets a unique ID.
    """
    if isinstance(dialect, str):
        try:
            dialect = ToolCallDialect(dialect)
        except ValueError:
            dialect = ToolCallDialect.GENERIC_JSON

    dial_to_fn = {
        ToolCallDialect.HERMES: _parse_hermes,
        ToolCallDialect.QWEN: _parse_qwen,
        ToolCallDialect.MISTRAL: _parse_mistral,
        ToolCallDialect.DEEPSEEK: _parse_deepseek,
        ToolCallDialect.LLAMA: _parse_llama,
        ToolCallDialect.OPENAI_NATIVE: _parse_openai,
        ToolCallDialect.GENERIC_JSON: _parse_generic_json,
    }

    parser = dial_to_fn.get(dialect, _parse_generic_json)
    return parser(text)


def _parse_hermes(text: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for idx, match in enumerate(_HERMES_TOOL_RE.finditer(text)):
        block = match.group(1).strip()
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls = _normalize_tool_json(data)
        for call in calls:
            call.dialect = ToolCallDialect.HERMES
            call.raw = block
            if not call.id:
                call.id = f"hermes_tc_{idx}"
        results.extend(calls)
    return results


def _parse_qwen(text: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for idx, match in enumerate(_QWEN_TOOL_RE.finditer(text)):
        block = match.group(1).strip()
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls = _normalize_tool_json(data)
        for call in calls:
            call.dialect = ToolCallDialect.QWEN
            call.raw = block
            if not call.id:
                call.id = f"qwen_tc_{idx}"
        results.extend(calls)
    return results


def _parse_deepseek(text: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for idx, match in enumerate(_DEEPSEEK_TOOL_RE.finditer(text)):
        block = match.group(1).strip()
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls = _normalize_tool_json(data)
        for call in calls:
            call.dialect = ToolCallDialect.DEEPSEEK
            call.raw = block
            if not call.id:
                call.id = f"deepseek_tc_{idx}"
        results.extend(calls)
    return results


def _parse_mistral(text: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for idx, match in enumerate(_MISTRAL_TOOL_RE.finditer(text)):
        block = match.group(1).strip()
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        calls = _normalize_tool_json(data)
        for call in calls:
            call.dialect = ToolCallDialect.MISTRAL
            call.raw = block
            if not call.id:
                call.id = f"mistral_tc_{idx}"
        results.extend(calls)
    return results


def _parse_llama(text: str) -> list[ToolCall]:
    """Llama models output tool calls via built-in API tool_calls field;
    this handles the case where they emit <|python_tag|> syntax as fallback."""
    results: list[ToolCall] = []
    text_parts = text.split("<|python_tag|>")
    for idx, part in enumerate(text_parts):
        part = part.strip()
        if not part:
            continue
        # Try extracting function(lambda ...) calls
        f_match = re.search(r"\{[^}]+\}", part)
        if f_match:
            try:
                data = json.loads(f_match.group(0))
                calls = _normalize_tool_json(data)
                for call in calls:
                    call.dialect = ToolCallDialect.LLAMA
                    call.raw = part
                    if not call.id:
                        call.id = f"llama_tc_{idx}"
                results.extend(calls)
            except json.JSONDecodeError:
                pass
    return results


def _parse_openai(text: str) -> list[ToolCall]:
    """Parse OpenAI-style JSON tool calls embedded in text.
    Typically { "function": "name", "arguments": {...} }
    or { "name": "...", "arguments": "..." }
    """
    results: list[ToolCall] = []
    for idx, match in enumerate(_GENERIC_JSON_RE.finditer(text)):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        calls = _normalize_tool_json(data)
        for call in calls:
            call.dialect = ToolCallDialect.OPENAI_NATIVE
            call.raw = match.group(0)
            if not call.id:
                call.id = f"openai_tc_{idx}"
        results.extend(calls)
    return results


def _parse_generic_json(text: str) -> list[ToolCall]:
    """Heuristic: find any JSON object that looks like a tool call."""
    results: list[ToolCall] = []
    for idx, match in enumerate(_GENERIC_JSON_RE.finditer(text)):
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        calls = _normalize_tool_json(data)
        for call in calls:
            call.dialect = ToolCallDialect.GENERIC_JSON
            call.raw = match.group(0)
            if not call.id:
                call.id = f"generic_tc_{idx}"
        results.extend(calls)
    return results


# ─── Normalization helpers ──────────────────────────────────────────────────


def _normalize_tool_json(data: dict[str, Any]) -> list[ToolCall]:
    """Normalize various tool call JSON shapes into ToolCall objects.

    Acceptable shapes:
      - {"name": "x", "arguments": {...} or "..."}
      - {"function": "x", "arguments": {...} or "..."}
      - {"function": {"name": "x", "arguments": {...}}}
      - {"tool": "x", "input": {...}}
      - [{"name": "x", ...}, ...]
      - {"name": "x", "parameters": {...}}
    """
    if isinstance(data, list):
        calls: list[ToolCall] = []
        for item in data:
            calls.extend(_normalize_tool_json(item))
        return calls

    calls = []
    name = ""
    arguments: dict[str, Any] = {}

    if "function" in data and isinstance(data["function"], dict):
        # {"function": {"name": "x", "arguments": {...}}}
        fn = data["function"]
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        arguments = dict(args)

    elif "function" in data:
        # {"function": "x", "arguments": {...}}
        name = str(data["function"])
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                args = {"raw": args}
        arguments = dict(args)

    elif "name" in data:
        name = data.get("name", "")
        for key in ("arguments", "parameters", "input", "params"):
            if key in data:
                args = data[key]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args}
                arguments = dict(args)
                break

    elif "tool" in data:
        name = data.get("tool", "")
        for key in ("input", "arguments", "parameters"):
            if key in data:
                args = data[key]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"raw": args}
                arguments = dict(args)
                break

    if name:
        calls.append(ToolCall(
            id=data.get("id", ""),
            name=name,
            arguments=arguments,
        ))

    return calls