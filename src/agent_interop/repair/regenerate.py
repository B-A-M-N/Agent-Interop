"""Hidden constrained regeneration — one-turn internal correction.

When deterministic repair fails, an optional ephemeral correction request
is sent to the same model with the same tool inventory, requesting only
the corrected tool call. The correction request is not added to the
client-visible conversation.

Backend-specific strategies:
- JSON Schema constrained decoding (when available)
- Grammar-constrained output
- Forced named tool selection
- Narrow repair prompt

Limits enforced:
``max_regenerations: int = 1``
``max_added_latency_ms: int = 15000``
``max_repair_input_bytes: int = 65536``
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from agent_interop.abi import CanonicalTool, SchemaIssue

logger = logging.getLogger("agent_interop.repair.regenerate")

# ─── Limits ───────────────────────────────────────────────────────────────────

MAX_REGENERATIONS = 1
MAX_ADDED_LATENCY_MS = 15000
MAX_REPAIR_INPUT_BYTES = 65536

# ─── Correction prompt ────────────────────────────────────────────────────────

_CORRECTION_TEMPLATE = """Your previous tool call was invalid. Please emit ONLY the corrected tool call.

## Previous call

Tool: {tool_name}
Arguments: {raw_arguments}

## Validation issues

{issues}

## Expected schema

```json
{schema}
```

## Instructions

1. Output ONLY a valid JSON tool call object.
2. Use the exact tool name: "{tool_name}".
3. Fix ALL the issues listed above.
4. Do NOT include any other text, explanation, or markdown formatting.
5. The output must be parseable as JSON with keys "name" and "arguments".

Corrected tool call:"""


def build_correction_request(
    tool_name: str,
    raw_arguments: Any,
    issues: list[SchemaIssue],
    tool: CanonicalTool,
) -> str:
    """Build the ephemeral correction prompt for the model.

    Args:
        tool_name: The canonical tool name.
        raw_arguments: The raw argument shape (truncated for size limits).
        issues: The validation issues to fix.
        tool: The tool definition with full schema.

    Returns:
        A prompt string for the model.
    """
    # Truncate raw arguments for safety
    raw_str = _truncate_str(repr(raw_arguments), 2000)

    # Format issues
    issue_lines = []
    for iss in issues[:10]:
        path = ".".join(str(p) for p in iss.path) if iss.path else "root"
        issue_lines.append(f"  - {path}: {iss.keyword} — {iss.message}")
    issues_str = "\n".join(issue_lines)

    schema_str = json.dumps(tool.input_schema, indent=2)
    if len(schema_str) > 8000:
        schema_str = json.dumps(tool.input_schema)

    return _CORRECTION_TEMPLATE.format(
        tool_name=tool_name,
        raw_arguments=raw_str,
        issues=issues_str,
        schema=schema_str,
    )


def _truncate_str(s: str, limit: int) -> str:
    return s if len(s) <= limit else s[: limit - 3] + "..."


# ─── Regeneration orchestrator ────────────────────────────────────────────────


class RegenerationOrchestrator:
    """Manages hidden constrained regeneration for tool-call repair.

    Usage:
        orchestrator = RegenerationOrchestrator()
        result = await orchestrator.attempt(
            tool_name="read_file",
            raw_arguments={"path": "/tmp/x"},
            issues=issues,
            tool=tool,
            regenerate_fn=async_callable,
        )
    """

    def __init__(
        self,
        max_attempts: int = MAX_REGENERATIONS,
        max_latency_ms: int = MAX_ADDED_LATENCY_MS,
        max_input_bytes: int = MAX_REPAIR_INPUT_BYTES,
    ) -> None:
        self.max_attempts = max_attempts
        self.max_latency_ms = max_latency_ms
        self.max_input_bytes = max_input_bytes
        self.start_time: float = 0.0
        self.attempts = 0
        self.total_latency_ms: float = 0.0

    def _check_limits(self) -> str | None:
        """Check if any limit has been exceeded.

        Returns an error message if a limit is exceeded, or None if ok.
        """
        elapsed = (time.monotonic() - self.start_time) * 1000
        if elapsed > self.max_latency_ms:
            return f"latency limit exceeded ({elapsed:.0f}ms > {self.max_latency_ms}ms)"
        if self.attempts >= self.max_attempts:
            return f"max regeneration attempts ({self.max_attempts}) exceeded"
        return None

    async def attempt(
        self,
        tool_name: str,
        raw_arguments: Any,
        issues: list[SchemaIssue],
        tool: CanonicalTool,
        regenerate_fn: Any,
    ) -> dict[str, Any] | None:
        """Attempt to regenerate a corrected tool call.

        Args:
            tool_name: The canonical tool name.
            raw_arguments: The raw argument shape.
            issues: Validation issues from the failed deterministic repair.
            tool: The tool definition.
            regenerate_fn: An async callable that takes a prompt string and
                returns the model's text response.

        Returns:
            A dict with ``name`` and ``arguments`` keys on success, or None.
        """
        self.start_time = time.monotonic()
        self.attempts = 0

        return await self._attempt_inner(tool_name, raw_arguments, issues, tool, regenerate_fn)

    async def _attempt_inner(
        self,
        tool_name: str,
        raw_arguments: Any,
        issues: list[SchemaIssue],
        tool: CanonicalTool,
        regenerate_fn: Any,
    ) -> dict[str, Any] | None:
        while True:
            limit_msg = self._check_limits()
            if limit_msg:
                logger.warning("regeneration aborted: %s", limit_msg)
                return None

            self.attempts += 1
            prompt = build_correction_request(tool_name, raw_arguments, issues, tool)

            try:
                response_text = await regenerate_fn(prompt)
            except Exception as exc:
                logger.warning("regeneration failed (attempt %d): %s", self.attempts, exc)
                continue

            if not response_text:
                continue

            # Parse the response — try to extract JSON
            parsed = self._extract_json(response_text)
            if parsed is None:
                continue

            # Validate shape
            name = parsed.get("name", parsed.get("tool", parsed.get("function", "")))
            args = parsed.get("arguments", parsed.get("input", parsed.get("parameters", {})))
            if not isinstance(name, str) or not isinstance(args, dict):
                continue

            # Use the canonical tool name
            if name != tool_name:
                # Accept if it's a recognized alias
                from agent_interop.repair.pipeline import canonicalize_tool_name
                canonical = canonicalize_tool_name(name, [tool])
                if canonical != tool_name:
                    continue

            return {"name": tool_name, "arguments": args}

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract a JSON object from model response text.

        Tries: full parse, then balanced scan.
        """
        text = text.strip()

        # Full parse
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Balanced scan
        from agent_interop.parsing.json_scan import BalancedJsonScanner
        candidates = BalancedJsonScanner.extract_tool_calls(text)
        if candidates:
            parsed = candidates[0].span.parse()
            if isinstance(parsed, dict):
                return parsed

        return None