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

Limits (P0-repair-budget): ONE call to the orchestrator = ONE attempted
model generation. The caller (transaction layer / request budget) owns
repetition and passes its configured ceilings in:
``max_added_latency_ms`` — wall-clock deadline for the correction
``max_input_bytes`` — serialized correction-request ceiling, enforced
before dispatch
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
    """One call = one attempted model generation (P0-repair-budget).

    The orchestrator owns NO repetition loop. Repetition policy (how many
    regenerations a request may make) belongs to the caller — the
    transaction layer / request budget — because only it sees every other
    generation the request spends. Each :meth:`attempt` dispatches at most
    one correction generation, enforces the caller's latency deadline and
    input-size ceiling, and returns the parsed correction or None.

    Usage:
        orchestrator = RegenerationOrchestrator(
            max_latency_ms=policy.max_added_latency_ms,
            max_input_bytes=policy.max_input_bytes,
        )
        result = await orchestrator.attempt(...)
    """

    def __init__(
        self,
        max_latency_ms: int = MAX_ADDED_LATENCY_MS,
        max_input_bytes: int = MAX_REPAIR_INPUT_BYTES,
    ) -> None:
        self.max_latency_ms = max_latency_ms
        self.max_input_bytes = max_input_bytes
        self.start_time: float = 0.0
        self.attempts = 0
        self.total_latency_ms: float = 0.0

    def _check_limits(self, prompt_bytes: int = 0) -> str | None:
        """Check the deadline and the rendered-correction input ceiling.

        ``prompt_bytes`` — the actual serialized size of the correction
        request about to be sent. The advertised ``max_input_bytes`` is a
        real ceiling: an oversized correction prompt is rejected BEFORE
        dispatch, not after the model has already paid for it.
        """
        elapsed = (time.monotonic() - self.start_time) * 1000
        if elapsed > self.max_latency_ms:
            return f"latency limit exceeded ({elapsed:.0f}ms > {self.max_latency_ms}ms)"
        if prompt_bytes > self.max_input_bytes:
            return (
                f"correction request too large ({prompt_bytes} bytes > "
                f"{self.max_input_bytes} max_input_bytes)"
            )
        return None

    async def attempt(
        self,
        tool_name: str,
        raw_arguments: Any,
        issues: list[SchemaIssue],
        tool: CanonicalTool,
        regenerate_fn: Any,
    ) -> dict[str, Any] | None:
        """Dispatch exactly one constrained correction generation.

        Returns a dict with ``name`` and ``arguments`` keys on success, or
        None (deadline exceeded, oversized correction, dispatch failure, or
        an unparseable answer — one generation, whatever the outcome).
        """
        self.start_time = time.monotonic()
        self.attempts = 1

        prompt = build_correction_request(tool_name, raw_arguments, issues, tool)
        limit_msg = self._check_limits(prompt_bytes=len(prompt.encode("utf-8")))
        if limit_msg:
            logger.warning("regeneration aborted before dispatch: %s", limit_msg)
            self.attempts = 0
            return None

        try:
            response_text = await regenerate_fn(prompt)
        except Exception as exc:
            logger.warning("regeneration failed: %s", exc)
            return None
        finally:
            self.total_latency_ms = (time.monotonic() - self.start_time) * 1000

        if not response_text:
            return None

        # Parse the response — try to extract JSON
        parsed = self._extract_json(response_text)
        if parsed is None:
            return None

        # Validate shape
        name = parsed.get("name", parsed.get("tool", parsed.get("function", "")))
        args = parsed.get("arguments", parsed.get("input", parsed.get("parameters", {})))
        if not isinstance(name, str) or not isinstance(args, dict):
            return None

        # Use the canonical tool name
        if name != tool_name:
            # Accept if it's a recognized alias
            from agent_interop.repair.pipeline import canonicalize_tool_name
            canonical = canonicalize_tool_name(name, [tool])
            if canonical != tool_name:
                return None

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
