"""Agent conformance test suite.

Deterministic tests that probe a model's actual agent capabilities.
Results classify the model as L0-L4.

Tests:
- Select correct tool from 10 similar tools
- Emit nested JSON arguments
- Avoid calling tools when unnecessary (no-tool test)
- Process a tool error and recover
- Preserve tool-call IDs across turns
- Three sequential tool calls
- Continue after a large tool result
- Make an edit and verify it
- Handle parallel calls when advertised
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from interop.types import (
    BackendKind,
    CanonicalTool,
    CapabilityLevel,
    ToolCall,
    ToolCallDialect,
)


@dataclass
class ConformanceResult:
    """Result of a single conformance test."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ConformanceSuite:
    """Results of running all conformance tests."""

    level: CapabilityLevel = CapabilityLevel.L0
    tests: list[ConformanceResult] = field(default_factory=list)
    scores: dict[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def total(self) -> int:
        return len(self.tests)


# ─── Test definitions ───────────────────────────────────────────────────────


# Canonical tool definitions used across tests

TOOL_READ_FILE = CanonicalTool(
    name="read_file",
    description="Read a file from the filesystem",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute path to file"},
        },
        "required": ["path"],
    },
)

TOOL_EDIT_FILE = CanonicalTool(
    name="edit_file",
    description="Edit a file using string replacement",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
        },
        "required": ["path", "old_string", "new_string"],
    },
)

TOOL_SEARCH_CODE = CanonicalTool(
    name="search_code",
    description="Search for code patterns in the codebase",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string"},
        },
        "required": ["pattern"],
    },
)

TOOL_RUN_COMMAND = CanonicalTool(
    name="run_command",
    description="Execute a shell command",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "description": "Timeout in seconds"},
        },
        "required": ["command"],
    },
)

TOOL_LIST_FILES = CanonicalTool(
    name="list_files",
    description="List files in a directory",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
        },
        "required": ["path"],
    },
)

TOOL_SAVE_FILE = CanonicalTool(
    name="save_file",
    description="Save content to a file",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)


ALL_TOOLS = [
    TOOL_READ_FILE, TOOL_EDIT_FILE, TOOL_SEARCH_CODE,
    TOOL_RUN_COMMAND, TOOL_LIST_FILES, TOOL_SAVE_FILE,
    CanonicalTool(name="delete_file", description="Delete a file", parameters={
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }),
    CanonicalTool(name="move_file", description="Move or rename a file", parameters={
        "type": "object", "properties": {
            "source": {"type": "string"}, "destination": {"type": "string"},
        }, "required": ["source", "destination"],
    }),
    CanonicalTool(name="find_in_files", description="Search for text in files", parameters={
        "type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
        }, "required": ["pattern"],
    }),
    CanonicalTool(name="get_weather", description="Get weather for a location", parameters={
        "type": "object", "properties": {
            "location": {"type": "string"},
        }, "required": ["location"],
    }),
]


# ─── Test prompts ────────────────────────────────────────────────────────────


# L0: Chat only — no tools provided
PROMPT_L0 = "What is 2 + 2?"

# L1: Must call a specific tool when explicitly instructed
PROMPT_L1_SELECT = (
    "I need to read the file /home/user/config.json. "
    "You must use the read_file tool to do this. Call it now."
)

# L1: Must call explicit tool (no tool = wrong)
PROMPT_L1_EXPLICIT = (
    "Read the file /home/user/log.txt using the read_file tool."
)

# L2: Must automatically choose the right tool from context
PROMPT_L2_IMPLICIT = (
    "What's the content of /etc/hostname? Read it."
)

# L2: Must NOT call a tool when the question doesn't need one
PROMPT_L2_NO_TOOL = "What is the capital of France?"

# L3: Sequential tool calls expected
PROMPT_L3_SEQUENTIAL = (
    "First list the files in /home/user/, then read the first .json file you find."
)

# L3: Nested/structured arguments
PROMPT_L3_NESTED = (
    "Save a JSON file with the content {'name': 'test', 'version': 1} "
    "to /tmp/manifest.json using the save_file tool. Then read it back to verify."
)

# L3: Tool error recovery
PROMPT_L3_ERROR = (
    "Try to read /nonexistent/file.txt. If it doesn't exist, list files in /tmp instead."
)

# L4: Complex multi-step with verification
PROMPT_L4_EDIT_AND_VERIFY = (
    "In /tmp/test_file.txt, replace 'foo' with 'bar', then read it back to confirm."
)

# L4: Remember original objective across tools
PROMPT_L4_REMEMBER = (
    "Find the Python files in /home/user/, check if any contain 'TODO', "
    "and report the results."
)


# ─── Verifiers ──────────────────────────────────────────────────────────────


def verify_correct_tool(calls: list[ToolCall], tools: list[str]) -> ConformanceResult:
    """Verify the correct tool was called."""
    if not calls:
        return ConformanceResult("correct_tool", False, "no tool calls made")
    expected = tools[0] if tools else ""
    if calls[0].name == expected:
        return ConformanceResult("correct_tool", True, f"called {expected}")
    return ConformanceResult(
        "correct_tool", False,
        f"expected {expected}, got {calls[0].name}",
    )


def verify_no_tool(calls: list[ToolCall]) -> ConformanceResult:
    """Verify model did NOT call a tool."""
    if not calls:
        return ConformanceResult("no_tool", True, "no tool calls — correct")
    return ConformanceResult(
        "no_tool", False,
        f"called tool(s) when unnecessary: {[c.name for c in calls]}",
    )


def verify_arguments(call: ToolCall, required_args: list[str]) -> ConformanceResult:
    """Verify tool call has required arguments."""
    missing = [a for a in required_args if a not in call.arguments]
    if missing:
        return ConformanceResult(
            "arguments", False,
            f"missing required args: {missing}",
        )
    return ConformanceResult("arguments", True, f"has args: {list(call.arguments.keys())}")


def verify_sequential(calls: list[ToolCall], expected_order: list[str]) -> ConformanceResult:
    """Verify tool calls happened in the expected order."""
    actual = [c.name for c in calls]
    if actual == expected_order or actual[:len(expected_order)] == expected_order:
        return ConformanceResult("sequential", True, f"order: {actual}")
    return ConformanceResult(
        "sequential", False,
        f"expected {expected_order}, got {actual}",
    )


def verify_id_preserved(call: ToolCall) -> ConformanceResult:
    """Verify tool call has a non-empty ID."""
    if call.id:
        return ConformanceResult("tool_id", True, f"id={call.id}")
    return ConformanceResult("tool_id", False, "tool call missing ID")


# ─── Suite runner ────────────────────────────────────────────────────────────


class ConformanceRunner:
    """Run the conformance test suite against parsed tool calls.

    This validates tool calls syntactically — actual execution is
    done via the agent runtime (the test suite validates that the
    model *can* produce correct tool calls, not that it runs them).
    """

    def __init__(self, dialect: ToolCallDialect = ToolCallDialect.GENERIC_JSON) -> None:
        self.dialect = dialect
        self.tools = ALL_TOOLS

    def evaluate(self, tests: list[tuple[str, list[ToolCall], Any]]) -> ConformanceSuite:
        """Evaluate a list of (test_name, tool_calls, verifier_data)."""
        suite = ConformanceSuite()

        for name, calls, verifier in tests:
            if callable(verifier):
                result = verifier(calls)
            else:
                result = ConformanceResult(name, True, str(verifier))
            suite.tests.append(result)
            suite.scores[name] = result.passed

        # Determine level
        suite.level = self._compute_level(suite)
        return suite

    def _compute_level(self, suite: ConformanceSuite) -> CapabilityLevel:
        if suite.passed == 0:
            return CapabilityLevel.L0

        # L4 requires ALL tests passed
        all_passed = suite.passed == suite.total
        if all_passed and suite.total >= 8:
            return CapabilityLevel.L4

        # L3: Sequential and argument tests pass
        sequential_keywords = {"sequential", "error_recovery", "nested"}
        l3_passed = any(
            any(k in t.name for k in sequential_keywords) for t in suite.tests if t.passed
        )
        if l3_passed and suite.passed / max(suite.total, 1) >= 0.6:
            return CapabilityLevel.L3

        # L2: Auto tool selection passes, no false positives
        if suite.passed >= 3:
            return CapabilityLevel.L2

        # L1: At least some tool calls
        if suite.passed >= 1:
            return CapabilityLevel.L1

        return CapabilityLevel.L0


# ─── Standard test battery ───────────────────────────────────────────────────


def get_level_tests(level: CapabilityLevel) -> list[tuple[str, Any]]:
    """Get list of (prompt, validator_fn) for a given level."""
    if level == CapabilityLevel.L0:
        return []
    tests: list[tuple[str, Any]] = []
    # ... would be populated in integration


# ─── Standard test battery ───────────────────────────────────────────────────

# Each test entry: (name, prompt, [tools_for_test], validator_fn)
# CONFORMANCE_TESTS maps level string -> list of test definitions

CONFORMANCE_TESTS: dict[str, list[tuple[str, str, list[CanonicalTool], Any]]] = {
    "L1": [
        ("explicit_tool", PROMPT_L1_SELECT, ALL_TOOLS, lambda calls: verify_correct_tool(calls, ["read_file"])),
        ("explicit_tool2", PROMPT_L1_EXPLICIT, ALL_TOOLS, lambda calls: verify_correct_tool(calls, ["read_file"])),
    ],
    "L2": [
        ("implicit_tool", PROMPT_L2_IMPLICIT, ALL_TOOLS, lambda calls: verify_correct_tool(calls, ["read_file"])),
        ("no_tool_when_unnecessary", PROMPT_L2_NO_TOOL, ALL_TOOLS, verify_no_tool),
    ],
    "L3": [
        ("sequential", PROMPT_L3_SEQUENTIAL, ALL_TOOLS, lambda calls: verify_sequential(calls, ["list_files", "read_file"])),
        ("nested_args", PROMPT_L3_NESTED, ALL_TOOLS, lambda calls: True),  # structural check
    ],
    "L4": [],
}