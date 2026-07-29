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

NOTE: this module's LEVEL_REQUIREMENTS/_compute_level operate on pre-parsed
CanonicalToolCallBlock lists and synthetic test names ("explicit_tool",
"implicit_tool", ...) — a standalone structural-validation utility, not
wired into the CLI/gateway/evidence store. cli.py `test`/`certify` and
`/v1/capabilities`'s observed-capability reporting use
interop.testing.levels.compute_conformance_level instead, which maps the
REAL RealConformanceRunner battery (testing/runner.get_standard_tests,
different test names) to a level with explicit infra-vs-behavioral and
battery-version semantics. Do not confuse the two mappings — they use
different test name sets and are not interchangeable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from agent_interop.abi import CanonicalTool, CanonicalToolCallBlock, ToolCallDialect
from agent_interop.types import CapabilityLevel


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
    input_schema={
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
    input_schema={
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
    input_schema={
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
    input_schema={
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
    input_schema={
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
    input_schema={
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
    CanonicalTool(name="delete_file", description="Delete a file", input_schema={
        "type": "object", "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }),
    CanonicalTool(name="move_file", description="Move or rename a file", input_schema={
        "type": "object", "properties": {
            "source": {"type": "string"}, "destination": {"type": "string"},
        }, "required": ["source", "destination"],
    }),
    CanonicalTool(name="find_in_files", description="Search for text in files", input_schema={
        "type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
        }, "required": ["pattern"],
    }),
    CanonicalTool(name="get_weather", description="Get weather for a location", input_schema={
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

# L4: Parallel calls — multiple independent operations in one response
PROMPT_L4_PARALLEL = (
    "Read /tmp/a.txt and /tmp/b.txt at the same time. "
    "Make both read_file calls in a single response."
)

# L4: Distinct tool call IDs for parallel disambiguation
PROMPT_L4_DISTINCT_IDS = (
    "Search for 'TODO' in /src and 'FIXME' in /tests simultaneously. "
    "Each search must have a unique call ID."
)


# ─── Verifiers ──────────────────────────────────────────────────────────────


def verify_correct_tool(calls: list[CanonicalToolCallBlock], tools: list[str]) -> ConformanceResult:
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


def verify_no_tool(calls: list[CanonicalToolCallBlock]) -> ConformanceResult:
    """Verify model did NOT call a tool."""
    if not calls:
        return ConformanceResult("no_tool", True, "no tool calls — correct")
    return ConformanceResult(
        "no_tool", False,
        f"called tool(s) when unnecessary: {[c.name for c in calls]}",
    )


def verify_arguments(call: CanonicalToolCallBlock, required_args: list[str]) -> ConformanceResult:
    """Verify tool call has required arguments."""
    missing = [a for a in required_args if a not in call.arguments]
    if missing:
        return ConformanceResult(
            "arguments", False,
            f"missing required args: {missing}",
        )
    return ConformanceResult("arguments", True, f"has args: {list(call.arguments.keys())}")


def verify_sequential(calls: list[CanonicalToolCallBlock], expected_order: list[str]) -> ConformanceResult:
    """Verify tool calls happened in the expected order."""
    actual = [c.name for c in calls]
    if actual == expected_order or actual[:len(expected_order)] == expected_order:
        return ConformanceResult("sequential", True, f"order: {actual}")
    return ConformanceResult(
        "sequential", False,
        f"expected {expected_order}, got {actual}",
    )


def verify_id_preserved(call: CanonicalToolCallBlock) -> ConformanceResult:
    """Verify tool call has a non-empty ID."""
    if call.id:
        return ConformanceResult("tool_id", True, f"id={call.id}")
    return ConformanceResult("tool_id", False, "tool call missing ID")


def verify_parallel_calls(calls: list[CanonicalToolCallBlock], min_count: int = 2) -> ConformanceResult:
    """Verify multiple tool calls were made in a single response (parallel)."""
    if len(calls) >= min_count:
        names = [c.name for c in calls]
        return ConformanceResult("parallel", True, f"parallel calls: {names}")
    return ConformanceResult(
        "parallel", False,
        f"expected >= {min_count} calls, got {len(calls)}",
    )


def verify_error_recovery(calls: list[CanonicalToolCallBlock], expected_after_error: str) -> ConformanceResult:
    """Verify model adapts after a tool error by calling a different tool."""
    if not calls:
        return ConformanceResult("error_recovery", False, "no calls made after error")
    if any(c.name == expected_after_error for c in calls):
        return ConformanceResult("error_recovery", True, f"recovered with {expected_after_error}")
    return ConformanceResult(
        "error_recovery", False,
        f"expected recovery call to {expected_after_error}, got {[c.name for c in calls]}",
    )


def verify_nested_arguments(call: CanonicalToolCallBlock, nested_key: str) -> ConformanceResult:
    """Verify a tool call argument contains nested/structured JSON."""
    if nested_key not in call.arguments:
        return ConformanceResult("nested_args", False, f"missing key: {nested_key}")
    val = call.arguments.get(nested_key)
    if isinstance(val, (dict, list)):
        return ConformanceResult("nested_args", True, f"nested {nested_key}: {type(val).__name__}")
    # If it's a string, it might be JSON that needs parsing
    if isinstance(val, str):
        import json
        try:
            parsed = json.loads(val)
            if isinstance(parsed, (dict, list)):
                return ConformanceResult("nested_args", True, f"parsed JSON {nested_key}")
        except (json.JSONDecodeError, ValueError):
            pass
    return ConformanceResult(
        "nested_args", False,
        f"{nested_key} is not structured (type={type(val).__name__})",
    )


def verify_distinct_tool_ids(calls: list[CanonicalToolCallBlock]) -> ConformanceResult:
    """Verify all tool calls have unique, non-empty IDs."""
    ids = [c.id for c in calls]
    if not all(ids):
        return ConformanceResult("distinct_ids", False, "some calls missing IDs")
    if len(set(ids)) != len(ids):
        return ConformanceResult("distinct_ids", False, f"duplicate IDs: {ids}")
    return ConformanceResult("distinct_ids", True, f"{len(ids)} unique IDs")


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

    def evaluate(self, tests: list[tuple[str, list[CanonicalToolCallBlock], Callable[[list[CanonicalToolCallBlock]], ConformanceResult]]]) -> ConformanceSuite:
        """Evaluate a list of (test_name, tool_calls, verifier_fn).

        Each verifier is a callable that takes the list of parsed tool calls
        and returns a ConformanceResult.
        """
        suite = ConformanceSuite()

        for name, calls, verifier in tests:
            result = verifier(calls)
            suite.tests.append(result)
            suite.scores[name] = result.passed

        # Determine level
        suite.level = self._compute_level(suite)
        return suite

    def _compute_level(self, suite: ConformanceSuite) -> CapabilityLevel:
        """Derive conformance level using cumulative evidence.

        Each level requires ALL tests for that level AND all lower levels
        to pass.  Test names are matched against LEVEL_REQUIREMENTS to
        determine which level they belong to.
        """
        if suite.total == 0:
            return CapabilityLevel.L0

        passed_names = {t.name for t in suite.tests if t.passed}

        # Check levels from highest to lowest — return the first fully
        # satisfied. Uses the CUMULATIVE requirement set (this level's own
        # tests plus every lower level's), matching the docstring's
        # documented contract — checking only LEVEL_REQUIREMENTS[level] in
        # isolation would let a model that fails every L1 test still reach
        # L4 merely by passing the three L4-named tests.
        for level in [CapabilityLevel.L4, CapabilityLevel.L3, CapabilityLevel.L2, CapabilityLevel.L1]:
            required = _cumulative_requirements(level.value)
            if required and required.issubset(passed_names):
                return level

        return CapabilityLevel.L0


# ─── Level requirements (cumulative) ─────────────────────────────────────────

# Each level requires ALL listed test names to pass.
# Levels are cumulative: achieving L3 requires all L1+L2+L3 tests.
LEVEL_REQUIREMENTS: dict[str, set[str]] = {
    "L1": {"explicit_tool", "explicit_tool2", "arguments_check"},
    "L2": {"implicit_tool", "no_tool_when_unnecessary"},
    "L3": {"sequential", "nested_args", "error_recovery"},
    "L4": {"parallel_calls", "edit_and_verify", "distinct_ids"},
}

# Full cumulative requirement set per level (includes all lower levels)
def _cumulative_requirements(level: str) -> set[str]:
    """Get all test names required for a level (including lower levels)."""
    level_order = ["L1", "L2", "L3", "L4"]
    idx = level_order.index(level) if level in level_order else -1
    reqs: set[str] = set()
    for i in range(idx + 1):
        reqs |= LEVEL_REQUIREMENTS.get(level_order[i], set())
    return reqs


# ─── Standard test battery ───────────────────────────────────────────────────


def get_level_tests(level: CapabilityLevel) -> list[tuple[str, str, list[CanonicalTool], Callable[[list[CanonicalToolCallBlock]], ConformanceResult]]]:
    """Get cumulative test cases for a given conformance level.

    Returns tests for the requested level AND all lower levels.
    Each entry is (name, prompt, tools, verifier_fn).
    L0 returns an empty list (chat only — no tool tests).
    """
    if level == CapabilityLevel.L0:
        return []

    level_str = level.value
    required = _cumulative_requirements(level_str)

    # Collect all test definitions that match the required names
    result: list[tuple[str, str, list[CanonicalTool], Callable[[list[CanonicalToolCallBlock]], ConformanceResult]]] = []
    for tests in CONFORMANCE_TESTS.values():
        for name, prompt, tools, verifier in tests:
            if name in required:
                result.append((name, prompt, tools, verifier))

    return result


# ─── Standard test battery ───────────────────────────────────────────────────

# Each test entry: (name, prompt, [tools_for_test], verifier_fn)
# CONFORMANCE_TESTS maps level string -> list of test definitions for that level only
# (not cumulative — get_level_tests() handles the cumulative expansion)

CONFORMANCE_TESTS: dict[str, list[tuple[str, str, list[CanonicalTool], Callable[[list[CanonicalToolCallBlock]], ConformanceResult]]]] = {
    "L1": [
        ("explicit_tool", PROMPT_L1_SELECT, ALL_TOOLS, lambda calls: verify_correct_tool(calls, ["read_file"])),
        ("explicit_tool2", PROMPT_L1_EXPLICIT, ALL_TOOLS, lambda calls: verify_correct_tool(calls, ["read_file"])),
        ("arguments_check", PROMPT_L1_SELECT, ALL_TOOLS, lambda calls: verify_arguments(calls[0], ["path"]) if calls else ConformanceResult("arguments_check", False, "no calls")),
    ],
    "L2": [
        ("implicit_tool", PROMPT_L2_IMPLICIT, ALL_TOOLS, lambda calls: verify_correct_tool(calls, ["read_file"])),
        ("no_tool_when_unnecessary", PROMPT_L2_NO_TOOL, ALL_TOOLS, verify_no_tool),
    ],
    "L3": [
        ("sequential", PROMPT_L3_SEQUENTIAL, ALL_TOOLS, lambda calls: verify_sequential(calls, ["list_files", "read_file"])),
        ("nested_args", PROMPT_L3_NESTED, ALL_TOOLS, lambda calls: verify_nested_arguments(calls[0], "content") if calls else ConformanceResult("nested_args", False, "no calls")),
        ("error_recovery", PROMPT_L3_ERROR, ALL_TOOLS, lambda calls: verify_error_recovery(calls, "list_files")),
    ],
    "L4": [
        ("parallel_calls", PROMPT_L4_PARALLEL, ALL_TOOLS, lambda calls: verify_parallel_calls(calls, min_count=2)),
        ("edit_and_verify", PROMPT_L4_EDIT_AND_VERIFY, ALL_TOOLS, lambda calls: verify_sequential(calls, ["edit_file", "read_file"])),
        ("distinct_ids", PROMPT_L4_DISTINCT_IDS, ALL_TOOLS, verify_distinct_tool_ids),
    ],
}