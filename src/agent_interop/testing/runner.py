"""Real conformance runner — invokes a model, executes a tool loop, persists results.

Replaces the old pre-parsed-object evaluator with an actual model runner
that sends prompts through the gateway, processes tool calls, sends
results back, and records the exact compatibility tuple.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    ProtocolKind,
    ToolChoiceMode,
)
from agent_interop.config import (
    FieldAliasPolicy,
    InteropServerConfig,
    MalformedJsonPolicy,
    ModelRoute,
    RepairConfig,
    UnknownToolPolicy,
)
from agent_interop.evidence import EvidenceStore, get_default_store
from agent_interop.gateway import Gateway
from agent_interop.replay.types import (
    CompatibilityKey,
    CompatibilityResult,
)

logger = logging.getLogger("agent_interop.testing.runner")


@dataclass
class ToolCallOutcome:
    """Outcome of a single tool execution."""

    tool_name: str
    arguments: dict[str, Any]
    result: str = ""
    is_error: bool = False
    # The model-issued tool_call id (CanonicalToolCallBlock.id), for
    # verifying distinct-ID criteria — the executor itself never sees this,
    # so it's set by the caller (run_test) after execution, not here.
    call_id: str = ""
    # Which conversation turn this call happened in (1-indexed, matching
    # ConformanceRunResult.turns) — lets criteria distinguish "N tool
    # calls in one response" (genuinely parallel) from "N tool calls
    # spread across N turns" (sequential).
    turn: int = 0


@dataclass
class ConformanceTest:
    """A single conformance test with explicit pass criteria."""

    name: str
    prompt: str
    tools: list[CanonicalTool] = field(default_factory=list)
    max_turns: int = 5
    tool_executor: Callable | None = None  # fn(tool_name, args) -> ToolCallOutcome
    tool_choice: CanonicalToolChoice | None = None  # Expected/required tool choice mode
    # Pass criteria
    expected_tools: list[str] = field(default_factory=list)  # tool names that MUST be called
    forbidden_tools: list[str] = field(default_factory=list)  # tool names that MUST NOT be called
    requires_final_text: bool = False  # must produce text after tool loop
    min_tool_calls: int = 0  # minimum number of tool calls required
    max_tool_calls: int | None = None  # maximum allowed (None = unlimited, 0 = zero)
    # Tool names that must appear in this RELATIVE order (as a subsequence
    # of the actual call sequence — other calls may be interspersed, but
    # these may not be reversed). Distinct from expected_tools, which only
    # checks membership.
    expected_tool_order: list[str] | None = None
    # Every recorded tool call must have a non-empty, mutually-unique
    # call_id (CanonicalToolCallBlock.id) — verifies the model/pipeline
    # actually assigns distinct IDs rather than reusing/omitting them.
    requires_distinct_call_ids: bool = False
    # If > 0, at least this many tool calls must land in the SAME turn
    # (single response) — distinguishes genuine same-response parallel
    # tool use from calls merely spread across sequential turns, which
    # min_tool_calls alone cannot tell apart.
    requires_same_turn_parallel: int = 0
    # tool_name -> {arg_name: expected_value}. At least one call to that
    # tool must match every key/value pair. Verifies actual ARGUMENT
    # content survived extraction/repair correctly, not just that the
    # right tool name was called.
    expected_arguments: dict[str, dict[str, Any]] | None = None
    # Zero-arg callback returning an error string on failure, None on
    # pass — checked independently of the tool loop's own recorded calls
    # and results. For a test like "edit_and_verify" this re-reads the
    # actual file from disk after the run, rather than trusting the
    # model's own read_file call to have reported real content — a model
    # (or a broken repair path) that emits a plausible-looking read_file
    # call with fabricated arguments would otherwise still "pass" purely
    # on tool-name/order criteria.
    post_run_verify: Callable[[], str | None] | None = None


@dataclass
class ConformanceRunResult:
    """Result of running a conformance test."""

    test_name: str
    passed: bool = False
    turns: int = 0
    tool_calls: list[ToolCallOutcome] = field(default_factory=list)
    final_text: str = ""
    error: str = ""
    # InteropErrorCode string when `error` came from a Gateway-reported
    # response.error (upstream/transport failure) — None for every other
    # failure path (criteria mismatch, forced-tool-choice violation,
    # unhandled exception). testing/levels.py uses this to tell an
    # infrastructure outage apart from a real behavioral failure instead
    # of letting a backend hiccup silently count as "the model can't do
    # this test".
    error_code: str | None = None
    duration_seconds: float = 0.0
    # The authoritative compatibility key for this test's route+request+context,
    # resolved via Gateway._resolve_invocation_plan_and_key so it byte-for-byte
    # matches what live traffic produces. None when the key could not be
    # resolved (e.g. tool-contract validation failure) — the test still runs.
    compat_key: CompatibilityKey | None = None


class RealConformanceRunner:
    """Runs conformance tests against an actual model via the gateway."""

    def __init__(
        self,
        config: InteropServerConfig,
        *,
        client_protocol: ProtocolKind = ProtocolKind.ANTHROPIC_MESSAGES,
        client_id: str = "",
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self._config = config
        self._client_protocol = client_protocol
        self._client_id = client_id
        self._evidence_store = evidence_store or get_default_store()
        self._gateway: Gateway | None = None

    async def start(self) -> None:
        """Initialize the gateway."""
        self._gateway = Gateway(self._config, evidence_store=self._evidence_store)
        await self._gateway.startup()

    async def close(self) -> None:
        """Shut down the gateway."""
        if self._gateway:
            await self._gateway.close()
            self._gateway = None

    @property
    def evidence_store(self) -> EvidenceStore:
        """The evidence store the gateway writes observations to.

        This is the SAME store instance the gateway's automatic live-traffic
        write-back uses. The CLI calls ``mark_verified`` on this exact instance
        explicitly, rather than relying on both sides independently resolving
        to the same singleton via ``get_default_store()`` (fragile).
        """
        return self._evidence_store

    async def run_test(
        self,
        test: ConformanceTest,
        *,
        model_name: str = "test-model",
        route: ModelRoute | None = None,
    ) -> ConformanceRunResult:
        """Run a single conformance test with tool loop execution."""
        import time

        start = time.monotonic()
        result = ConformanceRunResult(test_name=test.name)

        if not self._gateway:
            result.error = "gateway not started"
            return result

        # Build conversation
        messages: list[CanonicalMessage] = [
            CanonicalMessage(role="user", content=[CanonicalTextBlock(text=test.prompt)]),
        ]

        tools = test.tools

        # Build the context ONCE with the client_id threaded through. client_id is
        # part of CompatibilityKey, so real production traffic populates it (via
        # RequestContext.from_headers); conformance tests must do the same or the
        # resolved key will mismatch live traffic on this field.
        from agent_interop.context import RequestContext
        context = RequestContext(
            client_protocol=self._client_protocol,
            client_id=self._client_id,
        )

        # Build a representative request ONCE (before the turn loop). tools and
        # tool_choice are fixed per-test, and turns never change the key — so the
        # key computed here is valid for every turn of this test. Resolve the
        # authoritative compatibility key via the gateway so it byte-for-byte
        # matches what live traffic produces for the same route+request+context.
        if route is not None:
            key_request = CanonicalRequest(
                model=CanonicalModelReference(requested_name=model_name),
                messages=messages,
                tools=tools,
                tool_choice=test.tool_choice if test.tool_choice else CanonicalToolChoice(),
            )
            try:
                _, _, _, _, compat_key = self._gateway._resolve_invocation_plan_and_key(
                    route, key_request, context, streaming=False,
                )
                result.compat_key = compat_key
            except Exception as exc:
                # Key resolution is purely certification bookkeeping — never let a
                # failure here (e.g. ValueError from tool-contract validation)
                # break the actual test execution or pass/fail determination.
                result.compat_key = None
                logger.debug(
                    "conformance test %s: could not resolve compatibility key: %s",
                    test.name, exc,
                )

        try:
            for turn in range(test.max_turns):
                result.turns = turn + 1

                request = CanonicalRequest(
                    model=CanonicalModelReference(requested_name=model_name),
                    messages=messages,
                    tools=tools,
                    tool_choice=test.tool_choice if test.tool_choice else CanonicalToolChoice(),
                )

                # Call gateway
                response = await self._gateway.handle_request(request, context)

                # Fail immediately on error responses. Includes the error
                # CODE (not just the message) so callers can distinguish a
                # genuine upstream/transport failure (BACKEND_*,
                # MODEL_NOT_FOUND) from a behavioral validation failure
                # (e.g. TOOL_CHOICE_VIOLATION — the model just didn't
                # comply with a forced tool choice) — both surface through
                # this same response.error path, but only one of them is
                # infrastructure noise rather than capability evidence.
                if response.error:
                    result.error = f"Gateway error [{response.error.code}]: {response.error.message}"
                    result.error_code = str(response.error.code)
                    break

                # Check for tool calls
                tool_calls = [
                    block for block in response.content
                    if isinstance(block, CanonicalToolCallBlock)
                ]

                if not tool_calls:
                    # Check if tool calls were required
                    if test.tool_choice and test.tool_choice.mode in (
                        ToolChoiceMode.REQUIRED, ToolChoiceMode.NAMED,
                    ):
                        result.error = (
                            f"{test.tool_choice.mode.value.upper()} tool choice "
                            "but no tool calls in response"
                        )
                        break

                    # No tool calls — conversation complete
                    result.final_text = " ".join(
                        block.text for block in response.content
                        if isinstance(block, CanonicalTextBlock) and block.text
                    )
                    result.passed = True
                    break

                # Execute tool calls
                tool_results: list[CanonicalToolResultBlock] = []
                for tc in tool_calls:
                    outcome = await self._execute_tool(test, tc.name, tc.arguments)
                    outcome.call_id = tc.id
                    outcome.turn = result.turns
                    result.tool_calls.append(outcome)
                    tool_results.append(CanonicalToolResultBlock(
                        tool_call_id=tc.id or f"tc_{uuid.uuid4().hex[:8]}",
                        content=outcome.result,
                        is_error=outcome.is_error,
                    ))

                # Add assistant message with tool calls and tool results to conversation
                messages.append(CanonicalMessage(role="assistant", content=cast(list[CanonicalContentBlock], tool_calls)))
                messages.append(CanonicalMessage(role="tool", content=cast(list[CanonicalContentBlock], tool_results)))

        except Exception as exc:
            result.error = str(exc)
            logger.error("conformance test %s failed: %s", test.name, exc)

        # Verify explicit test criteria
        if result.passed:
            failure = self._verify_criteria(test, result)
            if failure:
                result.passed = False
                result.error = failure
            elif test.post_run_verify is not None:
                # Independent of what the tool loop recorded — re-checks
                # real external state (e.g. actual file contents).
                verify_failure = test.post_run_verify()
                if verify_failure:
                    result.passed = False
                    result.error = verify_failure

        result.duration_seconds = time.monotonic() - start
        return result

    def _verify_criteria(self, test: ConformanceTest, result: ConformanceRunResult) -> str | None:
        """Verify explicit pass criteria. Returns error message if failed, None if passed."""
        called_names = [tc.tool_name for tc in result.tool_calls]

        if test.expected_tools:
            for expected in test.expected_tools:
                if expected not in called_names:
                    return f"Expected tool '{expected}' was never called (called: {called_names})"

        if test.forbidden_tools:
            for forbidden in test.forbidden_tools:
                if forbidden in called_names:
                    return f"Forbidden tool '{forbidden}' was called"

        if test.min_tool_calls > 0 and len(result.tool_calls) < test.min_tool_calls:
            return f"Too few tool calls ({len(result.tool_calls)} < {test.min_tool_calls})"

        if test.max_tool_calls is not None and len(result.tool_calls) > test.max_tool_calls:
            return f"Too many tool calls ({len(result.tool_calls)} > {test.max_tool_calls})"

        if test.requires_final_text and not result.final_text.strip():
            return "Expected final text response but got none"

        if test.expected_tool_order:
            cursor = 0
            for expected in test.expected_tool_order:
                try:
                    idx = called_names.index(expected, cursor)
                except ValueError:
                    return (
                        f"Expected tool order {test.expected_tool_order} not satisfied "
                        f"— '{expected}' not found at/after position {cursor} "
                        f"(called: {called_names})"
                    )
                cursor = idx + 1

        if test.requires_distinct_call_ids:
            call_ids = [tc.call_id for tc in result.tool_calls]
            if not all(call_ids):
                return f"Distinct call IDs required, but at least one tool call had no id (ids: {call_ids})"
            if len(set(call_ids)) != len(call_ids):
                return f"Duplicate tool call ids detected: {call_ids}"

        if test.requires_same_turn_parallel > 0:
            from collections import Counter
            turn_counts = Counter(tc.turn for tc in result.tool_calls)
            if not any(count >= test.requires_same_turn_parallel for count in turn_counts.values()):
                return (
                    f"Expected >= {test.requires_same_turn_parallel} tool calls within a "
                    f"single turn (parallel), but calls were spread across turns: "
                    f"{dict(turn_counts)}"
                )

        if test.expected_arguments:
            for tool_name, expected_args in test.expected_arguments.items():
                matching = [tc for tc in result.tool_calls if tc.tool_name == tool_name]
                if not matching:
                    return f"Expected arguments check for '{tool_name}' but it was never called"
                if not any(
                    all(call.arguments.get(k) == v for k, v in expected_args.items())
                    for call in matching
                ):
                    got = [call.arguments for call in matching]
                    return (
                        f"No call to '{tool_name}' matched expected arguments "
                        f"{expected_args} (got: {got})"
                    )

        return None

    async def _execute_tool(
        self,
        test: ConformanceTest,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> ToolCallOutcome:
        """Execute a tool call."""
        if test.tool_executor:
            try:
                return test.tool_executor(tool_name, arguments)
            except Exception as exc:
                return ToolCallOutcome(
                    tool_name=tool_name,
                    arguments=arguments,
                    result=f"Error: {exc}",
                    is_error=True,
                )

        # Default: echo back
        return ToolCallOutcome(
            tool_name=tool_name,
            arguments=arguments,
            result=f"Executed {tool_name}({arguments})",
        )

    def store_result(
        self,
        key: CompatibilityKey,
        result: CompatibilityResult,
        *,
        failure_cases: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        """Store the result in the evidence store."""
        return self._evidence_store.store_result(key, result, failure_cases)


def with_repair_disabled(config: InteropServerConfig) -> InteropServerConfig:
    """Return a copy of ``config`` with every route's repair pipeline
    forced fully off (no syntax/safe-shape/coercive recovery, no
    regeneration).

    Used to run the same conformance battery twice — once against the
    route's real configured repair policy, once with repair off — so a
    computed level can report the model's OWN unaided capability
    alongside the repair-assisted one (see testing/levels.py,
    ConformanceLevelResult.repair_enabled), rather than only ever
    measuring the repair-assisted number under the unqualified label
    "the model's level".
    """
    from dataclasses import replace

    disabled = RepairConfig(
        max_regenerations=0,
        malformed_json=MalformedJsonPolicy.REJECT.value,
        unknown_tool=UnknownToolPolicy.REJECT.value,
        field_aliases=FieldAliasPolicy.DISABLED.value,
    )
    new_routes = {
        route_id: replace(route, repair=disabled)
        for route_id, route in config.routes.items()
    }
    return replace(config, routes=new_routes)


# ─── Standard test battery ──────────────────────────────────────────────────


def make_sandboxed_file_executor(workspace_dir: Path) -> Callable:
    """Execute read_file/edit_file/list_files against REAL files under
    ``workspace_dir`` instead of returning a fixed string regardless of
    what the model actually did.

    Re-audit P1#10: "edit_and_verify" is a canned-string test by name only
    unless the edit and the read-back are independently mediated by real
    disk I/O — a model that emits any edit_file call at all would
    previously get the same hardcoded "new_value" back from read_file, so
    a no-op or wrong-argument edit could still "pass". Operating on real
    files closes that gap: if old_string doesn't match what's on disk, or
    the model never actually issued the edit, the subsequent read_file
    call surfaces the real (unedited or errored) content.
    """
    def executor(tool_name: str, arguments: dict[str, Any]) -> ToolCallOutcome:
        raw_path = str(arguments.get("path", "test.txt"))
        # Confine every path to the workspace by basename only, regardless
        # of what the model sent — Path(raw_path).name strips any
        # directory component (including a leading "/" or "../"), so
        # "/etc/passwd" resolves to workspace_dir/passwd, never the real
        # /etc/passwd. Conformance tests run against a real filesystem
        # must not let untrusted model output touch anything outside the
        # sandbox, however that path was spelled.
        target = workspace_dir / Path(raw_path).name

        if tool_name == "read_file":
            try:
                return ToolCallOutcome(tool_name, arguments, target.read_text())
            except OSError as exc:
                return ToolCallOutcome(tool_name, arguments, f"Error: {exc}", is_error=True)

        if tool_name == "edit_file":
            old = str(arguments.get("old_string", ""))
            new = str(arguments.get("new_string", ""))
            try:
                content = target.read_text()
            except OSError as exc:
                return ToolCallOutcome(tool_name, arguments, f"Error: {exc}", is_error=True)
            if old not in content:
                return ToolCallOutcome(
                    tool_name, arguments,
                    f"Error: old_string {old!r} not found in file", is_error=True,
                )
            target.write_text(content.replace(old, new, 1))
            return ToolCallOutcome(tool_name, arguments, "Edit applied")

        if tool_name == "list_files":
            try:
                names = sorted(p.name for p in workspace_dir.iterdir())
            except OSError as exc:
                return ToolCallOutcome(tool_name, arguments, f"Error: {exc}", is_error=True)
            return ToolCallOutcome(tool_name, arguments, ", ".join(names))

        return ToolCallOutcome(tool_name, arguments, f"ok: {arguments}")

    return executor


def _make_edit_and_verify_post_check(workspace_dir: Path) -> Callable[[], str | None]:
    """Independent post-run check for the edit_and_verify test: re-reads
    test.txt from disk itself, rather than trusting the model's own
    read_file tool call to have reported the real content."""
    def check() -> str | None:
        actual = (workspace_dir / "test.txt").read_text()
        if "old_value" in actual:
            return f"file still contains 'old_value' after the edit loop: {actual!r}"
        if "new_value" not in actual:
            return f"file does not contain the expected 'new_value': {actual!r}"
        return None
    return check


def get_standard_tests(workspace_dir: Path | None = None) -> list[ConformanceTest]:
    """Get the standard 12-test conformance battery from the review.

    ``workspace_dir``, when given, is a real temporary directory used to
    seed and verify the "edit_and_verify" test's file with actual disk
    I/O (see make_sandboxed_file_executor) instead of a canned string
    executor. Callers that don't need a live filesystem (e.g. exercising
    ConformanceTest wiring/criteria in isolation) can omit it and get the
    original fixed-string behavior for every test.
    """
    tools = _get_standard_tools()

    def make_executor(results: dict[str, str]) -> Callable:
        def executor(tool_name: str, arguments: dict[str, Any]) -> ToolCallOutcome:
            return ToolCallOutcome(
                tool_name=tool_name,
                arguments=arguments,
                result=results.get(tool_name, f"ok: {arguments}"),
            )
        return executor

    if workspace_dir is not None:
        (workspace_dir / "test.txt").write_text("The value is old_value here.")
        edit_and_verify_executor: Callable = make_sandboxed_file_executor(workspace_dir)
        edit_and_verify_prompt = (
            f"Edit {workspace_dir / 'test.txt'} to replace 'old_value' with "
            "'new_value', then read the file back to verify the change was applied."
        )
        edit_and_verify_post_check = _make_edit_and_verify_post_check(workspace_dir)
    else:
        edit_and_verify_executor = make_executor({
            "edit_file": "Edit applied",
            "read_file": "new_value",
        })
        edit_and_verify_prompt = (
            "Edit /tmp/test.txt to replace 'old_value' with 'new_value', "
            "then read the file back to verify the change was applied."
        )
        edit_and_verify_post_check = None

    return [
        ConformanceTest(
            name="explicit_forced_tool",
            prompt="Read the file /tmp/test.txt using the read_file tool.",
            tools=tools,
            # A test named "forced" must actually force the choice — this
            # previously left tool_choice unset, which defaults to AUTO
            # (automatic selection), meaning it never exercised forced/
            # named tool-choice behavior at all despite its name.
            tool_choice=CanonicalToolChoice.named("read_file"),
            tool_executor=make_executor({"read_file": "File contents: hello"}),
            expected_tools=["read_file"],
            min_tool_calls=1,
        ),
        ConformanceTest(
            name="automatic_tool_selection",
            prompt="What files are in /tmp?",
            tools=tools,
            tool_executor=make_executor({"list_files": "file1.txt, file2.py"}),
            expected_tools=["list_files"],
            min_tool_calls=1,
        ),
        ConformanceTest(
            name="no_tool_request",
            prompt="Say hello to the user. Do not use any tools.",
            tools=tools,
            forbidden_tools=["read_file", "list_files", "search_code"],
            max_tool_calls=0,
        ),
        ConformanceTest(
            name="nested_arguments",
            prompt="Search for 'TODO' in /src with path='/src' and pattern='TODO'.",
            tools=tools,
            tool_executor=make_executor({"search_code": "Found 3 matches"}),
            expected_tools=["search_code"],
            min_tool_calls=1,
            # A test literally named "nested_arguments" must check the
            # argument VALUES survived correctly, not just that the tool
            # was called with SOME arguments.
            expected_arguments={"search_code": {"path": "/src", "pattern": "TODO"}},
        ),
        ConformanceTest(
            name="tool_result_continuation",
            prompt="Read /tmp/data.txt and tell me what it says.",
            tools=tools,
            tool_executor=make_executor({"read_file": "The data is important."}),
            expected_tools=["read_file"],
            requires_final_text=True,
        ),
        ConformanceTest(
            name="tool_error_recovery",
            prompt="Try to read /nonexistent/file.txt. If it fails, list files in /tmp instead.",
            tools=tools,
            tool_executor=lambda name, args: (
                ToolCallOutcome(name, args, "Error: file not found", is_error=True)
                if "nonexistent" in str(args.get("path", ""))
                else ToolCallOutcome(name, args, "file1.txt, file2.py")
            ),
            min_tool_calls=2,
            # "Recovery" requires actually calling the recovery tool after
            # the failing one, in that order — min_tool_calls=2 alone
            # would also pass a model that just retries the same failing
            # call twice and never recovers.
            expected_tools=["read_file", "list_files"],
            expected_tool_order=["read_file", "list_files"],
        ),
        ConformanceTest(
            name="sequential_calls",
            prompt="First list files in /tmp, then read /tmp/test.txt.",
            tools=tools,
            max_turns=5,
            tool_executor=make_executor({
                "list_files": "test.txt, data.py",
                "read_file": "Hello world",
            }),
            expected_tools=["list_files", "read_file"],
            min_tool_calls=2,
            # A test named for SEQUENCE must check order, not just that
            # both tools were called at some point.
            expected_tool_order=["list_files", "read_file"],
        ),
        ConformanceTest(
            name="malformed_call_repair",
            # Embedded double-quotes are the classic small/local-model
            # tool-call failure mode: the model emits arguments whose JSON
            # escaping breaks (a dropped backslash, an unescaped quote),
            # and Interop's repair pipeline (repair/rules.py) exists
            # specifically to recover from exactly this. A runner that
            # drives a REAL model end-to-end can't deterministically force
            # malformed JSON out of it — but it CAN, and must, verify the
            # argument value that comes out the other end is byte-correct,
            # not merely that some call to edit_file happened.
            prompt=(
                'Call edit_file on /tmp/test.txt: replace the text he said "hi" '
                'with the text she said "bye". Use exactly those two phrases, '
                "including the double quotes, as old_string and new_string."
            ),
            tools=tools,
            tool_executor=make_executor({"edit_file": "Edit applied"}),
            expected_tools=["edit_file"],
            expected_arguments={
                "edit_file": {
                    "old_string": 'he said "hi"',
                    "new_string": 'she said "bye"',
                },
            },
        ),
        ConformanceTest(
            name="history_round_trip",
            prompt="Read /tmp/a.txt then /tmp/b.txt.",
            tools=tools,
            max_turns=5,
            tool_executor=make_executor({
                "read_file": "Content",
            }),
            expected_tools=["read_file"],
            min_tool_calls=2,
        ),
        # L4 tests
        ConformanceTest(
            name="parallel_calls",
            prompt=(
                "Read both /tmp/a.txt and /tmp/b.txt at the same time, in a "
                "single response with two separate tool calls."
            ),
            tools=tools,
            max_turns=3,
            tool_executor=make_executor({"read_file": "file content"}),
            expected_tools=["read_file"],
            min_tool_calls=2,
            # "Parallel" means within one response — min_tool_calls=2
            # alone is equally satisfied by two calls spread across two
            # sequential turns, which is not what this test claims to
            # verify.
            requires_same_turn_parallel=2,
        ),
        ConformanceTest(
            name="edit_and_verify",
            prompt=edit_and_verify_prompt,
            tools=tools,
            max_turns=5,
            # Real disk I/O when workspace_dir is given (see
            # make_sandboxed_file_executor) — the edit and the read-back
            # are independently verified against an actual file, not a
            # canned string returned regardless of what the model did.
            tool_executor=edit_and_verify_executor,
            # A test named "edit_and_verify" must actually call edit_file —
            # the original definition only ever listed and read files.
            expected_tools=["edit_file", "read_file"],
            expected_tool_order=["edit_file", "read_file"],
            min_tool_calls=2,
            requires_final_text=True,
            post_run_verify=edit_and_verify_post_check,
        ),
        ConformanceTest(
            name="distinct_ids",
            prompt="Read /tmp/a.txt and /tmp/b.txt one after another.",
            tools=tools,
            max_turns=5,
            tool_executor=make_executor({"read_file": "content"}),
            expected_tools=["read_file"],
            min_tool_calls=2,
            # The point of this test's name: verify the model/pipeline
            # assigns distinct, non-empty call IDs — min_tool_calls=2
            # alone says nothing about IDs at all.
            requires_distinct_call_ids=True,
        ),
    ]


def _get_standard_tools() -> list[CanonicalTool]:
    """Standard tool set for conformance testing."""
    return [
        CanonicalTool(
            name="read_file",
            description="Read a file from the filesystem",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        CanonicalTool(
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
        ),
        CanonicalTool(
            name="list_files",
            description="List files in a directory",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        ),
        CanonicalTool(
            name="search_code",
            description="Search for patterns in code",
            input_schema={
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                "required": ["pattern"],
            },
        ),
        CanonicalTool(
            name="run_command",
            description="Execute a shell command",
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        ),
    ]
