"""Tests for the certify / replay / logs CLI command fixes.

These cover the three commands that were remediated in Step 12:

* ``certify`` — must run the full standard battery, build a real
  ``CompatibilityKey``, store evidence, and exit nonzero on any failure.
* ``replay`` — must invoke the real ``replay_all_policies`` engine and
  print a cross-policy comparison, exiting nonzero on a regression.
* ``logs`` / ``start`` — must agree on a single XDG-state log file path.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

from agent_interop.abi import (
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    ProtocolKind,
)
from agent_interop.cli import _log_file_path, app
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context import RequestContext
from agent_interop.evidence.store import EvidenceStore
from agent_interop.gateway import Gateway
from agent_interop.replay.types import ReplayResult
from agent_interop.testing.runner import (
    ConformanceRunResult,
    ConformanceTest,
    RealConformanceRunner,
)
from agent_interop.transport.http import UpstreamResponse, UpstreamTransport

# ─── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def valid_config(tmp_path: Path) -> Path:
    """Write a minimal valid interop.yaml and return its path."""
    import yaml  # type: ignore[import-untyped]

    config = {
        "host": "127.0.0.1",
        "port": 8090,
        "log_level": "info",
        "probe_on_startup": False,
        "default_route_id": "cli",
        "routes": {
            "cli": {
                "aliases": ["test-model"],
                "upstream_model": "test-model",
                "upstream": {
                    "kind": "ollama",
                    "base_url": "http://127.0.0.1:11434",
                    "wire_protocol": "ollama_chat",
                },
                "tool_mode": "auto",
                "translation_mode": "canonical",
            }
        },
    }
    config_path = tmp_path / "interop.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


# ─── certify ─────────────────────────────────────────────────────────────────


def _mock_runner(passed: bool) -> tuple[MagicMock, MagicMock]:
    """Build a mocked RealConformanceRunner whose run_test returns ``passed``.

    Used only by the exit-code tests, which pin the old control-flow contract
    (certify exits 0 on full pass, 1 on any failure) and do not exercise the
    new evidence-grouping path. The mocked results carry ``compat_key=None``,
    which the new certify routes into the "no resolvable key" branch — so the
    exit-code decision is unaffected by the verification rewrite.
    """
    instance = MagicMock()
    instance.start = AsyncMock()
    instance.close = AsyncMock()
    instance.run_test = AsyncMock(
        return_value=ConformanceRunResult(
            test_name="t", passed=passed, turns=1, error="" if passed else "boom",
        )
    )
    cls = MagicMock(return_value=instance)
    return cls, instance


# ─── A real upstream + store harness for the end-to-end certify tests ────────
#
# The old certify tests mocked ``RealConformanceRunner``, which meant
# ``run_test`` never actually ran the gateway — so the automatic live-traffic
# write-back never fired and the runner's dense ``compat_key`` was never
# computed. The rewrite verifies the NEW behaviour (dense-key grouping +
# mark_verified on full pass), so these tests drive a *real* runner against a
# *fake* upstream transport. The fake returns a body the configured codec can
# decode, so a test that passes criteria produces a real evidence record under
# the real dense key — exactly what the live gate looks up.

# A body the OpenAI-Chat codec decodes into a plain text response (no tool
# calls), which makes the "passing" conformance test succeed.
_OPENAI_TEXT_BODY: dict[str, Any] = {
    "id": "fake-chatcmpl",
    "object": "chat.completion",
    "created": 0,
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello there."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
}


class _FakeUpstreamTransport(UpstreamTransport):
    """Returns a pre-seeded OpenAI-Chat body for every ``send()``.

    Mirrors ``_FakeTransport`` in ``test_evidence_store_wiring.py``; kept local
    so this file's harness is self-contained. Subclasses ``UpstreamTransport``
    so it is assignable to ``Gateway._transport``.
    """

    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__()
        self._body = body

    async def send(self, request: Any) -> UpstreamResponse:
        return UpstreamResponse(
            status_code=200,
            headers={},
            body=json.dumps(self._body).encode("utf-8"),
        )

    async def stream(  # type: ignore[override]
        self, request: Any
    ) -> Any:
        # Never called by the non-streaming ``handle_request`` path; raises if
        # it ever is. Overrides the parent's async-generator signature.
        raise AssertionError("non-streaming certify tests must not stream")


def _certify_config_with(tmp_path: Path) -> Path:
    """Write a minimal config that routes to an OpenAI-Chat upstream and return
    its path. Using the OpenAI-Chat codec (rather than Ollama) lets the fake
    transport's body decode cleanly without standing up a real server."""
    config = {
        "host": "127.0.0.1",
        "port": 8090,
        "log_level": "error",
        "probe_on_startup": False,
        "default_route_id": "cli",
        "routes": {
            "cli": {
                "aliases": ["test-model"],
                "upstream_model": "test-model",
                "upstream": {
                    "kind": "openai_compatible",
                    "base_url": "http://127.0.0.1:0",
                    "wire_protocol": "openai_chat",
                },
                "tool_mode": "auto",
                "translation_mode": "canonical",
            }
        },
    }
    config_path = tmp_path / "interop_fake.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


@contextmanager
def _certify_with_fake_upstream(
    tests: list[ConformanceTest],
    *,
    body: dict[str, Any],
) -> Iterator[tuple[EvidenceStore, Gateway, ModelRoute]]:
    """Run certify against a *real* runner whose gateway talks to a fake
    upstream transport, persisting into a real in-memory ``EvidenceStore``.

    The fake transport is injected by subclassing ``RealConformanceRunner`` so
    that ``start()`` wires it into the gateway after startup. ``get_default_store``
    and ``get_standard_tests`` are patched so the CLI uses our in-memory store
    and our chosen test battery. Yields ``(store, gateway, route)`` for
    assertions.
    """
    store = EvidenceStore(db_path=":memory:")
    fake_transport = _FakeUpstreamTransport(body)

    class _RunnerWithFakeTransport(RealConformanceRunner):
        async def start(self) -> None:
            await super().start()
            if self._gateway is not None:
                self._gateway._transport = fake_transport

    # The CLI builds the runner from the config's default route ("cli").
    cfg = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "cli": ModelRoute(
                id="cli",
                client_model_aliases=["test-model"],
                upstream_model="test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    route = cfg.routes["cli"]

    with (
        patch(
            "agent_interop.testing.runner.RealConformanceRunner",
            _RunnerWithFakeTransport,
        ),
        patch("agent_interop.testing.runner.get_standard_tests", lambda **kwargs: tests),
        # The runner imports get_default_store into its own module namespace,
        # so the patch target is runner.get_default_store (not the source).
        patch("agent_interop.testing.runner.get_default_store", return_value=store),
    ):
        yield store, Gateway(cfg), route


class TestCertifyExitsNonzeroOnFailure:
    """certify must exit 1 when any test in the suite fails, 0 on full pass.

    These pin the exit-code contract (unchanged by the verification rewrite) via
    a mocked runner whose results carry ``compat_key=None`` — routing into the
    new "no resolvable key" branch without affecting the exit decision.
    """

    def _invoke(self, runner: CliRunner, config_path: Path, passed: bool):
        tests = [
            ConformanceTest(name="t1", prompt="p"),
            ConformanceTest(name="t2", prompt="p"),
        ]
        mock_cls, _ = _mock_runner(passed)
        with (
            patch("agent_interop.testing.runner.RealConformanceRunner", mock_cls),
            patch("agent_interop.testing.runner.get_standard_tests", lambda **kwargs: tests),
            patch("agent_interop.evidence.store.get_default_store", MagicMock()),
        ):
            return runner.invoke(
                app,
                ["certify", "--path", str(config_path), "--client", "claude_code"],
            )

    def test_exits_zero_on_full_pass(self, runner, valid_config):
        result = self._invoke(runner, valid_config, passed=True)
        assert result.exit_code == 0, f"certify failed: {result.output}"

    def test_exits_nonzero_on_failure(self, runner, valid_config):
        result = self._invoke(runner, valid_config, passed=False)
        assert result.exit_code == 1, f"certify should fail: {result.output}"

    def test_config_error_exits_nonzero(self, runner, tmp_path: Path):
        """Config/route errors must still exit 1 (pre-existing paths)."""
        missing = tmp_path / "nope.yaml"
        result = runner.invoke(app, ["certify", "--path", str(missing)])
        assert result.exit_code == 1

    def test_resolves_by_route_id_not_upstream_model(self, runner, tmp_path: Path):
        """MVP: certify must resolve routes by route_id, not
        route_config.upstream_model. Gateway route resolution only matches
        a route's own id or its client_model_aliases — never the backend's
        upstream_model string. A route whose upstream_model isn't ALSO one
        of its own aliases used to fail every test with "Unknown model"."""
        config = {
            "host": "127.0.0.1",
            "port": 8090,
            "log_level": "info",
            "probe_on_startup": False,
            "default_route_id": "my-custom-route",
            "routes": {
                "my-custom-route": {
                    "aliases": ["claude-interop-custom"],
                    # Deliberately NOT equal to the route id or any alias —
                    # this is the exact shape that triggered the bug.
                    "upstream_model": "internal-backend-model-name:latest",
                    "upstream": {
                        "kind": "ollama",
                        "base_url": "http://127.0.0.1:11434",
                        "wire_protocol": "ollama_chat",
                    },
                    "tool_mode": "auto",
                    "translation_mode": "canonical",
                }
            },
        }
        config_path = tmp_path / "interop_mismatched.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        tests = [ConformanceTest(name="t1", prompt="p")]
        mock_cls, mock_instance = _mock_runner(passed=True)
        with (
            patch("agent_interop.testing.runner.RealConformanceRunner", mock_cls),
            patch("agent_interop.testing.runner.get_standard_tests", lambda **kwargs: tests),
            patch("agent_interop.evidence.store.get_default_store", MagicMock()),
        ):
            result = runner.invoke(
                app,
                ["certify", "--path", str(config_path), "--client", "claude_code"],
            )

        assert result.exit_code == 0, f"certify failed: {result.output}"
        call_kwargs = mock_instance.run_test.call_args.kwargs
        assert call_kwargs["model_name"] == "my-custom-route", (
            f"expected certify to resolve by route_id, got model_name="
            f"{call_kwargs['model_name']!r}"
        )


class TestCertifyVerifiesDenseKeyEndToEnd:
    """certify must verify a *real dense* CompatibilityKey — the exact key the
    live traffic gate resolves — grouping test results by that key and calling
    ``mark_verified`` only when every test sharing the key passed.

    These drive a *real* runner against a fake upstream transport, so the
    automatic live-traffic write-back fires during ``handle_request`` and the
    runner's per-test ``compat_key`` is computed via the gateway's own
    ``_resolve_invocation_plan_and_key``. This is the faithful regression test
    for Bug B (the old hand-rolled sparse key never matched the live gate).
    """

    def test_passing_test_produces_verified_dense_record(
        self, runner: CliRunner, tmp_path: Path
    ):
        """A passing test must yield a verified record whose key is dense and
        resolvable by the live gate."""
        read_file = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        # A test with tools that passes on a plain text response (no tool
        # calls required). Tools being offered means the automatic write-back
        # fires, producing a real evidence record under the dense key.
        passing_test = ConformanceTest(
            name="no_tool_request",
            prompt="Say hello to the user.",
            tools=[read_file],
        )
        config_path = _certify_config_with(tmp_path)
        with _certify_with_fake_upstream(
            [passing_test], body=_OPENAI_TEXT_BODY
        ) as (store, _gw, _route):
            result = runner.invoke(
                app,
                ["certify", "--path", str(config_path), "--client", "claude_code"],
            )

        assert result.exit_code == 0, f"certify failed: {result.output}"
        assert "Suite passed (observed, not manually verified)" in result.output, result.output

        stored = store.query_results()
        assert len(stored) == 1, f"expected exactly 1 record, got {len(stored)}"
        compat_key, compat_result = stored[0]

        # certify is a fully automated CLI run — it must NEVER set
        # manually_verified=True itself. That flag is reserved for actual
        # human review; conflating the two would let an automated suite
        # pass silently activate compatibility-pack trust it hasn't earned.
        assert compat_result.manually_verified is False
        # Round-trip through the DB confirms persistence, not just in-memory.
        round_tripped = store.get_result(compat_key)
        assert round_tripped is not None
        assert round_tripped.manually_verified is False

        # The key is DENSE: real resolved identity, never empty placeholders.
        # (The old bug hand-rolled a key with empty model_digest/version/etc.)
        assert compat_key.model_id == "test-model"
        assert compat_key.backend_kind == "openai_compatible"
        assert compat_key.client_protocol == "anthropic_messages"
        assert compat_key.client_id == "claude_code"

    def test_effective_tool_mode_is_resolved_not_raw_auto(
        self, runner: CliRunner, tmp_path: Path
    ):
        """AUTO route mode must resolve to a concrete mode in the stored key."""
        read_file = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        passing_test = ConformanceTest(
            name="no_tool_request",
            prompt="Say hello to the user.",
            tools=[read_file],
        )
        config_path = _certify_config_with(tmp_path)
        with _certify_with_fake_upstream(
            [passing_test], body=_OPENAI_TEXT_BODY
        ) as (store, _gw, _route):
            result = runner.invoke(
                app,
                ["certify", "--path", str(config_path), "--client", "claude_code"],
            )

        assert result.exit_code == 0, f"certify failed: {result.output}"
        compat_key, _ = store.query_results()[0]
        # Must be a resolved concrete mode, never the raw "auto" config string.
        assert compat_key.effective_tool_mode != "auto"
        assert compat_key.effective_tool_mode in {"native", "prompted", "disabled"}


class TestCertifyDoesNotVerifyFailingRoutes:
    """Bug A regression: a route where SOME conformance tests fail must NOT
    produce a verified record for that route's key — failing tests must never
    be silently marked verified."""

    def test_failing_test_is_not_verified(self, runner: CliRunner, tmp_path: Path):
        """A test that fails its criteria must not yield a verified record."""
        read_file = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        # Requires a tool call that the fake text response never produces.
        failing_test = ConformanceTest(
            name="needs_tool_call",
            prompt="Read /tmp/x.txt using the read_file tool.",
            tools=[read_file],
            expected_tools=["read_file"],
            min_tool_calls=1,
        )
        config_path = _certify_config_with(tmp_path)
        with _certify_with_fake_upstream(
            [failing_test], body=_OPENAI_TEXT_BODY
        ) as (store, _gw, _route):
            result = runner.invoke(
                app,
                ["certify", "--path", str(config_path), "--client", "claude_code"],
            )

        # The suite failed => nonzero exit.
        assert result.exit_code == 1, f"certify should fail: {result.output}"
        assert "Observed, suite failed" in result.output, result.output

        # The live-traffic write-back still recorded an observation (tools were
        # offered and the gateway succeeded), but it must NOT be verified.
        stored = store.query_results()
        assert len(stored) == 1, f"expected 1 observation, got {len(stored)}"
        _, compat_result = stored[0]
        assert compat_result.manually_verified is False, (
            "a failing test must never produce a verified record (Bug A)"
        )


class TestCertifyKeyMatchesLiveGate:
    """Bug B regression: the key certify verifies must be byte-for-byte
    identical to the key a real live request for the same route+tools+client
    resolves to — i.e. the live gate would actually find certify's verified
    record, not a phantom."""

    def test_certify_key_matches_live_request_key(
        self, runner: CliRunner, tmp_path: Path
    ):
        """The dense key from certify must equal the key the live gate resolves
        for an identical request, so the verified record is findable."""
        read_file = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        prompt = "Say hello to the user."
        passing_test = ConformanceTest(
            name="no_tool_request",
            prompt=prompt,
            tools=[read_file],
        )
        config_path = _certify_config_with(tmp_path)
        with _certify_with_fake_upstream(
            [passing_test], body=_OPENAI_TEXT_BODY
        ) as (store, gateway, route):
            result = runner.invoke(
                app,
                ["certify", "--path", str(config_path), "--client", "claude_code"],
            )
            assert result.exit_code == 0, f"certify failed: {result.output}"

            stored = store.query_results()
            assert len(stored) == 1
            certified_key, certified_result = stored[0]
            assert certified_result.manually_verified is False

            # Build the *same* request a live client would send (same model,
            # tools, and client_id) and resolve its key via the gateway's own
            # path — the exact computation the live gate performs.
            live_request = CanonicalRequest(
                model=CanonicalModelReference(requested_name="test-model"),
                messages=[
                    CanonicalMessage(
                        role="user",
                        content=[CanonicalTextBlock(text=prompt)],
                    )
                ],
                tools=[read_file],
                tool_choice=CanonicalToolChoice(),
            )
            live_context = RequestContext(
                client_protocol=ProtocolKind.ANTHROPIC_MESSAGES,
                client_id="claude_code",
            )
            _, _, _, _, live_key = gateway._resolve_invocation_plan_and_key(
                route, live_request, live_context, streaming=False,
            )

        # Byte-for-byte equality: the live gate would find certify's record.
        assert live_key == certified_key, (
            "certify's verified key must match the live gate's key (Bug B)"
        )
        # And looking up that key returns the observed (not manually
        # verified — certify no longer sets that flag) record.
        live_lookup = store.get_result(live_key)
        assert live_lookup is not None
        assert live_lookup.manually_verified is False


class TestLegacyTestCommandKeyMatchesLiveGate:
    """The legacy `interop test` command (distinct from `certify`) used to
    hand-build its own CompatibilityKey from stub objects, including
    effective_tool_mode="auto" — a value resolve_tool_mode() never
    actually produces — so the evidence it stored could never be found by
    a live request's lookup. It now resolves the same authoritative key
    `certify` and the live gate use, via run_test(route=...)."""

    def test_test_command_key_matches_live_request_key(
        self, runner: CliRunner, tmp_path: Path
    ):
        read_file = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        prompt = "Say hello to the user."
        passing_test = ConformanceTest(
            name="no_tool_request",
            prompt=prompt,
            tools=[read_file],
        )

        store = EvidenceStore(db_path=":memory:")
        fake_transport = _FakeUpstreamTransport(_OPENAI_TEXT_BODY)

        class _RunnerWithFakeTransport(RealConformanceRunner):
            async def start(self) -> None:
                await super().start()
                if self._gateway is not None:
                    self._gateway._transport = fake_transport

        with (
            patch(
                "agent_interop.testing.runner.RealConformanceRunner",
                _RunnerWithFakeTransport,
            ),
            patch("agent_interop.testing.runner.get_standard_tests", lambda **kwargs: [passing_test]),
            patch("agent_interop.testing.runner.get_default_store", return_value=store),
        ):
            result = runner.invoke(
                app,
                [
                    "test", "test-model",
                    "--backend", "openai_compatible",
                    "--backend-url", "http://127.0.0.1:0",
                    "--client-profile", "claude-code",
                ],
            )
            assert result.exit_code == 0, f"test command failed: {result.output}"

            stored = store.query_results()
            assert len(stored) == 1
            written_key, _written_result = stored[0]

            # Build the SAME request a live client would send and resolve
            # its key via the gateway's own path — must match byte-for-byte.
            live_route = ModelRoute(
                id="test",
                client_model_aliases=["test-model"],
                upstream_model="test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OPENAI_COMPATIBLE,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                profile="auto",
            )
            live_config = InteropServerConfig(
                probe_on_startup=False,
                default_route_id="test",
                routes={"test": live_route},
            )
            live_gateway = Gateway(live_config)
            live_request = CanonicalRequest(
                model=CanonicalModelReference(requested_name="test-model"),
                messages=[
                    CanonicalMessage(role="user", content=[CanonicalTextBlock(text=prompt)])
                ],
                tools=[read_file],
                tool_choice=CanonicalToolChoice(),
            )
            live_context = RequestContext(
                client_protocol=ProtocolKind.ANTHROPIC_MESSAGES,
                client_id="claude_code",
            )
            _, _, _, _, live_key = live_gateway._resolve_invocation_plan_and_key(
                live_route, live_request, live_context, streaming=False,
            )

        assert live_key == written_key, (
            "the legacy `test` command's stored key must match the live "
            "gate's key, or the evidence it wrote is unfindable"
        )
        assert store.get_result(live_key) is not None


# ─── replay ──────────────────────────────────────────────────────────────────


class TestReplayInvokesRealEngine:
    """replay must call the real engine and print a policy comparison."""

    def _case_file(self, tmp_path: Path) -> Path:
        case_path = tmp_path / "case.json"
        case_path.write_text(json.dumps({
            "case_id": "test-case-1",
            "client_protocol": "anthropic_messages",
            "upstream_protocol": "ollama_chat",
        }))
        return case_path

    def test_default_mode_calls_engine_and_prints_comparison(self, runner, tmp_path):
        case_path = self._case_file(tmp_path)
        engine_result = {
            "repair_disabled": ReplayResult(
                policy_name="repair_disabled", executable=False,
                output_tool_name="read_file",
            ),
            "safe_shape": ReplayResult(
                policy_name="safe_shape", executable=True, arguments_valid=True,
                output_tool_name="read_file",
            ),
        }
        with patch("agent_interop.replay.replay_all_policies", AsyncMock(return_value=engine_result)) as mocked:
            result = runner.invoke(app, ["replay", str(case_path)])
        # The real engine entry point was actually invoked
        mocked.assert_awaited_once()
        assert result.exit_code == 0, f"replay failed: {result.output}"
        # Output contains the cross-policy comparison
        assert "Best policy" in result.output
        assert "safe_shape" in result.output
        assert "Repair helped" in result.output

    def test_exits_nonzero_when_regression_introduced(self, runner, tmp_path):
        """introduced_unintended execution must trigger exit 1."""
        case_path = self._case_file(tmp_path)
        engine_result = {
            "repair_disabled": ReplayResult(
                policy_name="repair_disabled", executable=False,
                output_tool_name="read_file",
            ),
            "coercive": ReplayResult(
                policy_name="coercive", executable=True, arguments_valid=True,
                output_tool_name="delete_file",  # not in baseline -> unintended
            ),
        }
        with patch("agent_interop.replay.replay_all_policies", AsyncMock(return_value=engine_result)):
            result = runner.invoke(app, ["replay", str(case_path)])
        assert result.exit_code == 1, f"replay should flag regression: {result.output}"
        assert "Regression" in result.output

    def test_single_policy_mode_uses_replay_case(self, runner, tmp_path):
        case_path = self._case_file(tmp_path)
        single = ReplayResult(policy_name="safe_shape", executable=True, arguments_valid=True)
        with (
            patch("agent_interop.replay.replay_all_policies", AsyncMock()) as all_policies,
            patch("agent_interop.replay.replay_case", AsyncMock(return_value=single)) as single_case,
        ):
            result = runner.invoke(app, ["replay", str(case_path), "--policy", "safe_shape"])
        single_case.assert_awaited_once()
        all_policies.assert_not_awaited()
        assert result.exit_code == 0, f"replay failed: {result.output}"
        assert "safe_shape" in result.output

    def test_unknown_policy_exits_nonzero(self, runner, tmp_path):
        case_path = self._case_file(tmp_path)
        result = runner.invoke(app, ["replay", str(case_path), "--policy", "not_a_policy"])
        assert result.exit_code == 1


# ─── logs / start path agreement ────────────────────────────────────────────


class TestLogsStartPathAgreement:
    """``logs`` and ``start`` must resolve to the same XDG-state log file."""

    def test_log_file_path_uses_xdg_state_home(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
        # XDG_STATE_HOME is the state root; helper namespaces under "interop"
        assert _log_file_path() == tmp_path / "interop" / "logs" / "interop.log"

    def test_log_file_path_fallback_without_xdg(self, monkeypatch, tmp_path: Path):
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        monkeypatch.setattr("os.path.expanduser", lambda p: str(tmp_path) if p == "~" else p)
        assert _log_file_path() == tmp_path / ".local" / "state" / "interop" / "logs" / "interop.log"

    def test_logs_reads_from_shared_helper(self, runner, tmp_path: Path):
        """logs must read from the exact path the helper resolves to."""
        log_path = tmp_path / "interop" / "logs" / "interop.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("line-one\nline-two\n")
        with patch("agent_interop.cli._log_file_path", return_value=log_path):
            result = runner.invoke(app, ["logs", "--lines", "50"])
        assert result.exit_code == 0, f"logs failed: {result.output}"
        assert "line-one" in result.output
        assert "line-two" in result.output

    def test_logs_no_file_is_graceful(self, runner, tmp_path: Path):
        missing = tmp_path / "missing" / "interop.log"
        with patch("agent_interop.cli._log_file_path", return_value=missing):
            result = runner.invoke(app, ["logs"])
        assert result.exit_code == 0
        assert "No log file found" in result.output
