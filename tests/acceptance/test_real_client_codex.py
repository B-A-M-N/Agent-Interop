"""Opt-in real-client acceptance test: Codex.

Skipped unless INTEROP_ACCEPTANCE_CODEX_BIN points at a real, installed
`codex` binary. NEVER runs in CI or by default `pytest` invocations — see
tests/acceptance/README.md and RELEASE.md's "Alpha vs. supported release
track". This has not been executed in this development sandbox (no real
`codex` binary or credentials are available here); see
test_real_client_claude.py's module docstring for the same caveat, which
applies equally here.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from agent_interop.agents.base import AgentLaunchContext
from agent_interop.agents.codex import CodexIntegration
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)

from ._harness import (
    ScriptedFakeTransport,
    start_acceptance_server,
    write_acceptance_result,
)

CODEX_BIN_ENV = "INTEROP_ACCEPTANCE_CODEX_BIN"

pytestmark = pytest.mark.skipif(
    not os.environ.get(CODEX_BIN_ENV),
    reason=(
        f"Opt-in acceptance test — set {CODEX_BIN_ENV} to a real `codex` "
        "binary path to run it. Not part of the default test suite or CI."
    ),
)


def _codex_version(codex_bin: str) -> str:
    try:
        out = subprocess.run(
            [codex_bin, "--version"], capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def test_codex_real_binary_completes_one_tool_round_trip(tmp_path):
    codex_bin = os.environ[CODEX_BIN_ENV]
    assert shutil.which(codex_bin) or os.path.isfile(codex_bin), (
        f"{CODEX_BIN_ENV}={codex_bin!r} is not an executable file"
    )

    port = 18092
    config = InteropServerConfig(
        host="127.0.0.1",
        port=port,
        probe_on_startup=False,
        default_route_id="acceptance",
        routes={
            "acceptance": ModelRoute(
                id="acceptance",
                client_model_aliases=["acceptance-test-model"],
                upstream_model="acceptance-test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:0",  # never actually dialed — transport is faked
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )

    transport = ScriptedFakeTransport(
        tool_name="read_file",
        tool_arguments={"path": "/tmp/acceptance-test.txt"},
        final_text="The file contains: hello from the acceptance test.",
    )
    handle = start_acceptance_server(config, transport)
    launch_spec = CodexIntegration().build_launch(
        AgentLaunchContext(
            route="acceptance",
            gateway_url=handle.base_url,
            model_name="acceptance-test-model",
            session_credential="acceptance-test-token",
        ),
    )
    try:
        assert launch_spec.command is not None, "Codex launch spec generation failed"
        env = {**os.environ, **launch_spec.env}

        # NOTE: exact non-interactive invocation/flags for `codex exec` are
        # the part most likely to need adjusting against the real CLI's
        # current interface.
        proc = subprocess.run(
            [*launch_spec.command, "exec", "Read /tmp/acceptance-test.txt and tell me what it says."],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            cwd=tmp_path,
        )

        passed = proc.returncode == 0 and "hello from the acceptance test" in proc.stdout
        write_acceptance_result(
            "Codex",
            _codex_version(codex_bin),
            passed=passed,
            scenario="single_tool_round_trip",
            detail=proc.stdout if passed else f"stdout={proc.stdout!r} stderr={proc.stderr!r}",
            argv=[*launch_spec.command, "exec", "Read /tmp/acceptance-test.txt and tell me what it says."],
            configuration_strategy=CodexIntegration().descriptor.integration_strategy,
            protocol=launch_spec.protocol.value if launch_spec.protocol else "",
            compatibility_path="adapted",
            model_digest="acceptance-test-model",
            verification={"read_test": passed, "multi_turn_continuation": passed},
        )
        assert passed, f"codex binary did not complete the round trip: {proc.stdout!r} / {proc.stderr!r}"
        assert len(transport.calls) >= 2, "expected at least 2 upstream calls (tool call + follow-up)"
    finally:
        if launch_spec.cleanup:
            launch_spec.cleanup()
        handle.stop()
