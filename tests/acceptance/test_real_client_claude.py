"""Opt-in real-client acceptance test: Claude Code.

Skipped unless INTEROP_ACCEPTANCE_CLAUDE_BIN points at a real, installed
`claude` binary. NEVER runs in CI or by default `pytest` invocations — see
tests/acceptance/README.md and RELEASE.md's "Alpha vs. supported release
track". This has not been executed in this development sandbox (no real
`claude` binary or credentials are available here); it is written and
reviewed, not proven — treat the exact subprocess invocation below as a
starting point to adjust against the real CLI's actual interface the
first time this is run for real.

What it proves when it passes: the real `claude` binary, launched exactly
the way `interop run claude` launches it (same LaunchSpec, same env vars),
can send a request through a live Interop gateway, receive a tool call
recovered from a scripted fake-model response, and produce a final
response — one full round trip through the real client process, not just
the gateway/protocol layer.
"""

from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from agent_interop.agents.base import AgentLaunchContext
from agent_interop.agents.claude_code import ClaudeCodeIntegration
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

CLAUDE_BIN_ENV = "INTEROP_ACCEPTANCE_CLAUDE_BIN"

pytestmark = pytest.mark.skipif(
    not os.environ.get(CLAUDE_BIN_ENV),
    reason=(
        f"Opt-in acceptance test — set {CLAUDE_BIN_ENV} to a real `claude` "
        "binary path to run it. Not part of the default test suite or CI."
    ),
)


def _claude_version(claude_bin: str) -> str:
    try:
        out = subprocess.run(
            [claude_bin, "--version"], capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def test_claude_code_real_binary_completes_one_tool_round_trip(tmp_path):
    claude_bin = os.environ[CLAUDE_BIN_ENV]
    assert shutil.which(claude_bin) or os.path.isfile(claude_bin), (
        f"{CLAUDE_BIN_ENV}={claude_bin!r} is not an executable file"
    )

    port = 18091  # fixed, uncommon local port for the acceptance run
    config = InteropServerConfig(
        host="127.0.0.1",
        port=port,
        probe_on_startup=False,
        default_route_id="acceptance",
        routes={
            "acceptance": ModelRoute(
                id="acceptance",
                client_model_aliases=["claude-interop-acceptance"],
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
    try:
        launch_spec = ClaudeCodeIntegration().build_launch(
            AgentLaunchContext(
                route="acceptance",
                gateway_url=handle.base_url,
                model_name="acceptance-test-model",
                session_credential="acceptance-test-token",
            ),
        )
        env = {**os.environ, **launch_spec.env}

        # NOTE: the exact non-interactive invocation is the part most
        # likely to need adjusting against the real CLI's current
        # interface — `--print`/`-p` is Claude Code's documented
        # non-interactive "print mode" flag as of this writing.
        proc = subprocess.run(
            [claude_bin, "--print", "Read /tmp/acceptance-test.txt and tell me what it says."],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            cwd=tmp_path,
        )

        passed = proc.returncode == 0 and "hello from the acceptance test" in proc.stdout
        write_acceptance_result(
            "Claude Code",
            _claude_version(claude_bin),
            passed=passed,
            scenario="single_tool_round_trip",
            detail=proc.stdout if passed else f"stdout={proc.stdout!r} stderr={proc.stderr!r}",
        )
        assert passed, f"claude binary did not complete the round trip: {proc.stdout!r} / {proc.stderr!r}"
        assert len(transport.calls) >= 2, "expected at least 2 upstream calls (tool call + follow-up)"
    finally:
        handle.stop()
