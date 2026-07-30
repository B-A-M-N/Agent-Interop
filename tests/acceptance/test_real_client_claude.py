"""Opt-in real-client acceptance test: Claude Code.

Skipped unless INTEROP_ACCEPTANCE_CLAUDE_BIN points at a real, installed
`claude` binary. NEVER runs in CI or by default `pytest` invocations — see
tests/acceptance/README.md and RELEASE.md's "Alpha vs. supported release
track".

This HAS been executed for real against the installed `claude` binary
(v2.1.220) — see `acceptance/results/claude-code-2.1.220.json`. The
initial version hand-built the subprocess argv and only reused
`launch_spec.env`, which meant it never actually exercised the `--model
claude-interop-<route>` flag `ClaudeCodeIntegration.build_launch()` adds
(a real fix found by an earlier live-launch test, where the CLI ignored
`CLAUDE_MODEL` and fell back to the operator's own persisted default
model). It now builds `cmd` from `launch_spec.command` itself so a run of
this test proves the exact argv `interop run claude` would produce.

What it proves when it passes: the real `claude` binary, launched exactly
the way `interop run claude` launches it (same LaunchSpec.command, same
env vars), can send a request through a live Interop gateway, receive a
tool call recovered from a scripted fake-model response, and produce a
final response — one full round trip through the real client process, not
just the gateway/protocol layer.
"""

from __future__ import annotations

import os
import re
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
    # `claude --version` prints e.g. "2.1.220 (Claude Code)" — extract just
    # the version number so it drops cleanly into a results filename
    # (write_acceptance_result only slugifies `client`, not `client_version`).
    try:
        out = subprocess.run(
            [claude_bin, "--version"], capture_output=True, text=True, timeout=10, check=False,
        )
        match = re.search(r"\d+\.\d+\.\d+", out.stdout)
        return match.group(0) if match else (out.stdout.strip() or "unknown")
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
        prompt = "Read /tmp/acceptance-test.txt and tell me what it says."
        launch_spec = ClaudeCodeIntegration().build_launch(
            AgentLaunchContext(
                route="acceptance",
                gateway_url=handle.base_url,
                model_name="acceptance-test-model",
                session_credential="acceptance-test-token",
                # NOTE: the exact non-interactive invocation is the part
                # most likely to need adjusting against the real CLI's
                # current interface — `--print`/`-p` is Claude Code's
                # documented non-interactive "print mode" flag as of this
                # writing. These land after `--model <alias>` in
                # launch_spec.command, same placement build_launch would
                # produce for `interop run claude -- --print "..."`.
                extra_args=("--print", prompt),
            ),
        )
        env = {**os.environ, **launch_spec.env}

        # launch_spec.command[0] is the placeholder binary name ("claude");
        # substitute the real resolved path so this exercises the exact
        # argv (including the --model flag) that `interop run claude`
        # would actually launch, not a hand-built stand-in for it.
        cmd = [claude_bin, *launch_spec.command[1:]]
        proc = subprocess.run(
            cmd,
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
