# Real-client acceptance tests

These tests are the one verification tier Interop has not yet earned for
any client (see the root `README.md`'s "Client integration status" table
and `RELEASE.md`'s "Alpha vs. supported release track"): an automated run
that actually launches the real client **binary** — not just the gateway
protocol or a hand-built launch spec — and drives one real tool-call round
trip through it.

**They have never been executed in this development sandbox.** There is no
real `claude`/`codex` binary or credentials available here. They are
written and reviewed, not proven by a real run — the first person who
actually runs one of these against a real client binary should expect to
adjust the exact subprocess invocation (CLI flags, output parsing) to match
that binary's current interface; the harness (`_harness.py`) around it —
starting a real Interop server, wiring a deterministic fake upstream, and
writing the result record — is what's actually load-bearing and reviewed.

## Running

Every module here is skipped by default and by CI. To run one:

```bash
INTEROP_ACCEPTANCE_CLAUDE_BIN=$(which claude) \
  uv run pytest tests/acceptance/test_real_client_claude.py -v

INTEROP_ACCEPTANCE_CODEX_BIN=$(which codex) \
  uv run pytest tests/acceptance/test_real_client_codex.py -v
```

Each test starts a real Interop gateway on a fixed local port
(`127.0.0.1:18091`/`18092`), wires a scripted fake upstream into it (so no
real Ollama/vLLM/llama.cpp backend or model is needed), launches the real
client binary with the exact `LaunchSpec` `interop run <agent>` would use,
and asserts the round trip completes.

## Result format

A successful run writes `acceptance/results/<client-slug>-<version>.json`:

```json
{
  "client": "Claude Code",
  "client_version": "1.2.3",
  "timestamp": "2026-07-29T12:00:00+00:00",
  "scenario": "single_tool_round_trip",
  "passed": true,
  "detail": "..."
}
```

`scripts/check_support_claims.sh` (part of the release gate) fails the
build if the root `README.md`'s client-status table ever claims a
"release-tested" tier for a client without a matching file here — that is
the only mechanism by which a client is allowed to graduate off the
alpha/unverified track. A failing run should NOT be deleted; leaving it in
place (with `"passed": false`) is a legitimate, honest record that the
last attempt didn't work.
