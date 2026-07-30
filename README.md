# Interop

## Agent Compatibility Gateway — local LLM compatibility layer for coding agents

Interop sits between a coding agent and a local inference backend and
translates between the wire formats each side expects.

### Why this exists

The trigger for this project was simple: pointing Claude Code, Codex, and
other coding agents at a local model through Ollama, and watching tool
calls fail — not because the model couldn't reason about the task, but
because it was speaking a slightly different tool-calling dialect than the
agent expected. A malformed JSON argument here, a missing envelope tag
there, a model that emits `<tool_call>` when the agent is listening for a
native function-call block. The task would stall or silently do the wrong
thing.

That's a real problem, not a cosmetic one: the entire point of running a
model locally — privacy, cost, control, no API bill — evaporates if the
agent built around it can't actually get anything done. A local model that
can't reliably call tools isn't a lighter-weight alternative to a hosted
one; it's just broken for agentic use, which is most of what coding agents
are for.

Interop exists to close that gap: it sits between the agent and the
backend, translates protocols, normalizes and repairs tool calls within
bounded and auditable limits, and is honest — via conformance testing and
declared-vs-verified capability reporting — about which models it can
actually make reliable and which ones it can't. See "Conformance levels"
and "Capability state" below for what that honesty looks like in practice.

### Client integration status (MVP)

Verification claims here are deliberately layered — "the gateway handles
this protocol correctly" and "this exact client binary works against it"
are different, separately-earned claims, and collapsing them into one
"Supported" label previously overstated what the test suite actually
proves. Claude Code is the one client with a test that launches and
drives the real binary end-to-end and passed; every other client stops at
the gateway-protocol or launch-spec level.

| Client | Backend | Verified as |
|--------|---------|-------------|
| Claude Code | Ollama | Reproducibly release-tested client (v2.1.220) |
| Codex | Ollama / OpenAI-compatible (vLLM, llama.cpp) | Gateway-tested protocol |
| Crush | Manual configuration | Unit-tested integration (no automatic launch) |
| Cline, OpenCode, Aider, Continue, Qwen Code | — | Unit-tested integration |

Verification tiers, weakest to strongest:

- **Unit-tested integration** — the launcher builds a correct launch spec
  (protocol, env vars, base URL, shim args) for this client and it's
  covered by unit tests against that spec, in isolation from a live
  gateway or backend.
- **Gateway-tested protocol** — the client's wire protocol (Anthropic
  Messages / OpenAI Chat / OpenAI Responses) is exercised end-to-end
  through a real `Gateway` instance against a fake or real backend in the
  test suite — request decode, tool-call extraction/repair, response
  encode — but not through the actual client binary.
- **Manually tested client** — a developer has run the real client
  binary against Interop by hand and confirmed the tool loop works.
- **Reproducibly release-tested client** — an automated, opt-in
  acceptance test invokes the real client binary end-to-end and is part
  of the release gate. Currently earned by Claude Code only — see
  `acceptance/results/claude-code-2.1.220.json` and
  `tests/acceptance/test_real_client_claude.py`.

MVP scope is Linux, Python 3.11+, loopback ingress, and one route per
process. Multi-route operation, remote ingress exposure, and any client
above are not part of the tested MVP surface beyond the tier listed — the
code paths exist but haven't been proven at a higher tier.

### Known gaps

- No test in this repository launches or drives a real `codex` binary.
  Codex, Crush, and the generic-integration clients are verified at the
  launch-spec or gateway-protocol level, not against the actual client
  process — see `tests/acceptance/test_real_client_codex.py` for the
  written-but-unrun harness.

### Quick start

```bash
pip install agent-interop
interop install
ollama launch claude --model qwen3-coder
```

The PyPI distribution is `agent-interop`; the CLI command stays `interop`.
The importable Python package is `agent_interop` (`import agent_interop`,
`from agent_interop.config import ...`) — this was renamed from the
original `interop` specifically so its top-level module name can't collide
with an unrelated third-party distribution that happens to also ship a
package literally named `interop`.

`interop install` puts a script literally named `ollama` at the front of
your PATH, shadowing the real `ollama` binary — every invocation of the
`ollama` command on this system, not just ones you type yourself, runs this
wrapper from then on. `ollama launch <agent>` is intercepted and routed
through Interop's format translation layer; every other subcommand (serve,
pull, push, etc.) execs straight through to the real binary unmodified.
`interop uninstall` removes the wrapper and restores the real binary at the
front of PATH. If you'd rather not modify PATH resolution at all, use
`interop run <agent>` instead — same result, no shim, no `ollama install`.

Contributing to Interop itself (not just using it) needs the dev extras —
see [Development](#development) below.

Not every model can be made fully reliable this way — some tool-call
defects are ambiguous and Interop deliberately won't guess at them (see
"Known gaps" and the extraction-safety notes throughout this file), so
effectiveness still varies by model. Use `interop test <model>` to check a
given model's conformance level (experimental — see "Evidence and
certification" below) instead of assuming.

### Architecture

```
ollama launch claude
  │
  ▼
Interop shim (intercepts launch subcommand)
  │
  ▼
Interop Gateway (protocol translation)
  ├── Client protocol adapters (Anthropic Messages, OpenAI Chat, OpenAI Responses)
  ├── Model-specific template rendering
  ├── Tool-call parsing (Hermes, Qwen, DeepSeek, Mistral, Llama, generic JSON)
  ├── Schema validation + bounded repair
  ├── Capability detection + per-route conformance levels
  └── Loop detection
  │
  ▼
Ollama / vLLM / llama.cpp
  │
  ▼
Local model
```

### Status

#### Implemented

- Protocol translation: Anthropic Messages ↔ OpenAI Chat ↔ OpenAI Responses,
  streaming and non-streaming, exercised through full ASGI-level tests (real
  HTTP requests against the FastAPI app, not just unit-level Gateway calls)
- Tool-call parsing for Hermes, Qwen, DeepSeek, Mistral, Llama, and generic
  JSON envelope dialects — generic/bare JSON tool-call scanning is **off by
  default** (opt-in per profile) because it can misinterpret ordinary JSON in
  model output as a tool call
- Schema validation with bounded cursor-scoped repair (one-issue/one-mutation
  at root level; nested `$ref`/`oneOf`/`anyOf` paths not yet supported)
- Streaming support for all three protocol adapters, including token usage
  emission (`usage_update`) before the terminal event, with error visibility
  and readiness/liveness endpoints (`/health/live`, `/health/ready`)
- Conformance test suite with 12 tests across cumulative L0-L4 levels
- Backends: Ollama (via upstream codecs), OpenAI-compatible (via upstream
  codecs, covers vLLM and llama.cpp)
- Shim installation for `ollama launch` interception, with an install
  manifest so uninstall restores the exact prior wrapper
- Config validation on startup, with a config schema version and a
  round-tripping `interop init` → load cycle
- History reconciliation with sequential pairing, safety checks, and
  deterministic ID synthesis (stable across retries of the same request)
- Centralized error registry wired into all protocol adapters, with
  credential/secret redaction applied to every client-visible error message
- Malformed stream frame handling with typed markers, a bounded retry
  threshold, and client-safe error details
- Loop detection wired into gateway response assembly, scoped per
  `(session, route)` and never triggered for a client that supplied no
  session identifier
- Request execution coordinator in both streaming and non-streaming paths —
  bookkeeping (evidence write-back, execution finalization) completes before
  the terminal event is yielded, not after
- Fenced-code masking applied to all textual extractors (Hermes, Mistral,
  Llama, generic)
- Structured tool corrections in rejection error details
- Backend constraints from codec (OpenAI Chat: 64-char names, 128 tools)
- systemd user service management (`interop service install/start/stop/logs`)

#### Experimental

- `interop certify` / evidence recording / replay — evidence is disabled by
  default and never activates compatibility-pack trust automatically; see
  "Evidence and certification" below
- `/v1/capabilities` — declared metadata only (see "Capability state" below),
  not a verified guarantee
- Multi-route fan-out (gateway serves multiple models simultaneously)
- MCP diagnostics and MCP tool-schema helpers (present in the codebase,
  not wired into any production request path)

#### Explicitly deferred scope

These items are architecturally bounded but not implemented:

1. **`$ref` resolution and `oneOf`/`anyOf` branch disambiguation** — Nested repair
   supports directly-addressable object and array paths. Repairs inside unresolved
   `$ref` targets or ambiguous `oneOf`/`anyOf` branches are rejected without mutation.

2. **Probe-derived model digest and quantization** — These fields depend on
   backend-specific discovery APIs. Fields remain empty rather than invented.
   Evidence is not marked verified when required identity dimensions are unavailable.

3. **Persistent multi-turn session management** — Loop detection uses a
   client-supplied session ID when present. No persistent session store or
   lifecycle contract exists beyond the bounded in-process LRU cache, and a
   request with no session ID gets no session tracking at all (by design —
   see "Implemented" above).

#### Planned (not yet implemented)

- Wiring `testing/conformance.py`'s cumulative L0-L4 level calculation
  (`_compute_level`/`_cumulative_requirements` — the algorithm itself
  already exists) into the live `interop test`/`interop certify` path and
  `/v1/capabilities`, which currently report the separate
  `ToolCapabilityLevel`/`AgentCapabilityLevel` scheme from capabilities.py
  instead
- Additional model profiles beyond the current packaged set
- Destination-aware request validation from live backend probe metadata

### Conformance levels

Interop classifies models into cumulative conformance levels. Each level
requires **all** behaviors of the previous level plus new ones. Levels are
determined by the conformance test suite, not assumed.

| Level | Required behavior |
|-------|-------------------|
| L0 | Chat only — no tool support |
| L1 | Call an explicitly-named tool with correct arguments |
| L2 | L1 + Automatically select the right tool + avoid tools when unnecessary |
| L3 | L2 + Sequential tool calls + error recovery + structured/nested arguments |
| L4 | L3 + Parallel tool calls + edit-and-verify cycles + distinct call IDs |

Use `interop test <model>` to run the conformance suite against a model.
The `/v1/capabilities` endpoint reports per-route level and degraded reason
— as **declared** metadata, not a verified result (see below).

### Capability state

`/v1/capabilities` reports declared capability state per route, derived from
static model-profile and codec metadata:

```json
{
  "source": "declared_profile_metadata",
  "verified": false,
  "capability_model": {
    "<route_id>": {
      "tool_level": {"value": "..."},
      "agent_level": {"value": "..."},
      "capabilities": {
        "<capability_name>": {"state": "declared|unsupported|...", "details": {}}
      },
      "compatibility": {"status": "...", "missing_capabilities": [], "warnings": [], "remediation": []}
    }
  }
}
```

`state` follows `unsupported → declared → probed → verified/degraded →
user-forced`. Only `probed`, `verified`, and `user-forced` count as actually
available (`CapabilityState.is_available()`); `declared` means only that
model/codec metadata *claims* the capability — nothing has confirmed it
against a live backend. Live conformance results (`interop certify`) are
recorded as evidence but do not automatically upgrade what this endpoint
reports — see "Evidence and certification" below.

### Evidence and certification (experimental)

The evidence store is **disabled by default** — it must be explicitly
enabled in config (`evidence.enabled: true`) before anything is recorded.
`interop certify` runs the conformance battery against a real backend and
records observations, but it does **not** mark results as manually verified
— that flag is reserved for actual human review, and an automated CLI run
conflating the two would let a passing suite silently activate trust it
hasn't earned. Treat `certify`, `evidence`, and `replay` as experimental
tooling for now, not a production trust mechanism.

### Install

```bash
# One-time
interop install

# Now ollama launch routes through Interop
ollama launch claude --model qwen3-coder

# Verify
interop status
```

### Development

```bash
git clone ...
cd interop
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Before opening a PR, run the release gate locally — the same checks CI runs
(lint, types, tests with coverage, wheel build/install/import, config
round-trip, CLI smoke tests):

```bash
./scripts/release.sh --check
```

(`--check` allows a dirty working tree for local iteration; drop the flag to
run the exact gate a real release requires.)

### Acknowledgements

Some of the development work on this project — testing changes against
real local models, iterating on tool-call parsing across the different
dialects Interop supports — used inference capacity provided by
[FreeInference.org](https://freeinference.org).

FreeInference did not commission, direct, fund, or pay me for this work.
No representative of FreeInference reviewed or approved this contribution,
and this acknowledgment does not indicate sponsorship, partnership, or
endorsement by FreeInference or any affiliated organization.

I'm acknowledging the service because access to capable inference
infrastructure can make meaningful open-source development more
accessible to developers and researchers who don't have the hardware or
budget to run these models themselves. Their inference meaningfully
contributed to the production of this work.

Organizations able to provide GPU capacity, hardware, cloud credits,
research funding, or other infrastructure resources should consider
supporting FreeInference so it can keep making this kind of capability
available for open-source development, research, and education. That's
the point of this note — not to send free-tier traffic their way, but to
direct attention toward supporting what makes work like this possible.
