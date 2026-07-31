# Agent Documentation: Interop

## Overview
**Interop** is an **Agent Compatibility Gateway** that sits between coding agents (e.g., Claude, Codex) and local inference backends (e.g., Ollama, vLLM, llama.cpp). It translates between the formats each side expects, normalizing tool calls, translating protocols, and checking capability conformance for local LLM coding agents.

It exists because a local model that can't reliably call tools isn't a viable substitute for a hosted one in an agentic coding workflow — the model may reason about the task correctly and still fail to act on it, because its tool-calling dialect doesn't match what the agent expects. See the README's "Why this exists" section for the full motivation.

### Key Features
- **Protocol Translation**: Converts between Anthropic Messages, OpenAI Chat, and OpenAI Responses protocols.
- **Tool-Call Normalization**: Parses and repairs tool calls from models like Qwen, Hermes, Mistral, Llama, and DeepSeek.
- **Capability Detection**: Classifies models into capability levels (L0-L4) based on real conformance-run evidence (`interop test`/`interop certify`, see agent_interop.testing.levels) and static profile metadata (`/v1/capabilities`'s `declared_profile_metadata` block). `interop status` reports installer/shim status only — it does not check model capability level; use `interop evidence list --route/--model` to see recorded compatibility evidence.
- **Model-Specific Rendering**: Uses declarative model profiles for chat templates, tool-call parsers, and repair strategies.
- **Streaming Support**: Handles both streaming and non-streaming requests.

### Architecture
```
ollama launch claude
  │
  ▼
Interop Shim (intercepts launch subcommand)
  │
  ▼
Interop Gateway (protocol translation)
  ├── Client protocol adapters (Anthropic Messages, OpenAI Chat, OpenAI Responses)
  ├── Model-specific template rendering
  ├── Tool-call parsing (Hermes, Qwen, DeepSeek, Mistral, Llama, generic JSON)
  ├── Schema validation + bounded repair
  ├── Capability detection + conformance levels
  └── Loop detection
  │
  ▼
Ollama / vLLM / llama.cpp
  │
  ▼
Local model
```

---

## Commands
### Essential Commands
| Command | Description | Example |
|---------|-------------|---------|
| `pip install -e ".[dev]"` | Install Interop in development mode (CLI/server deps are unconditional base dependencies, not separate extras). | `uv pip install -e ".[dev]"` |
| `pytest` | Run the test suite. | `pytest` |
| `interop start` | Start the Interop gateway server. | `interop start --model qwen3-coder --backend ollama` |
| `interop install` | Install the Interop shim to intercept `ollama launch` commands. | `interop install` |
| `interop status` | Verify the Interop installation/shim status (not model capability — see `interop evidence`). | `interop status` |
| `interop test {model}` | Run conformance tests for a specific model and compute its L0-L4 level. | `interop test qwen3-coder` |
| `interop evidence list --route/--model` | List recorded compatibility evidence (every distinct compatibility key, never collapsed). | `interop evidence list --model qwen3-coder` |

### CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Bind address for the gateway server. | `127.0.0.1` |
| `--port` | Listen port for the gateway server. | `8090` |
| `--backend` | Backend type (`ollama`, `vllm`, `llamacpp`, `openai`, `anthropic`, `openai_compatible`). | `ollama` |
| `--backend-url` | Backend server URL. | `http://127.0.0.1:11434` |
| `--model` | Model name to use. | `qwen3-coder` |
| `--probe/--no-probe` | Probe backend on startup. | `True` |
| `--log-level` | Log level (`debug`, `info`, `warn`, `error`). | `info` |
| `--backend-timeout` | Backend request timeout (seconds). | `120.0` |

---

## Code Organization
### Directory Structure
```
Interop/
├── src/
│   └── agent_interop/          # PyPI dist "agent-interop"; import name "agent_interop"; CLI command "interop"
│       ├── __init__.py
│       ├── abi.py              # Canonical ABI types (messages, blocks, requests)
│       ├── auth.py             # Authentication
│       ├── capabilities.py     # Capability level definitions
│       ├── cli.py              # CLI entrypoint and commands
│       ├── config.py           # Configuration loading
│       ├── context.py          # Request context and client identity
│       ├── enums.py            # Shared enumerations
│       ├── errors.py           # Custom errors and exceptions
│       ├── gateway.py          # Core gateway engine
│       ├── install.py          # Shim installation logic
│       ├── launcher.py         # Launcher logic for intercepting `ollama launch`
│       ├── session.py          # Session management
│       ├── transaction.py      # Transaction tracking
│       ├── types.py            # Legacy types (deprecated, use abi.py)
│       ├── extraction.py       # Tool-call extraction from model output
│       ├── request_validation.py
│       ├── agents/             # Agent-specific adapters (claude_code, codex)
│       ├── backends/           # Ollama admin/shim helpers (routing lives in upstreams/ + model/registry.py)
│       ├── context_budget/     # Runtime-aware token estimation and compaction planning
│       ├── controller/         # Compatibility-controller contracts and bounded state
│       ├── compatibility_packs/ # Per-agent compatibility shims
│       ├── data/profiles/      # Single source of truth for model profiles (YAML)
│       ├── evidence/           # Evidence store for conformance tracking
│       ├── execution_attempts/ # Bounded direct/adapted/controller fallback ladder
│       ├── history/            # History ledger and reconciliation
│       ├── mcp/                # MCP diagnostics and schemas
│       ├── model/              # Model profile registry and rendering
│       │   ├── contract_templates.py  # PROMPTED-mode contract text
│       │   ├── profiles_v2.py  # YAML profile loader (current)
│       │   └── registry.py     # Profile resolution registry
│       ├── parsing/            # JSON scanning and parsing utilities
│       ├── plugin/             # Plugin system for extensibility
│       ├── planning/           # Direct/adapted/controlled request compatibility planning
│       ├── protocols/          # Client protocol adapters
│       │   ├── anthropic_messages.py
│       │   ├── base.py
│       │   ├── openai_chat.py
│       │   ├── openai_responses.py
│       │   └── registry.py
│       ├── repair/             # Schema validation and repair pipeline
│       ├── qualification/      # Side-effect-free bootstrap model qualification
│       ├── replay/             # Request replay and comparison
│       ├── schemas/            # JSON schemas
│       ├── server/             # FastAPI server for the gateway
│       │   └── app.py
│       ├── streaming/          # Streaming coordinator (replaces the old stream.py)
│       ├── testing/            # Conformance test suite
│       │   ├── conformance.py  # Standalone syntactic validator (not wired — see levels.py)
│       │   ├── levels.py       # Real-battery -> L0-L4 mapping (wired into cli.py/`/v1/capabilities`)
│       │   ├── fake_upstream.py
│       │   └── runner.py       # RealConformanceRunner + the real 12-test battery
│       ├── tool/               # Tool-call normalization
│       ├── tool_surface/       # Deterministic model-visible tool selection
│       ├── transport/          # HTTP, NDJSON, SSE transport
│       └── upstreams/          # Upstream codec interfaces (routing/backend resolution lives here)
│           ├── anthropic.py
│           ├── codec.py
│           ├── ollama_chat.py
│           ├── openai_chat.py
│           ├── openai_responses.py
│           └── registry.py
├── tests/
├── pyproject.toml
├── README.md
├── PROPOSED_CHANGES.md         # Architecture plan (reference, not all implemented)
└── AGENTS.md                   # Agent documentation (this file)
```

### Key Components
| Component | Description |
|------------|-------------|
| **Gateway** (`gateway.py`) | Core engine that orchestrates protocol translation, model calls, and response conversion. |
| **Model Profiles** (`model/profiles_v2.py`, `data/profiles/`) | Declarative YAML definitions for supported models. `data/profiles/` is the single source of truth. |
| **Protocol Adapters** (`protocols/`) | Client protocol adapters for Anthropic Messages, OpenAI Chat, and OpenAI Responses. |
| **Upstream Codecs / Backend Adapters** (`upstreams/`) | Per-backend codecs (Ollama, vLLM, llama.cpp, OpenAI-compatible, Anthropic) that render requests and decode responses, preserving raw tool-call arguments; `backends/` itself only holds Ollama shim/admin helpers now. |
| **Compatibility Planning** (`planning/`, `context_budget/`, `tool_surface/`) | Builds direct/adapted/controlled path candidates from client requirements, live runtime facts, bootstrap qualification evidence, context capacity, and deterministic tool visibility. |
| **Runtime Inspection** (`backends/`) | Backend inspectors collect model digest, template, context allocation, and capability states without treating codec support as model proof. |
| **Embeddable Runtime** (`plugin/runtime.py`) | `InteropRuntime` exposes the gateway's inspect, plan, generate, stream, qualify, explain, and replay operations for in-process integrations. |
| **Diagnostics & Replay** (`replay/`, `paths.py`) | Failed/repaired requests can retain bounded, recursively-sanitized replay metadata in memory or schema-v2 durable state, retrievable by case ID. |
| **Tool-Call Extraction** (`extraction.py`, `parsing/`) | Extracts tool calls from model output (Hermes, Qwen, Mistral, etc.). |
| **Validation & Repair** (`repair/`) | Schema validation and bounded repair of malformed tool calls. |
| **Conformance Tests** (`testing/`) | Keeps the direct-model L0-L4 battery and adds bounded path suites (`testing/suites.py`) for adapted, controller, primary-worker, and client-contract validation. |
| **Transport** (`transport/`) | HTTP, NDJSON, and SSE transport layer. |
| **Streaming** (`streaming/`) | Streaming coordinator for multi-protocol streaming responses. |

---

## Conventions & Patterns
### Naming Conventions
- **Enums**: Use `PascalCase` for enum names and `UPPER_CASE` for enum values (e.g., `CapabilityLevel.L0`).
- **Classes**: Use `PascalCase` for class names (e.g., `Gateway`, `ModelProfile`).
- **Functions/Methods**: Use `snake_case` for function and method names (e.g., `handle_request`, `parse_tool_calls`).
- **Variables**: Use `snake_case` for variable names (e.g., `canonical_request`, `tool_call`).
- **Constants**: Use `UPPER_CASE` for constants (e.g., `BUILTIN_PROFILES`).

### Code Patterns
- **Async/Await**: Heavy use of `async/await` for non-blocking I/O operations (e.g., HTTP calls, streaming).
- **Dataclasses**: Use `dataclasses` for immutable data structures (e.g., `CanonicalRequest`, `ToolCall`).
- **Enums**: Use `Enum` for fixed sets of values (e.g., `CapabilityLevel`, `ProtocolKind`).
- **Pydantic**: Use `pydantic` for data validation and serialization. Note: `types.py` is legacy; new code should use `abi.py` for canonical types.
- **Type Hints**: Extensive use of type hints for static analysis and readability.

### Tool-Call Dialects
| Dialect | Parser ID | Description |
|---------|-----------|--------------------|
| Hermes | `hermes` | `<tool_call>JSON</tool_call>` envelope |
| Qwen | `qwen` | Qwen-tool-v1 prompted format |
| DeepSeek | `deepseek` | DeepSeek tool-call format |
| Mistral | `mistral` | Mistral function-call format |
| Llama | `llama` | Llama text-format tool calls |
| Generic JSON | `tool_call_envelope` | Fallback envelope-based parser |

---

### Conformance Levels

Conformance levels are **cumulative** -- each level requires all its own tests plus all lower-level tests to pass. The real 12-test battery (`agent_interop.testing.runner.get_standard_tests`) is mapped to levels in `agent_interop.testing.levels.TEST_LEVEL_MAP` (this is the only mapping actually wired into `interop test`/`interop certify`/`/v1/capabilities`; `agent_interop.testing.conformance`'s `LEVEL_REQUIREMENTS` is an older, standalone syntactic-validation module using different test names and is not wired anywhere).

| Level | Name | Required Tests |
|-------|-------|----------------------|
| L0 | No tools | N/A (baseline) |
| L1 | Explicit tools | `explicit_forced_tool`, `nested_arguments`, `malformed_call_repair` |
| L2 | Implicit tools | L1 + `automatic_tool_selection`, `no_tool_request` |
| L3 | Sequential workflow | L2 + `sequential_calls`, `tool_error_recovery`, `tool_result_continuation`, `history_round_trip` |
| L4 | Advanced | L3 + `parallel_calls`, `edit_and_verify`, `distinct_ids` |

A level result also carries a `battery_version` (invalidates stale evidence if the mapping changes), separates infra-inconclusive tests from real behavioral failures, and can be computed with the repair pipeline forced on or off (`interop test --repair`/`--no-repair`) so the model's own unaided level and the repair-assisted level are never conflated. Use `interop test <model>` or `interop certify` to run the conformance suite and determine a model's level.

---

### Implementation Status

- **Implemented**: Protocol translation (Anthropic Messages, OpenAI Chat, OpenAI Responses), tool-call parsing (Hermes, Qwen, generic), streaming, repair pipeline, conformance testing, profile-based rendering, managed-launch authentication, evidence store (key construction + lookup), tool transaction service, execution finalization.
- **Compatibility planning**: Runtime inspection, context/tool-surface planning, bounded attempt ladders, digest-scoped bootstrap qualification, and qualified controller routing are implemented. The controller may request bounded, private primary-worker refinements; those control calls are intercepted by Interop and never exposed to the coding client.
- **Hardened** (2026-07-25): Evidence write-back from live request path, Codex temp-file cleanup via `LaunchSpec.cleanup`, alias guard fix (`_is_key_sufficiently_populated` now requires model/profile dimensions), lockfile (`uv.lock`) for reproducible installs, release gate script (`scripts/release.sh`), ruff lint fixes, error registry completeness (`ROUTE_NOT_FOUND`, `SESSION_INVALID`, `SESSION_EXPIRED`), resource-leak cleanup (evidence store close, server lifespan error handling), execution finalization (fixed 3 streaming error paths missing `finalize_error()`), `asgi_lifespan` test dependency noted.
- **Experimental**: Upstream codecs, evidence-based conformance tracking, per-route capability state.
- **Planned**: See `PROPOSED_CHANGES.md` for the v2 architecture plan (reference document, not all items implemented).
