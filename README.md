# Interop

**Agent Compatibility Gateway** — lets coding agents talk to local LLMs without format mismatch.

## What it does

Coding agents (Claude Code, Codex, etc.) expect tool calls in a specific format. Local models rarely produce that format exactly — different envelope tags, bare JSON instead of structured blocks, malformed arguments. The agent stalls or silently does the wrong thing.

Interop sits between the agent and the backend and handles the translation:

```
Claude Code ─── Anthropic Messages ─┐
Codex ───────── OpenAI Responses ───┤                                     ┌─── Ollama
Other agents ── OpenAI Chat ─────────┤                                     ├─── vLLM
                                     ▼                                     └─── llama.cpp
                              Interop Gateway
                              ┌──────────────────────┐
                              │ Protocol translation  │
                              │ Tool-call extraction  │
                              │ Bounded repair        │
                              │ Conformance testing   │
                              └──────────────────────┘
                                     ▼
                                Local model
```

Specifically:
- **Protocol translation** between Anthropic Messages, OpenAI Chat, and OpenAI Responses.
- **Tool-call extraction** from model-native output (Hermes, Qwen, Mistral, DeepSeek, Llama, generic JSON dialects).
- **Bounded repair** of malformed tool calls within auditable limits (never changes tool selection, never invents argument values).
- **Conformance testing** that reports what a model can actually do, not what its profile claims.

## What it doesn't do

- **Make a model reason better.** Interop fixes format mismatches. It can't make a model smarter, more capable, or more reliable than it actually is.
- **Fix semantic errors.** If a model calls the right tool with wrong arguments, Interop won't correct the values. It only fixes structural issues (malformed JSON, wrong envelope format).
- **Guarantee reliability.** Effect varies by model. A 7B model will reliably fail tasks that a 14B model handles. Interop reports this honestly via conformance testing rather than hiding it.

## Install

```bash
pip install agent-interop
interop install
```

`interop install` places a wrapper at the front of PATH that intercepts `ollama launch <agent>` and routes it through Interop. All other `ollama` subcommands pass through unchanged. Use `interop uninstall` to remove.

## Use

```bash
# Start the gateway
interop start --model qwen3-coder --backend ollama

# Or intercept ollama launch
ollama launch claude --model qwen3-coder
```

### Run the demo

```bash
# Terminal 1: start the gateway
interop start --model qwen3-coder --backend ollama

# Terminal 2: run the demo
python3 demo/fix_bug_demo.py
```

The demo creates a buggy calculator, sends it to a local model through Interop's Anthropic Messages translation, and watches the model use tools to read, fix, and verify the code.

## Conformance levels

Interop classifies models by running a real test battery, not by reading profile metadata.

| Level | What it means |
|-------|---------------|
| L0 | Chat only — no tool support |
| L1 | Call an explicitly-named tool with correct arguments |
| L2 | L1 + automatically select the right tool + avoid tools when unnecessary |
| L3 | L2 + sequential calls + error recovery + nested arguments |
| L4 | L3 + parallel calls + edit-and-verify cycles + distinct call IDs |

Levels are cumulative — L3 requires all L1+L2+L3 tests to pass.

**What the tests actually measure:** format compliance (does the model emit tool calls in the right shape?) and basic tool-selection behavior. They don't measure reasoning quality, instruction-following precision, or whether the model can actually complete complex multi-step tasks. A model can pass L4 and still produce plausible-looking tool calls with wrong argument values.

### Verified results

| Model | Score | Level | Notes |
|-------|-------|-------|-------|
| qwen2.5-coder:14b | 12/12 | L4 | Passes all conformance tests |
| qwen2.5-coder:7b | 11/12 | L1 | Blocked on `malformed_call_repair` — model drops words from argument values |
| deepseek-coder:6.7b | 3/12 | L1 | Base model, limited instruction-following |
| mistral:7b | 3/12 | L1 | Base model, not fine-tuned for tool calling |

These scores reflect format compliance, not general capability. The 7B model can actually fix a real bug when driven through Interop (see the demo) — it just fails one specific conformance test because it drops words from argument values. The tests measure one thing; real tasks measure another.

## Client integration

| Client | Wire protocol | Launch |
|--------|---------------|--------|
| Claude Code | Anthropic Messages | `interop run claude` or `ollama launch claude` |
| Codex | OpenAI Responses | Manual config |
| hermes-agent | OpenAI Chat | `interop run hermes` |
| Generic OpenAI | OpenAI Chat | Point `OPENAI_BASE_URL` at Interop |

Only Claude Code has been tested end-to-end against the real binary. Other clients are verified at the protocol level (correct request/response translation) but not necessarily against the actual client binary.

## Development

```bash
git clone https://github.com/B-A-M-N/agent-interop
cd agent-interop
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest
```

Before submitting a PR, run the release gate:

```bash
./scripts/release.sh --check
```

## Acknowledgements

Some of the development and testing of this project — verifying behavior against real local models, iterating on tool-call parsing across the different dialects Interop supports — used inference capacity provided by [FreeInference.org](https://freeinference.org).

FreeInference did not commission, direct, fund, or pay for this work. No representative of FreeInference reviewed or approved this contribution, and this note does not indicate sponsorship, partnership, or endorsement by FreeInference or any affiliated organization.

The inference service access meaningfully contributed to this work. Organizations or individuals able to support open-source infrastructure (GPU capacity, cloud credits, research funding) should consider contributing to FreeInference to keep this capability available for developers and researchers who need it.
