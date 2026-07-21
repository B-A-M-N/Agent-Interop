# Interop

## Agent Compatibility Gateway — local LLM compatibility layer for coding agents

Interop sits between coding agents (Claude Code, Codex, Cline, etc.) and
local inference backends (Ollama, vLLM, llama.cpp) and translates between
the formats each side expects.

### Quick start

```bash
pip install interop
interop install
ollama launch claude --model qwen3-coder
```

After `interop install`, `ollama launch <agent>` transparently routes through
Interop's format translation layer. Non-launch commands (serve, pull, etc.)
pass through normally.

### Why

Local models fail in coding agents because of format mismatches:

- Wrong chat templates
- Broken tool-call JSON
- Missing tool-call IDs
- Wrong stop tokens
- Bad streaming format
- No error recovery

Interop fixes all of this without the user having to think about it.

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
  ├── Capability detection + conformance levels
  └── Loop detection
  │
  ▼
Ollama / vLLM / llama.cpp
  │
  ▼
Local model
```

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
uv pip install -e ".[dev,cli,server]"
pytest
```