# Proposed Changes — Interop v2 Architecture & Implementation Plan

> **Historical design document. Not an implementation-status document.**
> This describes the plan as originally proposed — not all items here were
> implemented, and some were implemented differently than described.
> README.md and RELEASE.md define current supported behavior; AGENTS.md
> tracks current implementation status. Treat this file as design-intent
> history, not a source of truth for what Interop does today.

## Context

The existing codebase has all the *components* for a repair pipeline but they are not
correctly connected. The current plan tried to fix this but suffered from:

1. **No phase gates** — testing deferred to Phase 7, no proof obligation per phase.
2. **Dual type system** — Phase 1 kept v1 backward-compat types and added more bridges.
3. **Codec boundary leaks** — upstream response decoders parse into dicts, not semantic objects.
4. **Prompted mode undefined** — no specification of what the prompt contract actually contains.
5. **Phase overlap** — Phase 5 and 6 both claim streaming work.
6. **No principal contracts** — `InvocationPlan`, `DecodedModelResponse`, `RawToolCallCandidate`,
   and `ToolCallDecision` are not explicitly defined.

This document replaces the previous gap-list approach with a phased architecture plan
where each phase has a defined scope, a mandatory gate test, and explicit non-goals.

---

## Principal Contracts (defined before any implementation)

Every phase operates in terms of these four contracts. They are the architecture.

### Contract 1: `InvocationPlan`

```python
@dataclass(frozen=True)
class InvocationPlan:
    effective_tool_mode: ToolMode          # NATIVE | PROMPTED | TEXTUAL | DISABLED
    native_tools_enabled: bool             # send provider-native schemas
    prompt_contract: list[CanonicalContentBlock]  # injected tool system prompt
    parser_id: str | None                  # textual dialect for parsing model output
    output_envelope: str | None            # textual envelope (e.g. "<tool_call>...</tool_call>")
    constrained_output: ConstrainedOutputConfig | None  # grammar/format constraints
```

Mode semantics:

| Mode       | Native schemas | Injected prompt | Textual parser | Envelope |
|------------|----------------|-----------------|----------------|----------|
| NATIVE     | Yes            | No              | No             | No       |
| PROMPTED   | No             | Yes             | Yes            | Yes      |
| TEXTUAL    | No             | No              | Yes            | Yes      |
| DISABLED   | No             | No              | No             | No       |
| AUTO       | Resolved at runtime from route + profile + conformance data               |                |          |          |

### Contract 2: `DecodedModelResponse` / `DecodedModelEvent`

```python
@dataclass
class DecodedModelResponse:
    content: list[CanonicalContentBlock]
    tool_candidates: list[RawToolCallCandidate]
    stop_reason: str
    usage: CanonicalUsage
    extra: dict[str, Any]

@dataclass
class DecodedModelEvent:
    type: Literal["text", "tool_arguments", "tool_complete", "error", "done"]
    text: str = ""
    tool_ordinal: int = 0
    tool_id: str = ""
    tool_name: str = ""
    arguments: str = ""          # raw JSON fragments or full string
    error: str = ""
```

Key property: `tool_candidates` contains `RawToolCallCandidate` objects, not
`CanonicalToolCallBlock`. Malformed JSON survives in `raw_arguments`.

### Contract 3: `RawToolCallCandidate`

```python
@dataclass
class RawToolCallCandidate:
    id: str | None
    name: str | None
    raw_arguments: str | dict[str, Any] | list[Any] | None
    source_protocol: ProtocolKind | str
    source_index: int | None
    source_text: str = ""
    raw_name: str | None = None
```

Already defined in `abi.py`. The critical rule: **the codec must not modify
`raw_arguments`**. A malformed JSON string like `{"path":"/tmp/x",}` stays as
that exact string through the codec boundary.

### Contract 4: `ToolCallDecision`

```python
@dataclass
class ToolCallDecision:
    candidate: RawToolCallCandidate
    outcome: RepairOutcome
    accepted_block: CanonicalToolCallBlock | None  # None if rejected
```

---

## Phase 1: Canonical ABI Unification — Remove the Dual Type System

### Scope

Remove every production import of the v1 backward-compat types (`AgentMessage`,
`ContentBlock`, `ToolCall`, `ToolResult`, `BackendRequest`, `BackendEvent`) from
protocol adapters and upstream renderers. All construction goes through canonical
types only.

Explicit requirements:

- `anthropic_messages.py`, `openai_chat.py`, `openai_responses.py` construct only
  `CanonicalMessage`, `CanonicalTextBlock`, `CanonicalToolCallBlock`,
  `CanonicalToolResultBlock`, `CanonicalReasoningBlock`, `CanonicalImageBlock`,
  `CanonicalRefusalBlock`, `CanonicalUnknownBlock`.
- v1 types move to `interop.compat` module (production code must not import them).
- Upstream renderers (`upstreams/openai_chat.py`, `upstreams/ollama_chat.py`)
  remove all `dict`/`hasattr` duck typing — accept only canonical ABI types.
- Request adapters populate `CanonicalModelRef.requested_name`.
- Tool results preserve `tool_call_id`.
- Unknown/unsupported content blocks wrap as `CanonicalUnknownBlock(raw=block)`.
- Correct the `events` variable bug in `gateway.py:_process_chunk_stream` (line ~502:
  `events` is referenced before assignment — must be `events = []` at function start).

### Gate (must pass before Phase 2)

```python
# A static test that fails if any production module imports v1 compat types
def test_no_v1_imports_in_production():
    production_modules = [
        "interop.protocols.anthropic_messages",
        "interop.protocols.openai_chat",
        "interop.protocols.openai_responses",
        "interop.gateway",
        "interop.upstreams.openai_chat",
        "interop.upstreams.ollama_chat",
        "interop.server.app",
    ]
    for mod_name in production_modules:
        mod = import_module(mod_name)
        source = getsource(mod)
        for v1_type in ("AgentMessage", "ContentBlock", "ToolCall", "ToolResult", "BackendRequest", "BackendEvent"):
            # Only flag imports, not string literals or comments
            if re.search(rf"(?m)^from.*import.*\b{v1_type}\b", source):
                pytest.fail(f"{mod_name} imports deprecated {v1_type}")
```

### Non-goals

- Do NOT add new conversion helpers between v1 and canonical types.
- Do NOT refactor streaming (Phase 6 owns that).
- Do NOT change the repair pipeline or schema validation.

---

## Phase 2: Upstream Codec Interfaces + Protocol Implementations

### Scope

Define and implement a `ModelCodec` interface. Every upstream protocol gets its own
codec that owns:

| Responsibility | Detail |
|---|---|
| Endpoint path | `/api/chat`, `/v1/chat/completions`, etc. |
| Required headers | Content-Type, Authorization, provider-specific |
| Request rendering | CanonicalRequest → upstream-native dict |
| Response decoding | Upstream-native dict → `DecodedModelResponse` |
| Streaming decoding | Upstream-native chunk → `DecodedModelEvent` |
| Usage conversion | Upstream usage raw → `CanonicalUsage` |
| Stop-reason conversion | Upstream finish → canonical stop_reason |
| Tool-call extraction | Upstream tool calls → `list[RawToolCallCandidate]` |

Required codecs:

| Codec | Endpoint | Upstream |
|---|---|---|
| `ollama_chat` | `/api/chat` | Ollama |
| `openai_chat` | `/v1/chat/completions` | vLLM, llama.cpp, OpenAI-compatible |
| `openai_responses` | `/v1/responses` | OpenAI |
| `anthropic_messages` | `/v1/messages` | Anthropic API |

Critical rule: **Response decoding must preserve `raw_arguments` exactly as the
upstream sent them.** The codec output is `DecodedModelResponse`, not
`CanonicalResponse`. The tool-call list is `list[RawToolCallCandidate]`, not
`list[CanonicalToolCallBlock]`.

### Gate

```python
def test_every_upstream_codec_preserves_malformed_json():
    """Each codec must return a DecodedModelResponse whose tool_candidates
    contain the original malformed raw_arguments string verbatim."""
    for codec_name, codec, raw_body in all_codec_fixtures():
        decoded = codec.decode_response(raw_body)
        for candidate in decoded.tool_candidates:
            assert isinstance(candidate, RawToolCallCandidate)
            # If the upstream sent malformed JSON, it must survive
            if test_malformed[codec_name]["raw"] == '{"path":"/tmp/x",}':
                assert candidate.raw_arguments == '{"path":"/tmp/x",}'
```

---

## Phase 3: Unified Profiles, Capability Resolution, InvocationPlan

### Scope

Build the decision layer that determines how tools are presented to the model and
how output is parsed.

Key components:

1. **Capability resolution**: route config + profile capability + backend capability
   → resolved `ToolMode` and `InvocationPlan`.

2. **`InvocationPlan` construction**: given a resolved mode and tool list, produce
   the exact prompt contract: injected tool definitions and the textual envelope.

3. **Deterministic serialization**: tool schemas serialized deterministically so
   repeated requests remain prompt-cache friendly (sorted keys, stable JSON).

Default textual contract when `PROMPTED` is selected:

```
To call a tool, emit exactly:

<tool_call>{"name":"tool_name","arguments":{"key":"value"}}</tool_call>

Text outside a <tool_call> block is ordinary assistant text.
A block means the tool is intended to execute.

Available tools:

<tool name="read_file">
{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}
</tool>
```

4. **Mode resolution rules**:

| Route config | Profile capability | Resolved mode |
|---|---|---|
| `AUTO` | Has structured tool support | `NATIVE` |
| `AUTO` | Textual dialect declared | `PROMPTED` |
| `AUTO` | L0 (chat only) | `DISABLED` |
| `NATIVE` | (any) | `NATIVE` |
| `PROMPTED` | (any) | `PROMPTED` |
| `TEXTUAL` | (any) | `TEXTUAL` |

### Gate

```python
def test_prompted_mode_contract():
    """Prompted mode produces a reproducible prompt contract with
    tool schemas, envelope, and no native tool_choice."""
    plan = build_invocation_plan(
        tools=sample_tools,
        mode=ToolMode.PROMPTED,
    )
    assert plan.native_tools_enabled is False
    assert plan.output_envelope == "tool_call"
    assert plan.parser_id is not None
    assert len(plan.prompt_contract) > 0  # contains tool descriptions
    # Verify determinism
    assert plan.prompt_contract == build_invocation_plan(
        tools=sample_tools, mode=ToolMode.PROMPTED
    ).prompt_contract
```

---

## Phase 4: Universal Tool Transaction Service

### Scope

Build the single authority for all tool-call processing. All sources produce
`RawToolCallCandidate`. One service performs the entire pipeline:

```
candidate
  → identify/canonicalize tool identity
  → recover outer call structure (envelope extraction)
  → recover argument JSON (syntax recovery)
  → validate against schema
  → deterministic repair
  → revalidate
  → optional regeneration
  → revalidate
  → accept or reject
```

### Sources of RawToolCallCandidate (all map here):

- Native structured provider output (Phase 2 codecs)
- Hermes `<tool_call>...</tool_call>` envelopes
- Qwen XML syntax
- Mistral inline JSON
- DeepSeek function_call blocks
- Prompted generic envelopes
- Complete streaming accumulations (from Phase 6)
- Regeneration output

### Explicit "malformed but recoverable" cases:

| Input | Recovery |
|---|---|
| `{"path":"/tmp/x",}` — trailing comma in JSON | JSON5 parsing or regex strip |
| `<tool_call>{"name":"read","args":{"path":"/tmp/x"}}</tool_call>` — different keys | Bounded alias mapping from envelope |
| `{"tool":"read_file","params":{"path":"/tmp/x"}}` — wrong wrapper/field names | Declared field-mapping only |
| `I should read /tmp/x.` — no structured output | Do NOT synthesize. Hidden regeneration only when `tool_choice=required` AND configured correction pass permits. |

### What is NOT repaired:

- Prose under `tool_choice=auto` is never synthesized as a tool call.
- Semantic repair is never claimed.
- Generic numeric clamping that changes model intent is forbidden.
- Partial execution of a parallel batch when one call is invalid is forbidden by default.

### Regeneration contract:

Only perform hidden regeneration when:

- `tool_choice=required` or a named tool is required
- Model emitted a declared but malformed tool envelope
- A configured single correction pass is permitted (`RepairConfig.max_regenerations >= 1`)

### Gate

```python
def test_tool_transaction_preserves_raw_malformed():
    """A RawToolCallCandidate with malformed JSON survives the transaction
    service as a RawToolCallCandidate with its raw_arguments preserved.
    The service does not 'fix' it into an empty dict before processing."""
    candidate = RawToolCallCandidate(
        id="tc_1", name="read_file",
        raw_arguments='{"path":"/tmp/x",}',  # trailing comma
        source_protocol="ollama_chat",
        source_index=0,
    )
    decision = tool_transaction_service(candidate, tools=[read_file_tool])
    assert decision.outcome.is_accepted
    assert decision.accepted_block is not None
    assert decision.accepted_block.arguments == {"path": "/tmp/x"}
```

---

## Phase 5: Complete Non-Streaming Application Wiring

### Scope

Wire everything end-to-end without streaming:

```
ASGI endpoint
  → adapter.decode_request()          → CanonicalRequest
  → route resolution                  → ModelRoute
  → InvocationPlan construction       → InvocationPlan
  → upstream codec render             → dict
  → HTTP transport                    → upstream response
  → upstream codec decode             → DecodedModelResponse
  → tool transaction service          → list[ToolCallDecision]
  → canonical response assembly       → CanonicalResponse
  → adapter.encode_response()         → client-native dict
  → HTTP response
```

Wire these services:

- **Per-route authentication**: `UpstreamAuthConfig` applied via route headers.
- **Configuration policy**: `InteropServerConfig` drives route selection.
- **Structured rejection errors**: When all tool calls are rejected, return a
  `stop_reason="error"` with structured error details. Never return 200 with
  empty content for a rejected tool response.
- **Regeneration**: After rejection, optionally re-request with corrective prompt.
- **Request/session identity**: `RequestContext` populated correctly from headers.
- **Telemetry**: Repair events carry session correlation and route provenance.
- **Route/profile/model provenance**: Every `CanonicalResponse` carries the
  resolved route and model.

### What is explicitly NOT wired in Phase 5:

- Streaming (Phase 6).
- Loop detection (designed in Phase 5 but wired in Phase 6 — the current
  repeated-call-only concept needs redesign to avoid false positives).
- Repair-note round-trip (designed but deferred: requires session state which
  is only meaningful with streaming context).

### Gate

```python
@pytest.mark.asyncio
async def test_non_streaming_all_three_protocols():
    """All three client protocols produce valid HTTP responses through the
    complete ASGI-to-upstream-to-ASGI path for non-streaming requests."""
    for protocol, body_factory in [
        ("anthropic_messages", anthropic_messages_body),
        ("openai_chat", openai_chat_body),
        ("openai_responses", openai_responses_body),
    ]:
        async with AsyncClient(app=app, base_url="http://test") as client:
            resp = await client.post(protocol_endpoint(protocol), json=body_factory())
            assert resp.status_code == 200
            data = resp.json()
            assert "content" in data or "output" in data or "choices" in data
```

---

## Phase 6: Streaming Lifecycle and Validated Delayed Tool Emission

### Scope

Streaming must not reuse an accumulator that concatenates strings and emits
empty-argument tool blocks. Requirements:

1. **Accumulator keyed by provider choice/index and tool index** — fragments
   for different tools and different parallel branches are kept separate.

2. **IDs, names, and arguments accumulated independently** — each fragment
   appends to the correct slot.

3. **Completed calls are drained once** — after emission, the accumulator
   must not re-emit the same call. No duplicates in the stream.

4. **Provider completion markers determine completion** — the upstream's
   `finish_reason` or `done` flag signals that a call is complete, not
   speculative heuristics like "name and arguments both have data."

5. **Incomplete calls are rejected or regenerated** — never merely logged
   and forwarded.

6. **Complete candidate goes through the same tool transaction service as
   non-streaming** — no separate repair path for streaming.

7. **No speculative argument fragments sent before validation** — the client
   must not receive partial tool arguments that might be discarded.

8. **Client stream encoders consume canonical block fields directly** — no
   adapter-specific streaming logic.

9. **Exactly one terminal event** — `message_stop` or equivalent appears
   exactly once.

10. **Streaming and non-streaming produce equivalent final results** —
    given the same input and upstream response, both paths produce the same
    `CanonicalToolCallBlock` objects.

For prompted textual streams: buffer text until the tool envelope boundary
is determined. Safe incremental envelope parsing can be optimized later.

### Gate

```python
@pytest.mark.asyncio
async def test_streaming_equivalence():
    """For the same upstream response, the streaming and non-streaming paths
    produce identical CanonicalToolCallBlock objects."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Non-streaming
        resp_non = await client.post("/v1/chat/completions", json=body_non_streaming)
        non_data = resp_non.json()
        non_calls = extract_tool_calls(non_data)

        # Streaming
        resp_stream = await client.post("/v1/chat/completions", json=body_streaming)
        stream_data = collect_sse(resp_stream)
        stream_calls = extract_streaming_tool_calls(stream_data)

        assert deep_equal_tool_calls(non_calls, stream_calls)
```

---

## Phase 7: Full ASGI Protocol Matrix, Conformance Harness, Coverage Enforcement

### Scope

Cross-product validation: every combination of (client protocol × upstream protocol
× tool mode × streaming flag) must be tested at the ASGI level.

Complete coverage matrix:

| Client protocol | Upstream protocol | Tool mode | Streaming | Status |
|---|---|---|---|---|
| anthropic_messages | ollama_chat | NATIVE | No | Test |
| anthropic_messages | ollama_chat | PROMPTED | No | Test |
| anthropic_messages | openai_chat | NATIVE | No | Test |
| anthropic_messages | openai_chat | NATIVE | Yes | Test |
| anthropic_messages | openai_chat | PROMPTED | No | Test |
| anthropic_messages | openai_chat | PROMPTED | Yes | Test |
| openai_chat | ollama_chat | NATIVE | No | Test |
| openai_chat | ollama_chat | NATIVE | Yes | Test |
| openai_chat | openai_chat | NATIVE | No | Test |
| openai_chat | openai_chat | NATIVE | Yes | Test |
| openai_chat | openai_chat | PROMPTED | No | Test |
| openai_chat | openai_chat | PROMPTED | Yes | Test |
| openai_responses | openai_chat | NATIVE | No | Test |
| openai_responses | openai_chat | NATIVE | Yes | Test |
| openai_responses | openai_chat | PROMPTED | No | Test |
| openai_responses | openai_chat | PROMPTED | Yes | Test |

### Cleanup and hardening:

- Remove dead code paths exposed by coverage gaps.
- Remove `encode_nonstream_response` legacy methods after verifying no callers.
- Remove `_legacy_dict_to_canonical` and `_process_backend_response` after
  verifying Phase 5 wiring makes them unreachable.
- Strengthen type annotations across all internal boundaries.

### Gate

```python
def test_full_protocol_matrix():
    """Every (client, upstream, mode, stream) combination produces
    a valid HTTP response with correct tool calls."""
    for client_proto in ProtocolKind:
        for upstream_codec in (ollama_chat, openai_chat):
            for mode in (ToolMode.NATIVE, ToolMode.PROMPTED):
                for stream in (False, True):
                    result = run_matrix_test(
                        client_protocol=client_proto,
                        upstream_codec=upstream_codec,
                        tool_mode=mode,
                        streaming=stream,
                    )
                    assert result.success, (
                        f"Failed: {client_proto}→{upstream_codec}({mode},stream={stream}): "
                        f"{result.error}"
                    )
```

---

## Phase Dependency Map

```
Phase 1 (Canonical ABI)
    │
    ▼
Phase 2 (Upstream Codecs)
    │
    ▼
Phase 3 (Profiles + InvocationPlan)
    │
    ▼
Phase 4 (Tool Transaction Service)
    │
    ├────────────────┐
    ▼                ▼
Phase 5          Phase 6
(Non-streaming)  (Streaming)
    │                │
    └───────┬────────┘
            ▼
      Phase 7 (Matrix + Coverage)
```

---

## Non-Goals (explicitly excluded from all phases)

1. **No arbitrary prose-to-command execution.** Natural language is never
   directly synthesized as an executable tool call.

2. **No acceptance merely because repair reduced the issue count.** If the
   repair pass reduces issues but does not eliminate them, the call is
   rejected (unless regeneration is configured and successful).

3. **No generic numeric clamping that changes model intent.** Clamping is
   only applied when the schema explicitly declares `minimum`/`maximum` and
   the value is a clear outlier.

4. **No claim of semantic repair in NATIVE or raw passthrough mode.**
   Passthrough means passthrough.

5. **No partial execution of a parallel batch when one call is invalid,**
   unless explicitly configured to do so.

6. **No provider-specific response parsing inside `Gateway`.** The Gateway
   consumes `DecodedModelResponse` and `DecodedModelEvent` — never raw
   upstream dicts.

7. **No additional compatibility bridges between the old and new type
   systems.** Phase 1 removes them. Phase 2 through 7 never rebuild them.

8. **No new YAML or file-based configuration.** Route configuration is
   code-defined `InteropServerConfig` objects.

9. **No loop detection without semantic redesign.** The current
   repeated-call-only concept risks false positives. Loop detection is
   deferred until a design that accounts for session-scoped state,
   repair-induced retries, and legitimate repeated calls.

---

## Appendix A: Current Runtime Blockers (must-fix in Phase 1)

### Bug 1: `events` undefined in `_process_chunk_stream`

**File**: `src/interop/gateway.py`, method `_process_chunk_stream` (line ~502)

The method references `events.append(...)` but `events` is never initialized.
This will raise `UnboundLocalError` on any streaming request that produces
tool calls or text deltas.

**Fix**: Add `events: list[CanonicalEvent] = []` at line 1 of the method body.

### Bug 2: Empty-content response when all tool calls rejected

**File**: `src/interop/gateway.py`, `_process_backend_response` (line ~287)

When every tool call is rejected, the method returns `CanonicalResponse` with
`content=[]`. The client sees a 200 with an empty assistant response — no
error, no indication of rejection.

**Fix**: After the repair loop, if `rejected` is non-empty and `accepted_blocks`
is empty, return `CanonicalResponse(stop_reason="error")` with structured error
details in `extra`.

### Bug 3: Legacy type usage still references `cb.tool_call`

**File**: `src/interop/protocols/openai_responses.py` (line ~246)
**File**: `src/interop/protocols/anthropic_messages.py` (line ~267)

Both `encode_stream_event` methods reference `event.content_block.tool_call` which
is a v1 legacy type attribute. After Phase 1, `event.content_block` is a
`CanonicalToolCallBlock` with direct `.id`, `.name`, `.arguments` fields.

---

## Appendix B: Files to Create, Modify, or Remove

### Phase 1 — Canonical ABI Unification

| Action | File | Change |
|---|---|---|
| Modify | `src/interop/protocols/anthropic_messages.py` | Build canonical content blocks directly; no v1 types |
| Modify | `src/interop/protocols/openai_chat.py` | Same |
| Modify | `src/interop/protocols/openai_responses.py` | Same |
| Modify | `src/interop/upstreams/openai_chat.py` | Remove dict/hasattr duck typing |
| Modify | `src/interop/upstreams/ollama_chat.py` | Same |
| Create | `src/interop/compat.py` | Move v1 types here |
| Modify | `src/interop/abi.py` | Remove v1 backward-compat types (lines 580-653) |
| Modify | `src/interop/types.py` | Re-export from compat, add deprecation warning |
| Modify | `src/interop/gateway.py` | Fix `events` bug; fix empty-content bug |
| Create | `tests/test_no_v1_imports.py` | Phase gate test |

### Phase 2 — Upstream Codecs

| Action | File | Change |
|---|---|---|
| Create | `src/interop/upstreams/codec.py` | `ModelCodec` interface, `DecodedModelResponse`, `DecodedModelEvent` |
| Modify | `src/interop/upstreams/openai_chat.py` | Implement `ModelCodec` |
| Modify | `src/interop/upstreams/ollama_chat.py` | Implement `ModelCodec` |
| Create | `src/interop/upstreams/openai_responses.py` | Implement `ModelCodec` |
| Create | `src/interop/upstreams/anthropic_messages.py` | Implement `ModelCodec` (may be stub initially) |
| Create | `tests/test_upstream_codecs.py` | Codec unit tests + malformed-JSON gate test |

### Phase 3 — InvocationPlan

| Action | File | Change |
|---|---|---|
| Create | `src/interop/plan.py` | `InvocationPlan`, `build_invocation_plan()`, mode resolution |
| Modify | `src/interop/config.py` | Wire `ToolMode` resolution into `ModelRoute` |
| Create | `tests/test_invocation_plan.py` | Plan construction + determinism tests |

### Phase 4 — Tool Transaction Service

| Action | File | Change |
|---|---|---|
| Create | `src/interop/transaction.py` | `ToolCallDecision`, `tool_transaction_service()` |
| Modify | `src/interop/repair/parse.py` | Envelope extraction separates from JSON parsing |
| Modify | `src/interop/repair/rules.py` | Bounded alias mappings, envelope recovery rules |
| Create | `tests/test_transaction.py` | Malformed JSON, envelope recovery, regeneration tests |

### Phase 5 — Non-Streaming Wiring

| Action | File | Change |
|---|---|---|
| Modify | `src/interop/gateway.py` | Complete rewrite of request handling to use Phase 1-4 |
| Modify | `src/interop/server/app.py` | Wire auth, config, session identity |
| Create | `tests/test_non_streaming_integration.py` | Phase gate: all three protocols, ASGI-level |

### Phase 6 — Streaming

| Action | File | Change |
|---|---|---|
| Rewrite | `src/interop/streaming/coordinator.py` | Full accumulator with independent ID/name/args slots |
| Modify | `src/interop/gateway.py` | New streaming path that buffers + validates before emit |
| Create | `tests/test_streaming_integration.py` | Equivalence test, delayed emission, completion markers |

### Phase 7 — Matrix + Cleanup

| Action | File | Change |
|---|---|---|
| Create | `tests/test_protocol_matrix.py` | Full cross-product test |
| Remove | — | `encode_nonstream_response`, `_legacy_dict_to_canonical`, `_process_backend_response` |
| Modify | `src/interop/gateway.py` | Remove dead code paths |
| Modify | Various | Type annotation hardening |