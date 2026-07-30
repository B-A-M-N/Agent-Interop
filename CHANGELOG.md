# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `agent_interop.testing.levels`: wires the real conformance test battery to
  L0-L4 capability levels, with explicit battery-versioning and
  infra-vs-behavioral failure classification (`interop test --repair/--no-repair`).
- Evidence-store `capability_source` classification (`observed` /
  `manually_approved` / `stale` / `revoked`) and a sibling `"observed"`
  block on `/v1/capabilities`, alongside the existing declared-metadata block.
- Manifest-backed installer identity (`interop install`/`uninstall`):
  canonical realpath + content hash instead of a marker-text substring
  check, plus a transaction lock and `--dry-run`.
- Opt-in real-client acceptance test harness (`tests/acceptance/`) and
  `scripts/check_support_claims.sh`, enforcing the "alpha vs. supported
  release track" distinction (see `RELEASE.md`) as part of the release gate.
- Per-request execution-nonce gating for the ambiguous whole-message-JSON
  fallback dialect under `tool_choice=auto`; builtin-tier profiles can no
  longer enable it at all.
- `validate_config` is now enforced at the actual construction boundary
  (`create_app`/`Gateway.__init__`), not only at CLI-level call sites.
- `SECURITY.md`, `CONTRIBUTING.md` (this file's siblings).
- Registered static compatibility packs (`compatibility_packs/<client>/`)
  now activate their field-alias mappings as soon as the resolved client
  identity and tool/schema contract match a sufficiently populated
  `CompatibilityKey` — they no longer require a separate
  `compatibility_verified` flag, since a maintainer-authored, statically
  reviewed alias mapping is a different trust category from a
  learned/observed one. Dynamic or user-supplied alias sources still
  require evidence verification; existing collision/ambiguity/
  discriminated-union safeguards in the repair rules are unchanged.
- Compact repair-note feedback: when the repair pipeline actually changes
  a tool call, the assistant turn now carries a short `[Interop]
  Normalized ...` text block naming what was renamed/coerced, so it
  persists into conversation history and can help a model stop repeating
  the same malformed schema within a session. Never emitted for calls
  that were already valid.
- `interop repair stats` (`--route`/`--model`/`--client`/`--since`/
  `--json`): grouped, queryable repair-pipeline effectiveness — accepted
  without repair, accepted after repair, rejected, rejected-with-partial-
  repair, and per-rule counts — backed by a new `repair_events` table
  (schema v7) written from real request handling.

### Changed
- PyPI distribution renamed to `agent-interop`; the CLI command remains
  `interop`. The importable Python package was renamed from `interop` to
  `agent_interop` — a decoy distribution also providing a top-level
  `interop` module was found to physically overwrite this package's own
  files on disk once installed into the same environment (confirmed via
  the release gate), so an in-package import-time collision guard could
  never reliably run; renaming the import name removes the shared
  top-level name entirely instead of trying to detect a collision on it.
- Project/user-tier model profiles that fail to load now raise instead of
  silently logging and skipping (builtin-tier profiles keep the
  warn-and-skip behavior for forward compatibility).
- `interop status` documented as installer/shim status only; compatibility
  evidence is queried via `interop evidence list --route/--model`.

### Fixed
- `interop doctor` now always closes its Gateway, including when startup
  or the per-route probe/diagnostics block raises.
- `ollama_list_models`/`ollama_has_model` no longer collapse "unreachable",
  "auth failed", "invalid response", and "genuinely no models" into the
  same empty list (see `ollama_list_models_detailed`).
- `validate_config` gained missing checks: `log_level` must be a
  recognized level name, `max_keepalive_connections` must not exceed
  `max_connections`.
- Fixed several installer transactional-safety gaps (interrupted-install
  visibility, rollback on shim-write failure, atomic+fsynced manifest writes).
- `interop run claude` (and `ClaudeCodeIntegration.build_launch()`) now
  passes `--model claude-interop-<route>` as an explicit CLI flag instead
  of relying solely on the `CLAUDE_MODEL` env var — a real live-launch
  test against Claude Code 2.1.220 found the installed CLI does not read
  that env var and falls back to the operator's own persisted default
  model, producing an "Unknown model" 400 from the gateway. The env var
  is still set for forward compatibility. No duplicate `--model` flag is
  emitted when the caller's own `extra_args` already supplies one.
- Four real correctness bugs found via a live round trip (real `claude`
  binary → real Interop gateway → real Ollama cloud model → tool call →
  tool execution → final answer), none of which the existing test suite
  caught because they only manifest on the Anthropic Messages *streaming*
  path, which no prior test exercised end-to-end against a real client:
  - `AnthropicStreamEncoder` dropped a response's first `text_delta`
    entirely — `content_block_start` hardcoded `"text": ""` and the
    actual text was never emitted as a following `content_block_delta`.
    Silent for multi-chunk native streaming (only the first token was
    lost), but total for any response whose whole text arrives in a
    single event — exactly what the `BUFFER_TEXTUAL_RESPONSE` path does
    for prompted-mode models.
  - Streamed tool calls could report `stop_reason: "end_turn"` instead of
    `"tool_use"` when the backend's own terminal frame disagreed with
    what was actually emitted (observed with Ollama/gpt-oss, which
    streams the tool_calls fragment in a non-terminal chunk and closes
    with `done_reason: "stop"`). A message containing a tool_use block
    reporting anything but `tool_use` is a protocol invariant violation;
    the streaming path now mirrors the non-streaming path's existing
    "accepted tool blocks force `stop_reason=tool_use`" guard.
  - `compatibility_packs/claude_code`'s `ALIASES` table used generic
    snake_case tool names (`read_file`, `edit_file`, `search_code`, ...)
    that never match Claude Code's actual PascalCase tool names (`Read`,
    `Edit`, `Grep`, ...), and had the canonical/alias direction backwards
    for the file-path field (treated `path` as canonical when Claude
    Code's real schema requires `file_path`). The pack had silently never
    activated for any real Claude Code session since it shipped.
  - `AnthropicStreamEncoder`'s `tool_use` event put the fully-populated
    arguments directly into `content_block_start`'s `input` field and
    never emitted an `input_json_delta` or a `content_block_stop` for the
    block — Claude Code's own SDK rebuilds `input` purely from
    accumulated `input_json_delta` chunks and finalizes it on
    `content_block_stop`; with neither ever sent, every real tool call
    failed client-side with "The model's tool call could not be parsed",
    even when the call itself (name, arguments) was perfect.
  - The Anthropic Messages decoder kept `role: "user"` for a message
    containing a pure `tool_result` block, instead of normalizing it to
    Interop's internal `role="tool"` convention (already used by the
    OpenAI Chat decoder, `upstreams/anthropic.py`'s outbound encoder, and
    required by `history/reconcile.py`'s safety check). Since Anthropic's
    real wire format has no dedicated "tool" role — a tool result is
    just a content block inside a `role: "user"` message — this made
    Interop's own history-safety check reject every real multi-turn tool
    call as "unsafe history" on the very next turn. A `user` message that
    mixes a `tool_result` with the user's own new text still correctly
    keeps `role="user"`.

### Verified
- Claude Code (v2.1.220): a real, opt-in acceptance run
  (`tests/acceptance/test_real_client_claude.py`) launched the actual
  `claude` binary with the exact `LaunchSpec` `interop run claude`
  builds and completed a full tool-call round trip through a live
  Interop gateway — see `acceptance/results/claude-code-2.1.220.json`
  and `RELEASE.md`'s "Alpha vs. supported release track". A second,
  independent live run — the real binary against a real Ollama
  cloud-hosted model (`gpt-oss:20b-cloud`), no fake transport — completed
  the full round trip end to end (read a file, quote its contents back
  correctly) after the four fixes above.

## [0.1.0] — 2026-07-21

Initial public-release candidate: protocol translation (Anthropic
Messages, OpenAI Chat, OpenAI Responses), tool-call extraction/repair for
non-native models (Hermes, Qwen, Mistral, Llama, DeepSeek dialects),
per-route capability negotiation, evidence store, and the `ollama` shim
installer.
