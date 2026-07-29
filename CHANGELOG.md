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

## [0.1.0] — 2026-07-21

Initial public-release candidate: protocol translation (Anthropic
Messages, OpenAI Chat, OpenAI Responses), tool-call extraction/repair for
non-native models (Hermes, Qwen, Mistral, Llama, DeepSeek dialects),
per-route capability negotiation, evidence store, and the `ollama` shim
installer.
