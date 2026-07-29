# Contributing

## Setup

```bash
uv sync --extra dev
```

## Before opening a change

```bash
uv run ruff check src tests
uv run mypy src/agent_interop
uv run pytest
```

All three must be clean. `./scripts/release.sh --check` runs the full
release gate (the same one CI runs) against a possibly-dirty tree — see
`RELEASE.md` for exactly what it checks and in what order.

## Code style

- No comments explaining WHAT code does — names should already make that
  clear. A comment earns its place only by explaining a non-obvious WHY: a
  hidden constraint, a workaround, an invariant a future reader could
  easily violate by "simplifying" the code.
- Don't add abstractions, config flags, or error handling for scenarios
  that can't happen. Trust internal invariants; validate only at real
  boundaries (user input, external responses).
- Prefer editing existing modules over adding new top-level ones unless
  the new concern is genuinely separable.

## Tests

- Every bug fix and every new capability gets a test that would have
  caught the bug / exercises the capability directly — not just a smoke
  test that the code runs without raising.
- Tests that exercise the Gateway/CLI/server boundary should go through
  the real objects (`Gateway`, `create_app`, the real Typer `app` via
  `CliRunner`) wherever practical, not a hand-rolled stand-in — several
  real bugs in this codebase were only ever visible at that level.

## Profiles

Model profiles (`src/agent_interop/data/profiles/*.yaml`) are executable
contracts, not descriptive metadata — see `model/profiles_v2.py`'s module
docstring. A new profile field must actually be read somewhere in the
invocation/extraction pipeline; an unknown field is a load-time error
(`validate_profile_schema`), not something silently ignored.

## Release-track claims

Never describe a client integration as more verified than the evidence
supports — see `README.md`'s "Verification tiers" and `RELEASE.md`'s
"Alpha vs. supported release track". `scripts/check_support_claims.sh`
(part of the release gate) enforces this mechanically; it is not
optional.

## Commit/PR expectations

- Keep changes focused; a bug fix doesn't need an accompanying refactor.
- Update `AGENTS.md` when you change something it documents (directory
  layout, CLI behavior, conformance-level mapping, etc.) — it drifting
  out of sync with the code is itself a bug.
