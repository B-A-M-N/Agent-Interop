# Release Process

## Prerequisites

- Working tree must be clean (no tracked, staged, *or* untracked files)
- `uv` must be installed (https://docs.astral.sh/uv/)
- Dependencies are pinned in `uv.lock` — regenerate with `uv lock` after changing `pyproject.toml`

## Steps

### 1. Run the release gate

```bash
./scripts/release.sh
```

This is the authoritative gate — CI runs the identical script (see "CI"
below), and a release should never be cut from a state this script hasn't
passed against. It runs, in order:

1. Repository is clean (tracked, staged, and untracked files)
2. `git diff --check HEAD` (no trailing whitespace / conflict markers)
3. Version in `pyproject.toml` matches semver
4. `uv lock --check` (lockfile in sync with `pyproject.toml`)
5. `uv sync --frozen --extra dev` + `uv pip check` (dependency compatibility)
6. `ruff check src/ tests/`
7. `mypy src/agent_interop`
8. `pytest` with coverage (`--cov-fail-under`, threshold via
   `INTEROP_RELEASE_COVERAGE_MIN`, default 70%) plus per-module coverage
   floors (`scripts/check_module_coverage.py`)
9. Build the wheel into a clean temporary directory (never `dist/` in the
   repo — a verification run leaves no artifacts behind)
10. Install that exact wheel into a fresh temporary venv
11. Import `agent_interop` from the installed wheel
12. Run `interop version` from the installed wheel
13. Verify every packaged model profile YAML loads
14. `interop init` → load → `validate_config` → `get_route_for_model("")`
    round-trip, then `interop config validate` — proves a config Interop
    generates is a config Interop can actually load, and that a fresh
    config always resolves a default route
15. `scripts/check_support_claims.sh` — fails the gate if any "fully
    supported"-style claim in `cli.py`/README.md lacks matching acceptance
    evidence (see "Alpha vs. supported release track" below)

Exit code is zero only if every step passes.

For local iteration, `./scripts/release.sh --check` runs the identical gate
but allows a dirty working tree — **not** a valid state for an actual
release; every other step still runs in full (no shortcuts, no skipped
build).

## Alpha vs. supported release track

Passing `scripts/release.sh` certifies an **alpha/source release only**. It
does **not** by itself license any "fully supported" or "release-tested"
claim about a client integration (Claude Code, Codex, Crush, or any of the
generic-integration clients) — the gate prints a banner saying so at the
end of every passing run.

There are exactly two release tracks:

- **Alpha/source release** — what this repo can claim today. Every client
  integration is labeled per the tier it has actually earned (see
  `README.md`'s "Client integration status" table and "Verification
  tiers"): unit-tested launch spec, gateway-tested protocol, or nothing
  higher. This is a legitimate, honestly-scoped release track — it is not
  a lesser or provisional state, it is the state the evidence supports.
- **Supported / release-tested claim for a specific client** — requires a
  REAL run of `tests/acceptance/test_real_client_<client>.py` (opt-in, see
  `tests/acceptance/README.md`) to have produced an
  `acceptance/results/<client-slug>-<version>.json` record. The record must
  (a) set `real_backend: true` — i.e. the run exercised a real local model
  backend (Ollama/vLLM/llama.cpp), not `ScriptedFakeTransport`; (b) prove
  the per-request execution-nonce gate (`verification.nonce_gated_recovery`);
  and (c) be bound to the current build — its `build.battery_version` and
  `build.git_commit` must match `agent_interop.build_info.get_build_info()`,
  checked mechanically by `scripts/check_support_claims.sh` (step 15 above).
  An evidence file recorded against a different conformance battery or git
  commit is stale and fails the gate until re-run. Until a client meets all
  three, the README/client-status table must NOT claim the release-tested
  tier for it — the gate fails the moment it does. The CLI/README can never
  claim more than recorded evidence supports, enforced mechanically rather
  than by convention.

As of this writing, **no client meets the release-tested tier.** Claude
Code has recorded acceptance runs (`acceptance/results/claude-code-2.1.220.json`,
`claude-code-2.1.251.json`) produced by actual runs of
`tests/acceptance/test_real_client_claude.py` against the real `claude`
binary — they launched with the exact `LaunchSpec` `interop run claude`
builds and completed a round trip through a live Interop gateway — but
those runs drove a **scripted upstream** (`ScriptedFakeTransport`), so
they sit at the "real client binary launched, scripted backend" tier, NOT
release-tested: `real_backend` is false and the nonce gate did not run in
those recordings. Reaching release-tested requires the
`test_real_claude_real_ollama_small_model.py` gate (real ~7B model,
`num_ctx <= 16,384`) to pass and record with `real_backend: true` and
`verification.nonce_gated_recovery: true`. Codex and the
generic-integration clients (Cline, OpenCode, Aider, Continue, Qwen Code)
have no recorded run yet and stay on the alpha/unverified track until
someone runs their harnesses for real.

### 2. Version bump

Update the `version` field in `pyproject.toml` following [SemVer](https://semver.org/):

- Patch (0.0.x): Bug fixes, minor corrections
- Minor (0.x.0): New features, backward-compatible additions
- Major (x.0.0): Breaking changes

### 3. Tag and release

```bash
git tag -a v$(python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])") -m "release v<VERSION>"
git push origin --tags
# Build the distribution
uv build
# Publish (requires PyPI credentials)
# uv publish
```

A tag should only be pushed once CI (see below) has passed on the commit
being tagged.

## Reproducibility

- Use `uv sync --frozen` to install exactly what the lockfile specifies
- Do not commit `pyproject.toml` dependency changes without regenerating `uv.lock`
- The `uv.lock` file must be committed to version control

## CI

`.github/workflows/ci.yml` runs the same `./scripts/release.sh --check` gate
described above on every push to `main` and every pull request, across
Python 3.11 and 3.12. The Python 3.11 job additionally builds and uploads
the wheel as a workflow artifact. Configure branch protection on `main` to
require this workflow before merging/tagging if that isn't already set up.

## Version Compatibility

- Python >= 3.11
- Dependencies are pinned by minimum version in `pyproject.toml` and exact in `uv.lock`
