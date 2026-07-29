#!/usr/bin/env bash
# ─── Interop Release Gate ───────────────────────────────────────────────────
# Runs the full validation pipeline for a release. This gate is authoritative:
# if it passes, the built wheel is what gets released — every step after the
# build operates on the ACTUAL wheel artifact, installed into a throwaway
# venv, never on the source checkout.
#
# Usage:
#   ./scripts/release.sh              # Full gate (default)
#   ./scripts/release.sh --check      # Same gate, but allows a dirty
#                                      # working tree (development use only —
#                                      # NOT a valid state for an actual release)
#
# Exit code is non-zero if any step fails.
#
# IMPORTANT: passing this gate certifies an ALPHA/SOURCE release only. It does
# NOT by itself license any "fully supported" client claim in the CLI/README —
# see the final step and RELEASE.md for the alpha vs. supported release-track
# distinction.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✘${NC} $1"; exit 1; }

MODE="${1:-default}"
case "$MODE" in
    default|--check) ;;
    *)
        fail "Unknown mode '$MODE'. Supported: (none), --check"
        ;;
esac

# --check allows a dirty tree for local iteration. It is NOT a lighter gate —
# every other step (lint, types, tests, wheel build/install/import, config
# round-trip, CLI smoke) still runs in full. The old --check additionally
# skipped the build entirely while claiming "not building" in its own
# comment yet still built later — that self-contradiction is why this gate
# was rewritten.
SKIP_DIRT_CHECK=false
if [ "$MODE" = "--check" ]; then
    SKIP_DIRT_CHECK=true
fi

COVERAGE_MIN=${INTEROP_RELEASE_COVERAGE_MIN:-70}

echo "=== Interop Release Gate ==="
echo ""

# ── 1. Repository is clean — tracked, staged, AND untracked files ──────────
echo "[1/15] Checking git status..."
if [ "$SKIP_DIRT_CHECK" = false ]; then
    STATUS_OUTPUT="$(git status --porcelain=v1 --untracked-files=all)"
    if [ -n "$STATUS_OUTPUT" ]; then
        echo "$STATUS_OUTPUT"
        fail "Repository is not clean (tracked, staged, or untracked files present). Commit, stash, or .gitignore them before releasing."
    fi
    pass "repository is clean"
else
    warn "dirty-tree check skipped (--check mode) — not valid for an actual release"
fi

# ── 2. git diff --check — no trailing whitespace / conflict markers ────────
echo "[2/15] Checking whitespace and conflict markers..."
if ! git diff --check HEAD -- 2>&1; then
    fail "git diff --check found trailing whitespace or unresolved conflict markers"
fi
pass "git diff --check clean"

# ── 3. Version in pyproject.toml ────────────────────────────────────────────
echo "[3/15] Checking version..."
VERSION=$(python3 -c "import tomllib; f=open('pyproject.toml','rb'); print(tomllib.load(f)['project']['version'])")
echo "  Version: $VERSION"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    fail "Version '$VERSION' does not match semver"
fi
pass "version $VERSION"

# ── 4. Lockfile is in sync with pyproject.toml ──────────────────────────────
echo "[4/15] Checking uv.lock is in sync..."
uv lock --check 2>&1 || fail "uv.lock is out of sync with pyproject.toml — run 'uv lock' and commit the result"
pass "uv.lock in sync"

# ── 5. Dependency compatibility ─────────────────────────────────────────────
echo "[5/15] Checking installed dependency compatibility..."
uv sync --frozen --extra dev --quiet 2>&1 || fail "uv sync --frozen failed"
uv pip check 2>&1 || fail "Dependency compatibility check failed"
pass "dependencies compatible"

# ── 6. Lint ──────────────────────────────────────────────────────────────────
echo "[6/15] Running linter..."
uv run ruff check src/ tests/ 2>&1 || fail "Linting failed"
pass "lint clean"

# ── 7. Type check ────────────────────────────────────────────────────────────
echo "[7/15] Running mypy..."
uv run mypy src/agent_interop 2>&1 || fail "Type check failed"
pass "types clean"

# ── 8. Tests with coverage (aggregate + per-module floors) ─────────────────
echo "[8/15] Running tests with coverage..."
COVERAGE_JSON="$(mktemp -d)/coverage.json"
uv run pytest -x --tb=short -q \
    --cov=src/agent_interop --cov-report=term-missing --cov-report="json:$COVERAGE_JSON" \
    --cov-fail-under="$COVERAGE_MIN" \
    2>&1 || fail "Tests failed (or coverage below ${COVERAGE_MIN}%)"
pass "all tests passed, coverage >= ${COVERAGE_MIN}%"

echo "  Checking per-module coverage floors..."
uv run python scripts/check_module_coverage.py "$COVERAGE_JSON" 2>&1 || fail "Per-module coverage floor not met"
pass "per-module coverage floors met"

# ── 9-11. Build into a clean dist dir, install into a fresh venv, import ───
echo "[9/15] Building wheel into a clean directory..."
DIST_DIR="$(mktemp -d)"
VENV_DIR="$(mktemp -d)"
CONFIG_DIR="$(mktemp -d)"
cleanup() { rm -rf "$DIST_DIR" "$VENV_DIR" "$CONFIG_DIR"; }
trap cleanup EXIT

uv build --out-dir "$DIST_DIR" --quiet 2>&1 || fail "Build failed"
WHEEL="$(find "$DIST_DIR" -maxdepth 1 -name 'agent_interop-*.whl' | head -n1)"
[ -n "$WHEEL" ] || fail "No wheel produced in $DIST_DIR"
pass "built $(basename "$WHEEL")"

echo "[10/15] Installing the built wheel into a fresh environment..."
uv venv "$VENV_DIR" --quiet 2>&1 || fail "Failed to create verification venv"
uv pip install --python "$VENV_DIR/bin/python" "$WHEEL" --quiet 2>&1 || fail "Failed to install built wheel"
pass "wheel installed into fresh venv"

echo "[11/15] Importing agent_interop from the installed wheel..."
"$VENV_DIR/bin/python" -c "
import agent_interop
print('  agent_interop', agent_interop.__version__)
assert agent_interop.__version__, 'agent_interop.__version__ is empty'
" 2>&1 || fail "Import from installed wheel failed"
pass "import verified from installed wheel"

# ── 12. interop version (CLI entry point resolves) ──────────────────────────
echo "[12/15] Running 'interop version' from the installed wheel..."
"$VENV_DIR/bin/interop" version 2>&1 || fail "'interop version' failed"
pass "CLI entry point works"

# ── 13. Packaged model profiles are present, strictly valid, and each
#        builds a real InvocationPlan — not just that their YAML parses.
#        Profiles are executable contracts: a field that doesn't affect
#        runtime behavior (an unknown key, a dangling parser/template
#        reference) must fail here, not ship silently inert.
echo "[13/15] Verifying packaged model profile resources..."
"$VENV_DIR/bin/python" -c "
import importlib.resources as resources
import yaml

from agent_interop.abi import CanonicalTool, CanonicalToolChoice
from agent_interop.config import ToolMode
from agent_interop.extraction import get_default_registry
from agent_interop.model.profiles_v2 import ModelProfile, ProfileIndex, validate_profile_schema
from agent_interop.model.registry import ModelProfileRegistry
from agent_interop.repair.invocation import build_invocation_plan

profiles_dir = resources.files('agent_interop.data.profiles')
yaml_files = [p for p in profiles_dir.iterdir() if p.name.endswith('.yaml')]
assert yaml_files, 'No profile YAML files found in the installed wheel'

index = ProfileIndex()
for p in yaml_files:
    data = yaml.safe_load(p.read_text())
    assert isinstance(data, dict), f'{p.name} did not parse to a dict'
    issues = validate_profile_schema(data, source=p.name)
    assert not issues, f'{p.name} failed strict validation: {issues}'
    profile = ModelProfile.from_yaml(data, matched_by=p.name)
    index.add_profile(profile, data)

# A duplicate profile id must not silently vanish one file's worth of
# behavior — add_profile() now raises on a duplicate id (see
# profiles_v2.py), but this assertion is the explicit, self-documenting
# guarantee: every packaged YAML file becomes one loaded profile.
assert len(index) == len(yaml_files), (
    f'loaded {len(index)} profiles from {len(yaml_files)} files — '
    'a profile id collision silently dropped one'
)

registry = ModelProfileRegistry(profiles=index)
tool = CanonicalTool(name='read_file', description='Read a file', input_schema={})
extractors = get_default_registry()
for profile_id in index.list_ids():
    resolved = registry.resolve(explicit_profile_id=profile_id)
    plan = build_invocation_plan(
        tools=[tool], tool_choice=CanonicalToolChoice.auto(),
        route_mode=ToolMode.AUTO, model_profile=resolved,
        repair_policy=None, codec_capabilities=None,
    )
    if plan.parser_id is not None:
        extractors.get(plan.parser_id)
    for strategy in resolved.fallback_strategies:
        extractors.get(strategy.parser_id)
    if plan.effective_tool_mode == ToolMode.PROMPTED:
        assert plan.prompt_contract.strip(), f'{profile_id}: empty PROMPTED contract'

print(f'  {len(yaml_files)} packaged profile(s) verified — strict schema + real invocation plan')
" 2>&1 || fail "Packaged profile verification failed"
pass "packaged profiles are strictly valid and build real invocation plans"

# ── 14. Generated configuration round-trips, then CLI smoke tests ──────────
echo "[14/15] Config round-trip and CLI smoke tests..."
CONFIG_PATH="$CONFIG_DIR/interop.yaml"

"$VENV_DIR/bin/interop" init --path "$CONFIG_PATH" --backend ollama --model qwen3-coder \
    2>&1 || fail "'interop init' failed"
[ -f "$CONFIG_PATH" ] || fail "'interop init' did not create $CONFIG_PATH"

"$VENV_DIR/bin/python" -c "
import yaml
from agent_interop.config import load_config_from_dict, validate_config

with open('$CONFIG_PATH') as f:
    raw = yaml.safe_load(f)

assert raw.get('default_route'), 'generated config is missing default_route'
config = load_config_from_dict(raw)
issues = validate_config(config)
assert not issues, f'generated config failed validation: {issues}'

route = config.get_route_for_model('')
assert route is not None, 'default route did not resolve from generated config'
print(f'  config round-trip OK — default route: {route.id}')
" 2>&1 || fail "Config round-trip verification failed"

"$VENV_DIR/bin/interop" config validate --path "$CONFIG_PATH" 2>&1 || fail "'interop config validate' failed"
pass "config generates, round-trips, and validates"

# ── 15. Support claims are backed by recorded acceptance evidence ──────────
# See RELEASE.md: a "fully supported" claim about a client integration in
# cli.py/README requires a matching acceptance/results/<client>-<version>.json
# record produced by a REAL run of tests/acceptance/. This gate never lets
# the CLI/README claim more than recorded evidence supports.
echo "[15/15] Checking support claims are backed by acceptance evidence..."
./scripts/check_support_claims.sh 2>&1 || fail "Unsupported claim found without matching acceptance evidence"
pass "support claims match recorded acceptance evidence"

# ── Preserve the verified wheel, if a caller wants it ───────────────────────
# $DIST_DIR is a throwaway mktemp dir, cleaned up on exit — by design, so a
# local verification run never leaves build artifacts in the repo. But a
# caller (CI) that wants to actually PUBLISH the artifact this gate just
# verified needs a copy that survives past cleanup. Without this, CI was
# running its own SEPARATE `uv build` after this script passed and
# uploading THAT wheel — a second, independent build never itself
# installed, imported, or smoke-tested by anything above. Setting
# INTEROP_RELEASE_WHEEL_OUT copies the exact verified artifact out instead.
if [ -n "${INTEROP_RELEASE_WHEEL_OUT:-}" ]; then
    mkdir -p "$INTEROP_RELEASE_WHEEL_OUT"
    cp "$WHEEL" "$INTEROP_RELEASE_WHEEL_OUT/"
    pass "verified wheel copied to $INTEROP_RELEASE_WHEEL_OUT/$(basename "$WHEEL")"
fi

echo ""
echo -e "${GREEN}=== Release gate passed for v$VERSION ===${NC}"
echo -e "${YELLOW}This certifies an ALPHA/SOURCE release only.${NC} No client integration is"
echo "\"fully supported\" until a real acceptance run (tests/acceptance/) has recorded"
echo "an evidence file under acceptance/results/ for that exact client + version."
echo "See RELEASE.md for the alpha vs. supported release-track distinction."
