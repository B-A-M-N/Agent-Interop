#!/usr/bin/env bash
# ─── Support-claims gate ────────────────────────────────────────────────────
# Ensures the CLI/README never claim more client-integration confidence than
# recorded evidence actually supports. See RELEASE.md's "Alpha vs. supported
# release track" section.
#
# Two checks:
#   1. The unqualified phrase "fully supported" must never appear in
#      cli.py/README.md at all — this project deliberately replaced that
#      phrase with tiered wording (see README's "Verification tiers");
#      re-introducing it would be exactly the P0-5 overclaim this gate
#      exists to catch.
#   2. Any client README's status table marks "release-tested" (the
#      strongest tier — an automated, opt-in acceptance test actually
#      drove the real client binary) must have a matching
#      acceptance/results/<client-slug>-*.json evidence file, produced by
#      a REAL run of tests/acceptance/ (see tests/acceptance/README.md).
#
# Exit code is non-zero if either check fails.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAILED=0

echo "[1/2] Checking for the unqualified phrase 'fully supported'..."
if grep -riIn "fully supported" README.md src/agent_interop/cli.py 2>/dev/null; then
    echo "FAIL: found the unqualified phrase 'fully supported' above — replace it with a tiered claim (see README's 'Verification tiers') backed by real evidence." >&2
    FAILED=1
else
    echo "  none found."
fi

echo "[2/2] Checking release-tested claims have matching acceptance evidence..."
mkdir -p acceptance/results
while IFS='|' read -r _ client _backend verified _; do
    client="$(echo "$client" | xargs)"
    verified="$(echo "$verified" | xargs)"
    [ -z "$client" ] && continue
    [ "$client" = "Client" ] && continue  # header row
    [ "$client" = "---" ] && continue
    case "$client" in --*) continue ;; esac
    case "$verified" in
        *"release-tested"*)
            slug="$(echo "$client" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '-' | sed 's/-\+/-/g; s/^-//; s/-$//')"
            matches=$(find acceptance/results -maxdepth 1 -name "${slug}-*.json" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$matches" -eq 0 ]; then
                echo "FAIL: README claims '$client' is release-tested, but no acceptance/results/${slug}-*.json evidence file exists." >&2
                FAILED=1
            else
                echo "  '$client': OK ($matches record(s) found)"
            fi
            ;;
    esac
done < <(grep '^|' README.md || true)

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
echo "check_support_claims: no unsupported claims found."
