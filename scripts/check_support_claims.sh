#!/usr/bin/env bash
# ─── Support-claims gate ────────────────────────────────────────────────────
# Thin wrapper (review #38): the checks live in check_support_claims.py so
# the evidence validation (passed=true, real_backend, nonce gate, build
# binding, ALL matching records) can be tested and extended in Python
# instead of fragile shell string handling.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/check_support_claims.py" "$@"
