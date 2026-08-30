#!/usr/bin/env python3
"""Support-claims gate (review #38).

Ensures the README never claims more client-integration confidence than the
recorded acceptance evidence actually supports. See RELEASE.md's "Alpha vs.
supported release track".

Checks:
  1. The unqualified phrase "fully supported" must never appear in
     README.md / cli.py — tiered wording only (P0-5).
  2. Any README status-table row marked "release-tested" must have a
     matching acceptance/results/<client-slug>-*.json evidence file where
     EVERY record for that client:
       - has passed=true,
       - was produced against a REAL local-model backend
         (real_backend=true) — a scripted fake transport does not count,
       - proved the per-request execution-nonce gate
         (verification.nonce_gated_recovery=true),
       - is bound to the current build (battery_version + git_commit).
     ALL matching records are validated (not just the first found) — one
     valid file must not launder a stale or failed sibling.

Exit code is non-zero if any check fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CLI = REPO_ROOT / "src" / "agent_interop" / "cli.py"
RESULTS_DIR = REPO_ROOT / "acceptance" / "results"

RELEASE_TESTED_LABEL = "reproducibly release-tested"


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)


def slugify(client: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", client.lower()).strip("-")
    return re.sub(r"-+", "-", slug)


def check_overclaim_phrase() -> bool:
    print("[1/2] Checking for the unqualified phrase 'fully supported'...")
    bad = False
    for path in (README, CLI):
        if not path.exists():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if "fully supported" in line.lower():
                print(f"  {path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
                bad = True
    if bad:
        fail(
            "found the unqualified phrase 'fully supported' — replace it with a "
            "tiered claim (see README's 'Verification tiers') backed by real evidence."
        )
        return False
    print("  none found.")
    return True


def _build_binding_errors(evidence_path: Path) -> list[str]:
    """Compare the evidence's build provenance against the current build."""
    sys.path.insert(0, str(REPO_ROOT / "src"))
    try:
        from agent_interop.build_info import get_build_info
    except Exception as exc:  # pragma: no cover - environment-dependent
        return [f"build provenance could not be verified ({exc}); treating as stale"]
    record = json.loads(evidence_path.read_text())
    build = record.get("build") or {}
    current = get_build_info()
    errors: list[str] = []
    battery = (build.get("battery_version") or "").strip()
    current_battery = (getattr(current, "battery_version", "") or "").strip()
    if battery != current_battery:
        errors.append(
            f"recorded against a different conformance battery "
            f"(battery_version={battery or '?'!r}, current={current_battery or '?'!r})"
        )
    commit = (build.get("git_commit") or "").strip()
    current_commit = (getattr(current, "git_commit", "") or "").strip()
    if commit and current_commit and commit != current_commit:
        errors.append(
            f"recorded against a different git commit ({commit[:12]} vs current "
            f"{current_commit[:12]})"
        )
    return errors


def check_release_tested_claims() -> bool:
    print("[2/2] Checking release-tested claims have real, passed, nonce-gated, "
          "build-bound evidence...")
    if not README.exists():
        print("  README.md missing; nothing to check.")
        return True

    table_rows = [
        line for line in README.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.startswith("|")
    ]
    claims: list[str] = []
    for row in table_rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 3 or cells[0] in {"Client", ""} or set(cells[0]) <= {"-"}:
            continue
        if "release-tested" in cells[-1].lower():
            claims.append(cells[0])

    if not claims:
        print("  no release-tested claims in README — nothing to verify.")
        return True

    ok = True
    for client in claims:
        slug = slugify(client)
        evidence_files = sorted(RESULTS_DIR.glob(f"{slug}-*.json")) if RESULTS_DIR.exists() else []
        if not evidence_files:
            fail(
                f"README claims '{client}' is release-tested, but no "
                f"acceptance/results/{slug}-*.json evidence file exists."
            )
            ok = False
            continue

        # Validate EVERY record — the newest file does not launder stale or
        # failed siblings (review #38: never just head -n1).
        for evidence_path in evidence_files:
            try:
                record = json.loads(evidence_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"'{client}': evidence {evidence_path.name} is unreadable ({exc}).")
                ok = False
                continue

            problems: list[str] = []
            if record.get("passed") is not True:
                problems.append("passed is not true")
            if record.get("real_backend") is not True:
                problems.append(
                    "was not produced against a real local-model backend "
                    "(real_backend=false — a scripted fake transport does not count)"
                )
            verification = record.get("verification") or {}
            if verification.get("nonce_gated_recovery") is not True:
                problems.append(
                    "did not prove the per-request execution-nonce gate "
                    "(verification.nonce_gated_recovery=false)"
                )
            problems.extend(
                f"{evidence_path.name} {err}" for err in _build_binding_errors(evidence_path)
            )

            if problems:
                for problem in problems:
                    fail(f"'{client}' is claimed release-tested, but {problem}.")
                ok = False
            else:
                print(f"  '{client}': OK ({evidence_path.name}: passed, real backend, "
                      f"nonce-gated, bound to current build).")
    return ok


def main() -> int:
    check1 = check_overclaim_phrase()
    check2 = check_release_tested_claims()
    if not (check1 and check2):
        return 1
    print("check_support_claims: no unsupported claims found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
