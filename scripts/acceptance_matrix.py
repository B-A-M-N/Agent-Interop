#!/usr/bin/env python3
"""Acceptance matrix — client x backend capability tiers, executed honestly.

Defines the full real-client x real-backend acceptance matrix for Interop
and drives as much of it as the local machine can genuinely execute,
reporting every cell as one of:

    PASSED          the real run completed and wrote its evidence record
    FAILED          the real run ran and failed
    SKIPPED         opt-in guard: required binaries/env absent on THIS machine
    NOT_RUN         declared in the matrix but this driver has no runner yet

An honest SKIPPED is a fine outcome — it is what a dev laptop reports for
a cell needing resources it does not have. What is NOT fine is a cell
reported as anything other than what actually happened, so this driver
never fabricates evidence: a PASSED cell always has a fresh
acceptance/results/<client>-<version>.json record written by the real test
harness, and a missing binary is a SKIP, not a pass.

The matrix (see RELEASE.md, "Alpha vs. supported release track"):

  Tier 1 — real client binary, scripted upstream (fast, no model needed).
           Proves the launch spec and protocol envelope, NOT the backend.
           Reaching release-tested does NOT require more than this + Tier 2.
  Tier 2 — real client binary, real local model (the 16K gate).
           The release-tested tier: real_backend=true + nonce gate.

Usage:
    uv run python scripts/acceptance_matrix.py                # drive what's runnable
    uv run python scripts/acceptance_matrix.py --list         # print the matrix only
    uv run python scripts/acceptance_matrix.py --tier 2       # only the real-model tier
    uv run python scripts/acceptance_matrix.py --json-out m.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "acceptance" / "results"

CLAUDE_BIN_ENV = "INTEROP_ACCEPTANCE_CLAUDE_BIN"
CODEX_BIN_ENV = "INTEROP_ACCEPTANCE_CODEX_BIN"
OLLAMA_URL_ENV = "INTEROP_ACCEPTANCE_OLLAMA_URL"
# Default assumes the local Ollama; override for tunnels/remote backends.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


@dataclass
class MatrixCell:
    """One client x backend acceptance cell."""

    client: str
    tier: int
    runner: str  # pytest node expression, empty = no runner module yet
    requires_bin: tuple[str, ...] = ()       # env vars that must name real binaries
    requires_env: tuple[str, ...] = ()       # other required env
    model: str = ""                          # Tier 2 only
    notes: str = ""

    @property
    def cell_id(self) -> str:
        return f"T{self.tier}:{self.client}" + (f"@{self.model}" if self.model else "")


# ─── The matrix definition ────────────────────────────────────────────────────

# Clients with a proven harness module. The generic-integration clients
# (Cline, OpenCode, Aider, Continue, Qwen Code) are declared in RELEASE.md's
# track table but have no harness yet — they are NOT_RUN cells, listed so the
# matrix cannot quietly forget them.
HARNESSED_CLIENTS = ("claude", "codex")

# Tier 2 runs against a real local model. qwen2.5-coder:7b is the gate's
# pinned model (tests/acceptance/test_real_claude_real_ollama_small_model.py
# pins MODEL_NAME); a matrix expansion to more models means editing that
# module to parametrize MODEL_NAME, not inventing a second gate here.
TIER2_MODEL = "qwen2.5-coder:7b"

MATRIX: tuple[MatrixCell, ...] = (
    # Tier 1 — real client, scripted upstream (test_real_client_*.py)
    MatrixCell(
        client="claude", tier=1,
        runner="tests/acceptance/test_real_client_claude.py",
        requires_bin=(CLAUDE_BIN_ENV,),
        notes="real claude binary vs ScriptedFakeTransport — launch-spec + envelope proof",
    ),
    MatrixCell(
        client="codex", tier=1,
        runner="tests/acceptance/test_real_client_codex.py",
        requires_bin=(CODEX_BIN_ENV,),
        notes="runner module reviewed but NEVER executed for real — expect argv/parse fixes on first run",
    ),
    # Tier 2 — real client, real ~7B local model at num_ctx <= 16,384
    MatrixCell(
        client="claude", tier=2,
        runner="tests/acceptance/test_real_claude_real_ollama_small_model.py",
        requires_bin=(CLAUDE_BIN_ENV,),
        requires_env=(OLLAMA_URL_ENV,),
        model=TIER2_MODEL,
        notes="the release-tested gate: 11 tasks across L0-L4, HARD_NUM_CTX=16384",
    ),
    MatrixCell(
        client="codex", tier=2,
        runner="",
        notes="no runner exists yet — Tier 1 must pass + a gate module written first",
    ),
    # Declared-but-unharnessed clients (RELEASE.md track table)
    MatrixCell(client="cline", tier=1, runner="", notes="generic-integration client, no harness yet"),
    MatrixCell(client="opencode", tier=1, runner="", notes="generic-integration client, no harness yet"),
    MatrixCell(client="aider", tier=1, runner="", notes="generic-integration client, no harness yet"),
    MatrixCell(client="continue", tier=1, runner="", notes="generic-integration client, no harness yet"),
    MatrixCell(client="qwen-code", tier=1, runner="", notes="generic-integration client, no harness yet"),
)


def _binary_available(env_var: str) -> bool:
    value = os.environ.get(env_var, "")
    if not value:
        return False
    return os.path.isfile(value) or shutil.which(value) is not None


def _ollama_reachable(url: str) -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/version", timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _model_pulled(url: str, model: str) -> bool | None:
    """True/False when Ollama answers, None when reachability is unknown."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=5) as resp:
            tags = json.load(resp)
        return any(m.get("name") == model for m in tags.get("models", []))
    except Exception:
        return None


def classify(cell: MatrixCell) -> str:
    """Report what WOULD happen to this cell on the current machine.

    runnable | missing:<reason> | no-runner
    """
    if not cell.runner:
        return "no-runner"
    for env_var in cell.requires_bin:
        if not _binary_available(env_var):
            return f"missing:{env_var}"
    for env_var in cell.requires_env:
        if not os.environ.get(env_var):
            return f"missing:{env_var}"
    if cell.model:
        url = os.environ.get(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)
        pulled = _model_pulled(url, cell.model)
        if pulled is False:
            return f"model-not-pulled:{cell.model}"
        if pulled is None:
            return f"missing:{OLLAMA_URL_ENV} (no reachable Ollama at {url})"
    return "runnable"


def _latest_result_for(client: str) -> Path | None:
    if not RESULTS_DIR.is_dir():
        return None
    candidates = sorted(RESULTS_DIR.glob(f"{client}-*.json"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_cell(cell: MatrixCell, timeout_s: int) -> dict[str, object]:
    """Execute one cell's runner for real and report what happened."""
    verdict = classify(cell)
    if verdict != "runnable":
        return {"cell": cell.cell_id, "status": "SKIPPED", "reason": verdict, "notes": cell.notes}

    cmd = [
        "uv", "run", "pytest", cell.runner, "-v", "--tb=short", "-x",
        "--no-header", "-q",
    ]
    env = dict(os.environ)
    env.setdefault(OLLAMA_URL_ENV, DEFAULT_OLLAMA_URL)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s, env=env)
    elapsed = time.monotonic() - t0

    # A PASSED verdict requires a fresh evidence record from this run —
    # the harness writes one only when the round trip actually completed.
    record = _latest_result_for(cell.client)
    record_name = record.name if record else None
    record_mtime = record.stat().st_mtime if record else 0

    passed = proc.returncode == 0
    return {
        "cell": cell.cell_id,
        "status": "PASSED" if passed else "FAILED",
        "duration_s": round(elapsed, 1),
        "returncode": proc.returncode,
        "evidence_record": record_name,
        "evidence_fresh": record is not None and record_mtime >= t0,
        "notes": cell.notes,
        "stdout_tail": proc.stdout[-2000:] if not passed else "",
        "stderr_tail": proc.stderr[-2000:] if not passed else "",
    }


def print_matrix(statuses: list[dict[str, object]]) -> None:
    print(f"{'CELL':<28} {'STATUS':<10} DETAIL")
    print("-" * 100)
    for entry in statuses:
        detail = entry.get("reason") or entry.get("evidence_record") or ""
        if entry.get("status") == "PASSED" and entry.get("evidence_fresh") is False:
            detail = "WARNING: no fresh evidence record written by this run"
        print(f"{entry['cell']:<28} {entry['status']:<10} {detail}")
    print()
    print("Status legend: PASSED/FAILED = real run executed; SKIPPED = opt-in")
    print("guard (missing binary/env on this machine); no-runner = declared in")
    print("RELEASE.md but no harness module exists yet.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the matrix and exit")
    parser.add_argument("--tier", type=int, choices=(1, 2), default=None, help="drive only one tier")
    parser.add_argument("--timeout", type=int, default=3600, help="per-cell pytest timeout in seconds")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    cells = [c for c in MATRIX if args.tier is None or c.tier == args.tier]

    statuses: list[dict[str, object]] = []
    if args.list:
        statuses = [{"cell": c.cell_id, "status": classify(c), "notes": c.notes} for c in cells]
    else:
        for cell in cells:
            if classify(cell) != "runnable":
                statuses.append({
                    "cell": cell.cell_id, "status": "SKIPPED",
                    "reason": classify(cell), "notes": cell.notes,
                })
                continue
            print(f"=== running {cell.cell_id}: {cell.runner} ===", flush=True)
            try:
                statuses.append(run_cell(cell, args.timeout))
            except subprocess.TimeoutExpired:
                statuses.append({
                    "cell": cell.cell_id, "status": "FAILED",
                    "reason": f"timeout after {args.timeout}s", "notes": cell.notes,
                })

    print_matrix(statuses)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "generated_at": datetime.now(UTC).isoformat(),
            "host_report": "which cells this machine could genuinely run",
            "cells": statuses,
        }, indent=2))

    # Exit 0 unless a REAL run failed. Skips are not failures — that is the
    # whole point of an honest opt-in matrix. A PASSED cell whose evidence
    # record is missing/stale IS a failure: the run claims more than it
    # recorded.
    real_failures = [
        s for s in statuses
        if s["status"] == "FAILED"
        or (s["status"] == "PASSED" and s.get("evidence_fresh") is False)
    ]
    return 1 if real_failures else 0


if __name__ == "__main__":
    sys.exit(main())
