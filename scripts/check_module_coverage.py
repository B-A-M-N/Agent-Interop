#!/usr/bin/env python3
"""Per-module coverage floors — a regression-prevention ratchet.

The release gate's aggregate --cov-fail-under (70% as of this writing) can
hide a critically under-tested module behind a healthy overall average —
audited and confirmed for install.py, launcher.py, agents/codex.py,
tool/normalize.py, and cli.py, each well below the aggregate floor.

Floors here are set at (roughly) each module's OWN coverage at the time
this script was introduced, not an aspirational target — writing enough
new tests to responsibly raise a module from 44% to, say, 90% is real,
separate work this script does not substitute for. Its job is narrower
and still real: once a module reaches a level, this prevents it from
silently regressing back down again, module by module, in a way the
aggregate number alone would not catch.

Usage:
    python scripts/check_module_coverage.py path/to/coverage.json
"""

from __future__ import annotations

import json
import sys

# module path suffix -> minimum percent_covered. Raise a floor (never
# lower it without a comment explaining why) as a module gets more tests.
MODULE_FLOORS: dict[str, float] = {
    "src/agent_interop/install.py": 80.0,
    "src/agent_interop/launcher.py": 55.0,
    "src/agent_interop/agents/codex.py": 40.0,
    "src/agent_interop/tool/normalize.py": 40.0,
    "src/agent_interop/cli.py": 45.0,
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: check_module_coverage.py <coverage.json>", file=sys.stderr)
        return 2

    with open(sys.argv[1]) as f:
        data = json.load(f)

    files = data.get("files", {})
    failures: list[str] = []
    for module_path, floor in MODULE_FLOORS.items():
        entry = files.get(module_path)
        if entry is None:
            failures.append(
                f"{module_path}: not found in coverage report — module may have "
                f"been renamed/removed; update MODULE_FLOORS"
            )
            continue
        actual = entry["summary"]["percent_covered"]
        if actual < floor:
            failures.append(
                f"{module_path}: {actual:.1f}% covered, below the {floor:.1f}% floor"
            )

    if failures:
        print("Per-module coverage floor violations:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"All {len(MODULE_FLOORS)} module coverage floors met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
