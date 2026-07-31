"""Build provenance carried into planning/evidence/diagnostics."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from importlib.resources import files
from pathlib import Path


@dataclass(frozen=True)
class BuildInfo:
    package_version: str = ""
    git_commit: str = ""
    git_dirty: bool | None = None
    module_path: str = ""
    executable_path: str = ""
    python_executable: str = ""
    profile_bundle_digest: str = ""
    agent_manifest_digest: str = ""
    planner_revision: str = "1"
    battery_version: str = ""


def _bundle_digest(folder: str) -> str:
    root = files("agent_interop.data").joinpath(folder)
    digest = hashlib.sha256()
    try:
        resources = sorted((item for item in root.iterdir() if item.is_file()), key=lambda item: item.name)
        for resource in resources:
            digest.update(resource.name.encode())
            digest.update(resource.read_bytes())
    except (FileNotFoundError, ModuleNotFoundError):
        return ""
    return digest.hexdigest()[:16]


def _git_info(module_path: Path) -> tuple[str, bool | None]:
    try:
        root = module_path.parents[2]
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "diff", "--quiet"], cwd=root, check=False).returncode)
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return "", None


def get_build_info() -> BuildInfo:
    import agent_interop
    module_path = Path(agent_interop.__file__ or "").resolve()
    commit, dirty = _git_info(module_path)
    try:
        version = metadata.version("agent-interop")
    except metadata.PackageNotFoundError:
        version = getattr(agent_interop, "__version__", "")
    try:
        from agent_interop.testing.levels import BATTERY_VERSION
    except ImportError:
        BATTERY_VERSION = ""
    return BuildInfo(
        package_version=version,
        git_commit=commit,
        git_dirty=dirty,
        module_path=str(module_path),
        executable_path=os.path.realpath(sys.argv[0]),
        python_executable=sys.executable,
        profile_bundle_digest=_bundle_digest("profiles"),
        agent_manifest_digest=_bundle_digest("agents"),
        battery_version=BATTERY_VERSION,
    )
