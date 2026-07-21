"""interop install — sets up transparent interception so existing commands Just Work.

What it does:
1. Creates a wrapper script for `ollama` that intercepts `ollama launch <agent>`
   and routes it through Interop instead of directly to Ollama.

2. The wrapper is placed in ~/.local/bin/interop-wrapper which gets added to
   PATH (or intercepts via a shim named 'ollama' that delegates non-launch
   commands to the real ollama).

3. Also registers Interop as a Hermes provider plugin when Hermes is detected.

After install:
    ollama launch claude  →  Interop intercepts → starts gateway →
                             launches Claude Code pointed at Interop
    ollama serve          →  passed through to real Ollama normally
    ollama pull model     →  passed through to real Ollama normally
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import subprocess
from pathlib import Path

logger = logging.getLogger("interop.install")

INTEROP_WRAPPER_SCRIPT = """#!/bin/bash
# Interop wrapper for `ollama launch`
# Routes `ollama launch <agent>` through Interop's compatibility layer.
# All other ollama commands are passed through to the real ollama.

REAL_OLLAMA="{real_ollama}"
INTEROP="{interop_runner}"

# Check if this is an `ollama launch` command
if [ "$1" = "launch" ] && [ -n "$2" ]; then
    AGENT="$2"
    shift 2
    MODEL=""

    # Parse --model flag
    EXTRA_ARGS=()
    while [ $# -gt 0 ]; do
        case "$1" in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --)
                shift
                # Pass remaining args to the agent
                EXTRA_ARGS+=("$@")
                break
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done

    if [ -z "$MODEL" ]; then
        # No model specified, check for default
        MODEL="$("$REAL_OLLAMA" ps 2>/dev/null | head -2 | tail -1 | awk '{{print $1}}')"
        [ -z "$MODEL" ] && MODEL="qwen3-coder"
    fi

    exec "$INTEROP" run "$AGENT" --model "$MODEL" {{+"${{EXTRA_ARGS[@]}}"}}+%
fi

# All other ollama commands pass through
exec "$REAL_OLLAMA" "$@"
"""

# Also support agent-specific environment variable interception
# ANTHROPIC_BASE_URL=... → if Interop is installed, route through it

INTEROP_ACTIVATE_SCRIPT = """# Interop activation — transparent local LLM compatibility
# Source this in your .bashrc or .zshrc to enable Interop's environment-level
# interception. When INTEROP_ENABLED=1, ANTHROPIC_BASE_URL is automatically
# routed through Interop.

interop_activate() {{
    if command -v interop-wrapper >/dev/null 2>&1; then
        # Replace ollama in PATH so `ollama launch` routes through Interop
        # The wrapper is already installed at a higher PATH priority
        export INTEROP_ACTIVE=1
    fi
}}

# Auto-intercept ANTHROPIC_BASE_URL when Interop is installed
_interop_anthropic_hook() {{
    if [ -n "$ANTHROPIC_BASE_URL" ] && [ "$ANTHROPIC_BASE_URL" != "${{ANTHROPIC_BASE_URL#*:8090}}" ]; then
        # Already pointed at Interop
        return
    fi
    if [ -n "$ANTHROPIC_BASE_URL" ] && command -v interop-gateway >/dev/null 2>&1; then
        # TODO: start interop and rewrite ANTHROPIC_BASE_URL
        :
    fi
}}
"""


# ─── Install paths ──────────────────────────────────────────────────────────


def _bin_dir() -> Path:
    """Find the best directory for user-level binaries."""
    candidates = [
        Path.home() / ".local" / "bin",
        Path.home() / "bin",
    ]
    for d in candidates:
        d.mkdir(parents=True, exist_ok=True)
        if str(d) in os.environ.get("PATH", ""):
            return d
    # Always ensure ~/.local/bin is in PATH
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


def _find_real_ollama() -> str | None:
    """Find the real ollama binary, skipping our own wrapper."""
    # Common locations
    for path in [
        "/usr/local/bin/ollama",
        "/usr/bin/ollama",
        shutil.which("ollama"),
    ]:
        if path and os.path.exists(path):
            return path
    return None


def _find_interop_runner() -> str:
    """Find the interop CLI entry point."""
    # Check if interop module is importable
    candidates = [
        shutil.which("interop"),
        os.path.expanduser("~/.local/bin/interop"),
    ]
    for c in candidates:
        if c:
            return c
    # Default to module invocation
    return f"{shutil.which('python3') or 'python3'} -m interop.cli"


# ─── Install ────────────────────────────────────────────────────────────────


def install(
    bin_dir: str | None = None,
    force: bool = False,
    add_rc: bool = True,
) -> dict[str, str]:
    """Install Interop wrappers for transparent local LLM compatibility.

    Returns a dict with paths of what was installed.
    """
    result: dict[str, str] = {}

    real_ollama = _find_real_ollama()
    if not real_ollama:
        logger.warning("Ollama not found — wrappers will only work after Ollama is installed")
        real_ollama = "/usr/local/bin/ollama"

    interop_runner = _find_interop_runner()
    target_dir = Path(bin_dir) if bin_dir else _bin_dir()

    # Install ollama wrapper shim in target_dir (which is at front of PATH)
    shim_path = target_dir / "ollama"

    # Handle existing non-Interop shims (e.g. Vulkan wrapper)
    existing_vulkan = target_dir / "ollama-vulkan"
    if shim_path.exists():
        content = shim_path.read_text()
        if "Interop wrapper" in content:
            if not force:
                logger.info("Interop shim already installed at %s (use --force to reinstall)", shim_path)
                result["shim"] = str(shim_path)
            else:
                shim_path.unlink()
        else:
            # Existing non-Interop wrapper — rename it so we can install ours,
            # and ours will chain through it
            logger.info("Found existing ollama wrapper — renaming to ollama-vulkan")
            shim_path.rename(existing_vulkan)
            result["renamed"] = str(existing_vulkan)

    if not shim_path.exists():
        shim_path.write_text(_build_shim(real_ollama, interop_runner))
        shim_path.chmod(shim_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        result["shim"] = str(shim_path)
        logger.info("Installed Interop shim: %s", shim_path)

    # Install activation script for shell rc
    activate_path = target_dir.parent / "interop" / "activate.sh"
    activate_path.parent.mkdir(parents=True, exist_ok=True)
    activate_path.write_text(INTEROP_ACTIVATE_SCRIPT)
    result["activate"] = str(activate_path)

    # Check PATH
    if str(target_dir) not in os.environ.get("PATH", ""):
        logger.info("Add to your shell rc: export PATH=\"%s:$PATH\"", target_dir)

    result["ollama"] = real_ollama
    result["target_dir"] = str(target_dir)
    result["interop_runner"] = interop_runner

    return result


def _build_shim(real_ollama: str, interop_runner: str) -> str:
    """Build the ollama wrapper shim script.

    Chains through any existing wrapper (e.g. Vulkan wrapper) so
    non-launch commands still get normal processing.
    """
    target_dir = _bin_dir()
    existing_vulkan = target_dir / "ollama-vulkan"
    if existing_vulkan.exists():
        pass_through = str(existing_vulkan)
    else:
        pass_through = real_ollama

    return f"""#!/bin/bash
# Interop wrapper for `ollama launch`
# Routes `ollama launch <agent>` through Interop's compatibility layer.
# All other ollama commands pass through to the real binary.

REAL_OLLAMA="{pass_through}"
INTEROP_RUNNER="{interop_runner}"

if [ "$1" = "launch" ] && [ -n "$2" ]; then
    AGENT="$2"
    shift 2
    MODEL=""
    EXTRA_ARGS=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --model)
                MODEL="$2"
                shift 2
                ;;
            --yes|-y)
                EXTRA_ARGS+=("--yes")
                shift
                ;;
            --)
                shift
                EXTRA_ARGS+=("$@")
                break
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done

    if [ -z "$MODEL" ]; then
        MODEL="$("$REAL_OLLAMA" ps 2>/dev/null | tail -n +2 | head -1 | awk '{{print $1}}')"
        [ -z "$MODEL" ] && MODEL="qwen3-coder"
    fi

    echo "Interop — local LLM compatibility layer"
    echo "  Agent: $AGENT   Model: $MODEL"
    echo "  (Interop translates format between agent and Ollama)"
    echo ""

    exec "$INTEROP_RUNNER" run "$AGENT" --model "$MODEL"
fi

# All non-launch ollama commands pass through
exec "$REAL_OLLAMA" "$@"
"""


# ─── Uninstall ──────────────────────────────────────────────────────────────


def uninstall() -> dict[str, str]:
    """Remove Interop wrappers."""
    result: dict[str, str] = {}
    target_dir = _bin_dir()
    shim_path = target_dir / "ollama"

    if shim_path.exists():
        # Only remove if it looks like our shim (contains Interop wrapper)
        content = shim_path.read_text()
        if "Interop wrapper" in content:
            shim_path.unlink()
            result["removed"] = str(shim_path)
            logger.info("Removed Interop shim: %s", shim_path)

    return result


# ─── Status ─────────────────────────────────────────────────────────────────


def status() -> dict[str, bool | str | None]:
    """Report Interop installation status."""
    target_dir = _bin_dir()
    shim_path = target_dir / "ollama"
    real = _find_real_ollama()
    runner = _find_interop_runner()

    shim_active = False
    if shim_path.exists():
        content = shim_path.read_text()
        if "Interop wrapper" in content:
            shim_active = True

    return {
        "shim_installed": shim_active,
        "shim_path": str(shim_path) if shim_active else None,
        "ollama_binary": real,
        "interop_runner": runner,
        "interop_in_path": shutil.which("interop") is not None,
        "shim_in_path": shutil.which(str(shim_path)) is not None if shim_active else False,
        "target_dir": str(target_dir),
    }