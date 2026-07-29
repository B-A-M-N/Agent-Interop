"""interop install — sets up transparent interception so existing commands Just Work.

What it does:
1. Writes a shim script literally named `ollama` into the front of PATH
   (see _bin_dir()) that intercepts `ollama launch <agent>` and routes it
   through Interop instead of directly to Ollama; every other `ollama`
   subcommand is passed through to the real binary unchanged.

2. If a non-Interop `ollama` wrapper already exists there, it's backed up
   (see the install manifest) so uninstall() can restore it exactly,
   rather than being overwritten and lost.

After install:
    ollama launch claude  →  Interop intercepts → starts gateway →
                             launches Claude Code pointed at Interop
    ollama serve          →  passed through to real Ollama normally
    ollama pull model     →  passed through to real Ollama normally

Ownership/identity: the install manifest is the single source of truth for
"is this file ours" (see _shim_matches_manifest) — a canonical realpath +
content-hash match, not a marker-text substring check, which can misidentify
an unrelated script that happens to quote the phrase "Interop wrapper" (e.g.
this very file). The marker-text check is used ONLY as a fallback when no
manifest exists yet (a first-ever install, nothing recorded to compare
against). Every install/uninstall transaction runs under an exclusive,
non-blocking file lock so two concurrent invocations fail fast instead of
interleaving writes.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import logging
import os
import shlex
import shutil
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger("agent_interop.install")

MANIFEST_FILENAME = "install_manifest.json"
LOCK_FILENAME = ".interop-install.lock"
INSTALLER_VERSION = 2


# ─── Install paths ──────────────────────────────────────────────────────────


def _bin_dir() -> Path:
    """Find the best directory for user-level binaries."""
    candidates = [
        Path.home() / ".local" / "bin",
        Path.home() / "bin",
    ]
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for d in candidates:
        d.mkdir(parents=True, exist_ok=True)
        if str(d) in path_dirs:
            return d
    # Always ensure ~/.local/bin is in PATH
    candidates[0].mkdir(parents=True, exist_ok=True)
    return candidates[0]


def _find_real_ollama(target_dir: Path | None = None) -> str | None:
    """Find the real ollama binary, skipping our own shim.

    When ``target_dir`` is given, the install manifest's recorded
    ``shim_realpath`` is used to positively exclude our own shim by
    canonical path — following symlinks, so a candidate that merely
    resolves TO the shim (rather than being a separate copy that happens
    to contain similar text) is never mistaken for a real installation.
    Falls back to the content-marker heuristic only when there is no
    manifest yet to compare against (nothing else to go on for a
    first-ever install). Keeps checking remaining candidates instead of
    returning the shim itself.
    """
    manifest = _read_manifest(target_dir) if target_dir is not None else None
    shim_realpath = manifest.get("shim_realpath") if manifest else None

    for path in ("/usr/local/bin/ollama", "/usr/bin/ollama", shutil.which("ollama")):
        if not path or not os.path.exists(path):
            continue
        real = os.path.realpath(path)
        if shim_realpath and real == shim_realpath:
            continue
        if not shim_realpath:
            try:
                if "Interop wrapper" in Path(path).read_text():
                    continue
            except (OSError, UnicodeDecodeError):
                pass
        return path
    return None


def _find_interop_runner() -> list[str]:
    """Find the interop CLI entry point as an argv list."""
    # Check if interop module is importable
    interop_bin = shutil.which("interop")
    if interop_bin:
        return [interop_bin]
    local_bin = os.path.expanduser("~/.local/bin/interop")
    if os.path.exists(local_bin):
        return [local_bin]
    # Default to module invocation as separate argv entries
    python_bin = shutil.which('python3') or 'python3'
    return [python_bin, "-m", "agent_interop.cli"]


# ─── Concurrency: one install/uninstall transaction at a time ──────────────


@contextlib.contextmanager
def _installer_lock(target_dir: Path):
    """Exclusive, non-blocking lock for the full install/uninstall
    transaction. A second concurrent invocation fails fast with a clear
    message instead of interleaving shim/manifest writes with the first."""
    target_dir.mkdir(parents=True, exist_ok=True)
    lock_path = target_dir / LOCK_FILENAME
    # Held for the whole `with` body below, not just this function — a
    # plain `with open(...)` doesn't fit.
    fh = open(lock_path, "w")  # noqa: SIM115
    try:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                f"Another interop install/uninstall is already in progress "
                f"(lock held on {lock_path}). Wait for it to finish before "
                f"retrying."
            ) from exc
        yield
    finally:
        fh.close()


# ─── Install manifest ───────────────────────────────────────────────────────
#
# Records exactly what a prior install() touched — in particular, the exact
# path any pre-existing non-Interop wrapper was moved to — so uninstall()
# can restore the original artifact precisely instead of guessing a fixed
# convention name that a real third-party file might also happen to use.
# Also the sole source of truth for shim ownership (shim_realpath +
# shim_sha256) and transaction visibility (transaction_state).


def _manifest_path(target_dir: Path) -> Path:
    return target_dir / f".{MANIFEST_FILENAME}"


def _read_manifest(target_dir: Path) -> dict[str, Any] | None:
    path = _manifest_path(target_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _fsync_path(path: Path, *, is_dir: bool = False) -> None:
    """fsync a file or directory fd — survives a crash immediately after,
    not just an interrupted rename. Best-effort: some filesystems/platforms
    don't support fsync on a directory fd; that's not fatal here."""
    flags = os.O_RDONLY | (os.O_DIRECTORY if is_dir and hasattr(os, "O_DIRECTORY") else 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _write_manifest(target_dir: Path, data: dict[str, Any]) -> None:
    """Atomic temp-file + os.replace, chmod 0o600 (the manifest records
    real filesystem paths — operator-only, not world-readable), and fsync
    both the temp file and the containing directory so the write survives
    a crash, not just an interrupted rename."""
    path = _manifest_path(target_dir)
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(json.dumps(data, indent=2))
    tmp_path.chmod(0o600)
    _fsync_path(tmp_path)
    os.replace(tmp_path, path)
    _fsync_path(target_dir, is_dir=True)


def _shim_matches_manifest(shim_path: Path, manifest: dict[str, Any] | None) -> bool:
    """Is ``shim_path`` the exact file Interop's own manifest recorded
    installing? Canonical realpath + content sha256, not a marker-text
    substring — a substring match can misidentify an unrelated script that
    happens to quote the phrase "Interop wrapper" (e.g. this very file, or
    a decoy). Falls back to the marker-text heuristic ONLY when no
    manifest exists at all — there is nothing else to compare against for
    a first-ever install.
    """
    if not shim_path.exists():
        return False
    if manifest and manifest.get("shim_realpath") and manifest.get("shim_sha256"):
        try:
            if os.path.realpath(shim_path) != manifest["shim_realpath"]:
                return False
            return hashlib.sha256(shim_path.read_bytes()).hexdigest() == manifest["shim_sha256"]
        except OSError:
            return False
    try:
        content = shim_path.read_text()
    except (OSError, UnicodeDecodeError):
        return False
    return "Interop wrapper" in content


def _atomic_write_executable(path: Path, content: str) -> None:
    """Write content to ``path`` atomically (temp file + rename), fsyncing
    the temp file and the containing directory.

    A reader (e.g. a shell already resolving `ollama` from PATH) can never
    observe a partially-written or truncated shim, which a direct
    ``write_text()`` does not guarantee if the process is interrupted
    mid-write — and without the fsync, neither does a rename that hasn't
    reached disk yet if the process is killed immediately after.
    """
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(content)
    tmp_path.chmod(tmp_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    _fsync_path(tmp_path)
    os.replace(tmp_path, path)
    _fsync_path(path.parent, is_dir=True)


# ─── Install ────────────────────────────────────────────────────────────────


def install(
    bin_dir: str | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, str]:
    """Install Interop wrappers for transparent local LLM compatibility.

    Returns a dict with paths of what was installed. Raises RuntimeError if
    no real ollama binary can be found (previously this fabricated
    "/usr/local/bin/ollama" and proceeded, which meant a shim could be
    installed pointing at a binary that was never actually verified to
    exist) or if another install/uninstall is already in progress.
    """
    target_dir = Path(bin_dir) if bin_dir else _bin_dir()

    with _installer_lock(target_dir):
        result: dict[str, str] = {}
        existing_manifest = _read_manifest(target_dir)

        real_ollama = _find_real_ollama(target_dir)
        if not real_ollama:
            raise RuntimeError(
                "Could not find a real ollama binary (checked "
                "/usr/local/bin/ollama, /usr/bin/ollama, and PATH). Install "
                "Ollama, or ensure it's reachable, before running "
                "`interop install`."
            )

        interop_runner = _find_interop_runner()
        shim_path = target_dir / "ollama"
        # Named uniquely to Interop's own bookkeeping — never collides with a
        # real third-party file the way a fixed "ollama-vulkan" guess could.
        backup_path = target_dir / "ollama.interop-backup"

        original_wrapper: str | None = (
            existing_manifest.get("original_wrapper") if existing_manifest else None
        )

        if dry_run:
            result["dry_run"] = "true"
            result["would_use_real_ollama"] = real_ollama
            result["would_install_shim_at"] = str(shim_path)
            if shim_path.exists() and not _shim_matches_manifest(shim_path, existing_manifest):
                result["would_back_up_existing_wrapper_to"] = str(backup_path)
            elif shim_path.exists() and not force:
                result["already_installed"] = "true"
            return result

        renamed_this_run = False
        if shim_path.exists():
            if _shim_matches_manifest(shim_path, existing_manifest):
                if not force:
                    logger.info("Interop shim already installed at %s (use --force to reinstall)", shim_path)
                    result["shim"] = str(shim_path)
                    return result
                shim_path.unlink()
            else:
                # Existing non-Interop wrapper — back it up so we can install
                # ours, and ours will chain through it. Never overwrite a file
                # that's already there under the backup name; that could be
                # unrelated user data, not something install() created.
                if backup_path.exists():
                    raise RuntimeError(
                        f"Cannot install: {shim_path} is a non-Interop wrapper, "
                        f"but the backup destination {backup_path} already "
                        f"exists. Resolve this manually (move or remove "
                        f"{backup_path}) before reinstalling."
                    )
                logger.info("Found existing ollama wrapper — backing up to %s", backup_path)
                shim_path.rename(backup_path)
                original_wrapper = str(backup_path)
                renamed_this_run = True
                result["renamed"] = str(backup_path)

        # Written BEFORE the shim itself, so a process killed mid-install
        # leaves a manifest whose transaction_state visibly says "pending"
        # instead of one that looks like a finished, trustworthy record.
        _write_manifest(target_dir, {
            "shim_path": str(shim_path),
            "original_wrapper": original_wrapper,
            "real_ollama": real_ollama,
            "installer_version": INSTALLER_VERSION,
            "transaction_state": "pending",
        })

        try:
            pass_through = original_wrapper or real_ollama
            shim_content = _build_shim(real_ollama, pass_through, interop_runner)
            _atomic_write_executable(shim_path, shim_content)
            shim_hash = hashlib.sha256(shim_content.encode()).hexdigest()
            shim_realpath = os.path.realpath(shim_path)
        except Exception:
            # Roll back: restore whatever was backed up this run so a
            # failed install doesn't leave the operator with no working
            # `ollama` at all.
            if renamed_this_run and backup_path.exists() and not shim_path.exists():
                backup_path.rename(shim_path)
            raise

        result["shim"] = str(shim_path)
        logger.info("Installed Interop shim: %s", shim_path)

        _write_manifest(target_dir, {
            "shim_path": str(shim_path),
            "original_wrapper": original_wrapper,
            "real_ollama": real_ollama,
            "shim_realpath": shim_realpath,
            "shim_sha256": shim_hash,
            "installer_version": INSTALLER_VERSION,
            "transaction_state": "complete",
        })

        # Check PATH
        if str(target_dir) not in os.environ.get("PATH", "").split(os.pathsep):
            logger.info("Add to your shell rc: export PATH=\"%s:$PATH\"", target_dir)

        result["ollama"] = real_ollama
        result["target_dir"] = str(target_dir)
        result["interop_runner"] = " ".join(interop_runner)

        return result


def _build_shim(real_ollama: str, pass_through: str, interop_runner: list[str]) -> str:
    """Build the ollama wrapper shim script.

    ``pass_through`` is the resolved binary/wrapper non-launch commands are
    chained to — either a previously-existing wrapper this install backed
    up (see ``install()``), or ``real_ollama`` when there was none.

    Uses safe shell quoting via shlex.quote() for all interpolated paths.
    The interop_runner is an argv list; the shim uses `exec` with proper
    word splitting rather than a single quoted string.
    """
    quoted_pass_through = shlex.quote(pass_through)
    quoted_interop_runner = " ".join(shlex.quote(a) for a in interop_runner)

    return f"""#!/bin/bash
# Interop wrapper for `ollama launch`
# Routes `ollama launch <agent>` through Interop's compatibility layer.
# All other ollama commands pass through to the real binary.

REAL_OLLAMA={quoted_pass_through}
INTEROP_RUNNER=({quoted_interop_runner})

if [ "$1" = "launch" ] && [ -n "$2" ]; then
    AGENT="$2"
    shift 2
    MODEL=""
    EXTRA_ARGS=()

    while [ $# -gt 0 ]; do
        case "$1" in
            --model)
                if [ $# -lt 2 ]; then
                    echo "Interop wrapper: --model requires a value" >&2
                    exit 1
                fi
                MODEL="$2"
                shift 2
                ;;
            --model=*)
                MODEL="${{1#--model=}}"
                shift
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

    # `--` separates Interop's own --model flag from EXTRA_ARGS so an agent
    # flag that looks like an Interop/Typer option (e.g. a leading `-`) is
    # never misparsed as one. "${{EXTRA_ARGS[@]}}" (no ":-" fallback) expands
    # to zero words when the array is empty — EXTRA_ARGS is always declared
    # above, so there is no unset case to fall back from, and the ":-" form
    # risked passing a single spurious empty-string argument on some bash
    # versions when the array had never been appended to.
    exec "${{INTEROP_RUNNER[@]}}" run "$AGENT" --model "$MODEL" -- "${{EXTRA_ARGS[@]}}"
fi

# All non-launch ollama commands pass through
exec "$REAL_OLLAMA" "$@"
"""


# ─── Uninstall ──────────────────────────────────────────────────────────────


def uninstall() -> dict[str, str]:
    """Remove Interop wrappers and restore the original artifact install()
    backed up, using the install manifest to know exactly what that was
    rather than guessing a fixed naming convention.

    Ownership is decided via manifest-backed identity (_shim_matches_manifest),
    never by re-checking marker text alone. A shim that doesn't match the
    manifest, or a manifest left in "pending" state by a crashed install,
    is left untouched with a message describing exactly what's inconsistent
    — never silently deleted.
    """
    target_dir = _bin_dir()
    shim_path = target_dir / "ollama"

    with _installer_lock(target_dir):
        result: dict[str, str] = {}
        manifest = _read_manifest(target_dir)

        if manifest is not None and manifest.get("transaction_state") == "pending":
            msg = (
                f"A previous install left an incomplete transaction in "
                f"{target_dir} (transaction_state=pending) — it may have been "
                f"interrupted mid-install. Refusing to uninstall automatically; "
                f"inspect {shim_path} and {_manifest_path(target_dir)} by hand."
            )
            logger.warning(msg)
            result["manual_recovery_required"] = msg
            return result

        if shim_path.exists():
            if not _shim_matches_manifest(shim_path, manifest):
                msg = (
                    f"{shim_path} does not match the recorded install manifest "
                    f"(different realpath/content hash) — refusing to delete a "
                    f"file Interop may not actually own."
                )
                logger.warning(msg)
                result["not_removed"] = msg
                return result

            shim_path.unlink()
            result["removed"] = str(shim_path)
            logger.info("Removed Interop shim: %s", shim_path)

            original_wrapper = manifest.get("original_wrapper") if manifest else None
            restore_path = (
                Path(original_wrapper) if original_wrapper
                # Fall back to the legacy fixed name for installs performed
                # before the manifest existed.
                else target_dir / "ollama-vulkan"
            )
            if restore_path.exists():
                restore_path.rename(shim_path)
                result["restored"] = str(shim_path)
                logger.info("Restored previous wrapper: %s → %s", restore_path, shim_path)

        manifest_path = _manifest_path(target_dir)
        if manifest_path.exists():
            manifest_path.unlink()

        return result


# ─── Status ─────────────────────────────────────────────────────────────────


def status() -> dict[str, bool | str | None]:
    """Report Interop installation status."""
    target_dir = _bin_dir()
    shim_path = target_dir / "ollama"
    manifest = _read_manifest(target_dir)
    real = _find_real_ollama(target_dir)
    runner = _find_interop_runner()

    shim_active = _shim_matches_manifest(shim_path, manifest)
    transaction_pending = bool(manifest and manifest.get("transaction_state") == "pending")

    # Check PATH precedence — does the user's `ollama` resolve to our shim?
    resolved_ollama = shutil.which("ollama")
    shim_in_path = (resolved_ollama is not None and os.path.samefile(resolved_ollama, str(shim_path))) if shim_active and shim_path.exists() else False

    return {
        "shim_installed": shim_active,
        "shim_path": str(shim_path) if shim_active else None,
        "ollama_binary": real,
        "interop_runner": " ".join(runner),
        "interop_in_path": shutil.which("interop") is not None,
        "shim_in_path": shim_in_path,
        "target_dir": str(target_dir),
        "transaction_pending": transaction_pending,
    }
