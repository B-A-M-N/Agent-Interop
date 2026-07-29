"""Transactional-safety tests for interop.install (REVISION #7).

Marker-text ("Interop wrapper" substring) was replaced as the SOLE
ownership check with manifest-backed identity (canonical realpath +
content sha256) — a substring match can misidentify an unrelated script
that happens to quote that exact phrase (e.g. install.py itself, or a
decoy). These tests cover: interrupted-install recovery, real-ollama
resolution never returning the shim's own path, uninstall never deleting
an unrelated/decoy script, concurrent install/uninstall locking, repeated
install/uninstall cycles, and a manifest/file hash mismatch being treated
as "not ours".
"""

from __future__ import annotations

import fcntl
import json
import os

import pytest

from agent_interop.install import (
    _find_real_ollama,
    _installer_lock,
    _manifest_path,
    _read_manifest,
    _shim_matches_manifest,
    install,
    uninstall,
)


class TestNoRealOllamaFound:
    def test_install_raises_instead_of_fabricating_a_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("agent_interop.install._find_real_ollama", lambda target_dir=None: None)
        with pytest.raises(RuntimeError, match="Could not find a real ollama"):
            install(bin_dir=str(tmp_path), force=True)
        # Nothing should have been written — a failed resolution must not
        # leave any shim/manifest artifacts behind.
        assert not (tmp_path / "ollama").exists()
        assert not _manifest_path(tmp_path).exists()


class TestDryRun:
    def test_dry_run_touches_nothing(self, tmp_path):
        result = install(bin_dir=str(tmp_path), dry_run=True)
        assert result.get("dry_run") == "true"
        assert not (tmp_path / "ollama").exists()
        assert not _manifest_path(tmp_path).exists()

    def test_dry_run_reports_backup_would_happen(self, tmp_path):
        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\necho 'some other wrapper'")
        result = install(bin_dir=str(tmp_path), dry_run=True)
        assert "would_back_up_existing_wrapper_to" in result
        # Still untouched.
        assert shim_path.read_text() == "#!/bin/bash\necho 'some other wrapper'"


class TestManifestBackedIdentity:
    def test_shim_matches_manifest_requires_realpath_and_hash(self, tmp_path):
        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\necho hi")
        wrong_manifest = {
            "shim_realpath": str(shim_path),
            "shim_sha256": "0" * 64,  # deliberately wrong hash
        }
        assert _shim_matches_manifest(shim_path, wrong_manifest) is False

    def test_falls_back_to_marker_text_only_without_manifest(self, tmp_path):
        shim_path = tmp_path / "ollama"
        shim_path.write_text("# Interop wrapper for ollama\necho hi")
        assert _shim_matches_manifest(shim_path, None) is True

        shim_path.write_text("echo not ours")
        assert _shim_matches_manifest(shim_path, None) is False

    def test_hash_mismatch_after_manual_edit_is_not_ours(self, tmp_path):
        """A shim that install() wrote, then someone hand-edited afterward
        (content changed, same path) must no longer be treated as ours —
        the manifest's sha256 won't match anymore."""
        install(bin_dir=str(tmp_path), force=True)
        shim_path = tmp_path / "ollama"
        manifest = _read_manifest(tmp_path)
        assert manifest is not None
        assert _shim_matches_manifest(shim_path, manifest) is True

        shim_path.write_text(shim_path.read_text() + "\n# tampered\n")
        assert _shim_matches_manifest(shim_path, manifest) is False


class TestUninstallNeverDeletesUnownedFiles:
    def test_decoy_with_marker_text_not_matching_manifest_is_preserved(self, tmp_path):
        """A real install() happens, THEN someone (or something) replaces
        the shim file with a decoy that still contains the literal phrase
        "Interop wrapper" (e.g. a comment quoting this exact module) but
        was never actually written by install(). uninstall() must refuse
        to delete it — marker-text alone is no longer sufficient."""
        install(bin_dir=str(tmp_path), force=True)
        shim_path = tmp_path / "ollama"
        decoy_content = (
            "#!/bin/bash\n"
            "# This script's name shows up in docs mentioning the phrase "
            "'Interop wrapper' but it is not one.\n"
            "echo decoy\n"
        )
        shim_path.write_text(decoy_content)

        result = _uninstall_at(tmp_path)
        assert "not_removed" in result
        assert shim_path.read_text() == decoy_content
        # The manifest itself is left alone too, since nothing was resolved.
        assert _manifest_path(tmp_path).exists()

    def test_unrelated_script_without_manifest_ever_existing_is_untouched(self, tmp_path):
        """No install() ever ran in this bin_dir — an unrelated script
        happening to sit at the 'ollama' path with no Interop marker at
        all must never be removed."""
        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\necho 'someone elses script'")
        result = _uninstall_at(tmp_path)
        assert result.get("removed") is None
        assert shim_path.exists()


class TestPendingTransactionVisibility:
    def test_pending_manifest_blocks_automatic_uninstall(self, tmp_path):
        """A manifest left in transaction_state='pending' (simulating a
        crash between the pending write and the completed write) must
        cause uninstall() to refuse and describe the inconsistency,
        rather than silently deleting whatever is at the shim path."""
        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\n# Interop wrapper (partial)\n")
        _manifest_path(tmp_path).write_text(json.dumps({
            "shim_path": str(shim_path),
            "original_wrapper": None,
            "real_ollama": "/usr/local/bin/ollama",
            "transaction_state": "pending",
        }))

        result = _uninstall_at(tmp_path)

        assert "manual_recovery_required" in result
        assert shim_path.exists()
        assert _manifest_path(tmp_path).exists()


class TestFindRealOllamaNeverReturnsShim:
    def test_shim_realpath_excluded_even_when_first_candidate(self, tmp_path, monkeypatch):
        shim_path = tmp_path / "ollama"
        shim_path.write_text("#!/bin/bash\necho shim")
        shim_path.chmod(0o755)
        manifest = {"shim_realpath": os.path.realpath(shim_path)}
        monkeypatch.setattr("agent_interop.install._read_manifest", lambda target_dir: manifest)
        # shutil.which("ollama") resolves to the shim itself (as it would
        # once the shim is on PATH) — must not be returned.
        monkeypatch.setattr("shutil.which", lambda name: str(shim_path))
        result = _find_real_ollama(tmp_path)
        assert result is None or os.path.realpath(result) != os.path.realpath(shim_path)


class TestConcurrentInstallLocking:
    def test_second_concurrent_install_fails_fast(self, tmp_path):
        with _installer_lock(tmp_path):
            with pytest.raises(RuntimeError, match="already in progress"):
                with _installer_lock(tmp_path):
                    pass  # pragma: no cover — should never reach here

    def test_lock_is_released_after_context_exits(self, tmp_path):
        with _installer_lock(tmp_path):
            pass
        # A fresh lock acquisition must succeed now that the first was released.
        with _installer_lock(tmp_path):
            pass

    def test_lock_file_uses_flock_semantics(self, tmp_path):
        """Sanity-check the lock is a real flock, not just a sentinel file
        whose mere existence blocks re-entry."""
        lock_path = tmp_path / ".interop-install.lock"
        with _installer_lock(tmp_path):
            assert lock_path.exists()
            # A non-blocking attempt from a separate fd must fail while held.
            with open(lock_path) as fh, pytest.raises(OSError):
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class TestRepeatedInstallUninstallCycles:
    def test_install_uninstall_install_uninstall(self, tmp_path):
        for _ in range(3):
            result = install(bin_dir=str(tmp_path), force=True)
            assert "shim" in result
            assert (tmp_path / "ollama").exists()
            uninstall_result = _uninstall_at(tmp_path)
            assert uninstall_result.get("removed") == str(tmp_path / "ollama")
            assert not (tmp_path / "ollama").exists()
            assert not _manifest_path(tmp_path).exists()


def _uninstall_at(target_dir) -> dict:
    import agent_interop.install as install_mod
    real_bin_dir = install_mod._bin_dir
    install_mod._bin_dir = lambda: target_dir
    try:
        return uninstall()
    finally:
        install_mod._bin_dir = real_bin_dir
