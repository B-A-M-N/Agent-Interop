"""Tests for launcher.py defects fixed in the re-audit (P1#12).

Covers:
- _get_ollama_bin(): manifest-first resolution, exact (not substring) shim
  exclusion, and raising rather than fabricating a nonexistent path.
- ollama_pull_model(): actually targets the given url via OLLAMA_HOST.
- ManagedGateway.ensure_ollama(): refuses to auto-start a local Ollama in
  place of an unreachable REMOTE one.
- ManagedGateway.start_gateway(): concurrency limit is configurable, not
  hardcoded; gateway stdout goes to a real log file, not DEVNULL.
- ManagedGateway.launch_agent(): an unregistered agent is rejected unless
  the caller explicitly opts in via assume_protocol.
"""

from __future__ import annotations

import os
import stat

import httpx
import pytest

from agent_interop import launcher

_HARDCODED_OLLAMA_PATHS = ("/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama")


@pytest.fixture(autouse=True)
def _reset_ollama_bin_cache():
    launcher._OLLAMA_BIN_CACHE = None
    yield
    launcher._OLLAMA_BIN_CACHE = None


@pytest.fixture()
def no_real_ollama_on_this_machine(monkeypatch):
    """_get_ollama_bin()'s "known non-shim install paths" tier checks
    fixed, real filesystem locations (e.g. /usr/local/bin/ollama) — on any
    dev machine that actually has Ollama installed there, that tier would
    resolve before a test's PATH/manifest fixtures ever get a chance to
    matter. This neutralizes exactly those three hardcoded paths without
    touching os.access for anything else, including the test's own
    tmp_path fixtures."""
    real_access = os.access

    def fake_access(path, mode, *a, **k):
        if str(path) in _HARDCODED_OLLAMA_PATHS:
            return False
        return real_access(path, mode, *a, **k)

    monkeypatch.setattr(launcher.os, "access", fake_access)


def _make_executable(path) -> None:
    path.write_text("#!/bin/sh\necho fake ollama\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestGetOllamaBinManifestResolution:
    def test_uses_manifest_real_ollama_when_present_and_valid(self, tmp_path, monkeypatch):
        real_ollama = tmp_path / "real-ollama"
        _make_executable(real_ollama)

        bin_dir = tmp_path / "bin_dir"
        bin_dir.mkdir()
        import json
        (bin_dir / ".install_manifest.json").write_text(json.dumps({
            "shim_path": str(bin_dir / "ollama"),
            "real_ollama": str(real_ollama),
        }))

        monkeypatch.setattr("agent_interop.install._bin_dir", lambda: bin_dir)
        assert launcher._get_ollama_bin() == str(real_ollama)

    def test_falls_back_when_manifest_real_ollama_is_stale(
        self, tmp_path, monkeypatch, no_real_ollama_on_this_machine,
    ):
        """If the manifest points at a path that no longer exists, resolution
        must continue to the next tier instead of returning a dead path."""
        bin_dir = tmp_path / "bin_dir"
        bin_dir.mkdir()
        import json
        (bin_dir / ".install_manifest.json").write_text(json.dumps({
            "shim_path": str(bin_dir / "ollama"),
            "real_ollama": str(tmp_path / "does-not-exist"),
        }))
        monkeypatch.setattr("agent_interop.install._bin_dir", lambda: bin_dir)
        # Ensure no known-path/PATH candidate exists either, so the only
        # way this test could pass is if the stale manifest path were
        # wrongly trusted despite not existing on disk.
        monkeypatch.setattr(launcher.shutil, "which", lambda *a, **k: None)
        monkeypatch.setenv("PATH", str(tmp_path / "empty"))
        with pytest.raises(launcher.OllamaBinaryNotFoundError):
            launcher._get_ollama_bin()

    def test_exact_shim_path_excluded_not_substring(
        self, tmp_path, monkeypatch, no_real_ollama_on_this_machine,
    ):
        """Only the manifest's exact shim directory must be excluded from
        PATH resolution — a real, unrelated directory that merely contains
        the substring "interop" (e.g. a checkout named
        "my-interop-project/bin") must NOT be skipped."""
        real_dir = tmp_path / "my-interop-project" / "bin"
        real_dir.mkdir(parents=True)
        real_ollama = real_dir / "ollama"
        _make_executable(real_ollama)

        shim_dir = tmp_path / "shim"
        shim_dir.mkdir()

        bin_dir = tmp_path / "manifest_dir"
        bin_dir.mkdir()
        import json
        (bin_dir / ".install_manifest.json").write_text(json.dumps({
            "shim_path": str(shim_dir / "ollama"),
            "real_ollama": str(tmp_path / "nonexistent"),  # forces fallthrough
        }))
        monkeypatch.setattr("agent_interop.install._bin_dir", lambda: bin_dir)
        monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{real_dir}")

        assert launcher._get_ollama_bin() == str(real_ollama)

    def test_raises_rather_than_fabricating_a_path(
        self, tmp_path, monkeypatch, no_real_ollama_on_this_machine,
    ):
        monkeypatch.setattr("agent_interop.install._bin_dir", lambda: tmp_path / "nope")
        monkeypatch.setattr(launcher.shutil, "which", lambda *a, **k: None)
        monkeypatch.setenv("PATH", str(tmp_path))  # empty dir, nothing resolvable
        with pytest.raises(launcher.OllamaBinaryNotFoundError):
            launcher._get_ollama_bin()


class TestOllamaPullModelTargetsUrl:
    def test_pull_sets_ollama_host_from_url(self, monkeypatch, tmp_path):
        captured_env = {}

        def fake_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)
        monkeypatch.setattr(launcher, "_get_ollama_bin", lambda: "/bin/true")

        launcher.ollama_pull_model("qwen3-coder", url="http://10.0.0.5:11434")
        assert captured_env.get("OLLAMA_HOST") == "10.0.0.5:11434"

    def test_pull_against_default_localhost_sets_host_too(self, monkeypatch):
        captured_env = {}

        def fake_run(cmd, **kwargs):
            captured_env.update(kwargs.get("env") or {})
            class R:
                returncode = 0
            return R()

        monkeypatch.setattr(launcher.subprocess, "run", fake_run)
        monkeypatch.setattr(launcher, "_get_ollama_bin", lambda: "/bin/true")

        launcher.ollama_pull_model("m")
        assert captured_env.get("OLLAMA_HOST") == "127.0.0.1:11434"


class TestIsLoopbackUrl:
    @pytest.mark.parametrize("url", [
        "http://127.0.0.1:11434", "http://localhost:11434", "http://[::1]:11434",
    ])
    def test_loopback_urls(self, url):
        assert launcher._is_loopback_url(url) is True

    @pytest.mark.parametrize("url", [
        "http://10.0.0.5:11434", "http://ollama.internal:11434", "http://192.168.1.50:11434",
    ])
    def test_non_loopback_urls(self, url):
        assert launcher._is_loopback_url(url) is False


class TestEnsureOllamaRefusesRemoteAutoStart:
    def test_unreachable_remote_url_does_not_spawn_local_ollama(self, monkeypatch):
        gw = launcher.ManagedGateway(ollama_url="http://10.0.0.5:11434")
        monkeypatch.setattr(launcher, "is_ollama_running", lambda url: False)

        popen_called = []
        monkeypatch.setattr(
            launcher.subprocess, "Popen",
            lambda *a, **k: popen_called.append((a, k)) or pytest.fail("must not spawn a process"),
        )

        with pytest.raises(SystemExit):
            gw.ensure_ollama()
        assert popen_called == []

    def test_unreachable_loopback_url_still_attempts_local_start(self, monkeypatch):
        """Unchanged behavior for the case this WAS designed for: a
        loopback Ollama that starts running shortly after being spawned."""
        gw = launcher.ManagedGateway(ollama_url="http://127.0.0.1:11434")
        monkeypatch.setattr(launcher, "_get_ollama_bin", lambda: "/bin/true")

        popen_calls = []

        class FakeProcess:
            pass

        def fake_popen(*a, **k):
            popen_calls.append((a, k))
            return FakeProcess()

        # First call (the initial reachability check) -> not running yet;
        # every call after Popen would have spawned it -> running. Avoids
        # a real 15s poll-loop wait for a case this test isn't about.
        running_calls = {"n": 0}

        def fake_is_running(url):
            running_calls["n"] += 1
            return running_calls["n"] > 1

        monkeypatch.setattr(launcher, "is_ollama_running", fake_is_running)
        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(launcher.time, "sleep", lambda s: None)
        monkeypatch.setattr(launcher, "ollama_has_model", lambda model, url: True)

        gw.ensure_ollama()  # must not raise — Popen was attempted for loopback
        assert len(popen_calls) == 1


class TestUnregisteredAgentRejected:
    def test_unregistered_agent_without_assume_protocol_is_rejected(self, monkeypatch):
        gw = launcher.ManagedGateway()
        monkeypatch.setattr("agent_interop.agents.registry.get_agent_integration", lambda name: None)
        with pytest.raises(SystemExit):
            gw.launch_agent("some-totally-unknown-tool")

    def test_unregistered_agent_with_assume_protocol_is_allowed(self, monkeypatch, tmp_path):
        gw = launcher.ManagedGateway()
        monkeypatch.setattr("agent_interop.agents.registry.get_agent_integration", lambda name: None)

        fake_bin = tmp_path / "some-tool"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(fake_bin.stat().st_mode | stat.S_IXUSR)
        monkeypatch.setattr(launcher, "find_agent", lambda name: str(fake_bin))

        captured = {}

        class FakeProcess:
            pass

        def fake_popen(cmd, env=None, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = env
            return FakeProcess()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)

        result = gw.launch_agent("some-tool", assume_protocol="anthropic")
        assert result is not None
        assert captured["env"]["ANTHROPIC_BASE_URL"] == gw._gateway_url


class TestStartGatewayConcurrencyConfigurable:
    def test_context_setting_reaches_managed_gateway_environment(self, monkeypatch, tmp_path):
        gw = launcher.ManagedGateway(ollama_num_ctx=16384)
        monkeypatch.setattr("agent_interop.paths.log_file", lambda: tmp_path / "interop.log")
        captured_kwargs = {}

        class FakeProcess:
            stderr = None

            def poll(self):
                return None

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeProcess()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(launcher.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no server")))
        monkeypatch.setattr(launcher.atexit, "register", lambda fn: None)

        with pytest.raises(RuntimeError, match="did not start"):
            gw.start_gateway(timeout=0.1)

        assert captured_kwargs["env"]["INTEROP_OLLAMA_NUM_CTX"] == "16384"

    def test_concurrency_limit_is_configurable_not_hardcoded(self, monkeypatch, tmp_path):
        gw = launcher.ManagedGateway()
        monkeypatch.setattr("agent_interop.paths.log_file", lambda: tmp_path / "interop.log")

        captured_cmd = {}

        class FakeProcess:
            stderr = None

            def poll(self):
                return None

        def fake_popen(cmd, **kwargs):
            captured_cmd["cmd"] = cmd
            return FakeProcess()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(launcher.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no server")))
        monkeypatch.setattr(launcher.atexit, "register", lambda fn: None)

        with pytest.raises(RuntimeError, match="did not start"):
            gw.start_gateway(timeout=0.1, concurrency_limit=42)

        assert "--limit-concurrency" in captured_cmd["cmd"]
        idx = captured_cmd["cmd"].index("--limit-concurrency")
        assert captured_cmd["cmd"][idx + 1] == "42"

    def test_gateway_stdout_goes_to_a_real_log_file_not_devnull(self, monkeypatch, tmp_path):
        gw = launcher.ManagedGateway()
        log_path = tmp_path / "interop.log"
        monkeypatch.setattr("agent_interop.paths.log_file", lambda: log_path)

        captured_kwargs = {}

        class FakeProcess:
            stderr = None

            def poll(self):
                return None

        def fake_popen(cmd, **kwargs):
            captured_kwargs.update(kwargs)
            return FakeProcess()

        monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
        monkeypatch.setattr(launcher.httpx, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no server")))
        monkeypatch.setattr(launcher.atexit, "register", lambda fn: None)

        with pytest.raises(RuntimeError, match="did not start"):
            gw.start_gateway(timeout=0.1)

        assert captured_kwargs["stdout"] is not None
        assert captured_kwargs["stdout"] != launcher.subprocess.DEVNULL
        assert log_path.exists()


class TestOllamaListModelsDetailed:
    """P2 fix: ollama_list_models() used to collapse 'unreachable',
    'auth failed', 'invalid response', and 'genuinely no models' into the
    same bare []. ollama_list_models_detailed() tells them apart;
    ollama_list_models() stays a backward-compatible thin wrapper."""

    class _FakeResponse:
        def __init__(self, status_code: int, json_body=None, json_error: bool = False):
            self.status_code = status_code
            self._json_body = json_body
            self._json_error = json_error

        def json(self):
            if self._json_error:
                raise ValueError("invalid JSON")
            return self._json_body

    def test_unreachable_backend(self, monkeypatch):
        def raise_error(*a, **k):
            raise httpx.ConnectError("no route to host")

        monkeypatch.setattr(launcher.httpx, "get", raise_error)
        result = launcher.ollama_list_models_detailed()
        assert result.status == launcher.OllamaListStatus.UNREACHABLE
        assert result.models == []
        assert launcher.ollama_list_models() == []

    def test_auth_failed(self, monkeypatch):
        monkeypatch.setattr(launcher.httpx, "get", lambda *a, **k: self._FakeResponse(401))
        result = launcher.ollama_list_models_detailed()
        assert result.status == launcher.OllamaListStatus.AUTH_FAILED
        assert result.models == []

    def test_invalid_response(self, monkeypatch):
        monkeypatch.setattr(
            launcher.httpx, "get",
            lambda *a, **k: self._FakeResponse(200, json_error=True),
        )
        result = launcher.ollama_list_models_detailed()
        assert result.status == launcher.OllamaListStatus.INVALID_RESPONSE

    def test_genuinely_empty(self, monkeypatch):
        monkeypatch.setattr(
            launcher.httpx, "get",
            lambda *a, **k: self._FakeResponse(200, json_body={"models": []}),
        )
        result = launcher.ollama_list_models_detailed()
        assert result.status == launcher.OllamaListStatus.EMPTY
        assert result.models == []

    def test_ok_with_models(self, monkeypatch):
        models = [{"name": "qwen3-coder:latest"}]
        monkeypatch.setattr(
            launcher.httpx, "get",
            lambda *a, **k: self._FakeResponse(200, json_body={"models": models}),
        )
        result = launcher.ollama_list_models_detailed()
        assert result.status == launcher.OllamaListStatus.OK
        assert result.models == models
        assert launcher.ollama_list_models() == models

    def test_server_error_status_is_unreachable(self, monkeypatch):
        monkeypatch.setattr(launcher.httpx, "get", lambda *a, **k: self._FakeResponse(503))
        result = launcher.ollama_list_models_detailed()
        assert result.status == launcher.OllamaListStatus.UNREACHABLE
