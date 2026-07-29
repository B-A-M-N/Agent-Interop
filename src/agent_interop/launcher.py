"""Managed gateway mode — Interop sits between the coding agent and Ollama.

Flow:
    olama launch claude
        → normally sets ANTHROPIC_BASE_URL=http://localhost:11434 (Ollama's /v1/messages)
        → problems: model format mismatches, broken tool calls, bad continuations
    
    interop run claude
        → sets ANTHROPIC_BASE_URL=http://localhost:<interop-port>
        → Interop proxies to Ollama's /api/chat or /v1/chat/completions
        → Interop translates agent protocol (Anthropic Messages) ↔ Ollama protocol
        → Interop handles: model-specific templates, tool-call parsing, validation,
          repair, streaming translation, capability negotiation
        → Result: model gets correct formatting, agent gets correct responses
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import IO

import httpx

logger = logging.getLogger("agent_interop.launcher")

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
# OLLAMA_BIN is resolved lazily at first use; never at module import.
# If Interop's own shim is on PATH, resolving ollama at import time
# would point to the wrapper, causing recursion when launching or pulling.
_OLLAMA_BIN_CACHE: str | None = None
INTEROP_PORT = 8090


class OllamaBinaryNotFoundError(RuntimeError):
    """Raised when the real ollama executable cannot be located anywhere —
    not via the install manifest, known install paths, or PATH."""


def _get_ollama_bin() -> str:
    """Resolve the real ollama executable, bypassing Interop's own shim.

    Preference: installed manifest -> direct known paths -> PATH resolve
    with EXACT shim-path exclusion (not a substring match).

    Raises OllamaBinaryNotFoundError rather than returning a path that may
    not exist — a caller silently getting back a fabricated
    "/usr/local/bin/ollama" that was never actually verified to exist
    would fail later with a much more confusing error (a Popen/subprocess
    FileNotFoundError deep in a launch attempt) instead of a clear message
    naming the actual problem up front.
    """
    global _OLLAMA_BIN_CACHE
    if _OLLAMA_BIN_CACHE is not None:
        return _OLLAMA_BIN_CACHE

    def _valid(path: str | None) -> bool:
        if path is None:
            return False
        return os.path.isfile(path) and os.access(path, os.X_OK)

    # 1. Installed manifest — records the exact real ollama path install()
    # resolved at install time, before the shim existed on PATH. This is
    # the most reliable source: every other strategy below risks
    # resolving back to Interop's own shim once it's on PATH.
    shim_path: str | None = None
    try:
        from agent_interop.install import _bin_dir, _read_manifest

        target_dir = _bin_dir()
        manifest = _read_manifest(target_dir)
        shim_path = manifest.get("shim_path") if manifest else None
        manifest_real_ollama = manifest.get("real_ollama") if manifest else None
        if manifest_real_ollama is not None and _valid(manifest_real_ollama):
            _OLLAMA_BIN_CACHE = manifest_real_ollama
            return manifest_real_ollama
    except ImportError:
        pass

    # 2. Known non-shim install paths.
    for candidate in ("/usr/local/bin/ollama", "/usr/bin/ollama", "/opt/homebrew/bin/ollama"):
        if _valid(candidate):
            _OLLAMA_BIN_CACHE = candidate
            return candidate

    # 3. PATH resolution, excluding ONLY the exact known shim path (from
    # the manifest, if one was read) — not every directory whose name
    # happens to contain "interop", which could wrongly skip a real,
    # unrelated directory (e.g. "/home/user/interop-project/bin").
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    shim_dir = os.path.dirname(shim_path) if shim_path else None
    filtered_dirs = [d for d in path_dirs if d != shim_dir]
    filtered_path = os.pathsep.join(filtered_dirs)

    found = shutil.which("ollama", path=filtered_path)
    if _valid(found):
        _OLLAMA_BIN_CACHE = found
        return found  # type: ignore[return-value]

    raise OllamaBinaryNotFoundError(
        "Could not locate the real ollama executable (checked the install "
        "manifest, common install paths, and PATH). Install Ollama or set "
        "its location explicitly."
    )


def _is_loopback_url(url: str) -> bool:
    """True if ``url``'s host is localhost/127.0.0.1/::1 — the only case
    where auto-starting a local Ollama server as a fallback is safe."""
    host = urllib.parse.urlsplit(url).hostname or ""
    return host in ("localhost", "127.0.0.1", "::1")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def is_ollama_running(url: str = OLLAMA_DEFAULT_URL) -> bool:
    try:
        r = httpx.get(f"{url}/api/version", timeout=2.0)
        return r.status_code == 200
    except (httpx.RequestError, ConnectionError):
        return False


class OllamaListStatus(str, Enum):
    """Why ollama_list_models_detailed() returned what it did — an auth
    failure and "genuinely no models installed" both used to collapse to
    the same empty list, indistinguishable to a caller deciding what to
    tell the operator."""

    OK = "ok"
    AUTH_FAILED = "auth_failed"
    UNREACHABLE = "unreachable"
    INVALID_RESPONSE = "invalid_response"
    EMPTY = "empty"


@dataclass
class OllamaListResult:
    status: OllamaListStatus
    models: list[dict] = field(default_factory=list)


def ollama_list_models_detailed(url: str = OLLAMA_DEFAULT_URL) -> OllamaListResult:
    """List models from Ollama's /api/tags, distinguishing WHY there are
    none — unreachable backend, auth failure, an invalid/unparseable
    response, or a reachable server that genuinely has no models pulled.
    """
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5.0)
    except httpx.HTTPError:
        return OllamaListResult(status=OllamaListStatus.UNREACHABLE)

    if r.status_code in (401, 403):
        return OllamaListResult(status=OllamaListStatus.AUTH_FAILED)
    if r.status_code >= 400:
        return OllamaListResult(status=OllamaListStatus.UNREACHABLE)

    try:
        body = r.json()
    except ValueError:
        return OllamaListResult(status=OllamaListStatus.INVALID_RESPONSE)

    models = body.get("models", []) if isinstance(body, dict) else []
    if not models:
        return OllamaListResult(status=OllamaListStatus.EMPTY)
    return OllamaListResult(status=OllamaListStatus.OK, models=models)


def ollama_list_models(url: str = OLLAMA_DEFAULT_URL) -> list[dict]:
    """Backward-compatible: the models list, or [] for ANY non-OK status
    (unreachable, auth failure, invalid response, or genuinely empty).
    Callers that need to tell those cases apart should use
    ollama_list_models_detailed() instead."""
    return ollama_list_models_detailed(url).models


def ollama_has_model(model: str, url: str = OLLAMA_DEFAULT_URL) -> bool:
    """Check if a model is available, using normalized tag-aware matching.

    Accounts for optional tags (e.g., 'qwen3-coder' matches 'qwen3-coder:latest').
    Uses the same normalizer as gateway readiness probing (model_names.py)
    so the two don't silently diverge on edge cases.
    """
    from agent_interop.model_names import model_names_match

    models = ollama_list_models(url)
    names = [m.get("name", "") for m in models]
    return any(model_names_match(model, name) for name in names)


def ollama_pull_model(model: str, url: str = OLLAMA_DEFAULT_URL) -> bool:
    """Pull a model via Ollama's CLI, targeting ``url``.

    Returns True on success.
    """
    logger.info("Pulling model %s via Ollama at %s...", model, url)
    try:
        # The `ollama pull` CLI subcommand has no --url flag — it always
        # targets whatever OLLAMA_HOST resolves to (default
        # 127.0.0.1:11434). Without setting it explicitly here, `url` was
        # accepted but silently ignored: a caller asking to pull to a
        # configured remote Ollama host would instead pull to whatever
        # local/default server OLLAMA_HOST happened to already point at.
        env = os.environ.copy()
        parsed = urllib.parse.urlsplit(url)
        if parsed.netloc:
            env["OLLAMA_HOST"] = parsed.netloc
        result = subprocess.run(
            [_get_ollama_bin(), "pull", model],
            capture_output=True, text=True, timeout=300, env=env,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.warning("Failed to pull model: %s", exc)
        return False


def find_agent(name: str) -> str | None:
    """Find a coding agent executable."""
    candidates: dict[str, list[str]] = {
        "claude": ["claude", "npx", "npx.cmd"],
        "codex": ["codex"],
        "cline": ["cline"],
        "opencode": ["opencode"],
        "continue": ["continue", "npx"],
        "aider": ["aider", "aider-chat"],
        "cursor": ["cursor"],
        "windsurf": ["windsurf"],
        "gemini-cli": ["gemini"],
        "roo-code": ["roo-code"],
        "copilot": ["copilot"],
    }
    for cmd in candidates.get(name, [name]):
        path = shutil.which(cmd)
        if path:
            return path
    return None


class ManagedGateway:
    """Manages an Interop gateway that sits between the coding agent and Ollama.

    Responsibilities:
    - Verify Ollama is running and the model exists (pull if needed)
    - Start Interop gateway on a local port
    - Set ANTHROPIC_BASE_URL to point at Interop (not Ollama directly)
    - Launch the coding agent
    - Forward signals, clean up on exit
    """

    def __init__(
        self,
        model: str = "qwen3-coder",
        ollama_url: str = OLLAMA_DEFAULT_URL,
        host: str = "127.0.0.1",
        port: int = 0,
        backend_type: str = "ollama",
        auto_pull: bool = True,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self.host = host
        self.port = port or find_free_port()
        self.backend_type = backend_type
        self.auto_pull = auto_pull

        self._session_credential = f"interop_{uuid.uuid4().hex[:24]}"
        self._gateway_url = f"http://{self.host}:{self.port}"
        self._gateway_process: subprocess.Popen | None = None
        self._agent_process: subprocess.Popen | None = None
        self._owned = False
        self._cleanup_fns: list[Callable[[], None]] = []
        self._log_file_path: Path | None = None
        self._log_fh: IO[bytes] | None = None

    # ─── Setup ────────────────────────────────────────────────────────

    def ensure_ollama(self) -> None:
        """Verify Ollama is reachable and the model exists."""
        if not is_ollama_running(self.ollama_url):
            if not _is_loopback_url(self.ollama_url):
                # Never spawn a LOCAL ollama server as a substitute for a
                # configured REMOTE one — that would silently start
                # talking to (and pulling models onto) the wrong machine
                # instead of surfacing that the real target is
                # unreachable.
                print(f"Ollama is not reachable at {self.ollama_url} (not localhost).")
                print("  Refusing to auto-start a local Ollama in its place — "
                      "check the remote host, port, and network path.")
                raise SystemExit(1)
            print(f"Ollama is not running at {self.ollama_url}")
            print("Starting Ollama...")
            subprocess.Popen(
                [_get_ollama_bin(), "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if is_ollama_running(self.ollama_url):
                    print("  Ollama started.")
                    break
                time.sleep(1)
            else:
                print("  Could not start Ollama. Is it installed?")
                print(f"  Try: {_get_ollama_bin()} serve")
                raise SystemExit(1)

        # Check model
        if not ollama_has_model(self.model, self.ollama_url):
            if self.auto_pull:
                print(f"Pulling model '{self.model}'...")
                if not ollama_pull_model(self.model, self.ollama_url):
                    print(f"  Failed to pull {self.model}")
                    print(f"  Try: ollama pull {self.model}")
                    raise SystemExit(1)
                print(f"  Model {self.model} ready.")
            else:
                print(f"Model '{self.model}' not found.")
                print(f"  Try: ollama pull {self.model}")
                raise SystemExit(1)

    def start_gateway(self, timeout: float = 15.0, concurrency_limit: int = 10) -> str:
        """Start the Interop gateway, wait for it to be ready."""
        import threading

        env = os.environ.copy()
        # Interop talks to Ollama
        env["INTEROP_BACKEND_URL"] = self.ollama_url
        env["INTEROP_BACKEND_TYPE"] = self.backend_type
        env["INTEROP_MODEL"] = self.model
        env["INTEROP_PORT"] = str(self.port)
        env["INTEROP_SESSION_CREDENTIAL"] = self._session_credential

        cmd = [
            sys.executable, "-m", "uvicorn",
            "agent_interop.server.app:create_app_from_env",
 "--factory",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "warning",
            "--timeout-graceful-shutdown", "30",
            "--limit-concurrency", str(concurrency_limit),
        ]

        logger.info("Starting Interop gateway on %s", self._gateway_url)
        # Gateway stdout previously went to DEVNULL — any print/log line
        # the app itself emitted on stdout (as opposed to via the
        # `logging` module, which uvicorn routes to stderr) was silently
        # lost, with no way for an operator to see it even by asking.
        # stderr stays piped (see _drain_stderr below) since the in-memory
        # ring buffer it feeds is used for crash diagnostics; both streams
        # are ALSO mirrored to the same on-disk log file so a launch that
        # succeeds still leaves a record an operator can open afterward.
        from agent_interop.paths import log_file
        self._log_file_path = log_file()
        self._log_file_path.parent.mkdir(parents=True, exist_ok=True)
        # Kept open for the process's lifetime (closed in cleanup()), so a
        # `with` block doesn't fit here — noqa is deliberate, not an oversight.
        self._log_fh = open(self._log_file_path, "ab")  # noqa: SIM115
        print(f"  Gateway log: {self._log_file_path}")
        try:
            self._gateway_process = subprocess.Popen(
                cmd,
                env=env,
                stdout=self._log_fh,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except Exception:
            self._log_fh.close()
            self._log_fh = None
            raise
        self._owned = True
        atexit.register(self.cleanup)

        # Drain stderr in a background thread to prevent pipe buffer
        # from filling and blocking the gateway process.
        self._stderr_lines: list[str] = []
        self._stderr_lock = threading.Lock()

        def _drain_stderr() -> None:
            if self._gateway_process is None or self._gateway_process.stderr is None:
                return
            try:
                for line in iter(self._gateway_process.stderr.readline, b""):
                    with self._stderr_lock:
                        # Keep only the last N lines to bound memory
                        self._stderr_lines.append(line.decode(errors="replace"))
                        if len(self._stderr_lines) > 100:
                            self._stderr_lines.pop(0)
                    try:
                        if self._log_fh is not None:
                            self._log_fh.write(line)
                            self._log_fh.flush()
                    except (ValueError, OSError):
                        pass  # log file closed under us during cleanup; stderr draining continues
            except (ValueError, OSError):
                pass

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        # Wait for the gateway PROCESS to come up (liveness, via
        # /v1/health/live — not backend readiness). The gateway is started
        # with ingress_auth.mode=session_token and the generated session
        # credential, so the endpoint is protected — the probe MUST
        # authenticate or it will be rejected with 401 and the launch will
        # time out waiting for a process that actually already started.
        #
        # This intentionally does NOT block on backend readiness: Ollama (or
        # whichever backend) may still be starting up or loading the model
        # when Interop itself is ready to accept connections, and failing
        # the whole launch on that would be worse than starting anyway and
        # letting the first real request surface a clear backend error.
        auth_headers = {"Authorization": f"Bearer {self._session_credential}"}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = httpx.get(
                    f"{self._gateway_url}/v1/health/live",
                    timeout=2.0,
                    headers=auth_headers,
                )
                if r.status_code == 200:
                    logger.info("Gateway process ready at %s", self._gateway_url)
                    # Backgrounded: this check has its own 10s HTTP
                    # timeout and previously ran inline here, meaning
                    # every launch waited out that full timeout whenever
                    # the backend was slow to come up — even though the
                    # gateway process itself was already confirmed live
                    # and start_gateway() has nothing further to verify.
                    threading.Thread(
                        target=self._log_backend_readiness,
                        args=(auth_headers,),
                        daemon=True,
                    ).start()
                    return self._gateway_url
            except Exception:
                pass
            time.sleep(0.3)

        # Check if process crashed
        retcode = self._gateway_process.poll()
        if retcode is not None:
            with self._stderr_lock:
                stderr_text = "".join(self._stderr_lines[-50:])
            raise RuntimeError(
                f"Gateway exited with code {retcode} before becoming ready.\n"
                f"Stderr: {stderr_text[:500]}"
            )
        raise RuntimeError(f"Gateway did not start within {timeout}s")

    def _log_backend_readiness(self, auth_headers: dict[str, str]) -> None:
        """Best-effort backend readiness check, run on a background thread
        (see start_gateway) so its 10s HTTP timeout never delays
        start_gateway()'s return.

        Runs once after the gateway process itself is confirmed live. Never
        raises — it exists purely so an operator watching logs finds out
        the backend isn't actually reachable yet, instead of only
        discovering it on the agent's first real request.
        """
        try:
            r = httpx.get(
                f"{self._gateway_url}/v1/health/ready",
                timeout=10.0,
                headers=auth_headers,
            )
            payload = r.json()
            if payload.get("ready"):
                logger.info("Backend route '%s' ready", payload.get("default_route"))
            else:
                logger.warning(
                    "Interop started, but the backend route is not ready yet: %s",
                    payload.get("routes"),
                )
        except Exception as exc:
            logger.warning("Backend readiness check failed: %s", exc)

    # ─── Agent launch ────────────────────────────────────────────────

    def launch_agent(
        self,
        agent: str,
        extra_args: list[str] | None = None,
        *,
        assume_protocol: str | None = None,
    ) -> subprocess.Popen | None:
        """Launch the coding agent through its AgentIntegration.

        Uses the AgentIntegration registry to build agent-specific launch
        commands, env vars, and config. An agent with no registered
        integration is REJECTED by default — it used to silently fall
        back to Anthropic-compatible env vars and blindly append
        ``--model <name>`` to whatever executable ``find_agent()``
        happened to resolve, for literally any string the user typed.
        That's a real behavior change and credential injection performed
        on a binary Interop knows nothing about; pass
        ``assume_protocol="anthropic"`` or ``"openai_chat"`` to explicitly
        opt into that fallback for a specific unregistered agent (the
        user asserting the contract, not Interop guessing it).

        Returns ``None`` for a ``configuration_required`` integration (e.g.
        Crush) — there is no agent process for us to launch or track; the
        user configures and starts it themselves, and the gateway is left
        running (see ``wait()``) so the credential and base_url just
        printed to them stay valid instead of dying with this process.
        """
        from agent_interop.agents.base import AgentLaunchContext
        from agent_interop.agents.registry import get_agent_integration

        integration = get_agent_integration(agent)
        if integration:
            installation = integration.discover()
            if installation.found:
                context = AgentLaunchContext(
                    route=self.model,
                    gateway_url=self._gateway_url,
                    model_name=self.model,
                    session_credential=self._session_credential,
                    extra_args=tuple(extra_args or ()),
                )
                spec = integration.build_launch(context)

                if spec.readiness == "configuration_required":
                    print(f"Configuration required for {agent}:")
                    for instruction in spec.config_instructions:
                        print(f"  {instruction}")
                    print()
                    print(
                        "  The Interop gateway above is still running so these "
                        "values stay valid — press Ctrl+C here once you're done."
                    )
                    return None

                if spec.readiness == "unsupported":
                    print(f"Agent '{agent}' is not supported for managed launch.")
                    raise SystemExit(1)

                if spec.command:
                    agent_env = os.environ.copy()
                    agent_env.update(spec.env)
                    agent_env["INTEROP_SESSION"] = "1"

                    # Centrally enforce credential injection as a safety net
                    self._inject_session_credential(agent_env, spec)

                    # Register cleanup from spec if provided
                    if spec.cleanup is not None:
                        self._cleanup_fns.append(spec.cleanup)

                    logger.info("Launching %s via integration: %s", agent, " ".join(spec.command))
                    self._agent_process = subprocess.Popen(
                        spec.command, env=agent_env, start_new_session=True
                    )
                    return self._agent_process
            else:
                print(f"Agent '{agent}' not found (integration registered but not installed).")
                print(f"  Integration: {integration.id}")
                raise SystemExit(1)

        # No registered integration for this agent — reject unless the
        # caller explicitly opted into a specific protocol contract for
        # it. Silently guessing Anthropic-compatible for an arbitrary
        # binary is exactly the "unknown agent" defect this replaces.
        if assume_protocol not in ("anthropic", "openai_chat"):
            print(f"'{agent}' has no registered Interop integration.")
            print(
                "  Interop won't guess a protocol/credential contract for an "
                "unknown agent. If it's actually Anthropic- or OpenAI-Chat-"
                "compatible, pass --assume-protocol anthropic (or "
                "openai_chat) to launch it that way explicitly."
            )
            raise SystemExit(1)

        agent_path = find_agent(agent)
        if not agent_path:
            print(f"Could not find '{agent}' in PATH.")
            print("  Install Claude Code: curl -fsSL https://claude.ai/install.sh | bash")
            print("  Install Codex: pip install codex")
            raise SystemExit(1)

        agent_env = os.environ.copy()
        if assume_protocol == "anthropic":
            agent_env["ANTHROPIC_BASE_URL"] = self._gateway_url
            agent_env["ANTHROPIC_AUTH_TOKEN"] = self._session_credential
            agent_env["ANTHROPIC_API_KEY"] = self._session_credential
        else:
            agent_env["OPENAI_BASE_URL"] = self._gateway_url
            agent_env["OPENAI_API_KEY"] = self._session_credential
        agent_env["INTEROP_SESSION"] = "1"

        cmd = [agent_path]
        if extra_args:
            cmd.extend(extra_args)
        if "--model" not in cmd and "-m" not in cmd:
            cmd.extend(["--model", self.model])

        logger.info(
            "Launching %s (unregistered, assume_protocol=%s): %s",
            agent, assume_protocol, " ".join(cmd),
        )
        self._agent_process = subprocess.Popen(cmd, env=agent_env, start_new_session=True)
        return self._agent_process

    def _inject_session_credential(
        self,
        agent_env: dict[str, str],
        spec,
    ) -> None:
        """Centrally enforce credential injection as a safety net.

        Sets the appropriate auth env vars based on the agent's protocol,
        ensuring the generated session credential reaches the agent even
        if an individual integration forgets to set it.
        """
        from agent_interop.abi import ProtocolKind

        protocol = spec.protocol
        if protocol is None:
            # Infer from env already set
            if "ANTHROPIC_BASE_URL" in agent_env:
                protocol = ProtocolKind.ANTHROPIC_MESSAGES
            elif "OPENAI_BASE_URL" in agent_env:
                protocol = ProtocolKind.OPENAI_CHAT
            else:
                return

        if protocol == ProtocolKind.ANTHROPIC_MESSAGES:
            agent_env.setdefault("ANTHROPIC_AUTH_TOKEN", self._session_credential)
            agent_env.setdefault("ANTHROPIC_API_KEY", self._session_credential)
        elif protocol in {ProtocolKind.OPENAI_CHAT, ProtocolKind.OPENAI_RESPONSES}:
            agent_env.setdefault("OPENAI_API_KEY", self._session_credential)

    def wait(self) -> int:
        """Wait for agent to exit. Returns exit code.

        When there is no managed agent process — the
        ``configuration_required`` path (e.g. Crush), where the user
        launches the agent themselves after editing its config — wait on
        the gateway process instead of returning immediately. Returning
        immediately here would let ``run()``'s ``finally: gateway.cleanup()``
        tear the gateway down right after printing its credential and
        base_url, before the user ever gets to use them.
        """
        if not self._agent_process:
            if self._gateway_process:
                try:
                    return self._gateway_process.wait()
                except KeyboardInterrupt:
                    return 0
            return 0
        try:
            return self._agent_process.wait()
        except KeyboardInterrupt:
            self._forward_signal(signal.SIGINT)
            return self._agent_process.wait()

    # ─── Cleanup ─────────────────────────────────────────────────────

    @staticmethod
    def _terminate_process_tree(proc: subprocess.Popen, timeout: float = 5.0) -> None:
        """Terminate ``proc`` and any children it spawned itself.

        ``proc`` is launched with ``start_new_session=True`` (its own
        process group, leader == proc.pid), so signaling the group via
        ``os.killpg`` reaches descendants the process spawned on its own
        (a shell, an MCP server, a pty helper) — ``proc.terminate()``
        alone only ever reaches the direct child, orphaning everything
        under it. Falls back to signaling just the process if the group
        can't be resolved (already reaped, or a platform without
        ``os.killpg``, e.g. Windows).
        """
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            proc.terminate()
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            proc.kill()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass

    def cleanup(self) -> None:
        """Stop agent and gateway, then run registered cleanup functions."""
        # Stop agent first
        if self._agent_process and self._agent_process.poll() is None:
            logger.info("Stopping agent...")
            self._terminate_process_tree(self._agent_process)

        # Stop gateway
        if self._gateway_process and self._gateway_process.poll() is None:
            logger.info("Stopping Interop gateway...")
            self._terminate_process_tree(self._gateway_process)

        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None

        self._owned = False

        # Run registered cleanup functions (temp dirs, etc.)
        for fn in self._cleanup_fns:
            try:
                fn()
            except Exception:
                logger.debug("cleanup function failed (non-fatal)", exc_info=True)
        self._cleanup_fns.clear()

    def _forward_signal(self, sig: signal.Signals) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            try:
                self._agent_process.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass


# ─── Main entry point ─────────────────────────────────────────────────────


def run(agent: str = "claude", model: str = "qwen3-coder",
        ollama_url: str = OLLAMA_DEFAULT_URL, extra_args: list[str] | None = None,
        assume_protocol: str | None = None) -> int:
    """High-level: ensure Ollama → start Interop → launch agent → wait → cleanup.

    This is the pipeline that makes local models work in coding agents:
    Ollama handles model inference, Interop handles the protocol mismatch.
    """
    gateway = ManagedGateway(model=model, ollama_url=ollama_url)

    print("Interop — local LLM compatibility layer")
    print(f"  Model:   {model}")
    print(f"  Backend: Ollama ({ollama_url})")
    print(f"  Agent:   {agent}")
    print()

    # Step 1: Make sure Ollama is running and has the model
    gateway.ensure_ollama()

    # Step 2: Start Interop gateway (translates between agent ↔ Ollama)
    gateway_url = gateway.start_gateway()
    print(f"  Gateway: {gateway_url}  (ANTHROPIC_BASE_URL)")
    print("  (Interop translates Anthropic Messages ↔ Ollama format)")
    print()

    # Step 3: Launch the coding agent pointed at Interop
    agent_process = gateway.launch_agent(agent, extra_args, assume_protocol=assume_protocol)

    # Step 4: Wait for it to finish
    if agent_process is not None:
        print(f"  {agent} started. Press Ctrl+C to stop.\n")
    try:
        return gateway.wait()
    finally:
        gateway.cleanup()