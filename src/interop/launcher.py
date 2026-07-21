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

import asyncio
import atexit
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("interop.launcher")

OLLAMA_DEFAULT_URL = "http://127.0.0.1:11434"
OLLAMA_BIN = shutil.which("ollama") or "/usr/local/bin/ollama"
INTEROP_PORT = 8090


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


def ollama_list_models(url: str = OLLAMA_DEFAULT_URL) -> list[dict]:
    try:
        r = httpx.get(f"{url}/api/tags", timeout=5.0)
        return r.json().get("models", [])
    except Exception:
        return []


def ollama_has_model(model: str, url: str = OLLAMA_DEFAULT_URL) -> bool:
    models = ollama_list_models(url)
    names = [m.get("name", "") for m in models]
    return any(model in n for n in names)


def ollama_pull_model(model: str, url: str = OLLAMA_DEFAULT_URL) -> bool:
    """Pull a model via Ollama's API. Returns True on success."""
    logger.info("Pulling model %s via Ollama...", model)
    try:
        result = subprocess.run(
            [OLLAMA_BIN, "pull", model],
            capture_output=True, text=True, timeout=300,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.warning("Failed to pull model: %s", exc)
        return False


def find_agent(name: str) -> str | None:
    """Find a coding agent executable."""
    candidates = {
        "claude": ["claude", "npx", "npx.cmd"],
        "codex": ["codex"],
        "cline": ["cline"],
        "opencode": ["opencode"],
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

    # ─── Setup ────────────────────────────────────────────────────────

    def ensure_ollama(self) -> None:
        """Verify Ollama is reachable and the model exists."""
        if not is_ollama_running(self.ollama_url):
            print(f"Ollama is not running at {self.ollama_url}")
            print("Starting Ollama...")
            subprocess.Popen(
                [OLLAMA_BIN, "serve"],
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
                print(f"  Try: {OLLAMA_BIN} serve")
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

    def start_gateway(self, timeout: float = 15.0) -> str:
        """Start the Interop gateway, wait for it to be ready."""
        env = os.environ.copy()
        # Interop talks to Ollama
        env["INTEROP_BACKEND_URL"] = self.ollama_url
        env["INTEROP_BACKEND_TYPE"] = self.backend_type
        env["INTEROP_MODEL"] = self.model
        env["INTEROP_PORT"] = str(self.port)
        env["INTEROP_SESSION_CREDENTIAL"] = self._session_credential

        cmd = [
            sys.executable, "-m", "uvicorn",
            "interop.server.app:create_app",
            "--host", self.host,
            "--port", str(self.port),
            "--log-level", "warning",
        ]

        logger.info("Starting Interop gateway on %s", self._gateway_url)
        self._gateway_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._owned = True
        atexit.register(self.cleanup)

        # Wait for readiness
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{self._gateway_url}/v1/health", timeout=2.0)
                if r.status_code == 200:
                    logger.info("Gateway ready at %s", self._gateway_url)
                    return self._gateway_url
            except Exception:
                pass
            time.sleep(0.3)

        # Check if process crashed
        retcode = self._gateway_process.poll()
        if retcode is not None:
            stderr = self._gateway_process.stderr.read().decode() if self._gateway_process.stderr else ""
            raise RuntimeError(
                f"Gateway exited with code {retcode} before becoming ready.\n"
                f"Stderr: {stderr[:500]}"
            )
        raise RuntimeError(f"Gateway did not start within {timeout}s")

    # ─── Agent launch ────────────────────────────────────────────────

    def launch_agent(self, agent: str, extra_args: list[str] | None = None) -> subprocess.Popen:
        """Launch the coding agent, pointing its Anthropic API at Interop.

        This is what ollama launch claude does, except ANTHROPIC_BASE_URL
        points to Interop instead of directly to Ollama.
        """
        agent_path = find_agent(agent)
        if not agent_path:
            print(f"Could not find '{agent}' in PATH.")
            print(f"  Install Claude Code: curl -fsSL https://claude.ai/install.sh | bash")
            print(f"  Install Codex: pip install codex")
            raise SystemExit(1)

        # Set the same env vars ollama launch would, but point at Interop
        agent_env = os.environ.copy()
        agent_env["ANTHROPIC_BASE_URL"] = self._gateway_url
        agent_env["ANTHROPIC_AUTH_TOKEN"] = self._session_credential
        agent_env["ANTHROPIC_API_KEY"] = self._session_credential
        agent_env["INTEROP_SESSION"] = "1"

        cmd = [agent_path]
        if extra_args:
            cmd.extend(extra_args)
        # Add --model so the agent knows which model name
        if "--model" not in cmd and "-m" not in cmd:
            cmd.extend(["--model", self.model])

        logger.info("Launching %s: %s", agent, " ".join(cmd))
        self._agent_process = subprocess.Popen(cmd, env=agent_env)
        return self._agent_process

    def wait(self) -> int:
        """Wait for agent to exit. Returns exit code."""
        if not self._agent_process:
            return 0
        try:
            return self._agent_process.wait()
        except KeyboardInterrupt:
            self._forward_signal(signal.SIGINT)
            return self._agent_process.wait()

    # ─── Cleanup ─────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop agent and gateway."""
        # Stop agent first
        if self._agent_process and self._agent_process.poll() is None:
            logger.info("Stopping agent...")
            self._agent_process.terminate()
            try:
                self._agent_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._agent_process.kill()

        # Stop gateway
        if self._gateway_process and self._gateway_process.poll() is None:
            logger.info("Stopping Interop gateway...")
            self._gateway_process.terminate()
            try:
                self._gateway_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._gateway_process.kill()

        self._owned = False

    def _forward_signal(self, sig: signal.Signals) -> None:
        if self._agent_process and self._agent_process.poll() is None:
            try:
                self._agent_process.send_signal(sig)
            except (ProcessLookupError, OSError):
                pass


# ─── Main entry point ─────────────────────────────────────────────────────


def run(agent: str = "claude", model: str = "qwen3-coder",
        ollama_url: str = OLLAMA_DEFAULT_URL, extra_args: list[str] | None = None) -> int:
    """High-level: ensure Ollama → start Interop → launch agent → wait → cleanup.

    This is the pipeline that makes local models work in coding agents:
    Ollama handles model inference, Interop handles the protocol mismatch.
    """
    gateway = ManagedGateway(model=model, ollama_url=ollama_url)

    print(f"Interop — local LLM compatibility layer")
    print(f"  Model:   {model}")
    print(f"  Backend: Ollama ({ollama_url})")
    print(f"  Agent:   {agent}")
    print()

    # Step 1: Make sure Ollama is running and has the model
    gateway.ensure_ollama()

    # Step 2: Start Interop gateway (translates between agent ↔ Ollama)
    gateway_url = gateway.start_gateway()
    print(f"  Gateway: {gateway_url}  (ANTHROPIC_BASE_URL)")
    print(f"  (Interop translates Anthropic Messages ↔ Ollama format)")
    print()

    # Step 3: Launch the coding agent pointed at Interop
    gateway.launch_agent(agent, extra_args)

    # Step 4: Wait for it to finish
    print(f"  {agent} started. Press Ctrl+C to stop.\n")
    try:
        return gateway.wait()
    finally:
        gateway.cleanup()