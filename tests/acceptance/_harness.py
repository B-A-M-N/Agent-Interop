"""Shared harness for real-client acceptance tests (P0-5).

Every test module in tests/acceptance/ is opt-in and skipped by default —
see each module's own skip guard. This harness:

1. Starts Interop's real FastAPI app (server/app.py:create_app) bound to a
   real localhost TCP port via uvicorn, so the actual client BINARY (not
   an in-process ASGI transport) can connect to it exactly as it would in
   production.
2. Swaps the Gateway's upstream transport for a deterministic FakeTransport
   once the app has started, so the test never depends on a real Ollama/
   vLLM/llama.cpp backend or a real model actually being loaded.
3. Writes a result record to acceptance/results/<client>-<version>.json on
   a successful run — this is the ONLY thing that ever promotes a client
   from the alpha/unverified track to "release-tested" in README.md (see
   RELEASE.md and scripts/check_support_claims.sh, which fails the release
   gate if README claims a release-tested tier without a matching file
   here).

NOTE: this harness has never been executed in this development sandbox —
there is no real `claude`/`codex` binary or credential available here. It
is written and structurally reviewed, not proven by a real run. Whoever
first runs it against a real client binary should expect to adjust the
exact subprocess invocation (CLI flags/output parsing) for that binary's
actual interface.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_interop.transport.http import (
    PreparedUpstreamRequest,
    UpstreamResponse,
    UpstreamTransport,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "acceptance" / "results"


class ScriptedFakeTransport(UpstreamTransport):
    """Deterministic fake upstream: returns a fixed Ollama-chat-shaped
    response for the first call (a tool call), then a fixed plain-text
    response for any subsequent call (the model's follow-up after seeing
    the tool result) — enough to drive exactly one tool-call round trip,
    which is all an acceptance smoke test needs to prove the real client
    can send a request, receive a tool call, execute it, and continue."""

    def __init__(self, tool_name: str, tool_arguments: dict[str, Any], final_text: str) -> None:
        self._tool_name = tool_name
        self._tool_arguments = tool_arguments
        self._final_text = final_text
        self.calls: list[PreparedUpstreamRequest] = []

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        self.calls.append(request)
        if len(self.calls) == 1:
            body = {
                "model": "acceptance-test-model",
                "message": {
                    "role": "assistant",
                    "content": (
                        f'<tool_call>{{"name":"{self._tool_name}",'
                        f'"arguments":{json.dumps(self._tool_arguments)}}}</tool_call>'
                    ),
                },
                "done": True,
                "done_reason": "stop",
            }
        else:
            body = {
                "model": "acceptance-test-model",
                "message": {"role": "assistant", "content": self._final_text},
                "done": True,
                "done_reason": "stop",
            }
        return UpstreamResponse(status_code=200, headers={}, body=json.dumps(body).encode())


@dataclass
class AcceptanceRunHandle:
    """A running Interop server plus the fake transport wired into it."""

    base_url: str
    transport: ScriptedFakeTransport
    _server: Any
    _thread: threading.Thread

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


def start_acceptance_server(config: Any, transport: ScriptedFakeTransport) -> AcceptanceRunHandle:
    """Start create_app(config) on a real TCP port in a background thread,
    with ``transport`` wired into the Gateway once it's up.

    Runs uvicorn in a dedicated thread (its own event loop) rather than
    in-process ASGI, because a real client subprocess needs an actual
    socket to connect to — httpx.ASGITransport (used by every in-process
    HTTP test elsewhere in this repo) never opens one.
    """
    import uvicorn

    from agent_interop.server.app import create_app

    app = create_app(config=config)
    uv_config = uvicorn.Config(app, host="127.0.0.1", port=config.port, log_level="warning")
    server = uvicorn.Server(uv_config)

    thread = threading.Thread(target=lambda: asyncio.run(server.serve()), daemon=True)
    thread.start()

    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("acceptance server did not start within 10s")

    # Gateway is constructed inside create_app's lifespan, which uvicorn
    # runs before accepting connections — by the time `started` is True,
    # app.state.gateway exists.
    gw = app.state.gateway
    gw._transport = transport

    return AcceptanceRunHandle(
        base_url=f"http://127.0.0.1:{config.port}",
        transport=transport,
        _server=server,
        _thread=thread,
    )


def write_acceptance_result(
    client: str,
    client_version: str,
    *,
    passed: bool,
    scenario: str,
    detail: str = "",
) -> Path:
    """Write acceptance/results/<client>-<version>.json.

    This is the ONLY evidence scripts/check_support_claims.sh accepts as
    grounds for a "release-tested" claim in README.md — see that script
    and RELEASE.md's "Alpha vs. supported release track" section.
    """
    import datetime

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "client": client,
        "client_version": client_version,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "scenario": scenario,
        "passed": passed,
        "detail": detail,
    }
    slug = "".join(c if c.isalnum() else "-" for c in client.lower()).strip("-")
    out_path = RESULTS_DIR / f"{slug}-{client_version}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return out_path
