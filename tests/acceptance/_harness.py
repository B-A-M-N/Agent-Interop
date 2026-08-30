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

NOTE: this harness has been executed for real against the installed
`claude` binary (v2.1.220) in this development sandbox — see
`acceptance/results/claude-code-2.1.220.json` and
`test_real_client_claude.py`. It has not been run against a real `codex`
binary yet; whoever first runs `test_real_client_codex.py` for real should
expect to adjust its exact subprocess invocation (CLI flags/output
parsing) to match that binary's actual interface, the same way
`test_real_client_claude.py` needed `launch_spec.command` wired through
instead of a hand-built argv once this harness was first run for real.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent_interop.build_info import get_build_info
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
    can send a request, receive a tool call, execute it, and continue.

    Review finding 36: the original implementation always returned the
    canned ``final_text`` on the second call without inspecting the
    client's tool-result request, so a broken client could "pass" by
    skipping tool execution entirely.  The nonce-gate on the second
    request proves the client actually sent the tool result before
    being given the follow-up answer.

    ``expect_result_for_call_id`` is the call-id the test expects to see
    on the tool-result message in the second request.  When empty no
    call-id check is performed (backward-compatible).

    ``nonce`` is a secret string that must appear in the tool-result
    body.  When empty no nonce check is performed (backward-compatible).
    """

    def __init__(
        self,
        tool_name: str,
        tool_arguments: dict[str, Any],
        final_text: str,
        expect_result_for_call_id: str = "",
        nonce: str = "",
    ) -> None:
        self._tool_name = tool_name
        self._tool_arguments = tool_arguments
        self._final_text = final_text
        self._expect_result_for_call_id = expect_result_for_call_id
        self._nonce = nonce
        self.calls: list[PreparedUpstreamRequest] = []
        # A synthetic call-id for the tool-call emitted in the first response.
        # This lets the test verify that the client echoes the same id back
        # in its tool-result message on the second request.
        self._tool_call_id: str = ""

    async def close(self) -> None:
        pass

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        self.calls.append(request)
        if len(self.calls) == 1:
            # Emit a unique call-id so we can verify the client echoes it
            # back in the tool-result on the next request (finding 36).
            from agent_interop.abi import new_tool_call_id

            self._tool_call_id = new_tool_call_id()
            body = {
                "model": "acceptance-test-model",
                "message": {
                    "role": "assistant",
                    "content": (
                        f'<tool_call>{{"id":"{self._tool_call_id}",'
                        f'"name":"{self._tool_name}",'
                        f'"arguments":{json.dumps(self._tool_arguments)}}}</tool_call>'
                    ),
                },
                "done": True,
                "done_reason": "stop",
            }
        else:
            # Nonce-gated second request (review finding 36).
            # The client MUST have included a tool-result message before
            # being given the final canned text.
            self._verify_nonce_gate(request)
            body = {
                "model": "acceptance-test-model",
                "message": {"role": "assistant", "content": self._final_text},
                "done": True,
                "done_reason": "stop",
            }
        return UpstreamResponse(status_code=200, headers={}, body=json.dumps(body).encode())

    def _verify_nonce_gate(self, request: PreparedUpstreamRequest) -> None:
        """Prove the model received the tool result before getting the
        final answer (review finding 36).

        Inspects the request body to verify:

        1. A message with role "tool" is present whose content carries the
           nonce (when ``_nonce`` is set), proving the model read the tool
           result before generating the next response.
        2. When ``_expect_result_for_call_id`` is set, the tool result
           references the exact call-id the fake emitted in step 1,
           preventing a silent pass when the client skips tool execution.

        Raises AssertionError on any missing requirement so the test
        fails loudly rather than hiding a broken client.
        """
        body = request.body  # dict[str, Any] — Ollama Chat render output
        messages: list[dict[str, Any]] = body.get("messages", [])

        # Collect all tool-result messages from the request.
        tool_messages: list[dict[str, Any]] = [
            msg for msg in messages if isinstance(msg, dict) and msg.get("role") == "tool"
        ]

        # 1. Verify nonce is present in at least one tool message content.
        if self._nonce:
            nonce_found = any(
                self._nonce in (msg.get("content", "") or "")
                for msg in tool_messages
            )
            if not nonce_found:
                raise AssertionError(
                    "Nonce-gate FAILED: the nonce is absent from all tool-result "
                    f"messages.  Nonce={self._nonce!r} was expected but the tool "
                    "result was never sent (client may have skipped tool execution)."
                )

        # 2. Verify the tool call-id matches when expect_result_for_call_id
        #    is configured.  The second request may carry the synthetic
        #    id the fake generated in step 1, proving the client used the
        #    correct tool-call correlation id instead of some fabricated one.
        if self._expect_result_for_call_id:
            call_id_found = any(
                msg.get("tool_call_id") == self._expect_result_for_call_id
                for msg in tool_messages
            )
            if not call_id_found:
                raise AssertionError(
                    "Call-id gate FAILED: the tool-result message does not contain "
                    f"call_id={self._expect_result_for_call_id!r}.  This usually means "
                    "the client either skipped tool execution or used a wrong call-id."
                )


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
    argv: list[str] | None = None,
    configuration_strategy: str = "",
    protocol: str = "",
    compatibility_path: str = "not_observed",
    controller_used: bool = False,
    model_digest: str = "",
    verification: dict[str, bool] | None = None,
    real_backend: bool = False,
    nonce_gated_recovery: bool = False,
) -> Path:
    """Write acceptance/results/<client>-<version>.json.

    This is the ONLY evidence scripts/check_support_claims.sh accepts as
    grounds for a "release-tested" claim in README.md — see that script
    and RELEASE.md's "Alpha vs. supported release track" section.

    ``real_backend`` records whether this run exercised an actual local
    model backend (Ollama/vLLM/llama.cpp) rather than a scripted fake
    transport. ``nonce_gated_recovery`` records whether the run proved the
    per-request execution-nonce gate actually rejects nonce-less recovery.
    Both feed P0.45/P0.47: a "release-tested" claim must be backed by
    evidence that drove a real backend and validated the nonce gate.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "client": client,
        "client_version": client_version,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "scenario": scenario,
        "passed": passed,
        "detail": detail,
        # Required acceptance provenance. A false check is intentionally more
        # useful than a missing field: it prevents a single read-only smoke
        # test from being presented as an edit/recovery certification.
        "argv": list(argv or ()),
        "configuration_strategy": configuration_strategy,
        "protocol": protocol,
        "build": asdict(get_build_info()),
        "compatibility_path": compatibility_path,
        "controller_used": controller_used,
        "model_digest": model_digest,
        "real_backend": bool(real_backend),
        "verification": {
            "read_test": bool((verification or {}).get("read_test", False)),
            "edit_test": bool((verification or {}).get("edit_test", False)),
            "tool_error_recovery": bool((verification or {}).get("tool_error_recovery", False)),
            "multi_turn_continuation": bool((verification or {}).get("multi_turn_continuation", False)),
            "cleanup_verification": bool((verification or {}).get("cleanup_verification", False)),
            "nonce_gated_recovery": bool(nonce_gated_recovery),
        },
        # Richer per-task detail (review 35/36): the flat booleans above feed
        # the claim checker; the classified layer results stay available for
        # humans interpreting a failure.
        "verification_detail": {
            k: v for k, v in (verification or {}).items()
            if isinstance(v, dict)
        },
    }
    slug = "".join(c if c.isalnum() else "-" for c in client.lower()).strip("-")
    out_path = RESULTS_DIR / f"{slug}-{client_version}.json"
    out_path.write_text(json.dumps(record, indent=2))
    return out_path
