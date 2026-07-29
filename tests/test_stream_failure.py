"""Tests for stream failure encoding across all three client protocols.

Each protocol's stream encoder must:

* emit a protocol-visible error frame when given a canonical ``error`` event,
* suppress the ordinary success-style terminal sequence,
* emit exactly one terminal sequence that reflects the failure,
* preserve issue paths and correction details from ``CanonicalError.details``.

These tests cover the encoder behavior in isolation; integration with the
gateway is exercised by the streaming integration tests.
"""

from __future__ import annotations

import json

import pytest

from agent_interop.abi import (
    CanonicalError,
    CanonicalEvent,
    CanonicalStopReason,
    CanonicalToolCorrection,
)
from agent_interop.protocols.anthropic_messages import AnthropicMessagesAdapter
from agent_interop.protocols.openai_chat import OpenAIChatAdapter
from agent_interop.protocols.openai_responses import OpenAIResponsesAdapter


def _parse_sse(text: str) -> list[dict[str, str | None]]:
    """Parse an SSE string into a list of (event, data) dicts."""
    out: list[dict[str, str | None]] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.rstrip("\r")
        if line == "":
            if event_name is not None or data_lines:
                out.append({"event": event_name, "data": "\n".join(data_lines)})
            event_name = None
            data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].lstrip())
        elif line.startswith(":"):
            # comment line
            continue
    return out


def _parse_data_lines(text: str) -> list[dict]:
    """Parse OpenAI Chat 'data:' frames into JSON dicts (skipping [DONE])."""
    out: list[dict] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


class TestOpenAIChatStreamError:
    def test_error_frame_visible_before_terminal(self):
        adapter = OpenAIChatAdapter()
        enc = adapter.create_stream_encoder(
            response_context={"response_id": "chatcmpl-test", "model": "qwen3"}
        )
        err = CanonicalError(
            code="TOOL_CALL_INVALID",
            message="bad args",
            request_id="req-1",
            details={
                "issue_paths": ["$.path"],
                "correction": "rename path to file_path",
            },
        )
        out_err = enc.encode(CanonicalEvent(type="error", error=err))
        out_stop = enc.encode(CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT))
        out_finish = enc.finish() or ""
        combined = (out_err or "") + (out_stop or "") + out_finish

        frames = _parse_data_lines(combined)
        assert any("error" in f for f in frames), f"expected error frame, got {frames!r}"

        # Terminal frame must NOT use plain 'stop' as finish_reason.
        terminal_frames = [f for f in frames if "choices" in f and f["choices"][0].get("finish_reason") is not None]
        assert terminal_frames, "expected a terminal frame"
        finish = terminal_frames[-1]["choices"][0]["finish_reason"]
        assert finish != "stop", f"failure must not produce stop finish_reason; got {finish!r}"

        # The error frame must include code + details.
        err_frames = [f for f in frames if "error" in f]
        err_payload = err_frames[0]["error"]
        assert err_payload["code"] == "TOOL_CALL_INVALID"
        assert err_payload["details"]["issue_paths"] == ["$.path"]

    def test_text_delta_suppressed_after_error(self):
        adapter = OpenAIChatAdapter()
        enc = adapter.create_stream_encoder()
        enc.encode(CanonicalEvent(
            type="error",
            error=CanonicalError(code="STREAM_ERROR", message="boom"),
        ))
        # Subsequent text_delta must not reach the wire.
        out = enc.encode(CanonicalEvent(type="text_delta", partial="recovered text"))
        assert out is None

    def test_terminal_then_done_emits_exactly_one_terminal(self):
        adapter = OpenAIChatAdapter()
        enc = adapter.create_stream_encoder()
        enc.encode(CanonicalEvent(
            type="error",
            error=CanonicalError(code="STREAM_ERROR", message="boom"),
        ))
        enc.encode(CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT))
        enc.finish()
        # Calling finish again should be a no-op (idempotent terminal).
        out = enc.finish() or ""
        assert out == ""


class TestAnthropicStreamError:
    def test_error_event_then_failure_terminal(self):
        adapter = AnthropicMessagesAdapter()
        enc = adapter.create_stream_encoder(
            response_context={"response_id": "msg_test", "model": "qwen3"}
        )
        err = CanonicalError(
            code="INVALID_OUTPUT",
            message="malformed tool call",
            details={"issue_paths": ["$.path"]},
        )
        err_out = enc.encode(CanonicalEvent(type="error", error=err))
        assert err_out is not None
        err_frame = _parse_sse(err_out)
        assert err_frame[0]["event"] == "error"
        data_str = err_frame[0]["data"]
        assert data_str is not None
        parsed_err = json.loads(data_str)
        assert parsed_err["type"] == "error"
        assert parsed_err["error"]["type"] == "api_error"
        assert parsed_err["error"]["message"] == "malformed tool call"

        # Terminal must mark stop_reason=invalid_output
        msg_start = enc.encode(CanonicalEvent(type="message_start"))
        # The encoder should ignore further text deltas.
        enc.encode(CanonicalEvent(type="text_delta", partial="hello"))
        stop_out = enc.encode(CanonicalEvent(
            type="message_stop",
            stop_reason=CanonicalStopReason.END_TURN,
        ))
        finish_out = enc.finish() or ""
        all_out = (msg_start or "") + (stop_out or "") + finish_out
        frames = _parse_sse(all_out)
        deltas = [json.loads(f["data"] or "{}") for f in frames if f.get("event") == "message_delta"]
        assert deltas, "expected a message_delta"
        assert deltas[0]["delta"]["stop_reason"] == "invalid_output"

    def test_no_phantom_text_after_error(self):
        adapter = AnthropicMessagesAdapter()
        enc = adapter.create_stream_encoder()
        enc.encode(CanonicalEvent(
            type="error",
            error=CanonicalError(code="STREAM_ERROR", message="boom"),
        ))
        out = enc.encode(CanonicalEvent(type="text_delta", partial="text"))
        assert out is None


class TestOpenAIResponsesStreamError:
    def test_error_event_emits_response_failed(self):
        adapter = OpenAIResponsesAdapter()
        enc = adapter.create_stream_encoder(
            response_context={"response_id": "resp_test", "model": "qwen3"}
        )
        err = CanonicalError(
            code="INVALID_OUTPUT",
            message="bad tool",
            details={"issue_paths": ["$.a"]},
        )
        err_out = enc.encode(CanonicalEvent(type="error", error=err))
        stop_out = enc.encode(CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT))
        finish_out = enc.finish() or ""
        all_out = (err_out or "") + (stop_out or "") + finish_out
        frames = _parse_sse(all_out)
        events = {f.get("event") for f in frames if f.get("event") is not None}
        # Must NOT emit a normal response.completed after a failure.
        assert "response.failed" in events
        assert "response.completed" not in events, f"completed leaked after failure: {events}"
        # [DONE] must NOT be emitted after response.failed.
        for f in frames:
            data = f.get("data") or ""
            if isinstance(data, str) and "DONE" in data:
                pytest.fail("DONE sentinel must not be emitted after response.failed")

    def test_error_then_text_suppressed(self):
        adapter = OpenAIResponsesAdapter()
        enc = adapter.create_stream_encoder()
        enc.encode(CanonicalEvent(
            type="error",
            error=CanonicalError(code="STREAM_ERROR", message="boom"),
        ))
        out = enc.encode(CanonicalEvent(type="text_delta", partial="hello"))
        assert out is None


class TestCanonicalToolCorrectionAsErrorDetail:
    def test_correction_serializes_via_details(self):
        """CanonicalToolCorrection must round-trip through CanonicalError.details."""
        correction = CanonicalToolCorrection(
            request_id="req-1",
            candidate_id="cand-1",
            raw_tool_name="write_file",
            canonical_tool_name="write",
            issue_paths=["$.path", "$.content"],
            repair_steps_attempted=["field_alias:path->file_path", "required_field:content"],
            correction_instruction="Use canonical tool name 'write' and required 'content' field.",
            retryable=True,
        )
        err = CanonicalError(
            code="TOOL_CALL_INVALID",
            message="rejected",
            details={"correction": correction.__dict__},
        )
        adapter = OpenAIChatAdapter()
        enc = adapter.create_stream_encoder()
        out_err = enc.encode(CanonicalEvent(type="error", error=err))
        out_stop = enc.encode(CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT))
        out = (out_err or "") + (out_stop or "") + (enc.finish() or "")
        frames = _parse_data_lines(out)
        err_frame = next(f for f in frames if "error" in f)
        details = err_frame["error"]["details"]
        assert details["correction"]["raw_tool_name"] == "write_file"
        assert details["correction"]["canonical_tool_name"] == "write"
        assert details["correction"]["issue_paths"] == ["$.path", "$.content"]
