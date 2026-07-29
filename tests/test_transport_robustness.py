"""Tests for transport robustness (Phase 4) and error classification."""

from __future__ import annotations

import pytest

from agent_interop.errors import (
    InteropError,
    InteropErrorCode,
    classify_http_status,
    err_backend_timeout,
    err_backend_unavailable,
    err_route_not_found,
    get_error_descriptor,
    parse_retry_after,
    serialize_client_error,
)
from agent_interop.transport.sse import SSEDecoder, SSEFrameTooLargeError

# ─── Error classification (item 77) ────────────────────────────────────────


class TestErrorClassification:
    def test_classify_400(self):
        assert classify_http_status(400) == InteropErrorCode.INVALID_REQUEST

    def test_classify_401(self):
        assert classify_http_status(401) == InteropErrorCode.BACKEND_AUTH_FAILED

    def test_classify_404(self):
        assert classify_http_status(404) == InteropErrorCode.MODEL_NOT_FOUND

    def test_classify_429(self):
        assert classify_http_status(429) == InteropErrorCode.BACKEND_RATE_LIMITED

    def test_classify_500(self):
        assert classify_http_status(500) == InteropErrorCode.BACKEND_ERROR

    def test_classify_503(self):
        assert classify_http_status(503) == InteropErrorCode.BACKEND_UNAVAILABLE

    def test_classify_504(self):
        assert classify_http_status(504) == InteropErrorCode.BACKEND_TIMEOUT

    def test_classify_unknown_5xx(self):
        assert classify_http_status(529) == InteropErrorCode.BACKEND_ERROR

    def test_parse_retry_after_seconds(self):
        assert parse_retry_after({"Retry-After": "5"}) == 5.0

    def test_parse_retry_after_missing(self):
        assert parse_retry_after({}) == 0.0

    def test_parse_retry_after_http_date(self):
        # Falls back to 1.0 for HTTP-date format
        assert parse_retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 1.0

    def test_get_error_descriptor_known(self):
        desc = get_error_descriptor(InteropErrorCode.BACKEND_UNAVAILABLE)
        assert desc.http_status == 503
        assert desc.retryable is True

    def test_get_error_descriptor_unknown_fallback(self):
        desc = get_error_descriptor("UNKNOWN_CODE")
        assert desc.code == InteropErrorCode.INTERNAL_ERROR


class TestInteropError:
    def test_err_backend_unavailable(self):
        err = err_backend_unavailable("http://localhost:11434", "refused")
        assert err.code == InteropErrorCode.BACKEND_UNAVAILABLE
        assert "not reachable" in err.message
        assert err.details["backend_url"] == "http://localhost:11434"

    def test_err_backend_timeout(self):
        err = err_backend_timeout("http://localhost:11434", 30.0)
        assert err.code == InteropErrorCode.BACKEND_TIMEOUT
        assert "30.0s" in err.message

    def test_err_route_not_found(self):
        err = err_route_not_found("my-model")
        assert err.code == InteropErrorCode.ROUTE_NOT_FOUND
        assert "my-model" in err.message

    def test_to_canonical_error(self):
        err = InteropError(code=InteropErrorCode.BACKEND_UNAVAILABLE, message="test")
        canonical = err.to_canonical_error()
        assert canonical.code == InteropErrorCode.BACKEND_UNAVAILABLE
        assert canonical.retryable is True

    def test_serialize_anthropic(self):
        from agent_interop.abi import CanonicalError
        err = CanonicalError(code=InteropErrorCode.BACKEND_UNAVAILABLE, message="down")
        encoded = serialize_client_error(err, "anthropic")
        assert encoded.body["error"]["type"] == "overloaded_error"
        assert encoded.status_code == 503

    def test_serialize_openai_chat(self):
        from agent_interop.abi import CanonicalError
        err = CanonicalError(code=InteropErrorCode.BACKEND_RATE_LIMITED, message="slow")
        encoded = serialize_client_error(err, "openai_chat")
        assert encoded.body["error"]["type"] == "rate_limit_error"


# ─── MVP-08: error.message must be redacted, not just .details ────────────


class TestRedactSecrets:
    def test_authorization_bearer_header_redacted(self):
        from agent_interop.errors import redact_secrets

        text = 'Upstream returned 401: {"error": "unauthorized"} Authorization: Bearer sk-abc123XYZ789'
        redacted = redact_secrets(text)
        assert "sk-abc123XYZ789" not in redacted
        assert "[REDACTED]" in redacted

    def test_bare_bearer_token_redacted(self):
        from agent_interop.errors import redact_secrets

        redacted = redact_secrets("failed with Bearer abcdef123456789token")
        assert "abcdef123456789token" not in redacted

    def test_api_key_field_redacted(self):
        from agent_interop.errors import redact_secrets

        redacted = redact_secrets('upstream body: {"api_key": "sk-live-987654321"}')
        assert "sk-live-987654321" not in redacted

    def test_credentialed_url_redacted(self):
        from agent_interop.errors import redact_secrets

        redacted = redact_secrets("connection to https://user:hunter2@internal.example.com/v1 failed")
        assert "hunter2" not in redacted
        assert "internal.example.com" in redacted  # host itself is not secret

    def test_ordinary_message_passes_through_unchanged(self):
        from agent_interop.errors import redact_secrets

        assert redact_secrets("connection refused") == "connection refused"

    def test_serialize_client_error_redacts_message(self):
        from agent_interop.abi import CanonicalError
        from agent_interop.errors import InteropErrorCode

        err = CanonicalError(
            code=InteropErrorCode.BACKEND_ERROR,
            message="Upstream returned 401: Authorization: Bearer sk-super-secret-token-value",
        )
        for protocol in ("anthropic", "openai_chat", "openai_responses"):
            encoded = serialize_client_error(err, protocol)
            body_str = str(encoded.body)
            assert "sk-super-secret-token-value" not in body_str

    def test_serialize_client_error_still_sanitizes_details(self):
        from agent_interop.abi import CanonicalError
        from agent_interop.errors import InteropErrorCode

        err = CanonicalError(
            code=InteropErrorCode.BACKEND_ERROR,
            message="failed",
            details={"api_key": "should-not-leak", "hint": "retry later"},
        )
        encoded = serialize_client_error(err, "openai_chat")
        assert "should-not-leak" not in str(encoded.body)
        assert encoded.body.get("details", {}).get("hint") == "retry later"


# ─── SSE byte limits (item 82) ─────────────────────────────────────────────


class TestSSEByteLimits:
    def test_normal_frame(self):
        dec = SSEDecoder(max_data_bytes=100)
        dec.feed("data: hello\n")
        frame = dec.feed("\n")
        assert frame is not None
        assert frame.data == "hello"

    def test_data_limit_enforced(self):
        dec = SSEDecoder(max_data_bytes=10)
        with pytest.raises(SSEFrameTooLargeError):
            dec.feed("data: this is way too long\n")

    def test_buffer_limit_enforced(self):
        dec = SSEDecoder(max_buffer_bytes=20)
        with pytest.raises(SSEFrameTooLargeError):
            dec.feed("data: abcdefghijklmnopqrstuvwxyz\n")

    def test_large_frame_emission(self):
        dec = SSEDecoder(max_data_bytes=1_048_576)
        # Feed a large but acceptable frame
        dec.feed("data: " + "x" * 1000 + "\n")
        frame = dec.feed("\n")
        assert frame is not None
        assert len(frame.data) == 1000


# ─── Unterminated-line buffer cap (raw_lines/sse_events read raw byte
#     chunks and bound their own buffer, instead of httpx's unbounded
#     aiter_lines()) ──────────────────────────────────────────────────────


class _UnboundedChunkResponse:
    """Simulates an upstream that never sends a newline: every chunk keeps
    extending a single line indefinitely. httpx's own aiter_lines() has no
    size cap of its own, so before the fix this would buffer without
    bound; UpstreamStream must now raise once its own cap is crossed,
    without ever needing to consume the full (here: deliberately huge)
    stream the fake is willing to produce."""

    def __init__(self, chunk: bytes, max_chunks: int) -> None:
        self._chunk = chunk
        self._max_chunks = max_chunks
        self.chunks_consumed = 0

    async def aiter_bytes(self):
        for _ in range(self._max_chunks):
            self.chunks_consumed += 1
            yield self._chunk


class TestUnterminatedLineBufferCap:
    @pytest.mark.asyncio
    async def test_sse_events_raises_on_unterminated_growth_without_consuming_everything(self):
        from agent_interop.transport.http import UpstreamStream

        chunk = b"x" * 1024  # no newline, ever
        # Fake is willing to produce far more than the cap — if the fix
        # didn't work, we'd either buffer all of it (memory blowup) or
        # hang; either way chunks_consumed would reach max_chunks.
        response = _UnboundedChunkResponse(chunk, max_chunks=100_000)
        stream = UpstreamStream(response=response, max_sse_buffer_bytes=4096)  # type: ignore[arg-type]

        with pytest.raises(SSEFrameTooLargeError):
            async for _ in stream.sse_events():
                pass

        # Must have raised long before exhausting the fake's supply —
        # proves the cap is enforced incrementally, not after full buffering.
        assert response.chunks_consumed < 100

    @pytest.mark.asyncio
    async def test_raw_lines_raises_on_unterminated_growth(self):
        from agent_interop.transport.http import UpstreamStream

        chunk = b"x" * 1024
        response = _UnboundedChunkResponse(chunk, max_chunks=100_000)
        stream = UpstreamStream(response=response, max_sse_buffer_bytes=4096)  # type: ignore[arg-type]

        with pytest.raises(SSEFrameTooLargeError):
            async for _ in stream.raw_lines():
                pass

        assert response.chunks_consumed < 100

    @pytest.mark.asyncio
    async def test_large_chunk_of_many_small_terminated_lines_not_falsely_rejected(self):
        """A single large chunk containing many well-terminated small
        lines must NOT trip the unterminated-line cap — only genuinely
        unterminated growth should count. Uses raw_lines() rather than
        sse_events() to isolate this from SSEDecoder's own (unrelated,
        pre-existing) per-record buffer limit."""
        from agent_interop.transport.http import UpstreamStream

        # One chunk, 8000 bytes total, but made of 1000 well-terminated
        # 8-byte lines — must not be rejected even though the whole chunk
        # exceeds a small cap, since nothing is ever left unterminated.
        big_chunk = (b"line 1\n") * 1000

        class _OneChunkResponse:
            async def aiter_bytes(self):
                yield big_chunk

        stream = UpstreamStream(response=_OneChunkResponse(), max_sse_buffer_bytes=4096)  # type: ignore[arg-type]
        lines = [line async for line in stream.raw_lines()]
        assert len(lines) == 1000
        assert all(line == "line 1" for line in lines)


# ─── MVP-06: SSE tail flush must only happen on normal EOF ────────────────


class _FakeHttpxResponse:
    """Minimal stand-in for httpx.Response exposing aiter_bytes(), which
    is what UpstreamStream reads from (see _bounded_lines — reading raw
    byte chunks and splitting on our own bounded buffer, rather than
    httpx's unbounded aiter_lines(), is the actual fix under test in the
    sibling module)."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_bytes(self):
        for line in self._lines:
            yield (line + "\n").encode("utf-8")


class TestSSEEventsTailFlush:
    @pytest.mark.asyncio
    async def test_trailing_data_without_blank_line_is_flushed_on_normal_eof(self):
        """A final `data:` record with no trailing blank line must still be
        surfaced — the decoder's flush-on-EOF behavior."""
        from agent_interop.transport.http import UpstreamStream

        # No blank line after the last "data: hello" — normal EOF must still
        # flush it via the decoder.
        response = _FakeHttpxResponse(["data: hello"])
        stream = UpstreamStream(response=response)  # type: ignore[arg-type]

        frames = [frame async for frame in stream.sse_events()]
        assert len(frames) == 1
        assert frames[0].data == "hello"

    @pytest.mark.asyncio
    async def test_aclose_mid_stream_does_not_raise(self):
        """Closing the generator mid-stream (GeneratorExit) must not raise —
        yielding from `finally` on abnormal exit produces a RuntimeError
        ("async generator ignored GeneratorExit") under the old code."""
        from agent_interop.transport.http import UpstreamStream

        response = _FakeHttpxResponse(["data: first", "", "data: second"])
        stream = UpstreamStream(response=response)  # type: ignore[arg-type]

        gen = stream.sse_events()
        first = await gen.__anext__()
        assert first.data == "first"
        # Close before the generator reaches normal EOF.
        await gen.aclose()


# ─── MVP-05: non-streaming responses must be bounded before buffering ──────


class TestSendResponseSizeBound:
    @pytest.mark.asyncio
    async def test_oversized_response_raises_before_full_body_returned(self):
        import httpx

        from agent_interop.transport.http import (
            PreparedUpstreamRequest,
            UpstreamResponseTooLargeError,
            UpstreamTransport,
        )

        oversized_body = b"x" * 10_000

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=oversized_body)

        transport = UpstreamTransport(max_retries=0, max_response_bytes=1024)
        transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        with pytest.raises(UpstreamResponseTooLargeError):
            await transport.send(PreparedUpstreamRequest(method="POST", url="http://fake/x"))

    @pytest.mark.asyncio
    async def test_response_within_limit_succeeds(self):
        import httpx

        from agent_interop.transport.http import PreparedUpstreamRequest, UpstreamTransport

        small_body = b'{"ok": true}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=small_body)

        transport = UpstreamTransport(max_retries=0, max_response_bytes=1024)
        transport._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        resp = await transport.send(PreparedUpstreamRequest(method="POST", url="http://fake/x"))
        assert resp.status_code == 200
        assert resp.body == small_body


# ─── Transport response error_code property (item 77) ────────────────────────


class TestUpstreamResponse:
    def test_error_code_classification(self):
        from agent_interop.transport.http import UpstreamResponse
        resp = UpstreamResponse(status_code=503, body=b"")
        assert resp.error_code == InteropErrorCode.BACKEND_UNAVAILABLE

    def test_error_code_success(self):
        from agent_interop.transport.http import UpstreamResponse
        resp = UpstreamResponse(status_code=200, body=b"")
        assert resp.is_error() is False
