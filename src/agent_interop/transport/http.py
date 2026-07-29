"""HTTP transport — shared httpx client with proper streaming lifecycle.

Key improvements over the old ``http_client.post()`` + ``_stream_events()``:
- Uses ``async with client.stream()`` for true streaming proxy (no buffering).
- Connection pooling, timeout handling, upstream cancellation.
- ``send()`` for non-streaming, ``stream()`` for streaming.
- Connection/concurrency limits via httpx.Limits (item 80).
- Bounded retries with exponential backoff (item 79).
- SSE/NDJSON byte limits enforced at the stream level (item 82).
- Centralized HTTP error classification (item 77).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from agent_interop.errors import classify_http_status, parse_retry_after
from agent_interop.transport.ndjson import NDJSONDecoder
from agent_interop.transport.sse import SSEDecoder, SSEFrame, SSEFrameTooLargeError

logger = logging.getLogger("agent_interop.transport")

# ─── Defaults ───────────────────────────────────────────────────────────────

DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE = 20
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BASE_DELAY = 0.5
DEFAULT_RETRY_MAX_DELAY = 8.0
DEFAULT_MAX_SSE_DATA_BYTES = 1_048_576  # 1 MiB per frame
DEFAULT_MAX_SSE_BUFFER_BYTES = 4_194_304  # 4 MiB total buffer
DEFAULT_MAX_NDJSON_FRAME_BYTES = 1_048_576  # 1 MiB per line
DEFAULT_MAX_TOTAL_STREAM_BYTES = 256 * 1024 * 1024  # 256 MiB per stream
DEFAULT_MAX_RESPONSE_BYTES = 256 * 1024 * 1024  # 256 MiB per non-streaming response


class UpstreamResponseTooLargeError(Exception):
    """A non-streaming upstream response exceeded max_response_bytes.

    Raised while the body is still being read incrementally — before it is
    fully buffered — so an oversized response cannot force an unbounded
    allocation.
    """


@dataclass
class PreparedUpstreamRequest:
    """A fully prepared upstream request ready to send."""

    method: str = "POST"
    url: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    stream: bool = True
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


@dataclass
class UpstreamResponse:
    """A non-streaming upstream response."""

    status_code: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    # True only for the synthetic response send() returns after every retry
    # attempt failed at the TRANSPORT level (connect/timeout error) — the
    # backend never actually answered. Distinguishes that case from a
    # genuine HTTP status the backend returned (including a real 503),
    # which callers like route probing need to tell apart: "we couldn't
    # reach it at all" is not the same signal as "it reached us and said no".
    transport_failed: bool = False

    def json(self) -> dict[str, Any]:
        return json.loads(self.body)

    def is_error(self) -> bool:
        return self.status_code >= 400

    @property
    def error_code(self) -> str:
        """Classify this response's error status using the centralized registry."""
        return classify_http_status(self.status_code)


@dataclass
class UpstreamStream:
    """A streaming upstream response yielding frames."""

    response: httpx.Response
    max_sse_data_bytes: int = DEFAULT_MAX_SSE_DATA_BYTES
    max_sse_buffer_bytes: int = DEFAULT_MAX_SSE_BUFFER_BYTES
    max_ndjson_frame_bytes: int = DEFAULT_MAX_NDJSON_FRAME_BYTES
    max_total_stream_bytes: int = DEFAULT_MAX_TOTAL_STREAM_BYTES
    _total_bytes: int = 0
    _closed: bool = False

    async def _bounded_lines(self, max_unterminated_bytes: int) -> AsyncIterator[str]:
        """Decode text lines from the raw byte stream, capping how large
        an UNTERMINATED line's buffer may grow.

        httpx's own ``aiter_lines()`` buffers internally with no size
        limit of its own — a pathological (or malicious) upstream that
        never sends a newline would have httpx accumulate the entire line
        in memory before we ever see it, bypassing every length check
        that only runs on an already-materialized ``line`` object. Reading
        raw byte chunks and splitting on ``b"\\n"`` ourselves lets the cap
        apply to the buffer as it grows, not after the fact.

        Splitting on the raw newline byte before UTF-8 decoding is always
        safe: 0x0A never appears as a continuation byte of a multi-byte
        UTF-8 sequence, so a decoded codepoint can never be split across
        the boundary we cut on.
        """
        buf = bytearray()
        async for chunk in self.response.aiter_bytes():
            if not chunk:
                continue
            buf.extend(chunk)
            while True:
                nl = buf.find(b"\n")
                if nl == -1:
                    break
                raw_line = bytes(buf[:nl])
                del buf[: nl + 1]
                if raw_line.endswith(b"\r"):
                    raw_line = raw_line[:-1]
                yield raw_line.decode("utf-8", errors="replace")
            # Only the genuinely-unterminated residual counts toward the
            # cap — checking before draining complete lines would falsely
            # reject a single large chunk containing many well-terminated
            # lines.
            if len(buf) > max_unterminated_bytes:
                raise SSEFrameTooLargeError(
                    f"Unterminated stream line exceeds {max_unterminated_bytes} bytes"
                )
        if buf:
            tail = bytes(buf)
            if tail.endswith(b"\r"):
                tail = tail[:-1]
            yield tail.decode("utf-8", errors="replace")

    async def sse_events(self) -> AsyncIterator[SSEFrame]:
        """Iterate over SSE events from the stream.

        The underlying ``SSEDecoder`` is flushed at end-of-stream so
        that a final record without a trailing blank line is still
        surfaced to callers.

        Enforces SSE frame byte limits (item 82). Yields SSEFrame
        objects; raises SSEFrameTooLargeError if limits are exceeded.
        """
        decoder = SSEDecoder(
            max_data_bytes=self.max_sse_data_bytes,
            max_buffer_bytes=self.max_sse_buffer_bytes,
        )
        # Yielding from a `finally` block is unsafe: if this generator is
        # closed via GeneratorExit (cancellation, or the consumer breaking
        # out of an `async for` and letting the generator go out of scope),
        # Python raises a RuntimeError for a `finally` that yields instead
        # of re-raising, and the yielded value would be a truncated/partial
        # tail from an abnormal exit anyway. Only flush and yield the tail
        # on a real, normal end-of-stream.
        normal_eof = False
        try:
            async for line in self._bounded_lines(self.max_sse_buffer_bytes):
                self._total_bytes += len(line.encode("utf-8")) + 1  # +1 for newline
                if self._total_bytes > self.max_total_stream_bytes:
                    raise SSEFrameTooLargeError(
                        f"Stream exceeds {self.max_total_stream_bytes} total bytes"
                    )
                frame = decoder.feed(line + "\n")
                if frame is not None:
                    decoder.reset_buffer_counter()
                    yield frame
            normal_eof = True
        finally:
            if not normal_eof:
                logger.debug("SSE stream ended abnormally; discarding decoder tail")

        if normal_eof:
            try:
                tail = decoder.flush()
            except Exception:
                tail = None
            if tail is not None:
                yield tail

    async def ndjson_events(self) -> AsyncIterator[Any]:
        """Iterate over NDJSON events from the stream.

        Yields a mix of parsed dict frames and
        ``MalformedNDJSONLine`` markers for parse failures.  The
        decoder is flushed at end-of-stream so trailing data is
        surfaced.

        Enforces per-frame byte limits via NDJSONDecoder (item 82).
        """
        import codecs

        decoder = NDJSONDecoder(max_frame_bytes=self.max_ndjson_frame_bytes)
        utf8_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            async for chunk in self.response.aiter_bytes():
                if not chunk:
                    continue
                self._total_bytes += len(chunk)
                if self._total_bytes > self.max_total_stream_bytes:
                    raise SSEFrameTooLargeError(
                        f"Stream exceeds {self.max_total_stream_bytes} total bytes"
                    )
                text = utf8_decoder.decode(chunk, final=False)
                if not text:
                    continue
                emitted = list(decoder.feed(text))
                for item in emitted:
                    yield item
            # Flush the UTF-8 decoder for any trailing bytes
            trailing_text = utf8_decoder.decode(b"", final=True)
            if trailing_text:
                for item in decoder.feed(trailing_text):
                    yield item
            # Flush trailing buffered data in NDJSON decoder
            for item in decoder.flush():
                yield item
        finally:
            pass

    async def raw_lines(self) -> AsyncIterator[str]:
        """Iterate over raw text lines, with the same unterminated-line
        buffer cap as ``sse_events()`` (see ``_bounded_lines``)."""
        async for line in self._bounded_lines(self.max_sse_buffer_bytes):
            yield line

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self.response.aclose()

    @property
    def status_code(self) -> int:
        return self.response.status_code


class UpstreamTransport:
    """Shared HTTP transport for all upstream backends.

    Features:
    - Connection pooling with configurable limits (item 80)
    - Bounded retries with exponential backoff (item 79)
    - Configurable timeouts per request
    """

    def __init__(
        self,
        *,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
        max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
        retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        connect_timeout: float | None = None,
        write_timeout: float | None = None,
        pool_timeout: float | None = None,
        max_sse_data_bytes: int = DEFAULT_MAX_SSE_DATA_BYTES,
        max_sse_buffer_bytes: int = DEFAULT_MAX_SSE_BUFFER_BYTES,
        max_ndjson_frame_bytes: int = DEFAULT_MAX_NDJSON_FRAME_BYTES,
        max_total_stream_bytes: int = DEFAULT_MAX_TOTAL_STREAM_BYTES,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        tls_verify: bool = True,
        **client_kwargs: Any,
    ) -> None:
        self._client_kwargs = client_kwargs
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._max_retries = max_retries
        self._retryable_statuses = retryable_statuses
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._timeout_seconds = timeout_seconds
        # timeout_seconds is the READ timeout — the dimension a route-
        # specific override replaces. connect/write/pool default to it when
        # not given explicitly so a caller that only sets timeout_seconds
        # (legacy usage) still gets a single coherent timeout everywhere.
        self._connect_timeout = connect_timeout if connect_timeout is not None else timeout_seconds
        self._write_timeout = write_timeout if write_timeout is not None else timeout_seconds
        self._pool_timeout = pool_timeout if pool_timeout is not None else timeout_seconds
        self._max_sse_data_bytes = max_sse_data_bytes
        self._max_sse_buffer_bytes = max_sse_buffer_bytes
        self._max_ndjson_frame_bytes = max_ndjson_frame_bytes
        self._max_total_stream_bytes = max_total_stream_bytes
        self._max_response_bytes = max_response_bytes
        self._tls_verify = tls_verify
        self._client: httpx.AsyncClient | None = None

    def _default_timeout(self) -> httpx.Timeout:
        """Build the client-level timeout with all four dimensions wired
        independently, rather than collapsing them into one scalar."""
        return httpx.Timeout(
            connect=self._connect_timeout,
            read=self._timeout_seconds,
            write=self._write_timeout,
            pool=self._pool_timeout,
        )

    def _request_timeout(self, read_timeout_seconds: float) -> httpx.Timeout:
        """Build a per-request timeout that overrides only the READ
        dimension with the route's configured timeout — connect/write/pool
        stay at their client-level defaults rather than being replaced
        wholesale by a request-specific value that was only ever meant to
        bound how long a (possibly slow-generating) model may take."""
        return httpx.Timeout(
            connect=self._connect_timeout,
            read=read_timeout_seconds,
            write=self._write_timeout,
            pool=self._pool_timeout,
        )

    def _build_client(self) -> httpx.AsyncClient:
        """Create the httpx client with connection limits (item 80)."""
        limits = httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive,
        )
        return httpx.AsyncClient(
            timeout=self._default_timeout(),
            limits=limits,
            verify=self._tls_verify,
            **self._client_kwargs,
        )

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    # Methods with no generation side effect — safe to retry after the
    # backend has actually seen the request (a timed-out or failed GET
    # probe can simply be re-asked; a timed-out or failed POST inference
    # call may already be running on the backend, and re-sending it
    # duplicates real (often GPU-bound, locally-limited) compute for a
    # non-idempotent operation, not just wasted network round-trips).
    _RETRY_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})

    async def send(self, request: PreparedUpstreamRequest) -> UpstreamResponse:
        """Send a non-streaming request with bounded retries (item 79).

        Uses ``client.stream()`` and reads the body incrementally, bounded
        against ``max_response_bytes``, rather than ``client.request()`` +
        ``.aread()``. ``client.request()`` already buffers the complete
        response body internally before returning — by the time ``.aread()``
        ran, an oversized response had already been fully allocated.

        Retry eligibility is gated by HTTP method beyond connection-phase
        failures: a ``ConnectError``/``ConnectTimeout`` means nothing was
        ever transmitted, so retrying is always safe regardless of method.
        A read timeout or a retryable status *response*, though, means the
        backend already received the request — for POST (inference calls),
        that request may already be executing, so automatically retrying
        it would silently duplicate a non-idempotent, possibly expensive
        generation instead of just re-trying a safe, side-effect-free call.
        """
        last_exc: Exception | None = None
        retry_safe = request.method.upper() in self._RETRY_SAFE_METHODS

        for attempt in range(self._max_retries + 1):
            try:
                json_body = request.body if request.method not in ("GET", "HEAD") else None
                async with self.client.stream(
                    request.method,
                    request.url,
                    headers=request.headers,
                    json=json_body,
                    timeout=self._request_timeout(request.timeout_seconds),
                ) as resp:
                    # Retry on configured retryable statuses (e.g. 500, 503, 429).
                    # httpx does not raise on 4xx/5xx by default, so an explicit
                    # status check is required to avoid silently dropping retries.
                    # Only for retry-safe methods — see docstring above.
                    if (
                        resp.status_code in self._retryable_statuses
                        and retry_safe
                        and attempt < self._max_retries
                    ):
                        delay = self._backoff_delay(attempt, dict(resp.headers))
                        logger.warning(
                            "Upstream retryable status %d (attempt %d/%d), retrying in %.1fs",
                            resp.status_code, attempt + 1, self._max_retries + 1, delay,
                        )
                        await self._async_sleep(delay)
                        continue
                    body_buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body_buf.extend(chunk)
                        if len(body_buf) > self._max_response_bytes:
                            raise UpstreamResponseTooLargeError(
                                f"Upstream response exceeds "
                                f"{self._max_response_bytes} bytes"
                            )
                    return UpstreamResponse(
                        status_code=resp.status_code,
                        headers=dict(resp.headers),
                        body=bytes(body_buf),
                    )
            except httpx.ConnectTimeout as exc:
                # Connection-phase timeout — nothing was ever sent, so
                # retrying is safe for any method.
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Upstream connect timeout (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    await self._async_sleep(delay)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if not retry_safe:
                    # The request body was already fully sent by the time a
                    # read/write timeout fires — looping back here would
                    # resend a request that may already be executing on the
                    # backend. Stop now rather than let the `for` loop's
                    # next iteration silently resend it.
                    break
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Upstream timeout (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    await self._async_sleep(delay)
            except httpx.ConnectError as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Upstream connection error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    await self._async_sleep(delay)
            except httpx.HTTPStatusError as exc:
                # Don't retry client errors (4xx), only server errors (5xx)
                if exc.response.status_code < 500:
                    body = await exc.response.aread()
                    return UpstreamResponse(
                        status_code=exc.response.status_code,
                        headers=dict(exc.response.headers),
                        body=body,
                    )
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._backoff_delay(attempt, dict(exc.response.headers))
                    logger.warning(
                        "Upstream server error %d (attempt %d/%d), retrying in %.1fs",
                        exc.response.status_code, attempt + 1,
                        self._max_retries + 1, delay,
                    )
                    await self._async_sleep(delay)

        # All retries exhausted — return a synthetic error response
        # rather than raising, so callers can classify the failure
        # through the normal error pipeline
        error_msg = str(last_exc) if last_exc else "Unknown transport error"
        return UpstreamResponse(
            status_code=503,
            headers={},
            body=json.dumps({
                "error": f"Upstream request failed after {self._max_retries + 1} attempts: {error_msg}",
            }).encode("utf-8"),
            transport_failed=True,
        )

    @asynccontextmanager
    async def stream(self, request: PreparedUpstreamRequest) -> AsyncIterator[UpstreamStream]:
        """Open a streaming connection.

        Note: Retries for streaming are not supported because the
        body is consumed on the first attempt. If the stream fails
        mid-flight, the caller should retry at a higher level.
        """
        async with self.client.stream(
            request.method,
            request.url,
            headers=request.headers,
            json=request.body,
            timeout=self._request_timeout(request.timeout_seconds),
        ) as resp:
            yield UpstreamStream(
                response=resp,
                max_sse_data_bytes=self._max_sse_data_bytes,
                max_sse_buffer_bytes=self._max_sse_buffer_bytes,
                max_ndjson_frame_bytes=self._max_ndjson_frame_bytes,
                max_total_stream_bytes=self._max_total_stream_bytes,
            )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _backoff_delay(self, attempt: int, headers: dict[str, str] | None = None) -> float:
        """Calculate exponential backoff delay with jitter.

        Respects Retry-After header if present (item 77).
        """
        import random

        # Check for Retry-After header from upstream
        if headers:
            retry_after = parse_retry_after(headers)
            if retry_after > 0:
                return min(retry_after, self._retry_max_delay)

        # Exponential backoff: 0.5s, 1s, 2s, 4s, ... capped at retry_max_delay
        delay = self._retry_base_delay * (2 ** attempt)
        # Add jitter: ±25%
        jitter = delay * 0.25 * (2 * random.random() - 1)
        return min(delay + jitter, self._retry_max_delay)

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep that respects cancellation."""
        import asyncio
        await asyncio.sleep(seconds)
