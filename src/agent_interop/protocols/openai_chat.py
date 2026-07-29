"""OpenAI Chat Completions API adapter (/v1/chat/completions)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalError,
    CanonicalEvent,
    CanonicalGenerationOptions,
    CanonicalImageBlock,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalReasoningBlock,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
    CanonicalUnknownBlock,
    MetadataForwardingPolicy,
    ProviderMetadata,
    canonical_tool_choice,
    tool_from_openai,
)
from agent_interop.errors import EncodedErrorResponse
from agent_interop.protocols.base import ClientProtocolAdapter, StreamEncoder


class OpenAIChatAdapter(ClientProtocolAdapter):
    """Translate OpenAI Chat Completions API to/from canonical form."""

    protocol = "openai_chat"
    id = "openai-chat"
    supported_provider_metadata = frozenset({"reasoning_content"})

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        return bool(path.rstrip("/").endswith("/v1/chat/completions"))

    @staticmethod
    def _parse_content_blocks(content: Any) -> list[CanonicalContentBlock]:
        """Parse an OpenAI ``content`` value (string or block array) into
        canonical blocks. Shared by user and developer message decoding so
        both handle text/image/unknown blocks identically."""
        if not isinstance(content, list):
            return [CanonicalTextBlock(text=content)]
        blocks: list[CanonicalContentBlock] = []
        for block in content:
            btype = block.get("type")
            if btype == "text":
                text_val = block.get("text", "")
                if isinstance(text_val, str):
                    blocks.append(CanonicalTextBlock(text=text_val))
            elif btype == "image_url" or btype == "image":
                url_data = block.get("image_url", {})
                if isinstance(url_data, dict):
                    blocks.append(CanonicalImageBlock(
                        source_type="url",
                        url=url_data.get("url", ""),
                        detail=url_data.get("detail"),
                    ))
                else:
                    blocks.append(CanonicalUnknownBlock(source_type=btype, raw=block))
            else:
                # Preserve unknown blocks rather than dropping them
                blocks.append(CanonicalUnknownBlock(source_type=btype, raw=block))
        return blocks

    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        messages: list[CanonicalMessage] = []
        system_content: list[CanonicalContentBlock] = []

        for msg in body.get("messages", []):
            role = msg["role"]
            content = msg.get("content", "")

            if role == "system":
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            system_content.append(CanonicalTextBlock(text=block.get("text", "")))
                else:
                    system_content.append(CanonicalTextBlock(text=content))
                continue

            if role == "developer":
                # Distinct from "system": preserved as an ordered message
                # (role="developer") rather than hoisted into the top-level
                # system field, matching what the OpenAI Chat egress renderer
                # already expects (upstreams/openai_chat.py._render_message
                # maps role="developer" -> "system" per-message). Coding
                # agents commonly send high-priority policy/workspace
                # constraints on this role; silently dropping it changes
                # agent behavior without telling the client.
                blocks = self._parse_content_blocks(content)
                messages.append(CanonicalMessage(
                    role="developer",
                    content=blocks if blocks else [CanonicalTextBlock(text="")],
                ))
                continue

            if role == "assistant":
                asst_blocks: list[CanonicalContentBlock] = []
                tc: list[CanonicalToolCallBlock] = []

                # Capture provider reasoning content (e.g.
                # DeepSeek's ``reasoning_content`` field) so it
                # round-trips back to clients that support it.
                reasoning_content = msg.get("reasoning_content")
                if isinstance(reasoning_content, str) and reasoning_content:
                    asst_blocks.append(CanonicalReasoningBlock(
                        content=reasoning_content,
                        signature=None,
                        provider_metadata=ProviderMetadata(
                            origin_protocol="openai_chat",
                            origin_provider="openai",
                            origin_model=body.get("model", ""),
                            metadata_kind="reasoning_content",
                            opaque_value=reasoning_content,
                            required_for_replay=True,
                            forwarding_policy=MetadataForwardingPolicy.PRESERVE_IF_COMPATIBLE,
                        ),
                    ))

                for raw_tc in msg.get("tool_calls") or []:
                    args_raw = raw_tc.get("function", {}).get("arguments", "{}")
                    if isinstance(args_raw, str):
                        try:
                            args = json.loads(args_raw)
                            raw_arguments = args_raw
                            validated = True
                        except json.JSONDecodeError:
                            args = {}
                            raw_arguments = args_raw
                            validated = False
                    else:
                        args = args_raw if isinstance(args_raw, dict) else {}
                        raw_arguments = args_raw
                        validated = isinstance(args_raw, dict)
                    tool_call = CanonicalToolCallBlock(
                        id=raw_tc.get("id", ""),
                        name=raw_tc.get("function", {}).get("name", ""),
                        arguments=args,
                        raw_arguments=raw_arguments,
                        arguments_validated=validated,
                    )
                    tc.append(tool_call)
                    asst_blocks.append(tool_call)

                if content:
                    asst_blocks.insert(0, CanonicalTextBlock(text=content))

                messages.append(CanonicalMessage(
                    role="assistant",
                    content=asst_blocks if asst_blocks else content,
                ))

            elif role == "user":
                blocks = self._parse_content_blocks(content)
                messages.append(CanonicalMessage(
                    role="user",
                    content=blocks if blocks else [CanonicalTextBlock(text="")],
                ))

            elif role == "tool":
                messages.append(CanonicalMessage(
                    role="tool",
                    content=[CanonicalToolResultBlock(
                        tool_call_id=msg.get("tool_call_id", ""),
                        content=str(msg.get("content", "")),
                    )],
                ))

        tools: list[CanonicalTool] = []
        for tool_spec in body.get("tools") or body.get("functions") or []:
            tools.append(tool_from_openai(tool_spec))

        tc = body.get("tool_choice", "auto")
        if tc == "none":
            tool_choice = canonical_tool_choice("none")
        elif tc == "auto":
            tool_choice = canonical_tool_choice("auto")
        elif tc == "required":
            tool_choice = canonical_tool_choice("required")
        elif isinstance(tc, dict) and tc.get("type") == "function":
            tool_choice = canonical_tool_choice("named", tc.get("function", {}).get("name", ""))
        else:
            # Silently falling back to "auto" here would change the
            # client's request semantics without telling them — a client
            # that explicitly asked for "required" (typo'd, or an
            # unsupported shape) deserves a clear rejection, not quiet
            # behavior it never asked for.
            raise ValueError(f"'tool_choice' has an unrecognized value: {tc!r}")

        return CanonicalRequest(
            request_id=self._resolve_request_id(headers, body),
            session_id=self._resolve_session_id(headers),
            model=CanonicalModelReference(requested_name=str(body.get("model", ""))),
            system=system_content,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            generation=CanonicalGenerationOptions(
                max_output_tokens=self.validate_max_tokens(body.get("max_tokens", 4096)),
                temperature=self.validate_temperature(body.get("temperature", 0.0)),
                top_p=self.validate_top_p(body.get("top_p")),
                stop=self.validate_stop(body.get("stop")),
                stream=body.get("stream", False),
            ),
        )

    def encode_error(self, error: CanonicalError) -> EncodedErrorResponse:
        """Encode a canonical error as an OpenAI error response."""
        from agent_interop.errors import serialize_client_error

        return serialize_client_error(error, "openai_chat")

    @staticmethod
    def _nonstreaming_finish_reason(
        stop_reason: CanonicalStopReason,
        tool_calls_out: list[dict],
    ) -> str:
        """Map a canonical stop reason to OpenAI Chat finish_reason for nonstreaming."""
        from agent_interop.abi import CanonicalStopReason

        if tool_calls_out:
            return "tool_calls"
        if stop_reason == CanonicalStopReason.MAX_TOKENS:
            return "length"
        if stop_reason in (
            CanonicalStopReason.INVALID_OUTPUT,
            CanonicalStopReason.CONTENT_FILTER,
        ):
            return "content_filter"
        if stop_reason in (
            CanonicalStopReason.STOP_SEQUENCE,
            CanonicalStopReason.END_TURN,
        ):
            return "stop"
        # Default
        return "stop"

    def encode_response(self, response: CanonicalResponse) -> dict[str, Any]:
        text = ""
        reasoning_content = ""
        tool_calls_out: list[dict] = []
        for block in response.content:
            if isinstance(block, CanonicalTextBlock) and block.text:
                text += block.text
            elif isinstance(block, CanonicalReasoningBlock) and block.content:
                # Preserve provider reasoning content when compatible
                if self.should_forward_provider_metadata(block.provider_metadata):
                    reasoning_content = block.content
            elif isinstance(block, CanonicalToolCallBlock):
                tc = block
                tool_calls_out.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                })

        finish_reason = self._nonstreaming_finish_reason(response.stop_reason, tool_calls_out)
        message = {
            "role": "assistant",
            "content": text or None,
            "tool_calls": tool_calls_out if tool_calls_out else None,
        }
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
        return {
            "id": response.response_id or f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "model": response.model.requested_name or "unknown",
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def encode_event(self, event: CanonicalEvent) -> str | None:
        if event.type == "text_delta":
            data = {"choices": [{"delta": {"content": event.partial}, "index": 0}]}
            return f"data: {json.dumps(data)}\n\n"
        if event.type == "text":
            return None
        if event.type in ("tool_use_delta", "tool_use"):
            if event.content_block and isinstance(event.content_block, CanonicalToolCallBlock):
                tc = event.content_block
                data = {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": event.index or 0,
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }]
                        },
                        "index": 0,
                        "finish_reason": None,
                    }]
                }
                return f"data: {json.dumps(data)}\n\n"
            return None
        if event.type == "message_stop":
            if event.stop_reason == CanonicalStopReason.TOOL_CALL:
                finish = "tool_calls"
            else:
                finish = "stop"
            data = {
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": finish,
                }]
            }
            return f"data: {json.dumps(data)}\n\n"
        return None

    def parse_tool_result(self, body: dict[str, Any]) -> str:
        return body.get("content", "")

    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": body.get("messages", []),
            "model": body.get("model", ""),
        }

    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "object": "token_count",
            "input_tokens": backend_body.get("input_tokens", 0) or backend_body.get("prompt_tokens", 0),
            "output_tokens": backend_body.get("output_tokens", 0) or backend_body.get("completion_tokens", 0),
        }

    def create_stream_encoder(
        self,
        response_context: dict[str, Any] | None = None,
    ) -> StreamEncoder:
        return OpenAIChatStreamEncoder(self, response_context)


class OpenAIChatStreamEncoder(StreamEncoder):
    """Stateful stream encoder for OpenAI Chat Completions.

    Tracks content block index, tool call lifecycle, and ensures
    exactly one terminal finish_reason event.
    """

    def __init__(self, adapter: OpenAIChatAdapter, response_context: dict[str, Any] | None = None) -> None:
        super().__init__(response_context)
        self._adapter = adapter
        self._tool_indexes: dict[str, int] = {}
        self._next_tool_index: int = 0
        self._usage: dict[str, int] | None = None

    def encode(self, event: CanonicalEvent) -> str | None:
        if event.type == "text_delta":
            if self.state.failure_pending:
                # Suppress text after a streaming error so the model
                # does not appear to recover before the terminal.
                return None
            data = {"choices": [{"delta": {"content": event.partial}, "index": 0}]}
            return f"data: {json.dumps(data)}\n\n"

        if event.type == "tool_use":
            if self.state.failure_pending:
                return None
            cb = event.content_block
            if cb and isinstance(cb, CanonicalToolCallBlock):
                idx = self._next_tool_index
                self._next_tool_index += 1
                self._tool_indexes[cb.name] = idx
                data = {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": idx,
                                "id": cb.id,
                                "type": "function",
                                "function": {"name": cb.name, "arguments": json.dumps(cb.arguments)},
                            }]
                        },
                        "index": 0,
                        "finish_reason": None,
                    }]
                }
                return f"data: {json.dumps(data)}\n\n"
            return None

        if event.type == "tool_use_delta":
            if self.state.failure_pending:
                return None
            idx = self._tool_indexes.get(
                getattr(event.content_block, "name", "") if event.content_block else "",
                self.state.content_block_index,
            )
            data = {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": idx,
                            "function": {"arguments": event.partial},
                        }]
                    },
                    "index": 0,
                    "finish_reason": None,
                }]
            }
            return f"data: {json.dumps(data)}\n\n"

        if event.type == "error":
            # Record failure state and emit a protocol-visible error
            # data frame.  finish() will still emit [DONE] but the
            # terminal ``message_stop`` chunk is suppressed so the
            # client observes the failure.
            self.state.failure_pending = True
            self.state.pending_error = event.error
            err = event.error
            from agent_interop.errors import redact_secrets, sanitize_error_details
            err_payload: dict[str, Any] = {
                "message": redact_secrets(getattr(err, "message", "") or "Stream error"),
                "type": "interop_error",
                "code": getattr(err, "code", "") or "STREAM_ERROR",
                "param": None,
            }
            details = sanitize_error_details(getattr(err, "details", None) if err else None)
            if details:
                err_payload["details"] = details
            err_data: dict[str, Any] = {"error": err_payload}
            return f"data: {json.dumps(err_data)}\n\n"

        if event.type == "usage_update":
            usage: dict[str, int] = {}
            if event.input_tokens is not None:
                usage["prompt_tokens"] = event.input_tokens
            if event.output_tokens is not None:
                usage["completion_tokens"] = event.output_tokens
            if event.input_tokens is not None and event.output_tokens is not None:
                usage["total_tokens"] = event.input_tokens + event.output_tokens
            if usage:
                self._usage = usage
            return None

        if event.type == "message_stop":
            if self.state.terminal_emitted:
                return None
            self.state.terminal_emitted = True
            if self.state.failure_pending:
                # Emit a chunk that exposes the failure as the
                # finish_reason, not as ordinary success.  The
                # client can distinguish this from ``stop`` because
                # a preceding ``error`` frame was already sent.
                stop_reason = event.stop_reason or CanonicalStopReason.INVALID_OUTPUT
                data = {
                    "choices": [{
                        "delta": {},
                        "index": 0,
                        "finish_reason": self._openai_finish_reason(stop_reason, failure=True),
                    }]
                }
                return f"data: {json.dumps(data)}\n\n"
            # When failure_pending is not set, this is a normal completion
            if event.stop_reason == CanonicalStopReason.TOOL_CALL:
                finish = "tool_calls"
            elif event.stop_reason == CanonicalStopReason.MAX_TOKENS:
                finish = "length"
            elif event.stop_reason == CanonicalStopReason.INVALID_OUTPUT:
                # Map INVALID_OUTPUT to content_filter even without failure_pending
                finish = "content_filter"
            else:
                finish = "stop"
            data = {"choices": [{"delta": {}, "index": 0, "finish_reason": finish}]}
            return f"data: {json.dumps(data)}\n\n"

        return None

    def _openai_finish_reason(self, stop_reason: CanonicalStopReason | None, *, failure: bool) -> str:
        """Map a canonical stop reason to an OpenAI Chat ``finish_reason``.

        When ``failure`` is true, ensure the wire value cannot be
        mistaken for ordinary success (``stop``).
        """
        if stop_reason == CanonicalStopReason.TOOL_CALL:
            return "tool_calls"
        if stop_reason == CanonicalStopReason.MAX_TOKENS:
            return "length"
        if stop_reason == CanonicalStopReason.INVALID_OUTPUT:
            return "content_filter"
        if stop_reason == CanonicalStopReason.CONTENT_FILTER:
            return "content_filter"
        if failure:
            return "content_filter"
        return "stop"

    def finish(self) -> str | None:
        """Emit the [DONE] sentinel.

        For OpenAI Chat, the ``[DONE]`` sentinel is still emitted after
        both successful and failed streams; the failure state has
        already been advertised by the ``error`` data frame and the
        non-``stop`` finish_reason on the terminal chunk.  ``finish()``
        is idempotent: subsequent calls return ``None`` so the same
        encoder cannot accidentally re-emit ``[DONE]``.
        """
        if self.state.done_emitted:
            return None
        self.state.done_emitted = True
        if self.state.terminal_emitted:
            self.state.terminal_was_failure = self.state.failure_pending
            frames: list[str] = []
            if self._usage and not self.state.terminal_was_failure:
                # Trailing usage-only chunk, matching OpenAI's
                # stream_options.include_usage shape: empty choices,
                # populated usage.
                frames.append(f"data: {json.dumps({'choices': [], 'usage': self._usage})}\n\n")
            frames.append("data: [DONE]\n\n")
            return "".join(frames)
        # If no terminal was emitted yet, do not synthesise a fake
        # success path; emit a failure-aware terminal chunk first.
        if self.state.failure_pending:
            self.state.terminal_emitted = True
            self.state.terminal_was_failure = True
            data = {
                "choices": [{
                    "delta": {},
                    "index": 0,
                    "finish_reason": "content_filter",
                }]
            }
            return f"data: {json.dumps(data)}\n\ndata: [DONE]\n\n"
        return None