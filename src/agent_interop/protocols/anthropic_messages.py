"""Anthropic Messages API adapter (/v1/messages, /v1/messages/count_tokens).

Translates between the Anthropic Messages API and Interop canonical form.
Supports content blocks, tool_use, tool_result, thinking blocks, and streaming.
"""

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
    RequestedCapabilities,
    canonical_tool_choice,
    tool_from_anthropic,
)
from agent_interop.errors import EncodedErrorResponse
from agent_interop.protocols.base import ClientProtocolAdapter, StreamEncoder


class AnthropicMessagesAdapter(ClientProtocolAdapter):
    """Translate Anthropic Messages API to/from canonical form."""

    protocol = "anthropic_messages"
    id = "anthropic-messages"

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        if path.rstrip("/").endswith("/v1/messages"):
            return True
        return bool(headers.get("anthropic-version") or headers.get("x-api-key", "").startswith("sk-ant-"))

    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        system: list[CanonicalContentBlock] = []
        system_blocks = body.get("system", [])
        if isinstance(system_blocks, str):
            system.append(CanonicalTextBlock(text=system_blocks))
        else:
            for sys_block in system_blocks:
                if sys_block.get("type") == "text":
                    system.append(CanonicalTextBlock(text=sys_block.get("text", "")))
                elif sys_block.get("type") == "thinking":
                    system.append(CanonicalReasoningBlock(
                        content=sys_block.get("thinking", ""),
                        signature=sys_block.get("signature"),
                    ))
                else:
                    system.append(CanonicalUnknownBlock(source_type=sys_block.get("type", "unknown"), raw=sys_block))

        messages: list[CanonicalMessage] = []
        for msg in body.get("messages", []):
            role_str = msg["role"]
            raw_content = msg.get("content", "")

            if role_str == "assistant":
                if isinstance(raw_content, list):
                    blocks: list[CanonicalContentBlock] = []
                    for block in raw_content:
                        btype = block.get("type", "text")
                        if btype == "text":
                            blocks.append(CanonicalTextBlock(text=block.get("text", "")))
                        elif btype == "tool_use":
                            tid = block.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
                            inp = block.get("input", {})
                            if isinstance(inp, dict):
                                args = inp
                                raw_arguments = inp
                                validated = True
                            else:
                                args = {}
                                raw_arguments = inp
                                validated = False
                            blocks.append(CanonicalToolCallBlock(
                                id=tid,
                                name=block.get("name", ""),
                                arguments=args,
                                raw_arguments=raw_arguments,
                                arguments_validated=validated,
                            ))
                        elif btype == "thinking":
                            blocks.append(CanonicalReasoningBlock(
                                content=block.get("thinking", ""),
                                signature=block.get("signature"),
                            ))
                        else:
                            blocks.append(CanonicalUnknownBlock(source_type=btype, raw=block))
                    messages.append(CanonicalMessage(role="assistant", content=blocks))
                else:
                    messages.append(CanonicalMessage(role="assistant", content=[CanonicalTextBlock(text=str(raw_content))]))

            if role_str == "user":
                if isinstance(raw_content, list):
                    # Preserve all content blocks in original order
                    user_blocks: list[CanonicalContentBlock] = []
                    for block in raw_content:
                        if block.get("type") == "text":
                            user_blocks.append(CanonicalTextBlock(text=block.get("text", "")))
                        elif block.get("type") == "tool_result":
                            # Render structured tool-result content deterministically
                            result_content = block.get("content", "")
                            if isinstance(result_content, list):
                                text_parts = [b.get("text", "") for b in result_content if isinstance(b, dict)]
                                result_content = "\n".join(text_parts)
                            user_blocks.append(CanonicalToolResultBlock(
                                tool_call_id=block.get("tool_use_id", ""),
                                content=str(result_content) if result_content else "",
                                is_error=bool(block.get("is_error", False)),
                            ))
                        elif block.get("type") == "image":
                            user_blocks.append(CanonicalImageBlock(
                                source_type=block.get("source", {}).get("type", ""),
                                media_type=block.get("source", {}).get("media_type", ""),
                                data=block.get("source", {}).get("data", ""),
                                url=block.get("source", {}).get("url", ""),
                                detail=block.get("detail"),
                            ))
                        else:
                            user_blocks.append(CanonicalUnknownBlock(source_type=block.get("type", "unknown"), raw=block))
                    # Anthropic's wire format has no dedicated "tool" role —
                    # a tool result is a content block inside a role:"user"
                    # message, per the real API contract (confirmed via a
                    # live Claude Code request). Interop's own canonical
                    # model uses a dedicated role="tool" internally (as the
                    # OpenAI Chat decoder already does below, and as
                    # history/reconcile.py's safety check requires) — a
                    # pure tool-result turn is normalized to role="tool"
                    # here so canonical messages have one consistent shape
                    # regardless of source protocol. A message mixing a
                    # tool_result with the user's own new text/images keeps
                    # role="user" — that combination is a real user turn
                    # with a tool_result attached, not a pure tool-result
                    # message the same way an OpenAI role:"tool" message is.
                    if user_blocks and all(
                        isinstance(b, CanonicalToolResultBlock) for b in user_blocks
                    ):
                        messages.append(CanonicalMessage(role="tool", content=user_blocks))
                    else:
                        messages.append(CanonicalMessage(role="user", content=user_blocks))
                else:
                    messages.append(CanonicalMessage(role="user", content=[CanonicalTextBlock(text=str(raw_content))]))

            elif role_str == "tool":
                messages.append(CanonicalMessage(
                    role="tool",
                    content=[CanonicalToolResultBlock(
                        tool_call_id=msg.get("tool_use_id", ""),
                        content=str(raw_content),
                    )],
                ))

            elif role_str == "system" or role_str == "developer":
                # System/developer role messages in the messages array —
                # treat content as a system block prepended to the next user message
                # (Anthropic API allows system blocks inside messages)
                if isinstance(raw_content, list):
                    sys_blocks: list[CanonicalContentBlock] = []
                    for block in raw_content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            sys_blocks.append(CanonicalTextBlock(text=block.get("text", "")))
                        elif isinstance(block, dict):
                            sys_blocks.append(CanonicalUnknownBlock(source_type=block.get("type", "unknown"), raw=block))
                    if sys_blocks:
                        messages.append(CanonicalMessage(role="system", content=sys_blocks))
                elif isinstance(raw_content, str) and raw_content:
                    messages.append(CanonicalMessage(role="system", content=[CanonicalTextBlock(text=raw_content)]))

        tools: list[CanonicalTool] = []
        for tspec in body.get("tools", []):
            tools.append(tool_from_anthropic(tspec))

        tc = body.get("tool_choice", {"type": "auto"})
        if isinstance(tc, str):
            tc_map = {"auto": "auto", "any": "required", "tool": "named", "none": "none"}
            if tc not in tc_map:
                # Silently falling back to "auto" would change the
                # client's request semantics without telling them.
                raise ValueError(f"'tool_choice' has an unrecognized value: {tc!r}")
            tool_choice = canonical_tool_choice(tc_map[tc])
        elif isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "tool":
                tool_choice = canonical_tool_choice("named", tc.get("name", ""))
            elif tc_type == "any":
                tool_choice = canonical_tool_choice("required")
            elif tc_type == "auto":
                tool_choice = canonical_tool_choice("auto")
            elif tc_type == "none":
                tool_choice = canonical_tool_choice("none")
            else:
                raise ValueError(f"'tool_choice.type' has an unrecognized value: {tc_type!r}")
        else:
            raise ValueError(f"'tool_choice' must be a string or object, got {tc!r}")

        return CanonicalRequest(
            request_id=self._resolve_request_id(headers, body),
            session_id=self._resolve_session_id(headers),
            model=CanonicalModelReference(requested_name=str(body.get("model", ""))),
            system=system,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            generation=CanonicalGenerationOptions(
                max_output_tokens=self.validate_max_tokens(body.get("max_tokens", 4096)),
                temperature=self.validate_temperature(body.get("temperature", 0.0)),
                top_p=self.validate_top_p(body.get("top_p")),
                stop=self.validate_stop(body.get("stop_sequences")),
                stream=body.get("stream", False),
            ),
            requested_capabilities=RequestedCapabilities(
                tools=bool(tools),
                parallel_tools=bool(tc.get("disable_parallel_tool_use") is False) if isinstance(tc, dict) else False,
                reasoning=bool(body.get("thinking")),
                images=any(getattr(block, "type", "") == "image" for message in messages for block in message.content),
                structured_output=False,
                tool_result_continuation=any(getattr(block, "type", "") == "tool_result" for message in messages for block in message.content),
                sequential_tools=any(getattr(block, "type", "") == "tool_result" for message in messages for block in message.content),
                exact_named_tool=tool_choice.mode.value == "named",
            ),
            metadata={
                "anthropic_version": headers.get("anthropic-version", body.get("anthropic-version", "2023-06-01")),
                "metadata": body.get("metadata", {}),
            },
        )

    def encode_error(self, error: CanonicalError) -> EncodedErrorResponse:
        """Encode a canonical error as an Anthropic error response."""
        from agent_interop.errors import serialize_client_error

        return serialize_client_error(error, "anthropic")

    def encode_response(self, response: CanonicalResponse) -> dict[str, Any]:
        content: list[dict[str, Any]] = []
        for block in response.content:
            if isinstance(block, CanonicalTextBlock):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, CanonicalToolCallBlock):
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                })
            elif isinstance(block, CanonicalReasoningBlock):
                content.append({
                    "type": "thinking",
                    "thinking": block.content,
                    "signature": block.signature or "",
                })

        # Map canonical stop_reason back to Anthropic native value
        canonical_stop_map = {
            CanonicalStopReason.END_TURN: "end_turn",
            CanonicalStopReason.TOOL_CALL: "tool_use",
            CanonicalStopReason.MAX_TOKENS: "max_tokens",
            CanonicalStopReason.STOP_SEQUENCE: "stop_sequence",
            CanonicalStopReason.INVALID_OUTPUT: "invalid_output",
            CanonicalStopReason.BACKEND_ERROR: "error",
        }
        anthropic_stop = canonical_stop_map.get(response.stop_reason, "end_turn")

        return {
            "id": response.response_id or f"msg_{uuid.uuid4().hex[:16]}",
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": response.model.requested_name or "unknown",
            "stop_reason": anthropic_stop,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def encode_event(self, event: CanonicalEvent) -> str | None:
        """Encode canonical events as Anthropic SSE events."""
        if event.type == "text_delta":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "text_delta", "text": event.partial},
            })
        if event.type == "text":
            return self._sse("content_block_start", {
                "type": "content_block_start",
                "index": event.index,
                "content_block": {"type": "text", "text": event.partial or ""},
            })
        if event.type == "thinking_delta":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "thinking_delta", "thinking": event.partial},
            })
        if event.type == "thinking":
            return self._sse("content_block_start", {
                "type": "content_block_start",
                "index": event.index,
                "content_block": {
                    "type": "thinking",
                    "thinking": event.partial or "",
                    "signature": getattr(event.content_block, "signature", None) if event.content_block else None,
                },
            })
        if event.type == "thinking_signature":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {
                    "type": "signature_delta",
                    "signature": event.partial,
                },
            })
        if event.type == "tool_use":
            cb = event.content_block
            if cb and isinstance(cb, CanonicalToolCallBlock):
                return self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": event.index,
                    "content_block": {
                        "type": "tool_use",
                        "id": cb.id,
                        "name": cb.name,
                        "input": cb.arguments,
                    },
                })
        if event.type == "tool_use_delta":
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": event.index,
                "delta": {"type": "input_json_delta", "partial_json": event.partial},
            })
        if event.type == "content_block_stop":
            return self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": event.index,
            })
        if event.type == "message_stop":
            return self._sse("message_stop", {"type": "message_stop"})
        return None

    def encode_stream_done(self) -> str:
        # Anthropic Messages streaming terminates with message_stop event,
        # not an OpenAI-style [DONE] sentinel. Return empty string.
        return ""

    def parse_tool_result(self, body: dict[str, Any]) -> str:
        content = body.get("content", "")
        if isinstance(content, list):
            return " ".join(b.get("text", "") for b in content if b.get("type") == "text")
        return str(content)

    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "messages": body.get("messages", []),
            "system": body.get("system", ""),
            "model": body.get("model", ""),
        }

    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_tokens": backend_body.get("input_tokens", 0),
            "output_tokens": backend_body.get("output_tokens", 0),
        }

    _EVENT_COUNTER = 0

    @classmethod
    def _sse(cls, event_type: str, data: dict[str, Any]) -> str:
        cls._EVENT_COUNTER += 1
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def create_stream_encoder(
        self,
        response_context: dict[str, Any] | None = None,
    ) -> StreamEncoder:
        return AnthropicStreamEncoder(self, response_context)


class AnthropicStreamEncoder(StreamEncoder):
    """Stateful stream encoder for Anthropic Messages protocol.

    Tracks content block lifecycle (start/delta/stop) and emits
    valid Anthropic SSE sequences: message_start → content_block_start →
    content_block_delta* → content_block_stop → message_delta → message_stop.
    """

    def __init__(self, adapter: AnthropicMessagesAdapter, response_context: dict[str, Any] | None = None) -> None:
        super().__init__(response_context)
        self._adapter = adapter
        self._block_indexes: dict[str, int] = {}  # stable key -> index
        self._next_index: int = 0
        self._text_block_started: bool = False
        self._current_text_index: int | None = None
        self._usage: dict[str, int] = {}

    def _sse(self, event_type: str, data: dict[str, Any]) -> str:
        """Format an SSE frame for Anthropic Messages."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def encode(self, event: CanonicalEvent) -> str | None:
        if event.type == "error":
            # Record failure state and emit Anthropic's error SSE
            # event.  Any subsequent text/tool events are suppressed
            # and ``message_stop`` is emitted as a failure terminal
            # (``message_delta`` with ``stop_reason=invalid_output``
            # and an attached ``error`` payload).
            self.state.failure_pending = True
            self.state.pending_error = event.error
            err = event.error
            # Look up the error type from ERROR_REGISTRY — single
            # source of truth rather than a duplicated hardcoded map.
            from agent_interop.errors import (
                get_error_descriptor,
                redact_secrets,
                sanitize_error_details,
            )
            err_type = "api_error"
            if err is not None and hasattr(err, "code") and err.code:
                desc = get_error_descriptor(err.code)
                err_type = desc.anthropic_type
            err_payload: dict[str, Any] = {
                "type": err_type,
                "message": redact_secrets(getattr(err, "message", "") or "Stream error"),
            }
            details = sanitize_error_details(getattr(err, "details", None) if err else None)
            if details:
                err_payload["details"] = details
            return self._sse("error", {
                "type": "error",
                "error": err_payload,
            })

        if event.type == "message_start":
            return self._sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": self.state.response_id,
                    "type": "message",
                    "role": "assistant",
                    "model": self.state.model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })

        if event.type == "text_delta":
            if self.state.failure_pending:
                return None
            # Start text block if not started. Anthropic's real API always
            # opens a block with empty text and sends the actual content as
            # a separate content_block_delta — this must never fold the
            # first delta's text into content_block_start's hardcoded ""
            # (that silently drops it, which is fatal for any response
            # whose entire text arrives in a single text_delta event, e.g.
            # the BUFFER_TEXTUAL_RESPONSE path that buffers a whole prompted-
            # mode turn and yields it as one event).
            if not self._text_block_started:
                idx = self._next_index
                self._next_index += 1
                self._current_text_index = idx
                self._text_block_started = True
                start_frame = self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "text",
                        "text": "",
                    },
                })
                if not event.partial:
                    return start_frame
                delta_frame = self._sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": event.partial},
                })
                return start_frame + delta_frame
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": self._current_text_index,
                "delta": {"type": "text_delta", "text": event.partial},
            })

        if event.type == "tool_use":
            if self.state.failure_pending:
                return None
            cb = event.content_block
            if cb and isinstance(cb, CanonicalToolCallBlock):
                # Use stable key for tool call (id if available, else name+index)
                key = cb.id or f"{cb.name}_{self._next_index}"
                idx = self._next_index
                self._next_index += 1
                self._block_indexes[key] = idx
                start_frame = self._sse("content_block_start", {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": cb.id,
                        "name": cb.name,
                        # Anthropic's real streaming contract always opens
                        # a tool_use block with an EMPTY input — the real
                        # arguments arrive via subsequent input_json_delta
                        # chunks that the client concatenates and parses
                        # once the block closes. The gateway hands this
                        # encoder the complete, already-decided call in one
                        # shot (never a separate tool_use_delta/
                        # content_block_stop pair for this path — see
                        # gateway.py's _emit_batch_decision_events, the
                        # only caller), so this single encode() call must
                        # synthesize the full start+delta+stop sequence
                        # itself. Found via a real live-client run: with a
                        # fully-populated "input" here and no delta/stop
                        # ever following, Claude Code's own SDK — which
                        # rebuilds input purely from accumulated deltas —
                        # got an empty, unparseable string and reported
                        # "the model's tool call could not be parsed".
                        "input": {},
                    },
                })
                delta_frame = self._sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(cb.arguments),
                    },
                })
                stop_frame = self._sse("content_block_stop", {
                    "type": "content_block_stop",
                    "index": idx,
                })
                return start_frame + delta_frame + stop_frame

        if event.type == "tool_use_delta":
            if self.state.failure_pending:
                return None
            # Find the index for this tool
            tool_key: str | None = None
            if event.content_block and isinstance(event.content_block, CanonicalToolCallBlock):
                tool_key = event.content_block.id or f"{event.content_block.name}_{self._block_indexes.get(event.content_block.name, '')}"
            if tool_key and tool_key in self._block_indexes:
                idx = self._block_indexes[tool_key]
            else:
                # Fallback
                idx = self.state.content_block_index
            return self._sse("content_block_delta", {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "input_json_delta", "partial_json": event.partial},
            })

        if event.type == "content_block_stop":
            # Emit content_block_stop for open block
            idx = self._current_text_index if self._current_text_index is not None else self.state.content_block_index
            self._text_block_started = False
            self._current_text_index = None
            return self._sse("content_block_stop", {
                "type": "content_block_stop",
                "index": idx,
            })

        if event.type == "usage_update":
            # Track usage for message_delta
            if event.input_tokens is not None:
                self._usage["input_tokens"] = event.input_tokens
            if event.output_tokens is not None:
                self._usage["output_tokens"] = event.output_tokens
            return None

        if event.type == "message_stop":
            if not self.state.terminal_emitted:
                self.state.terminal_emitted = True
                frames: list[str] = []

                # Close any open text block
                if self._text_block_started and self._current_text_index is not None:
                    frames.append(self._sse("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._current_text_index,
                    }))
                    self._text_block_started = False
                    self._current_text_index = None

                # Map stop reason.  If a streaming error is pending we
                # always report ``invalid_output`` regardless of the
                # event's declared stop_reason so the client cannot
                # confuse this with a normal completion.
                if self.state.failure_pending:
                    stop_reason = "invalid_output"
                    self.state.terminal_was_failure = True
                elif event.stop_reason == CanonicalStopReason.TOOL_CALL:
                    stop_reason = "tool_use"
                elif event.stop_reason == CanonicalStopReason.MAX_TOKENS:
                    stop_reason = "max_tokens"
                elif event.stop_reason == CanonicalStopReason.INVALID_OUTPUT:
                    stop_reason = "invalid_output"
                else:
                    stop_reason = "end_turn"

                delta_payload: dict[str, Any] = {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason},
                }
                # When failure is pending, attach error details to message_delta
                if self.state.failure_pending and self.state.pending_error:
                    from agent_interop.errors import redact_secrets, sanitize_error_details
                    err = self.state.pending_error
                    delta_payload["error"] = {
                        "type": "api_error",
                        "message": redact_secrets(getattr(err, "message", "") or "Stream error"),
                    }
                    if hasattr(err, "code") and err.code:
                        delta_payload["error"]["code"] = err.code
                    details = sanitize_error_details(getattr(err, "details", None))
                    if details:
                        delta_payload["error"]["details"] = details

                if self._usage:
                    delta_payload["usage"] = self._usage
                frames.append(self._sse("message_delta", delta_payload))
                return "".join(frames)
            return None

        return None

    def finish(self) -> str | None:
        """Emit message_stop after the message_delta has been sent."""
        if self.state.terminal_emitted:
            return self._sse("message_stop", {"type": "message_stop"})
        return None
