"""OpenAI Responses API adapter (/v1/responses).

Adopted by Codex as its primary protocol. Extends the chat format with
structured response objects, response-level IDs, and continuation tokens.
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
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
    CanonicalUnknownBlock,
    ProviderMetadata,
    canonical_tool_choice,
    tool_from_openai,
)
from agent_interop.errors import EncodedErrorResponse
from agent_interop.protocols.base import ClientProtocolAdapter, StreamEncoder

# provider_metadata.metadata_kind for a Responses "function_call" item's
# own response-item id, kept distinct from the call_id used for tool-loop
# pairing (item id and call id are two different identifiers on the wire —
# see decode_request's function_call branch).
_RESPONSES_ITEM_ID_KIND = "responses_item_id"


class OpenAIResponsesAdapter(ClientProtocolAdapter):
    """Translate OpenAI Responses API to/from canonical form."""

    protocol = "openai_responses"
    id = "openai-responses"

    def matches(self, path: str, headers: dict[str, str], body: dict[str, Any]) -> bool:
        if path.rstrip("/").endswith("/v1/responses"):
            return True
        return bool("input" in body and "tools" in body)

    def decode_request(self, body: dict[str, Any], headers: dict[str, str]) -> CanonicalRequest:
        messages: list[CanonicalMessage] = []
        system = body.get("instructions", body.get("system", ""))

        prev_id = body.get("previous_response_id")
        raw_input = body.get("input", [])
        if isinstance(raw_input, str):
            # Plain string input → single user text message
            messages.append(CanonicalMessage(role="user", content=[CanonicalTextBlock(text=raw_input)]))
            raw_input = []
        for raw_msg in raw_input:
            if not isinstance(raw_msg, dict):
                continue
            item_type = raw_msg.get("type", "")
            # Handle role-based items without explicit type (backward compat)
            if not item_type and "role" in raw_msg:
                item_type = "message"

            if item_type == "message":
                role = raw_msg.get("role", "user")
                content = raw_msg.get("content", "")

                if role == "system":
                    system = content if isinstance(content, str) else system
                    continue
                if role == "developer":
                    system = content if not system else system + "\n" + content
                    continue

                if role == "assistant":
                    blocks: list[CanonicalContentBlock] = []
                    tc: list[CanonicalToolCallBlock] = []
                    for raw_tc in raw_msg.get("tool_calls") or []:
                        args_raw = raw_tc.get("arguments", "{}")
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
                        tcall = CanonicalToolCallBlock(
                            id=raw_tc.get("id", f"tc_{uuid.uuid4().hex[:12]}"),
                            name=raw_tc.get("name", ""),
                            arguments=args,
                            raw_arguments=raw_arguments,
                            arguments_validated=validated,
                        )
                        tc.append(tcall)
                        blocks.append(tcall)
                    if content:
                        blocks.insert(0, CanonicalTextBlock(text=content))
                    messages.append(CanonicalMessage(
                        role="assistant",
                        content=blocks if blocks else [CanonicalTextBlock(text="")],
                    ))
                elif role == "user":
                    if isinstance(content, list):
                        # Preserve every block, not just text — silently
                        # joining only the "input_text"/"text" items and
                        # dropping the rest (input_image, input_file, ...)
                        # loses content the client actually sent rather
                        # than just its formatting.
                        user_blocks: list[CanonicalContentBlock] = []
                        for b in content:
                            if not isinstance(b, dict):
                                continue
                            if b.get("type") in ("input_text", "text"):
                                user_blocks.append(CanonicalTextBlock(text=b.get("text", "")))
                            else:
                                user_blocks.append(CanonicalUnknownBlock(source_type=b.get("type", ""), raw=b))
                        messages.append(CanonicalMessage(
                            role="user",
                            content=user_blocks if user_blocks else [CanonicalTextBlock(text="")],
                        ))
                    else:
                        messages.append(CanonicalMessage(role="user", content=[CanonicalTextBlock(text=str(content))]))
                elif role == "tool":
                    messages.append(CanonicalMessage(
                        role="tool",
                        content=[CanonicalToolResultBlock(
                            tool_call_id=raw_msg.get("tool_call_id", ""),
                            content=str(raw_msg.get("content", "")),
                        )],
                    ))

            elif item_type == "function_call":
                # This is the native Responses-API shape (what Codex, the
                # real client for this protocol, actually sends) — it must
                # carry raw_arguments/arguments_validated exactly like the
                # backward-compat "message role=assistant with tool_calls"
                # branch above, or a parse failure here silently defaults
                # to arguments_validated=True (the dataclass default) with
                # an empty {} argument dict and the original text gone —
                # mislabeling malformed data as validated.
                args_raw = raw_msg.get("arguments", "{}")
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
                # Responses distinguishes the response item's own "id" from
                # the "call_id" used to pair this call with its later
                # function_call_output. Tool-loop pairing (history
                # reconciliation, transaction bookkeeping) must key off
                # call_id — using "id" here silently orphaned the pairing
                # whenever a client sent distinct item_/call_ ids (item_123
                # vs call_456), since function_call_output only ever carries
                # call_id. The item id is preserved separately so egress can
                # round-trip both.
                item_id = raw_msg.get("id")
                call_id = raw_msg.get("call_id") or item_id or f"tc_{uuid.uuid4().hex[:12]}"
                provider_metadata = None
                if item_id and item_id != call_id:
                    provider_metadata = ProviderMetadata(
                        origin_protocol="openai_responses",
                        metadata_kind=_RESPONSES_ITEM_ID_KIND,
                        opaque_value=item_id,
                    )
                messages.append(CanonicalMessage(
                    role="assistant",
                    content=[CanonicalToolCallBlock(
                        id=call_id,
                        name=raw_msg.get("name", ""),
                        arguments=args,
                        raw_arguments=raw_arguments,
                        arguments_validated=validated,
                        provider_metadata=provider_metadata,
                    )],
                ))

            elif item_type == "function_call_output":
                messages.append(CanonicalMessage(
                    role="tool",
                    content=[CanonicalToolResultBlock(
                        tool_call_id=raw_msg.get("call_id", ""),
                        content=str(raw_msg.get("output", "")),
                    )],
                ))

            elif item_type == "reasoning":
                # Preserve reasoning as unknown block for now
                messages.append(CanonicalMessage(
                    role="assistant",
                    content=[CanonicalUnknownBlock(source_type="reasoning", raw=raw_msg)],
                ))

            else:
                # Unknown item type — preserve as unknown
                messages.append(CanonicalMessage(
                    role="user",
                    content=[CanonicalUnknownBlock(source_type=item_type, raw=raw_msg)],
                ))

        tools: list[CanonicalTool] = []
        for tspec in body.get("tools", []) + (body.get("functions", [])):
            tools.append(tool_from_openai(tspec))

        tc = body.get("tool_choice", "auto")
        if isinstance(tc, str):
            if tc not in ("auto", "none", "required"):
                # Silently falling back to "auto" would change the
                # client's request semantics without telling them.
                raise ValueError(f"'tool_choice' has an unrecognized value: {tc!r}")
            tool_choice = canonical_tool_choice(tc)
        else:
            raise ValueError(f"'tool_choice' must be a string, got {tc!r}")

        return CanonicalRequest(
            request_id=self._resolve_request_id(headers, body),
            session_id=self._resolve_session_id(headers),
            model=CanonicalModelReference(requested_name=str(body.get("model", ""))),
            system=[CanonicalTextBlock(text=system)] if system else [],
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            generation=CanonicalGenerationOptions(
                max_output_tokens=self.validate_max_tokens(
                    body.get("max_output_tokens", body.get("max_tokens", 4096))
                ),
                temperature=self.validate_temperature(body.get("temperature", 0.0)),
                top_p=self.validate_top_p(body.get("top_p")),
                stream=body.get("stream", False),
            ),
            previous_response_id=prev_id,
            metadata={
                "response_format": body.get("text", {}),
            },
        )

    def encode_error(self, error: CanonicalError) -> EncodedErrorResponse:
        """Encode a canonical error as an OpenAI Responses error response."""
        from agent_interop.errors import serialize_client_error

        return serialize_client_error(error, "openai_responses")

    @staticmethod
    def _response_status(stop_reason: CanonicalStopReason) -> str:
        """Map canonical stop reason to OpenAI Responses API status.

        OpenAI Responses uses:
        - completed: Normal completion
        - failed: Error or content filter
        - in_progress: Still generating (streaming only)
        """
        if stop_reason in (
            CanonicalStopReason.INVALID_OUTPUT,
            CanonicalStopReason.CONTENT_FILTER,
            CanonicalStopReason.BACKEND_ERROR,
        ):
            return "failed"
        return "completed"

    def encode_response(self, response: CanonicalResponse) -> dict[str, Any]:
        output: list[dict[str, Any]] = []
        text = ""
        for block in response.content:
            if isinstance(block, CanonicalTextBlock) and block.text:
                text += block.text
            elif isinstance(block, CanonicalToolCallBlock):
                tc = block
                item_id = tc.id
                meta = tc.provider_metadata
                if meta is not None and meta.metadata_kind == _RESPONSES_ITEM_ID_KIND:
                    item_id = meta.opaque_value
                output.append({
                    "type": "function_call",
                    "id": item_id,
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                    "status": "completed",
                })

        if text:
            output.insert(0, {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            })

        return {
            "id": response.response_id or f"resp_{uuid.uuid4().hex[:16]}",
            "object": "response",
            "status": self._response_status(response.stop_reason),
            "output": output,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        }

    def encode_event(self, event: CanonicalEvent) -> str | None:
        """Encode canonical events as OpenAI Responses SSE format.

        Responses uses a specific SSE format with typed events.
        """
        if event.type == "text_delta":
            return self._sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "delta": event.partial,
                "index": event.index,
            })
        if event.type == "text":
            return self._sse("response.output_text.annotated", {
                "type": "response.output_text.annotated",
                "text": event.partial or "",
            })
        if event.type == "tool_use":
            cb = event.content_block
            if cb and isinstance(cb, CanonicalToolCallBlock):
                item_id = cb.id
                meta = cb.provider_metadata
                if meta is not None and meta.metadata_kind == _RESPONSES_ITEM_ID_KIND:
                    item_id = meta.opaque_value
                return self._sse("response.function_call_arguments.done", {
                    "type": "response.function_call_arguments.done",
                    "item_id": item_id,
                    "name": cb.name,
                    "arguments": json.dumps(cb.arguments),
                    "call_id": cb.id,
                })
        if event.type == "message_stop":
            return self._sse("response.completed", {
                "type": "response.completed",
            })
        return None

    def parse_tool_result(self, body: dict[str, Any]) -> str:
        output = body.get("output", [])
        for item in output:
            if item.get("type") == "function_call_output":
                return item.get("output", "")
        return body.get("content", "")

    def count_tokens_request(self, body: dict[str, Any]) -> dict[str, Any]:
        return {
            "input": body.get("input", []),
            "model": body.get("model", ""),
        }

    def count_tokens_response(self, backend_body: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_tokens": backend_body.get("input_tokens", 0),
            "output_tokens": backend_body.get("output_tokens", 0),
        }

    @classmethod
    def _sse(cls, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def create_stream_encoder(
        self,
        response_context: dict[str, Any] | None = None,
    ) -> StreamEncoder:
        return OpenAIResponsesStreamEncoder(self, response_context)


class OpenAIResponsesStreamEncoder(StreamEncoder):
    """Stateful stream encoder for OpenAI Responses protocol.

    Tracks response item lifecycle with stable IDs and sequence numbers.
    Emits typed Responses events:
      response.created → output_text.delta | output_item.added →
      function_call_arguments.delta → output_item.done →
      response.completed | response.error
    """

    def __init__(self, adapter: OpenAIResponsesAdapter, response_context: dict[str, Any] | None = None) -> None:
        super().__init__(response_context)
        self._adapter = adapter
        self._tool_call_ids: dict[str, str] = {}  # stable key -> call_id
        self._output_index: int = 0
        self._content_index: int = 0
        self._sequence_number: int = 0
        self._usage: dict[str, int] = {}

    def _next_sequence(self) -> int:
        self._sequence_number += 1
        return self._sequence_number

    def encode(self, event: CanonicalEvent) -> str | None:
        if event.type == "error":
            # Record the error so the message_stop terminal becomes
            # ``response.failed`` instead of ``response.completed``.
            self.state.failure_pending = True
            self.state.pending_error = event.error
            if event.error:
                from agent_interop.errors import redact_secrets, sanitize_error_details
                err = event.error
                err_payload: dict[str, Any] = {
                    "type": "error",
                    "code": err.code or "server_error",
                    "message": redact_secrets(err.message or "Stream error"),
                }
                details = sanitize_error_details(getattr(err, "details", None))
                if details:
                    err_payload["details"] = details
                return self._sse("error", err_payload)
            return None

        if event.type == "message_start":
            return self._sse("response.created", {
                "type": "response.created",
                "response": {
                    "id": self.state.response_id,
                    "object": "response",
                    "status": "in_progress",
                    "model": self.state.model,
                    "output": [],
                },
            })

        if event.type == "text_delta":
            if self.state.failure_pending:
                return None
            seq = self._next_sequence()
            return self._sse("response.output_text.delta", {
                "type": "response.output_text.delta",
                "item_id": f"{self.state.response_id}_text",
                "output_index": 0,
                "content_index": 0,
                "sequence_number": seq,
                "delta": event.partial,
            })

        if event.type == "tool_use":
            if self.state.failure_pending:
                return None
            cb = event.content_block
            if cb and isinstance(cb, CanonicalToolCallBlock):
                idx = self._output_index
                self._output_index += 1
                # Use stable key: id if available, else name+index
                key = cb.id or f"{cb.name}_{idx}"
                self._tool_call_ids[key] = cb.id or key
                item_id = cb.id or key
                meta = cb.provider_metadata
                if meta is not None and meta.metadata_kind == _RESPONSES_ITEM_ID_KIND:
                    item_id = meta.opaque_value
                seq = self._next_sequence()
                return self._sse("response.output_item.added", {
                    "type": "response.output_item.added",
                    "output_index": idx,
                    "sequence_number": seq,
                    "item": {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": cb.id or key,
                        "name": cb.name,
                        "arguments": "",
                        "status": "in_progress",
                    },
                })

        if event.type == "tool_use_delta":
            if self.state.failure_pending:
                return None
            # Find call_id by stable key from content block
            key = ""
            if event.content_block and isinstance(event.content_block, CanonicalToolCallBlock):
                key = event.content_block.id or event.content_block.name
            call_id = self._tool_call_ids.get(key, key or "unknown")
            seq = self._next_sequence()
            return self._sse("response.function_call_arguments.delta", {
                "type": "response.function_call_arguments.delta",
                "item_id": key or call_id,
                "call_id": call_id,
                "output_index": self._output_index - 1 if self._output_index > 0 else 0,
                "sequence_number": seq,
                "delta": event.partial,
            })

        if event.type == "usage_update":
            if event.input_tokens is not None:
                self._usage["input_tokens"] = event.input_tokens
            if event.output_tokens is not None:
                self._usage["output_tokens"] = event.output_tokens
            return None

        if event.type == "message_stop":
            if not self.state.terminal_emitted:
                self.state.terminal_emitted = True
                if self.state.failure_pending or self.state.pending_error is not None:
                    from agent_interop.errors import redact_secrets
                    self.state.terminal_was_failure = True
                    err = self.state.pending_error
                    raw_message = getattr(err, "message", "") or "Stream error" if err else "Stream error"
                    return self._sse("response.failed", {
                        "type": "response.failed",
                        "response": {
                            "id": self.state.response_id,
                            "status": "failed",
                            "error": {
                                "code": getattr(err, "code", "") or "server_error" if err else "server_error",
                                "message": redact_secrets(raw_message),
                            },
                        },
                    })
                # Only emit response.completed when there is no failure
                response_payload: dict[str, Any] = {
                    "id": self.state.response_id,
                    "status": "completed",
                }
                if self._usage:
                    response_payload["usage"] = dict(self._usage)
                return self._sse("response.completed", {
                    "type": "response.completed",
                    "response": response_payload,
                })
            return None

        return None

    def finish(self) -> str | None:
        """Emit final [DONE] sentinel.

        For OpenAI Responses, the protocol's failure terminal
        (``response.failed``) is treated as a complete stream
        termination, so ``[DONE]`` is omitted when the stream ended
        in failure.  This prevents the client from interpreting
        ``[DONE]`` after ``response.failed`` as ordinary success.
        """
        if self.state.terminal_was_failure:
            return None
        return "data: [DONE]\n\n"

    def _sse(self, event_type: str, data: dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"