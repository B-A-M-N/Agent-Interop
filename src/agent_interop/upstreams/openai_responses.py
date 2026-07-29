"""OpenAI Responses upstream codec — ModelCodec for /v1/responses.

Renders canonical ABI to OpenAI Responses API format and decodes responses.
Handles input items, instructions, tools, previous_response_id,
text/config metadata, and streaming SSE events.
"""

from __future__ import annotations

import json
from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalMessage,
    CanonicalReasoningBlock,
    CanonicalRequest,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
    CanonicalToolResultBlock,
    CanonicalUsage,
    ProtocolKind,
    RawToolCallCandidate,
    ToolChoiceMode,
)
from agent_interop.config import UpstreamProtocol
from agent_interop.upstreams.codec import (
    CodecCapabilities,
    DecodedModelResponse,
    DecodedStreamComplete,
    DecodedStreamError,
    DecodedStreamEvent,
    DecodedTextDelta,
    DecodedToolBatchComplete,
    DecodedToolFragment,
    ModelCodec,
    StreamFraming,
    upstream_extra,
)


class OpenAIResponsesCodec(ModelCodec):
    """Codec for OpenAI Responses API (/v1/responses)."""

    protocol = UpstreamProtocol.OPENAI_RESPONSES

    def endpoint_path(self) -> str:
        return "/v1/responses"

    def probe_endpoint(self) -> str:
        return "/v1/models"

    def required_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def render_request(
        self,
        canonical: CanonicalRequest,
        model_name: str,
        stream: bool = True,
    ) -> dict[str, Any]:
        return render_canonical_to_responses(canonical, model_name, stream)

    def decode_response(
        self,
        body: dict[str, Any],
        tools: list[CanonicalTool] | None = None,
    ) -> DecodedModelResponse:
        content: list[CanonicalContentBlock] = []
        candidates: list[RawToolCallCandidate] = []

        for idx, item in enumerate(body.get("output", [])):
            item_type = item.get("type", "")

            if item_type == "message":
                for subblock in item.get("content", []):
                    if subblock.get("type") == "output_text":
                        content.append(CanonicalTextBlock(text=subblock.get("text", "")))

            elif item_type == "function_call":
                candidates.append(RawToolCallCandidate(
                    id=item.get("id"),
                    name=item.get("name"),
                    raw_arguments=item.get("arguments", "{}"),
                    source_protocol=ProtocolKind.OPENAI_RESPONSES,
                    source_index=idx,
                ))

            elif item_type == "reasoning":
                # Reasoning content — extract text if present
                text = item.get("summary", [{}])[0].get("text", "") if item.get("summary") else ""
                if text:
                    content.append(CanonicalTextBlock(text=text))

            else:
                # Preserve unknown output item types (item 68)
                from agent_interop.abi import CanonicalUnknownBlock
                content.append(CanonicalUnknownBlock(
                    source_type=f"responses_{item_type}",
                    raw=item,
                ))

        extra = {"response_id": body.get("id", "")}
        extra.update(upstream_extra(body))
        return DecodedModelResponse(
            content=content,
            tool_candidates=candidates,
            stop_reason=self.extract_stop_reason(body),
            usage=self.extract_usage(body),
            extra=extra,
        )

    def decode_stream_chunk(
        self,
        chunk: dict[str, Any],
    ) -> list[DecodedStreamEvent]:
        """Decode an OpenAI Responses streaming event.

        Supports the full Responses event lifecycle:
            response.output_text.delta → DecodedTextDelta
            response.function_call_arguments.delta → DecodedToolFragment
            response.function_call_arguments.done → DecodedToolBatchComplete
            response.completed → DecodedStreamComplete
            response.incomplete → DecodedStreamComplete (max_tokens)
            response.failed → DecodedStreamError
        """
        events: list[DecodedStreamEvent] = []
        etype = chunk.get("type", "")

        if etype == "response.output_text.delta":
            events.append(DecodedTextDelta(text=chunk.get("delta", "")))

        elif etype == "response.function_call_arguments.delta":
            events.append(DecodedToolFragment(
                choice_index=0,
                tool_index=chunk.get("output_index", 0),
                call_id_fragment=chunk.get("call_id", ""),
                argument_fragment=chunk.get("delta", ""),
            ))

        elif etype == "response.function_call_arguments.done":
            events.append(DecodedToolBatchComplete(
                choice_index=chunk.get("output_index", 0),
                stop_reason=CanonicalStopReason.TOOL_CALL,
            ))

        elif etype == "response.completed":
            usage_data = chunk.get("response", {}).get("usage", {})
            events.append(DecodedStreamComplete(
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                ) if usage_data else None,
            ))

        elif etype == "response.incomplete":
            events.append(DecodedStreamComplete(
                stop_reason=CanonicalStopReason.MAX_TOKENS,
            ))

        elif etype == "response.failed":
            events.append(DecodedStreamError(
                error=chunk.get("error", {}).get("message", str(chunk)),
            ))

        return events

    def extract_usage(self, body: dict[str, Any]) -> CanonicalUsage:
        raw = body.get("usage", {})
        return CanonicalUsage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            total_tokens=raw.get("total_tokens", 0),
            confidence="backend-reported",
        )

    def extract_stop_reason(self, body: dict[str, Any]) -> CanonicalStopReason:
        from agent_interop.abi import CanonicalStopReason
        status = body.get("status", "completed")
        status_map = {
            "completed": CanonicalStopReason.END_TURN,
            "incomplete": CanonicalStopReason.MAX_TOKENS,
        }
        return status_map.get(status, CanonicalStopReason.END_TURN)

    def is_stream_complete(self, chunk: dict[str, Any]) -> bool:
        etype = chunk.get("type", "")
        return etype in ("response.completed", "response.incomplete", "response.failed")

    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            supports_native_tools=True,
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            supports_system_messages=False,  # Uses "instructions" instead
            max_tools=128,
            streaming_framing=StreamFraming.SSE,
        )


# ─── Legacy render function for gateway backward compatibility ────────────


def render_canonical_to_responses(
    canonical: CanonicalRequest,
    model_name: str,
    stream: bool = True,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "stream": stream,
    }

    if canonical.system:
        body["instructions"] = _render_instructions(canonical.system)

    if canonical.generation.max_output_tokens:
        body["max_output_tokens"] = canonical.generation.max_output_tokens

    if canonical.generation.temperature is not None:
        body["temperature"] = canonical.generation.temperature

    if canonical.generation.top_p is not None:
        body["top_p"] = canonical.generation.top_p

    body["input"] = _render_input(canonical.messages)

    # Use the canonical field, not metadata (single source of truth)
    prev_id = canonical.previous_response_id
    if prev_id:
        body["previous_response_id"] = prev_id

    if canonical.tools:
        body["tools"] = [_render_responses_tool(t) for t in canonical.tools]

    body["tool_choice"] = _render_tool_choice_responses(canonical.tool_choice)

    text_config = canonical.metadata.get("response_format", canonical.metadata.get("text", {}))
    if text_config:
        body["text"] = text_config

    meta = canonical.metadata.get("metadata", {})
    if meta:
        body["metadata"] = meta

    return body


def _render_instructions(system: list[CanonicalContentBlock]) -> str:
    if isinstance(system, str):
        return system
    parts: list[str] = []
    for block in system:
        if isinstance(block, CanonicalTextBlock) and block.text:
            parts.append(block.text)
    return "\n".join(parts)


def _render_input(messages: list[CanonicalMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.role

        if role == "tool":
            for block in msg.content:
                if isinstance(block, CanonicalToolResultBlock):
                    output = block.content if isinstance(block.content, str) else json.dumps(block.content)
                    item: dict[str, Any] = {
                        "type": "function_call_output",
                        "output": output,
                    }
                    if block.tool_call_id:
                        item["call_id"] = block.tool_call_id
                    if block.is_error:
                        item["status"] = "error"
                    items.append(item)
                elif isinstance(block, CanonicalTextBlock):
                    items.append({
                        "type": "function_call_output",
                        "output": block.text,
                    })
            continue

        if role == "developer":
            role = "user"

        if role in ("user", "assistant"):
            content_blocks: list[dict[str, Any]] = []

            for block in msg.content:
                if isinstance(block, CanonicalTextBlock) and block.text:
                    content_type = "input_text" if role == "user" else "output_text"
                    content_blocks.append({"type": content_type, "text": block.text})
                elif isinstance(block, CanonicalToolCallBlock):
                    items.append({
                        "type": "function_call",
                        "id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments) if isinstance(block.arguments, dict) else str(block.arguments),
                    })
                elif isinstance(block, CanonicalToolResultBlock):
                    output = block.content if isinstance(block.content, str) else json.dumps(block.content)
                    fc_item: dict[str, Any] = {
                        "type": "function_call_output",
                        "output": output,
                    }
                    if block.tool_call_id:
                        fc_item["call_id"] = block.tool_call_id
                    if block.is_error:
                        fc_item["status"] = "error"
                    items.append(fc_item)
                elif isinstance(block, CanonicalReasoningBlock) and block.content and role == "assistant":
                    content_blocks.append({"type": "output_text", "text": block.content})

            if content_blocks:
                items.append({
                    "type": "message",
                    "role": role,
                    "content": content_blocks,
                })
            elif role == "user":
                text = _extract_text(msg.content)
                if text:
                    items.append({
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    })

    return items


def _render_responses_tool(tool: CanonicalTool) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": tool.strict,
    }


def _render_tool_choice_responses(tc: CanonicalToolChoice) -> str | dict[str, Any]:
    mode = tc.mode
    mode_str = mode.value if isinstance(mode, ToolChoiceMode) else str(mode)

    if mode_str in ("auto", ToolChoiceMode.AUTO.value):
        return "auto"
    if mode_str in ("none", ToolChoiceMode.NONE.value):
        return "none"
    if mode_str in ("required", ToolChoiceMode.REQUIRED.value):
        return "required"
    if mode_str in ("named", ToolChoiceMode.NAMED.value):
        return {"type": "function", "name": tc.name}
    return "auto"


def _extract_text(content: list[CanonicalContentBlock]) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, CanonicalTextBlock) and block.text:
            parts.append(block.text)
        elif isinstance(block, CanonicalToolResultBlock) and block.content:
            if isinstance(block.content, str):
                parts.append(block.content)
    return "\n".join(parts)