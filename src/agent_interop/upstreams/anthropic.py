"""Anthropic upstream codec — ModelCodec for /v1/messages.

Renders canonical ABI to Anthropic Messages API format and decodes responses.
Handles content blocks (text, tool_use, tool_result, thinking), streaming SSE
events, cache-control headers, and anthropic-version headers.
"""

from __future__ import annotations

from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalImageBlock,
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
    DecodedUsageUpdate,
    ModelCodec,
    StreamFraming,
    upstream_extra,
)


class AnthropicCodec(ModelCodec):
    """Codec for Anthropic Messages API (/v1/messages)."""

    protocol = UpstreamProtocol.ANTHROPIC_MESSAGES

    def endpoint_path(self) -> str:
        return "/v1/messages"

    def probe_endpoint(self) -> str:
        return "/v1/messages"

    def required_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def render_request(
        self,
        canonical: CanonicalRequest,
        model_name: str,
        stream: bool = True,
    ) -> dict[str, Any]:
        return render_canonical_to_anthropic(canonical, model_name, stream)

    def decode_response(
        self,
        body: dict[str, Any],
        tools: list[CanonicalTool] | None = None,
    ) -> DecodedModelResponse:
        content: list[CanonicalContentBlock] = []
        candidates: list[RawToolCallCandidate] = []

        for idx, raw_block in enumerate(body.get("content", [])):
            btype = raw_block.get("type", "")
            if btype == "text":
                content.append(CanonicalTextBlock(text=raw_block.get("text", "")))
            elif btype == "tool_use":
                candidates.append(RawToolCallCandidate(
                    id=raw_block.get("id"),
                    name=raw_block.get("name"),
                    raw_arguments=raw_block.get("input", {}),
                    source_protocol=ProtocolKind.ANTHROPIC_MESSAGES,
                    source_index=idx,
                ))
            elif btype == "thinking":
                content.append(CanonicalReasoningBlock(
                    content=raw_block.get("thinking", ""),
                    signature=raw_block.get("signature"),
                ))
            else:
                # Preserve unknown content block types (item 68)
                from agent_interop.abi import CanonicalUnknownBlock
                content.append(CanonicalUnknownBlock(
                    source_type=btype,
                    raw=raw_block,
                ))

        # Map Anthropic stop_reason to CanonicalStopReason
        raw_stop = body.get("stop_reason", "end_turn")
        anthropic_stop_map = {
            "end_turn": CanonicalStopReason.END_TURN,
            "tool_use": CanonicalStopReason.TOOL_CALL,
            "max_tokens": CanonicalStopReason.MAX_TOKENS,
            "stop_sequence": CanonicalStopReason.STOP_SEQUENCE,
        }
        stop_reason = anthropic_stop_map.get(raw_stop, CanonicalStopReason.END_TURN)

        extra = {"response_id": body.get("id", "")}
        extra.update(upstream_extra(body, model_key="id"))
        return DecodedModelResponse(
            content=content,
            tool_candidates=candidates,
            stop_reason=stop_reason,
            usage=self.extract_usage(body),
            extra=extra,
        )

    def decode_stream_chunk(
        self,
        chunk: dict[str, Any],
    ) -> list[DecodedStreamEvent]:
        """Decode an Anthropic streaming SSE chunk.

        Uses the unified discriminated event contract:
            content_block_start(tool_use) → DecodedToolFragment (id+name)
            content_block_delta(input_json_delta) → DecodedToolFragment (args)
            content_block_stop → DecodedToolBatchComplete
            message_delta → update usage/stop metadata
            message_stop → DecodedStreamComplete
        """
        events: list[DecodedStreamEvent] = []
        etype = chunk.get("type", "")

        if etype == "content_block_delta":
            delta = chunk.get("delta", {})
            dtype = delta.get("type", "")
            if dtype == "text_delta":
                events.append(DecodedTextDelta(text=delta.get("text", "")))
            elif dtype == "input_json_delta":
                # Carry partial arguments as a fragment
                events.append(DecodedToolFragment(
                    choice_index=0,
                    tool_index=chunk.get("index", 0),
                    argument_fragment=delta.get("partial_json", ""),
                ))

        elif etype == "content_block_start":
            block = chunk.get("content_block", {})
            btype = block.get("type", "")
            idx = chunk.get("index", 0)
            if btype == "tool_use":
                # Carry ID and name from block start
                events.append(DecodedToolFragment(
                    choice_index=0,
                    tool_index=idx,
                    call_id_fragment=block.get("id", ""),
                    name_fragment=block.get("name", ""),
                ))

        elif etype == "content_block_stop":
            # Complete the tool candidate at this block index
            idx = chunk.get("index", 0)
            events.append(DecodedToolBatchComplete(
                choice_index=0,
                stop_reason=CanonicalStopReason.TOOL_CALL,
            ))

        elif etype == "message_delta":
            delta = chunk.get("delta", {})
            stop_reason = delta.get("stop_reason")
            usage = chunk.get("usage")
            if usage:
                events.append(DecodedUsageUpdate(
                    usage=CanonicalUsage(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    ),
                ))
            if stop_reason:
                anthropic_stop_map = {
                    "end_turn": CanonicalStopReason.END_TURN,
                    "tool_use": CanonicalStopReason.TOOL_CALL,
                    "max_tokens": CanonicalStopReason.MAX_TOKENS,
                    "stop_sequence": CanonicalStopReason.STOP_SEQUENCE,
                }
                events.append(DecodedStreamComplete(
                    stop_reason=anthropic_stop_map.get(stop_reason, CanonicalStopReason.END_TURN),
                ))

        elif etype == "message_stop":
            events.append(DecodedStreamComplete(stop_reason=CanonicalStopReason.END_TURN))

        elif etype == "error":
            events.append(DecodedStreamError(
                error=chunk.get("error", {}).get("message", str(chunk)),
            ))

        return events

    def extract_usage(self, body: dict[str, Any]) -> CanonicalUsage:
        raw = body.get("usage", {})
        return CanonicalUsage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            total_tokens=raw.get("input_tokens", 0) + raw.get("output_tokens", 0),
            confidence="backend-reported",
        )

    def is_stream_complete(self, chunk: dict[str, Any]) -> bool:
        etype = chunk.get("type", "")
        return etype in ("message_stop", "error")

    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            supports_native_tools=True,
            supports_streaming=True,
            supports_parallel_tool_calls=False,
            supports_vision=True,
            supports_system_messages=True,
            max_tools=200,
            streaming_framing=StreamFraming.SSE,
        )


# ─── Legacy render function for gateway backward compatibility ────────────


def render_canonical_to_anthropic(
    canonical: CanonicalRequest,
    model_name: str,
    stream: bool = True,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "max_tokens": canonical.generation.max_output_tokens,
        "stream": stream,
    }

    if canonical.system:
        body["system"] = _render_system(canonical.system)

    if canonical.generation.temperature is not None:
        body["temperature"] = canonical.generation.temperature

    if canonical.generation.top_p is not None:
        body["top_p"] = canonical.generation.top_p

    if canonical.generation.stop:
        body["stop_sequences"] = canonical.generation.stop

    body["messages"] = [_render_anthropic_message(msg) for msg in canonical.messages]

    if canonical.tools:
        body["tools"] = [_render_anthropic_tool(t) for t in canonical.tools]

    body["tool_choice"] = _render_tool_choice(canonical.tool_choice)

    meta = canonical.metadata.get("metadata", {})
    if meta:
        body["metadata"] = meta

    return body


def _render_system(system: list[CanonicalContentBlock]) -> str | list[dict[str, Any]]:
    if isinstance(system, str):
        return system
    blocks: list[dict[str, Any]] = []
    for block in system:
        if isinstance(block, CanonicalTextBlock):
            b: dict[str, Any] = {"type": "text", "text": block.text}
            cache = getattr(block, "cache_control", None)
            if cache:
                b["cache_control"] = cache
            blocks.append(b)
    return blocks if blocks else str(system)


def _render_anthropic_message(msg: CanonicalMessage) -> dict[str, Any]:
    role = msg.role

    if role == "tool":
        content_parts: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, CanonicalToolResultBlock):
                content_parts.append({
                    "type": "tool_result",
                    "tool_use_id": block.tool_call_id,
                    "content": block.content,
                    "is_error": block.is_error,
                })
            elif isinstance(block, CanonicalTextBlock):
                content_parts.append({"type": "text", "text": block.text})
        if not content_parts:
            content_parts = [{"type": "text", "text": _extract_text(msg.content)}]
        return {"role": "user", "content": content_parts}

    if role == "developer":
        text = _extract_text(msg.content)
        return {"role": "user", "content": f"[developer]\n{text}"}

    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, CanonicalTextBlock):
                blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, CanonicalToolCallBlock):
                blocks.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                })
            elif isinstance(block, CanonicalReasoningBlock):
                tb: dict[str, Any] = {"type": "thinking", "thinking": block.content}
                if block.signature:
                    tb["signature"] = block.signature
                blocks.append(tb)
        if not blocks:
            text = _extract_text(msg.content)
            if text:
                blocks = [{"type": "text", "text": text}]
        return {"role": "assistant", "content": blocks}

    if role == "user":
        blocks = _render_user_content(msg.content)
        if len(blocks) == 1 and blocks[0].get("type") == "text":
            return {"role": "user", "content": blocks[0]["text"]}
        return {"role": "user", "content": blocks}

    text = _extract_text(msg.content)
    return {"role": "user", "content": text or "..."}


def _render_user_content(content: list[CanonicalContentBlock]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, CanonicalTextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, CanonicalToolResultBlock):
            blocks.append({
                "type": "tool_result",
                "tool_use_id": block.tool_call_id,
                "content": block.content,
                "is_error": block.is_error,
            })
        elif isinstance(block, CanonicalImageBlock):
            if block.data:
                source: dict[str, Any] = {"type": "base64", "media_type": block.media_type, "data": block.data}
            else:
                source = {"type": "url", "url": block.url}
            blocks.append({"type": "image", "source": source})
        else:
            blocks.append({"type": getattr(block, "type", "unknown"), **block.__dict__})
    return blocks


def _render_anthropic_tool(tool: CanonicalTool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _render_tool_choice(tc: CanonicalToolChoice) -> dict[str, Any]:
    mode = tc.mode
    mode_str = mode.value if isinstance(mode, ToolChoiceMode) else str(mode)

    if mode_str in ("auto", ToolChoiceMode.AUTO.value):
        return {"type": "auto"}
    if mode_str in ("none", ToolChoiceMode.NONE.value):
        return {"type": "none"}
    if mode_str in ("required", ToolChoiceMode.REQUIRED.value):
        return {"type": "any"}
    if mode_str in ("named", ToolChoiceMode.NAMED.value):
        return {"type": "tool", "name": tc.name}
    return {"type": "auto"}


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