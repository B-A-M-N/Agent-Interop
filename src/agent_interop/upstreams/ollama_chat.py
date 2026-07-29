"""Ollama native chat upstream codec — ModelCodec for /api/chat.

Renders canonical ABI to Ollama /api/chat format and decodes responses.
Ollama uses a Messages-like JSON format with optional tools in the
OpenAI function-calling style, and streams via NDJSON.
"""

from __future__ import annotations

import json
from typing import Any

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalImageBlock,
    CanonicalMessage,
    CanonicalReasoningBlock,
    CanonicalRefusalBlock,
    CanonicalRequest,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolResultBlock,
    CanonicalUnknownBlock,
    CanonicalUsage,
    ProtocolKind,
    RawToolCallCandidate,
)
from agent_interop.config import UpstreamProtocol
from agent_interop.upstreams.codec import (
    CodecCapabilities,
    DecodedModelResponse,
    DecodedStreamComplete,
    DecodedStreamEvent,
    DecodedTextDelta,
    DecodedToolBatchComplete,
    DecodedToolFragment,
    ModelCodec,
    StreamFraming,
    upstream_extra,
)


class OllamaChatCodec(ModelCodec):
    """Codec for Ollama /api/chat (OpenAI-compatible chat format over NDJSON)."""

    protocol = UpstreamProtocol.OLLAMA_CHAT
    stream_framing = StreamFraming.NDJSON

    def endpoint_path(self) -> str:
        return "/api/chat"

    def probe_endpoint(self) -> str:
        return "/api/tags"

    def required_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def render_request(
        self,
        canonical: CanonicalRequest,
        model_name: str,
        stream: bool = True,
    ) -> dict[str, Any]:
        return render_canonical_to_ollama(canonical, model_name, stream)

    def decode_response(
        self,
        body: dict[str, Any],
        tools: list[CanonicalTool] | None = None,
    ) -> DecodedModelResponse:
        msg = body.get("message", {})
        text = msg.get("content", "")

        content: list[CanonicalContentBlock] = []
        if text:
            content.append(CanonicalTextBlock(text=text))

        # Preserve unknown message-level content (item 68)
        if msg.get("images"):
            from agent_interop.abi import CanonicalUnknownBlock
            content.append(CanonicalUnknownBlock(
                source_type="ollama_images",
                raw=msg["images"],
            ))

        candidates: list[RawToolCallCandidate] = []
        for idx, raw_tc in enumerate(msg.get("tool_calls", [])):
            fn = raw_tc.get("function", {})
            raw_args = fn.get("arguments", {})
            candidate = RawToolCallCandidate(
                id=raw_tc.get("id"),
                name=fn.get("name"),
                raw_arguments=raw_args,
                source_protocol=ProtocolKind.OPENAI_CHAT,
                source_index=idx,
            )
            candidates.append(candidate)

        return DecodedModelResponse(
            content=content,
            tool_candidates=candidates,
            stop_reason=self.extract_stop_reason(body),
            usage=self.extract_usage(body),
            extra=upstream_extra(body, "model"),
        )

    def decode_stream_chunk(
        self,
        chunk: dict[str, Any],
    ) -> list[DecodedStreamEvent]:
        events: list[DecodedStreamEvent] = []
        is_done = chunk.get("done", False)
        msg = chunk.get("message", {})
        text = msg.get("content", "") if isinstance(msg, dict) else ""

        if text:
            events.append(DecodedTextDelta(text=text))

        # Accept tool-call fragments or complete calls from non-final messages
        tool_calls = msg.get("tool_calls", []) if isinstance(msg, dict) else []
        for idx, raw_tc in enumerate(tool_calls):
            fn = raw_tc.get("function", {})
            raw_args = fn.get("arguments")
            arguments_str = raw_args if isinstance(raw_args, str) else json.dumps(raw_args) if raw_args is not None else "{}"
            events.append(DecodedToolFragment(
                choice_index=0,
                tool_index=idx,
                call_id_fragment=raw_tc.get("id", ""),
                name_fragment=fn.get("name", ""),
                argument_fragment=arguments_str,
            ))

        # On the final done frame, complete the batch
        if is_done:
            done_reason = chunk.get("done_reason")
            if tool_calls or done_reason == "tool_calls":
                events.append(DecodedToolBatchComplete(
                    choice_index=0,
                    stop_reason=CanonicalStopReason.TOOL_CALL,
                ))
            else:
                stop_map = {
                    "stop": CanonicalStopReason.END_TURN,
                    "length": CanonicalStopReason.MAX_TOKENS,
                }
                events.append(DecodedStreamComplete(
                    stop_reason=stop_map.get(done_reason or "stop", CanonicalStopReason.END_TURN),
                ))

        return events

    def extract_usage(self, body: dict[str, Any]) -> CanonicalUsage:
        return CanonicalUsage(
            input_tokens=body.get("prompt_eval_count", 0),
            output_tokens=body.get("eval_count", 0),
            total_tokens=body.get("prompt_eval_count", 0) + body.get("eval_count", 0),
            confidence="backend-reported",
        )

    def extract_stop_reason(self, body: dict[str, Any]) -> CanonicalStopReason:
        from agent_interop.abi import CanonicalStopReason
        reason = body.get("done_reason", "stop")
        stop_map = {
            "stop": CanonicalStopReason.END_TURN,
            "tool_calls": CanonicalStopReason.TOOL_CALL,
            "length": CanonicalStopReason.MAX_TOKENS,
        }
        return stop_map.get(reason, CanonicalStopReason.END_TURN)

    def is_stream_complete(self, chunk: dict[str, Any]) -> bool:
        return bool(chunk.get("done", False))

    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            supports_native_tools=True,
            supports_streaming=True,
            supports_parallel_tool_calls=False,
            supports_vision=True,
            supports_system_messages=True,
            max_tools=64,
            streaming_framing=StreamFraming.NDJSON,
        )


# ─── Legacy render function for gateway backward compatibility ────────────


def render_canonical_to_ollama(
    canonical: CanonicalRequest,
    model_name: str,
    stream: bool = True,
) -> dict[str, Any]:
    """Render a canonical request to Ollama /api/chat format."""
    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []

    # Extract system content
    for block in canonical.system:
        if isinstance(block, CanonicalTextBlock) and block.text:
            system_parts.append(block.text)

    system_text = "\n".join(system_parts).strip()

    # Build messages array
    for msg in canonical.messages:
        ollama_msg = _render_ollama_message(msg)
        if ollama_msg:
            messages.append(ollama_msg)

    # Prepend system as first message if present
    if system_text:
        messages.insert(0, {"role": "system", "content": system_text})

    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "options": {},
    }

    if canonical.generation.temperature is not None:
        body["options"]["temperature"] = canonical.generation.temperature

    if canonical.generation.max_output_tokens:
        body["options"]["num_predict"] = canonical.generation.max_output_tokens

    if canonical.generation.top_p is not None:
        body["options"]["top_p"] = canonical.generation.top_p

    if canonical.generation.stop:
        body["options"]["stop"] = canonical.generation.stop

    # Ollama supports tools in the OpenAI function-calling style
    if canonical.tools:
        body["tools"] = [_render_ollama_tool(t) for t in canonical.tools]

    # Tool-choice semantics: Ollama doesn't support explicit tool_choice
    # in its native API. We signal required/named choice by removing
    # the tool array and injecting instructions into the system prompt,
    # which is handled by the InvocationPlan via PROMPTED mode.
    # The codec simply passes through whatever the plan provides.

    return body


def _render_ollama_message(msg: CanonicalMessage) -> dict[str, Any] | None:
    """Render a canonical message to Ollama format."""
    role = msg.role

    if role == "system":
        return {"role": "system", "content": _extract_text(msg.content)}

    if role == "assistant":
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in msg.content:
            if isinstance(block, CanonicalTextBlock):
                content_parts.append(block.text)
            elif isinstance(block, CanonicalToolCallBlock):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": block.arguments,
                    },
                })
            elif isinstance(block, CanonicalReasoningBlock) and block.content:
                content_parts.append(block.content)
            elif isinstance(block, CanonicalRefusalBlock) and block.refusal:
                content_parts.append(block.refusal)

        result: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    if role == "tool":
        for block in msg.content:
            if isinstance(block, CanonicalToolResultBlock):
                content = block.content if isinstance(block.content, str) else json.dumps(block.content)
                tool_msg: dict[str, Any] = {"role": "tool", "content": content}
                if block.tool_call_id:
                    tool_msg["tool_call_id"] = block.tool_call_id
                return tool_msg
        content = _extract_text(msg.content)
        return {"role": "tool", "content": content}

    user_parts: list[str] = []
    user_images: list[str] = []
    for block in msg.content:
        if isinstance(block, CanonicalTextBlock) and block.text:
            user_parts.append(block.text)
        elif isinstance(block, CanonicalReasoningBlock) and block.content:
            user_parts.append(block.content)
        elif isinstance(block, CanonicalImageBlock):
            if block.data:
                user_images.append(block.data)
            elif block.url:
                user_parts.append(f"[image: {block.url}]")
        elif isinstance(block, CanonicalRefusalBlock) and block.refusal:
            user_parts.append(block.refusal)
        elif isinstance(block, CanonicalUnknownBlock) and block.raw is not None:
            user_parts.append(json.dumps(block.raw) if not isinstance(block.raw, str) else block.raw)

    user_result: dict[str, Any] = {"role": role, "content": "".join(user_parts)}
    if user_images:
        user_result["images"] = user_images
    return user_result


def _render_ollama_tool(tool: CanonicalTool) -> dict[str, Any]:
    """Render a CanonicalTool to Ollama tool format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, CanonicalTextBlock) and block.text:
            parts.append(block.text)
        elif isinstance(block, CanonicalToolResultBlock) and block.content:
            c = block.content
            parts.append(c if isinstance(c, str) else json.dumps(c))
    return "\n".join(parts)