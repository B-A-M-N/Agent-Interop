"""OpenAI Chat upstream codec — /v1/chat/completions.

Implements ModelCodec for the OpenAI Chat Completions protocol used by
vLLM, llama.cpp, and most OpenAI-compatible backends.
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
    DecodedUsageUpdate,
    ModelCodec,
    StreamFraming,
    upstream_extra,
)


class OpenAIChatCodec(ModelCodec):
    """Codec for OpenAI Chat Completions (/v1/chat/completions)."""

    protocol = UpstreamProtocol.OPENAI_CHAT

    def endpoint_path(self) -> str:
        return "/v1/chat/completions"

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
        return render_canonical_to_chat(canonical, model_name, stream)

    def decode_response(
        self,
        body: dict[str, Any],
        tools: list[CanonicalTool] | None = None,
    ) -> DecodedModelResponse:
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message", {})
        text = msg.get("content", "")

        content: list[CanonicalContentBlock] = []

        # Preserve unknown content from provider extensions (item 68)
        # e.g., audio, refusal blocks not in standard text/reasoning
        for key in ("audio", "refusal"):
            if msg.get(key):
                from agent_interop.abi import CanonicalUnknownBlock
                content.append(CanonicalUnknownBlock(
                    source_type=f"openai_{key}",
                    raw=msg[key],
                ))

        # Capture reasoning_content from upstream (o1, DeepSeek, etc.)
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        if reasoning and isinstance(reasoning, str):
            from agent_interop.abi import MetadataForwardingPolicy, ProviderMetadata
            content.append(CanonicalReasoningBlock(
                content=reasoning,
                provider_metadata=ProviderMetadata(
                    origin_protocol="openai_chat",
                    origin_provider="",
                    origin_model="",
                    metadata_kind="reasoning_content",
                    opaque_value=reasoning,
                    required_for_replay=True,
                    forwarding_policy=MetadataForwardingPolicy.PRESERVE_IF_COMPATIBLE,
                ),
            ))

        if text:
            content.append(CanonicalTextBlock(text=text))

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

        extra = {"response_id": body.get("id", "")}
        extra.update(upstream_extra(body))
        return DecodedModelResponse(
            content=content,
            tool_candidates=candidates,
            stop_reason=self.extract_stop_reason(choice),
            usage=self.extract_usage(body),
            extra=extra,
        )

    def decode_stream_chunk(
        self,
        chunk: dict[str, Any],
    ) -> list[DecodedStreamEvent]:
        events: list[DecodedStreamEvent] = []

        # Process top-level usage before checking choices.
        # Some providers send a terminal chunk with usage but empty choices.
        top_usage = chunk.get("usage")
        if top_usage:
            events.append(DecodedUsageUpdate(
                usage=CanonicalUsage(
                    input_tokens=top_usage.get("prompt_tokens", 0),
                    output_tokens=top_usage.get("completion_tokens", 0),
                    total_tokens=top_usage.get("total_tokens", 0),
                ),
            ))

        choices = chunk.get("choices", [])

        # Process ALL choices, not just choices[0]
        for choice in choices:
            choice_index = choice.get("index", 0)
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")
            usage = choice.get("usage")
            text = delta.get("content", "")

            if text:
                events.append(DecodedTextDelta(text=text))

            for raw_tc in delta.get("tool_calls") or []:
                tool_index = raw_tc.get("index", 0)
                fn = raw_tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                tc_id = raw_tc.get("id", "")

                # Emit fragment with full identity (id/name/args)
                events.append(DecodedToolFragment(
                    choice_index=choice_index,
                    tool_index=tool_index,
                    call_id_fragment=tc_id,
                    name_fragment=name,
                    argument_fragment=args if isinstance(args, str) else json.dumps(args) if args else "",
                ))

            # Choice-level completion driven by provider finish_reason
            if finish_reason == "tool_calls":
                events.append(DecodedToolBatchComplete(
                    choice_index=choice_index,
                    stop_reason=CanonicalStopReason.TOOL_CALL,
                ))
            elif finish_reason == "stop":
                events.append(DecodedStreamComplete(
                    stop_reason=CanonicalStopReason.END_TURN,
                ))
            elif finish_reason == "length":
                events.append(DecodedStreamComplete(
                    stop_reason=CanonicalStopReason.MAX_TOKENS,
                ))

            # Capture streamed usage
            if usage:
                events.append(DecodedUsageUpdate(
                    usage=CanonicalUsage(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                    ),
                ))

        return events

    def extract_usage(self, body: dict[str, Any]) -> CanonicalUsage:
        raw = body.get("usage", {})
        return CanonicalUsage(
            input_tokens=raw.get("prompt_tokens", 0),
            output_tokens=raw.get("completion_tokens", 0),
            total_tokens=raw.get("total_tokens", 0),
            confidence="backend-reported",
        )

    def extract_stop_reason(self, body: dict[str, Any]) -> CanonicalStopReason:
        from agent_interop.abi import CanonicalStopReason
        finish = body.get("finish_reason", "stop")
        stop_map = {
            "stop": CanonicalStopReason.END_TURN,
            "tool_calls": CanonicalStopReason.TOOL_CALL,
            "length": CanonicalStopReason.MAX_TOKENS,
        }
        return stop_map.get(finish, CanonicalStopReason.END_TURN)

    def is_stream_complete(self, chunk: dict[str, Any]) -> bool:
        if chunk.get("done", False):
            return True
        # Terminal usage-only chunk (choices empty, usage at top level)
        if chunk.get("usage") and not chunk.get("choices"):
            return True
        choices = chunk.get("choices", [])
        if choices:
            return bool(choices[0].get("finish_reason"))
        return False

    def capabilities(self) -> CodecCapabilities:
        return CodecCapabilities(
            supports_native_tools=True,
            supports_streaming=True,
            supports_parallel_tool_calls=True,
            supports_system_messages=True,
            max_tools=128,
            streaming_framing=StreamFraming.SSE,
        )

    def backend_constraints(self):
        """Return destination constraints for OpenAI-compatible backends."""
        import re

        from agent_interop.request_validation import BackendConstraints
        return BackendConstraints(
            name_pattern=re.compile(r"^[a-zA-Z0-9_-]+$"),
            max_name_length=64,
            max_tools=128,
            supports_parallel=True,
            supports_strict_schema=False,
        )


# ─── Legacy render function for gateway backward compatibility ────────────


def render_canonical_to_chat(
    canonical: CanonicalRequest,
    model_name: str,
    stream: bool = True,
) -> dict[str, Any]:
    """Render a canonical request to OpenAI Chat Completions format."""
    messages: list[dict[str, Any]] = []

    if canonical.system:
        messages.append({
            "role": "system",
            "content": _render_system(canonical.system),
        })

    for msg in canonical.messages:
        messages.append(_render_message(msg))

    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": stream,
        "max_tokens": canonical.generation.max_output_tokens,
    }

    if canonical.generation.temperature is not None:
        body["temperature"] = canonical.generation.temperature

    if canonical.generation.top_p is not None:
        body["top_p"] = canonical.generation.top_p

    if canonical.generation.stop:
        body["stop"] = canonical.generation.stop

    if canonical.tools:
        body["tools"] = [_render_tool(t) for t in canonical.tools]
        body["tool_choice"] = _render_tool_choice(canonical.tool_choice)

    return body


def _render_system(blocks: list[CanonicalContentBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, CanonicalTextBlock) and block.text:
            parts.append(block.text)
    return "\n".join(parts)


def _render_message(msg: CanonicalMessage) -> dict[str, Any]:
    role = msg.role
    if role == "developer":
        role = "system"

    if role == "tool":
        tool_call_id = ""
        content = ""
        for block in msg.content:
            if isinstance(block, CanonicalToolResultBlock):
                tool_call_id = block.tool_call_id
                content = block.content if isinstance(block.content, str) else str(block.content)
                break
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }

    if role == "assistant":
        result: dict[str, Any] = {"role": "assistant"}
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        for block in msg.content:
            if isinstance(block, CanonicalTextBlock):
                content_parts.append(block.text)
            elif isinstance(block, CanonicalReasoningBlock):
                # Only forward reasoning_content if the metadata permits
                meta = getattr(block, "provider_metadata", None)
                if meta and meta.forwarding_policy == "drop":
                    continue
                reasoning_parts.append(block.content)
            elif isinstance(block, CanonicalToolCallBlock):
                # Preserve raw arguments when available (historical malformed calls)
                # to avoid corrupting conversation history by replacing with {}
                if block.raw_arguments is not None and not block.arguments_validated:
                    args_str = block.raw_arguments if isinstance(block.raw_arguments, str) else json.dumps(block.raw_arguments, ensure_ascii=False, default=str)
                else:
                    args_str = json.dumps(block.arguments, ensure_ascii=False, default=str)
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": args_str,
                    },
                })
        result["content"] = "".join(content_parts) or None
        if reasoning_parts:
            result["reasoning_content"] = "\n".join(reasoning_parts)
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    content = _render_content(msg.content)
    return {
        "role": role,
        "content": content if content else "",
    }


def _render_content(content: list[CanonicalContentBlock]) -> str:
    parts: list[str] = []
    for block in content:
        if isinstance(block, CanonicalTextBlock) and block.text:
            parts.append(block.text)
        elif isinstance(block, CanonicalToolResultBlock) and block.content:
            if isinstance(block.content, str):
                parts.append(block.content)
    return "\n".join(parts)


def _render_tool(tool: CanonicalTool) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
            "strict": tool.strict,
        },
    }


def _render_tool_choice(tc: CanonicalToolChoice) -> str | dict[str, Any]:
    mode = tc.mode
    if isinstance(mode, str):
        mode_str = mode
    else:
        mode_str = mode.value if hasattr(mode, "value") else "auto"

    if mode_str == "auto":
        return "auto"
    if mode_str == "none":
        return "none"
    if mode_str == "required":
        return "required"
    if mode_str == "named":
        return {"type": "function", "function": {"name": tc.name}}
    return "auto"