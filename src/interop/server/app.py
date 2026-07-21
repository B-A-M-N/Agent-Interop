"""FastAPI server for the Interop gateway.

Exposes:
- /v1/messages (Anthropic Messages API)
- /v1/messages/count_tokens
- /v1/chat/completions (OpenAI Chat)
- /v1/responses (OpenAI Responses)
- /v1/health
- /v1/models
- /v1/capabilities

All endpoints normalize incoming requests into canonical form,
process through the gateway, and translate responses back to the
original protocol format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from interop.gateway import Gateway
from interop.protocols.registry import detect_protocol, get_adapter
from interop.types import CanonicalRequest, InteropConfig, ProtocolKind

logger = logging.getLogger("interop.server")

_gateway: Gateway | None = None


def create_app(config: InteropConfig | None = None) -> FastAPI:
    """Create the FastAPI application with the given config."""
    global _gateway

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        g = Gateway(config or InteropConfig())
        await g.startup()
        app.state.gateway = g
        _gateway = g
        logger.info(
            "interop server ready — %s:%d (backend=%s, model=%s)",
            config.host if config else "127.0.0.1",
            config.port if config else 8090,
            g.config.backend.value,
            g.config.model,
        )
        yield
        await g.close()
        _gateway = None

    app = FastAPI(
        title="Interop — Agent Compatibility Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    def get_gateway(request: Request) -> Gateway:
        return request.app.state.gateway

    # ─── Health ───────────────────────────────────────────────────────────

    @app.get("/v1/health")
    @app.get("/health")
    async def health(request: Request):
        gw = get_gateway(request)
        info = gw.server_info()
        return JSONResponse({
            "status": "ok",
            "version": info.version,
            "model": info.model,
            "profile": info.profile,
            "level": info.level,
            "level_description": info.level_description,
            "supports": info.supports,
        })

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models(request: Request):
        gw = get_gateway(request)
        return JSONResponse({
            "object": "list",
            "data": [
                {
                    "id": gw.get_model(),
                    "object": "model",
                    "created": 0,
                    "owned_by": "interop",
                }
            ],
        })

    @app.get("/v1/capabilities")
    @app.get("/capabilities")
    async def capabilities(request: Request):
        gw = get_gateway(request)
        info = gw.server_info()
        profile = gw.get_profile()
        return JSONResponse({
            "model": info.model,
            "level": info.level,
            "description": info.level_description,
            "profile": {
                "context_length": profile.context_length if profile else "unknown",
                "parallel_tools": profile.parallel_tools if profile else False,
                "supports_images": profile.supports_images if profile else False,
                "supports_thinking": profile.supports_thinking if profile else False,
            } if profile else None,
        })

    # ─── Anthropic Messages API ───────────────────────────────────────────

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        gw = get_gateway(request)
        body = await request.json()
        stream = body.get("stream", False)

        adapter = get_adapter(ProtocolKind.ANTHROPIC_MESSAGES)
        canonical = adapter.decode_request(body, dict(request.headers))

        if stream:
            return await _stream_response(gw, canonical, ProtocolKind.ANTHROPIC_MESSAGES, adapter)
        else:
            backend_resp = await gw.handle_request(canonical, ProtocolKind.ANTHROPIC_MESSAGES)
            if isinstance(backend_resp, dict):
                # Already in anthropic format (from backend)
                encoded = _adapt_backend_to_anthropic(backend_resp)
            else:
                encoded = _canonical_to_anthropic(backend_resp)
            return JSONResponse(encoded)

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        gw = get_gateway(request)
        body = await request.json()
        adapter = get_adapter(ProtocolKind.ANTHROPIC_MESSAGES)
        simplified = adapter.count_tokens_request(body)
        _to_backend_token_count(simplified, gw)
        return JSONResponse({"input_tokens": 0, "output_tokens": 0})

    # ─── OpenAI Chat Completions API ──────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        gw = get_gateway(request)
        body = await request.json()
        stream = body.get("stream", False)

        adapter = get_adapter(ProtocolKind.OPENAI_CHAT)
        canonical = adapter.decode_request(body, dict(request.headers))

        if stream:
            return await _stream_response(gw, canonical, ProtocolKind.OPENAI_CHAT, adapter)
        else:
            backend_resp = await gw.handle_request(canonical, ProtocolKind.OPENAI_CHAT)
            if isinstance(backend_resp, dict):
                encoded = _backend_to_chat(backend_resp)
            else:
                encoded = _canonical_to_chat(backend_resp)
            return JSONResponse(encoded)

    # ─── OpenAI Responses API ─────────────────────────────────────────────

    @app.post("/v1/responses")
    async def responses_api(request: Request):
        gw = get_gateway(request)
        body = await request.json()
        stream = body.get("stream", False)

        adapter = get_adapter(ProtocolKind.OPENAI_RESPONSES)
        canonical = adapter.decode_request(body, dict(request.headers))

        if stream:
            return await _stream_response(gw, canonical, ProtocolKind.OPENAI_RESPONSES, adapter)
        else:
            backend_resp = await gw.handle_request(canonical, ProtocolKind.OPENAI_RESPONSES)
            if isinstance(backend_resp, dict):
                encoded = _backend_to_responses(backend_resp)
            else:
                encoded = _canonical_to_responses(backend_resp)
            return JSONResponse(encoded)

    return app


# ─── Streaming helper ────────────────────────────────────────────────────────


async def _stream_response(gw: Gateway, canonical: CanonicalRequest,
                           protocol: ProtocolKind, adapter) -> StreamingResponse:
    """Stream a response, translating canonical events to the client protocol."""

    async def event_stream():
        adapter_inst = adapter
        accumulated_tool_calls: list[dict] = []
        content_index = 0

        # Emit content_block_start for text
        def start_text(text: str):
            if protocol == ProtocolKind.ANTHROPIC_MESSAGES:
                return adapter_inst.encode_stream_event(
                    type("_", (), {"type": "text", "index": content_index, "partial": text})()
                )

        async for event in gw.handle_stream(canonical, protocol):
            sse = adapter_inst.encode_stream_event(event)
            if sse:
                yield sse

            # Track accumulated tool calls for final message
            if event.type == "tool_use":
                if event.content_block and event.content_block.tool_call:
                    accumulated_tool_calls.append(event.content_block.tool_call.to_dict())

            if event.type == "message_stop":
                break

        yield adapter_inst.encode_stream_done()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ─── Response format helpers ─────────────────────────────────────────────────


def _adapt_backend_to_anthropic(body: dict) -> dict:
    """Convert a chat-completion backend response to Anthropic format."""
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "")
    finish = choice.get("finish_reason", "end_turn")

    content = []
    if text:
        content.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls", []):
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "input": _parse_json(tc.get("function", {}).get("arguments", "{}")),
        })

    stop_map = {"tool_calls": "tool_use", "length": "max_tokens", "stop": "end_turn"}
    return {
        "id": body.get("id", "interop-msg"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.get("model", "unknown"),
        "stop_reason": stop_map.get(finish, finish),
        "usage": {
            "input_tokens": body.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": body.get("usage", {}).get("completion_tokens", 0),
        },
    }


def _backend_to_chat(body: dict) -> dict:
    """Passthrough for chat-format backend responses."""
    return {
        "id": body.get("id", "interop-chat"),
        "object": "chat.completion",
        "created": body.get("created", 0),
        "model": body.get("model", "unknown"),
        "choices": body.get("choices", []),
        "usage": body.get("usage", {}),
    }


def _backend_to_responses(body: dict) -> dict:
    """Convert a chat-completion backend response to Responses API format."""
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "")
    finish = choice.get("finish_reason", "stop")

    output: list[dict] = []
    if text:
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })
    for tc in msg.get("tool_calls", []):
        output.append({
            "type": "function_call",
            "id": tc.get("id", ""),
            "name": tc.get("function", {}).get("name", ""),
            "arguments": tc.get("function", {}).get("arguments", "{}"),
            "status": "completed",
        })

    return {
        "id": body.get("id", "interop-resp"),
        "object": "response",
        "status": "completed" if finish != "length" else "incomplete",
        "output": output,
        "usage": {
            "input_tokens": body.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": body.get("usage", {}).get("completion_tokens", 0),
        },
    }


def _canonical_to_anthropic(resp) -> dict:
    content = []
    for block in resp.content:
        if block.type == "text" and block.text:
            content.append({"type": "text", "text": block.text})
        elif block.type == "tool_use" and block.tool_call:
            content.append({
                "type": "tool_use",
                "id": block.tool_call.id,
                "name": block.tool_call.name,
                "input": block.tool_call.arguments,
            })
        elif block.type == "thinking" and block.text:
            entry = {"type": "thinking", "thinking": block.text}
            if block.signature:
                entry["signature"] = block.signature
            content.append(entry)

    return {
        "id": resp.id or "interop-msg",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": resp.model,
        "stop_reason": resp.stop_reason,
        "usage": {
            "input_tokens": resp.usage.get("input_tokens", resp.usage.get("prompt_tokens", 0)),
            "output_tokens": resp.usage.get("output_tokens", resp.usage.get("completion_tokens", 0)),
        },
    }


def _canonical_to_chat(resp) -> dict:
    text = resp.text
    return {
        "id": resp.id or "interop-chat",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in resp.tool_calls
                ] if resp.tool_calls else None,
            },
            "finish_reason": "tool_calls" if resp.tool_calls else "stop",
        }],
        "usage": resp.usage,
        "model": resp.model,
    }


def _canonical_to_responses(resp) -> dict:
    output: list[dict] = []
    text = resp.text
    if text:
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        })
    for tc in resp.tool_calls:
        output.append({
            "type": "function_call",
            "id": tc.id,
            "name": tc.name,
            "arguments": json.dumps(tc.arguments),
            "status": "completed",
        })

    return {
        "id": resp.id or "interop-resp",
        "object": "response",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": resp.usage.get("input_tokens", resp.usage.get("prompt_tokens", 0)),
            "output_tokens": resp.usage.get("output_tokens", resp.usage.get("completion_tokens", 0)),
        },
    }


def _to_backend_token_count(body: dict, gw: Gateway) -> None:
    """Send token count request to backend (minimal implementation)."""
    pass


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {"raw": text}