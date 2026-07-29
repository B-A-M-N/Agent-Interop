"""Fake upstream backend for testing — returns configurable responses.

Used by conformance tests to exercise the full Interop pipeline without
a real model backend. Supports both streaming and non-streaming modes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from agent_interop.abi import CanonicalRequest

# ─── Response Templates ───────────────────────────────────────────────────────


@dataclass
class FakeResponseTemplate:
    """Template for a fake upstream response.

    Can be static or dynamic (computed from the request).
    """

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 10, "output_tokens": 10})
    latency_ms: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# ─── Fake Upstream ────────────────────────────────────────────────────────────


class FakeUpstream:
    """A fake upstream backend that returns configurable responses.

    Usage:
        upstream = FakeUpstream()
        upstream.set_response("read_file", FakeResponseTemplate(
            text="", tool_calls=[{"function": {"name": "read_file", "arguments": {"path": "/tmp/x"}}}],
        ))
        body = await upstream.handle_request(canonical)
    """

    def __init__(self) -> None:
        self._responses: list[FakeResponseTemplate] = []
        self._tool_responses: dict[str, FakeResponseTemplate] = {}
        self._call_index = 0

    def set_response(self, template: FakeResponseTemplate) -> None:
        """Set the next response (queued)."""
        self._responses.append(template)

    def set_tool_response(self, tool_name: str, template: FakeResponseTemplate) -> None:
        """Set a response keyed by the tool name that was called."""
        self._tool_responses[tool_name] = template

    def set_sequential_responses(self, templates: list[FakeResponseTemplate]) -> None:
        """Set a sequence of responses for multi-turn tests."""
        self._responses = list(templates)

    def reset(self) -> None:
        self._responses.clear()
        self._tool_responses.clear()
        self._call_index = 0

    async def handle_request(
        self,
        canonical: CanonicalRequest,
        model_name: str = "fake-model",
    ) -> dict[str, Any]:
        """Handle a non-streaming request and return an OpenAI Chat-formatted body."""
        template = self._next_response(canonical)
        await self._simulate_latency(template.latency_ms)
        return self._build_openai_body(template, model_name)

    async def handle_stream(
        self,
        canonical: CanonicalRequest,
        model_name: str = "fake-model",
    ) -> AsyncIterator[dict[str, Any] | str]:
        """Handle a streaming request, yielding OpenAI Chat SSE chunks."""
        template = self._next_response(canonical)
        await self._simulate_latency(template.latency_ms)

        # Text content
        if template.text:
            yield {
                "choices": [{"delta": {"content": template.text}, "index": 0, "finish_reason": None}]
            }

        # Tool calls
        if template.has_tool_calls:
            for tc in template.tool_calls:
                fn = tc.get("function", {})
                yield {
                    "choices": [{
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": tc.get("id", "tc_fake_001"),
                                "type": "function",
                                "function": {
                                    "name": fn.get("name", ""),
                                    "arguments": json.dumps(fn.get("arguments", {})),
                                },
                            }]
                        },
                        "index": 0,
                        "finish_reason": None,
                    }]
                }

        # Finish
        yield {
            "choices": [{"delta": {}, "index": 0, "finish_reason": template.finish_reason}]
        }
        yield "[DONE]"

    def _next_response(self, canonical: CanonicalRequest) -> FakeResponseTemplate:
        """Get the next response, either from queue, tool map, or a text default."""
        # Check if queue has responses
        if self._call_index < len(self._responses):
            template = self._responses[self._call_index]
            self._call_index += 1
            return template

        # Check tool responses by looking at response-level tool registrations
        # Match based on the last assistant message's tool calls
        for msg in reversed(canonical.messages):
            if msg.role == "assistant":
                blocks = msg.content if isinstance(msg.content, list) else [msg.content]
                for block in blocks:
                    if isinstance(block, dict):
                        name = block.get("name", "")
                    else:
                        name = getattr(block, "name", "") if hasattr(block, "name") else ""
                    if name and name in self._tool_responses:
                        return self._tool_responses[name]
                break

        # Also check by tool_call_id on tool-role messages
        for msg in canonical.messages:
            if msg.role == "tool":
                call_id = getattr(msg, "tool_call_id", "") or ""
                if call_id and call_id in self._tool_responses:
                    return self._tool_responses[call_id]
                # Fallback: match any tool response
                if self._tool_responses:
                    return next(iter(self._tool_responses.values()))

        # Default text response
        return FakeResponseTemplate(text="This is a fake response.")

    async def _simulate_latency(self, ms: float) -> None:
        if ms > 0:
            await asyncio.sleep(ms / 1000)

    def _build_openai_body(self, template: FakeResponseTemplate, model_name: str) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": template.text or None}
        if template.has_tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.get("id", f"tc_fake_{i:03d}"),
                    "type": "function",
                    "function": tc.get("function", {"name": "", "arguments": "{}"}),
                }
                for i, tc in enumerate(template.tool_calls)
            ]
        return {
            "id": "fake-chat-response",
            "object": "chat.completion",
            "created": 0,
            "model": model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": template.finish_reason}],
            "usage": template.usage,
        }


# ─── Supporting types for conformance testing ─────────────────────────────────


def make_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    call_id: str | None = None,
    finish_reason: str = "tool_calls",
) -> FakeResponseTemplate:
    """Create a FakeResponseTemplate with exactly one tool call."""
    import uuid
    return FakeResponseTemplate(
        text="",
        tool_calls=[{
            "id": call_id or f"tc_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments or {},
            },
        }],
        finish_reason=finish_reason,
    )


def make_text(text: str, finish_reason: str = "stop") -> FakeResponseTemplate:
    """Create a FakeResponseTemplate with text content only."""
    return FakeResponseTemplate(text=text, finish_reason=finish_reason)


def make_tool_result_response(
    text: str = "Tool executed successfully.",
    finish_reason: str = "stop",
) -> FakeResponseTemplate:
    """Create a response after a tool result has been sent."""
    return FakeResponseTemplate(text=text, finish_reason=finish_reason)