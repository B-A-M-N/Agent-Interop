"""Tests for provider metadata round-tripping in protocol adapters.

Provider metadata must survive a decode/encode round trip for fields
like ``reasoning_content`` so that provider-specific replay-critical
state is not silently lost.
"""

from __future__ import annotations

from agent_interop.abi import (
    CanonicalReasoningBlock,
)
from agent_interop.protocols.openai_chat import OpenAIChatAdapter


class TestProviderMetadataRoundTrip:
    def test_reasoning_content_preserved(self):
        adapter = OpenAIChatAdapter()
        body = {
            "model": "qwen3",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "secret-chain",
                },
            ],
        }
        req = adapter.decode_request(body, {})
        # Find the reasoning block in the assistant message.
        asst = next(m for m in req.messages if m.role == "assistant")
        reasoning_blocks = [
            b for b in asst.content if isinstance(b, CanonicalReasoningBlock)
        ]
        assert reasoning_blocks, "expected a ReasoningBlock"
        rb = reasoning_blocks[0]
        assert rb.content == "secret-chain"
        assert rb.provider_metadata is not None
        assert rb.provider_metadata.origin_protocol == "openai_chat"
        assert rb.provider_metadata.required_for_replay is True
        assert rb.provider_metadata.opaque_value == "secret-chain"
