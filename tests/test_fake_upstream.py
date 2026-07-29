"""Tests for fake upstream backend."""
import pytest

from agent_interop.abi import (
    CanonicalGenerationOptions,
    CanonicalMessage,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalTextBlock,
    CanonicalToolChoice,
)
from agent_interop.testing.fake_upstream import (
    FakeResponseTemplate,
    FakeUpstream,
    make_text,
    make_tool_call,
)


@pytest.mark.asyncio
async def test_fake_upstream_text_response():
    upstream = FakeUpstream()
    upstream.set_response(FakeResponseTemplate(text="Hello, world!"))
    canonical = _make_canonical("Hello")
    body = await upstream.handle_request(canonical)
    assert body["choices"][0]["message"]["content"] == "Hello, world!"
    assert body["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_fake_upstream_tool_call():
    upstream = FakeUpstream()
    upstream.set_response(make_tool_call("read_file", {"path": "/tmp/x"}))
    canonical = _make_canonical("Read file pls")
    body = await upstream.handle_request(canonical)
    msg = body["choices"][0]["message"]
    assert msg["content"] is None
    assert len(msg["tool_calls"]) == 1
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"


@pytest.mark.asyncio
async def test_fake_upstream_sequential():
    upstream = FakeUpstream()
    upstream.set_sequential_responses([
        make_tool_call("read_file", {"path": "/a"}),
        make_text("Done."),
    ])
    c1 = _make_canonical("Read file")
    b1 = await upstream.handle_request(c1)
    assert b1["choices"][0]["finish_reason"] == "tool_calls"

    c2 = _make_canonical("Read file")
    b2 = await upstream.handle_request(c2)
    assert b2["choices"][0]["message"]["content"] == "Done."


@pytest.mark.asyncio
async def test_fake_upstream_tool_response():
    upstream = FakeUpstream()
    upstream.set_tool_response("read_file", make_text("File content here."))
    canonical = _make_canonical("Read file", role="tool", tool_call_id="tc_001")
    body = await upstream.handle_request(canonical)
    assert body["choices"][0]["message"]["content"] == "File content here."


@pytest.mark.asyncio
async def test_fake_upstream_streaming():
    upstream = FakeUpstream()
    upstream.set_response(make_tool_call("read_file", {"path": "/tmp/x"}))
    canonical = _make_canonical("Read file pls")
    chunks = []
    async for chunk in upstream.handle_stream(canonical):
        if isinstance(chunk, dict):
            chunks.append(chunk)

    assert len(chunks) >= 2  # tool call delta + finish


@pytest.mark.asyncio
async def test_fake_upstream_default_text():
    upstream = FakeUpstream()
    canonical = _make_canonical("Hello")
    body = await upstream.handle_request(canonical)
    assert body["choices"][0]["message"]["content"] == "This is a fake response."


@pytest.mark.asyncio
async def test_fake_upstream_reset():
    upstream = FakeUpstream()
    upstream.set_response(make_text("First"))
    upstream.set_response(make_text("Second"))
    upstream.reset()
    canonical = _make_canonical("Hello")
    body = await upstream.handle_request(canonical)
    # After reset, uses default
    assert body["choices"][0]["message"]["content"] == "This is a fake response."


def _make_canonical(
    text: str = "",
    role: str = "user",
    tool_call_id: str | None = None,
) -> CanonicalRequest:
    msg = CanonicalMessage(
        role=role,
        content=[CanonicalTextBlock(text=text)],
    )
    if tool_call_id:
        msg.tool_call_id = tool_call_id  # type: ignore[attr-defined]
    return CanonicalRequest(
        model=CanonicalModelReference(requested_name="fake-model"),
        messages=[msg],
        generation=CanonicalGenerationOptions(max_output_tokens=100),
        tool_choice=CanonicalToolChoice.auto(),
    )