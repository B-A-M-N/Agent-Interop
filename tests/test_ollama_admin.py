"""Tests for OllamaAdminClient (interop.backends.ollama_admin).

This module previously had 0% test coverage — genuinely never imported by
any test in the suite. It's explicitly NOT wired into any live gateway
request path (see the module's own docstring: "salvaged... currently
unreachable"), but it's still real, shipped code an operator or external
caller can import and use directly, and every method was completely
unverified.

Since each method opens its own httpx.AsyncClient() internally (no
injectable transport), these tests monkeypatch interop.backends.
ollama_admin.httpx.AsyncClient with a fake that records the request and
returns a canned response — proving each method hits the right endpoint,
with the right method/body, and parses the response shape correctly.
"""

from __future__ import annotations

import json

import pytest

from agent_interop.backends.ollama_admin import OllamaAdminClient

BASE_URL = "http://127.0.0.1:11434"


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncClient:
    """Records every call made through it; returns the queued response(s)."""

    calls: list[dict] = []
    _response: _FakeResponse | None = None
    _stream_response: _FakeStreamResponse | None = None

    def __init__(self, *a, **k) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, **kwargs):
        _FakeAsyncClient.calls.append({"method": "GET", "url": url, **kwargs})
        return _FakeAsyncClient._response

    async def post(self, url, **kwargs):
        _FakeAsyncClient.calls.append({"method": "POST", "url": url, **kwargs})
        return _FakeAsyncClient._response

    async def request(self, method, url, **kwargs):
        _FakeAsyncClient.calls.append({"method": method, "url": url, **kwargs})
        return _FakeAsyncClient._response

    def stream(self, method, url, **kwargs):
        _FakeAsyncClient.calls.append({"method": method, "url": url, **kwargs})
        return _FakeAsyncClient._stream_response


@pytest.fixture(autouse=True)
def _patch_client(monkeypatch):
    _FakeAsyncClient.calls = []
    _FakeAsyncClient._response = None
    _FakeAsyncClient._stream_response = None
    monkeypatch.setattr("agent_interop.backends.ollama_admin.httpx.AsyncClient", _FakeAsyncClient)
    yield


def _set_response(payload: dict) -> None:
    _FakeAsyncClient._response = _FakeResponse(payload)


def _set_stream(lines: list[str]) -> None:
    _FakeAsyncClient._stream_response = _FakeStreamResponse(lines)


def _run(coro):
    import asyncio
    return asyncio.run(coro)


async def _collect(agen):
    return [item async for item in agen]


class TestDiscovery:
    def test_list_models(self):
        _set_response({"models": [{"name": "qwen3-coder:latest"}]})
        client = OllamaAdminClient()
        result = _run(client.list_models(BASE_URL))
        assert result == [{"name": "qwen3-coder:latest"}]
        assert _FakeAsyncClient.calls[0]["url"] == f"{BASE_URL}/api/tags"
        assert _FakeAsyncClient.calls[0]["method"] == "GET"

    def test_list_models_missing_key_returns_empty(self):
        _set_response({})
        client = OllamaAdminClient()
        assert _run(client.list_models(BASE_URL)) == []

    def test_show_model(self):
        _set_response({"modelfile": "FROM qwen3-coder", "parameters": "temperature 0.7"})
        client = OllamaAdminClient()
        result = _run(client.show_model(BASE_URL, "qwen3-coder"))
        assert result["modelfile"] == "FROM qwen3-coder"
        call = _FakeAsyncClient.calls[0]
        assert call["url"] == f"{BASE_URL}/api/show"
        assert call["method"] == "POST"
        assert call["json"] == {"model": "qwen3-coder"}

    def test_ps(self):
        _set_response({"models": [{"name": "qwen3-coder", "size": 4_000_000}]})
        client = OllamaAdminClient()
        result = _run(client.ps(BASE_URL))
        assert result == [{"name": "qwen3-coder", "size": 4_000_000}]
        assert _FakeAsyncClient.calls[0]["url"] == f"{BASE_URL}/api/ps"

    def test_ps_missing_key_returns_empty(self):
        _set_response({})
        client = OllamaAdminClient()
        assert _run(client.ps(BASE_URL)) == []


class TestModelManagement:
    def test_pull_model_streams_progress(self):
        _set_stream([
            json.dumps({"status": "pulling manifest"}),
            json.dumps({"status": "success"}),
        ])
        client = OllamaAdminClient()
        events = _run(_collect(client.pull_model(BASE_URL, "qwen3-coder")))
        assert events == [{"status": "pulling manifest"}, {"status": "success"}]
        call = _FakeAsyncClient.calls[0]
        assert call["url"] == f"{BASE_URL}/api/pull"
        assert call["method"] == "POST"
        assert call["json"] == {"model": "qwen3-coder", "stream": True}

    def test_pull_model_skips_blank_lines(self):
        _set_stream(["", json.dumps({"status": "success"}), "   "])
        client = OllamaAdminClient()
        events = _run(_collect(client.pull_model(BASE_URL, "m")))
        assert events == [{"status": "success"}]

    def test_pull_model_non_streaming_flag_still_sent(self):
        _set_stream([json.dumps({"status": "success"})])
        client = OllamaAdminClient()
        _run(_collect(client.pull_model(BASE_URL, "m", stream=False)))
        assert _FakeAsyncClient.calls[0]["json"]["stream"] is False

    def test_push_model(self):
        _set_stream([json.dumps({"status": "pushing"})])
        client = OllamaAdminClient()
        events = _run(_collect(client.push_model(BASE_URL, "m")))
        assert events == [{"status": "pushing"}]
        assert _FakeAsyncClient.calls[0]["url"] == f"{BASE_URL}/api/push"

    def test_create_model_with_modelfile_and_path(self):
        _set_stream([json.dumps({"status": "creating"})])
        client = OllamaAdminClient()
        events = _run(_collect(client.create_model(
            BASE_URL, "custom-model", modelfile="FROM base", path="/tmp/Modelfile",
        )))
        assert events == [{"status": "creating"}]
        body = _FakeAsyncClient.calls[0]["json"]
        assert body["model"] == "custom-model"
        assert body["modelfile"] == "FROM base"
        assert body["path"] == "/tmp/Modelfile"

    def test_create_model_without_modelfile_or_path_omits_keys(self):
        _set_stream([json.dumps({"status": "creating"})])
        client = OllamaAdminClient()
        _run(_collect(client.create_model(BASE_URL, "m")))
        body = _FakeAsyncClient.calls[0]["json"]
        assert "modelfile" not in body
        assert "path" not in body

    def test_delete_model(self):
        _set_response({})
        client = OllamaAdminClient()
        _run(client.delete_model(BASE_URL, "old-model"))
        call = _FakeAsyncClient.calls[0]
        assert call["method"] == "DELETE"
        assert call["url"] == f"{BASE_URL}/api/delete"
        assert call["json"] == {"model": "old-model"}

    def test_copy_model(self):
        _set_response({})
        client = OllamaAdminClient()
        _run(client.copy_model(BASE_URL, "source-model", "dest-model"))
        call = _FakeAsyncClient.calls[0]
        assert call["url"] == f"{BASE_URL}/api/copy"
        assert call["json"] == {"source": "source-model", "destination": "dest-model"}


class TestEmbeddings:
    def test_embed_single_string(self):
        _set_response({"embeddings": [[0.1, 0.2, 0.3]]})
        client = OllamaAdminClient()
        result = _run(client.embed(BASE_URL, "embed-model", "hello world"))
        assert result == [[0.1, 0.2, 0.3]]
        call = _FakeAsyncClient.calls[0]
        assert call["url"] == f"{BASE_URL}/api/embed"
        assert call["json"] == {"model": "embed-model", "input": "hello world"}

    def test_embed_list_of_strings(self):
        _set_response({"embeddings": [[0.1], [0.2]]})
        client = OllamaAdminClient()
        result = _run(client.embed(BASE_URL, "embed-model", ["a", "b"]))
        assert result == [[0.1], [0.2]]
        assert _FakeAsyncClient.calls[0]["json"]["input"] == ["a", "b"]

    def test_embed_missing_key_returns_empty(self):
        _set_response({})
        client = OllamaAdminClient()
        assert _run(client.embed(BASE_URL, "m", "x")) == []

    def test_legacy_embeddings(self):
        _set_response({"embedding": [0.5, 0.6]})
        client = OllamaAdminClient()
        result = _run(client.embeddings(BASE_URL, "embed-model", "hello"))
        assert result == [0.5, 0.6]
        call = _FakeAsyncClient.calls[0]
        assert call["url"] == f"{BASE_URL}/api/embeddings"
        assert call["json"] == {"model": "embed-model", "prompt": "hello"}

    def test_legacy_embeddings_missing_key_returns_empty(self):
        _set_response({})
        client = OllamaAdminClient()
        assert _run(client.embeddings(BASE_URL, "m", "x")) == []
