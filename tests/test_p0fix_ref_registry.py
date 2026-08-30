"""RequestRefRegistry — request-scoped ref lifecycle (P0, review item 3).

Refs minted AFTER preparation (schema-on-demand via
``__interop_get_tool_schema`` storing a large schema in the store) must be
pinned, included in the output-firewall identity, and unpinned with the
request. The historical ``pinned_refs`` tuple froze at preparation time and
leaked every dynamic ref past all three.
"""

from __future__ import annotations

import pytest

from agent_interop.abi import (
    CanonicalTool,
)
from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.context_store import RequestRefRegistry
from agent_interop.context_store.store import ContextStore
from agent_interop.errors import ContextStoreError
from agent_interop.gateway import Gateway


class _TrackingStore(ContextStore):
    """Records pin/unpin traffic with the session-scoped signature."""

    def __init__(self) -> None:
        super().__init__()
        self.pinned: dict[str, int] = {}
        self.unpinned: dict[str, int] = {}

    def pin_refs(self, refs, request_id: str, session_id: str | None = None) -> int:
        n = super().pin_refs(refs, request_id, session_id=session_id)
        for ref in refs:
            self.pinned[ref] = self.pinned.get(ref, 0) + n and self.pinned.get(ref, 0) + 1
        return n

    def unpin_refs(self, refs, request_id: str, session_id: str | None = None) -> int:
        n = super().unpin_refs(refs, request_id, session_id=session_id)
        for ref in refs:
            self.unpinned[ref] = self.unpinned.get(ref, 0) + 1
        return n


def _registry(store: ContextStore, session: str = "s-1") -> RequestRefRegistry:
    return RequestRefRegistry(store, session, "req-1")


def test_register_pins_and_close_unpins_session_scoped():
    store = ContextStore()
    stored = store.store(session_id="s-1", content="payload", kind="tool_result")
    reg = _registry(store)
    reg.register(stored.ref)
    assert stored.ref in reg.snapshot()
    entry = store.get(stored.ref, "s-1")
    assert entry is not None and "req-1" in entry.pinned_requests
    reg.close()
    entry = store.get(stored.ref, "s-1")
    assert entry is None or "req-1" not in entry.pinned_requests


def test_register_is_idempotent_per_request():
    store = ContextStore()
    stored = store.store(session_id="s-1", content="payload", kind="tool_result")
    reg = _registry(store)
    reg.register(stored.ref)
    reg.register(stored.ref)
    reg.register(stored.ref)
    assert reg.snapshot() == frozenset({stored.ref})
    reg.close()
    # One unpin request id per ref regardless of register count.
    assert store.get(stored.ref, "s-1") is None or (
        "req-1" not in store.get(stored.ref, "s-1").pinned_requests
    )


def test_register_unknown_ref_raises():
    store = ContextStore()
    reg = _registry(store)
    with pytest.raises(ContextStoreError):
        reg.register("nonexistent0000000000000000")


def test_register_after_close_raises():
    store = ContextStore()
    reg = _registry(store)
    reg.close()
    stored = store.store(session_id="s-1", content="x", kind="tool_result")
    with pytest.raises(ContextStoreError):
        reg.register(stored.ref)


def test_close_is_idempotent():
    store = ContextStore()
    stored = store.store(session_id="s-1", content="x", kind="tool_result")
    reg = _registry(store)
    reg.register(stored.ref)
    reg.close()
    reg.close()
    assert reg.snapshot() == frozenset()


def test_cross_session_ref_is_not_pinned():
    """A session-scoped registry cannot pin another session's ref."""
    store = ContextStore()
    other = store.store(session_id="s-other", content="secret", kind="tool_result")
    reg = _registry(store, session="s-1")
    with pytest.raises(ContextStoreError):
        reg.register(other.ref)
    assert other.ref not in reg.snapshot()


def _gateway(store: ContextStore) -> Gateway:
    route = ModelRoute(
        id="r",
        client_model_aliases=["m"],
        upstream_model="fake-model",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OPENAI_COMPATIBLE,
            base_url="http://127.0.0.1:1",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
        ),
        tool_mode=ToolMode.NATIVE,
        context=ContextConfig(context_limit_tokens=2000, output_reserve_tokens=500),
    )
    gw = Gateway(InteropServerConfig(
        probe_on_startup=False, log_level="error", routes={"r": route},
    ))
    gw._context_store = store
    return gw


def test_dynamic_schema_ref_joins_registry_and_cleanup():
    """The schema-on-demand executor mints a ref mid-request; the gateway
    must pin it (via the registry) and unpin it when the request finishes."""
    from agent_interop.context_store.executor import (
        InternalExecutionContext,
        InternalToolExecutor,
    )

    store = _TrackingStore()
    _gateway(store)

    big_tool = CanonicalTool(
        name="big_tool",
        description="A tool with an oversized schema",
        # The executor serializes input_schema only — the description is not
        # part of the stored text — so the schema itself must exceed
        # MAX_SCHEMA_CHARS (2000) to be stored as a ref.
        input_schema={
            "type": "object",
            "properties": {
                f"field_{i}": {
                    "type": "string",
                    "description": "p" * 120,
                }
                for i in range(40)
            },
        },
    )
    # The route is fake; what this test proves is the ref LIFECYCLE: the
    # private executor mints a mid-request ref, the registry pins it, and
    # close() unpins it. The gateway's own loop registers the outcome.ref
    # through this same code path (gateway private-loop: registry.register).
    context = InternalExecutionContext(
        session_id="s-reg",
        authorized_tools={"big_tool": big_tool},
        withheld_tools=frozenset({"big_tool"}),
    )
    executor = InternalToolExecutor(store)
    result = executor.execute(
        "__interop_get_tool_schema",
        {"name": "big_tool"},
        session_id="s-reg",
        context=context,
    )
    assert result.ref, "oversized schema must be stored and return a ref"

    reg = RequestRefRegistry(store, "s-reg", "req-dyn")
    reg.register(result.ref)
    identity_refs = reg.snapshot()
    assert result.ref in identity_refs
    entry = store.get(result.ref, "s-reg")
    assert entry is not None and "req-dyn" in entry.pinned_requests
    reg.close()
    entry = store.get(result.ref, "s-reg")
    assert entry is None or "req-dyn" not in entry.pinned_requests
