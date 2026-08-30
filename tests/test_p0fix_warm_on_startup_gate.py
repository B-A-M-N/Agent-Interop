"""P1.11 (review #30): warm_on_startup gate honored by the gateway.

Operators who turn off ``resources.warm_on_startup=False`` must NOT see
runtime-metadata warm-up probes in ``startup()``.  The pre-fix code
ignored the gate and always warmed.
"""

from __future__ import annotations

import asyncio

from agent_interop.config import (
    ContextConfig,
    InteropServerConfig,
    ModelRoute,
    RuntimeInspectionConfig,
    ToolMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.gateway import Gateway


def _route() -> ModelRoute:
    return ModelRoute(
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


def _gateway(*, warm_on_startup: bool) -> Gateway:
    return Gateway(InteropServerConfig(
        probe_on_startup=False,
        log_level="error",
        routes={"r": _route()},
        runtime_inspection=RuntimeInspectionConfig(
            mode="cached_metadata", warm_on_startup=warm_on_startup,
        ),
    ))


def test_warm_on_startup_true_calls_inspect():
    """Default: warm_on_startup=True → metadata warm-up runs."""
    gw = _gateway(warm_on_startup=True)
    calls = {"n": 0}

    async def fake_inspect(route):
        calls["n"] += 1
        from agent_interop.backends.base import ModelRuntimeCapabilities
        return ModelRuntimeCapabilities()

    gw._inspect_model_runtime = fake_inspect  # type: ignore[method-assign]
    asyncio.run(gw._warm_runtime_metadata())
    assert calls["n"] == 1


def test_warm_on_startup_false_skips_inspect():
    """Operator-disabled warm-up must NOT call inspect_model_runtime."""
    gw = _gateway(warm_on_startup=False)
    calls = {"n": 0}

    async def fake_inspect(route):
        calls["n"] += 1
        from agent_interop.backends.base import ModelRuntimeCapabilities
        return ModelRuntimeCapabilities()

    gw._inspect_model_runtime = fake_inspect  # type: ignore[method-assign]
    asyncio.run(gw._warm_runtime_metadata())
    assert calls["n"] == 0, (
        "warm_on_startup=False must skip the inspect_model_runtime calls"
    )


def test_warm_on_startup_default_is_true_for_backwards_compat():
    """The default value of warm_on_startup must be True (warm) so existing
    deployments do not see a behavior change."""
    from agent_interop.config import RuntimeInspectionConfig
    rc = RuntimeInspectionConfig()
    assert rc.warm_on_startup is True