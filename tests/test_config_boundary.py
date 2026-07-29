"""REVISION #4: validate_config enforced at the actual construction
boundary, not just at CLI-level call sites (deploy/check).

Before this fix, `create_app()`/`Gateway.__init__` never called
validate_config at all — only CLI commands did, before ever constructing
either object. An embedding application (or any other direct caller)
constructing `create_app(my_config)` or `Gateway(my_config)` itself bypassed
validation entirely. These tests call both constructors DIRECTLY (no CLI
involved) with a config carrying a known-invalid field, and assert both
raise immediately.
"""

from __future__ import annotations

import pytest

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairConfig,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    validate_config,
)
from agent_interop.gateway import Gateway
from agent_interop.server.app import create_app


def _valid_config() -> InteropServerConfig:
    return InteropServerConfig(
        host="127.0.0.1",
        port=0,
        probe_on_startup=False,
        default_route_id="r",
        routes={
            "r": ModelRoute(
                id="r",
                client_model_aliases=["m"],
                upstream_model="m",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:11434",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )


def _config_with_negative_max_regenerations() -> InteropServerConfig:
    config = _valid_config()
    config.routes["r"].repair = RepairConfig(max_regenerations=-1)
    return config


def _config_with_no_routes() -> InteropServerConfig:
    return InteropServerConfig(probe_on_startup=False, routes={})


class TestSanityCheckOwnFixture:
    def test_valid_config_actually_passes_validate_config(self) -> None:
        """Guards against this file's own 'valid' fixture silently drifting
        out of sync with validate_config and making every other test here
        meaningless."""
        assert validate_config(_valid_config()) == []


class TestGatewayConstructionBoundary:
    def test_gateway_rejects_negative_max_regenerations(self) -> None:
        with pytest.raises(ValueError, match="Invalid InteropServerConfig"):
            Gateway(_config_with_negative_max_regenerations())

    def test_gateway_rejects_no_routes(self) -> None:
        with pytest.raises(ValueError, match="Invalid InteropServerConfig"):
            Gateway(_config_with_no_routes())

    def test_gateway_accepts_valid_config(self) -> None:
        Gateway(_valid_config())  # must not raise

    def test_gateway_allow_invalid_config_escape_hatch_bypasses_check(self) -> None:
        """The escape hatch exists for tests that intentionally probe
        invalid-config behavior further downstream (e.g. startup()'s own
        redundant check) — it must still let construction succeed."""
        Gateway(_config_with_no_routes(), allow_invalid_config=True)  # must not raise


class TestCreateAppConstructionBoundary:
    def test_create_app_rejects_negative_max_regenerations(self) -> None:
        with pytest.raises(ValueError, match="Invalid InteropServerConfig"):
            create_app(_config_with_negative_max_regenerations())

    def test_create_app_rejects_no_routes(self) -> None:
        with pytest.raises(ValueError, match="Invalid InteropServerConfig"):
            create_app(_config_with_no_routes())

    def test_create_app_accepts_valid_config(self) -> None:
        create_app(_valid_config())  # must not raise

    def test_create_app_allow_invalid_escape_hatch_bypasses_check(self) -> None:
        create_app(_config_with_no_routes(), allow_invalid=True)  # must not raise
