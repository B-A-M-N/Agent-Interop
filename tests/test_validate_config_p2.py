"""P2 fix: validate_config gained two previously-missing checks:

- log_level must be a recognized logging-level name (previously a typo
  silently fell back to INFO via getattr(..., logging.INFO) in
  cli.py's _configure_process_logging, with no config-level signal at all).
- max_keepalive_connections must not exceed max_connections (a keepalive
  pool larger than the total pool is nonsensical).

(Alias/route-id collision checks already existed prior to this fix —
see validate_config's existing "Check alias collisions" block.)
"""

from __future__ import annotations

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    validate_config,
)


def _config(**overrides) -> InteropServerConfig:
    base = {
        "host": "127.0.0.1",
        "port": 0,
        "probe_on_startup": False,
        "default_route_id": "r",
        "routes": {
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
    }
    base.update(overrides)
    return InteropServerConfig(**base)


class TestLogLevelValidation:
    def test_valid_log_levels_pass(self):
        for level in ("debug", "info", "warning", "warn", "error", "critical", "DEBUG", "Info"):
            issues = validate_config(_config(log_level=level))
            assert not any("log_level" in i for i in issues), (level, issues)

    def test_invalid_log_level_flagged(self):
        issues = validate_config(_config(log_level="verbose"))
        assert any("log_level" in i and "verbose" in i for i in issues)

    def test_typo_log_level_flagged(self):
        """The exact real-world case: a typo silently became INFO with no
        signal anywhere before this fix."""
        issues = validate_config(_config(log_level="infor"))
        assert any("log_level" in i for i in issues)


class TestKeepaliveNotExceedingMaxConnections:
    def test_keepalive_within_bounds_passes(self):
        issues = validate_config(_config(max_connections=100, max_keepalive_connections=20))
        assert not any("keepalive" in i.lower() for i in issues)

    def test_keepalive_equal_to_max_passes(self):
        issues = validate_config(_config(max_connections=20, max_keepalive_connections=20))
        assert not any("keepalive" in i.lower() for i in issues)

    def test_keepalive_exceeding_max_flagged(self):
        issues = validate_config(_config(max_connections=10, max_keepalive_connections=50))
        assert any("keepalive" in i.lower() and "max_connections" in i for i in issues)
