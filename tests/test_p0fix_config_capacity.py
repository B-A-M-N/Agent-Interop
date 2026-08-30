"""P0.19 (review finding #14): unknown_capacity_policy operator control.

Before this fix, ResourceConfig had no policy for the case where a
request's context capacity could not be established — it silently
meant 8K tokens.  This adds operator-controlled fields and wires them
through the YAML/dict loader.
"""

from __future__ import annotations

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ResourceConfig,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    load_config_from_dict,
    validate_config,
)

_VALID_ROUTE = ModelRoute(
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
)


def _minimal_server_config(rc: ResourceConfig) -> InteropServerConfig:
    """Build an InteropServerConfig with one minimal route and the given ResourceConfig.

    validate_config returns early when config.routes is empty, so every
    validate_config call must have at least one route.
    """
    return InteropServerConfig(
        default_route_id="r",
        routes={"r": _VALID_ROUTE},
        resources=rc,
    )


class TestResourceConfigDefaults:
    """Verify the new fields exist and carry their documented defaults."""

    def test_default_unknown_capacity_policy_is_reject(self) -> None:
        rc = ResourceConfig()
        assert rc.unknown_capacity_policy == "reject"

    def test_default_unknown_capacity_fallback_tokens(self) -> None:
        rc = ResourceConfig()
        assert rc.unknown_capacity_fallback_tokens == 8192


class TestValidateConfigUnknownCapacity:
    """validate_config must reject invalid values for the new fields."""

    def test_invalid_policy_rejected(self) -> None:
        """Policy values outside {reject, fallback} must produce an issue."""
        issues = validate_config(
            _minimal_server_config(ResourceConfig(unknown_capacity_policy="guess"))
        )
        matching = [i for i in issues if "unknown_capacity_policy" in i]
        assert len(matching) == 1, f"Expected one issue about unknown_capacity_policy, got: {matching}"

    def test_invalid_policy_rejected_values(self) -> None:
        """Each invalid value should be rejected."""
        for bad_value in ("guess", "ignore", "warn", ""):
            issues = validate_config(
                _minimal_server_config(ResourceConfig(unknown_capacity_policy=bad_value))
            )
            assert any("unknown_capacity_policy" in i for i in issues), (
                f"Expected validation to reject policy={bad_value!r}"
            )

    def test_valid_policy_values_pass(self) -> None:
        for good_value in ("reject", "fallback"):
            issues = validate_config(
                _minimal_server_config(ResourceConfig(unknown_capacity_policy=good_value))
            )
            policy_issues = [i for i in issues if "unknown_capacity_policy" in i]
            assert policy_issues == [], (
                f"Expected no issues for policy={good_value!r}, got: {policy_issues}"
            )

    def test_fallback_tokens_zero_rejected(self) -> None:
        issues = validate_config(
            _minimal_server_config(ResourceConfig(unknown_capacity_fallback_tokens=0))
        )
        assert any("unknown_capacity_fallback_tokens" in i for i in issues), (
            f"Expected validation to reject fallback_tokens=0, got: {issues}"
        )

    def test_fallback_tokens_negative_rejected(self) -> None:
        issues = validate_config(
            _minimal_server_config(ResourceConfig(unknown_capacity_fallback_tokens=-100))
        )
        assert any("unknown_capacity_fallback_tokens" in i for i in issues), (
            f"Expected validation to reject fallback_tokens=-100, got: {issues}"
        )

    def test_fallback_tokens_over_cap_rejected(self) -> None:
        issues = validate_config(
            _minimal_server_config(ResourceConfig(unknown_capacity_fallback_tokens=2_000_000))
        )
        assert any("unknown_capacity_fallback_tokens" in i for i in issues), (
            f"Expected validation to reject fallback_tokens=2_000_000, got: {issues}"
        )

    def test_fallback_tokens_positive_passes(self) -> None:
        """Any value in (0, 1_000_000] should pass."""
        for good_value in (1, 8192, 100_000, 1_000_000):
            issues = validate_config(
                _minimal_server_config(ResourceConfig(unknown_capacity_fallback_tokens=good_value))
            )
            token_issues = [i for i in issues if "unknown_capacity_fallback_tokens" in i]
            assert token_issues == [], (
                f"Expected no issues for fallback_tokens={good_value}, got: {token_issues}"
            )


class TestLoadConfigFromDict:
    """The dict/YAML loader must round-trip both new fields."""

    def test_defaults_when_resources_missing(self) -> None:
        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {"kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                },
            },
        })
        assert config.resources.unknown_capacity_policy == "reject"
        assert config.resources.unknown_capacity_fallback_tokens == 8192

    def test_defaults_when_resources_empty(self) -> None:
        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {"kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                },
            },
            "resources": {},
        })
        assert config.resources.unknown_capacity_policy == "reject"
        assert config.resources.unknown_capacity_fallback_tokens == 8192

    def test_roundtrip_both_fields(self) -> None:
        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {"kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                },
            },
            "resources": {
                "unknown_capacity_policy": "fallback",
                "unknown_capacity_fallback_tokens": 4096,
            },
        })
        assert config.resources.unknown_capacity_policy == "fallback"
        assert config.resources.unknown_capacity_fallback_tokens == 4096

    def test_roundtrip_reject_policy_explicit(self) -> None:
        config = load_config_from_dict({
            "routes": {
                "default": {
                    "aliases": ["test-model"],
                    "upstream": {"kind": "ollama", "base_url": "http://127.0.0.1:11434"},
                },
            },
            "resources": {
                "unknown_capacity_policy": "reject",
                "unknown_capacity_fallback_tokens": 16384,
            },
        })
        assert config.resources.unknown_capacity_policy == "reject"
        assert config.resources.unknown_capacity_fallback_tokens == 16384


class TestPolicyFallbackValidates:
    """Policy 'fallback' must validate without issues."""

    def test_fallback_policy_passes_validation(self) -> None:
        issues = validate_config(
            _minimal_server_config(ResourceConfig(unknown_capacity_policy="fallback"))
        )
        policy_issues = [i for i in issues if "unknown_capacity_policy" in i]
        assert policy_issues == [], (
            f"Policy 'fallback' should not produce validation issues: {issues}"
        )
