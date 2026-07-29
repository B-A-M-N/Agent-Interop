"""Tests for translation mode policy.

Only ``CANONICAL`` is implemented.  ``RAW_PASSTHROUGH`` and
``REPAIR_AWARE_SAME_PROTOCOL`` are accepted by the enum for API
stability but must be rejected at config-validation time and at
request-dispatch time.
"""

from __future__ import annotations

import pytest

from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
    load_config_from_dict,
    validate_config,
)


def _build_route(mode: TranslationMode) -> ModelRoute:
    return ModelRoute(
        id=f"r-{mode.value}",
        client_model_aliases=["*"],
        upstream_model="qwen3",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://127.0.0.1:11434",
            wire_protocol=UpstreamProtocol.OPENAI_CHAT,
            timeout_seconds=10.0,
        ),
        translation_mode=mode,
    )


class TestTranslationModePolicy:
    def test_canonical_accepted(self):
        config = InteropServerConfig(routes={"r": _build_route(TranslationMode.CANONICAL)})
        issues = validate_config(config)
        assert not [i for i in issues if "translation_mode" in i]

    @pytest.mark.parametrize("mode", [
        TranslationMode.RAW_PASSTHROUGH,
        TranslationMode.REPAIR_AWARE_SAME_PROTOCOL,
    ])
    def test_non_canonical_rejected_in_validate_config(self, mode):
        config = InteropServerConfig(routes={"r": _build_route(mode)})
        issues = validate_config(config)
        assert any(
            "translation_mode" in i and "not implemented" in i for i in issues
        ), f"expected translation_mode rejection in {issues!r}"

    def test_loaded_yaml_with_raw_passthrough_is_rejected(self):
        data = {
            "routes": {
                "r": {
                    "client_model_aliases": ["*"],
                    "upstream_model": "qwen3",
                    "upstream": {
                        "base_url": "http://127.0.0.1:11434",
                        "wire_protocol": "openai_chat",
                    },
                    "translation_mode": "raw_passthrough",
                }
            }
        }
        config = load_config_from_dict(data)
        issues = validate_config(config)
        assert any("not implemented" in i for i in issues)
