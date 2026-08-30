"""P0.42: TokenMeter is calibrated from real backend usage.

Each completed generation must feed the backend's actual input token count
back into the gateway's TokenMeter, keyed by model digest, so the next
request's budget uses an exact count instead of a conservative estimate.
"""

import json

from agent_interop.abi import CanonicalUsage
from agent_interop.backends.base import ModelRuntimeCapabilities
from agent_interop.config import InteropServerConfig, UpstreamKind
from agent_interop.context_budget.meter import TokenMeter
from agent_interop.gateway import Gateway


def _make_gateway():
    config = InteropServerConfig(
        routes={},
        probe_on_startup=False,
        log_level="error",
    )
    return Gateway(config, allow_invalid_config=True)


def test_gateway_constructs_token_meter():
    gw = _make_gateway()
    assert isinstance(gw._token_meter, TokenMeter)


def test_calibrate_token_meter_updates_estimate():
    gw = _make_gateway()
    digest = "abc123"
    # Build a minimal ResolvedInvocation-like object with runtime_capabilities
    caps = ModelRuntimeCapabilities(
        backend_kind=UpstreamKind.OLLAMA,
        model_name="qwen",
        model_digest=digest,
    )

    class _Inv:
        runtime_capabilities = caps

    rendered = {"tools": [{"type": "function", "function": {"name": "x"}}], "messages": []}
    # First pass: no usage → no calibration
    gw._calibrate_token_meter(_Inv(), rendered, None)
    # Second pass: real usage of 500 tokens for ~2000 rendered bytes
    rendered_body = json.dumps(rendered).encode("utf-8")
    usage = CanonicalUsage(input_tokens=500, output_tokens=10)
    gw._calibrate_token_meter(_Inv(), rendered, usage)

    measured = gw._token_meter.measure_rendered(rendered_body, model_digest=digest)
    # After calibration the meter should use the real backend-derived ratio,
    # surfacing the exact prompt_eval_count (not the conservative estimate).
    assert measured.confidence == "calibrated"
    assert measured.input_tokens == 500


def test_calibrate_skips_without_digest():
    gw = _make_gateway()

    class _InvNoCaps:
        runtime_capabilities = None

    rendered = {"messages": []}
    # No digest → no calibration, meter stays uncalibrated
    gw._calibrate_token_meter(_InvNoCaps(), rendered, CanonicalUsage(input_tokens=100))
    rendered_body = json.dumps(rendered).encode("utf-8")
    measured = gw._token_meter.measure_rendered(rendered_body, model_digest="abc123")
    assert measured.confidence != "exact"
