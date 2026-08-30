"""P0.43: acceptance harness nonce validation.

The acceptance round trip must prove the per-request execution-nonce gate
actually rejects a nonce-less whole-message-JSON tool call. This is the
same safety boundary unit-tested in test_whole_message_json.py, but here
it's asserted through the *acceptance harness's* extraction path (the real
Gateway extractor registry + whole_message_json fallback) so a real client
run can record `verification.nonce_gated_recovery: true`.

The gateway's non-streaming send path calls
``gw._extractor_registry.extract(..., expected_execution_nonce=plan.execution_nonce)``
— exactly what this test drives.
"""


from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.gateway import Gateway


def _build_acceptance_gateway():
    config = InteropServerConfig(
        probe_on_startup=False,
        default_route_id="acceptance",
        routes={
            "acceptance": ModelRoute(
                id="acceptance",
                client_model_aliases=["acceptance-model"],
                upstream_model="acceptance-test-model",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
        log_level="error",
    )
    return Gateway(config, allow_invalid_config=True)


def _extract(gw: Gateway, model_text: str, expected_nonce: str | None):
    """Drive one whole-message-JSON recovery through the real extractor.

    Mirrors the call site in Gateway._extract_tool_candidates for the
    whole_message_json fallback path used by acceptance scenarios.
    """
    from agent_interop.abi import CanonicalTextBlock, CanonicalTool, CanonicalToolChoice
    from agent_interop.model.profiles_v2 import ExtractionStrategy

    strategy = ExtractionStrategy(
        parser_id="whole_message_json",
        skip_when_native_present=False,
        allowed_tool_choice_modes=frozenset({"auto"}),
    )
    tools = [
        CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        ),
    ]
    plan = type(
        "_Plan",
        (),
        {
            "parser_id": "tool_call_envelope",
            "output_envelope": "tool_call",
            "validation_tools": tools,
            "fallback_strategies": (strategy,),
            "original_tool_choice": CanonicalToolChoice.auto(),
            "execution_nonce": expected_nonce,
        },
    )()
    content = [CanonicalTextBlock(text=model_text)]
    return gw._extractor_registry.extract(
        content,
        extractor_id=plan.parser_id,
        tools=plan.validation_tools,
        envelope=plan.output_envelope,
        fallback_strategies=plan.fallback_strategies,
        tool_choice=plan.original_tool_choice,
        native_candidates_present=False,
        expected_execution_nonce=plan.execution_nonce,
    )


def test_acceptance_nonce_gate_rejects_missing_nonce():
    gw = _build_acceptance_gateway()
    # A nonce was configured for this request...
    expected = "abc123nonce"
    # ...but the model text carries no interop_call_id
    text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
    result = _extract(gw, text, expected)
    # The gate must NOT recover a tool call without the matching nonce
    assert not result.candidates, "nonce-less whole-message-JSON must NOT be recovered"


def test_acceptance_nonce_gate_accepts_matching_nonce():
    gw = _build_acceptance_gateway()
    expected = "abc123nonce"
    text = '{"name":"read_file","arguments":{"path":"/tmp/x"},"interop_call_id":"abc123nonce"}'
    result = _extract(gw, text, expected)
    assert result.candidates, "matching-nonce whole-message-JSON SHOULD be recovered"
    assert result.candidates[0].name == "read_file"
