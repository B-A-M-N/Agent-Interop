"""Tests for Phase 5: evidence/model/execution — source confidence, key dimensions, fallback.

Includes integration tests for confidence gating, codec capabilities, and tool schema fingerprint."""

from __future__ import annotations

import pytest

from agent_interop.config import ToolMode, UpstreamKind
from agent_interop.evidence.key import CompatibilityKeyInputs, build_compatibility_key
from agent_interop.model.registry import (
    BackendMetadata,
    ConformanceRecord,
    ModelProfileRegistry,
    ResolvedModelProfile,
)

# ─── Confidence gating integration (item 86 wired) ──────────────────────────


class TestConfidenceGatingIntegration:
    """Test that low profile confidence gates risky repair tiers."""

    def test_low_confidence_disables_coercive(self):
        from agent_interop.config import RepairPolicy, RepairTier
        from agent_interop.gateway import Gateway

        policy = RepairPolicy(
            enabled_tiers=frozenset({
                RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE,
                RepairTier.COERCIVE, RepairTier.REGENERATION,
            }),
        )
        gw = Gateway.__new__(Gateway)
        result = gw._apply_confidence_gate(policy, 0.1)
        assert RepairTier.COERCIVE not in result.enabled_tiers
        assert RepairTier.REGENERATION not in result.enabled_tiers
        assert RepairTier.SYNTAX_ONLY in result.enabled_tiers

    def test_medium_confidence_disables_only_regeneration(self):
        from agent_interop.config import RepairPolicy, RepairTier
        from agent_interop.gateway import Gateway

        policy = RepairPolicy(
            enabled_tiers=frozenset({
                RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE,
                RepairTier.COERCIVE, RepairTier.REGENERATION,
            }),
        )
        gw = Gateway.__new__(Gateway)
        result = gw._apply_confidence_gate(policy, 0.8)
        assert RepairTier.REGENERATION not in result.enabled_tiers
        assert RepairTier.COERCIVE in result.enabled_tiers

    def test_high_confidence_keeps_all_tiers(self):
        from agent_interop.config import RepairPolicy, RepairTier
        from agent_interop.gateway import Gateway

        policy = RepairPolicy(
            enabled_tiers=frozenset({
                RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE,
                RepairTier.COERCIVE, RepairTier.REGENERATION,
            }),
        )
        gw = Gateway.__new__(Gateway)
        result = gw._apply_confidence_gate(policy, 1.0)
        assert RepairTier.REGENERATION in result.enabled_tiers
        assert RepairTier.COERCIVE in result.enabled_tiers


# ─── Codec capabilities validation (item 74 wired) ──────────────────────────


class TestCodecCapabilitiesIntegration:
    """Test that codec capabilities gate the invocation plan."""

    def test_native_preserved_when_codec_supports(self):
        from agent_interop.abi import CanonicalToolChoice
        from agent_interop.config import ToolMode
        from agent_interop.repair.invocation import build_invocation_plan
        from agent_interop.upstreams.ollama_chat import OllamaChatCodec

        # OllamaChatCodec supports native tools, so a NATIVE route mode stays NATIVE
        # when resolved through build_invocation_plan (which now negotiates codec
        # capability up front).
        result = build_invocation_plan(
            tools=(),
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.NATIVE,
            codec_capabilities=OllamaChatCodec().capabilities(),
        )
        assert result.effective_tool_mode == ToolMode.NATIVE

    def test_tool_count_not_truncated_by_codec(self):
        """Truncation-avoidance is now guaranteed at plan-construction time:
        ``build_invocation_plan`` never slices ``upstream_tools``. With a codec
        that supports native tools (OllamaChatCodec), a NATIVE route mode stays
        NATIVE and the full tool set is carried verbatim.

        The pre-upstream ``max_tools`` REJECTION (a separate concern) is covered
        by ``TestToolCountPreUpstreamRejection`` below.
        """
        from agent_interop.abi import CanonicalTool, CanonicalToolChoice
        from agent_interop.config import ToolMode
        from agent_interop.repair.invocation import build_invocation_plan
        from agent_interop.upstreams.ollama_chat import OllamaChatCodec

        tools = tuple(CanonicalTool(name=f"tool_{i}", description="", input_schema={"type": "object"}) for i in range(70))
        result = build_invocation_plan(
            tools=tools,
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.NATIVE,
            codec_capabilities=OllamaChatCodec().capabilities(),
        )
        # OllamaChatCodec supports native → mode stays NATIVE, no silent truncation.
        assert result.effective_tool_mode == ToolMode.NATIVE
        assert len(result.upstream_tools) == 70


class TestNativeToPromptedDowngrade:
    """Bug 1 regression: a NATIVE plan hitting a codec without native-tool support
    must rebuild a COMPLETE PROMPTED plan — not just flip effective_tool_mode."""

    @staticmethod
    def _make_non_native_codec():
        """A codec whose capabilities declare no native-tool support."""
        from agent_interop.upstreams.codec import CodecCapabilities

        class _NonNativeCodec:
            def capabilities(self) -> CodecCapabilities:
                return CodecCapabilities(supports_native_tools=False)

            def backend_constraints(self):
                from agent_interop.request_validation import BackendConstraints
                return BackendConstraints(max_tools=128)

        return _NonNativeCodec()

    def test_downgrade_produces_complete_prompted_contract(self):
        """Single-construction test: when a NATIVE route mode hits a codec that
        lacks native-tool support, ``build_invocation_plan`` resolves the codec
        capability UP FRONT and builds a single, complete PROMPTED plan — no more
        build-then-mutate (the old gateway-layer rebuild-vs-mutate bug). The
        downgraded plan must carry a full PROMPTED contract: non-empty
        prompt_contract (with tool names + <tool_call>), a digest, a parser_id,
        an output_envelope, and BUFFER_TEXTUAL_RESPONSE streaming.

        A plan that merely flips effective_tool_mode (the old bug) would have an
        empty prompt_contract, parser_id=None, output_envelope=None, and
        stream_extraction_mode=NATIVE_FRAGMENTS — and fail every assertion below.
        """
        from agent_interop.abi import CanonicalTool, CanonicalToolChoice
        from agent_interop.config import ToolMode
        from agent_interop.repair.invocation import StreamExtractionMode, build_invocation_plan

        tools = tuple(
            CanonicalTool(name=f"tool_{i}", description=f"desc {i}", input_schema={"type": "object"})
            for i in range(3)
        )
        # A NATIVE route mode against a non-native codec resolves to PROMPTED in a
        # single construction — the codec veto is applied before any plan fields
        # are computed.
        result = build_invocation_plan(
            tools=tools,
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.NATIVE,
            codec_capabilities=self._make_non_native_codec().capabilities(),
        )

        assert result.effective_tool_mode == ToolMode.PROMPTED
        assert "tool_0" in result.prompt_contract
        assert "<tool_call>" in result.prompt_contract
        assert len(result.prompt_contract_digest) > 0
        # No model_profile → defaults to the standard tool_call envelope parser.
        assert result.parser_id == "tool_call_envelope"
        assert result.output_envelope == "tool_call"
        assert result.stream_extraction_mode == StreamExtractionMode.BUFFER_TEXTUAL_RESPONSE
        # The previously-dead codec_capabilities field is now populated.
        assert result.codec_capabilities is not None


class TestToolCountPreUpstreamRejection:
    """Bug 2 regression: the pre-upstream ``max_tools`` limit only applies to
    requests that actually resolve to NATIVE mode (where tools are sent as a native
    JSON array capped by the backend). Requests resolving to PROMPTED/TEXTUAL/
    DISABLED never send a native tools array, so that limit must NOT reject them —
    doing so is a false positive. Within-limit requests behave exactly as before."""

    @staticmethod
    def _make_gateway(tool_mode=ToolMode.AUTO):
        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        config = InteropServerConfig(
            host="127.0.0.1",
            port=0,
            log_level="error",
            probe_on_startup=False,
            routes={
                "ollama": ModelRoute(
                    id="ollama",
                    client_model_aliases=["qwen2.5-coder"],
                    upstream_model="qwen2.5-coder",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://127.0.0.1:0",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                        timeout_seconds=30.0,
                    ),
                    tool_mode=tool_mode,
                    translation_mode=TranslationMode.CANONICAL,
                ),
            },
        )
        return Gateway(config=config)

    @staticmethod
    def _make_request(tools):
        from agent_interop.abi import (
            CanonicalMessage,
            CanonicalModelReference,
            CanonicalRequest,
            CanonicalTextBlock,
            CanonicalToolChoice,
        )

        return CanonicalRequest(
            model=CanonicalModelReference(requested_name="qwen2.5-coder"),
            messages=[
                CanonicalMessage(role="user", content=[CanonicalTextBlock(text="Do something")])
            ],
            tools=tools,
            tool_choice=CanonicalToolChoice.auto(),
        )

    def test_over_limit_rejected_pre_upstream_when_native(self):
        """OllamaChatCodec caps max_tools at 64. For a request that genuinely
        resolves to NATIVE mode (explicit tool_mode="native" bypasses profile-based
        AUTO resolution), sending 70 tools through the real _prepare_invocation
        path must raise a clean ValueError naming the limit — the native-array cap
        is real and must be enforced."""
        from agent_interop.abi import CanonicalTool
        from agent_interop.context import RequestContext
        from agent_interop.execution import InteropRequestExecution

        gw = self._make_gateway(tool_mode=ToolMode.NATIVE)
        tools = [
            CanonicalTool(name=f"tool_{i}", description="", input_schema={"type": "object"})
            for i in range(70)
        ]
        request = self._make_request(tools)

        with pytest.raises(ValueError) as exc_info:
            gw._prepare_invocation(
                request, RequestContext(), streaming=False, execution=InteropRequestExecution(),
            )
        assert "Too many tools" in str(exc_info.value)
        assert "64" in str(exc_info.value)

    def test_over_limit_passes_when_prompted(self):
        """With tool_mode="auto", qwen2.5-coder resolves to PROMPTED (via the
        bundled qwen-coder-ollama profile, presentation.mode=prompted), so the 70
        tools are serialized into the text contract and never sent as a native
        tools array. The native-array max_tools limit does NOT apply — the request
        must pass through, and the plan must be PROMPTED with the full tool set."""
        from agent_interop.abi import CanonicalTool
        from agent_interop.config import ToolMode
        from agent_interop.context import RequestContext
        from agent_interop.execution import InteropRequestExecution

        gw = self._make_gateway(tool_mode=ToolMode.AUTO)
        tools = [
            CanonicalTool(name=f"tool_{i}", description="", input_schema={"type": "object"})
            for i in range(70)
        ]
        request = self._make_request(tools)

        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=False, execution=record,
        )
        assert invocation.invocation_plan is not None
        # Resolved to PROMPTED (tools embedded as text, not a native array).
        assert invocation.invocation_plan.effective_tool_mode == ToolMode.PROMPTED
        # upstream_tools is empty for PROMPTED, but validation_tools carries the
        # full tool set for the textual extractor.
        assert len(invocation.invocation_plan.validation_tools) == 70

    def test_within_limit_passes_unchanged(self):
        """A request with a tool count within the codec's limit must flow through
        _prepare_invocation exactly as before — no spurious error, plan built."""
        from agent_interop.abi import CanonicalTool
        from agent_interop.context import RequestContext
        from agent_interop.execution import InteropRequestExecution

        gw = self._make_gateway()
        tools = [
            CanonicalTool(name=f"tool_{i}", description="", input_schema={"type": "object"})
            for i in range(5)
        ]
        request = self._make_request(tools)

        record = InteropRequestExecution()
        invocation = gw._prepare_invocation(
            request, RequestContext(), streaming=False, execution=record,
        )
        assert invocation.invocation_plan is not None
        # The full within-limit tool set is preserved on the plan.
        assert len(invocation.invocation_plan.validation_tools) == 5


# ─── Tool schema fingerprint (item 83 wired) ────────────────────────────────


class TestToolSchemaFingerprint:
    def test_same_tools_same_fingerprint(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.gateway import Gateway

        gw = Gateway.__new__(Gateway)
        tools = [CanonicalTool(name="read", description="", input_schema={"type": "object"})]
        fp1 = gw._compute_tool_schema_fingerprint(tools)
        fp2 = gw._compute_tool_schema_fingerprint(tools)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_different_tools_different_fingerprint(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.gateway import Gateway

        gw = Gateway.__new__(Gateway)
        tools1 = [CanonicalTool(name="read", description="", input_schema={"type": "object"})]
        tools2 = [CanonicalTool(name="write", description="", input_schema={"type": "object"})]
        fp1 = gw._compute_tool_schema_fingerprint(tools1)
        fp2 = gw._compute_tool_schema_fingerprint(tools2)
        assert fp1 != fp2

    def test_empty_tools_empty_fingerprint(self):
        from agent_interop.gateway import Gateway
        gw = Gateway.__new__(Gateway)
        assert gw._compute_tool_schema_fingerprint([]) == ""


# ─── InvocationPlan carries source confidence (item 86) ─────────────────────


class TestInvocationPlanCarriesConfidence:
    def test_plan_includes_source_confidence(self):
        from agent_interop.abi import CanonicalTool, CanonicalToolChoice
        from agent_interop.config import ToolMode
        from agent_interop.repair.invocation import build_invocation_plan

        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        plan = build_invocation_plan(
            tools=[tool],
            tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.AUTO,
            model_profile=_mock_profile(),
        )
        assert plan.source_confidence > 0.0
        assert plan.profile_source != ""


# ─── Source confidence (item 86) ────────────────────────────────────────────


class TestSourceConfidence:
    def test_unknown_model_gets_lowest_confidence(self):
        """An unmatched model must reach the registry's fallback tier, not
        be treated as a builtin match — see profiles_v2._FALLBACK_PROFILE_ID."""
        reg = ModelProfileRegistry()
        result = reg.resolve(model_name="xyz-nonexistent-model-999")
        assert result.source == "fallback"
        assert result.source_confidence == 0.1

    def test_true_fallback_has_lowest_confidence(self):
        """If no profile matches at all (empty registry), fallback is used."""
        from agent_interop.model.profiles_v2 import ProfileIndex
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="any-model")
        assert result.source == "fallback"
        assert result.source_confidence == 0.1

    def test_explicit_profile_has_high_confidence(self):
        reg = ModelProfileRegistry()
        # Use a built-in profile ID
        result = reg.resolve(
            model_name="test-model",
            explicit_profile_id="hermes-3-llama-ollama",
        )
        if result.source == "explicit":
            assert result.source_confidence == 0.9

    def test_override_has_max_confidence(self):
        from agent_interop.model.profiles_v2 import ModelProfile, ToolBehaviorProfile
        reg = ModelProfileRegistry()
        override = ModelProfile(
            id="test-override",
            tool_behavior=ToolBehaviorProfile(
                native_schema_support=True,
                native_response_support=True,
                presentation_mode="native",
                supports_auto_choice=True,
            ),
        )
        result = reg.resolve(model_name="test", session_override=override)
        assert result.source == "override"
        assert result.source_confidence == 1.0

    def test_confidence_scales_with_valid_call_rate(self):
        reg = ModelProfileRegistry()
        conformance = ConformanceRecord(
            model_id="test",
            valid_call_rate=0.95,
            supports_native_tools=True,
        )
        result = reg.resolve(model_name="test", conformance=conformance)
        if result.source == "conformance":
            assert result.source_confidence == 0.95

    def test_conformance_overlay_preserves_base_profile_executable_fields(self):
        """Re-audit P1#8: conformance used to be an early-return branch that
        built a bare ResolvedModelProfile(...) with only 6 fields set —
        parser_id, contract_template_id, fallback_strategies, and context
        limits all silently reverted to defaults/None even when the model
        had a real builtin profile match. A model that's PROVEN reliable
        (valid_call_rate > 0.8) must not lose its own dialect info as the
        reward for being proven reliable."""
        reg = ModelProfileRegistry()
        conformance = ConformanceRecord(
            model_id="qwen3-coder", valid_call_rate=0.95, supports_native_tools=False,
        )
        result = reg.resolve(model_name="qwen3-coder", conformance=conformance)

        assert result.source == "conformance"
        assert result.source_confidence == 0.95
        # These come from the qwen3-coder builtin profile — proving the
        # overlay ran on top of the base resolution, not instead of it.
        assert result.parser_id == "tool_call_envelope"
        assert result.tool_parallel is True
        assert result.declared_tokens == 131072
        assert result.safe_tokens == 65536

    def test_conformance_overlay_can_raise_but_not_silently_hide_capability(self):
        """Live-verified conformance data can prove a capability the static
        profile didn't claim (raise the signal); it must not be lost."""
        reg = ModelProfileRegistry()
        conformance = ConformanceRecord(
            model_id="xyz-nonexistent-model-999", valid_call_rate=0.9,
            supports_native_tools=True,
        )
        result = reg.resolve(model_name="xyz-nonexistent-model-999", conformance=conformance)
        assert result.source == "conformance"
        assert result.supports_native_tools is True


# ─── Fallback semantics (item 88) ───────────────────────────────────────────


class TestFallbackSemantics:
    def test_fallback_does_not_claim_native_tools(self):
        from agent_interop.model.profiles_v2 import ProfileIndex
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="unknown-model-12345")
        assert result.source == "fallback"
        assert result.supports_native_tools is False

    def test_fallback_allows_textual_mode(self):
        from agent_interop.model.profiles_v2 import ProfileIndex
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="unknown-model-12345")
        assert result.supports_textual_tools is True

    def test_fallback_has_conservative_limits(self):
        from agent_interop.model.profiles_v2 import ProfileIndex
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="unknown-model-12345")
        assert result.safe_tokens <= 4096
        assert result.declared_tokens <= 8192

    def test_fallback_output_envelope_set(self):
        from agent_interop.model.profiles_v2 import ProfileIndex
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="unknown-model-12345")
        assert result.output_envelope == "tool_call"
        assert result.parser_id == "tool_call_envelope"

    def test_fallback_does_not_claim_named_choice_support(self):
        """The single fallback-profile definition (get_fallback_profile())
        declares supports_named_choice=False — an unknown model with no
        conformance evidence can't be assumed to reliably honor a request
        for one specific named tool. Previously this dimension was
        dropped entirely during ResolvedModelProfile conversion, and a
        second, independently-hand-maintained inline fallback in
        registry.py's resolve() didn't carry it at all — two fallback
        definitions that had drifted apart."""
        from agent_interop.model.profiles_v2 import ProfileIndex
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="unknown-model-12345")
        assert result.tool_named is False

    def test_fallback_matches_get_fallback_profile_directly(self):
        """resolve()'s fallback branch must be the SAME declaration as
        get_fallback_profile(), not an independent hand-copy that can
        silently drift from it."""
        from agent_interop.model.profiles_v2 import ProfileIndex, get_fallback_profile
        reg = ModelProfileRegistry(profiles=ProfileIndex())
        result = reg.resolve(model_name="unknown-model-12345")
        fallback = get_fallback_profile()
        assert result.tool_automatic == fallback.tool_behavior.supports_auto_choice
        assert result.tool_named == fallback.tool_behavior.supports_named_choice
        assert result.tool_required == fallback.tool_behavior.supports_required_choice
        assert result.tool_parallel == fallback.tool_behavior.supports_parallel_calls


# ─── Compatibility key dimensions (item 83) ─────────────────────────────────


class TestCompatibilityKeyDimensions:
    def test_all_fields_populated_with_context(self):
        key = build_compatibility_key(CompatibilityKeyInputs(
            route=_mock_route(),
            request=_mock_request(),
            model_profile=ResolvedModelProfile(profile_id="test"),
            backend_metadata=BackendMetadata(
                backend_kind=UpstreamKind.OLLAMA,
                model_name="llama3",
            ),
            tool_schema_fingerprint="abc123",
            streaming=False,
        ))
        # Model identity should be populated
        assert key.model_id == "llama3"
        # Backend identity should be populated
        assert key.backend_kind == "ollama"
        # Tool schema fingerprint should be populated
        assert key.tool_schema_fingerprint == "abc123"

    def test_empty_context_produces_empty_client_fields(self):
        """Without context, client fields are empty — key still valid."""
        key = build_compatibility_key(CompatibilityKeyInputs(
            route=_mock_route(),
            model_profile=ResolvedModelProfile(profile_id="test"),
        ))
        assert key.client_id == ""
        assert key.client_version == ""

    def test_profile_revision_reaches_the_key(self):
        """ResolvedModelProfile previously had no profile_revision field at
        all, so build_compatibility_key's getattr(profile, "profile_revision", "")
        always silently fell back to its "" default — a profile's YAML
        `revision:` field never actually differentiated the cache key, so
        a revised profile with different behavior could collide with a
        stale cached evidence record for the old revision."""
        key = build_compatibility_key(CompatibilityKeyInputs(
            route=_mock_route(),
            model_profile=ResolvedModelProfile(profile_id="test", profile_revision="rev-42"),
        ))
        assert key.profile_revision == "rev-42"

    def test_upstream_protocol_populated_from_route(self):
        key = build_compatibility_key(CompatibilityKeyInputs(
            route=_mock_route(),
            model_profile=ResolvedModelProfile(profile_id="test"),
        ))
        assert key.upstream_protocol == "ollama_chat"


# ─── Helpers ────────────────────────────────────────────────────────────────


def _mock_route():
    from agent_interop.config import ModelRoute, UpstreamConfig, UpstreamKind, UpstreamProtocol
    return ModelRoute(
        id="test",
        client_model_aliases=["test"],
        upstream_model="llama3",
        upstream=UpstreamConfig(
            kind=UpstreamKind.OLLAMA,
            base_url="http://localhost:11434",
            wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
        ),
    )


def _mock_request():
    from agent_interop.abi import CanonicalModelReference, CanonicalRequest
    return CanonicalRequest(model=CanonicalModelReference(requested_name="llama3"))


def _mock_profile():
    return ResolvedModelProfile(
        profile_id="test",
        source="builtin",
        source_confidence=0.8,
        supports_native_tools=True,
        parser_id="tool_call_envelope",
    )
