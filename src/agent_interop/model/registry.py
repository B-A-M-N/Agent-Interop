"""Model profile registry — consolidated profile resolution.

Resolves a model + backend + conformance tuple into a ResolvedModelProfile
with effective capabilities. This is the single source of truth for model
behavior, replacing the ad-hoc capability guessing in Gateway.

Resolution priority (highest first):
1. Per-route/session override
2. Verified conformance record for exact model/runtime tuple
3. Explicit profile ID
4. Built-in profile match
5. Backend metadata
6. Conservative generic fallback
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_interop.config import UpstreamKind
from agent_interop.model.profiles_v2 import (
    ModelProfile,
    ProfileIndex,
    get_fallback_profile,
    load_profiles,
)


@dataclass(frozen=True)
class BackendMetadata:
    """Metadata reported by a backend about a model."""

    backend_kind: UpstreamKind = UpstreamKind.OLLAMA
    backend_version: str = ""
    model_name: str = ""
    model_digest: str = ""
    context_length: int = 0
    chat_template: str = ""
    quantization: str = ""


@dataclass(frozen=True)
class ConformanceRecord:
    """Verified conformance result for a specific model/runtime tuple."""

    model_id: str = ""
    backend_kind: UpstreamKind = UpstreamKind.OLLAMA
    tool_selection_rate: float = 0.0
    valid_call_rate: float = 0.0
    supports_native_tools: bool = False
    supports_textual_tools: bool = False
    notes: str = ""


@dataclass(frozen=True)
class ResolvedModelProfile:
    """Fully resolved model profile with effective capabilities."""

    profile_id: str = ""
    source: str = "fallback"  # override | conformance | explicit | builtin | backend | fallback
    source_confidence: float = 0.0  # 0.0–1.0 confidence in the resolution (item 86)

    # Effective capabilities
    supports_native_tools: bool = False
    supports_textual_tools: bool = False
    tool_call_dialect: str = "generic"
    output_envelope: str | None = None
    parser_id: str | None = None
    contract_template_id: str | None = None
    fallback_strategies: tuple[Any, ...] = ()

    # Behavior
    tool_automatic: bool = False
    tool_parallel: bool = False
    tool_named: bool = True
    tool_required: bool = True
    streaming_supported: bool = True
    reasoning_supported: bool = False

    # Identity (item 89: profile_revision was previously read via
    # getattr() off this class but never actually set here, so a
    # profile's YAML `revision:` field silently never affected the
    # evidence/certification cache key)
    profile_revision: str = ""

    # Limits
    declared_tokens: int = 4096
    safe_tokens: int = 2048

    # Raw profile reference
    raw_profile: ModelProfile | None = None


class ModelProfileRegistry:
    """Registry that resolves model profiles from multiple sources.

    Injected into Gateway as the single source of truth for model behavior.
    """

    def __init__(self, profiles: ProfileIndex | None = None) -> None:
        self._profiles = load_profiles() if profiles is None else profiles

    def resolve(
        self,
        model_name: str = "",
        backend: UpstreamKind = UpstreamKind.OLLAMA,
        backend_metadata: BackendMetadata | None = None,
        conformance: ConformanceRecord | None = None,
        explicit_profile_id: str | None = None,
        session_override: ModelProfile | None = None,
    ) -> ResolvedModelProfile:
        """Resolve the best profile for a model/backend tuple.

        Conservative: does NOT assume native tool support merely because
        the upstream protocol is OpenAI/Anthropic/Ollama. The model itself
        must demonstrate tool capability through conformance or profile.
        """
        # 1. Session override
        if session_override is not None:
            return self._from_profile(session_override, source="override", confidence=1.0)

        # 2-6. Resolve a full base profile first (explicit -> builtin ->
        # backend -> fallback), THEN apply verified conformance as a narrow
        # overlay on top of it — never as an early-return replacement.
        # Conformance data proves a capability *signal* (native tools:
        # yes/no, a valid-call rate) for one model/runtime tuple; it says
        # nothing about which parser, contract template, fallback
        # strategies, or context limits that model needs. Short-circuiting
        # straight to a bare ResolvedModelProfile(...) here used to throw
        # all of that away — a model that also had a real builtin profile
        # match would silently lose its own dialect info the moment
        # conformance data existed for it.
        base = self._resolve_base(model_name, backend, backend_metadata, explicit_profile_id)

        if conformance is not None and conformance.valid_call_rate > 0.8:
            return self._apply_conformance_overlay(base, conformance)

        return base

    def _resolve_base(
        self,
        model_name: str,
        backend: UpstreamKind,
        backend_metadata: BackendMetadata | None,
        explicit_profile_id: str | None,
    ) -> ResolvedModelProfile:
        """Resolve the base profile: explicit ID -> builtin match -> backend
        metadata match -> conservative generic fallback. Factored out of
        resolve() so conformance can be layered on top of whatever this
        returns, instead of needing its own independent code path."""
        # 3. Explicit profile ID
        if explicit_profile_id:
            profile = self._profiles.get_by_id(explicit_profile_id)
            if profile is not None:
                return self._from_profile(profile, source="explicit", confidence=0.9)

        # 4. Built-in profile match by name pattern
        if model_name:
            profile = self._profiles.resolve(model_name, backend.value)
            if profile is not None:
                return self._from_profile(profile, source="builtin", confidence=0.8)

        # 5. Backend metadata
        if backend_metadata and backend_metadata.model_name:
            profile = self._profiles.resolve(
                backend_metadata.model_name, backend_metadata.backend_kind.value
            )
            if profile is not None:
                return self._from_profile(profile, source="backend", confidence=0.7)

        # 6. Conservative generic fallback (item 88)
        # We know nothing about this model. Assume it can at least try
        # prompted mode (textual), but don't claim native tool support.
        # The invocation plan will downgrade to PROMPTED/DISABLED as needed.
        #
        # Routed through get_fallback_profile() + _from_profile() rather
        # than an independent inline ResolvedModelProfile(...) — a second,
        # hand-maintained copy of "what the fallback profile declares" had
        # drifted from get_fallback_profile()'s actual declaration (which
        # was itself dead code, never called from anywhere): this branch
        # claimed tool_automatic=False but never carried the fallback
        # profile's supports_named_choice=False at all, so a "generic,
        # unknown model" could still be asked for a named tool_choice with
        # nothing in the resolved profile reflecting that the profile
        # explicitly says it can't reliably do that.
        return self._from_profile(get_fallback_profile(), source="fallback", confidence=0.1)

    @staticmethod
    def _apply_conformance_overlay(
        base: ResolvedModelProfile, conformance: ConformanceRecord,
    ) -> ResolvedModelProfile:
        """Apply verified conformance as a narrow overlay on ``base``.

        Only refines confidence and capability signal; every executable
        field the base profile carries (parser_id, contract_template_id,
        fallback_strategies, context limits, choice support) survives
        unchanged. Capability flags can only be raised by conformance
        (live-verified evidence trumps a static guess), never silently
        lowered — a model conformance proved supports native tools should
        say so even if its static profile predates that proof.
        """
        from dataclasses import replace

        return replace(
            base,
            profile_id=conformance.model_id or base.profile_id,
            source="conformance",
            source_confidence=conformance.valid_call_rate,
            supports_native_tools=base.supports_native_tools or conformance.supports_native_tools,
            supports_textual_tools=base.supports_textual_tools or conformance.supports_textual_tools,
            tool_call_dialect=(
                "native" if conformance.supports_native_tools else base.tool_call_dialect
            ),
        )

    def _from_profile(
        self,
        profile: ModelProfile,
        source: str,
        confidence: float = 0.5,
    ) -> ResolvedModelProfile:
        """Convert a ModelProfile to a ResolvedModelProfile.

        CRITICAL: Uses explicit capability dimensions from ToolBehaviorProfile,
        NOT the legacy tool_automatic field. `automatic` means "can auto-select
        tools" — it does NOT mean "supports native structured transport".
        """
        tb = profile.tool_behavior

        # Use explicit dimensions: native_schema_support is separate from
        # supports_auto_choice. A Qwen model may support automatic selection
        # while requiring textual dialect → PROMPTED, not NATIVE.
        supports_native = (
            tb.native_schema_support
            and tb.native_response_support
            and tb.presentation_mode == "native"
        )
        supports_textual = (
            tb.presentation_mode in ("prompted", "textual")
            or tb.extractor_id is not None
        )

        return ResolvedModelProfile(
            profile_id=profile.id or profile.matched_by,
            source=source,
            source_confidence=confidence,
            supports_native_tools=supports_native,
            supports_textual_tools=supports_textual,
            tool_call_dialect=tb.presentation_mode,
            output_envelope=tb.output_envelope,
            parser_id=tb.extractor_id,
            contract_template_id=tb.contract_template_id,
            fallback_strategies=tb.fallback_strategies,
            tool_automatic=tb.supports_auto_choice,
            tool_parallel=tb.supports_parallel_calls,
            tool_named=tb.supports_named_choice,
            tool_required=tb.supports_required_choice,
            streaming_supported=profile.streaming_supported,
            reasoning_supported=profile.reasoning_supported,
            declared_tokens=profile.declared_tokens,
            safe_tokens=profile.safe_tokens,
            profile_revision=profile.profile_revision,
            raw_profile=profile,
        )

    def list_profiles(self) -> list[ModelProfile]:
        """Return all loaded profiles for CLI display and diagnostics."""
        return self._profiles.list_profiles()


# ─── Module-level default ──────────────────────────────────────────────────

_default_registry: ModelProfileRegistry | None = None


def get_default_registry() -> ModelProfileRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelProfileRegistry()
    return _default_registry
