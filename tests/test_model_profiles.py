"""Tests for model profiles — profile lookup and metadata.

Exercises the v2 profile system (interop.model.profiles_v2 / registry):
YAML profiles loaded from data/profiles/, resolved by model name + backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_interop.config import UpstreamKind
from agent_interop.model.profiles_v2 import ModelProfile, load_profiles
from agent_interop.model.registry import ModelProfileRegistry

# Load the real built-in profiles once for the whole module.
_PROFILES = load_profiles([Path("src/agent_interop/data/profiles")])
_REGISTRY = ModelProfileRegistry(profiles=_PROFILES)


def _resolve(model: str, backend: str = "ollama") -> ModelProfile | None:
    """Resolve a model name against the real built-in profile index."""
    return _PROFILES.resolve(model, backend)


class TestResolve:
    def test_exact_match(self):
        p = _resolve("qwen3-coder")
        assert p is not None
        assert p.id == "qwen3-coder"
        assert p.tool_behavior.supports_parallel_calls is True

    def test_tag_suffix_match(self):
        # "qwen3-coder:latest" should resolve to the qwen3-coder profile.
        p = _resolve("qwen3-coder:latest")
        assert p is not None
        assert p.id == "qwen3-coder"

    def test_prefix_match(self):
        # A longer model name still resolves to its family profile.
        p = _resolve("llama-3.1-8b-instruct")
        assert p is not None
        assert "llama" in p.id

    def test_unknown_model_falls_back(self):
        # Nothing specific matches. ProfileIndex.resolve() deliberately
        # excludes generic-fallback from catch-all matching (it's only
        # reachable through ModelProfileRegistry's own fallback tier), so
        # an unmatched model resolves to no profile at all here.
        p = _resolve("totally-fake-model-v99")
        assert p is None

    def test_list_returns_all(self):
        ids = _PROFILES.list_ids()
        assert len(ids) >= 8  # we have at least 8 builtins


class TestBuiltinProfiles:
    def test_qwen_profile(self):
        p = _resolve("qwen3-coder")
        assert p is not None
        assert p.declared_tokens == 131072
        # L3-capable: parallel tool calls supported.
        assert p.tool_behavior.supports_parallel_calls is True
        assert p.tool_behavior.supports_auto_choice is True

    def test_hermes_profile(self):
        p = _resolve("hermes-3-llama-3.1-405b")
        assert p is not None
        # Resolved by the existing hermes-3-llama-ollama profile.
        assert p.id == "hermes-3-llama-ollama"
        assert p.tool_behavior.supports_parallel_calls is True

    def test_deepseek_profile(self):
        p = _resolve("deepseek-v4-0324")
        assert p is not None
        assert p.id == "deepseek-v4-0324"
        # DeepSeek V4 supports reasoning / thinking.
        assert p.reasoning_supported is True
        # Was `parser: deepseek` (only matches DeepSeek's native
        # \x14...\x14 delimiter) while the rendered contract taught
        # <tool_call>...</tool_call> — a real mismatch, fixed by using
        # the extractor that actually matches what's taught.
        assert p.tool_behavior.extractor_id == "tool_call_envelope"

    def test_fallback_profile(self):
        # generic-fallback is excluded from pattern-based resolve(); reach
        # it directly via get_by_id() the way an explicit_profile_id would.
        p = _PROFILES.get_by_id("generic-fallback")
        assert p is not None
        # The generic fallback has tool calling disabled.
        assert p.tool_behavior.supports_parallel_calls is False


class TestRegistryResolve:
    """Exercise ModelProfileRegistry.resolve with real profiles."""

    def test_builtin_resolution_has_source_and_confidence(self):
        result = _REGISTRY.resolve(model_name="qwen3-coder", backend=UpstreamKind.OLLAMA)
        assert result.source == "builtin"
        assert result.source_confidence == 0.8
        assert result.profile_id == "qwen3-coder"

    def test_unknown_model_resolves_to_generic_fallback(self):
        # generic-fallback is excluded from ordinary pattern matching, so an
        # unknown model reaches the registry's own conservative fallback
        # tier — not a "builtin" match with builtin-level confidence.
        result = _REGISTRY.resolve(
            model_name="totally-fake-model-v99", backend=UpstreamKind.OLLAMA
        )
        assert result.source == "fallback"
        assert result.source_confidence == 0.1
        assert result.profile_id == "generic-fallback"
        # Conservative: the generic fallback never claims native tool support.
        assert result.supports_native_tools is False

    def test_truly_empty_registry_uses_synthetic_fallback(self):
        # With no profiles at all, the registry's own fallback fires.
        from agent_interop.model.profiles_v2 import ProfileIndex

        empty_registry = ModelProfileRegistry(profiles=ProfileIndex())
        result = empty_registry.resolve(
            model_name="totally-fake-model-v99", backend=UpstreamKind.OLLAMA
        )
        assert result.source == "fallback"
        assert result.supports_native_tools is False

    def test_native_profile_reports_native_tools(self):
        # gpt-4o-mini-compat is the one native-mode profile.
        result = _REGISTRY.resolve(
            model_name="gpt-4o-mini", backend=UpstreamKind.OPENAI_COMPATIBLE
        )
        assert result.profile_id == "gpt-4o-mini-compat"
        assert result.supports_native_tools is True

    def test_backend_metadata_tier_used_when_model_name_not_given(self):
        """resolve()'s documented priority-5 tier ("backend metadata") is
        only reachable when the caller doesn't already know the model name
        up front but backend introspection reveals it — e.g. resolving
        before a name is confirmed. Confirms the tier itself works;
        Gateway._resolve_profile wiring it through in production is
        covered separately in test_gateway.py-adjacent tests."""
        from agent_interop.model.registry import BackendMetadata

        result = _REGISTRY.resolve(
            model_name="",  # not known directly by the caller
            backend=UpstreamKind.OLLAMA,
            backend_metadata=BackendMetadata(
                backend_kind=UpstreamKind.OLLAMA, model_name="qwen3-coder",
            ),
        )
        assert result.source == "backend"
        assert result.profile_id == "qwen3-coder"


class TestGatewayResolveProfileWiring:
    """Gateway._resolve_profile() previously called
    ModelProfileRegistry.resolve() without ever passing backend_metadata —
    the caller (Gateway._prepare_invocation) already computed it (for the
    compatibility key) one line above, but never threaded it through, so
    the registry's documented priority-5 "backend metadata" tier was dead
    code from every real request."""

    def test_resolve_profile_passes_backend_metadata_through(self):
        from unittest.mock import MagicMock

        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            ToolMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.gateway import Gateway

        route = ModelRoute(
            id="test",
            client_model_aliases=["test-model"],
            upstream_model="qwen3-coder",
            upstream=UpstreamConfig(
                kind=UpstreamKind.OLLAMA,
                base_url="http://127.0.0.1:11434",
                wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
            ),
            tool_mode=ToolMode.AUTO,
        )
        config = InteropServerConfig(probe_on_startup=False, routes={"test": route})
        gw = Gateway(config)
        gw._profile_registry = MagicMock(wraps=gw._profile_registry)

        backend_metadata = gw._get_backend_metadata(route)
        gw._resolve_profile(route, backend_metadata)

        _, kwargs = gw._profile_registry.resolve.call_args
        assert kwargs.get("backend_metadata") is backend_metadata
        assert kwargs["backend_metadata"] is not None


# ─── Release-gate: every packaged profile is a genuine executable contract ──


class TestPackagedProfilesAreExecutableContracts:
    """Profiles are executable contracts, not descriptive metadata — a
    profile that fails strict validation, references an unimplemented
    parser/template, or can't actually build a real InvocationPlan is
    broken regardless of whether its YAML parses. This is the check that
    would have caught the qwen/mistral/deepseek/llama taught-vs-parsed
    envelope mismatch (see those profiles' YAML comments) automatically,
    if it had existed before that bug was introduced.
    """

    def _all_profile_paths(self) -> list[Path]:
        return sorted(Path("src/agent_interop/data/profiles").glob("*.yaml"))

    def test_every_packaged_profile_file_exists_and_parses(self) -> None:
        paths = self._all_profile_paths()
        assert len(paths) >= 8, "expected at least 8 packaged profiles"

    def test_every_packaged_profile_passes_strict_validation(self) -> None:
        import yaml

        from agent_interop.model.profiles_v2 import validate_profile_schema

        failures = {}
        for path in self._all_profile_paths():
            data = yaml.safe_load(path.read_text())
            issues = validate_profile_schema(data, source=path.name)
            if issues:
                failures[path.name] = issues
        assert not failures, f"profiles failed strict validation: {failures}"

    def test_no_packaged_profile_is_silently_dropped_by_the_loader(self) -> None:
        """load_profiles() logs-and-skips a profile that fails strict
        validation rather than raising — this test is what actually
        notices a silently-dropped profile, by requiring every *.yaml
        file on disk to produce a loaded profile with a matching id."""
        on_disk = set()
        for path in self._all_profile_paths():
            import yaml
            data = yaml.safe_load(path.read_text())
            if isinstance(data, dict) and data.get("id"):
                on_disk.add(data["id"])

        loaded = set(_PROFILES.list_ids())
        missing = on_disk - loaded
        assert not missing, f"profiles present on disk but not loaded (see logs for why): {missing}"

    def test_every_loaded_profile_builds_a_real_invocation_plan(self) -> None:
        """Resolve each profile through the SAME path a live request
        uses (ModelProfileRegistry -> ResolvedModelProfile ->
        build_invocation_plan) and assert the plan reflects what the
        profile actually declared — not just that construction didn't
        raise."""
        from agent_interop.abi import CanonicalTool, CanonicalToolChoice
        from agent_interop.config import ToolMode
        from agent_interop.model.registry import ModelProfileRegistry
        from agent_interop.repair.invocation import build_invocation_plan

        registry = ModelProfileRegistry(profiles=_PROFILES)
        tool = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )

        for profile_id in _PROFILES.list_ids():
            resolved = registry.resolve(explicit_profile_id=profile_id)
            assert resolved.profile_id == profile_id or resolved.raw_profile is not None, (
                f"resolve(explicit_profile_id={profile_id!r}) did not resolve the "
                f"profile it was asked for"
            )

            plan = build_invocation_plan(
                tools=[tool],
                tool_choice=CanonicalToolChoice.auto(),
                route_mode=ToolMode.AUTO,
                model_profile=resolved,
                repair_policy=None,
                codec_capabilities=None,
            )

            # The plan must reference a real, registered parser — not a
            # dangling id (build_invocation_plan doesn't validate this
            # itself; a live request would only discover a bad parser_id
            # when ExtractorRegistry.get() raises mid-request).
            if plan.parser_id is not None:
                from agent_interop.extraction import get_default_registry
                get_default_registry().get(plan.parser_id)  # raises if unknown

            # PROMPTED mode must have actually rendered a non-empty
            # contract — an empty prompt_contract would mean the model
            # gets no instructions at all despite tools being declared.
            if plan.effective_tool_mode == ToolMode.PROMPTED:
                assert plan.prompt_contract.strip(), (
                    f"profile {profile_id!r}: PROMPTED mode produced an empty prompt_contract"
                )

            # Every fallback strategy's parser must also be a real,
            # registered extractor.
            for strategy in resolved.fallback_strategies:
                from agent_interop.extraction import get_default_registry
                get_default_registry().get(strategy.parser_id)

    def test_contract_template_actually_changes_rendered_text(self) -> None:
        """qwen-coder-ollama declares contract_template: qwen-tool-v1 —
        prove the rendered prompt_contract actually differs from the
        universal default, i.e. the field is real, not decorative."""
        from agent_interop.abi import CanonicalTool, CanonicalToolChoice
        from agent_interop.config import ToolMode
        from agent_interop.model.registry import ModelProfileRegistry
        from agent_interop.repair.invocation import build_invocation_plan

        registry = ModelProfileRegistry(profiles=_PROFILES)
        tool = CanonicalTool(name="read_file", description="Read a file", input_schema={})

        qwen_resolved = registry.resolve(explicit_profile_id="qwen-coder-ollama")
        assert qwen_resolved.contract_template_id == "qwen-tool-v1"
        qwen_plan = build_invocation_plan(
            tools=[tool], tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.AUTO, model_profile=qwen_resolved,
            repair_policy=None, codec_capabilities=None,
        )

        generic_resolved = registry.resolve(explicit_profile_id="hermes-3-llama-ollama")
        assert generic_resolved.contract_template_id is None
        generic_plan = build_invocation_plan(
            tools=[tool], tool_choice=CanonicalToolChoice.auto(),
            route_mode=ToolMode.AUTO, model_profile=generic_resolved,
            repair_policy=None, codec_capabilities=None,
        )

        assert qwen_plan.prompt_contract != generic_plan.prompt_contract
        assert "fence" in qwen_plan.prompt_contract.lower() or "backtick" in qwen_plan.prompt_contract.lower()


class TestStrictValueTypeValidation:
    """Re-audit P1#7: validate_profile_schema checked unknown keys but not
    value types — a YAML boolean field given as a truthy string (e.g.
    ``streaming.supported: "false"``) previously passed validation and ran
    as if the string were "supported". Same gap existed for match.backends
    (unvalidated free-text) and match.model_patterns (a non-string entry
    reached re.compile() uncaught — TypeError, not caught by
    load_profiles()'s except clause, crashing the whole directory load)."""

    def _base(self, **overrides) -> dict:
        data = {
            "schema_version": "interop.model-profile.v2",
            "id": "test-profile",
            "match": {"model_patterns": ["^test$"], "backends": ["ollama"]},
            "tool_calling": {
                "presentation": {"mode": "prompted"},
                "extraction": {"parser": "tool_call_envelope", "envelope": "tool_call"},
                "choice": {"automatic": False, "required": True, "named": False, "parallel": False},
            },
        }
        data.update(overrides)
        return data

    def test_string_bool_in_streaming_supported_rejected(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base(streaming={"supported": "false"})
        issues = validate_profile_schema(data)
        assert any("streaming.supported" in i for i in issues), issues

    def test_string_bool_in_reasoning_supported_rejected(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base(reasoning={"supported": "true"})
        issues = validate_profile_schema(data)
        assert any("reasoning.supported" in i for i in issues), issues

    def test_string_bool_in_choice_fields_rejected(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base()
        data["tool_calling"]["choice"]["parallel"] = "yes"
        issues = validate_profile_schema(data)
        assert any("tool_calling.choice.parallel" in i for i in issues), issues

    def test_real_booleans_accepted(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base(streaming={"supported": True}, reasoning={"supported": False})
        assert validate_profile_schema(data) == []

    def test_unknown_backend_name_rejected(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base()
        data["match"]["backends"] = ["ollama", "not_a_real_backend"]
        issues = validate_profile_schema(data)
        assert any("match.backends" in i for i in issues), issues

    def test_non_string_model_pattern_rejected_not_crashed(self) -> None:
        """Previously this reached re.compile(123) uncaught, raising
        TypeError — a type load_profiles() doesn't catch, so one bad
        profile file could crash the entire directory load instead of
        being cleanly skipped with a validation issue."""
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base()
        data["match"]["model_patterns"] = ["^ok$", 123]
        issues = validate_profile_schema(data)  # must not raise
        assert any("match.model_patterns" in i for i in issues), issues

    def test_non_string_backends_list_rejected_not_crashed(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base()
        data["match"]["backends"] = ["ollama", None]
        issues = validate_profile_schema(data)  # must not raise
        assert any("match.backends" in i for i in issues), issues

    def test_unknown_schema_format_rejected(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base()
        data["tool_calling"]["presentation"]["schema_format"] = "made_up_format"
        issues = validate_profile_schema(data)
        assert any("schema_format" in i for i in issues), issues

    def test_non_string_revision_rejected(self) -> None:
        from agent_interop.model.profiles_v2 import validate_profile_schema

        data = self._base(revision=1)
        issues = validate_profile_schema(data)
        assert any("revision" in i for i in issues), issues


class TestDuplicateProfileIdFailsLoudly:
    """Re-audit P1#7: a duplicate profile id used to be logged and
    silently dropped — the second file's declarations simply vanished
    with no load-time signal. It must now fail loudly enough that the
    release gate (scripts/release.sh stage 13) catches it."""

    def test_add_profile_raises_on_duplicate_id(self) -> None:
        from agent_interop.model.profiles_v2 import ModelProfile, ProfileIndex

        data = {
            "schema_version": "interop.model-profile.v2",
            "id": "dup-test",
            "match": {"model_patterns": ["^dup$"], "backends": ["ollama"]},
        }
        index = ProfileIndex()
        profile_a = ModelProfile.from_yaml(data, matched_by="a")
        profile_b = ModelProfile.from_yaml(data, matched_by="b")
        index.add_profile(profile_a, data)
        with pytest.raises(ValueError, match="duplicate profile id"):
            index.add_profile(profile_b, data)
        # The first registration must survive — one raised duplicate must
        # not corrupt or remove the already-loaded profile.
        assert index.get_by_id("dup-test") is not None
        assert len(index) == 1


class TestProfileTierPrecedence:
    """Re-audit P1#8: project/user profile directories were documented
    (module docstring) but never actually discovered or loaded — and the
    module had no concept of a profile "overriding" another at all,
    since add_profile() only knew how to reject or accept. These tests
    exercise the real tiered load_profiles() default (project > user >
    builtin) via a monkeypatched cwd/XDG dir, and the override semantics
    in ProfileIndex.add_profile directly."""

    def _profile_yaml(self, profile_id: str, declared_tokens: int) -> str:
        return (
            "schema_version: interop.model-profile.v2\n"
            f"id: {profile_id}\n"
            "match:\n"
            "  model_patterns: ['^tier-test-model$']\n"
            "  backends: [ollama]\n"
            "tool_calling:\n"
            "  presentation: {mode: prompted}\n"
            "  extraction: {parser: tool_call_envelope, envelope: tool_call}\n"
            "  choice: {automatic: false, required: true, named: false, parallel: false}\n"
            f"context: {{declared_tokens: {declared_tokens}, safe_tokens: 1024}}\n"
        )

    def test_project_tier_overrides_builtin_tier_same_id(self, tmp_path, monkeypatch) -> None:
        from agent_interop.model.profiles_v2 import load_profiles

        builtin_dir = tmp_path / "builtin"
        project_dir = tmp_path / "project" / ".interop" / "profiles"
        builtin_dir.mkdir(parents=True)
        project_dir.mkdir(parents=True)
        (builtin_dir / "x.yaml").write_text(self._profile_yaml("shared-id", 4096))
        (project_dir / "x.yaml").write_text(self._profile_yaml("shared-id", 999999))

        monkeypatch.chdir(tmp_path / "project")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", lambda: [builtin_dir])

        index = load_profiles()
        profile = index.get_by_id("shared-id")
        assert profile is not None
        # The project-tier value won, not the builtin-tier value.
        assert profile.declared_tokens == 999999
        assert len(index) == 1  # not two separate "shared-id" entries

    def test_user_tier_overrides_builtin_but_not_project(self, tmp_path, monkeypatch) -> None:
        from agent_interop.model.profiles_v2 import load_profiles

        builtin_dir = tmp_path / "builtin"
        project_dir = tmp_path / "proj" / ".interop" / "profiles"
        user_dir = tmp_path / "xdg" / "interop" / "profiles"
        for d in (builtin_dir, project_dir, user_dir):
            d.mkdir(parents=True)
        (builtin_dir / "x.yaml").write_text(self._profile_yaml("shared-id", 2048))
        (user_dir / "x.yaml").write_text(self._profile_yaml("shared-id", 4096))
        (project_dir / "x.yaml").write_text(self._profile_yaml("shared-id", 8192))

        monkeypatch.chdir(tmp_path / "proj")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", lambda: [builtin_dir])

        index = load_profiles()
        profile = index.get_by_id("shared-id")
        assert profile is not None
        assert profile.declared_tokens == 8192  # project wins over user+builtin

    def test_add_profile_same_tier_duplicate_still_raises_even_with_tiers(self) -> None:
        from agent_interop.model.profiles_v2 import ModelProfile, ProfileIndex

        data = {
            "schema_version": "interop.model-profile.v2",
            "id": "dup-tier-test",
            "match": {"model_patterns": ["^dup$"], "backends": ["ollama"]},
        }
        index = ProfileIndex()
        index.add_profile(ModelProfile.from_yaml(data, matched_by="a"), data, tier="project")
        with pytest.raises(ValueError, match="duplicate profile id"):
            index.add_profile(ModelProfile.from_yaml(data, matched_by="b"), data, tier="project")

    def test_add_profile_cross_tier_duplicate_does_not_raise(self) -> None:
        """A builtin-tier id already claimed by a higher-precedence
        project-tier profile is silently superseded, not an error — this
        is the whole point of tiered override."""
        from agent_interop.model.profiles_v2 import ModelProfile, ProfileIndex

        data = {
            "schema_version": "interop.model-profile.v2",
            "id": "override-test",
            "match": {"model_patterns": ["^x$"], "backends": ["ollama"]},
        }
        index = ProfileIndex()
        index.add_profile(ModelProfile.from_yaml(data, matched_by="proj"), data, tier="project")
        index.add_profile(ModelProfile.from_yaml(data, matched_by="builtin"), data, tier="builtin")
        assert len(index) == 1
        winner = index.get_by_id("override-test")
        assert winner is not None
        assert winner.matched_by == "proj"


class TestAmbiguousAutoFallbackTierBan:
    """P0-3 fix: a *builtin*-tier profile must never be able to enable
    'auto' for the ambiguous whole_message_json fallback dialect — see
    profiles_v2._check_builtin_tier_restrictions. Only a project/user-tier
    override (an operator's own, locally-installed profile) may set it,
    and even then extraction.py additionally requires a live per-request
    execution nonce before trusting a recovered candidate (see
    tests/test_whole_message_json.py)."""

    def _profile_yaml_with_auto_fallback(self, profile_id: str) -> str:
        return (
            "schema_version: interop.model-profile.v2\n"
            f"id: {profile_id}\n"
            "match:\n"
            "  model_patterns: ['^ambiguous-auto-test$']\n"
            "  backends: [ollama]\n"
            "tool_calling:\n"
            "  presentation: {mode: prompted}\n"
            "  extraction:\n"
            "    parser: tool_call_envelope\n"
            "    envelope: tool_call\n"
            "    fallbacks:\n"
            "      - parser: whole_message_json\n"
            "        tool_choice_modes: [auto, required, named]\n"
            "  choice: {automatic: true, required: true, named: true, parallel: false}\n"
        )

    def test_builtin_tier_profile_with_auto_fallback_is_rejected(self, tmp_path, monkeypatch) -> None:
        from agent_interop.model.profiles_v2 import load_profiles

        builtin_dir = tmp_path / "builtin"
        builtin_dir.mkdir(parents=True)
        (builtin_dir / "bad.yaml").write_text(self._profile_yaml_with_auto_fallback("bad-auto"))

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", lambda: [builtin_dir])

        index = load_profiles()
        assert index.get_by_id("bad-auto") is None

    def test_project_tier_profile_with_auto_fallback_is_accepted(self, tmp_path, monkeypatch) -> None:
        """The same YAML that's rejected at builtin tier loads fine as a
        project-tier override — the ban is tier-specific, not a blanket
        schema rejection. Runtime nonce verification (not load-time
        validation) is what protects this path once loaded."""
        from agent_interop.model.profiles_v2 import load_profiles

        project_dir = tmp_path / "proj" / ".interop" / "profiles"
        project_dir.mkdir(parents=True)
        (project_dir / "override.yaml").write_text(
            self._profile_yaml_with_auto_fallback("allowed-auto")
        )

        monkeypatch.chdir(tmp_path / "proj")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", list)

        index = load_profiles()
        profile = index.get_by_id("allowed-auto")
        assert profile is not None
        assert any(
            fs.parser_id == "whole_message_json" and "auto" in fs.allowed_tool_choice_modes
            for fs in profile.tool_behavior.fallback_strategies
        )

    def test_required_named_only_auto_fallback_accepted_at_builtin_tier(self, tmp_path, monkeypatch) -> None:
        """Sanity check the ban is scoped to 'auto' specifically — the same
        dialect restricted to required/named (no ambiguity, the client
        already asked for a specific tool) is fine at builtin tier."""
        from agent_interop.model.profiles_v2 import load_profiles

        builtin_dir = tmp_path / "builtin"
        builtin_dir.mkdir(parents=True)
        yaml_text = self._profile_yaml_with_auto_fallback("fine-auto").replace(
            "tool_choice_modes: [auto, required, named]",
            "tool_choice_modes: [required, named]",
        )
        (builtin_dir / "fine.yaml").write_text(yaml_text)

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", lambda: [builtin_dir])

        index = load_profiles()
        assert index.get_by_id("fine-auto") is not None


class TestStrictProjectUserTierLoading:
    """P1 fix: an invalid builtin-tier profile logs-and-skips (forward
    compat with a possibly-older Interop reading a newer packaged file),
    but a project/user-tier profile is something an operator deliberately
    placed there — silently dropping it with only a log line risks their
    intended override (or safety restriction) never applying. Those tiers
    now raise ProfileLoadError immediately, naming the exact file."""

    def test_invalid_project_tier_profile_raises(self, tmp_path, monkeypatch) -> None:
        from agent_interop.model.profiles_v2 import ProfileLoadError, load_profiles

        project_dir = tmp_path / "proj" / ".interop" / "profiles"
        project_dir.mkdir(parents=True)
        (project_dir / "bad.yaml").write_text(
            "schema_version: interop.model-profile.v2\n"
            "id: bad-project-profile\n"
            "unknown_top_level_field: true\n"
        )

        monkeypatch.chdir(tmp_path / "proj")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", list)

        with pytest.raises(ProfileLoadError, match="bad.yaml"):
            load_profiles()

    def test_invalid_user_tier_profile_raises(self, tmp_path, monkeypatch) -> None:
        from agent_interop.model.profiles_v2 import ProfileLoadError, load_profiles

        user_dir = tmp_path / "xdg" / "interop" / "profiles"
        user_dir.mkdir(parents=True)
        (user_dir / "bad.yaml").write_text(
            "schema_version: interop.model-profile.v2\n"
            "id: bad-user-profile\n"
            "tool_calling:\n"
            "  extraction:\n"
            "    parser: nonexistent_parser_id\n"
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", list)

        with pytest.raises(ProfileLoadError, match="bad.yaml"):
            load_profiles()

    def test_invalid_builtin_tier_profile_still_warns_and_skips(self, tmp_path, monkeypatch) -> None:
        """The pre-existing, less-strict behavior is preserved for builtin
        specifically — this is a deliberate asymmetry, not an oversight."""
        from agent_interop.model.profiles_v2 import load_profiles

        builtin_dir = tmp_path / "builtin"
        builtin_dir.mkdir(parents=True)
        (builtin_dir / "bad.yaml").write_text(
            "schema_version: interop.model-profile.v2\n"
            "id: bad-builtin-profile\n"
            "unknown_top_level_field: true\n"
        )

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", lambda: [builtin_dir])

        index = load_profiles()  # must not raise
        assert index.get_by_id("bad-builtin-profile") is None

    def test_malformed_yaml_at_project_tier_also_raises(self, tmp_path, monkeypatch) -> None:
        """Not just schema-validation failures — a YAML parse error in a
        project-tier file must raise too, not silently vanish."""
        from agent_interop.model.profiles_v2 import ProfileLoadError, load_profiles

        project_dir = tmp_path / "proj" / ".interop" / "profiles"
        project_dir.mkdir(parents=True)
        (project_dir / "broken.yaml").write_text("{ not: valid: yaml: [")

        monkeypatch.chdir(tmp_path / "proj")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-user-profiles-here"))

        import agent_interop.model.profiles_v2 as p2
        monkeypatch.setattr(p2, "_builtin_profile_dirs", list)

        with pytest.raises(ProfileLoadError, match="broken.yaml"):
            load_profiles()
