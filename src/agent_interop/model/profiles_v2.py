"""Model profile loader — loads YAML profiles from directories.

Profile precedence (highest first):
1. Session override
2. Explicit profile ID
3. Project profiles (./.interop/profiles/*.yaml, cwd-relative)
4. User profiles (<XDG config dir>/interop/profiles/*.yaml)
5. Built-in profile (packaged interop.data.profiles resources)
6. Backend metadata
7. Generic fallback profile

Verified conformance (interop.model.registry.ModelProfileRegistry) is not
its own tier — it's a narrow overlay applied ON TOP OF whichever tier above
resolved, refining confidence/capability signal without discarding the base
profile's executable fields (parser, contract template, fallback
strategies, context limits). See ModelProfileRegistry._apply_conformance_overlay.

Same-id collision across tiers 3/4/5 is a deliberate override (project >
user > builtin); a collision WITHIN one tier is treated as an authoring
error (see ProfileIndex.add_profile).

Schema v2: profiles are EXECUTABLE CONTRACTS, not descriptive metadata — a
field that doesn't affect runtime behavior must not exist in the schema.
v1 carried several fields that were parsed from YAML but never read again
anywhere (profile-level repair settings, reasoning.parser/send_to_client,
context.reserve_output_tokens, streaming.tool_arguments_incremental,
template.source/fallback) — those are removed here rather than kept as
inert-but-validated syntax, per the same principle that motivated the
whole_message_json fallback tier: declaring a capability a profile doesn't
actually have is worse than not declaring it. `load_profiles()` rejects
unknown fields and invalid references (see `validate_profile_schema`)
instead of silently ignoring them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from importlib.resources import files as _resources_files
except ImportError:
    from importlib_resources import files as _resources_files  # type: ignore

logger = logging.getLogger("agent_interop.profiles")

SCHEMA_VERSION = "interop.model-profile.v2"


class ProfileLoadError(ValueError):
    """A project/user-tier profile file failed to load.

    Builtin-tier profiles warn-and-skip on failure (forward-compat: an
    older Interop version encountering a newer-schema packaged profile
    shouldn't crash the whole load). A project/user-tier file is
    different — an operator deliberately placed it there and expects it to
    take effect; silently skipping it with only a log line means their
    intended override (or, worse, their intended SAFETY restriction —
    see the ambiguous-auto tier ban) may silently never apply. This raises
    immediately instead, naming the exact file and issues.
    """


def _builtin_profile_dirs() -> list[Path]:
    """Return directories to search for built-in profiles.

    Uses importlib.resources for wheel-compatible access, with a fallback
    to the source tree for development.  Deduplicates resolved paths so
    that the same directory is never traversed twice — previously
    editable installs reported each profile twice.
    """
    seen: set[Path] = set()
    ordered: list[Path] = []
    try:
        pkg = _resources_files("agent_interop.data").joinpath("profiles")
        if pkg.is_dir():
            resolved = Path(str(pkg)).resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)
    except (ImportError, ModuleNotFoundError, TypeError):
        pass
    # Fallback to source-tree data/profiles/ for editable installs
    src_profiles = Path(__file__).resolve().parent.parent / "data" / "profiles"
    if src_profiles.is_dir():
        resolved = src_profiles.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def _project_profile_dir() -> Path:
    """Project-level profile directory: ``./.interop/profiles/`` relative
    to the current working directory. Lets a repo pin or override a
    model's behavioral profile alongside its code (checked into version
    control), without touching the packaged builtins."""
    return Path.cwd() / ".interop" / "profiles"


def _user_profile_dir() -> Path:
    """User-level profile directory, under the XDG config directory
    (see interop.paths) — e.g. ``~/.config/interop/profiles/``. Lets one
    operator's local overrides apply across every project on that
    machine, at lower precedence than a project's own ``.interop/profiles/``."""
    from agent_interop.paths import config_dir

    return config_dir() / "profiles"


# ─── Extraction strategy (ordered fallback pipeline) ───────────────────────

_VALID_TOOL_CHOICE_MODES = frozenset({"auto", "required", "named", "none"})


@dataclass(frozen=True)
class ExtractionStrategy:
    """One fallback extraction tier, tried in order after the primary
    parser finds nothing.

    Generalizes what were previously two independent, hardcoded booleans
    (allow_generic_json_fallback, allow_whole_message_json) — each new
    fallback SHAPE a model needs no longer requires its own bespoke
    profile field, registry plumbing, and gateway call-site change; it's
    another entry in this list.
    """

    parser_id: str
    skip_when_native_present: bool = True
    allowed_tool_choice_modes: frozenset[str] = field(
        default_factory=lambda: frozenset({"auto", "required", "named"})
    )


@dataclass
class ToolBehaviorProfile:
    """Explicit tool capability dimensions.

    These are orthogonal: a model may support automatic tool selection
    while still requiring a textual dialect (Qwen), or consume native
    schemas but not return native structured calls.
    """

    presentation_mode: str = "native"  # native | prompted | textual | disabled
    # Which rendered instruction text a PROMPTED-mode model gets — see
    # model/contract_templates.py. None uses the universal default
    # template. Actually consulted by build_invocation_plan() (v1's
    # `template.contract_template` field was parsed but never read).
    contract_template_id: str | None = None
    extractor_id: str | None = None
    output_envelope: str | None = None
    fallback_strategies: tuple[ExtractionStrategy, ...] = ()

    # Choice capabilities
    supports_auto_choice: bool = True
    supports_required_choice: bool = True
    supports_named_choice: bool = True
    supports_parallel_calls: bool = False

    # Transport capabilities (orthogonal from choice)
    native_schema_support: bool = False
    native_response_support: bool = False


@dataclass
class ModelProfile:
    """Resolved model profile loaded from YAML."""

    schema_version: str = SCHEMA_VERSION
    id: str = ""
    matched_by: str = ""

    tool_behavior: ToolBehaviorProfile = field(default_factory=ToolBehaviorProfile)

    reasoning_supported: bool = False

    declared_tokens: int = 4096
    safe_tokens: int = 2048

    streaming_supported: bool = True

    profile_revision: str = ""

    @classmethod
    def from_yaml(cls, data: dict[str, Any], matched_by: str = "") -> ModelProfile:
        """Load from a parsed YAML dict.

        Callers must run ``validate_profile_schema(data)`` first — this
        only constructs the dataclass from data already known to be
        well-formed; it does not itself reject unknown or invalid fields.
        """
        tc = data.get("tool_calling", {})
        r = data.get("reasoning", {})
        ctx = data.get("context", {})
        st = data.get("streaming", {})

        presentation = tc.get("presentation", {})
        extraction = tc.get("extraction", {})
        choice = tc.get("choice", {})

        fallback_strategies = tuple(
            ExtractionStrategy(
                parser_id=fb["parser"],
                skip_when_native_present=fb.get("skip_when_native_present", True),
                allowed_tool_choice_modes=frozenset(
                    fb.get("tool_choice_modes", ["auto", "required", "named"])
                ),
            )
            for fb in extraction.get("fallbacks", [])
        )

        tool_behavior = ToolBehaviorProfile(
            presentation_mode=presentation.get("mode", "native"),
            contract_template_id=presentation.get("contract_template"),
            extractor_id=extraction.get("parser"),
            output_envelope=extraction.get("envelope"),
            fallback_strategies=fallback_strategies,
            supports_auto_choice=choice.get("automatic", True),
            supports_required_choice=choice.get("required", True),
            supports_named_choice=choice.get("named", True),
            supports_parallel_calls=choice.get("parallel", False),
            native_schema_support=presentation.get("schema_format") == "openai_function",
            native_response_support=presentation.get("mode") == "native",
        )

        return cls(
            schema_version=data.get("schema_version", cls.schema_version),
            id=data.get("id", ""),
            matched_by=matched_by,
            tool_behavior=tool_behavior,
            reasoning_supported=r.get("supported", cls.reasoning_supported),
            declared_tokens=ctx.get("declared_tokens", cls.declared_tokens),
            safe_tokens=ctx.get("safe_tokens", cls.safe_tokens),
            streaming_supported=st.get("supported", cls.streaming_supported),
            profile_revision=data.get("revision", ""),
        )


# ─── Strict schema validation ───────────────────────────────────────────────

_TOP_LEVEL_FIELDS = frozenset({
    "schema_version", "id", "match", "tool_calling", "reasoning", "context",
    "streaming", "revision",
})
_MATCH_FIELDS = frozenset({"model_patterns", "backends"})
_TOOL_CALLING_FIELDS = frozenset({"presentation", "extraction", "choice"})
_PRESENTATION_FIELDS = frozenset({"mode", "schema_format", "contract_template"})
_EXTRACTION_FIELDS = frozenset({"parser", "envelope", "fallbacks"})
_FALLBACK_FIELDS = frozenset({"parser", "skip_when_native_present", "tool_choice_modes"})
_CHOICE_FIELDS = frozenset({"automatic", "required", "named", "parallel"})
_REASONING_FIELDS = frozenset({"supported"})
_CONTEXT_FIELDS = frozenset({"declared_tokens", "safe_tokens"})
_STREAMING_FIELDS = frozenset({"supported"})
_VALID_PRESENTATION_MODES = frozenset({"native", "prompted", "textual", "disabled"})


_VALID_BACKENDS = frozenset({
    "ollama", "vllm", "llamacpp", "openai", "anthropic", "openai_compatible", "*",
})
_VALID_SCHEMA_FORMATS = frozenset({"openai_function"})


def _check_unknown(obj: Any, allowed: frozenset[str], path: str, issues: list[str]) -> bool:
    """Returns True if obj is a well-formed dict with no unknown keys."""
    if not isinstance(obj, dict):
        issues.append(f"'{path}' must be an object, got {type(obj).__name__}")
        return False
    unknown = set(obj.keys()) - allowed
    if unknown:
        issues.append(f"Unknown field(s) in '{path}': {sorted(unknown)}")
    return True


def _check_bool(obj: dict[str, Any], key: str, path: str, issues: list[str]) -> None:
    """A YAML value like ``"false"`` or ``1`` parses fine but is truthy in
    Python — silently inverting the author's intent (e.g. a fallback
    profile that means to declare ``streaming.supported: false`` but ships
    ``streaming.supported: "false"`` would run as if streaming were
    supported). Only real booleans are accepted; absence is fine, any
    present-but-wrong-type value is not."""
    if key not in obj:
        return
    if not isinstance(obj[key], bool):
        issues.append(f"'{path}.{key}' must be a boolean, got {obj[key]!r}")


def _check_string_list(obj: Any, path: str, issues: list[str]) -> list[Any]:
    """Validate obj is a list of strings, returning it (or [] if invalid)
    so callers can still safely iterate without a second isinstance check.
    A non-string entry here previously reached ``re.compile()``/set
    operations uncaught, raising TypeError instead of a clean validation
    issue — TypeError is not in load_profiles()'s caught exception set, so
    it crashed the entire profile directory load, not just this file."""
    if not isinstance(obj, list) or not all(isinstance(v, str) for v in obj):
        issues.append(f"'{path}' must be a list of strings, got {obj!r}")
        return []
    return obj


def validate_profile_schema(data: dict[str, Any], *, source: str = "") -> list[str]:
    """Strictly validate a raw profile dict. Returns a list of issues
    (empty = valid). A profile with any issue must not be loaded — an
    executable contract that silently ignores an unknown or malformed
    field is worse than one that fails to load at all.
    """
    issues: list[str] = []
    prefix = f"{source}: " if source else ""

    if not _check_unknown(data, _TOP_LEVEL_FIELDS, "<root>", issues):
        return [prefix + i for i in issues]

    if not isinstance(data.get("id", ""), str) or not data.get("id"):
        issues.append("'id' is required and must be a non-empty string")

    if "revision" in data and not isinstance(data["revision"], str):
        issues.append(f"'revision' must be a string, got {data['revision']!r}")

    match = data.get("match", {})
    if _check_unknown(match, _MATCH_FIELDS, "match", issues):
        if "model_patterns" in match:
            for pat in _check_string_list(match["model_patterns"], "match.model_patterns", issues):
                try:
                    re.compile(pat)
                except re.error as exc:
                    issues.append(f"Invalid match.model_patterns regex {pat!r}: {exc}")
        if "backends" in match:
            backends = _check_string_list(match["backends"], "match.backends", issues)
            unknown_backends = set(backends) - _VALID_BACKENDS
            if unknown_backends:
                issues.append(
                    f"Unknown match.backends {sorted(unknown_backends)} "
                    f"(valid: {sorted(_VALID_BACKENDS)})"
                )

    tc = data.get("tool_calling", {})
    if _check_unknown(tc, _TOOL_CALLING_FIELDS, "tool_calling", issues):
        presentation = tc.get("presentation", {})
        if _check_unknown(presentation, _PRESENTATION_FIELDS, "tool_calling.presentation", issues):
            mode = presentation.get("mode", "native")
            if mode not in _VALID_PRESENTATION_MODES:
                issues.append(
                    f"Invalid tool_calling.presentation.mode {mode!r} "
                    f"(valid: {sorted(_VALID_PRESENTATION_MODES)})"
                )
            contract_template = presentation.get("contract_template")
            if contract_template is not None:
                from agent_interop.model.contract_templates import CONTRACT_TEMPLATES
                if contract_template not in CONTRACT_TEMPLATES:
                    issues.append(
                        f"Unknown tool_calling.presentation.contract_template "
                        f"{contract_template!r} (known: {sorted(CONTRACT_TEMPLATES)})"
                    )
            schema_format = presentation.get("schema_format")
            if schema_format is not None and schema_format not in _VALID_SCHEMA_FORMATS:
                issues.append(
                    f"Unknown tool_calling.presentation.schema_format {schema_format!r} "
                    f"(valid: {sorted(_VALID_SCHEMA_FORMATS)}) — an unrecognized value "
                    "silently means 'no native schema support' rather than erroring"
                )

        extraction = tc.get("extraction", {})
        if _check_unknown(extraction, _EXTRACTION_FIELDS, "tool_calling.extraction", issues):
            from agent_interop.extraction import get_default_registry
            registry = get_default_registry()

            parser = extraction.get("parser")
            if parser is not None:
                try:
                    registry.get(parser)
                except ValueError:
                    issues.append(f"Unknown tool_calling.extraction.parser {parser!r}")

            fallbacks = extraction.get("fallbacks", [])
            if not isinstance(fallbacks, list):
                issues.append("'tool_calling.extraction.fallbacks' must be a list")
            else:
                for i, fb in enumerate(fallbacks):
                    path = f"tool_calling.extraction.fallbacks[{i}]"
                    if not _check_unknown(fb, _FALLBACK_FIELDS, path, issues):
                        continue
                    fb_parser = fb.get("parser")
                    if not fb_parser:
                        issues.append(f"{path} missing required 'parser'")
                    else:
                        try:
                            registry.get(fb_parser)
                        except ValueError:
                            issues.append(f"{path}: unknown parser {fb_parser!r}")
                    _check_bool(fb, "skip_when_native_present", path, issues)
                    modes = fb.get("tool_choice_modes", ["auto", "required", "named"])
                    if not isinstance(modes, list) or not set(modes) <= _VALID_TOOL_CHOICE_MODES:
                        issues.append(
                            f"{path}: invalid tool_choice_modes {modes!r} "
                            f"(valid: {sorted(_VALID_TOOL_CHOICE_MODES)})"
                        )

        choice = tc.get("choice", {})
        if _check_unknown(choice, _CHOICE_FIELDS, "tool_calling.choice", issues):
            for field_name in _CHOICE_FIELDS:
                _check_bool(choice, field_name, "tool_calling.choice", issues)

    r = data.get("reasoning", {})
    if _check_unknown(r, _REASONING_FIELDS, "reasoning", issues):
        _check_bool(r, "supported", "reasoning", issues)

    ctx = data.get("context", {})
    if _check_unknown(ctx, _CONTEXT_FIELDS, "context", issues):
        declared = ctx.get("declared_tokens", 4096)
        safe = ctx.get("safe_tokens", 2048)
        if not (isinstance(declared, int) and isinstance(safe, int)):
            issues.append("context.declared_tokens/safe_tokens must be integers")
        elif not (0 < safe <= declared):
            issues.append(
                f"context limits must satisfy 0 < safe_tokens ({safe}) <= "
                f"declared_tokens ({declared})"
            )

    st = data.get("streaming", {})
    if _check_unknown(st, _STREAMING_FIELDS, "streaming", issues):
        _check_bool(st, "supported", "streaming", issues)

    return [prefix + i for i in issues]


def _check_builtin_tier_restrictions(data: dict[str, Any], tier: str, *, source: str = "") -> list[str]:
    """Builtin-tier (packaged) profiles must never enable the ambiguous
    whole_message_json fallback dialect under tool_choice=auto.

    That dialect recovers a bare or fenced JSON object with no taught
    envelope — under ``auto`` there is no way to distinguish a genuine tool
    call from demonstration content the model happened to emit, and a
    *builtin* profile ships to every install by default, so enabling it
    there would mean a stock install executes ambiguous output with no
    operator having opted in. Only a project/user-tier profile (an
    operator's own override, loaded from their own machine) may set
    ``auto`` for this dialect — and even then, extraction additionally
    requires a live per-request execution nonce to match before treating a
    recovered candidate as real (see repair/invocation.py and
    extraction.py's ExtractorRegistry.extract).

    This check runs unconditionally (independent of load_profiles'
    ``strict`` flag) — it is a safety invariant, not a schema-strictness
    preference a caller should be able to relax.
    """
    if tier != "builtin":
        return []
    issues: list[str] = []
    fallbacks = data.get("tool_calling", {}).get("extraction", {}).get("fallbacks", [])
    if isinstance(fallbacks, list):
        for i, fb in enumerate(fallbacks):
            if not isinstance(fb, dict):
                continue
            if fb.get("parser") == "whole_message_json" and "auto" in fb.get("tool_choice_modes", []):
                issues.append(
                    f"tool_calling.extraction.fallbacks[{i}]: builtin-tier profiles "
                    "must not enable 'auto' for the ambiguous whole_message_json "
                    "fallback dialect — this is only permitted in a project/user-tier "
                    "profile override, where it additionally requires a live "
                    "per-request execution nonce at runtime"
                )
    prefix = f"{source}: " if source else ""
    return [prefix + i for i in issues]


# ─── Profile Index ──────────────────────────────────────────────────────────

_FALLBACK_PROFILE_ID = "generic-fallback"


class ProfileIndex:
    """Index of all loaded profiles with match-based resolution.

    Tracks a "tier" per profile (project > user > builtin, in precedence
    order — see load_profiles()) so a project/user profile can
    deliberately override a builtin profile of the same id, while two
    profiles declaring the same id within the SAME tier is still a real
    authoring bug that must fail loudly.
    """

    def __init__(self) -> None:
        self._profiles: list[tuple[ModelProfile, list[re.Pattern], list[str], str]] = []

    def add_profile(self, profile: ModelProfile, data: dict[str, Any], *, tier: str = "builtin") -> None:
        """Add a profile with its match rules.

        ``tier`` distinguishes an intentional cross-tier override from a
        same-tier collision. Callers add tiers in precedence order
        (highest first — see load_profiles()), so by the time a lower-tier
        duplicate is seen, the higher-tier profile for that id has already
        won and is left untouched; the lower-tier registration is simply
        not added. A duplicate id within the SAME tier still raises: two
        files at the same precedence level disagreeing about what one
        profile means has no correct resolution, so it must fail loudly
        rather than silently dropping whichever file lost the race — the
        release gate's file-count check (scripts/release.sh) only catches
        this because add_profile() fails loudly instead of swallowing it.
        """
        for existing_profile, _, _, existing_tier in self._profiles:
            if profile.id and profile.id == existing_profile.id:
                if existing_tier == tier:
                    raise ValueError(f"duplicate profile id {profile.id!r} within {tier!r} tier")
                # A higher-precedence tier already registered this id —
                # that's the intentional override target. This
                # registration is silently superseded, not an error.
                return
        match_section = data.get("match", {})
        patterns = [re.compile(p) for p in match_section.get("model_patterns", [])]
        # "openai_compatible" (underscore) matches UpstreamKind.OPENAI_COMPATIBLE.value
        # — a hyphenated "openai-compatible" here would silently never match
        # any openai_compatible-backed route for a profile that omits
        # match.backends entirely.
        backends = match_section.get("backends", ["ollama", "openai_compatible"])
        self._profiles.append((profile, patterns, backends, tier))

        # Sort: most specific (longest non-wildcard patterns, specific backends) first
        def specificity(item):
            _p, pats, backends, _tier = item
            # Reward specific backends (smaller set = more specific)
            backend_score = len(backends)
            # Penalize wildcards heavily; reward specific chars
            pattern_score = 0
            for pat in pats:
                s = pat.pattern
                if s == ".*":
                    pattern_score -= 1000  # generic catch-all
                elif s.startswith("^") and s.endswith("$"):
                    pattern_score += 500 + len(s)  # anchored exact
                elif s.startswith("^"):
                    pattern_score += 200 + len(s)  # anchored prefix
                else:
                    pattern_score += len(s)
            # Lower sort_key = more specific
            return (pattern_score, backend_score)  # high pattern_score = more specific

        self._profiles.sort(key=specificity, reverse=True)

    def resolve(self, model_name: str, backend: str = "ollama") -> ModelProfile | None:
        """Find the best matching profile for a model/backend pair.

        Returns a copy of the profile with matched_by set, so concurrent
        requests don't overwrite each other's matched_by on the shared object.

        The packaged generic-fallback profile is deliberately excluded here
        even though it declares a catch-all ``.*`` pattern: it must never be
        returned as an ordinary "builtin" match (confidence 0.8), only as
        ModelProfileRegistry's own conservative fallback tier (confidence
        0.1). Otherwise every unknown model would silently receive
        builtin-level confidence and become eligible for repairs meant only
        for models Interop actually has evidence about. Use
        ``get_fallback_profile()`` / ``get_by_id("generic-fallback")`` to
        reach it directly.
        """
        from dataclasses import replace

        for profile, patterns, backends, _tier in self._profiles:
            if profile.id == _FALLBACK_PROFILE_ID:
                continue
            if backend not in backends and "*" not in backends:
                continue
            for pattern in patterns:
                if pattern.search(model_name):
                    return replace(profile, matched_by=pattern.pattern)
        return None

    def get_by_id(self, profile_id: str) -> ModelProfile | None:
        """Look up a profile by its ID."""
        for profile, _, _, _tier in self._profiles:
            if profile.id == profile_id:
                return profile
        return None

    def list_ids(self) -> list[str]:
        """List all loaded profile IDs."""
        return [p.id for p, _, _, _tier in self._profiles]

    def list_profiles(self) -> list[ModelProfile]:
        """List all loaded profiles."""
        return [p for p, _, _, _tier in self._profiles]

    def __len__(self) -> int:
        return len(self._profiles)


# ─── Loader ──────────────────────────────────────────────────────────────────


def load_profiles(dirs: list[Path] | None = None, *, strict: bool = True) -> ProfileIndex:
    """Load all YAML profiles, applying project/user/builtin precedence.

    When ``dirs`` is omitted (the production default), profiles are loaded
    from three tiers in precedence order — a profile id declared in a
    higher tier overrides the same id in a lower one (see
    ``ProfileIndex.add_profile``):

    1. Project: ``./.interop/profiles/*.yaml`` (relative to cwd)
    2. User: ``<XDG config dir>/interop/profiles/*.yaml``
    3. Builtin: the packaged ``interop.data.profiles`` resources

    Passing an explicit ``dirs`` list (tests, the release gate's isolated
    wheel check) treats every directory as the single "builtin" tier —
    override semantics only matter for the real multi-tier default.

    ``strict`` (default True) rejects a profile that fails
    ``validate_profile_schema`` — unknown fields, invalid parser/template
    references, or contradictory settings — instead of silently loading
    it. Set False only for tests intentionally exercising the loader
    against non-conforming input.
    """
    index = ProfileIndex()
    if dirs is not None:
        tiered_dirs: list[tuple[Path, str]] = [(d, "builtin") for d in dirs]
    else:
        tiered_dirs = (
            [(_project_profile_dir(), "project")]
            + [(_user_profile_dir(), "user")]
            + [(d, "builtin") for d in _builtin_profile_dirs()]
        )

    for d, tier in tiered_dirs:
        if not d.exists():
            continue
        for fpath in sorted(d.glob("*.yaml")):
            try:
                raw = fpath.read_text()
                data = yaml.safe_load(raw)
                if not isinstance(data, dict):
                    continue
                if not data.get("schema_version", "").startswith("interop.model-profile"):
                    continue
                if data.get("schema_version") != SCHEMA_VERSION:
                    logger.warning(
                        "Skipping profile %s: schema_version %r is not the "
                        "supported %r",
                        fpath, data.get("schema_version"), SCHEMA_VERSION,
                    )
                    continue
                issues: list[str] = []
                if strict:
                    issues = validate_profile_schema(data, source=fpath.name)
                issues += _check_builtin_tier_restrictions(data, tier, source=fpath.name)
                if issues:
                    if tier != "builtin":
                        # An operator deliberately placed this file — fail
                        # loudly rather than silently skipping it with only
                        # a log line (see ProfileLoadError).
                        raise ProfileLoadError(
                            f"{fpath} ({tier} tier): " + "; ".join(issues)
                        )
                    for issue in issues:
                        logger.warning("Profile validation failed: %s", issue)
                    continue
                profile = ModelProfile.from_yaml(data, matched_by=fpath.stem)
                index.add_profile(profile, data, tier=tier)
            except ProfileLoadError:
                raise
            except (yaml.YAMLError, OSError, ValueError, KeyError) as exc:
                if tier != "builtin":
                    raise ProfileLoadError(
                        f"{fpath} ({tier} tier): failed to load: {exc}"
                    ) from exc
                # Builtin tier: log and skip bad profiles (forward-compat).
                logger.warning("Failed to load profile %s: %s", fpath, exc)

    return index


def get_fallback_profile() -> ModelProfile:
    """Return the generic fallback profile used when nothing matches."""
    return ModelProfile(
        id=_FALLBACK_PROFILE_ID,
        tool_behavior=ToolBehaviorProfile(
            presentation_mode="prompted",
            extractor_id="tool_call_envelope",
            output_envelope="tool_call",
            supports_auto_choice=False,
            supports_required_choice=True,
            supports_named_choice=False,
            supports_parallel_calls=False,
        ),
        reasoning_supported=False,
        declared_tokens=4096,
        safe_tokens=2048,
    )
