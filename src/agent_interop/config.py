"""Route-based configuration for Interop.

Replaces the single-model ``InteropConfig`` with a multi-route
configuration that determines the model, backend, wire protocol,
tool mode, and repair settings per request.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolMode(Enum):
    """How tools are presented to the upstream model."""

    AUTO = "auto"
    NATIVE = "native"
    PROMPTED = "prompted"
    TEXTUAL = "textual"
    DISABLED = "disabled"


class ToolSurfaceMode(str, Enum):
    """How much of the client tool registry is visible to a model."""

    TRANSPARENT = "transparent"
    SCOPED = "scoped"
    DYNAMIC = "dynamic"


class UpstreamKind(Enum):
    """The inference server software."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class UpstreamProtocol(str, Enum):
    """Wire protocol spoken by the upstream."""

    OLLAMA_CHAT = "ollama_chat"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    ANTHROPIC_MESSAGES = "anthropic_messages"


_DEFAULT_WIRE_PROTOCOL_BY_KIND: dict[UpstreamKind, UpstreamProtocol] = {
    UpstreamKind.OLLAMA: UpstreamProtocol.OLLAMA_CHAT,
    UpstreamKind.VLLM: UpstreamProtocol.OPENAI_CHAT,
    UpstreamKind.LLAMACPP: UpstreamProtocol.OPENAI_CHAT,
    UpstreamKind.OPENAI: UpstreamProtocol.OPENAI_CHAT,
    UpstreamKind.ANTHROPIC: UpstreamProtocol.ANTHROPIC_MESSAGES,
    UpstreamKind.OPENAI_COMPATIBLE: UpstreamProtocol.OPENAI_CHAT,
}


def default_wire_protocol_for_kind(kind: UpstreamKind) -> UpstreamProtocol:
    """The wire protocol a backend kind actually speaks by default.

    Single source of truth for this mapping — previously duplicated
    independently in cli.py's ``_resolve_wire_protocol`` and
    server/app.py's ``create_app_from_env``, while the YAML config loader
    (``load_config_from_dict``) had no kind-aware default at all: it
    always defaulted a missing ``wire_protocol`` to "openai_chat"
    regardless of ``kind``, so an Ollama route with wire_protocol omitted
    silently got OpenAI-Chat framing sent to an Ollama endpoint.
    """
    return _DEFAULT_WIRE_PROTOCOL_BY_KIND.get(kind, UpstreamProtocol.OPENAI_CHAT)


class TranslationMode(Enum):
    """How the route translates between client and upstream protocols.

    Explicit semantics:
    - CANONICAL: Full decode into canonical ABI, then re-encode for the
      client protocol. Maximum compatibility and repair capability.
      This is the default and recommended mode.
    - RAW_PASSTHROUGH: **NOT IMPLEMENTED** — config-time-rejected.  The
      option is retained for API stability but routing a request with
      this mode is a configuration error.
    - REPAIR_AWARE_SAME_PROTOCOL: **NOT IMPLEMENTED** —
      config-time-rejected for the same reason.

    Only ``CANONICAL`` is currently honored.  Setting a non-canonical
    value will fail ``validate_config`` with a clear error rather than
    silently substituting canonical behavior.

    Do NOT call a route "passthrough" when Interop mutates its contents.
    """

    RAW_PASSTHROUGH = "raw_passthrough"
    REPAIR_AWARE_SAME_PROTOCOL = "repair_aware_same_protocol"
    CANONICAL = "canonical"


# ─── Route types ──────────────────────────────────────────────────────────────


@dataclass
class UpstreamConfig:
    """Describes how to reach and authenticate with an inference server."""

    kind: UpstreamKind
    base_url: str
    wire_protocol: UpstreamProtocol
    api_key_env: str | None = None
    static_headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 120.0
    auth: dict[str, str] = field(default_factory=dict)


@dataclass
class RepairConfig:
    """Per-route tool-call repair settings."""

    max_regenerations: int = 0
    max_added_latency_ms: int = 15000
    max_repair_input_bytes: int = 65536
    malformed_json: str = "safe"
    unknown_tool: str = "safe_normalization"
    # COMPATIBILITY_PACK, not SCHEMA_ONLY, is the default: a registered
    # pack only ever activates for a resolved, sufficiently-populated
    # client identity (see repair/aliases.py's module docstring and
    # _is_key_sufficiently_populated) — it is maintainer-authored and
    # reviewed, not learned/dynamic, so there is nothing for an unknown or
    # unresolved client to inherit by enabling this. Turning it on by
    # default is what makes the field-alias repairs Interop ships for
    # claude_code/hermes_agent/etc. actually apply out of the box, instead
    # of requiring every operator to discover and set this per route.
    field_aliases: str = "compatibility_pack"
    batch_policy: str = "atomic"


@dataclass
class ToolSurfaceConfig:
    mode: ToolSurfaceMode = ToolSurfaceMode.DYNAMIC
    toolsets: tuple[str, ...] = ()
    allow_tools: tuple[str, ...] = ()
    deny_tools: tuple[str, ...] = ()
    max_initial_tools: int = 8
    max_schema_tokens: int = 8192
    max_selection_rounds: int = 1
    selector: str = "lexical"
    allow_embedding_selector: bool = False


@dataclass
class ContextConfig:
    strategy: str = "auto"
    output_reserve_tokens: int = 4096
    context_limit_tokens: int = 0
    allow_tool_reduction: bool = True
    allow_history_compaction: bool = True
    allow_controller_decomposition: bool = True


@dataclass
class QualificationConfig:
    """When bounded model qualification is allowed to run."""

    # Programmatic/legacy routes stay opt-in. Schema-v2 configuration opts
    # into the target blocking behavior below, preserving existing callers.
    bootstrap: str = "on_demand"
    full_battery: str = "on_demand"
    cache_by_digest: bool = True


@dataclass
class DiagnosticsConfig:
    """Privacy-preserving replay/diagnostic capture policy."""

    capture: str = "failures"
    content_mode: str = "metadata_only"
    max_case_bytes: int = 1024 * 1024
    max_frames: int = 128
    retention_count: int = 100
    # Legacy/programmatic configs retain in-memory capture. Schema-v2 config
    # enables durable, sanitized case retention by default below.
    persist: bool = False


@dataclass
class ControllerConfig:
    enabled: bool = True
    route_id: str = ""
    auto_select_route: bool = True
    minimum_controller_level: str = "L3"
    # Schema-v2 config enables this below; legacy programmatic routes retain
    # their historical behavior until operators opt into controller gating.
    require_verified: bool = False
    max_controller_turns: int = 4
    max_primary_turns: int = 4
    allow_primary_tool_calls: bool = False
    preserve_primary_reasoning: bool = False


@dataclass
class CompatibilityConfig:
    mode: str = "auto"
    allow_direct: bool = True
    allow_adapted: bool = True
    allow_controlled: bool = True
    # Schema-v2 routes buffer unknown/non-direct streaming attempts until the
    # same ladder that protects non-streaming output has accepted one.
    buffer_unverified_streaming: bool = False
    max_attempts: int = 3


class MalformedJsonPolicy(str, Enum):
    """How to handle malformed JSON in tool-call arguments."""

    REJECT = "reject"
    SAFE = "safe"
    AGGRESSIVE = "aggressive"


class FieldAliasPolicy(str, Enum):
    """How to handle non-canonical field names in tool arguments."""

    DISABLED = "disabled"
    SCHEMA_ONLY = "schema_only"
    COMPATIBILITY_PACK = "compatibility_pack"


class UnknownToolPolicy(str, Enum):
    """How to handle tool calls for undeclared tools."""

    REJECT = "reject"
    SAFE_NORMALIZATION = "safe_normalization"


class RepairTier(str, Enum):
    """Semantic risk tiers for repair rules.

    Hierarchy (ascending risk):
    T1  SYNTAX_ONLY           — trailing comma, control chars, closure fix
    T2  SAFE_SHAPE            — schema-declared aliases, stringified arrays
    T3  COMPATIBILITY_REPAIR  — verified compatibility-tuple-only transforms
    T4  COERCIVE              — scalar coercion, external aliases
    T5  REGENERATION          — constrained second model pass

    Default enablement:
    - T1: enabled when malformed_json >= safe
    - T2: enabled when malformed_json >= safe or field_aliases != disabled
    - T3: enabled ONLY for an exact verified compatibility tuple
    - T4: disabled unless route explicitly opts in (malformed_json=aggressive)
    - T5: bounded and explicit (max_regenerations > 0)
    """

    SYNTAX_ONLY = "syntax_only"                    # T1
    SAFE_SHAPE = "safe_shape"                      # T2
    COMPATIBILITY_REPAIR = "compatibility_repair"  # T3
    COERCIVE = "coercive"                          # T4
    REGENERATION = "regeneration"                  # T5


@dataclass(frozen=True)
class RepairPolicy:
    """Normalized, executable repair policy derived from RepairConfig.

    Passed through the invocation pipeline to candidate extraction,
    transaction service, and regeneration.
    """

    enabled_tiers: frozenset[RepairTier] = field(
        default_factory=lambda: frozenset({RepairTier.SYNTAX_ONLY, RepairTier.SAFE_SHAPE})
    )
    max_input_bytes: int = 65536
    max_added_latency_ms: int = 15000
    max_regenerations: int = 0
    malformed_json_policy: MalformedJsonPolicy = MalformedJsonPolicy.SAFE
    field_alias_policy: FieldAliasPolicy = FieldAliasPolicy.SCHEMA_ONLY
    unknown_tool_policy: UnknownToolPolicy = UnknownToolPolicy.SAFE_NORMALIZATION
    batch_policy: str = "atomic"  # "atomic" | "best_effort"

    @classmethod
    def from_config(cls, config: RepairConfig) -> RepairPolicy:
        """Create a RepairPolicy from a RepairConfig.

        Builds the enabled_tiers independently from each config dimension
        rather than keying everything off malformed_json. This ensures that
        disabling syntax recovery doesn't disable safe-shape rules and
        vice versa.
        """
        tiers: set[RepairTier] = set()

        malformed = MalformedJsonPolicy(config.malformed_json)
        if malformed == MalformedJsonPolicy.REJECT:
            pass  # No syntax recovery tiers
        else:
            tiers.add(RepairTier.SYNTAX_ONLY)
            if malformed in (MalformedJsonPolicy.SAFE, MalformedJsonPolicy.AGGRESSIVE):
                tiers.add(RepairTier.SAFE_SHAPE)
            if malformed == MalformedJsonPolicy.AGGRESSIVE:
                tiers.add(RepairTier.COERCIVE)

        # Field-aliases enable safe-shape tier for rename rules
        field_alias = FieldAliasPolicy(config.field_aliases)
        if field_alias != FieldAliasPolicy.DISABLED:
            tiers.add(RepairTier.SAFE_SHAPE)

        # Regeneration tier is orthogonal — controlled by max_regenerations
        if config.max_regenerations > 0:
            tiers.add(RepairTier.REGENERATION)

        # COMPATIBILITY_REPAIR tier for compatibility pack transformations
        if field_alias == FieldAliasPolicy.COMPATIBILITY_PACK:
            tiers.add(RepairTier.COMPATIBILITY_REPAIR)

        return cls(
            enabled_tiers=frozenset(tiers),
            max_input_bytes=config.max_repair_input_bytes,
            max_added_latency_ms=config.max_added_latency_ms,
            max_regenerations=config.max_regenerations,
            malformed_json_policy=malformed,
            field_alias_policy=field_alias,
            unknown_tool_policy=UnknownToolPolicy(config.unknown_tool),
            batch_policy=config.batch_policy,
        )


@dataclass
class ModelRoute:
    """A single model route that the gateway can serve."""

    id: str
    client_model_aliases: list[str]
    upstream_model: str
    upstream: UpstreamConfig
    profile: str = "auto"
    tool_mode: ToolMode = ToolMode.AUTO
    translation_mode: TranslationMode = TranslationMode.CANONICAL
    repair: RepairConfig = field(default_factory=RepairConfig)
    tool_surface: ToolSurfaceConfig = field(default_factory=ToolSurfaceConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)
    controller: ControllerConfig | None = None
    compatibility: CompatibilityConfig = field(default_factory=CompatibilityConfig)


@dataclass
class EvidenceConfig:
    """Opt-in configuration for the compatibility evidence store.

    The evidence store is NEVER enabled by default — it must be explicitly
    opted into via server config. This preserves the "opt-in only, never
    silently enable persistent state" principle: a Gateway constructed
    without an store (the default) behaves byte-for-byte identically to
    before.
    """

    enabled: bool = False
    db_path: str | None = None  # None -> EvidenceStore default (XDG state dir)


@dataclass
class InteropServerConfig:
    """Top-level server configuration (replaces old InteropConfig)."""

    host: str = "127.0.0.1"
    port: int = 8090
    log_level: str = "info"
    probe_on_startup: bool = True
    default_route_id: str = ""
    routes: dict[str, ModelRoute] = field(default_factory=dict)
    ingress_auth: dict[str, str] = field(default_factory=dict)
    backend_timeout: float = 120.0
    evidence: EvidenceConfig | None = None  # Opt-in; None = no evidence store
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)

    # Operational transport settings (P0.6)
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    write_timeout: float = 30.0
    pool_timeout: float = 5.0
    max_connections: int = 100
    max_keepalive_connections: int = 20
    max_retries: int = 2
    max_request_bytes: int = 32 * 1024 * 1024  # 32 MiB
    max_response_bytes: int = 256 * 1024 * 1024  # 256 MiB
    max_stream_frame_bytes: int = 1 * 1024 * 1024  # 1 MiB
    max_tool_argument_bytes: int = 16 * 1024 * 1024  # 16 MiB
    max_simultaneous_tool_calls: int = 64
    max_malformed_stream_frames: int = 2
    tls_verify: bool = True
    retryable_statuses: tuple[int, ...] = (429, 500, 502, 503, 504)

    def get_route_for_model(self, model_name: str) -> ModelRoute | None:
        """Resolve a model name to a route by alias or ID.

        This is the authoritative single resolver.

        Resolution order:
        1. Exact alias or route ID match
        2. If model_name is empty/None → configured default (if any)
        3. If no match and model_name was explicitly provided → None (unknown model)

        Ordinary aliases are NEVER treated as regex patterns.
        """
        if not model_name:
            # No model supplied → use configured default
            if self.default_route_id and self.default_route_id in self.routes:
                return self.routes[self.default_route_id]
            return None

        # 1. Exact alias or route ID match
        for route in self.routes.values():
            if model_name in route.client_model_aliases or model_name == route.id:
                return route

        # 2. Explicit model requested but not found → None (do not silently fallback)
        return None

    def all_model_aliases(self) -> list[str]:
        """Return all client-facing model aliases for /v1/models."""
        aliases: list[str] = []
        for route in self.routes.values():
            aliases.extend(route.client_model_aliases)
        return sorted(set(aliases))

    def get_models_response(self) -> list[dict[str, Any]]:
        """Build the /v1/models response."""
        data: list[dict[str, Any]] = []
        for alias in self.all_model_aliases():
            data.append({
                "id": alias,
                "object": "model",
                "created": 0,
                "owned_by": "interop",
            })
        return data

    @property
    def is_multi_route(self) -> bool:
        return len(self.routes) > 1


def validate_config(config: InteropServerConfig) -> list[str]:
    """Validate server configuration, returning a list of issue strings.

    An empty list means the configuration is valid.
    """
    issues: list[str] = []

    if not config.routes:
        issues.append("At least one route must be configured")
        return issues

    if config.default_route_id and config.default_route_id not in config.routes:
        issues.append(f"default_route_id '{config.default_route_id}' not found in routes")

    # Check alias collisions
    all_aliases: dict[str, str] = {}
    for route_id, route in config.routes.items():
        # Route ID should equal the dict key
        if route_id != route.id:
            issues.append(f"Route key '{route_id}' does not match route.id '{route.id}'")

        if not route.client_model_aliases:
            issues.append(f"Route '{route_id}' has no client_model_aliases")
        for alias in route.client_model_aliases:
            if not alias or not alias.strip():
                issues.append(f"Route '{route_id}' has empty alias")
                continue
            if alias in all_aliases:
                issues.append(f"Alias '{alias}' collides between route '{all_aliases[alias]}' and '{route_id}'")
            all_aliases[alias] = route_id

        # upstream_model must be present
        if not route.upstream_model:
            issues.append(f"Route '{route_id}': upstream_model is required")

    # Valid enum value sets for validation
    _valid_malformed = {e.value for e in MalformedJsonPolicy}
    _valid_unknown = {e.value for e in UnknownToolPolicy}
    _valid_alias = {e.value for e in FieldAliasPolicy}
    _valid_tool_modes = {e.value for e in ToolMode}
    _valid_translation = {e.value for e in TranslationMode}

    # Supported kind → protocol combinations
    _supported_kind_protocol: dict[UpstreamKind, set[UpstreamProtocol]] = {
        UpstreamKind.OLLAMA: {UpstreamProtocol.OLLAMA_CHAT},
        UpstreamKind.VLLM: {UpstreamProtocol.OPENAI_CHAT},
        UpstreamKind.LLAMACPP: {UpstreamProtocol.OPENAI_CHAT},
        UpstreamKind.OPENAI: {UpstreamProtocol.OPENAI_CHAT, UpstreamProtocol.OPENAI_RESPONSES},
        UpstreamKind.ANTHROPIC: {UpstreamProtocol.ANTHROPIC_MESSAGES},
        UpstreamKind.OPENAI_COMPATIBLE: {UpstreamProtocol.OPENAI_CHAT},
    }

    # Validate upstream config and repair config
    HOP_BY_HOP = {"host", "content-length", "transfer-encoding", "connection"}
    PROTECTED_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "authorization"}
    for route_id, route in config.routes.items():
        if not route.upstream.base_url.startswith(("http://", "https://")):
            issues.append(f"Route '{route_id}': upstream URL must start with http:// or https://")
        if route.upstream.timeout_seconds <= 0:
            issues.append(f"Route '{route_id}': timeout_seconds must be positive")

        # Validate upstream kind/protocol combination
        supported_protocols = _supported_kind_protocol.get(route.upstream.kind, set())
        if route.upstream.wire_protocol not in supported_protocols:
            issues.append(
                f"Route '{route_id}': wire_protocol '{route.upstream.wire_protocol.value}' "
                f"is not supported for kind '{route.upstream.kind.value}'. "
                f"Supported: {[p.value for p in supported_protocols]}"
            )

        for header in route.upstream.static_headers:
            if header.lower() in HOP_BY_HOP:
                issues.append(f"Route '{route_id}': static header '{header}' is hop-by-hop")
            if header.lower() in PROTECTED_HEADERS:
                issues.append(f"Route '{route_id}': static header '{header}' is protected and cannot be set")

        # Validate upstream auth mode (fail-closed): an invalid/unrecognized
        # mode must fail at load time rather than being silently downgraded to
        # NONE inside _build_upstream_auth_config.
        from agent_interop.auth import UpstreamAuthMode
        auth_dict = route.upstream.auth
        if auth_dict:
            auth_mode_str = auth_dict.get("mode", "none")
            try:
                UpstreamAuthMode(auth_mode_str)
            except ValueError:
                issues.append(
                    f"Route '{route_id}': upstream auth mode '{auth_mode_str}' is not valid "
                    f"(valid: {[m.value for m in UpstreamAuthMode]})"
                )

        # Validate repair config values
        r = route.repair
        if r.max_regenerations < 0:
            issues.append(f"Route '{route_id}': max_regenerations must be nonnegative (got {r.max_regenerations})")
        if r.max_repair_input_bytes <= 0:
            issues.append(f"Route '{route_id}': max_repair_input_bytes must be positive (got {r.max_repair_input_bytes})")
        if r.max_added_latency_ms <= 0:
            issues.append(f"Route '{route_id}': max_added_latency_ms must be positive (got {r.max_added_latency_ms})")
        if r.malformed_json not in _valid_malformed:
            issues.append(f"Route '{route_id}': malformed_json '{r.malformed_json}' is not valid (options: {_valid_malformed})")
        if r.unknown_tool not in _valid_unknown:
            issues.append(f"Route '{route_id}': unknown_tool '{r.unknown_tool}' is not valid (options: {_valid_unknown})")
        if r.field_aliases not in _valid_alias:
            issues.append(f"Route '{route_id}': field_aliases '{r.field_aliases}' is not valid (options: {_valid_alias})")
        if r.batch_policy not in ("atomic", "best_effort"):
            issues.append(f"Route '{route_id}': batch_policy '{r.batch_policy}' is not valid (options: atomic, best_effort)")

        # Validate enum-based route config
        if route.tool_mode.value not in _valid_tool_modes:
            issues.append(f"Route '{route_id}': tool_mode '{route.tool_mode}' is not valid")
        if route.translation_mode.value not in _valid_translation:
            issues.append(f"Route '{route_id}': translation_mode '{route.translation_mode}' is not valid")
        # RAW_PASSTHROUGH and REPAIR_AWARE_SAME_PROTOCOL are
        # currently not implemented.  Reject them explicitly rather
        # than silently substituting canonical behavior.
        if route.translation_mode != TranslationMode.CANONICAL:
            issues.append(
                f"Route '{route_id}': translation_mode={route.translation_mode.value!r} "
                "is not implemented; only 'canonical' is supported"
            )

        surface = route.tool_surface
        if surface.max_initial_tools <= 0:
            issues.append(f"Route '{route_id}': tool_surface.max_initial_tools must be positive")
        if surface.max_schema_tokens <= 0:
            issues.append(f"Route '{route_id}': tool_surface.max_schema_tokens must be positive")
        if surface.max_selection_rounds <= 0:
            issues.append(f"Route '{route_id}': tool_surface.max_selection_rounds must be positive")
        if surface.selector != "lexical" and not surface.allow_embedding_selector:
            issues.append(f"Route '{route_id}': unsupported tool_surface.selector '{surface.selector}'")
        if route.context.output_reserve_tokens <= 0:
            issues.append(f"Route '{route_id}': context.output_reserve_tokens must be positive")
        if route.context.context_limit_tokens < 0:
            issues.append(f"Route '{route_id}': context.context_limit_tokens must be nonnegative")
        if route.compatibility.mode not in {"auto", "direct", "adapted", "controlled"}:
            issues.append(f"Route '{route_id}': compatibility.mode must be auto, direct, adapted, or controlled")
        if route.compatibility.max_attempts <= 0:
            issues.append(f"Route '{route_id}': compatibility.max_attempts must be positive")

        # Cross-field: explicit profile must exist unless "auto"
        if route.profile and route.profile != "auto":
            from agent_interop.model.profiles_v2 import load_profiles
            profile_index = load_profiles()
            if profile_index.get_by_id(route.profile) is None:
                issues.append(
                    f"Route '{route_id}': profile '{route.profile}' not found in profile index"
                )

        # Cross-field: NATIVE mode requires a profile that supports it
        # Only flag if an explicit profile is set AND it demonstrably lacks native support
        if route.tool_mode == ToolMode.NATIVE and route.profile and route.profile != "auto":
            from agent_interop.model.profiles_v2 import load_profiles
            profile_index = load_profiles()
            prof = profile_index.get_by_id(route.profile)
            if prof is not None:
                tb = prof.tool_behavior
                supports_native = (
                    tb.native_schema_support
                    and tb.native_response_support
                    and tb.presentation_mode == "native"
                )
                if not supports_native:
                    issues.append(
                        f"Route '{route_id}': tool_mode=native but profile '{route.profile}' "
                        "does not support native tools"
                    )

        # Cross-field: regeneration requires a codec that supports repair requests
        if r.max_regenerations > 0:
            from agent_interop.upstreams.registry import get_codec
            try:
                codec = get_codec(route.upstream.wire_protocol)
                # Verify the codec has repair request capability
                if not hasattr(codec, 'build_repair_request'):
                    issues.append(
                        f"Route '{route_id}': max_regenerations > 0 but codec "
                        f"'{route.upstream.wire_protocol.value}' does not support repair requests"
                    )
            except ValueError:
                issues.append(
                    f"Route '{route_id}': max_regenerations > 0 but no codec available "
                    f"for protocol '{route.upstream.wire_protocol.value}'"
                )

    # Validate that ingress_auth.mode is a recognized enum value (Bug 4,
    # config layer): a bad mode fails fast at startup rather than only being
    # caught by the middleware at request time.
    from agent_interop.auth import IngressAuthMode

    auth_config = config.ingress_auth
    auth_mode = auth_config.get("mode", "none_loopback")
    try:
        IngressAuthMode(auth_mode)
    except ValueError:
        issues.append(
            f"ingress_auth.mode '{auth_mode}' is not a valid IngressAuthMode "
            f"(valid: {[m.value for m in IngressAuthMode]})"
        )

    # Cross-field: ingress auth mandatory when binding outside loopback
    if config.host not in ("127.0.0.1", "::1", "localhost", ""):
        mode = auth_config.get("mode", "none_loopback")
        if mode == "none_loopback":
            issues.append(
                f"Binding to non-loopback address '{config.host}' requires "
                "ingress authentication (ingress_auth.mode must not be 'none_loopback')"
            )

    # Validate transport settings are positive
    _positive_fields = {
        "connect_timeout": config.connect_timeout,
        "read_timeout": config.read_timeout,
        "write_timeout": config.write_timeout,
        "pool_timeout": config.pool_timeout,
        "max_connections": config.max_connections,
        "max_keepalive_connections": config.max_keepalive_connections,
        "max_request_bytes": config.max_request_bytes,
        "max_response_bytes": config.max_response_bytes,
        "max_stream_frame_bytes": config.max_stream_frame_bytes,
        "max_tool_argument_bytes": config.max_tool_argument_bytes,
        "max_simultaneous_tool_calls": config.max_simultaneous_tool_calls,
        "max_malformed_stream_frames": config.max_malformed_stream_frames,
    }
    for field_name, value in _positive_fields.items():
        if isinstance(value, (int, float)) and value <= 0:
            issues.append(f"Transport setting '{field_name}' must be positive (got {value})")

    for status in config.retryable_statuses:
        if not isinstance(status, int) or not (100 <= status <= 599):
            issues.append(
                f"Transport setting 'retryable_statuses' contains an invalid "
                f"HTTP status code: {status!r}"
            )

    # Cross-field: a keepalive pool larger than the total connection pool
    # is nonsensical (it can never hold more idle connections than the
    # pool allows in total) and usually means one of the two was edited
    # without the other.
    if config.max_keepalive_connections > config.max_connections:
        issues.append(
            f"max_keepalive_connections ({config.max_keepalive_connections}) must not "
            f"exceed max_connections ({config.max_connections})"
        )

    # log_level must be a real logging level name — cli.py's
    # _configure_process_logging() silently falls back to INFO via
    # getattr(logging, log_level.upper(), logging.INFO) for anything
    # unrecognized, so a typo here previously never surfaced as a
    # config problem, only as "why is my debug logging not showing up".
    _valid_log_levels = {"debug", "info", "warning", "warn", "error", "critical"}
    if config.log_level.lower() not in _valid_log_levels:
        issues.append(
            f"log_level {config.log_level!r} is not valid (valid: {sorted(_valid_log_levels)})"
        )

    return issues


SUPPORTED_CONFIG_SCHEMA_VERSIONS = (1, 2)


def load_config_from_dict(data: dict[str, Any]) -> InteropServerConfig:
    """Load server config from a dict (YAML or JSON source)."""
    schema_version = data.get("schema_version")
    if schema_version is not None and schema_version not in SUPPORTED_CONFIG_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported config schema_version {schema_version!r} "
            f"(supported: {SUPPORTED_CONFIG_SCHEMA_VERSIONS})"
        )

    routes: dict[str, ModelRoute] = {}
    for route_id, rd in data.get("routes", {}).items():
        upstream_data = rd.get("upstream", {})
        upstream_kind = UpstreamKind(upstream_data.get("kind", "ollama"))
        upstream = UpstreamConfig(
            kind=upstream_kind,
            base_url=upstream_data.get("base_url", "http://127.0.0.1:11434"),
            wire_protocol=UpstreamProtocol(
                upstream_data.get(
                    "wire_protocol",
                    default_wire_protocol_for_kind(upstream_kind).value,
                )
            ),
            api_key_env=upstream_data.get("api_key_env"),
            static_headers=upstream_data.get("static_headers", {}),
            timeout_seconds=upstream_data.get("timeout_seconds", 120.0),
            auth=upstream_data.get("auth", {}),
        )
        repair_data = rd.get("repair", {})
        repair = RepairConfig(
            max_regenerations=repair_data.get("max_regenerations", RepairConfig().max_regenerations),
            max_added_latency_ms=repair_data.get("max_added_latency_ms", RepairConfig().max_added_latency_ms),
            max_repair_input_bytes=repair_data.get("max_repair_input_bytes", RepairConfig().max_repair_input_bytes),
            malformed_json=repair_data.get("malformed_json", RepairConfig().malformed_json),
            unknown_tool=repair_data.get("unknown_tool", RepairConfig().unknown_tool),
            field_aliases=repair_data.get("field_aliases", RepairConfig().field_aliases),
            batch_policy=repair_data.get("batch_policy", RepairConfig().batch_policy),
        )
        tool_surface_data = rd.get("tool_surface", {})
        tool_surface = ToolSurfaceConfig(
            mode=ToolSurfaceMode(tool_surface_data.get("mode", ToolSurfaceMode.DYNAMIC.value)),
            toolsets=tuple(tool_surface_data.get("toolsets", ())),
            allow_tools=tuple(tool_surface_data.get("allow_tools", ())),
            deny_tools=tuple(tool_surface_data.get("deny_tools", ())),
            max_initial_tools=tool_surface_data.get("max_initial_tools", 8),
            max_schema_tokens=tool_surface_data.get("max_schema_tokens", 8192),
            max_selection_rounds=tool_surface_data.get("max_selection_rounds", 1),
            selector=tool_surface_data.get("selector", "lexical"),
            allow_embedding_selector=tool_surface_data.get("allow_embedding_selector", False),
        )
        context_data = rd.get("context", {})
        context = ContextConfig(
            strategy=context_data.get("strategy", "auto"),
            output_reserve_tokens=context_data.get("output_reserve_tokens", 4096),
            context_limit_tokens=context_data.get("context_limit_tokens", 0),
            allow_tool_reduction=context_data.get("allow_tool_reduction", True),
            allow_history_compaction=context_data.get("allow_history_compaction", True),
            allow_controller_decomposition=context_data.get("allow_controller_decomposition", True),
        )
        qualification_data = dict(rd.get("qualification", {}))
        if schema_version == 2 and "bootstrap" not in qualification_data:
            qualification_data["bootstrap"] = "blocking_for_tool_requests"
        qualification = QualificationConfig(**qualification_data)
        controller_data = rd.get("controller")
        if schema_version == 2 and isinstance(controller_data, dict) and "require_verified" not in controller_data:
            controller_data = {**controller_data, "require_verified": True}
        controller = ControllerConfig(**controller_data) if isinstance(controller_data, dict) else None
        compatibility_data = dict(rd.get("compatibility", {}))
        if schema_version == 2 and "buffer_unverified_streaming" not in compatibility_data:
            compatibility_data["buffer_unverified_streaming"] = True
        compatibility = CompatibilityConfig(**compatibility_data)
        routes[route_id] = ModelRoute(
            id=route_id,
            client_model_aliases=rd.get("aliases", []),
            upstream_model=rd.get("upstream_model", rd.get("model", "")),
            upstream=upstream,
            profile=rd.get("profile", "auto"),
            tool_mode=ToolMode(rd.get("tool_mode", "auto")),
            translation_mode=TranslationMode(rd.get("translation_mode", "canonical")),
            repair=repair,
            tool_surface=tool_surface,
            context=context,
            qualification=qualification,
            controller=controller,
            compatibility=compatibility,
        )

    # Transport settings (P0.6)
    transport = data.get("transport", {})

    # Evidence store (opt-in only). Absent/disabled -> None (no store).
    evidence_data = data.get("evidence")
    evidence = None
    if evidence_data:
        evidence = EvidenceConfig(
            enabled=evidence_data.get("enabled", False),
            db_path=evidence_data.get("db_path"),
        )
    controller_data = dict(data.get("controller", {}))
    if schema_version == 2 and "require_verified" not in controller_data:
        controller_data["require_verified"] = True
    controller = ControllerConfig(**controller_data)
    diagnostics_data = dict(data.get("diagnostics", {}))
    if schema_version == 2 and "persist" not in diagnostics_data:
        diagnostics_data["persist"] = True
    diagnostics = DiagnosticsConfig(**diagnostics_data)

    return InteropServerConfig(
        host=data.get("host", "127.0.0.1"),
        port=data.get("port", 8090),
        log_level=data.get("log_level", "info"),
        probe_on_startup=data.get("probe_on_startup", True),
        default_route_id=data.get("default_route", data.get("default_route_id", "")),
        routes=routes,
        ingress_auth=data.get("ingress_auth", {}),
        backend_timeout=data.get("backend_timeout", 120.0),
        connect_timeout=transport.get("connect_timeout", 5.0),
        read_timeout=transport.get("read_timeout", 120.0),
        write_timeout=transport.get("write_timeout", 30.0),
        pool_timeout=transport.get("pool_timeout", 5.0),
        max_connections=transport.get("max_connections", 100),
        max_keepalive_connections=transport.get("max_keepalive_connections", 20),
        max_retries=transport.get("max_retries", 2),
        max_request_bytes=transport.get("max_request_bytes", 32 * 1024 * 1024),
        max_response_bytes=transport.get("max_response_bytes", 256 * 1024 * 1024),
        max_stream_frame_bytes=transport.get("max_stream_frame_bytes", 1 * 1024 * 1024),
        max_tool_argument_bytes=transport.get("max_tool_argument_bytes", 16 * 1024 * 1024),
        max_simultaneous_tool_calls=transport.get("max_simultaneous_tool_calls", 64),
        max_malformed_stream_frames=transport.get("max_malformed_stream_frames", 2),
        tls_verify=transport.get("tls_verify", True),
        retryable_statuses=tuple(
            transport.get("retryable_statuses", (429, 500, 502, 503, 504))
        ),
        evidence=evidence,
        controller=controller,
        diagnostics=diagnostics,
    )


def example_config() -> InteropServerConfig:
    """Return an example configuration for documentation purposes."""
    return InteropServerConfig(
        routes={
            "qwen-local": ModelRoute(
                id="qwen-local",
                client_model_aliases=["qwen3-coder", "claude-interop-qwen3-coder"],
                upstream_model="qwen3-coder:latest",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:11434",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                ),
                tool_mode=ToolMode.AUTO,
                profile="auto",
            ),
            "deepseek-local": ModelRoute(
                id="deepseek-local",
                client_model_aliases=["deepseek-local", "claude-interop-deepseek"],
                upstream_model="deepseek-r1:70b",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.VLLM,
                    base_url="http://127.0.0.1:8000",
                    wire_protocol=UpstreamProtocol.OPENAI_CHAT,
                ),
                tool_mode=ToolMode.PROMPTED,
                profile="auto",
            ),
        },
    )
