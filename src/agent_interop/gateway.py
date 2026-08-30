"""The core gateway engine — orchestrates protocol translation, model calls,
and response conversion.

The Gateway is the central object that ties together:
1. Client protocol adapters (inbound protocol parsing)
2. Model profile registry (capability-aware resolution)
3. Upstream codecs (protocol-native rendering/decoding)
4. Tool-call parsers (extraction from model output)
5. Transaction service (validation and repair)
6. Response encoding (back to client protocol)

Production path:
    HTTP request → client protocol adapter → canonical request → request context
    → route resolution → history reconciliation → model/backend profile resolution
    → repair policy → invocation plan → upstream codec rendering
    → authenticated transport → upstream codec decoding
    → model-dialect extraction → universal tool transaction
    → canonical response/events → client protocol adapter encoding
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, cast

from agent_interop import __version__
from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalError,
    CanonicalEvent,
    CanonicalModelReference,
    CanonicalRequest,
    CanonicalResponse,
    CanonicalStopReason,
    CanonicalTextBlock,
    CanonicalToolCallBlock,
    CanonicalUsage,
    RawToolCallCandidate,
    RepairStatus,
    ToolChoiceMode,
)
from agent_interop.admission import AdmissionConfig, InferenceAdmissionController
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairPolicy,
    ToolMode,
)
from agent_interop.context import RequestContext
from agent_interop.context_budget.adaptation import (
    AdaptationState,
    run_context_adaptation,
)
from agent_interop.context_budget.meter import TokenMeter
from agent_interop.context_store.executor import InternalToolExecutor
from agent_interop.context_store.store import ContextStore
from agent_interop.enums import RESERVED_INTERNAL_TOOL_PREFIX, ToolAuthority
from agent_interop.errors import InteropErrorCode, classify_http_status
from agent_interop.evidence.recorder import (
    record_evidence_observation,
    selected_evidence_key,
)
from agent_interop.evidence.store import EvidenceStore
from agent_interop.execution import InteropRequestExecution
from agent_interop.extraction import get_default_registry
from agent_interop.history.reconcile import reconcile_history
from agent_interop.history.summary import (
    replace_compacted_history_with_controller_summary,
)
from agent_interop.model.registry import ModelProfileRegistry
from agent_interop.model.registry import get_default_registry as get_default_profile_registry
from agent_interop.projection import rebuild_invocation_atomic
from agent_interop.qualification import QualificationCoordinator, state_meets_controller_level
from agent_interop.repair.invocation import build_invocation_plan
from agent_interop.transaction import ToolBatchPolicy, ToolTransactionContext, process_tool_batch
from agent_interop.transport.http import (
    UpstreamTransport,
)
from agent_interop.types import ServerInfo
from agent_interop.upstreams.codec import (
    DecodedModelResponse,
)
from agent_interop.upstreams.registry import get_codec

logger = logging.getLogger("agent_interop.gateway")


# Minimum sample base before a piece of evidence is trusted to gate
# compatibility-pack activation. Below this, a record is too thin to act on.
MIN_EVIDENCE_SAMPLE_COUNT = 5


# ─── ResolvedInvocation (P0.1 contract) ────────────────────────────────────


@dataclass(frozen=True)
class PrivateCapabilityPlan:
    """P0.7: Request-scoped set of private tools that may be used."""
    read_result: bool = False
    recall_history: bool = False
    search_history: bool = False
    get_tool_schema: bool = False

    @property
    def has_any(self) -> bool:
        return bool(self.read_result or self.recall_history or self.search_history or self.get_tool_schema)


@dataclass(frozen=True)
class ResolvedInvocation:
    """Request-scoped preparation result (P0.1).

    Created once per request through :meth:`Gateway._prepare_invocation`.
    Carries every resolved component needed by both streaming and
    non-streaming request paths.
    """

    request_context: Any  # RequestContext
    original_request: CanonicalRequest
    reconciled_request: CanonicalRequest
    route: ModelRoute
    backend_metadata: Any  # BackendMetadata
    model_profile: Any  # ResolvedModelProfile
    repair_policy: RepairPolicy
    invocation_plan: Any  # InvocationPlan
    codec: Any  # ModelCodec
    compatibility_key: Any  # CompatibilityKey
    evidence_record: Any | None  # EvidenceRecord when one exists
    repair_budget: Any  # RepairBudget
    execution_record: Any  # InteropRequestExecution
    # P0 compatibility-planning facts are intentionally retained alongside
    # the legacy backend metadata/InvocationPlan fields during migration.
    runtime_capabilities: Any | None = None
    behavioral_capabilities: Any | None = None
    request_requirements: Any | None = None
    compatibility_plan: Any | None = None
    context_plan: Any | None = None
    tool_surface_plan: Any | None = None
    compatibility_attempt: Any | None = None
    authoritative_request: Any | None = None
    model_request: Any | None = None
    model_view: Any | None = None
    # P0.4: Private capability plan — which __interop_* tools are usable.
    private_capabilities: Any | None = None  # PrivateCapabilityPlan
    # P0.4: The actual model-visible tool surface (client + selected private).
    model_visible_tools: tuple[Any, ...] = ()
    # P0.3/P0.6: ContextStore refs this request created/pinned. The gateway
    # pins them for the WHOLE request lifecycle (private continuations may
    # still need the data after the first worker generation) and unpins in
    # the caller's finally block.
    pinned_refs: tuple[str, ...] = ()


def _no_private_capabilities() -> Any:
    """Review #21: all-off PrivateCapabilityPlan for tool-free sub-requests
    (controller primary worker turns).  Built lazily to avoid an import
    cycle at module load."""
    from agent_interop.projection.types import PrivateCapabilityPlan
    return PrivateCapabilityPlan(read_result=False, recall_history=False,
                                 search_history=False, get_tool_schema=False)


def _canonicalize_json_ish(value: Any) -> str:
    """Canonicalize a JSON-ish value to a stable string representation.

    JSON *strings* are parsed and re-serialized with sorted keys so that
    semantically-identical payloads differing only in key order or whitespace
    collapse to the same output. This is what makes argument-based
    de-duplication and loop detection robust to a model re-encoding the same
    logical arguments differently.

    A string that fails to parse falls back to ``str(value)`` rather than
    raising, since this helper is used in non-fatal bookkeeping paths.
    """
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return str(value)
    return json.dumps(parsed, sort_keys=True, default=str)


class Gateway:
    """Core agent compatibility gateway.

    Accepts route-based InteropServerConfig. Each request resolves a route
    by model name, and the route determines upstream model, wire protocol,
    tool mode, and repair settings.

    Dependency injection via constructor arguments enables tests to prove
    which codec, transport, profile, and evidence records were selected.
    """

    @staticmethod
    def prepare_model_request_for_attempt(
        invocation: Any,
        plan: Any,
    ) -> Any:
        """P0-1: Produce the canonical request the codec should render.

        The codec is presentation-agnostic: it renders whatever tools are on
        the request as a native tool array. The compatibility mode is applied
        HERE, before rendering, so a PROMPTED/TEXTUAL/DISABLED attempt can
        never leak private Interop tools (or client tools) into the native
        array — a model being prompted precisely because native tools are
        unreliable must not receive them back as a native surface.
        """
        from agent_interop.config import ToolMode

        model_request = getattr(invocation, "model_request", None)
        if model_request is None:
            return model_request

        mode = getattr(plan, "effective_tool_mode", ToolMode.NATIVE) if plan is not None else ToolMode.NATIVE

        if mode == ToolMode.NATIVE:
            return model_request

        from dataclasses import replace
        return replace(model_request, tools=[], tool_choice=None)

    @staticmethod
    def _classify_tool_authority(name: str) -> ToolAuthority:
        """P0.4: Classify a tool call by execution authority."""
        if name.startswith(RESERVED_INTERNAL_TOOL_PREFIX):
            return ToolAuthority.INTEROP_INTERNAL
        return ToolAuthority.CLIENT

    @staticmethod
    def _check_reserved_namespace_collision(tools: list[Any]) -> None:
        """Reject client-declared tools that collide with Interop's reserved
        private namespace.

        The ``__interop_*`` prefix is Interop's execution boundary: the model
        only ever sees internal tools that the private capability plan admits,
        and their execution is resolved through the ContextStore — never
        through client authority.  A client tool named ``__interop_evil``
        would otherwise be projected onto the model surface as if it were
        Interop infrastructure (and classified INTEROP_INTERNAL if the model
        ever called it), bypassing the public/private firewall.
        """
        from agent_interop.enums import RESERVED_INTERNAL_TOOL_PREFIX as _prefix
        for tool in tools or []:
            name = getattr(tool, "name", "")
            if isinstance(name, str) and name.startswith(_prefix):
                raise ValueError(
                    f"Client tool '{name}' uses the reserved '{_prefix}*' namespace "
                    "(Interop private tools); rename the tool."
                )

    def _partition_accepted_by_authority(
        self, decision: Any, session_id: str = "",
    ) -> tuple[list[Any], list[Any]]:
        """Split accepted tool-call blocks into (client, internal) by authority.

        Unlike ``private_loop.partition_blocks_by_authority`` (which is
        isinstance-strict as a firewall), this adapter partitions whatever the
        transaction already accepted — authority here is a property of the
        tool NAME, so real and validated-duck-typed blocks partition the same.
        """
        from agent_interop.enums import ToolAuthority
        from agent_interop.private_loop import classify_tool_authority
        client: list[Any] = []
        internal: list[Any] = []
        for block in getattr(decision, "accepted_blocks", ()) or ():
            if classify_tool_authority(getattr(block, "name", "")) == ToolAuthority.INTEROP_INTERNAL:
                internal.append(block)
            else:
                client.append(block)
        return client, internal

    @staticmethod
    def _build_internal_tool_surface(base_tools, capabilities=None):
        """P0.7: Append private Interop retrieval tools to the model-visible surface."""
        existing = {t.name for t in base_tools}
        from agent_interop.context_store.schema_tools import all_schema_tools
        from agent_interop.context_store.tools import all_internal_tools
        extra = []
        if capabilities is None:
            extra.extend(t for t in all_internal_tools() if t.name not in existing)
            extra.extend(t for t in all_schema_tools() if t.name not in existing)
        else:
            if capabilities.read_result:
                extra.extend(
                    t for t in all_internal_tools()
                    if t.name == "__interop_read_result" and t.name not in existing
                )
            if capabilities.recall_history or capabilities.search_history:
                extra.extend(
                    t for t in all_internal_tools()
                    if t.name in ("__interop_recall_history", "__interop_search_history")
                    and t.name not in existing
                )
            if capabilities.get_tool_schema:
                extra.extend(
                    t for t in all_schema_tools() if t.name not in existing
                )
        return list(base_tools) + extra

    def __init__(
        self,
        config: InteropServerConfig,
        *,
        transport: UpstreamTransport | None = None,
        profile_registry: ModelProfileRegistry | None = None,
        extractor_registry: Any | None = None,
        session_manager: Any | None = None,
        telemetry: Any | None = None,
        evidence_store: EvidenceStore | None = None,
        allow_invalid_config: bool = False,
    ) -> None:
        """Construct a Gateway directly from a config.

        Validates ``config`` with ``validate_config`` and raises
        ``ValueError`` on any issue — this is the lowest construction
        boundary; CLI-level validation (``deploy``/``check``) and
        ``server.app.create_app`` both happen ABOVE this, but a caller that
        constructs ``Gateway(config)`` directly (bypassing both) previously
        reached startup with no validation at all. ``allow_invalid_config``
        exists strictly for tests that intentionally probe invalid-config
        behavior; production call sites must never pass it.
        """
        if not allow_invalid_config:
            from agent_interop.config import validate_config
            issues = validate_config(config)
            if issues:
                raise ValueError(
                    "Invalid InteropServerConfig:\n" + "\n".join(f"  - {i}" for i in issues)
                )
        self.config = config
        self._transport = transport
        self._extractor_registry = extractor_registry or get_default_registry()
        self._profile_registry = profile_registry or get_default_profile_registry()
        self._session_manager = session_manager
        self._telemetry = telemetry
        # Opt-in only: defaults to None (disabled). Do NOT default to
        # get_default_store() here — that would make every unconfigured
        # Gateway silently read/write a real on-disk store (~/.local/state/...),
        # polluting tests and any deployment that doesn't explicitly opt in.
        self._evidence_store = evidence_store
        # Keyed by route_id (not append-only) and cleared at the start of
        # every probe pass, so a route that stops failing doesn't leave its
        # earlier failure entries lingering alongside the new success.
        from agent_interop.readiness import ReadinessProber

        self._readiness = ReadinessProber(gateway=self)
        from agent_interop.history.summary import ControllerHistorySummarizer

        self._history_summarizer = ControllerHistorySummarizer(gateway=self)
        from agent_interop.controller import ControllerStateStore

        self._controller_state = ControllerStateStore()
        # Bootstrap qualification is deliberately scoped to the immutable
        # served-model identity. It informs low-risk presentation decisions;
        # it never enables semantic/coercive repair on its own.
        # New schema-v2 configurations opt into durable diagnostics/state;
        # legacy programmatic setups retain the historical in-memory scope.
        qualification_store: Any | None = None
        if config.diagnostics.persist:
            from agent_interop.paths import qualification_file
            from agent_interop.qualification import QualificationStore

            qualification_store = QualificationStore(qualification_file())
        self._qualification = QualificationCoordinator(
            gateway=self, store=qualification_store,
        )
        from agent_interop.backends.runtime_cache import RuntimeCapabilityCache

        self._runtime_capability_cache = RuntimeCapabilityCache(
            ttl_seconds=config.runtime_inspection.ttl_seconds,
        )
        # P0-45: operational attempt-path hints.  NOT compatibility evidence —
        # a short-lived "this serving tuple already failed native, prompted
        # worked" note that only reorders ladder rungs the planner already
        # permitted.  Keyed by the complete serving tuple; any change to
        # model/template/serving/client/surface invalidates it.
        from agent_interop.planning.hints import AttemptHintCache

        self._attempt_hints = AttemptHintCache()

        # P0-7: stream-safety observations — the buffered-stream gate's
        # automatic path.  One fully-accepted unbuffered streaming turn on
        # a serving tuple unlocks later streams of the SAME tuple; a
        # rejected batch revokes it.  NOT evidence-store material and
        # never persisted: the opt-in evidence store keeps its role as the
        # durable, operator-certified channel.
        from agent_interop.planning.stream_safety import StreamSafetyCache

        self._stream_safety = StreamSafetyCache()

        # P1.8: ONE planner for the gateway's lifetime so its plan cache can
        # actually serve across requests — instantiating per request made
        # the cache a permanent miss.
        from agent_interop.planning import RequestCompatibilityPlanner

        self._compatibility_planner = RequestCompatibilityPlanner()

        # Resource initialization
        rc = config.resources
        self._context_store = ContextStore(
            max_bytes_per_session=rc.max_bytes_per_session,
            max_total_bytes=rc.max_total_bytes,
            max_sessions=rc.max_sessions,
            max_entry_bytes=rc.max_entry_bytes,
            ttl_seconds=rc.ttl_seconds,
        )
        self._internal_executor = InternalToolExecutor(self._context_store)
        self._admission_controller = InferenceAdmissionController(
            AdmissionConfig(
                max_concurrent_generations=rc.max_concurrent_generations,
                max_queued_generations=rc.max_queued_generations,
                queue_timeout_seconds=rc.admission_timeout_seconds,
            ),
        )
        self._token_meter = TokenMeter(
            backend_tokenizer_endpoint=getattr(config, "backend_tokenizer_endpoint", None)
        )

        # P1.13: THE single model-generation seam (render → gate → reserve →
        # admit → transport → reconcile), extracted out of this class so the
        # public, private, and streaming paths cannot drift on presentation,
        # serialization, budgeting, or admission — the drift that produced
        # the historical bypass/double-reservation bug class. One instance
        # per gateway; it holds no request-scoped state.
        from agent_interop.generation_seam import GenerationSeam

        self._generation_seam = GenerationSeam(
            gateway=self,
            admission_controller=self._admission_controller,
            token_meter=self._token_meter,
            apply_invocation_plan=self._apply_invocation_plan_to_request,
            build_upstream_headers=self._build_upstream_headers,
        )

        # P0.4/P0.5: the bounded private-tool continuation loop, extracted
        # alongside the seam — the gateway supplies the machinery (step
        # sending, internal execution authority, extraction, identity) and
        # the loop owns the iteration/firewall discipline.
        from agent_interop.private_loop import PrivateContinuationLoop

        self._private_loop = PrivateContinuationLoop(
            send_step=self._send_one_model_step,
            internal_executor=self._internal_executor,
            extract_candidates=self._extract_tool_candidates,
            enabled_internal_tools=self._enabled_internal_tools,
            request_identity=self._request_identity,
        )

        # The streaming engine: frame loop, tool-fragment accumulation,
        # atomic batch decision, event emission, and the stream tail's
        # budget reconciliation — extracted so the frame machinery is not
        # Gateway's. The generation seam supplies the render/gate prologue
        # so both dispatch paths render byte-identical bodies.
        from agent_interop.stream_engine import StreamEngine

        self._stream_engine = StreamEngine(
            config=config,
            admission_controller=self._admission_controller,
            transport_provider=lambda: self.transport,
            prepare_generation=self._prepare_model_generation,
            context_limit_error=self._context_limit_error,
            build_upstream_headers=self._build_upstream_headers,
            disabled_tool_choice_conflict=self._disabled_tool_choice_conflict,
            extract_tool_candidates=self._extract_tool_candidates,
            dedup_tool_candidates=self._dedup_tool_candidates,
            enabled_internal_tools=self._enabled_internal_tools,
            private_loop=self._private_loop,
            build_transaction_context=self._build_transaction_context,
            record_repairs_to_session=self._record_repairs_to_session,
            record_tool_decisions=self._record_tool_decisions,
            record_evidence_observation=self._record_evidence_observation,
            build_batch_rejection_error=self._build_batch_rejection_error,
            record_stream_safety=self._record_stream_safety_observation,
        )

        # The controlled (controller-mediated) attempt loop: route selection,
        # session turn budgets, work-product selection, decision cycle,
        # tool-choice enforcement, provenance labelling.
        from agent_interop.controller.attempt import ControllerAttemptExecutor

        self._controller_attempt = ControllerAttemptExecutor(
            gateway=self,
            config=config,
            controller_state=self._controller_state,
            select_controller_route=self._select_controller_route,
            inspect_model_runtime=self._inspect_model_runtime,
            backend_metadata_from_runtime=self._backend_metadata_from_runtime,
            resolve_profile=self._resolve_profile,
            no_private_capabilities=_no_private_capabilities,
        )

        from agent_interop.paths import diagnostic_cases_dir
        from agent_interop.replay.store import DiagnosticCaseStore

        self._diagnostic_cases = DiagnosticCaseStore(
            config.diagnostics.retention_count,
            diagnostic_cases_dir() if config.diagnostics.persist else None,
            config.diagnostics.max_case_bytes,
        )

    def _record_stream_safety_observation(
        self,
        invocation: ResolvedInvocation,
        accepted: bool,
    ) -> None:
        """P0-7: one stream's batch outcome updates the tuple's observation.

        A fully-accepted tool batch records stream safety; a rejected batch
        revokes it. Failures here must never break the client stream, so
        the whole operation is wrapped.
        """
        try:
            key = self._stream_safety_key(invocation)
            if not key:
                return
            if accepted:
                self._stream_safety.record(key)
            else:
                self._stream_safety.revoke(key)
        except Exception:  # pragma: no cover - defensive
            logger.debug("stream-safety observation failed", exc_info=True)

    @property
    def transport(self) -> UpstreamTransport:
        """Lazily build a default ``UpstreamTransport`` from config when none
        was injected. Transport settings (P0.6) are mapped from the
        ``InteropServerConfig`` fields."""
        if self._transport is None:
            cfg = self.config
            max_conn = getattr(cfg, "max_connections", 100) or 100
            max_keepalive = getattr(cfg, "max_keepalive_connections", 20) or 20
            max_retries = getattr(cfg, "max_retries", 2) or 2
            read_timeout = getattr(cfg, "read_timeout", cfg.backend_timeout) or cfg.backend_timeout or 120.0
            connect_timeout = getattr(cfg, "connect_timeout", None)
            write_timeout = getattr(cfg, "write_timeout", None)
            pool_timeout = getattr(cfg, "pool_timeout", None)
            max_stream_frame = getattr(cfg, "max_stream_frame_bytes", 1 * 1024 * 1024)
            max_response = getattr(cfg, "max_response_bytes", 256 * 1024 * 1024)
            retryable_statuses = getattr(cfg, "retryable_statuses", (429, 500, 502, 503, 504))
            tls_verify = getattr(cfg, "tls_verify", True)

            self._transport = UpstreamTransport(
                max_connections=max_conn,
                max_keepalive=max_keepalive,
                max_retries=max_retries,
                retryable_statuses=retryable_statuses,
                timeout_seconds=read_timeout,
                connect_timeout=connect_timeout,
                write_timeout=write_timeout,
                pool_timeout=pool_timeout,
                max_sse_data_bytes=max_stream_frame,
                max_ndjson_frame_bytes=max_stream_frame,
                max_total_stream_bytes=max_response,
                max_response_bytes=max_response,
                tls_verify=tls_verify,
            )
        return self._transport

    async def close(self) -> None:
        if self._transport is not None:
            await self._transport.close()
            self._transport = None

    # ─── Startup / Probe ──────────────────────────────────────────────────

    async def startup(self) -> None:
        """Initialize the gateway.

        Validates config, then probes every configured route when
        ``probe_on_startup`` is True.
        """
        from agent_interop.config import validate_config

        issues = validate_config(self.config)
        if issues:
            raise RuntimeError(
                f"Invalid gateway configuration: {'; '.join(issues)}"
            )

        if not self.config.routes:
            logger.warning("interop starting — no routes configured")
            return

        logger.info(
            "interop starting — routes=%d default=%s",
            len(self.config.routes),
            self.config.default_route_id,
        )

        # P1-H: per-route admission caps are applied ONCE at startup — each
        # route may tighten (never widen) the global
        # resources.max_concurrent_generations for its own (backend URL,
        # served model) key. Doing this here, not per request, keeps route
        # resolution free of admission bookkeeping.
        for route in self.config.routes.values():
            route_capacity = getattr(route.upstream, "max_concurrent_generations", 0)
            if route_capacity:
                await self._admission_controller.set_route_capacity(
                    route.upstream.base_url, route.upstream_model, route_capacity,
                )

        if self.config.probe_on_startup:
            await self._probe_routes()

        # P0-1: warm planning metadata concurrently with (or after) health
        # probing so first-request latency carries no inspection cost.
        # Metadata-only — never a behavioral generation.
        await self._warm_runtime_metadata()

    async def _probe_routes(self, *, force: bool = False, ttl: float = 5.0) -> None:
        await self._readiness.probe_routes(force=force, ttl=ttl)

    def readiness(self) -> dict[str, Any]:
        return self._readiness.readiness()

    # ─── Server info ──────────────────────────────────────────────────────

    def server_info(self) -> ServerInfo:
        """Return aggregate service information and route summaries."""
        route_summaries = []
        for route_id, route in self.config.routes.items():
            route_summaries.append({
                "route_id": route_id,
                "upstream_model": route.upstream_model,
                "upstream_kind": route.upstream.kind.value,
                "wire_protocol": route.upstream.wire_protocol.value,
                "tool_mode": route.tool_mode.value,
                "profile": route.profile,
                "default": route_id == self.config.default_route_id,
            })

        return ServerInfo(
            version=__version__,
            model=",".join(
                r.upstream_model for r in self.config.routes.values()
            ),
            routes=route_summaries,
        )

    def get_route_for_model(self, model_name: str) -> ModelRoute | None:
        """Resolve a model name to a route."""
        return self.config.get_route_for_model(model_name)

    def _resolve_route(
        self,
        canonical: CanonicalRequest,
        *,
        route_override: ModelRoute | None = None,
    ) -> ModelRoute:
        """Resolve the route for a request, raising on unknown model.

        ``route_override`` (review #22) lets an internal caller — currently
        the bootstrap probe executor — run an isolated route configuration
        (e.g. a forced tool_mode) WITHOUT mutating ``self.config.routes``.
        The override must already be resolved; it is never registered.
        """
        if route_override is not None:
            return route_override
        requested = canonical.model.requested_name
        route = self.config.get_route_for_model(requested)
        if route is None:
            if requested:
                raise ValueError(
                    f"Unknown model: '{requested}'. "
                    f"Available: {self.config.all_model_aliases()}",
                )
            raise ValueError(
                "No model specified and no default route configured",
            )
        return route

    def _get_session_state(self, context: Any) -> Any:
        """Resolve session state for loop detection — the single touch
        point for a request (increments request_count exactly once, via
        SessionManager.begin_request). Every other lookup of the same
        session within this request must use ``.get()`` instead, or
        request_count ends up counting internal lookups rather than
        requests.

        Returns None (no session tracking) when the client supplied no
        session ID — a synthetic per-request ID would create one
        one-shot session-store entry per stateless request, both
        polluting the bounded store and being useless for loop detection
        (which needs repeated requests in the SAME session to say anything).
        """
        if self._session_manager is None:
            return None
        session_id = getattr(context, 'session_id', None) if context else None
        if not session_id:
            return None
        route_id = getattr(context, 'route_id', '') if context else ''
        return self._session_manager.begin_request(session_id, route_id=route_id)

    def _record_repairs_to_session(
        self,
        batch_decision: Any,
        context: Any,
    ) -> None:
        """Record repair outcomes from a batch decision into session state.

        This feeds the session manager's loop detection with data about
        which tools were repaired, rejected, or succeeded. Without this
        call, the loop detector is starved of input data.

        Does NOT touch the session (no begin_request call) — by the time
        this runs, `_get_session_state` has already been called once for
        this request and created/touched the session if a session_id was
        present. Calling begin_request again here double-counted
        request_count for every batch of repairs recorded in one request.
        """
        if self._session_manager is None:
            return
        session_id = getattr(context, 'session_id', None) if context else None
        if not session_id:
            return
        route_id = getattr(context, 'route_id', '') if context else ''

        for decision in batch_decision.decisions:
            outcome = decision.outcome
            tool_name = outcome.call_name or decision.candidate.name
            # Digest the raw candidate arguments so the loop detector can
            # distinguish repairs of the same tool with DIFFERENT arguments
            # (legitimate) from repeated identical-argument repairs (a loop).
            # Prefer the raw candidate arguments: a rejected call may not have
            # a populated outcome.accepted, but the candidate always carries
            # the original arguments the model produced.
            argument_digest = self._compute_argument_digest(
                decision.candidate.raw_arguments,
            )
            if outcome.was_repaired:
                self._session_manager.record_repair(
                    session_id,
                    route_id,
                    tool_name,
                    len(outcome.initial_issues),
                    "repaired",
                    argument_digest=argument_digest,
                )
            elif not outcome.is_accepted:
                self._session_manager.record_repair(
                    session_id,
                    route_id,
                    tool_name,
                    len(outcome.initial_issues),
                    "rejected",
                    argument_digest=argument_digest,
                )

    def _record_tool_decisions(
        self,
        batch_decision: Any,
        execution: InteropRequestExecution,
    ) -> None:
        """Record per-call repair outcomes onto the shared execution record.

        The in-memory ``execution.tool_decisions`` record is always populated
        so downstream consumers (``finalize_response`` outcome classification,
        summary logging, replay/evidence) can observe what happened on this
        request regardless of whether an evidence store is configured.

        Persisting to the evidence store itself remains opt-in: that write-back
        happens in :meth:`_record_evidence_observation`, which is separately
        gated on ``self._evidence_store``.
        """
        for decision in batch_decision.decisions:
            execution.record_tool_decision(
                tool_name=decision.outcome.call_name or decision.candidate.name or "",
                candidate_id=decision.candidate.id or "",
                outcome=decision.outcome,
                accepted=decision.is_accepted,
            )

    def _record_evidence_observation(
        self,
        invocation: ResolvedInvocation,
        execution: InteropRequestExecution,
    ) -> None:
        """Persist a single-request compatibility observation.

        Read-modify-write: merges this request's outcome into any existing
        record for the exact compatibility tuple (see
        ``agent_interop.evidence.recorder`` for the merge rules: exact
        per-decision counters, pre-v4 seeding, rate re-derivation, and the
        never-touch-certification-state rule). A persistence failure must
        never break the client request, so the whole operation is wrapped
        to log and swallow.
        """
        store = self._evidence_store
        if store is None:
            return
        record_evidence_observation(invocation, execution, store)

    # ─── Request preparation (P0.1 contract) ───────────────────────────

    def _resolve_invocation_plan_and_key(
        self,
        route: ModelRoute,
        request: CanonicalRequest,
        context: Any,
        streaming: bool,
    ) -> Any:
        """Return an awaitable resolution, with a temporary sync unpack shim.

        Runtime inspection is asynchronous.  The iterable behavior exists only
        for the pre-v2 diagnostic callers that unpack the historical five
        values outside an event loop; live traffic and all new callers await
        the full resolution tuple.  It can be removed once the public
        diagnostic API is migrated.
        """
        coroutine = self._resolve_invocation_plan_and_key_async(route, request, context, streaming)

        class _Resolution:
            def __await__(self):
                return coroutine.__await__()

            def __iter__(self):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return iter(asyncio.run(self._resolve()))
                raise RuntimeError("await compatibility resolution inside an active event loop")

            async def _resolve(self):
                resolved = await coroutine
                return resolved[:5]

        return _Resolution()

    async def _resolve_invocation_plan_and_key_async(
        self,
        route: ModelRoute,
        request: CanonicalRequest,
        context: Any,
        streaming: bool,
        *,
        inspect_runtime: bool = True,
        cost_snapshot: Any | None = None,
    ) -> tuple[Any, Any, RepairPolicy, Any, Any, Any, Any, Any, Any]:
        """Resolve backend metadata, model profile, repair policy, invocation
        plan, and the authoritative compatibility key for an already-routed
        request — the exact computation ``_prepare_invocation`` performs,
        factored out so other callers (certify/conformance tooling) can
        obtain a key that will byte-for-byte match what live traffic
        produces for the same route+request+context, instead of hand-rolling
        a sparse key that can never be found by the live gate.

        The ``route`` must already be resolved; ``request`` should be the
        history-reconciled request that will actually be sent upstream.
        ``cost_snapshot`` is the caller's single serialization pass over
        that same request — when supplied, planning and key fingerprinting
        consume it instead of re-serializing.

        Returns:
            (backend_metadata, model_profile, repair_policy, invocation_plan,
            compat_key, runtime_capabilities, behavioral_capabilities,
            compatibility_plan, context_plan)
        """
        from agent_interop.evidence.key import CompatibilityKeyInputs, build_compatibility_key

        # Runtime inspection is intentionally before profile resolution. Codec
        # capabilities describe the transport only; they never prove that this
        # served model can invoke tools.
        codec = get_codec(route.upstream.wire_protocol)
        runtime_capabilities = (
            await self._inspect_model_runtime(route)
            if inspect_runtime else self._static_runtime_capabilities(route)
        )
        backend_metadata = self._backend_metadata_from_runtime(runtime_capabilities)
        model_profile = self._resolve_profile(route, backend_metadata)

        from agent_interop.agents.base import ClientRequirementProfile
        from agent_interop.agents.manifests import load_builtin_descriptor
        behavioral = self._behavioral_capabilities(runtime_capabilities)
        descriptor = load_builtin_descriptor(getattr(context, "client_id", ""))
        client_requirements = descriptor.required_capabilities if descriptor else ClientRequirementProfile()
        planning_route = route if route.controller is not None else replace(route, controller=self.config.controller)
        compatibility_plan = await self._compatibility_planner.plan(
            request=request,
            context=context,
            route=planning_route,
            client_requirements=client_requirements,
            codec_capabilities=codec.capabilities(),
            runtime_capabilities=runtime_capabilities,
            behavioral_capabilities=behavioral,
            # P0.19 (review #13/#14): operator policy flows from server config
            unknown_capacity_policy=self.config.resources.unknown_capacity_policy,
            unknown_capacity_fallback_tokens=self.config.resources.unknown_capacity_fallback_tokens,
            cost_snapshot=cost_snapshot,
        )
        context_plan = compatibility_plan.context_plan
        tool_surface_plan = compatibility_plan.tool_surface_plan
        # An oversized request may gain an adapted path after deterministic
        # old-result compaction in ``_prepare_invocation_async``.  Defer the
        # unavailable-path error for that one case; all ordinary impossible
        # requests keep the historical preflight failure.
        if not compatibility_plan.attempts:
            if compatibility_plan.context_plan.compaction_required:
                # The caller will apply only deterministic safe adaptation,
                # then invoke this resolver again.  No presentation plan or
                # evidence key may be created for a request known not to fit.
                return (
                    backend_metadata,
                    model_profile,
                    RepairPolicy.from_config(route.repair),
                    None,
                    None,
                    runtime_capabilities,
                    behavioral,
                    compatibility_plan,
                    context_plan,
                )
            raise ValueError(
                "REQUEST_PLAN_UNAVAILABLE: no direct, adapted, or configured controller path "
                f"can satisfy {', '.join(compatibility_plan.missing_capabilities) or 'this request'}"
            )

        # A profile that declares streaming_supported=False is a real,
        # executable constraint (unlike declared_tokens/safe_tokens, which
        # remain informational-only pending real token counting) — silently
        # downgrading to non-streaming or silently proceeding would give the
        # client output framed as a stream when the model can't actually
        # produce one incrementally. Reject before contacting the backend,
        # the same way an invalid tool contract is rejected above.
        if (
            streaming
            and model_profile is not None
            and not getattr(model_profile, "streaming_supported", True)
        ):
            raise ValueError(
                f"Model profile '{getattr(model_profile, 'profile_id', '')}' does not "
                "support streaming, but the request asked for stream=true"
            )

        # Resolve the effective tool mode ONCE, from route config × profile × codec,
        # BEFORE anything downstream computes plan fields. This is the single source
        # of truth for tool-mode negotiation — so the plan is built exactly once, already
        # correctly negotiated, and never needs post-hoc mutation.
        from agent_interop.config import ToolMode
        codec_caps = codec.capabilities()
        effective_tool_mode = compatibility_plan.attempts[0].tool_mode

        # Validate tool contract (pre-upstream). This can raise ValueError for an
        # invalid contract, mirroring exactly what _prepare_invocation does —
        # callers that cannot compute a key for this request (e.g. a conformance
        # test whose tools violate the backend contract) must catch locally.
        from agent_interop.request_validation import validate_tool_contract
        backend_constraints = codec.backend_constraints()
        if effective_tool_mode != ToolMode.NATIVE:
            # max_tools models a native tool-array limit. PROMPTED/TEXTUAL/DISABLED
            # never send a native tools array (tools are embedded as text or absent),
            # so that limit does not apply — enforcing it anyway produces false
            # rejections of valid requests that would never hit the native array cap.
            from dataclasses import replace as _replace
            backend_constraints = _replace(backend_constraints, max_tools=0)
        is_valid, validation_issues = validate_tool_contract(
            # A native route's hard backend array cap remains a preflight
            # contract on the client declaration. Surface reduction may later
            # shrink the rendered array, but must not disguise an explicitly
            # native request that the backend could not accept as declared.
            tools=(list(request.tools) if effective_tool_mode == ToolMode.NATIVE
                   else list(tool_surface_plan.visible_tools)),
            tool_choice=request.tool_choice,
            tool_mode=route.tool_mode,
            backend_constraints=backend_constraints,
        )
        if not is_valid:
            issue_messages = "; ".join(i.message for i in validation_issues)
            raise ValueError(f"Invalid tool contract: {issue_messages}")

        # P0.4 (review: private-tool authority): the reserved __interop_*
        # namespace is Interop's execution boundary. Enforce the collision
        # check against the CLIENT DECLARATION (not the reduced surface) so a
        # withheld-by-surface-reduction colliding tool is still rejected.
        self._check_reserved_namespace_collision(list(request.tools))

        # Construct repair policy with confidence gating.
        repair_policy = RepairPolicy.from_config(route.repair)
        profile_confidence = getattr(model_profile, 'source_confidence', 0.5) if model_profile else 0.5
        repair_policy = self._apply_confidence_gate(repair_policy, profile_confidence)

        # Build invocation plan exactly once. The mode is already fully resolved
        # (route × profile × codec), so the plan is correct as built — no post-hoc
        # codec validation or mutation needed.
        plan = build_invocation_plan(
            tools=None,
            tool_choice=request.tool_choice,
            route_mode=effective_tool_mode,
            model_profile=model_profile,
            repair_policy=repair_policy,
            codec_capabilities=codec_caps,
            upstream_tools=tool_surface_plan.visible_tools,
            validation_tools=tool_surface_plan.validation_tools,
        )

        # Compute compatibility key. P1-F: the fingerprint comes from the
        # request's single serialization snapshot (identical canonical form)
        # — local fallback preserves the standalone diagnostic path.
        if cost_snapshot is not None and cost_snapshot.tool_schema_fingerprint:
            tool_schema_fingerprint = cost_snapshot.tool_schema_fingerprint
        else:
            tool_schema_fingerprint = self._compute_tool_schema_fingerprint(request.tools)
        compat_key = build_compatibility_key(CompatibilityKeyInputs(
            # Pass the context OBJECT, not context.client_id — the builder reads
            # .client_id/.client_version/.client_protocol off it. Passing the
            # string client_id crashes with AttributeError whenever it is non-empty.
            request_context=context,
            route=route,
            request=request,
            backend_metadata=backend_metadata,
            model_profile=model_profile,
            invocation_plan=plan,
            tool_schema_fingerprint=tool_schema_fingerprint,
            streaming=streaming,
            runtime_capabilities=runtime_capabilities,
            compatibility_plan=compatibility_plan,
            context_plan=context_plan,
            tool_surface_plan=tool_surface_plan,
            selected_attempt=compatibility_plan.attempts[0],
        ))

        return (
            backend_metadata, model_profile, repair_policy, plan, compat_key,
            runtime_capabilities, behavioral, compatibility_plan, context_plan,
        )

    def _prepare_invocation(
        self,
        request: CanonicalRequest,
        context: Any,
        streaming: bool,
        execution: InteropRequestExecution,
    ) -> Any:
        """Return an awaitable preparation with legacy synchronous access.

        Runtime inspection made preparation asynchronous.  A few diagnostic
        and evidence callers intentionally use this internal helper outside
        an event loop, so preserve that narrow compatibility boundary while
        live request paths await the real implementation below.
        """
        # This compatibility entry point is intentionally offline.  It is used
        # by diagnostics and older integrations that synchronously inspect the
        # resolved plan; live traffic calls the async implementation below and
        # performs runtime inspection before planning.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Preserve the old eager error surface for ordinary synchronous
            # callers (notably contract validation).
            return asyncio.run(self._prepare_invocation_async(
                request, context, streaming, execution, inspect_runtime=False,
            ))

        coroutine = self._prepare_invocation_async(
            request, context, streaming, execution, inspect_runtime=False,
        )

        class _PreparedInvocation:
            _value: ResolvedInvocation | None = None

            async def _resolve(self) -> ResolvedInvocation:
                if self._value is None:
                    self._value = await coroutine
                return self._value

            def __await__(self):
                return self._resolve().__await__()

            def __getattr__(self, name: str) -> Any:
                if self._value is not None:
                    return getattr(self._value, name)
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return getattr(asyncio.run(self._resolve()), name)
                # The offline diagnostic path performs no I/O.  Its nested
                # coroutines all complete immediately, which lets legacy
                # callers access a field inside an async test without opening
                # a second event loop (and without moving :memory: evidence
                # stores to another thread).
                try:
                    coroutine.send(None)
                except StopIteration as stop:
                    self._value = stop.value
                else:  # pragma: no cover - protects this compatibility path
                    raise RuntimeError("offline preparation unexpectedly suspended")
                return getattr(self._value, name)

        prepared = _PreparedInvocation()
        # The compatibility boundary is offline and must complete without an
        # event-loop suspension. Resolve it eagerly so preflight validation
        # preserves its historical synchronous error behavior.
        try:
            coroutine.send(None)
        except StopIteration as stop:
            prepared._value = stop.value
        else:  # pragma: no cover - protects the offline-only contract
            raise RuntimeError("offline preparation unexpectedly suspended")
        return prepared

    @staticmethod
    def _replace_compacted_history_with_controller_summary(
        request: CanonicalRequest,
        context_plan: Any,
        summary: str,
    ) -> CanonicalRequest:
        return replace_compacted_history_with_controller_summary(
            request, context_plan, summary,
        )

    async def _summarize_old_history_with_controller(
        self,
        *,
        route: ModelRoute,
        request: CanonicalRequest,
        context: Any,
        context_plan: Any,
        inspect_runtime: bool,
        execution: InteropRequestExecution | None = None,
    ) -> CanonicalRequest | None:
        return await self._history_summarizer.summarize(
            route=route,
            request=request,
            context=context,
            context_plan=context_plan,
            inspect_runtime=inspect_runtime,
            execution=execution,
        )

    async def _prepare_invocation_async(
        self,
        request: CanonicalRequest,
        context: Any,
        streaming: bool,
        execution: InteropRequestExecution,
        *,
        inspect_runtime: bool = True,
        allow_controller_summary: bool = True,
        route_override: ModelRoute | None = None,
    ) -> ResolvedInvocation:
        """Prepare a resolved invocation for one request (P0.1).

        Creates the request-scoped structure containing every resolved
        component needed by both streaming and non-streaming paths.

        Preparation order:
            1. Resolve route from request.model.requested_name
            2. Reject explicit unknown model (no silent fallback)
            3. Use default_route_id only when model omitted
            4. Reconcile conversation history
            5. Reject unsafe history before contacting backend
            6. Resolve backend metadata and model profile
            7. Construct repair policy
            8. Build invocation plan exactly once
        """
        from agent_interop.repair.pipeline import RepairBudget

        # 1-3. Resolve the route (explicit override for internal probes —
        # review #22; never registers anything in self.config)
        route = self._resolve_route(request, route_override=route_override)
        # Attach to the caller-supplied execution record as soon as it is
        # available so it is populated even on the early-exit branches below.
        execution.route = route

        # 3.25 Config-level contradiction check BEFORE any I/O: a DISABLED
        # route + REQUIRED/NAMED choice can never be satisfied, and refusing
        # it here means the request triggers no metadata inspection, no
        # probe, and no generation — the backend is never contacted at all.
        conflict = self._config_tool_choice_conflict(route, request)
        if conflict is not None:
            raise ValueError(f"TOOL_CHOICE_VIOLATION: {conflict.message}")

        # 3.5 Check for session loop (before expensive preparation)
        session_state = self._get_session_state(context)
        if session_state is not None and session_state.flagged:
            execution.finalize_error(CanonicalError(
                code="GENERATION_LOOP_DETECTED",
                message="Session flagged for generation loop — refusing new requests",
            ))
            raise ValueError(
                f"Session '{getattr(context, 'session_id', '?')}' flagged for generation loop"
            )

        # 4. Reconcile conversation history
        history_result = reconcile_history(
            request.messages,
            session_id=getattr(context, "session_id", "") or "",
            request_id=getattr(context, "request_id", "") or request.request_id or "",
        )
        # Attach history diagnostics to the shared execution record as soon as
        # they are computed so they survive the unsafe-history early exit below.
        execution.history_diagnostics.extend(history_result.diagnostics)

        # 5. Reject unsafe history
        if not history_result.is_safe:
            from dataclasses import replace
            reconciled = replace(request, messages=history_result.messages)
            runtime_capabilities = (
                await self._inspect_model_runtime(route)
                if inspect_runtime else self._static_runtime_capabilities(route)
            )
            unsafe_backend_metadata = self._backend_metadata_from_runtime(runtime_capabilities)
            return ResolvedInvocation(
                request_context=context,
                original_request=request,
                reconciled_request=reconciled,
                route=route,
                backend_metadata=unsafe_backend_metadata,
                model_profile=self._resolve_profile(route, unsafe_backend_metadata),
                repair_policy=RepairPolicy.from_config(route.repair),
                invocation_plan=None,
                codec=None,
                compatibility_key=None,
                evidence_record=None,
                repair_budget=None,
                execution_record=execution,
                runtime_capabilities=runtime_capabilities,
            )

        from dataclasses import replace
        # P0-4: the AUTHORITATIVE request is frozen immediately after history
        # reconciliation. It carries the client's semantics + reconciled
        # history ONLY — never virtualized results, paged history, or
        # model-generated summaries. All capacity adaptation below operates
        # on a separate projected request.
        authoritative_request = replace(request, messages=history_result.messages)
        reconciled_request = authoritative_request

        # P1-F (review item 29): the request's ONE full serialization pass.
        # System/history/tool-schema costs and the canonical tool-schema
        # fingerprint come from this snapshot; the planning pipeline and the
        # compatibility key below consume it instead of each re-serializing.
        from agent_interop.context_budget import build_request_cost_snapshot
        cost_snapshot = build_request_cost_snapshot(reconciled_request)

        # Resolve the codec up front — it only depends on the route (already in
        # hand) and is needed for the ResolvedInvocation returned below.
        codec = get_codec(route.upstream.wire_protocol)

        # Resolve backend metadata, model profile, repair policy, invocation
        # plan, and the authoritative compatibility key — the exact computation
        # live traffic performs, factored into _resolve_invocation_plan_and_key
        # so certify/conformance tooling can obtain a key that byte-for-byte
        # matches what the live gate looks up. This includes the tool-contract
        # validation step that can raise ValueError (propagates unchanged).
        backend_metadata, model_profile, repair_policy, plan, compat_key, runtime_capabilities, behavioral, compatibility_plan, context_plan = (
            await self._resolve_invocation_plan_and_key_async(
                route, reconciled_request, context, streaming,
                inspect_runtime=inspect_runtime,
                cost_snapshot=cost_snapshot,
            )
        )
        # P0-4: the projected request is what adaptation is allowed to
        # reshape. It starts equal to the authoritative request and diverges
        # only through explicit, recorded transformations.
        projected_request = authoritative_request
        # Track what this request's adaptation did — drives private read_result
        # capability and ref pinning below.
        from agent_interop.context_budget.compaction import ContextAdaptationResult
        # P1.1: the request lifecycle is the TTL cleanup hook — a store that
        # is never swept by traffic would otherwise only evict on insert.
        self._context_store.evict_expired()
        adaptation = ContextAdaptationResult(projected_request)
        # Refs accumulated by this function's own adaptation passes (result
        # virtualization lands in adaptation.stored_refs; history paging in
        # history_page_refs). Both are seeded into the final projection.
        history_page_refs: tuple[str, ...] = ()
        if context_plan.compaction_required:
            adapt = AdaptationState(
                projected_request=projected_request,
                cost_snapshot=cost_snapshot,
                adaptation=adaptation,
                backend_metadata=backend_metadata,
                model_profile=model_profile,
                repair_policy=repair_policy,
                plan=plan,
                compat_key=compat_key,
                runtime_capabilities=runtime_capabilities,
                behavioral=behavioral,
                compatibility_plan=compatibility_plan,
                context_plan=context_plan,
            )
            await run_context_adaptation(
                self, adapt,
                route=route, context=context, history_result=history_result,
                streaming=streaming, inspect_runtime=inspect_runtime,
                allow_controller_summary=allow_controller_summary,
                execution=execution,
            )
            projected_request = adapt.projected_request
            cost_snapshot = adapt.cost_snapshot
            adaptation = adapt.adaptation
            history_page_refs = adapt.history_page_refs
            backend_metadata = adapt.backend_metadata
            model_profile = adapt.model_profile
            repair_policy = adapt.repair_policy
            plan = adapt.plan
            compat_key = adapt.compat_key
            runtime_capabilities = adapt.runtime_capabilities
            behavioral = adapt.behavioral
            compatibility_plan = adapt.compatibility_plan
            context_plan = adapt.context_plan
        if not compatibility_plan.attempts:
            raise ValueError(
                "REQUEST_PLAN_UNAVAILABLE: no direct, adapted, or configured controller path "
                f"can satisfy {', '.join(compatibility_plan.missing_capabilities) or 'this request'}"
            )
        execution.invocation_plan = plan
        execution.compatibility_key = compat_key

        # P0-4: the authoritative request stays frozen as established above
        # (client semantics + reconciled history ONLY). The projected request
        # carries every recorded transformation from this point on — result
        # virtualization, history paging, and the controller summary are real
        # work paid for below and must never be discarded by re-deriving the
        # projection from authoritative state here. The single authoritative
        # initialization lives at the top of this function.

        # Look up verified evidence for this exact tuple (opt-in only). A record
        # only qualifies when it is present, manually verified, not revoked, not
        # stale, and backed by a sufficient sample base. Absent that, packs must
        # NOT be activated on a merely well-formed key.
        # P1-G: ONE fetch — the historical form read the row, then is_stale()
        # re-read the identical row before evaluating it.
        evidence_record = None
        if self._evidence_store is not None:
            from agent_interop.evidence.store import result_is_stale

            candidate = self._evidence_store.get_result(compat_key)
            if (
                candidate is not None
                and candidate.manually_verified
                and not result_is_stale(candidate)
                and candidate.sample_count >= MIN_EVIDENCE_SAMPLE_COUNT
            ):
                evidence_record = candidate

        # Create request-scoped repair budget
        repair_budget = RepairBudget()
        execution.repair_budget = repair_budget

        # P0-5: ModelProjector.project() is the SOLE projection authority.
        # The gateway no longer hand-builds the model-facing request; the
        # projector owns result virtualization carried in `adaptation`,
        # history paging, system projection, tool-surface selection, private
        # capabilities, the withheld-tool index, and the ModelView. The
        # gateway only orchestrates ordering and owns ref pinning.
        from agent_interop.projection import ModelProjector
        projection = ModelProjector.project(
            authoritative_request=authoritative_request,
            route=route,
            runtime_capabilities=runtime_capabilities,
            compatibility_plan=compatibility_plan,
            policy=None,
            context_store=self._context_store,
            session_context=context,
            perform_history_paging=False,
            adaptation=adaptation if adaptation.changed else None,
            invocation_plan=plan,
            # P0-4: the projection pipeline continues from the request this
            # function's adaptation actually produced (virtualized results,
            # paged history, controller summary). Re-deriving from the
            # authoritative request would silently discard that paid work.
            projected_request=projected_request,
            seed_refs=tuple(dict.fromkeys((*adaptation.stored_refs, *history_page_refs))),
        )
        model_request = projection.request
        model_view = projection.model_view
        private_caps = projection.private_capabilities
        # Every ref the projected request exposes — gateway-adapted, result
        # virtualization, and history paging — participates in pinning.
        virtualized_refs = projection.referenced_refs

        resolved_invocation = ResolvedInvocation(
            request_context=context,
            original_request=request,
            route=route,
            backend_metadata=backend_metadata,
            model_profile=model_profile,
            repair_policy=repair_policy,
            invocation_plan=plan,
            codec=codec,
            compatibility_key=compat_key,
            evidence_record=evidence_record,
            repair_budget=repair_budget,
            execution_record=execution,
            runtime_capabilities=runtime_capabilities,
            behavioral_capabilities=behavioral,
            request_requirements=compatibility_plan.requirements,
            compatibility_plan=compatibility_plan,
            context_plan=context_plan,
            tool_surface_plan=compatibility_plan.tool_surface_plan,
            # P0-4: authoritative vs projected are distinct fields with
            # distinct meanings — validation/audit reads authoritative,
            # rendering reads model_request.
            authoritative_request=authoritative_request,
            reconciled_request=projected_request,
            model_request=model_request,
            model_view=model_view,
            private_capabilities=private_caps,
            model_visible_tools=tuple(model_request.tools),
            pinned_refs=virtualized_refs,
        )
        execution.configure_token_efficiency(resolved_invocation)
        return resolved_invocation

    # ─── Non-streaming request ────────────────────────────────────────────

    async def handle_request(
        self,
        canonical: CanonicalRequest,
        context: Any,
    ) -> CanonicalResponse:
        """Handle a non-streaming request end-to-end.

        Production path:
            _prepare_invocation → codec render → transport → decode
            → extraction → transaction → canonical assembly
        """
        exec_record = InteropRequestExecution(context=context)
        try:
            # P0-16: the budget exists BEFORE preparation — every model
            # generation on this request (public attempt, private
            # continuation, controller turn) spends against the same
            # ledger, and preparation itself can consult it.  The route's
            # attempt ceiling is stamped in as soon as the route is known.
            from agent_interop.execution_attempts import AttemptBudget

            budget = AttemptBudget()
            exec_record.attempt_budget = budget
            # Prepare the resolved invocation (passes in the shared record so
            # diagnostics/route/plan all land on the object that gets finalized)
            invocation = await self._prepare_invocation_async(
                canonical, context, streaming=False, execution=exec_record,
            )
            if await self._ensure_bootstrap_qualification(invocation):
                # P0-3: qualification contributes only low-risk behavior
                # evidence. Rebuild JUST the evidence-dependent facts — the
                # resolved route/metadata/profile, history reconciliation,
                # tool registry, and context estimate are all unchanged —
                # instead of re-running the entire preparation pipeline.
                invocation = await self._replan_after_qualification(
                    invocation, exec_record,
                )
            budget.max_upstream_attempts = invocation.route.compatibility.max_attempts
            from agent_interop.execution_attempts import CompatibilityAttemptExecutor

            executor = CompatibilityAttemptExecutor(budget)
            # P0.3/P0.6: pin this request's refs for the WHOLE request
            # lifecycle — private continuations and controller turns may still
            # need the stored content after the first worker generation. The
            # registry below unpins on every exit path (success, error,
            # cancellation), so pins can never leak past the request.
            request_id = getattr(context, "request_id", "") or canonical.request_id
            session_id = getattr(context, "session_id", "") or ""
            from agent_interop.context_store import RequestRefRegistry
            ref_registry = RequestRefRegistry(
                self._context_store, session_id, request_id,
            )
            # P0: initial projection refs join through the registry too, so
            # the snapshot the firewall reads and the pins the store hold
            # have exactly one source of truth.
            ref_registry.register_all(getattr(invocation, "pinned_refs", ()) or ())
            invocation = replace(invocation, pinned_refs=())
            exec_record.ref_registry = ref_registry
            try:
                result = await executor.execute(
                    invocation,
                    build_invocation=self._invocation_for_attempt,
                    execute_attempt=lambda attempt_invocation: self._execute_compatibility_attempt(
                        attempt_invocation, exec_record
                    ),
                    replan_withheld_tool=self._invocation_with_withheld_tool,
                    hint_key=lambda inv: self._attempt_hint_key(
                        inv, bool(getattr(inv.reconciled_request.generation, "stream", False)),
                    ),
                    hint_get=self._attempt_hints.get,
                    hint_record=self._attempt_hints.record,
                )
            finally:
                ref_registry.close()
            # Live evidence write-back: only on the success path, only when tools
            # were offered, only when an evidence store was injected. Backend
            # errors carry no tool-calling signal, so they are skipped.
            if (
                self._evidence_store is not None
                and canonical.tools
                and result.error is None
            ):
                self._record_evidence_observation(invocation, exec_record)
            # finalize_response() logs the summary itself (see execution.py) —
            # relying on a caller to do it separately after the fact is what
            # let the streaming path silently skip logging entirely (the ASGI
            # consumer never resumes the generator far enough to reach a
            # trailing call).
            exec_record.finalize_response(result)
            self._capture_diagnostic_case(invocation, exec_record, result)
            return result
        except asyncio.CancelledError:
            # Mid-request cancellation: finalize the record as CANCELLED so it
            # is not left permanently ACTIVE, then re-raise. Do NOT swallow —
            # the caller must see the cancellation.
            exec_record.finalize_cancelled()
            raise
        except Exception as exc:
            error = self._preflight_error(exc)
            if error is not None:
                result = CanonicalResponse(error=error)
                exec_record.finalize_response(result)
                return result
            exc_err = CanonicalError(code="HANDLE_ERROR", message=str(exc)) if not isinstance(exc, CanonicalError) else exc
            exec_record.finalize_error(exc_err)
            raise

    def diagnostic_case(self, case_id: str) -> Any:
        """Retrieve an in-memory sanitized diagnostic case by ID."""
        return self._diagnostic_cases.get(case_id)

    @staticmethod
    def _selected_evidence_key(
        invocation: ResolvedInvocation,
        execution: InteropRequestExecution,
    ) -> Any:
        """Use an enriched key only when a controller actually selected it."""
        return selected_evidence_key(invocation, execution)

    @staticmethod
    def _preflight_error(exc: Exception) -> CanonicalError | None:
        """Translate known planning preflight failures to canonical errors."""
        from agent_interop.context_budget import ContextLimitExceededError
        from agent_interop.context_budget.planner import ContextCapacityUnknownError

        if isinstance(exc, ContextCapacityUnknownError):
            # P0.19 (review #13/#14): unknown capacity is a hard reject for
            # tool-bearing requests — never a silent 8K guess.
            return CanonicalError(
                code=InteropErrorCode.CONTEXT_CAPACITY_UNKNOWN,
                message=str(exc),
                details=exc.details(),
            )
        if isinstance(exc, ContextLimitExceededError):
            return CanonicalError(
                code=InteropErrorCode.CONTEXT_LIMIT_EXCEEDED,
                message="Request does not fit the model's effective context limit after safe adaptation",
                details=exc.details(),
            )
        message = str(exc)
        if message.startswith("REQUEST_PLAN_UNAVAILABLE:"):
            return CanonicalError(
                code=InteropErrorCode.REQUEST_PLAN_UNAVAILABLE,
                message="No direct, adapted, or controller path can satisfy this request",
                details={
                    "path": "unavailable",
                    "responsible": "model_or_controller",
                    "reason": message.removeprefix("REQUEST_PLAN_UNAVAILABLE:").strip(),
                    "next": "configure a qualified controller or select a compatible route",
                },
            )
        return None

    def _capture_diagnostic_case(
        self,
        invocation: ResolvedInvocation,
        execution: InteropRequestExecution,
        response: CanonicalResponse,
    ) -> None:
        """Retain bounded failure/repair diagnostics according to route policy."""
        policy = self.config.diagnostics
        repaired = any(decision.repair_steps for decision in execution.tool_decisions)
        should_capture = policy.capture == "all" or response.error is not None or (
            policy.capture in {"repairs", "failures_and_repairs"} and repaired
        )
        if not should_capture or policy.capture == "off":
            return
        from dataclasses import asdict, is_dataclass

        from agent_interop.build_info import get_build_info
        from agent_interop.replay.capture import capture_case

        def metadata(value: Any) -> dict[str, Any]:
            # is_dataclass accepts classes as well as instances; only an
            # instance can be expanded, so exclude the class form explicitly.
            if value is None or isinstance(value, type) or not is_dataclass(value):
                return {}
            # asdict() rejects dataclass *classes*; the isinstance check above
            # already excluded them. _typeshed's DataclassInstance is
            # typeshed-only (never importable at runtime), hence TYPE_CHECKING.
            return asdict(cast("DataclassInstance", value))

        def plan_metadata() -> dict[str, Any]:
            compatibility = invocation.compatibility_plan
            surface = invocation.tool_surface_plan
            context = invocation.context_plan
            if compatibility is None or surface is None or context is None:
                return {"path": "unavailable", "reason": "preflight_failed"}
            return {
                "path": getattr(compatibility.path, "value", str(compatibility.path)),
                "planner_revision": compatibility.planner_revision,
                "attempts": [attempt.kind.value for attempt in compatibility.attempts],
                "missing_capabilities": list(compatibility.missing_capabilities),
                "context_strategy": context.selected_strategy,
                "context_transformations": list(context.transformations),
                "visible_tools": [tool.name for tool in surface.visible_tools],
                "withheld_tools": list(surface.withheld_tool_names),
            }

        diagnostics = {
            "build": asdict(get_build_info()),
            "runtime": metadata(invocation.runtime_capabilities),
            "requirements": metadata(invocation.request_requirements),
            "plan": plan_metadata(),
            "execution": execution.to_sanitized_dict(),
            "response": {
                "error_code": response.error.code if response.error else "",
                "stop_reason": getattr(response.stop_reason, "value", str(response.stop_reason)),
                "tool_call_count": sum(isinstance(block, CanonicalToolCallBlock) for block in response.content),
            },
        }
        inbound = {
            "request_id": invocation.reconciled_request.request_id,
            "model": invocation.reconciled_request.model.requested_name,
            "message_count": len(invocation.reconciled_request.messages),
            "tool_count": len(invocation.reconciled_request.tools),
        }
        canonical = invocation.reconciled_request if policy.content_mode != "metadata_only" else None
        case = capture_case(
            client_protocol=getattr(invocation.request_context.client_protocol, "value", ""),
            upstream_protocol=invocation.route.upstream.wire_protocol.value,
            inbound_request=inbound,
            canonical_request=canonical,
            upstream_request={"route": invocation.route.id, "model": invocation.route.upstream_model},
            tool_registry=invocation.reconciled_request.tools,
            compatibility_key=self._selected_evidence_key(invocation, execution),
            diagnostics=diagnostics,
        )
        execution.diagnostic_case_id = case.case_id
        self._diagnostic_cases.put(case)

    def _build_transaction_context(
        self,
        invocation: ResolvedInvocation,
        canonical: CanonicalRequest,
    ) -> ToolTransactionContext:
        """Build the ``ToolTransactionContext`` for a tool-call batch.

        Shared by the streaming and non-streaming paths so both use the exact
        same repair policy (confidence-gated), the request-scoped repair budget
        (shared across every batch in the request), the correct request_id, and
        the compatibility key. Duplicating this logic in each path caused the
        streaming path to skip the confidence gate, reset the budget per batch,
        and drop telemetry/compatibility_key.
        """
        request_context = invocation.request_context
        from agent_interop.repair.adapter import make_regenerate_fn
        return ToolTransactionContext(
            request_id=request_context.request_id if request_context else "",
            session_id=getattr(request_context, 'session_id', '') if request_context else '',
            tool_choice=canonical.tool_choice,
            repair_policy=invocation.repair_policy,
            client_id=request_context.client_id if request_context else None,
            # P0-wire-regeneration: the hidden repair generation is real —
            # it reserves budget, passes admission, and is accounted under
            # purpose="tool_repair". None (AUTO default) means the
            # transaction service performs deterministic repair only.
            regenerate_fn=make_regenerate_fn(
                self, invocation, invocation.execution_record,
            ),
            telemetry=self._telemetry,
            budget=invocation.repair_budget,
            compatibility_key=invocation.compatibility_key,
            # evidence_record is only ever non-null when it passed all the
            # verification gates in _prepare_invocation, so its presence is the
            # verified signal. A merely well-formed key is not sufficient.
            compatibility_verified=invocation.evidence_record is not None,
        )

    async def _replan_after_qualification(
        self,
        invocation: ResolvedInvocation,
        execution: InteropRequestExecution,
        *,
        streaming: bool = False,
    ) -> ResolvedInvocation:
        """P0-3: replan ONLY the qualification-dependent facts.

        A full ``_prepare_invocation_async`` re-runs history reconciliation,
        requirement derivation, tool ranking, token estimates, schema hashes,
        and projection — all unchanged by qualification. What DID change is
        the behavioral-capability evidence, so this rebuilds exactly:
        behavioral capabilities → compatibility plan → invocation plan →
        compatibility key → model presentation.

        P1.5 (review #13): ``streaming`` is threaded through so the
        compatibility key fingerprints the same streaming shape the planner
        downstream will see — a stream that requalified and replanned
        without ``streaming=True`` would otherwise compute a stale key.

        P1.6 (review #12): ModelView + private_capabilities + model_request +
        pinned_refs are rebuilt atomically through ``ModelProjector.project``
        AFTER the replan, using the FRESH ``compatibility_plan`` /
        ``tool_surface_plan`` / ``context_plan``. The previous code left
        ModelView (and the codec-native tool surface it describes) pointing
        at the pre-qualification facts — so a freshly-promoted model could
        observe telemetry contradicting the surface it was actually rendered
        with.
        """
        behavioral = self._behavioral_capabilities(invocation.runtime_capabilities)
        (
            backend_metadata, model_profile, repair_policy, plan,
            compat_key, runtime_capabilities, _behavioral,
            compatibility_plan, context_plan,
        ) = await self._resolve_invocation_plan_and_key_async(
            invocation.route,
            invocation.reconciled_request,
            invocation.request_context,
            streaming=streaming,
            inspect_runtime=False,  # metadata is already resolved and cached
        )
        # Atomic ModelView rebuild: project from the SAME authoritative
        # request the original preparation used, with the FRESH surface /
        # plan. ``seed_refs`` carries forward refs the original preparation
        # minted (history pages) so the new virtualized set does not lose
        # work the prior projection already paid for.
        from agent_interop.projection import ModelProjector
        projection = ModelProjector.project(
            authoritative_request=invocation.authoritative_request
                or invocation.reconciled_request,
            route=invocation.route,
            runtime_capabilities=runtime_capabilities,
            compatibility_plan=compatibility_plan,
            policy=None,
            context_store=self._context_store,
            session_context=invocation.request_context,
            # P1.5: history paging was already paid for in preparation —
            # the replan does not re-paging, it only re-presents.
            perform_history_paging=False,
            # No new adaptation at replan time.
            adaptation=None,
            invocation_plan=plan,
            projected_request=invocation.reconciled_request,
            seed_refs=tuple(getattr(invocation, "pinned_refs", ()) or ()),
        )
        updated = replace(
            invocation,
            behavioral_capabilities=behavioral,
            backend_metadata=backend_metadata,
            model_profile=model_profile,
            repair_policy=repair_policy,
            invocation_plan=plan,
            compatibility_key=compat_key,
            runtime_capabilities=runtime_capabilities,
            compatibility_plan=compatibility_plan,
            context_plan=context_plan,
            request_requirements=compatibility_plan.requirements,
            tool_surface_plan=compatibility_plan.tool_surface_plan,
            evidence_record=None,
            # P1.6: the fresh presentation fields.
            model_request=projection.request,
            model_view=projection.model_view,
            private_capabilities=projection.private_capabilities,
            pinned_refs=projection.referenced_refs,
        )
        execution.invocation_plan = plan
        execution.compatibility_key = compat_key
        return updated

    def _attempt_hint_key(self, invocation: ResolvedInvocation, streaming: bool) -> str:
        """P0-45: the complete serving tuple behind an attempt-path hint."""
        from agent_interop.planning.hints import attempt_hint_key

        runtime = invocation.runtime_capabilities
        fingerprint = ""
        requirements = invocation.request_requirements
        if requirements is not None:
            fingerprint = getattr(requirements, "tool_schema_fingerprint", "") or ""
        if not fingerprint:
            fingerprint = self._compute_tool_schema_fingerprint(
                invocation.reconciled_request.tools,
            )
        choice = invocation.reconciled_request.tool_choice
        choice_class = {
            "auto": "auto",
            "none": "none",
            "required": "required",
        }.get(getattr(choice.mode, "value", str(choice.mode)), "named")
        # P1.9 (review #19): the textual contract carries prompt-mode
        # behavior. Same model + surface + tool_choice with different
        # contracts produces different ladder outcomes — cache by it.
        plan = invocation.invocation_plan
        contract_fingerprint = (
            getattr(plan, "prompt_contract_digest", "") or ""
            if plan is not None else ""
        )
        return attempt_hint_key(
            model_digest=getattr(runtime, "model_digest", "") or "",
            template_digest=getattr(runtime, "chat_template_digest", "") or "",
            serving_config_digest=getattr(runtime, "serving_config_digest", "") or "",
            profile_revision=str(getattr(invocation.model_profile, "profile_revision", "") or ""),
            client_protocol=(
                f"{getattr(invocation.request_context, 'client_id', '') or ''}"
                f"/{getattr(invocation.request_context, 'wire_protocol', '') or ''}"
            ),
            tool_surface_fingerprint=fingerprint,
            streaming=streaming,
            tool_choice_class=choice_class,
            prompted_contract_fingerprint=contract_fingerprint,
        )

    def _invocation_for_attempt(self, invocation: ResolvedInvocation, attempt: Any) -> ResolvedInvocation:
        """Build the request-specific InvocationPlan for one ladder rung."""
        from agent_interop.evidence.key import CompatibilityKeyInputs, build_compatibility_key
        from agent_interop.repair.invocation import build_invocation_plan

        surface = invocation.tool_surface_plan
        plan = build_invocation_plan(
            tools=None,
            tool_choice=invocation.reconciled_request.tool_choice,
            route_mode=attempt.tool_mode,
            model_profile=invocation.model_profile,
            repair_policy=invocation.repair_policy,
            codec_capabilities=invocation.codec.capabilities(),
            upstream_tools=list(surface.visible_tools) if surface is not None else [],
            validation_tools=list(surface.validation_tools) if surface is not None else [],
            capabilities=getattr(invocation, "private_capabilities", None),
        )
        if attempt.constrained_output:
            plan = replace(plan, constrained_output=True)
        # P1-F: reuse the fingerprint from the request's single serialization
        # snapshot (carried on requirements) — history reconciliation never
        # touches tools, so the attempt-level key needs no re-serialization.
        fingerprint = ""
        requirements = invocation.request_requirements
        if requirements is not None:
            fingerprint = getattr(requirements, "tool_schema_fingerprint", "") or ""
        if not fingerprint:
            fingerprint = self._compute_tool_schema_fingerprint(
                invocation.reconciled_request.tools,
            )
        compatibility_key = build_compatibility_key(CompatibilityKeyInputs(
            request_context=invocation.request_context,
            route=invocation.route,
            request=invocation.reconciled_request,
            backend_metadata=invocation.backend_metadata,
            model_profile=invocation.model_profile,
            invocation_plan=plan,
            tool_schema_fingerprint=fingerprint,
            streaming=invocation.reconciled_request.generation.stream,
            runtime_capabilities=invocation.runtime_capabilities,
            compatibility_plan=invocation.compatibility_plan,
            context_plan=invocation.context_plan,
            tool_surface_plan=surface,
            selected_attempt=attempt,
        ))
        return replace(
            invocation,
            invocation_plan=plan,
            compatibility_key=compatibility_key,
            # Evidence for a previous rung is not evidence for this exact
            # path/presentation tuple.
            evidence_record=None,
            compatibility_attempt=attempt,
        )

    def _invocation_with_withheld_tool(
        self, invocation: ResolvedInvocation, tool_name: str,
    ) -> ResolvedInvocation:
        """Rebuild an attempt once after an unseen declared tool was requested."""
        from agent_interop.tool_surface.selector import ToolSurfacePlanner

        if invocation.tool_surface_plan is None:
            return invocation
        surface = ToolSurfacePlanner.replan_with_tool(
            invocation.tool_surface_plan, tool_name,
        )
        if surface is invocation.tool_surface_plan:
            return invocation
        if invocation.execution_record is not None:
            invocation.execution_record.record_compatibility_event(
                f"withheld_tool_replanned:{tool_name}"
            )
        replanned = replace(invocation, tool_surface_plan=surface)
        return self._invocation_for_attempt(replanned, invocation.compatibility_attempt)

    def _rebuild_invocation_from_request(
        self,
        invocation: ResolvedInvocation,
        authoritative_request: Any,
        *,
        plan: Any,
        tool_surface_plan: Any | None = None,
        private_capabilities: Any | None = None,
    ) -> ResolvedInvocation:
        """Atomic invocation rebuild — the canonical implementation lives in
        ``agent_interop.projection.rebuild_invocation_atomic``."""
        return rebuild_invocation_atomic(
            invocation, authoritative_request,
            plan=plan,
            tool_surface_plan=tool_surface_plan,
            private_capabilities=private_capabilities,
        )

    async def _execute_compatibility_attempt(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
    ) -> CanonicalResponse:
        """Dispatch one validated attempt without granting tool authority."""
        if getattr(invocation.compatibility_attempt, "use_controller", False):
            return await self._execute_controller_attempt(invocation, exec_record)
        return await self._handle_request_send(invocation, exec_record)

    async def _execute_controller_attempt(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
    ) -> CanonicalResponse:
        """Delegate to the extracted controlled-attempt executor (bounded
        primary↔controller loop)."""
        return await self._controller_attempt.execute(invocation, exec_record)

    async def _select_controller_route(self, primary_route: ModelRoute, config: Any) -> ModelRoute | None:
        """Select an explicitly configured or verified installed controller."""
        from agent_interop.controller.companion import ControllerRegistry

        if not config.enabled:
            return None
        registry = ControllerRegistry()
        candidates = registry.candidates(self.config, primary_route)
        explicit_id = config.route_id
        for candidate in candidates:
            is_explicit = bool(explicit_id and candidate.id == explicit_id)
            # Automatic selection must never promote an arbitrary installed
            # route. Explicit legacy configuration may still opt out of the
            # qualification gate for backwards compatibility.
            if await self._controller_route_is_qualified(candidate, config.minimum_controller_level):
                return candidate
            if is_explicit and not config.require_verified:
                return candidate
        return None

    async def _controller_route_is_qualified(self, route: ModelRoute, minimum_level: str) -> bool:
        """Check a controller against the bounded bootstrap qualification state."""
        runtime = await self._inspect_model_runtime(route)
        record = self._qualification_record_for_runtime(runtime)
        if record is None:
            return False
        return state_meets_controller_level(
            getattr(record, "state", None), minimum_level,
        )

    async def _prepare_model_generation(
        self,
        invocation: ResolvedInvocation,
        *,
        stream: bool,
    ) -> tuple[Any, bytes, Any, int, int] | CanonicalResponse:
        """Delegate to the generation seam: render, serialize once, gate."""
        return await self._generation_seam.prepare(
            invocation,
            stream=stream,
            attempt_request=self.prepare_model_request_for_attempt(
                invocation, invocation.invocation_plan,
            ),
            context_limit_error=self._context_limit_error,
        )

    async def _dispatch_model_generation(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
        *,
        prepared: tuple[Any, bytes, Any, int, Any],
        purpose: str,
    ) -> tuple[Any, bytes, Any | None]:
        """Delegate to the generation seam: reserve, admit, send.

        On success the third element is the still-OPEN reservation — the
        caller commits it with post-decode actuals via the seam's
        ``finalize_reservation``; on failure it is None and reconciliation
        already happened inside the seam.
        """
        return await self._generation_seam.dispatch(
            invocation, exec_record, prepared=prepared, purpose=purpose,
            context_limit_error=self._context_limit_error,
        )

    async def _handle_request_send(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
    ) -> CanonicalResponse:
        """Send a prepared invocation to the upstream and decode the response.

        Steps:
            1. Handle unsafe history (return error response)
            2. Render the reconciled request through the codec
            3. Apply the invocation plan (tool mode adjustments)
            4. Build typed PreparedUpstreamRequest
            5. Send with bounded retries
            6. Decode the upstream response
            7. Extract tool calls from model-dialect output
            8. Run the tool transaction pipeline
            9. Assemble the canonical response
        """
        # Unsafe history — return error response
        if invocation.invocation_plan is None or invocation.codec is None:
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=invocation.reconciled_request.model.requested_name,
                    resolved_name=invocation.route.upstream_model,
                ),
                error=CanonicalError(
                    code=InteropErrorCode.HISTORY_UNSAFE,
                    message="History reconciliation detected unsafe history",
                ),
            )

        route = invocation.route
        plan = invocation.invocation_plan
        codec = invocation.codec
        canonical = invocation.reconciled_request

        choice_conflict = self._disabled_tool_choice_conflict(plan)
        if choice_conflict is not None:
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=choice_conflict,
            )

        # 1-2. THE generation seam: render, serialize once, exact-context
        # gate. Every generation kind passes here (P0-12) — presentation and
        # context safety cannot drift between public, private, controller,
        # and probe generations.
        prepared = await self._prepare_model_generation(invocation, stream=False)
        if isinstance(prepared, CanonicalResponse):
            return prepared
        # Rendered form + byte count are consumed by calibration below; the
        # gate/meter/output-reserve tuple elements stay inside the seam.
        _request_local, rendered_bytes, _meter, _reserve, rendered = prepared

        # 3-4. Dispatch under admission with full budget reconciliation
        # (GenerationReservation commit/release, rendered-byte accounting).
        # On success the reservation stays OPEN — actual usage exists only
        # after the codec decodes below, so committing at dispatch time
        # would strand every generation on its estimate.
        purpose = (
            "controller" if getattr(invocation.compatibility_attempt, "use_controller", False)
            else "worker"
        )
        response, rendered_bytes, open_reservation = await self._dispatch_model_generation(
            invocation, exec_record, prepared=prepared, purpose=purpose,
        )
        # The seam returns CanonicalResponse (always carrying .error) on
        # admission/context/budget/transport failure, and the raw
        # UpstreamResponse on success so the decode pipeline can feed the
        # codec. Every CanonicalResponse return is terminal here (the seam
        # already reconciled the reservation on those paths).
        if isinstance(response, CanonicalResponse):
            return response

        def _reconcile_decode_failure() -> None:
            # Dispatched but the wire answer was unusable — conservative
            # commit_estimated (the backend spent the tokens), matching the
            # stream path's dispatched-failure rule.
            if open_reservation is not None:
                open_reservation.commit_estimated()
                budget = getattr(exec_record, "attempt_budget", None)
                if budget is not None:
                    budget.record_rendered_bytes(len(rendered_bytes))

        if response.is_error():
            _reconcile_decode_failure()
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=CanonicalError(
                    code=classify_http_status(response.status_code),
                    message=(
                        f"Upstream returned {response.status_code}: "
                        f"{response.body[:500].decode('utf-8', errors='replace')}"
                    ),
                ),
            )

        # 5. Decode upstream response
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            _reconcile_decode_failure()
            return CanonicalResponse(
                content=[],
                stop_reason=CanonicalStopReason.END_TURN,
                usage=CanonicalUsage(),
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=CanonicalError(
                    code="INVALID_UPSTREAM_OUTPUT",
                    message=f"Upstream returned non-JSON response (status={response.status_code})",
                ),
            )

        decoded = codec.decode_response(data)

        # 5b. The generation happened and its REAL usage is now known —
        # replace the reservation's estimate with actuals (exactly one
        # commit per generation, same rule as the stream tail).
        if open_reservation is not None:
            self._generation_seam.finalize_reservation(decoded, open_reservation)
            budget = getattr(exec_record, "attempt_budget", None)
            if budget is not None:
                budget.record_rendered_bytes(len(rendered_bytes))

        # P0.42 (review #20): feed the backend's REAL prompt token count back
        # into the meter, keyed by the full serving identity, so the next
        # request's budget uses an exact count instead of a conservative
        # estimate.
        self._calibrate_token_meter(
            invocation, rendered, getattr(decoded, "usage", None),
            rendered_byte_count=len(rendered_bytes),
        )

        # 6. Extract tool calls from model-dialect output
        candidates = self._extract_tool_candidates(decoded, invocation)

        # P0.4/P0.5: Any internal call means the model's turn is NOT finished —
        # execute the enabled internal tools privately and continue the turn.
        # The continuation returns the FINAL public turn content, from which
        # candidates are re-derived; internal identity never crosses the
        # client boundary.
        enabled_internal = self._enabled_internal_tools(invocation)
        internal_candidates = [c for c in candidates if c.name in enabled_internal]
        if internal_candidates:
            continued = await self._private_loop.run(
                invocation,
                exec_record,
                internal_candidates=internal_candidates,
                decoded=decoded,
                budget=getattr(exec_record, "attempt_budget", None),
            )
            if isinstance(continued, CanonicalResponse):
                return continued  # terminal: loop/budget failure, already firewalled
            decoded = continued
            candidates = self._extract_tool_candidates(decoded, invocation)

        # A withheld declared tool is not executable merely because a model
        # guessed its identifier. Record the event and let the bounded ladder
        # expose it for exactly one retry.
        surface = invocation.tool_surface_plan
        withheld = set(surface.withheld_tool_names) if surface is not None else set()
        requested_withheld = next((candidate.name for candidate in candidates if candidate.name in withheld), "")
        if requested_withheld:
            if exec_record is not None:
                exec_record.record_compatibility_event(
                    f"withheld_tool_requested:{requested_withheld}"
                )
            return CanonicalResponse(
                model=CanonicalModelReference(
                    requested_name=canonical.model.requested_name,
                    resolved_name=route.upstream_model,
                ),
                error=CanonicalError(
                    code=InteropErrorCode.TOOL_SELECTION_FAILED,
                    message="Model requested a declared tool outside its visible surface",
                    details={"withheld_tool_requested": requested_withheld},
                ),
            )

        # 7. Run tool transaction pipeline

        transaction_context = self._build_transaction_context(invocation, canonical)
        batch_decision = await process_tool_batch(
            candidates,
            canonical.tools,
            context=transaction_context,
            policy=ToolBatchPolicy(invocation.repair_policy.batch_policy),
        )

        # 7.5 Record repairs into session state for loop detection
        self._record_repairs_to_session(batch_decision, invocation.request_context)

        # 7.6 Record per-call decisions onto the shared execution record. The
        # in-memory record is always populated (so finalize_response's outcome
        # classification sees the decisions); evidence-store write-back is a
        # separate, opt-in step in _record_evidence_observation.
        self._record_tool_decisions(batch_decision, exec_record)

        # 8. Assemble canonical response
        return self._assemble_response(decoded, batch_decision, canonical, route, exec_record=exec_record)

    def _disabled_tool_choice_conflict(self, plan: Any) -> CanonicalError | None:
        """A DISABLED route combined with a required/named tool choice is a
        contradiction: the client demands a tool call the route guarantees
        will never be produced. Reject before contacting the backend rather
        than silently ignoring the choice."""
        if plan is None or plan.effective_tool_mode != ToolMode.DISABLED:
            return None
        if plan.original_tool_choice is None:
            return None
        mode = plan.original_tool_choice.mode
        if mode == ToolChoiceMode.REQUIRED or mode == ToolChoiceMode.NAMED:
            return CanonicalError(
                code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                message=(
                    f"tool_choice={mode.value!r} requires a tool call, but this route's "
                    "tool_mode is disabled"
                ),
            )
        return None

    def _config_tool_choice_conflict(self, route: ModelRoute, request: CanonicalRequest) -> CanonicalError | None:
        """Route-config-level contradiction check, runnable BEFORE any I/O.

        When the route's configured ``tool_mode`` is DISABLED outright, a
        REQUIRED/NAMED tool choice is already unsatisfiable — resolution
        (which consults profile/codec) can only keep it DISABLED or leave
        it, never make the contradiction satisfiable. Detecting it here
        means a contradictory request never triggers even metadata I/O,
        let alone a generation.
        """
        if route.tool_mode != ToolMode.DISABLED:
            return None
        choice = request.tool_choice
        if choice is not None and choice.mode in (ToolChoiceMode.REQUIRED, ToolChoiceMode.NAMED):
            return CanonicalError(
                code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                message=(
                    f"tool_choice={choice.mode.value!r} requires a tool call, but this route's "
                    "tool_mode is DISABLED"
                ),
            )
        return None

    def _context_limit_error(
        self,
        invocation: Any,
        reason: str,
    ) -> CanonicalResponse:
        """Bounded CONTEXT_LIMIT_EXCEEDED response after final rendered
        measurement (P0-12). Unlike planning-time failures this fires on the
        EXACT body about to be sent, including prompted contracts, private
        schemas, and continuation history.

        P1.12 (review #32): takes a :class:`ResolvedInvocation` so the
        canonical request, route, and plan cannot drift out of sync at
        the gate. Pre-fix callers passed the three components
        separately, which made it trivial to mis-thread them in
        streaming/continuation paths and silently mis-report the
        responsible limit.
        """
        canonical = invocation.reconciled_request
        route = invocation.route
        context_plan = getattr(invocation, "context_plan", None)
        details: dict[str, Any] = {
            "gate": "rendered_body",
            "reason": reason,
            "path": "preflight",
            "responsible": "request_size",
        }
        if context_plan is not None:
            details["safe_limit_tokens"] = getattr(context_plan, "safe_limit_tokens", 0)
            details["runtime_limit_tokens"] = getattr(context_plan, "runtime_limit_tokens", 0)
        return CanonicalResponse(
            content=[],
            stop_reason=CanonicalStopReason.END_TURN,
            usage=CanonicalUsage(),
            model=CanonicalModelReference(
                requested_name=canonical.model.requested_name,
                resolved_name=route.upstream_model,
            ),
            error=CanonicalError(
                code=InteropErrorCode.CONTEXT_LIMIT_EXCEEDED,
                message="Rendered request exceeds the model's effective context limit",
                details=details,
            ),
        )

    # ─── Private internal-tool continuation loop (P0.4/P0.5) ──────────────

    def _enabled_internal_tools(self, invocation: ResolvedInvocation) -> dict[str, Any]:
        """The executable internal-tool registry for THIS request.

        P0-24: capability IS authority — a request may privately execute only
        the internal tools its own :class:`PrivateCapabilityPlan` admitted to
        the model surface. The previous registry handed out every internal
        tool globally, so a capability the projection never advertised (and
        the client never opted into) was still executable if the model
        hallucinated its name — surface and execution disagreed.

        Every internal tool remains a side-effect-free READ scoped to the
        caller's own session (the ContextStore fails closed across
        sessions), so the blast radius of a granted capability is the
        request's own session state. Unknown non-Interop names are never
        here — they fall through to the client transaction layer.
        """
        from agent_interop.context_store.schema_tools import all_schema_tools
        from agent_interop.context_store.tools import all_internal_tools

        capabilities = invocation.private_capabilities
        if capabilities is None:
            # No capability plan → no private authority at all.  (Defensive:
            # _no_private_capabilities() is the normal producer of this state.)
            return {}
        by_name: dict[str, Any] = {}
        for tool in (*all_internal_tools(), *all_schema_tools()):
            by_name[tool.name] = tool
        enabled: dict[str, Any] = {}
        if capabilities.read_result:
            enabled["__interop_read_result"] = by_name["__interop_read_result"]
        if capabilities.recall_history:
            enabled["__interop_recall_history"] = by_name["__interop_recall_history"]
        if capabilities.search_history:
            enabled["__interop_search_history"] = by_name["__interop_search_history"]
        if capabilities.get_tool_schema:
            enabled["__interop_get_tool_schema"] = by_name["__interop_get_tool_schema"]
        return enabled

    def _request_identity(self, invocation: ResolvedInvocation, call_ids: set[str]) -> Any:
        """Build the request-scoped identity for the output firewall.

        Refs created by this request's projection AND refs minted later by
        internal executors (schema-on-demand) are the authoritative leak
        identity: a ref is an unguessable token, so its appearance in public
        content IS the leak signal — no marker-syntax pattern needed. The
        request's ref registry is the single source for both.
        """
        from agent_interop.private_loop import InternalIdentity

        registry = getattr(invocation.execution_record, "ref_registry", None)
        if registry is not None:
            refs = registry.snapshot()
        else:
            refs = frozenset(getattr(invocation, "pinned_refs", ()) or ())
        names = tuple(self._enabled_internal_tools(invocation))
        return InternalIdentity(
            call_ids=frozenset(ids for ids in call_ids if ids),
            refs=refs,
            tool_names=frozenset(names),
        )

    async def _send_one_model_step(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
        *,
        purpose: str = "private_continuation",
        attempt_request: CanonicalRequest | None = None,
    ) -> tuple[CanonicalResponse, bytes]:
        """One generation with no extraction and no transaction pipeline —
        the private continuation loop's unit of work, delegated to the
        generation seam (which owns budget reservation, admission, and
        decoding into the uniform CanonicalResponse shape).

        ``attempt_request`` narrows the rendered surface for this step only
        (the tool-repair adapter passes a no-tool correction request); the
        invocation's resolved facts are reused untouched."""
        return await self._generation_seam.run_step(
            invocation, exec_record, purpose=purpose,
            context_limit_error=self._context_limit_error,
            attempt_request=attempt_request,
        )

    def _extract_tool_candidates(
        self,
        decoded: DecodedModelResponse,
        invocation: ResolvedInvocation,
    ) -> list[RawToolCallCandidate]:
        """Extract raw tool call candidates from a decoded model response.

        Merges codec-native candidates (``decoded.tool_candidates``), any
        pre-structured ``CanonicalToolCallBlock`` entries in content, and —
        when the invocation plan specifies a textual parser — candidates
        recovered by the ``ExtractorRegistry`` from the model's raw text
        output. Mutates ``decoded.content`` in place to remove consumed
        envelope text so it never leaks into the assembled response.
        """
        route = invocation.route
        plan = invocation.invocation_plan

        # Fail-closed boundary: a DISABLED route must never surface a tool
        # call regardless of what the model or backend emits. This check is
        # deliberately redundant with build_invocation_plan() clearing
        # parser_id/fallback_strategies for DISABLED plans — a future
        # plan-construction regression must not silently reactivate tools.
        if plan is None or plan.effective_tool_mode == ToolMode.DISABLED:
            return []

        native: list[RawToolCallCandidate] = list(decoded.tool_candidates)

        for block in decoded.content:
            if isinstance(block, CanonicalToolCallBlock):
                native.append(RawToolCallCandidate(
                    id=block.id,
                    name=block.name,
                    raw_arguments=json.dumps(block.arguments) if isinstance(block.arguments, dict) else str(block.arguments),
                    source_protocol=route.upstream.wire_protocol.value,
                    source_index=0,
                    choice_index=0,
                    tool_index=0,
                ))

        textual: list[RawToolCallCandidate] = []
        if plan is not None and plan.parser_id:
            # Textual extraction always runs when a parser is configured, even
            # when native candidates are already present. A hybrid response (one
            # that carries a native tool call AND a distinct textual <tool_call>
            # envelope) must contribute both: the native call and the textual
            # one. Shadow duplicates — a textual echo that exactly matches a
            # native candidate — are removed by _dedup_tool_candidates below,
            # which merges on (choice_index, tool_index, name, normalized_args),
            # so a well-formed native response is never masked by its own echo
            # while a genuinely distinct hybrid call survives.
            result = self._extractor_registry.extract(
                decoded.content,
                extractor_id=plan.parser_id,
                tools=plan.validation_tools,
                envelope=plan.output_envelope,
                fallback_strategies=plan.fallback_strategies,
                tool_choice=plan.original_tool_choice,
                native_candidates_present=bool(native),
                expected_execution_nonce=plan.execution_nonce,
            )
            textual = list(result.candidates)
            decoded.content = list(result.remaining_content)
            if invocation.execution_record is not None:
                for diag in result.diagnostics:
                    invocation.execution_record.record_parser_diagnostic(
                        f"[{diag.level}] {diag.envelope}: {diag.message}"
                    )

        return self._dedup_tool_candidates(native, textual)

    def _dedup_tool_candidates(
        self,
        native: list[RawToolCallCandidate],
        textual: list[RawToolCallCandidate],
    ) -> list[RawToolCallCandidate]:
        """Merge native and textually-extracted candidates, dropping a
        textual echo of a call the backend already reported natively.

        Every native candidate always survives unchanged — this function
        only ever decides whether a TEXTUAL candidate is a duplicate of
        some native one, never native-vs-native or textual-vs-textual, so
        genuinely parallel identical calls within either list are never
        touched here.

        Two matching strategies, applied in order:

        1. Provider call ID, when both sides have a real (non-empty) one.
           An exact ID match is the strongest possible duplicate signal —
           unrelated to name/arguments/index — so it settles the question
           on its own. Two candidates with DIFFERENT non-empty IDs are
           never merged by content alone; that would collapse genuinely
           distinct parallel calls that happen to share identical
           name+arguments (audit finding: "collapse distinct identical
           calls with separate IDs").
        2. Content signature (name + normalized arguments) — used ONLY as
           a fallback when the textual candidate has no ID to compare
           (the common case: text-dialect extractors rarely have access
           to the backend's native call ID). Deliberately excludes
           choice_index/tool_index: pre-structured candidates (e.g.
           whole-message JSON) are constructed with those indexes forced
           to 0 regardless of their real position, so index equality is
           neither necessary (a genuine echo can land at a different
           index) nor sufficient (two unrelated zero-indexed candidates
           would falsely look identical) for this decision.
        """
        if not textual:
            return native
        if not native:
            return textual

        def _sig(c: RawToolCallCandidate) -> tuple:
            return (c.name, _canonicalize_json_ish(c.raw_arguments))

        native_ids = {c.id for c in native if c.id}
        native_sigs = {_sig(c) for c in native}

        merged = list(native)
        for c in textual:
            if c.id:
                if c.id in native_ids:
                    continue  # confirmed duplicate — exact provider ID match
            elif _sig(c) in native_sigs:
                continue  # no ID to compare — fall back to content echo suppression
            merged.append(c)
        return merged

    def _apply_invocation_plan_to_request(
        self,
        rendered: dict[str, Any],
        plan: Any,
        route: ModelRoute,
    ) -> dict[str, Any]:
        """Apply the invocation plan to the rendered request.

        - NATIVE: send plan.upstream_tools (already in rendered)
        - PROMPTED: remove native tools, inject prompt_contract
        - DISABLED: remove tools, reject required/named choice
        - TEXTUAL: remove native tools
        """
        from agent_interop.config import ToolMode

        if plan.effective_tool_mode == ToolMode.NATIVE:
            # Tools already rendered
            pass
        elif plan.effective_tool_mode == ToolMode.PROMPTED:
            # Remove native tools and inject the contract
            rendered.pop("tools", None)
            rendered.pop("tool_choice", None)
            # Inject prompt contract into system message
            if plan.prompt_contract:
                rendered = self._inject_prompt_contract(rendered, plan.prompt_contract)
        elif plan.effective_tool_mode == ToolMode.DISABLED:
            # Remove all tools
            rendered.pop("tools", None)
            rendered.pop("tool_choice", None)
        elif plan.effective_tool_mode == ToolMode.TEXTUAL:
            # Remove native tools
            rendered.pop("tools", None)
            rendered.pop("tool_choice", None)

        self._apply_route_runtime_options(rendered, route)
        return rendered

    @staticmethod
    def _apply_route_runtime_options(rendered: dict[str, Any], route: ModelRoute) -> None:
        """Apply route-owned inference settings without overriding client controls."""
        from agent_interop.config import UpstreamKind

        if route.upstream.kind != UpstreamKind.OLLAMA or not route.upstream.ollama_num_ctx:
            return
        options = rendered.setdefault("options", {})
        if isinstance(options, dict):
            options.setdefault("num_ctx", route.upstream.ollama_num_ctx)

    def _inject_prompt_contract(self, rendered: dict[str, Any], contract: str) -> dict[str, Any]:
        """Inject the prompt contract into the system message."""
        messages = rendered.get("messages", [])
        if messages and messages[0].get("role") == "system":
            # Append to existing system message
            messages[0]["content"] = (messages[0].get("content", "") + "\n\n" + contract).strip()
        else:
            # Prepend a new system message
            messages.insert(0, {"role": "system", "content": contract})
        rendered["messages"] = messages
        return rendered

    def _resolve_profile(self, route: ModelRoute, backend_metadata: Any = None) -> Any:
        """Resolve the model profile for a route using ModelProfileRegistry.

        ``backend_metadata`` was previously computed by the caller (for the
        compatibility key) but never actually passed into ``resolve()`` —
        the registry's documented priority-5 tier ("backend metadata") was
        unreachable dead code from the live request path; only explicit
        profile ID, built-in pattern match, and the conservative fallback
        could ever fire.
        """
        return self._profile_registry.resolve(
            model_name=route.upstream_model,
            backend=route.upstream.kind,
            backend_metadata=backend_metadata,
            explicit_profile_id=route.profile if route.profile != "auto" else None,
        )

    def _get_backend_metadata(self, route: ModelRoute) -> Any:
        """Build BackendMetadata from route config (item 83).

        Populates what's available; empty strings for unknown dimensions.
        The evidence key still works — it just has fewer discriminating fields.
        """
        from agent_interop.model.registry import BackendMetadata

        return BackendMetadata(
            backend_kind=route.upstream.kind,
            model_name=route.upstream_model,
        )

    async def _inspect_model_runtime(self, route: ModelRoute) -> Any:
        """Inspect the live backend/model tuple through the shared transport.

        P0-1: METADATA-ONLY. Live request traffic never runs behavioral
        probe generations — those belong to explicit qualification tooling
        (``interop qualify``, conformance commands). Ollama's four metadata
        reads (/version, /tags, /show, /ps) are concurrent and TTL-cached,
        so a warm route pays zero inspection latency.

        ``runtime_inspection.mode`` governs this independently of
        ``probe_on_startup``: health probing and planning metadata are
        separate concerns.
        """
        from agent_interop.backends.registry import get_backend_inspector

        if self.config.runtime_inspection.mode == "off":
            return self._static_runtime_capabilities(route)

        cached = self._runtime_capability_cache.get_for_route(
            route.upstream.base_url, route.upstream_model,
        )
        if cached is not None:
            return cached

        inspector = get_backend_inspector(route.upstream.kind)
        try:
            if hasattr(inspector, "inspect_runtime_metadata"):
                runtime = await inspector.inspect_runtime_metadata(
                    route,
                    self.transport,
                )
            else:  # pragma: no cover - non-Ollama inspectors without the split
                runtime = await inspector.inspect(route, self.transport)
            self._runtime_capability_cache.put(runtime, route.upstream.base_url)
            return runtime
        except (AttributeError, OSError, RuntimeError) as exc:
            # Inspection enriches planning but must not break a working route
            # when a test/dedicated streaming transport exposes only ``stream``
            # or when a backend's metadata endpoint is unavailable.
            logger.debug("Runtime inspection unavailable for %s: %s", route.id, exc)
            return self._static_runtime_capabilities(route)

    async def _warm_runtime_metadata(self) -> None:
        """Warm the runtime-metadata cache for every route concurrently.

        Called from ``startup()`` so the user's FIRST generation does not pay
        the (cheap but nonzero) metadata round trips. Purely metadata — no
        model generations, matching the P0-1 invariant.

        P1.11 (review #30): the operator can disable warm-up via
        ``runtime_inspection.warm_on_startup=False`` — useful when the
        upstream is slow to respond on cold boot or when the operator
        wants to make the first-request behavior explicit instead of
        implicit. Default is on (warm) for backwards-compatible behavior.
        """
        if not getattr(self.config.runtime_inspection, "warm_on_startup", True):
            logger.debug(
                "Runtime metadata warm-up disabled (runtime_inspection.warm_on_startup=False)",
            )
            return
        if self.config.runtime_inspection.mode != "cached_metadata":
            return
        if not self.config.routes:
            return
        results = await asyncio.gather(
            *(self._inspect_model_runtime(route) for route in self.config.routes.values()),
            return_exceptions=True,
        )
        for route, outcome in zip(self.config.routes.values(), results):
            if isinstance(outcome, BaseException):
                logger.debug(
                    "Runtime metadata warm-up skipped for %s: %s", route.id, outcome,
                )

    @staticmethod
    def _static_runtime_capabilities(route: ModelRoute) -> Any:
        """Return conservative offline facts for synchronous diagnostics.

        This never upgrades a model to native/direct tool capability.  It
        exists solely to keep the historical inspection helper side-effect
        free; live gateway requests always use ``_inspect_model_runtime``.
        """
        from agent_interop.backends.base import ModelRuntimeCapabilities

        return ModelRuntimeCapabilities(
            backend_kind=route.upstream.kind,
            model_name=route.upstream_model,
        )

    def _calibrate_token_meter(
        self,
        invocation: ResolvedInvocation,
        rendered: Any,
        usage: Any | None,
        rendered_byte_count: int | None = None,
    ) -> None:
        """P0.42 (review #20): calibrate the TokenMeter from a real response.

        Each completed generation feeds the backend's actual input token
        count back into the meter, keyed by the full serving identity
        (model_digest, chat_template_digest, wire_protocol), so subsequent
        budget decisions use an exact count instead of the conservative
        bytes/token estimate.  Missing usage or digest → no-op.

        P0-11: callers pass ``rendered_byte_count`` (the length of the bytes
        serialized exactly once in the send path); re-serializing `rendered`
        here is a fallback only for legacy callers.
        """
        if usage is None:
            return
        runtime = getattr(invocation, "runtime_capabilities", None)
        digest = getattr(runtime, "model_digest", "") if runtime else ""
        if not digest:
            return
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        if input_tokens <= 0:
            return
        if rendered_byte_count is None:
            try:
                rendered_byte_count = len(json.dumps(rendered).encode("utf-8", "replace"))
            except (TypeError, ValueError):
                return
        wire_protocol = ""
        route = getattr(invocation, "route", None)
        if route is not None:
            wire_protocol = getattr(route.upstream.wire_protocol, "value", "")
        self._token_meter.calibrate(
            digest,
            input_tokens,
            rendered_byte_count,
            chat_template_digest=getattr(runtime, "chat_template_digest", "") or "",
            wire_protocol=wire_protocol,
        )

    # ─── Qualification (delegated to qualification.coordinator) ──────────

    @staticmethod
    def _qualification_key(runtime: Any) -> str:
        return QualificationCoordinator.key(runtime)

    def record_qualification(self, record: Any) -> None:
        self._qualification.record(record)

    def _qualification_record_for_runtime(self, runtime: Any) -> Any | None:
        return self._qualification.record_for_runtime(runtime)

    def _behavioral_capabilities(self, runtime: Any) -> Any:
        return self._qualification.behavioral_capabilities(runtime)

    def _required_probes_for_request(
        self,
        invocation: ResolvedInvocation,
        record: Any | None,
    ) -> tuple[str, ...]:
        return self._qualification.required_probes_for_request(invocation, record)

    async def _ensure_bootstrap_qualification(self, invocation: ResolvedInvocation) -> bool:
        return await self._qualification.ensure_bootstrap(invocation)

    async def _execute_bootstrap_probe(self, invocation: ResolvedInvocation, probe: Any) -> bool:
        return await self._qualification.execute_bootstrap_probe(invocation, probe)

    async def qualify_route(
        self,
        model_digest: str,
        runtime: Any,
        *,
        scope: str = "full",
    ) -> Any:
        """Run the synthetic bootstrap battery for one resolved model.

        Restores the entry point ``interop qualify`` and
        ``InteropRuntime.qualify`` call: the dee1a97 god-object extraction
        moved probe execution into the qualification coordinator but left
        no public route-qualification entry, so both callers raised
        ``AttributeError`` at runtime. This delegates to the same
        coordinator the request path uses — probes run through
        ``execute_bootstrap_probe``'s isolated route overrides, never by
        mutating serving config — and records the resulting evidence under
        the runtime's digest key. ``scope`` is accepted for API
        compatibility; the battery is the same bounded synthetic set
        either way.
        """
        from agent_interop.qualification import BootstrapQualifier

        key = self._qualification_key(runtime) or model_digest
        existing = self._qualification.record_for_runtime(runtime)

        async def execute(probe: Any) -> bool:
            # Build a minimal probe invocation through the standard
            # preparation path so the probe exercises the same machinery
            # (plan, admission, seam) as any request.
            invocation = await self._probe_invocation_for(runtime)
            return await self._qualification.execute_bootstrap_probe(invocation, probe)

        record = await BootstrapQualifier().qualify_demand(
            key,
            execute,
            existing=existing,
            want_native=True,
            need_continuation=True,
            template_digest=getattr(runtime, "chat_template_digest", ""),
        )
        self.record_qualification(record)
        return record

    async def _probe_invocation_for(self, runtime: Any) -> Any:
        """Build the minimal invocation one synthetic probe runs against."""
        from agent_interop.abi import (
            CanonicalGenerationOptions,
            CanonicalMessage,
            CanonicalRequest,
            CanonicalTextBlock,
            CanonicalToolChoice,
        )
        from agent_interop.execution import InteropRequestExecution

        route = self.get_route_for_model(getattr(runtime, "model_name", "") or "")
        if route is None:
            from agent_interop.errors import InteropError, InteropErrorCode

            raise InteropError(
                code=InteropErrorCode.MODEL_NOT_FOUND,
                message=(
                    f"no route serves model "
                    f"{getattr(runtime, 'model_name', '')!r} — cannot qualify"
                ),
            )
        request = CanonicalRequest(
            model=CanonicalModelReference(requested_name=getattr(runtime, "model_name", "") or ""),
            generation=CanonicalGenerationOptions(max_output_tokens=64, stream=False),
            messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="probe")])],
            tool_choice=CanonicalToolChoice.none(),
        )
        execution = InteropRequestExecution(context=RequestContext())
        return await self._prepare_invocation_async(
            request,
            execution.context,
            streaming=False,
            execution=execution,
        )

    @staticmethod
    def _backend_metadata_from_runtime(runtime: Any) -> Any:
        from agent_interop.model.registry import BackendMetadata

        return BackendMetadata(
            backend_kind=runtime.backend_kind,
            backend_version=runtime.backend_version,
            model_name=runtime.model_name,
            model_digest=runtime.model_digest,
            context_length=runtime.effective_context_tokens,
            chat_template=runtime.chat_template,
            quantization=runtime.quantization,
        )

    def _compute_tool_schema_fingerprint(self, tools: list[Any]) -> str:
        """Compute a fingerprint of the tool schema set for evidence lookup (item 83).

        Uses a hash of tool names + schema structure so that evidence is
        invalidated when tools change.
        """
        import hashlib
        import json

        if not tools:
            return ""
        # Canonical representation: sorted by name, with schema
        canonical = sorted(
            [{"name": t.name, "schema": t.input_schema} for t in tools],
            key=lambda x: x["name"],
        )
        raw = json.dumps(canonical, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _compute_argument_digest(self, arguments: Any) -> str:
        """Compute a stable digest of a tool call's arguments for loop detection.

        Matches the style of ``_compute_tool_schema_fingerprint``: a SHA-256
        prefix of a canonical JSON representation. Returns "" when no
        arguments are present so that argument-less calls collapse to a
        single digest (and thus still flag genuine repeat calls).
        """
        import hashlib

        if not arguments:
            return ""
        raw = _canonicalize_json_ish(arguments)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _apply_confidence_gate(self, policy: Any, confidence: float) -> Any:
        """Gate risky repair tiers when profile confidence is low (item 86 integration).

        Low confidence (fallback, <0.5): only SYNTAX_ONLY and SAFE_SHAPE tiers.
        Medium confidence (builtin, ~0.8): add COERCIVE.
        High confidence (override/explicit, ≥0.9): all tiers including REGENERATION.
        """
        from dataclasses import replace

        if not hasattr(policy, 'enabled_tiers'):
            return policy

        from agent_interop.config import RepairTier

        current = set(policy.enabled_tiers)

        if confidence < 0.5:
            # Low confidence — disable coercive and regeneration
            current.discard(RepairTier.COERCIVE)
            current.discard(RepairTier.REGENERATION)
        elif confidence < 0.9:
            # Medium confidence — allow coercive but not regeneration
            current.discard(RepairTier.REGENERATION)
        # High confidence — keep all enabled tiers

        if current == set(policy.enabled_tiers):
            return policy
        return replace(policy, enabled_tiers=frozenset(current))

    @staticmethod
    def _build_repair_note(decisions: list[Any]) -> str | None:
        """Compact, model-facing note describing what the repair pipeline
        actually changed this turn — never emitted for calls that were
        already valid (VALID_UNCHANGED), only for ones that needed and
        got a repair. Intent: showing the model what was normalized (e.g.
        "old_str -> old_string") can reduce repeat mistakes of the same
        kind within a session, the same way a compiler/lint error shown
        back to a coding agent helps it self-correct.
        """
        notes: list[str] = []
        for d in decisions:
            outcome = getattr(d, "outcome", None)
            if outcome is None or outcome.status != RepairStatus.REPAIRED or not outcome.steps:
                continue
            tool_name = outcome.call_name or getattr(d.candidate, "name", None) or "tool call"
            step_msgs = "; ".join(step.message for step in outcome.steps if step.message)
            if step_msgs:
                notes.append(f"{tool_name}: {step_msgs}")
        if not notes:
            return None
        return "[Interop] Normalized before executing — " + " | ".join(notes)

    def _assemble_response(
        self,
        decoded: DecodedModelResponse,
        batch_decision: Any,
        canonical: CanonicalRequest,
        route: ModelRoute,
        exec_record: InteropRequestExecution | None = None,
    ) -> CanonicalResponse:
        """Assemble the canonical response from decoded + transaction output."""

        content: list[CanonicalContentBlock] = []

        # Add non-tool content
        for block in decoded.content:
            if not isinstance(block, CanonicalToolCallBlock):
                content.append(block)

        # Add accepted tool blocks from the transaction decision
        content.extend(batch_decision.accepted_blocks)

        # Repair-note feedback (P1.4): a short, structured note naming exactly
        # what got normalized, only when a repair actually fired. It is
        # recorded on the EXECUTION RECORD — never appended to the
        # client-visible response, where it would corrupt the coding client's
        # transcript (the client owns assistant-turn content). The internal
        # recall surface resurfaces it to the model on later turns.
        repair_note = self._build_repair_note(getattr(batch_decision, "decisions", []))
        if repair_note and exec_record is not None:
            exec_record.record_repair_hint(repair_note)

        # Determine stop reason
        stop_reason = decoded.stop_reason
        if batch_decision.accepted_blocks and stop_reason == CanonicalStopReason.END_TURN:
            stop_reason = CanonicalStopReason.TOOL_CALL

        # Check for batch-level errors
        error = None
        if not batch_decision.is_accepted and not batch_decision.accepted_blocks:
            # Complete batch rejection. Never report a TOOL_CALL stop reason
            # when the whole batch was rejected — the client must see
            # INVALID_OUTPUT so it knows none of the calls were executed.
            stop_reason = CanonicalStopReason.INVALID_OUTPUT
            error = self._build_batch_rejection_error(batch_decision, canonical.request_id)

        return CanonicalResponse(
            content=content,
            stop_reason=stop_reason,
            usage=decoded.usage,
            model=CanonicalModelReference(
                requested_name=canonical.model.requested_name,
                resolved_name=route.upstream_model,
            ),
            request_id=canonical.request_id,
            response_id=decoded.extra.get("response_id", ""),
            error=error,
        )

    def _build_batch_rejection_error(
        self,
        batch_decision: Any,
        request_id: str,
    ) -> CanonicalError:
        """Build a structured ``CanonicalError`` for a fully-rejected tool batch.

        Two distinct scenarios:
        - **Tool-choice policy violation** (``choice_error`` is set): the batch
          was rejected wholesale because the calls violated the
          REQUIRED/NAMED/NONE choice contract. This gets its own code.
        - **Per-call rejections**: individual calls failed validation/repair.
          The batch-level error is synthesized from the per-decision data so
          the client gets structured, actionable feedback.
        """
        # Tool-choice policy violation — whole batch rejected before per-call
        # processing. This is the canonical "the model disobeyed the choice
        # contract" signal.
        if batch_decision.choice_error:
            return CanonicalError(
                code=InteropErrorCode.TOOL_CHOICE_VIOLATION,
                message=batch_decision.choice_error,
                retryable=True,
                request_id=request_id,
            )

        # Per-call rejections: synthesize a batch-level error from the
        # per-decision data.
        rejected = [d for d in batch_decision.decisions if d.is_rejected]
        messages = [d.outcome.error for d in rejected if d.outcome.error]
        if not messages:
            messages = [d.outcome.final_issues[0].message for d in rejected
                        if d.outcome.final_issues]
        message = "; ".join(messages) if messages else "All tool calls in the batch were rejected"

        # A repair attempt that still failed is retryable (a regeneration may
        # succeed next time); a call that was never valid to begin with is not.
        repair_attempted_failed = any(
            d.outcome.status == RepairStatus.REJECTED and d.outcome.steps
            for d in rejected
        )
        code = (
            InteropErrorCode.TOOL_CALL_REPAIR_FAILED
            if repair_attempted_failed
            else InteropErrorCode.TOOL_CALL_INVALID
        )

        # Structured per-call correction info so the client can see exactly
        # which calls failed and why.
        rejected_calls: list[dict[str, Any]] = []
        for d in rejected:
            correction = d.correction
            if correction is not None:
                rejected_calls.append({
                    "tool_name": correction.tool_name,
                    "candidate_id": correction.candidate_id,
                    "issue_path": correction.issue_path,
                    "schema_keyword": correction.schema_keyword,
                    "observed_type": correction.observed_type,
                    "expected_type": correction.expected_type,
                    "allowed_values": correction.allowed_values,
                    "message": correction.message,
                    "retryable": correction.retryable,
                })
            else:
                rejected_calls.append({
                    "tool_name": d.outcome.call_name or d.candidate.name or "",
                    "candidate_id": d.candidate.id or "",
                    "message": d.outcome.error,
                })

        return CanonicalError(
            code=code,
            message=message,
            retryable=repair_attempted_failed,
            request_id=request_id,
            details={"rejected_calls": rejected_calls},
        )

    # ─── Streaming request ────────────────────────────────────────────────

    async def handle_stream(
        self,
        canonical: CanonicalRequest,
        context: Any,
    ) -> AsyncGenerator[CanonicalEvent, None]:
        """Handle a streaming request, yielding canonical events.

        The streaming path uses the same preparation pipeline as non-streaming:
            _prepare_invocation → codec render → transport → decode
            → extraction → transaction → canonical events

        For NATIVE_FRAGMENTS mode: text streams through immediately, tool fragments
        are accumulated and validated through the transaction service before emission.

        For BUFFER_TEXTUAL_RESPONSE mode: buffer model text until complete, then
        extract and validate.
        """
        exec_record = InteropRequestExecution(context=context)
        try:
            # P0-16: the budget exists BEFORE preparation — and covers BOTH
            # stream shapes, so the direct-streaming path is no longer the
            # only request kind running without a token/attempt ledger.
            from agent_interop.execution_attempts import AttemptBudget

            budget = AttemptBudget()
            exec_record.attempt_budget = budget
            # Prepare the resolved invocation (same as non-streaming; passes in
            # the shared record so diagnostics/route/plan land on the record
            # that gets finalized)
            invocation = await self._prepare_invocation_async(
                canonical, context, streaming=True, execution=exec_record,
            )
            if await self._ensure_bootstrap_qualification(invocation):
                # A stream that requires qualification must not leak its first
                # unqualified attempt.  Re-plan (NOT a full re-prep) so the
                # compatibility key fingerprints the streaming shape the
                # dispatcher will use — a key computed under streaming=False
                # would never match the live gate's stream request.
                invocation = await self._replan_after_qualification(
                    invocation, exec_record, streaming=True,
                )
            budget.max_upstream_attempts = invocation.route.compatibility.max_attempts
            # P0-27: pin this request's refs for the WHOLE streaming
            # generator lifecycle — private continuations inside the stream
            # still need stored content after the first generation, and the
            # finally guarantees unpinning on every exit path including
            # client cancellation. The registry also captures refs minted
            # mid-stream (schema-on-demand) and feeds the firewall identity.
            request_id = getattr(context, "request_id", "") or canonical.request_id
            session_id = getattr(context, "session_id", "") or ""
            from agent_interop.context_store import RequestRefRegistry
            ref_registry = RequestRefRegistry(
                self._context_store, session_id, request_id,
            )
            ref_registry.register_all(getattr(invocation, "pinned_refs", ()) or ())
            invocation = replace(invocation, pinned_refs=())
            exec_record.ref_registry = ref_registry
            try:
                async for event in self._stream_dispatch(
                    invocation, canonical, context, exec_record, budget,
                ):
                    yield event
            finally:
                ref_registry.close()
        except asyncio.CancelledError:
            # Mid-request cancellation: finalize the record as CANCELLED so it is
            # not left permanently ACTIVE, then re-raise. Do NOT yield
            # any further frames — the client connection is already gone.
            exec_record.finalize_cancelled()
            raise
        except Exception as exc:
            exc_err = self._preflight_error(exc)
            if exc_err is None:
                exc_err = CanonicalError(code="STREAM_ERROR", message=str(exc)) if not isinstance(exc, CanonicalError) else exc
            exec_record.finalize_error(exc_err)
            yield CanonicalEvent(type="error", error=exc_err)
            yield CanonicalEvent(type="message_stop")
            return

    async def _stream_dispatch(
        self,
        invocation: Any,
        canonical: CanonicalRequest,
        context: Any,
        exec_record: InteropRequestExecution,
        budget: Any,
    ) -> AsyncGenerator[CanonicalEvent, None]:
        """Buffered-vs-direct stream dispatch (P0-27 refactor): the pin
        lifecycle in handle_stream wraps everything this method yields."""
        if self._requires_buffered_stream_validation(invocation):
            from agent_interop.execution_attempts import CompatibilityAttemptExecutor

            executor = CompatibilityAttemptExecutor(budget)
            response = await executor.execute(
                invocation,
                build_invocation=self._invocation_for_attempt,
                execute_attempt=lambda attempt_invocation: self._execute_compatibility_attempt(
                    attempt_invocation, exec_record,
                ),
                replan_withheld_tool=self._invocation_with_withheld_tool,
                hint_key=lambda inv: self._attempt_hint_key(
                    inv, bool(getattr(inv.reconciled_request.generation, "stream", False)),
                ),
                hint_get=self._attempt_hints.get,
                hint_record=self._attempt_hints.record,
            )
            if (
                self._evidence_store is not None
                and canonical.tools
                and response.error is None
            ):
                self._record_evidence_observation(invocation, exec_record)
            exec_record.finalize_response(response)
            self._capture_diagnostic_case(invocation, exec_record, response)
            async for event in self._events_from_buffered_response(response):
                yield event
            return
        # The sub-generator finalizes the record (success or internal
        # terminal error) — and logs the summary — BEFORE yielding its
        # terminal event, not after. The ASGI server stops consuming this
        # generator as soon as it sees message_stop, so nothing here can
        # rely on running after the last yield to be reached.
        async for event in self._handle_stream_send(invocation, exec_record):
            yield event

    def _stream_safety_key(self, invocation: ResolvedInvocation) -> str:
        """P0-7: the serving tuple a stream-safety observation attaches to."""
        from agent_interop.planning.stream_safety import stream_safety_key

        runtime = getattr(invocation, "runtime_capabilities", None)
        fingerprint = ""
        requirements = getattr(invocation, "request_requirements", None)
        if requirements is not None:
            fingerprint = getattr(requirements, "tool_schema_fingerprint", "") or ""
        if not fingerprint:
            fingerprint = self._compute_tool_schema_fingerprint(
                invocation.reconciled_request.tools,
            )
        choice = invocation.reconciled_request.tool_choice
        choice_class = {
            "auto": "auto",
            "none": "none",
            "required": "required",
        }.get(getattr(choice.mode, "value", str(choice.mode)), "named")
        return stream_safety_key(
            model_digest=getattr(runtime, "model_digest", "") or "",
            template_digest=getattr(runtime, "chat_template_digest", "") or "",
            serving_config_digest=getattr(runtime, "serving_config_digest", "") or "",
            profile_revision=str(getattr(
                getattr(invocation, "model_profile", None), "profile_revision", "",
            ) or ""),
            client_protocol=(
                f"{getattr(getattr(invocation, 'request_context', None), 'client_id', '') or ''}"
                f"/{getattr(getattr(invocation, 'request_context', None), 'wire_protocol', '') or ''}"
            ),
            tool_surface_fingerprint=fingerprint,
            tool_choice_class=choice_class,
        )

    def _requires_buffered_stream_validation(self, invocation: ResolvedInvocation) -> bool:
        """Instance entry: resolve the tuple's stream-safety observation,
        then apply the pure policy."""
        return self._buffered_stream_policy(
            invocation,
            observation_unlocks=self._stream_safety.is_safe(
                self._stream_safety_key(invocation),
            ),
        )

    @staticmethod
    def _buffered_stream_policy(
        invocation: ResolvedInvocation,
        *,
        observation_unlocks: bool,
    ) -> bool:
        """Whether this stream must wait for an accepted ladder response.

        P0-24/P0-25 (review): buffering is a TTFT tax, so it applies ONLY
        when tool interpretation could actually convert prose into
        execution:

          1. P0-25 — private capabilities active ⇒ ALWAYS buffer. Already-
             streamed text cannot be withdrawn if the model later requests
             __interop_read_result; only the firewalled final turn may be
             emitted.
          2. No client tools AND no private capabilities ⇒ never buffer —
             there is no tool surface to misinterpret; text streams
             immediately.
          3. tool_choice=NONE (no private caps) ⇒ never buffer, same reason.
          4. Otherwise: unverified native/prompted tool streams buffer until
             a validated response exists.  Two independent unlock paths —
             P0-7: an operator-certified evidence record for this tuple,
             OR a recent in-process stream-safety observation (one prior
             unbuffered stream on this exact tuple came back fully
             accepted). A deployment that never configured the evidence
             store still earns the streaming fast path.
        """
        if not invocation.route.compatibility.buffer_unverified_streaming:
            return False
        private_capabilities = getattr(invocation, "private_capabilities", None)
        if private_capabilities is not None and private_capabilities.has_any:
            return True
        request = invocation.authoritative_request or invocation.reconciled_request
        if not request.tools:
            return False
        choice = getattr(request, "tool_choice", None)
        if choice is not None and getattr(choice.mode, "value", choice.mode) == "none":
            return False
        plan = invocation.invocation_plan
        compatibility = invocation.compatibility_plan
        if not (
            compatibility is not None
            and getattr(compatibility.path, "value", compatibility.path) == "direct"
            and plan is not None
            and plan.effective_tool_mode == ToolMode.NATIVE
        ):
            return True
        return not (
            invocation.evidence_record is not None or observation_unlocks
        )

    async def _events_from_buffered_response(self, response: CanonicalResponse) -> AsyncIterator[CanonicalEvent]:
        """Encode one validated buffered response as canonical stream events."""
        if response.error is not None:
            yield CanonicalEvent(type="error", error=response.error)
            yield CanonicalEvent(type="message_stop", stop_reason=CanonicalStopReason.INVALID_OUTPUT)
            return
        for index, block in enumerate(response.content):
            if isinstance(block, CanonicalTextBlock) and block.text:
                yield CanonicalEvent(type="text_delta", index=index, partial=block.text)
            elif isinstance(block, CanonicalToolCallBlock):
                yield CanonicalEvent(type="tool_use", index=index, content_block=block)
        if response.usage.input_tokens or response.usage.output_tokens:
            yield CanonicalEvent(
                type="usage_update",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        yield CanonicalEvent(type="message_stop", stop_reason=response.stop_reason)

    async def _handle_stream_send(
        self,
        invocation: ResolvedInvocation,
        exec_record: InteropRequestExecution,
    ) -> AsyncIterator[CanonicalEvent]:
        """Delegate to the extracted streaming engine (frame loop → batch →
        events)."""
        async for event in self._stream_engine.run_send_stream(invocation, exec_record):
            yield event

    # ─── Helpers ───────────────────────────────────────────────────────────

    def _build_upstream_auth_config(self, route: ModelRoute) -> Any:
        """Build a typed UpstreamAuthConfig from the route's loose auth dict.

        The legacy ``route.upstream.api_key_env`` field (used by the probe
        path) is translated into the equivalent typed ``API_KEY`` config here
        when no explicit ``auth`` dict is present, so probing, inference,
        streaming, and count_tokens all resolve upstream auth identically.
        """
        from agent_interop.auth import UpstreamAuthConfig, UpstreamAuthMode

        auth = route.upstream.auth
        if not auth:
            # Translate the legacy api_key_env field into the typed
            # UpstreamAuthConfig mechanism so real requests honor it too.
            if route.upstream.api_key_env:
                return UpstreamAuthConfig(
                    mode=UpstreamAuthMode.API_KEY,
                    env_key=route.upstream.api_key_env,
                )
            return UpstreamAuthConfig(mode=UpstreamAuthMode.NONE)

        mode_str = auth.get("mode", "none")
        try:
            mode = UpstreamAuthMode(mode_str)
        except ValueError:
            # An invalid mode string is a config error, not a silent NONE
            # fallback — but this method can't raise per its contract, so the
            # invalid value is caught and rejected by validate_config at load
            # time (which see). Treat as NONE here as a last-resort default.
            mode = UpstreamAuthMode.NONE

        # The "command" field may be a list or a stringified list; normalize to list
        raw_command: Any = auth.get("command", [])
        if isinstance(raw_command, str):
            import shlex
            command = shlex.split(raw_command)
        elif isinstance(raw_command, list):
            command = raw_command
        else:
            command = []

        return UpstreamAuthConfig(
            mode=mode,
            api_key=auth.get("token") or auth.get("api_key"),
            api_key_header=auth.get("api_key_header", "Authorization"),
            env_key=auth.get("env_key"),
            command=command,
        )

    def _build_upstream_headers(
        self,
        route: ModelRoute,
        client_headers: dict[str, str] | None = None,
        codec_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Build upstream headers using the auth module (item 92).

        Merges codec-required headers with route auth headers.
        """
        from agent_interop.auth import build_upstream_headers

        auth_config = self._build_upstream_auth_config(route)
        headers = build_upstream_headers(
            client_headers or {},
            auth_config,
            route.upstream.static_headers,
        )
        # Codec-required headers (Content-Type, anthropic-version, etc.)
        if codec_headers:
            headers.update(codec_headers)
        return headers
