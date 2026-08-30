"""Model projection aggregation (P1.6).

Projection owns:
  - system projection (deterministic, version-aware)
  - history paging (bounded via ContextStore refs)
  - tool-surface + private-capability selection
  - withheld-tool index (schema-on-demand discovery)
  - ModelView telemetry construction

The gateway only orchestrates ordering (adapt before plan, project after plan)
and owns pinning/unpinning of referenced_refs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from agent_interop.abi import CanonicalRequest, CanonicalTextBlock
from agent_interop.context_budget.compaction import (
    ContextAdaptationResult,
    compact_safe_tool_results,
)
from agent_interop.context_budget.model_view import ModelView
from agent_interop.projection.system import SystemProjectionResult, project_system
from agent_interop.projection.types import PrivateCapabilityPlan

# ─── Module-level tool-surface builder (migrated from Gateway._build_internal_) ──


def build_internal_tool_surface(base_tools, capabilities=None):
    """Append private Interop retrieval tools to the model-visible surface.

    Migrated from ``Gateway._build_internal_tool_surface`` to break the
    backwards dependency of projection on gateway.  The filtering semantics
    are identical: when ``capabilities`` is None, all internal + schema
    tools are included (deduplicated by existing names); when
    ``capabilities`` is provided, only the tools corresponding to the
    active capabilities are added.
    """
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


# ─── Projection result with transformations tracking ────────────────────────


def _was_surface_narrowed(tool_surface_plan: Any, authorized_tool_count: int) -> bool:
    """P1.10 (review #25): did the projection actually narrow the surface?

    A surface is narrowed only when its visible tool count is strictly less
    than the declared authoritative count. Tests sometimes pass stubbed
    surfaces without ``visible_tools``; in that case we conservatively treat
    the surface as not narrowed (the note would otherwise leak into a
    surface whose narrowing state we cannot verify).
    """
    if tool_surface_plan is None:
        return False
    visible = getattr(tool_surface_plan, "visible_tools", None)
    if visible is None:
        return False
    try:
        return len(visible) < authorized_tool_count
    except TypeError:
        return False


@dataclass
class ProjectionResult:
    """Single object describing the actual request that reaches the model.

    P1.6: every transformation (system projection, history paging, result
    virtualization, tool-surface narrowing, private-capability selection)
    is reflected here so the gateway's final pre-transport check can
    consult one authoritative structure.

    ``transformations`` is a tuple of tags aggregating virtualization /
    history-paging / withheld-tool-index events so downstream telemetry
    can observe the full pipeline without re-deriving facts.
    """

    request: Any  # CanonicalRequest — final model-facing request
    model_view: ModelView
    tool_surface: Any = None  # ToolSurfacePlan
    private_capabilities: Any = None  # PrivateCapabilityPlan
    referenced_refs: tuple[str, ...] = ()
    context_plan: Any = None  # ContextPlan
    system_projection: SystemProjectionResult | None = None
    authoritative_request: Any = None  # CanonicalRequest — immutable original
    transformations: tuple[str, ...] = ()


# ─── ModelProjector ─────────────────────────────────────────────────────────


class ModelProjector:
    """Aggregate the per-request projection into one ``ProjectionResult``."""

    @staticmethod
    def build_model_view(
        *,
        authorized_tool_count: int = 0,
        visible_client_tool_names: tuple[str, ...] = (),
        visible_private_tool_names: tuple[str, ...] = (),
        route: Any = None,
        context_plan: Any = None,
        tool_surface_plan: Any = None,
        results_virtualized: bool = False,
        history_paged: bool = False,
        system_projected: bool = False,
        effective_context_limit: int = 0,
        safe_context_limit: int = 0,
        virtualized_refs_count: int = 0,
        history_ref_count: int = 0,
    ) -> ModelView:
        """Construct a ``ModelView`` that describes the ACTUAL final request.

        P0.40: the view must describe what was sent, not an approximation.
        ``authorized_tool_count`` is the full authoritative registry size
        (used for validation); ``visible_*_tool_names`` describe the narrowed
        client surface and the request-scoped private tools respectively.
        """
        max_ctx = effective_context_limit or (route.upstream.ollama_num_ctx if route else 0)
        return ModelView(
            authorized_tool_count=authorized_tool_count,
            visible_tool_count=len(visible_client_tool_names),
            visible_tool_names=visible_client_tool_names,
            visible_client_tool_names=visible_client_tool_names,
            visible_private_tool_names=visible_private_tool_names,
            system_projected=system_projected,
            results_virtualized=results_virtualized,
            history_paged=history_paged,
            max_context_tokens=max_ctx,
            virtualized_refs_count=virtualized_refs_count,
            history_ref_count=history_ref_count,
            effective_context_limit=effective_context_limit,
            safe_context_limit=safe_context_limit or max_ctx,
            # P1.10 (review #25): the note reflects whether the surface
            # was actually narrowed — visible set smaller than the
            # declared surface → narrowed; otherwise unchanged surfaces
            # stay silent. The historical code added the note on every
            # ``tool_surface_plan is not None`` which falsely flagged
            # unchanged surfaces as narrowed.
            notes=(
                ("tool_surface_narrowed",)
                if _was_surface_narrowed(tool_surface_plan, authorized_tool_count)
                else ()
            ),
        )

    @staticmethod
    def project_system(system: list[Any], client_id: str = "") -> SystemProjectionResult:
        """Wrap ``project_system`` so callers route through the projector."""
        return project_system(system, client_id=client_id)

    @staticmethod
    def adapt_context(
        request: CanonicalRequest,
        *,
        plan: Any = None,
        store: Any = None,
        session_id: str = "",
        exchanges: tuple[Any, ...] | list[Any] | None = None,
    ) -> ContextAdaptationResult:
        """SINGLE wrapper for pre-plan context adaptation.

        When ``store`` is not None, ``session_id`` is non-empty, and the
        plan permits result virtualization AND compaction is required,
        delegates to ``compact_safe_tool_results``.

        Otherwise returns a NO-OP ``ContextAdaptationResult(request)``
        unchanged.  When virtualization is not permitted the adaptation
        is a NO-OP — legacy lossy compaction must never run in the
        projection path.
        """
        if store is None or not session_id:
            return ContextAdaptationResult(request)
        if plan is None:
            return ContextAdaptationResult(request)
        if not getattr(plan, "allow_result_virtualization", False):
            # Virtualization not permitted — do NOT fall through to legacy
            # lossy compaction. The caller must handle capacity differently.
            return ContextAdaptationResult(request)
        if not getattr(plan, "compaction_required", False):
            return ContextAdaptationResult(request)

        if exchanges is None:
            exchanges = ()
        adaptation = compact_safe_tool_results(
            request,
            exchanges=exchanges,
            plan=plan,
            store=store,
            session_id=session_id,
        )
        return adaptation

    @staticmethod
    def project(
        authoritative_request: CanonicalRequest,
        route: Any,
        runtime_capabilities: Any,
        compatibility_plan: Any,
        policy: Any,
        context_store: Any = None,
        session_context: Any = None,
        *,
        perform_history_paging: bool = False,
        adaptation: ContextAdaptationResult | None = None,
        invocation_plan: Any = None,
        projected_request: CanonicalRequest | None = None,
        seed_refs: tuple[str, ...] = (),
    ) -> ProjectionResult:
        """Own the full projection pipeline in deterministic order.

        Sequence:
            1.  result virtualization  (adapt_context — stored refs)
            2.  history paging         (ContextStore refs for old turns)
            3.  system projection      (deterministic, version-aware)
            4.  client tool selection  (mode-negotiated via invocation_plan)
            5.  private capability selection
            6.  withheld-tool index    (schema-on-demand)
            7.  final model-visible tools
            8.  ModelView construction

        The gateway should not separately rediscover or alter any of those
        facts later.

        ``projected_request`` seeds step 1 with an ALREADY-projected request
        (e.g. one carrying virtualized results, paged history, or a controller
        summary from the gateway's adaptation pipeline). Omitting it starts
        from the authoritative request — the two must never be silently
        conflated, because transformation work paid for upstream would be
        discarded and downstream accounting would price a request that is
        never rendered.

        ``seed_refs`` carries the ContextStore refs already present in that
        pre-projected request (result virtualization / history paging done by
        the gateway). They are merged into ``referenced_refs`` alongside refs
        created inside this pipeline so nothing the model can see goes
        unpinned or unfirewalled.
        """
        # P0-28 (review): the historical whole-request deepcopy duplicated
        # the entire canonical graph (every message, block, and tool schema)
        # on every projection pass — O(history) memory churn on large
        # conversations.  Every transformation below now builds a NEW
        # request via ``replace(...)`` with freshly-constructed lists, so a
        # shallow copy carries the same safety at O(1).  Codecs and the
        # projector never mutate canonical inputs.
        projected_request = replace(projected_request or authoritative_request)
        context_plan = compatibility_plan.context_plan
        tool_surface_plan = compatibility_plan.tool_surface_plan
        session_id = getattr(session_context, "session_id", "") or ""
        exchanges = ()

        transformations_list: list[str] = []

        # ── 1. Result virtualization via adapt_context ──────────────────
        # Ref identities are tracked per source: refs the gateway's own
        # adaptation already stored (seed), result virtualization refs, and
        # history paging refs. Every one of them is content the model can
        # observe a placeholder for, so every one must be pinned and
        # firewalled for the request's lifetime.
        seed_refs = tuple(seed_refs or ())
        result_refs: tuple[str, ...] = ()
        history_refs: tuple[str, ...] = ()
        referenced_refs: tuple[str, ...] = seed_refs
        virtualized_refs_count = 0

        if adaptation is None:
            adaptation = ModelProjector.adapt_context(
                projected_request,
                plan=context_plan,
                store=context_store,
                session_id=session_id,
                exchanges=exchanges,
            )

        if adaptation.changed:
            projected_request = adaptation.request
            # P0-7: only ContextStore refs count here. Tool-call IDs are
            # semantic correlation IDs, NOT store handles — pinning them
            # would be a no-op at best and a leak signal at worst (the
            # output firewall treats refs as unguessable tokens).
            result_refs = tuple(getattr(adaptation, "stored_refs", ()) or ())
            virtualized_refs_count = len(result_refs)
            referenced_refs = tuple(dict.fromkeys((*referenced_refs, *result_refs)))
            transformations_list.append("virtualize_tool_results")

        # ── 2. History paging ───────────────────────────────────────────
        history_paged = False
        history_ref_count = 0
        history_capabilities_requested = False

        if (
            perform_history_paging
            and context_store is not None
            and session_id
        ):
            from agent_interop.history.projector import (
                build_history_index_prompt,
                project_history,
            )

            history_result = project_history(
                list(projected_request.messages),
                store=context_store,
                session_id=session_id,
                max_recent_turns=6,
            )
            if history_result.refs:
                projected_request = replace(
                    projected_request, messages=history_result.messages
                )
                # Append index prompt as a CanonicalTextBlock in system blocks
                index_prompt = build_history_index_prompt(history_result.refs)
                if index_prompt:
                    # Build projected system with the index block appended
                    projected_system = list(projected_request.system) + [
                        CanonicalTextBlock(text=index_prompt)
                    ]
                    projected_request = replace(
                        projected_request, system=projected_system
                    )
                    history_paged = True
                    transformations_list.append("history_paged")

                # History refs exist => history retrieval capabilities are
                # enabled. P0-8: ProjectionResult.private_capabilities is the
                # ONLY output channel — the previous attempt to mutate the
                # frozen CompatibilityPlan silently swallowed an exception
                # on every call and left no trace for downstream consumers.
                history_capabilities_requested = True

                # P0: history pages are model-visible refs. Accumulate them
                # separately from result-virtualization refs so the final
                # referenced_refs covers every store handle the projected
                # request exposes.
                history_refs = tuple(history_result.stored_refs)
                referenced_refs = tuple(dict.fromkeys((*referenced_refs, *history_refs)))

                # Count refs
                ref_count = len(
                    getattr(history_result, "refs", history_result.refs)
                )
                history_ref_count = ref_count
                history_paged = True

        # ── 3. System projection ────────────────────────────────────────
        client_id = getattr(session_context, "client_id", "") or ""
        system_projection = project_system(
            list(projected_request.system), client_id=client_id
        )
        if system_projection.transformations:
            projected_request = replace(
                projected_request, system=system_projection.projected
            )
            transformations_list.append("system_projected")

        # ── 4. Client tool selection (from compatibility plan) ──────────
        # P0-5: the InvocationPlan's upstream_tools is the MODE-NEGOTIATED
        # surface (empty under PROMPTED/DISABLED, where tools move into the
        # prompt or nowhere). The raw ToolSurfacePlan.visible_tools is only
        # the pre-negotiation lexical narrowing — using it here would leak
        # schema tools into a prompted contract that must not carry them.
        plan_upstream_tools = getattr(invocation_plan, "upstream_tools", None)
        if plan_upstream_tools is not None:
            visible_tools = tuple(plan_upstream_tools)
        else:
            visible_tools = tool_surface_plan.visible_tools
        validation_tools = tool_surface_plan.validation_tools

        # ── 5. Private capability selection ─────────────────────────────
        # P0-23: enabling get_tool_schema implies read_result — a withheld
        # tool's schema can itself exceed the inline limit and page through
        # the store, so the model needs result paging to consume it.
        private_caps = PrivateCapabilityPlan(
            read_result=bool(virtualized_refs_count) or history_paged,
            recall_history=history_capabilities_requested,
            search_history=history_capabilities_requested,
            get_tool_schema=bool(
                getattr(tool_surface_plan, "withheld_tool_names", None)
            ),
        )
        if private_caps.get_tool_schema:
            private_caps = replace(private_caps, read_result=True)

        # ── 6. Build final model-visible tools ──────────────────────────
        base_tools = list(visible_tools) if visible_tools else []
        if private_caps.has_any:
            final_tools = build_internal_tool_surface(
                base_tools, capabilities=private_caps
            )
        else:
            final_tools = base_tools

        # ── 7. Withheld-tool index (schema-on-demand) ───────────────────
        if private_caps.get_tool_schema:
            withheld = getattr(
                tool_surface_plan, "withheld_tool_names", ()
            )
            if withheld:
                # Build from validation_tools registry (names + first
                # sentence of description) for a compact index block
                index_lines = [
                    (
                        "Additional authorized tools available via "
                        "__interop_get_tool_schema:"
                    )
                ]
                for tool in (validation_tools or []):
                    name = getattr(tool, "name", "")
                    if name in withheld:
                        desc = getattr(tool, "description", "") or ""
                        # First sentence only
                        first_sentence = desc.split(".")[0] if desc else desc
                        index_lines.append(f"- {name} — {first_sentence}")
                if index_lines:
                    index_block = CanonicalTextBlock(
                        text="\n".join(index_lines)
                    )
                    projected_request = replace(
                        projected_request,
                        system=list(projected_request.system)
                        + [index_block],
                    )
                    transformations_list.append("withheld_tool_index")

        # ── 8. Build model_request ──────────────────────────────────────
        model_request = replace(projected_request, tools=final_tools)

        # ── 9. Build ModelView ──────────────────────────────────────────
        visible_client_names = tuple(
            t.name for t in final_tools if not t.name.startswith("__interop_")
        )
        visible_private_names = tuple(
            t.name for t in final_tools if t.name.startswith("__interop_")
        )

        # History ref count: use the ref count computed during paging
        # (already set above). If still zero, fall back to getattr.
        if history_paged and history_ref_count == 0:
            try:
                session_refs = getattr(
                    context_store, "session_refs", None
                )
                if session_refs is not None:
                    history_ref_count = len(session_refs(session_id))
            except Exception:
                pass  # ContextStore may not have session_refs yet

        model_view = ModelProjector.build_model_view(
            authorized_tool_count=len(authoritative_request.tools),
            visible_client_tool_names=visible_client_names,
            visible_private_tool_names=visible_private_names,
            route=route,
            context_plan=context_plan,
            tool_surface_plan=tool_surface_plan,
            results_virtualized=bool(virtualized_refs_count),
            history_paged=history_paged,
            system_projected=bool(system_projection.transformations),
            effective_context_limit=context_plan.runtime_limit_tokens,
            safe_context_limit=context_plan.safe_limit_tokens,
            virtualized_refs_count=virtualized_refs_count,
            history_ref_count=history_ref_count,
        )

        return ProjectionResult(
            request=model_request,
            model_view=model_view,
            tool_surface=tool_surface_plan,
            private_capabilities=private_caps,
            referenced_refs=referenced_refs,
            context_plan=context_plan,
            system_projection=system_projection,
            authoritative_request=authoritative_request,
            transformations=tuple(transformations_list),
        )


def rebuild_invocation_atomic(
    invocation: Any,
    authoritative_request: Any,
    *,
    plan: Any,
    tool_surface_plan: Any | None = None,
    private_capabilities: Any | None = None,
) -> Any:
    """Review #21: ATOMIC invocation rebuild from one authoritative request.

    Every field derived from ``reconciled_request`` is regenerated TOGETHER
    so a rebuild can never produce an invocation whose plan mentions tools
    the model-visible surface lacks (or vice versa).  Callers that today
    hand-roll partial ``replace(invocation, ...)`` updates must use this:
    a partial update is exactly how a controller invocation ended up with
    a delegate tool in its plan but not in ``model_request``, so the tool
    never reached the wire.

    Regenerated here:
      - model_request          (authoritative tools + system projection)
      - private_capabilities   (explicit arg, default preserved/derived)
      - model_visible_tools    (from the rebuilt surface)
      - tool_surface_plan      (explicit arg when the surface changes)
    NOT regenerated (route/runtime-level, unaffected by message edits):
      invocation_plan (caller supplies), compatibility_key (attempt-level,
      rebuilt by _invocation_for_attempt), codec, evidence, budgets.
    """
    from dataclasses import replace as _replace

    capabilities = private_capabilities
    if capabilities is None:
        capabilities = invocation.private_capabilities
    final_tools = build_internal_tool_surface(
        list(authoritative_request.tools), capabilities=capabilities,
    )
    model_request = _replace(authoritative_request, tools=list(final_tools))

    client_id = getattr(invocation.request_context, "client_id", "") or ""
    system_projection = project_system(list(model_request.system), client_id=client_id)
    if system_projection.transformations:
        model_request = _replace(model_request, system=system_projection.projected)

    surface = tool_surface_plan
    if surface is None:
        surface = invocation.tool_surface_plan
    # Refs live in the request's ref registry once the request lifecycle
    # begins; before that (mid-preparation rebuilds) they are still on
    # the invocation. Either source describes the same model-visible set.
    registry = getattr(invocation.execution_record, "ref_registry", None)
    virtualized_refs = (
        tuple(sorted(registry.snapshot()))
        if registry is not None else tuple(invocation.pinned_refs)
    )
    model_view = ModelProjector.build_model_view(
        authorized_tool_count=len(authoritative_request.tools),
        visible_client_tool_names=tuple(
            t.name for t in final_tools if not t.name.startswith("__interop_")
        ),
        visible_private_tool_names=tuple(
            t.name for t in final_tools if t.name.startswith("__interop_")
        ),
        route=invocation.route,
        tool_surface_plan=surface,
        context_plan=invocation.context_plan,
        results_virtualized=bool(virtualized_refs),
        system_projected=bool(system_projection.transformations),
        effective_context_limit=getattr(invocation.context_plan, "runtime_limit_tokens", 0)
        if invocation.context_plan is not None else 0,
        safe_context_limit=getattr(invocation.context_plan, "safe_limit_tokens", 0)
        if invocation.context_plan is not None else 0,
        virtualized_refs_count=len(virtualized_refs),
    )
    return _replace(
        invocation,
        original_request=authoritative_request,
        reconciled_request=authoritative_request,
        authoritative_request=authoritative_request,
        invocation_plan=plan,
        tool_surface_plan=surface,
        model_request=model_request,
        model_view=model_view,
        private_capabilities=capabilities,
        model_visible_tools=tuple(final_tools),
    )
