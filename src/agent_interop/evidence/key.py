"""Authoritative compatibility key factory for the evidence store.

The CompatibilityKey identifies an exact client/model/backend/profile tuple
for empirical evidence lookup.  Every caller that stores or retrieves
evidence must use the same factory to build keys — hand-built keys with
sparse fields will never match.

Usage::

    inputs = CompatibilityKeyInputs(
        request_context=ctx,
        route=route,
        request=canonical_request,
        backend_metadata=backend_meta,
        model_profile=resolved_profile,
        invocation_plan=plan,
        tool_schema_fingerprint=fp,
        streaming=streaming,
    )
    key = build_compatibility_key(inputs)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_interop.replay.types import CompatibilityKey


def _strval(v: Any) -> str:
    """Safely coerce a value to string, handling None and enums."""
    if v is None:
        return ""
    if hasattr(v, "value"):
        return str(v.value)
    return str(v)


@dataclass(frozen=True)
class CompatibilityKeyInputs:
    """All inputs required to build an authoritative CompatibilityKey.

    Collecting all dimensions in one place prevents callers from
    accidentally omitting fields that the evidence store relies on
    for exact tuple matching.
    """

    request_context: Any = None
    route: Any = None
    request: Any = None
    backend_metadata: Any = None
    model_profile: Any = None
    invocation_plan: Any = None
    tool_schema_fingerprint: str = ""
    streaming: bool = False
    runtime_capabilities: Any = None
    compatibility_plan: Any = None
    context_plan: Any = None
    tool_surface_plan: Any = None
    selected_attempt: Any = None


def build_compatibility_key(
    inputs: CompatibilityKeyInputs,
) -> CompatibilityKey:
    """Build an authoritative CompatibilityKey from resolved inputs.

    All dimensions are extracted from the resolved runtime objects.
    Callers must resolve the invocation plan first so that
    ``effective_tool_mode`` and ``parser_id`` are available at key
    construction time.

    Args:
        inputs: Fully resolved CompatibilityKeyInputs.

    Returns:
        A frozen CompatibilityKey with all fields populated.
    """
    from agent_interop.replay.types import CompatibilityKey

    ctx = inputs.request_context
    route = inputs.route
    request = inputs.request
    backend_meta = inputs.backend_metadata
    profile = inputs.model_profile
    plan = inputs.invocation_plan

    # ── Client identity ────────────────────────────────────────────
    client_id = ""
    client_version = ""
    client_protocol = ""
    if ctx:
        client_id = ctx.client_id or ""
        client_version = ctx.client_version or ""
        client_protocol = (
            ctx.client_protocol.value if hasattr(ctx.client_protocol, "value")
            else _strval(ctx.client_protocol)
        )

    # ── Model identity ─────────────────────────────────────────────
    model_id = ""
    if route:
        model_id = route.upstream_model or ""
    if not model_id and request:
        model_id = request.model.requested_name or ""

    model_digest = ""
    quantization = ""
    if backend_meta is not None:
        model_digest = _strval(getattr(backend_meta, "model_digest", ""))
        quantization = _strval(getattr(backend_meta, "quantization", ""))

    # ── Backend identity ───────────────────────────────────────────
    backend_kind = ""
    backend_version = ""
    upstream_protocol = ""
    chat_template_digest = ""
    if backend_meta is not None:
        backend_version = _strval(getattr(backend_meta, "backend_version", ""))
        # Hash the raw chat template instead of storing it
        raw_template = _strval(getattr(backend_meta, "chat_template", ""))
        if raw_template:
            import hashlib
            chat_template_digest = hashlib.sha256(raw_template.encode()).hexdigest()[:16]
    if route:
        backend_kind = (
            route.upstream.kind.value if hasattr(route.upstream.kind, "value")
            else str(route.upstream.kind)
        )
        upstream_protocol = (
            route.upstream.wire_protocol.value
            if hasattr(route.upstream.wire_protocol, "value")
            else str(route.upstream.wire_protocol)
        )

    # ── Profile identity ───────────────────────────────────────────
    profile_id = ""
    profile_revision = ""
    if profile is not None:
        profile_id = (
            _strval(getattr(profile, "profile_id", ""))
            or _strval(getattr(profile, "id", ""))
            or ""
        )
        profile_revision = _strval(getattr(profile, "profile_revision", ""))

    # ── Plan dimensions ────────────────────────────────────────────
    effective_tool_mode = ""
    parser_id = ""
    template_revision = ""
    if plan is not None:
        resolved_tool_mode = getattr(plan, "effective_tool_mode", None)
        if resolved_tool_mode is not None:
            effective_tool_mode = (
                resolved_tool_mode.value if hasattr(resolved_tool_mode, "value")
                else str(resolved_tool_mode)
            )
        parser_id = _strval(getattr(plan, "parser_id", ""))
        template_revision = _strval(getattr(plan, "template_revision", ""))

    # Fallback parser from profile if plan doesn't carry it
    if not parser_id and profile is not None:
        parser_id = _strval(getattr(profile, "parser_id", ""))

    backend_serving_config = ""

    runtime = inputs.runtime_capabilities
    runtime_context_tokens = int(getattr(runtime, "effective_context_tokens", 0) or 0)
    runtime_capability_digest = ""
    if runtime is not None:
        import hashlib
        import json

        runtime_key_facts = {
            name: value for name, value in vars(runtime).items()
            # Observation time is telemetry, not a compatibility dimension.
            # Including it makes identical requests miss evidence every turn.
            if name != "probed_at"
        }
        runtime_capability_digest = hashlib.sha256(
            json.dumps(runtime_key_facts, sort_keys=True, default=_strval).encode()
        ).hexdigest()[:16]

    compatibility = inputs.compatibility_plan
    surface = inputs.tool_surface_plan
    context_plan = inputs.context_plan
    selected_attempt = inputs.selected_attempt
    path = _strval(getattr(compatibility, "path", ""))
    planner_revision = _strval(getattr(compatibility, "planner_revision", ""))
    attempt_kind = _strval(getattr(selected_attempt, "kind", ""))
    controller_model_id = _strval(getattr(getattr(route, "controller", None), "route_id", ""))
    streaming_policy = "direct"
    if path == "controlled":
        streaming_policy = "controller"
    elif inputs.streaming and (
        effective_tool_mode != "native"
        or bool(getattr(getattr(route, "compatibility", None), "buffer_unverified_streaming", False))
    ):
        streaming_policy = "buffered_validation"

    from agent_interop.build_info import get_build_info
    build = get_build_info()

    return CompatibilityKey(
        client_id=client_id,
        client_version=client_version,
        client_protocol=client_protocol,
        model_id=model_id,
        model_digest=model_digest,
        quantization=quantization,
        backend_kind=backend_kind,
        backend_version=backend_version,
        upstream_protocol=upstream_protocol,
        chat_template_digest=chat_template_digest,
        profile_id=profile_id,
        profile_revision=profile_revision,
        tool_schema_fingerprint=inputs.tool_schema_fingerprint,
        streaming=inputs.streaming,
        effective_tool_mode=effective_tool_mode,
        parser_id=parser_id,
        template_revision=template_revision,
        backend_serving_config=backend_serving_config,
        interop_build_commit=build.git_commit,
        interop_build_dirty=build.git_dirty,
        planner_revision=planner_revision or build.planner_revision,
        runtime_context_tokens=runtime_context_tokens,
        runtime_capability_digest=runtime_capability_digest,
        compatibility_path=path,
        attempt_kind=attempt_kind,
        controller_model_id=controller_model_id,
        tool_surface_mode=_strval(getattr(surface, "mode", "")),
        visible_tool_fingerprint=_strval(getattr(surface, "fingerprint", "")),
        tool_selector_revision=_strval(getattr(surface, "selector_revision", "")),
        context_strategy=_strval(getattr(context_plan, "selected_strategy", "")),
        context_plan_revision="1" if context_plan is not None else "",
        streaming_policy=streaming_policy,
    )
