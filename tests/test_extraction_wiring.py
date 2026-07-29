"""Tests for textual tool-call extraction wiring into the Gateway request path.

These confirm that ``Gateway._extract_tool_candidates`` actually invokes the
``ExtractorRegistry`` for prompted/local models, that native and textual
candidates are merged with dedup, that the envelope text is consumed (never
leaks to the client), and that the qwen-coder-ollama profile selects the
``tool_call_envelope`` extractor.
"""
from __future__ import annotations

import json

from agent_interop.abi import (
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
)
from agent_interop.config import (
    InteropServerConfig,
    ModelRoute,
    RepairPolicy,
    ToolMode,
    TranslationMode,
    UpstreamConfig,
    UpstreamKind,
    UpstreamProtocol,
)
from agent_interop.execution import InteropRequestExecution
from agent_interop.extraction import get_default_registry
from agent_interop.gateway import Gateway, ResolvedInvocation
from agent_interop.model.registry import ModelProfileRegistry
from agent_interop.repair.invocation import build_invocation_plan
from agent_interop.upstreams.codec import DecodedModelResponse

TEST_TOOL = CanonicalTool(
    name="test_tool",
    description="A test tool",
    input_schema={
        "type": "object",
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    },
)


def _make_gateway() -> Gateway:
    config = InteropServerConfig(
        host="127.0.0.1",
        port=0,
        log_level="error",
        probe_on_startup=False,
        routes={
            "qwen": ModelRoute(
                id="qwen",
                client_model_aliases=["qwen2.5-coder"],
                upstream_model="qwen2.5-coder",
                upstream=UpstreamConfig(
                    kind=UpstreamKind.OLLAMA,
                    base_url="http://127.0.0.1:0",
                    wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    timeout_seconds=30.0,
                ),
                tool_mode=ToolMode.AUTO,
                translation_mode=TranslationMode.CANONICAL,
            ),
        },
    )
    return Gateway(config=config)


def _resolve_qwen_profile() -> object:
    gw = _make_gateway()
    route = next(iter(gw.config.routes.values()))
    return ModelProfileRegistry().resolve(
        model_name=route.upstream_model,
        backend=route.upstream.kind,
        explicit_profile_id=route.profile if route.profile != "auto" else None,
    )


def _prompted_invocation(profile: object, tools: list[CanonicalTool] | None = None):
    gw = _make_gateway()
    route = next(iter(gw.config.routes.values()))
    plan = build_invocation_plan(
        tools=tools or [TEST_TOOL],
        tool_choice=CanonicalToolChoice.auto(),
        route_mode=route.tool_mode,
        model_profile=profile,
        repair_policy=RepairPolicy.from_config(route.repair),
    )
    return gw, ResolvedInvocation(
        request_context=None,
        original_request=None,
        reconciled_request=None,
        route=route,
        backend_metadata=None,
        model_profile=profile,
        repair_policy=None,
        invocation_plan=plan,
        codec=None,
        compatibility_key=None,
        evidence_record=None,
        repair_budget=None,
        execution_record=InteropRequestExecution(),
    )


def _extraction_invocation() -> tuple[Gateway, ResolvedInvocation]:
    """Invocation whose plan uses the tool_call_envelope parser with test_tool
    declared, so textual extraction can recognize the envelope."""
    profile = _resolve_qwen_profile()
    return _prompted_invocation(profile)


# ─── Incomplete / truncated envelope rejection ─────────────────────────────


class TestIncompleteEnvelopeRejection:
    def test_truncated_envelope_not_executed_with_diagnostic(self):
        """A truncated <tool_call> envelope must not produce a candidate and
        must emit an error diagnostic (raw_arguments are incomplete)."""
        registry = get_default_registry()
        content = [CanonicalTextBlock(text='<tool_call>{"name":"test_tool", "argum')]
        result = registry.extract(
            content,
            extractor_id="tool_call_envelope",
            tools=[TEST_TOOL],
            envelope="tool_call",
        )
        assert len(result.candidates) == 0, (
            f"truncated envelope should not produce candidates, got {result.candidates}"
        )
        assert any(d.level == "error" for d in result.diagnostics), (
            f"expected an error diagnostic, got {result.diagnostics}"
        )


# ─── Fenced-code masking ───────────────────────────────────────────────────


class TestFencedCodeNonExecution:
    def test_fenced_tool_call_example_not_executed(self):
        """A <tool_call> envelope inside a fenced code block must not be
        extracted (it is a literal example, not a real call)."""
        registry = get_default_registry()
        content = [
            CanonicalTextBlock(
                text='Example:\n```text\n<tool_call>{"name":"test_tool","arguments":{}}\n```\n'
            )
        ]
        result = registry.extract(
            content,
            extractor_id="tool_call_envelope",
            tools=[TEST_TOOL],
            envelope="tool_call",
        )
        assert len(result.candidates) == 0, (
            f"fenced example should not produce candidates, got {result.candidates}"
        )


# ─── Native + textual merge / dedup ────────────────────────────────────────


class TestNativeTextualDedup:
    def test_native_and_textual_duplicate_collapses(self):
        """When a native candidate is already present and an identical textual
        echo appears in content, only one candidate should survive.

        Textual extraction now runs even when a native candidate exists, so
        the echo IS produced as a textual candidate — but it is an exact shadow
        duplicate of the native one, so ``_dedup_tool_candidates`` collapses the
        pair to a single candidate. The result is still one candidate.
        """
        gw, invocation = _extraction_invocation()
        decoded = DecodedModelResponse(
            tool_candidates=[
                # Native candidate from the codec
                type(
                    "RC",
                    (),
                    {
                        "id": "native_001",
                        "name": "test_tool",
                        "raw_arguments": json.dumps({"key": "value"}),
                        "source_protocol": "openai_chat",
                        "source_index": 0,
                        "choice_index": 0,
                        "tool_index": 0,
                    },
                )(),
            ],
            content=[
                CanonicalTextBlock(
                    text='<tool_call>{"name":"test_tool","arguments":{"key":"value"}}</tool_call>'
                )
            ],
        )
        candidates = gw._extract_tool_candidates(decoded, invocation)
        assert len(candidates) == 1, f"expected dedup to one, got {len(candidates)}: {candidates}"
        assert candidates[0].name == "test_tool"

    def test_distinct_parallel_calls_remain_distinct(self):
        """Two different <tool_call> blocks with different arguments must both
        survive extraction (different tool_index keeps them distinct)."""
        gw, invocation = _extraction_invocation()
        decoded = DecodedModelResponse(
            content=[
                CanonicalTextBlock(
                    text=(
                        '<tool_call>{"name":"test_tool","arguments":{"key":"value"}}</tool_call>'
                        '<tool_call>{"name":"test_tool","arguments":{"key":"other"}}</tool_call>'
                    )
                )
            ],
        )
        candidates = gw._extract_tool_candidates(decoded, invocation)
        assert len(candidates) == 2, f"expected 2 distinct candidates, got {len(candidates)}: {candidates}"
        # Each candidate carries the arguments value under the "key" field.
        # The dedup key includes normalized_args, so differing args keep both.
        keys = {
            (json.loads(c.raw_arguments) if isinstance(c.raw_arguments, str) else c.raw_arguments).get("key")
            for c in candidates
        }
        assert keys == {"value", "other"}, f"unexpected arg keys: {keys}"

    def test_dedup_helper_exact_duplicate_collapses(self):
        """Direct unit check on the merge helper: identical name and
        normalized arguments, with no ID on either side, collapses to
        one — the content-signature echo-suppression fallback."""
        gw = _make_gateway()
        a = type("RC", (), {
            "id": None, "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        # Same identity but arguments serialized differently (whitespace).
        b = type("RC", (), {
            "id": None, "name": "test_tool", "raw_arguments": '{"key":"value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        merged = gw._dedup_tool_candidates([a], [b])
        assert len(merged) == 1, f"expected collapse to 1, got {len(merged)}"

    def test_dedup_helper_index_alone_no_longer_distinguishes_calls(self):
        """Re-audit P1#13: choice_index/tool_index are deliberately NOT
        part of the dedup signature anymore. Pre-structured candidates
        (e.g. whole-message JSON) are constructed with both forced to 0
        regardless of their real position, so index equality/inequality
        was neither a reliable duplicate signal nor a reliable distinctness
        signal — two candidates with the same name+arguments and no ID now
        collapse to one EVEN IF their indexes differ, since index carries
        no trustworthy information here. Genuinely distinct parallel calls
        must be distinguished by differing arguments or provider IDs
        instead (see the tests below)."""
        gw = _make_gateway()
        first = type("RC", (), {
            "id": None, "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        second = type("RC", (), {
            "id": None, "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 1,
        })()
        merged = gw._dedup_tool_candidates([first], [second])
        assert len(merged) == 1

    def test_dedup_helper_distinct_arguments_preserved_regardless_of_index(self):
        gw = _make_gateway()
        first = type("RC", (), {
            "id": None, "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        second = type("RC", (), {
            "id": None, "name": "test_tool", "raw_arguments": '{"key": "other"}',
            "choice_index": 0, "tool_index": 0,
        })()
        merged = gw._dedup_tool_candidates([first], [second])
        assert len(merged) == 2

    def test_dedup_helper_distinct_provider_ids_never_collapse_even_if_identical_content(self):
        """Re-audit P1#13: two candidates with the SAME name/arguments but
        DIFFERENT non-empty provider IDs must never be merged by content
        alone — that would collapse genuinely distinct parallel calls
        (e.g. two independent 'read_file(\"/tmp/x\")' calls) that just
        happen to share identical arguments."""
        gw = _make_gateway()
        first = type("RC", (), {
            "id": "call_1", "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        second = type("RC", (), {
            "id": "call_2", "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        merged = gw._dedup_tool_candidates([first], [second])
        assert len(merged) == 2

    def test_dedup_helper_matching_provider_ids_collapse_even_with_different_index(self):
        """The strongest duplicate signal: identical non-empty provider
        IDs settle the question regardless of index."""
        gw = _make_gateway()
        first = type("RC", (), {
            "id": "call_1", "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 0, "tool_index": 0,
        })()
        second = type("RC", (), {
            "id": "call_1", "name": "test_tool", "raw_arguments": '{"key": "value"}',
            "choice_index": 1, "tool_index": 3,
        })()
        merged = gw._dedup_tool_candidates([first], [second])
        assert len(merged) == 1


# ─── Envelope consumption (no leak) ────────────────────────────────────────


class TestEnvelopeConsumed:
    def test_envelope_text_not_in_remaining_content(self):
        """After extraction, the <tool_call> envelope must be consumed so it
        never leaks into the client-visible response."""
        gw, invocation = _extraction_invocation()
        decoded = DecodedModelResponse(
            content=[
                CanonicalTextBlock(
                    text='before <tool_call>{"name":"test_tool","arguments":{"key":"value"}}</tool_call> after'
                )
            ],
        )
        gw._extract_tool_candidates(decoded, invocation)
        remaining_text = " ".join(getattr(b, "text", "") for b in decoded.content)
        assert "<tool_call>" not in remaining_text, f"envelope leaked: {remaining_text!r}"
        # Surrounding text is preserved
        assert "before" in remaining_text
        assert "after" in remaining_text


# ─── Profile regression ────────────────────────────────────────────────────


class TestQwenProfileParser:
    def test_qwen_profile_uses_tool_call_envelope_parser(self):
        """Regression: the shipped qwen-coder-ollama profile must select the
        tool_call_envelope parser (matches <tool_call>...</tool_call>), not the
        qwen parser (which matches <tool>...</tool>)."""
        profile = _resolve_qwen_profile()
        assert getattr(profile, "parser_id", None) == "tool_call_envelope", (
            f"qwen-coder-ollama parser_id should be 'tool_call_envelope', "
            f"got {getattr(profile, 'parser_id', None)!r}"
        )

    def test_qwen_profile_builds_prompted_plan_with_envelope_parser(self):
        """End-to-end: a prompted plan built from the qwen profile resolves to
        the tool_call_envelope parser and tool_call envelope."""
        profile = _resolve_qwen_profile()
        _, invocation = _prompted_invocation(profile)
        plan = invocation.invocation_plan
        assert plan is not None
        assert plan.parser_id == "tool_call_envelope"
        assert plan.output_envelope == "tool_call"


class TestNativeStructuredExcludedFromGenericFallback:
    """Regression: a route whose primary extractor is the ``native_structured``
    passthrough must NOT fall through to the ``generic_balanced_json`` prose
    scanner, even when ``allow_generic_fallback=True``. The native passthrough
    returning zero candidates means "tool calls come from the codec, not from
    here" — not "the extractor failed, scan the prose for JSON"."""

    def test_native_route_with_prose_json_yields_no_textual_candidate(self):
        """The shipped ``gpt-4o-mini-compat`` profile resolves to ToolMode.NATIVE
        with ``parser_id='native_structured'`` and ``allow_generic_fallback=True``.
        A response whose prose contains a JSON object whose ``name`` matches a
        declared tool (here ``test_tool``) must NOT produce a bogus textual
        candidate from that prose."""
        registry = get_default_registry()
        prose = (
            'Here is an example config: {"name": "test_tool", "arguments": {"key": "example"}} '
            "- not a real call."
        )
        content = (CanonicalTextBlock(text=prose),)

        from agent_interop.model.profiles_v2 import ExtractionStrategy

        result = registry.extract(
            content,
            extractor_id="native_structured",
            tools=(TEST_TOOL,),
            envelope=None,
            fallback_strategies=[ExtractionStrategy(parser_id="generic_balanced_json")],
        )

        assert len(result.candidates) == 0, (
            f"native_structured must not fall through to generic JSON-in-prose "
            f"scanning; got {len(result.candidates)} spurious candidate(s): "
            f"{[(c.name, c.raw_arguments) for c in result.candidates]}"
        )
        # The prose is untouched — the native passthrough returns it verbatim.
        assert len(result.remaining_content) == 1
        assert isinstance(result.remaining_content[0], CanonicalTextBlock)
        assert result.remaining_content[0].text == prose
