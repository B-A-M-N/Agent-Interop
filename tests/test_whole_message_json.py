"""Tests for the whole_message_json fallback tier (extraction.py).

Closes a real gap found via a live-model benchmark: qwen2.5-coder:7b's
default habit is a bare or fenced JSON tool-call object with no envelope
tags at all — a shape none of the tag-based extractors (or the fence-
masked GenericBalancedJsonExtractor) can ever recover. This is a distinct,
profile-approved output DIALECT (WholeMessageJsonExtractor), not envelope-
defect repair (that's envelope_scan.py's job) — see the module's docstring
for the full rationale.

Three layers are tested:
  1. WholeMessageJsonExtractor.extract() directly — the structural safety
     bounds (whole-response-only, narrow key set, declared-tool match).
  2. ExtractorRegistry.extract() — the cross-cutting policy this
     extractor's fixed protocol signature can't enforce itself: profile
     opt-in, tool_choice gating, skip-when-native-present.
  3. A couple of full-gateway, non-streaming, fake-transport tests proving
     the wiring is correct end-to-end.
"""

from __future__ import annotations

import pytest

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalRefusalBlock,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolCallBlock,
    CanonicalToolChoice,
)
from agent_interop.extraction import ExtractorRegistry, WholeMessageJsonExtractor
from agent_interop.model.profiles_v2 import ExtractionStrategy

_WHOLE_MESSAGE_JSON_STRATEGY = [ExtractionStrategy(parser_id="whole_message_json")]

_TOOLS = [
    CanonicalTool(
        name="read_file",
        description="Read a file",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
    ),
]


def _extract(text: str, tools=_TOOLS):
    ext = WholeMessageJsonExtractor()
    return ext.extract([CanonicalTextBlock(text=text)], tools=tools, envelope=None)


# ─── Layer 1: extractor-level acceptance ───────────────────────────────────


class TestAcceptance:
    def test_bare_whole_message_json(self):
        r = _extract('{"name":"read_file","arguments":{"path":"/tmp/x"}}')
        assert len(r.candidates) == 1
        assert r.candidates[0].name == "read_file"
        assert r.candidates[0].raw_arguments == '{"path":"/tmp/x"}'
        assert r.remaining_content == ()

    def test_fenced_whole_message_json(self):
        r = _extract('```json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n```')
        assert len(r.candidates) == 1
        assert r.candidates[0].name == "read_file"

    def test_fenced_with_empty_language_tag(self):
        r = _extract('```\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n```')
        assert len(r.candidates) == 1

    def test_tilde_fence(self):
        r = _extract('~~~json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n~~~')
        assert len(r.candidates) == 1

    def test_mismatched_fence_delimiters_rejected(self):
        """Re-audit P2#15: the opening/closing fence must be the SAME
        delimiter — the regex previously used two independent (backtick|
        tilde) alternations instead of a captured-and-backreferenced one,
        so a message opening with backticks and closing with tildes (which
        markdown never treats as a single fenced block) was incorrectly
        accepted as one."""
        r = _extract('```json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n~~~')
        assert len(r.candidates) == 0

        r2 = _extract('~~~json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n```')
        assert len(r2.candidates) == 0

    def test_id_key_permitted(self):
        r = _extract('{"name":"read_file","arguments":{"path":"/tmp/x"},"id":"call_1"}')
        assert len(r.candidates) == 1

    def test_id_key_is_assigned_to_the_candidate(self):
        """Re-audit P1#13: 'id' was an accepted top-level key but was
        parsed and then silently discarded — the candidate's own .id
        stayed None regardless. Downstream dedup (Gateway.
        _dedup_tool_candidates) relies on candidate.id when present, so
        losing it here made the strongest duplicate signal unavailable
        for every whole-message-JSON call that carried one."""
        r = _extract('{"name":"read_file","arguments":{"path":"/tmp/x"},"id":"call_1"}')
        assert len(r.candidates) == 1
        assert r.candidates[0].id == "call_1"

    def test_missing_id_key_leaves_candidate_id_none(self):
        r = _extract('{"name":"read_file","arguments":{"path":"/tmp/x"}}')
        assert r.candidates[0].id is None

    def test_whitespace_around_message_tolerated(self):
        r = _extract('  \n  {"name":"read_file","arguments":{"path":"/tmp/x"}}  \n  ')
        assert len(r.candidates) == 1

    def test_raw_arguments_preserves_exact_source_text(self):
        """Raw argument evidence must survive verbatim for the repair
        layer, not a re-serialized/normalized copy."""
        r = _extract('{"name":"read_file","arguments":{"path":  "/tmp/x" , "extra":1}}')
        assert len(r.candidates) == 1
        assert r.candidates[0].raw_arguments == '{"path":  "/tmp/x" , "extra":1}'

    def test_malformed_arguments_json_still_recovered_for_repair(self):
        """Re-audit P2#15: the extractor previously required the ENTIRE
        outer object to parse as valid JSON — a whole-message call whose
        "arguments" value had the classic small-model failure (an
        embedded, unescaped double quote inside a string) failed to parse
        as a whole and was discarded outright, even though "name" was
        clean and unambiguous. It must now recover the call with the raw
        (malformed) argument text intact, so the repair pipeline gets a
        chance to fix it, instead of losing the call entirely."""
        tools = [
            CanonicalTool(
                name="edit_file", description="Edit a file",
                input_schema={"type": "object", "properties": {}},
            ),
        ]
        text = (
            '{"name":"edit_file","arguments":'
            '{"old_string":"he said "hi"","new_string":"she said "bye""}}'
        )
        r = _extract(text, tools=tools)
        assert len(r.candidates) == 1
        assert r.candidates[0].name == "edit_file"
        # Raw (still-malformed) text survives verbatim for repair to work on.
        assert r.candidates[0].raw_arguments == (
            '{"old_string":"he said "hi"","new_string":"she said "bye""}'
        )
        assert r.diagnostics
        assert r.diagnostics[0].level == "warning"

    def test_malformed_arguments_with_id_still_recovers_id(self):
        text = (
            '{"name":"read_file","arguments":{"path":"say "hi""},"id":"call_1"}'
        )
        r = _extract(text)
        assert len(r.candidates) == 1
        assert r.candidates[0].id == "call_1"

    def test_malformed_arguments_but_no_clean_name_not_recovered(self):
        """If even "name" fails to parse cleanly, there's nothing safe to
        recover — must fall through to rejection, not guess."""
        text = '{"name":123,"arguments":{"path":"say "hi""}}'
        assert _extract(text).candidates == ()

    def test_malformed_arguments_undeclared_tool_not_recovered(self):
        text = '{"name":"delete_everything","arguments":{"path":"say "hi""}}'
        assert _extract(text).candidates == ()

    def test_malformed_arguments_missing_arguments_key_not_recovered(self):
        text = '{"name":"read_file","note":"say "hi"" }'
        assert _extract(text).candidates == ()

    def test_malformed_with_extra_unrelated_key_not_recovered(self):
        """The bounded-recovery path enforces the same allowed-key set as
        the well-formed path — an unrelated top-level key must still
        reject the whole thing, malformed or not."""
        text = (
            '{"name":"read_file","arguments":{"path":"say "hi""},'
            '"note":"be careful"}'
        )
        assert _extract(text).candidates == ()


# ─── Layer 1: extractor-level rejection ────────────────────────────────────


class TestRejection:
    def test_prose_before_fence_rejected(self):
        text = 'Here is an example:\n```json\n{"name":"read_file","arguments":{}}\n```'
        assert _extract(text).candidates == ()

    def test_prose_after_fence_rejected(self):
        text = '```json\n{"name":"read_file","arguments":{}}\n```\nDone.'
        assert _extract(text).candidates == ()

    def test_prose_only_no_json_rejected(self):
        assert _extract("I would call it like this if asked.").candidates == ()

    def test_two_json_objects_rejected(self):
        text = '{"name":"read_file","arguments":{}} {"name":"read_file","arguments":{}}'
        assert _extract(text).candidates == ()

    def test_two_fenced_blocks_rejected(self):
        text = (
            '```json\n{"name":"read_file","arguments":{}}\n```\n'
            '```json\n{"name":"read_file","arguments":{}}\n```'
        )
        assert _extract(text).candidates == ()

    def test_unclosed_fence_rejected(self):
        text = '```json\n{"name":"read_file","arguments":{}}'
        assert _extract(text).candidates == ()

    def test_unknown_language_tag_rejected(self):
        text = '```python\n{"name":"read_file","arguments":{}}\n```'
        assert _extract(text).candidates == ()

    def test_json_array_rejected(self):
        assert _extract('[{"name":"read_file","arguments":{}}]').candidates == ()

    def test_missing_arguments_key_rejected(self):
        assert _extract('{"name":"read_file"}').candidates == ()

    def test_undeclared_tool_name_rejected(self):
        text = '{"name":"delete_everything","arguments":{}}'
        assert _extract(text).candidates == ()

    def test_extra_unrelated_top_level_key_rejected(self):
        text = '{"name":"read_file","arguments":{},"note":"be careful"}'
        assert _extract(text).candidates == ()

    def test_no_tools_declared_rejected(self):
        text = '{"name":"read_file","arguments":{}}'
        assert _extract(text, tools=[]).candidates == ()

    def test_name_not_a_string_rejected(self):
        assert _extract('{"name":123,"arguments":{}}').candidates == ()

    def test_empty_message_rejected(self):
        assert _extract("").candidates == ()
        assert _extract("   ").candidates == ()

    def test_second_content_block_present_rejected(self):
        ext = WholeMessageJsonExtractor()
        content: list[CanonicalContentBlock] = [
            CanonicalTextBlock(text='{"name":"read_file","arguments":{}}'),
            CanonicalRefusalBlock(refusal="I can't help with that."),
        ]
        r = ext.extract(content, tools=_TOOLS, envelope=None)
        assert r.candidates == ()

    def test_existing_tool_call_block_present_rejected(self):
        """A native tool-call block already in content means this isn't
        'the entire response is one bare JSON object' — reject."""
        ext = WholeMessageJsonExtractor()
        content: list[CanonicalContentBlock] = [
            CanonicalTextBlock(text='{"name":"read_file","arguments":{}}'),
            CanonicalToolCallBlock(id="x", name="other_tool", arguments={}),
        ]
        r = ext.extract(content, tools=_TOOLS, envelope=None)
        assert r.candidates == ()


# ─── Layer 2: ExtractorRegistry policy gating ──────────────────────────────


class TestRegistryPolicyGating:
    """The extractor itself has no notion of tool_choice or profile
    opt-in — ExtractorRegistry.extract() is where those cross-cutting
    decisions are enforced."""

    def _registry_result(self, text, **kwargs):
        registry = ExtractorRegistry()
        return registry.extract(
            [CanonicalTextBlock(text=text)],
            extractor_id="tool_call_envelope",  # primary finds nothing for bare JSON
            tools=_TOOLS,
            envelope="tool_call",
            **kwargs,
        )

    def test_disabled_by_default_even_when_shape_matches(self):
        """allow_whole_message_json defaults to False — a well-formed
        whole-message JSON call must NOT be recovered unless a profile
        explicitly opts in."""
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(text)
        assert result.candidates == ()

    def test_enabled_recovers_under_auto_choice_with_matching_nonce(self):
        """Auto-mode recovery additionally requires a live per-request
        execution nonce (P0-3 fix): the whole-response/narrow-key-set/
        declared-tool-name constraints alone were judged insufficient for
        the ambiguous auto case, since a bare/fenced JSON object can't be
        told apart from demonstration content on shape alone. With a
        matching nonce, recovery still proceeds — that's the common case
        for a real, opted-in coding-agent use."""
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"},"interop_call_id":"nonce-123"}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY, tool_choice=CanonicalToolChoice.auto(),
            expected_execution_nonce="nonce-123",
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].name == "read_file"

    def test_auto_choice_rejected_without_nonce(self):
        """The actual P0-3 safety boundary: no live nonce configured for
        this request at all (the ordinary case for a builtin profile,
        which can never enable this path — see profiles_v2.py) means an
        ambiguous auto-mode candidate is never trusted, regardless of
        shape."""
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY, tool_choice=CanonicalToolChoice.auto(),
        )
        assert result.candidates == ()

    def test_auto_choice_rejected_with_mismatched_nonce(self):
        """A nonce that doesn't match the live per-request value (e.g.
        copied from an earlier turn or the model's own habit) must not be
        trusted either — only an exact match counts."""
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"},"interop_call_id":"stale-nonce"}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY, tool_choice=CanonicalToolChoice.auto(),
            expected_execution_nonce="nonce-123",
        )
        assert result.candidates == ()

    def test_enabled_recovers_under_required_choice(self):
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY, tool_choice=CanonicalToolChoice.required(),
        )
        assert len(result.candidates) == 1

    def test_never_runs_under_none_choice(self):
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY, tool_choice=CanonicalToolChoice.none(),
        )
        assert result.candidates == ()

    def test_named_choice_matching_name_recovers(self):
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY,
            tool_choice=CanonicalToolChoice.named("read_file"),
        )
        assert len(result.candidates) == 1

    def test_named_choice_mismatch_does_not_substitute_a_different_tool(self):
        """A real, well-formed call was recovered — just not the specific
        tool the client asked for. Must not silently substitute it."""
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY,
            tool_choice=CanonicalToolChoice.named("some_other_tool"),
        )
        assert result.candidates == ()

    def test_skipped_when_native_candidates_present(self):
        """Native evidence outranks this weaker fallback — bare textual
        JSON alongside a native call is most likely an echo of it."""
        text = '{"name":"read_file","arguments":{"path":"/tmp/x"}}'
        result = self._registry_result(
            text, fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY, tool_choice=CanonicalToolChoice.auto(),
            native_candidates_present=True,
        )
        assert result.candidates == ()

    def test_skipped_when_primary_extractor_already_found_candidates(self):
        """Only a fallback of last resort — never overrides a real primary
        (tag-envelope) match."""
        registry = ExtractorRegistry()
        text = '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>'
        result = registry.extract(
            [CanonicalTextBlock(text=text)],
            extractor_id="tool_call_envelope",
            tools=_TOOLS,
            envelope="tool_call",
            fallback_strategies=_WHOLE_MESSAGE_JSON_STRATEGY,
            tool_choice=CanonicalToolChoice.auto(),
        )
        assert len(result.candidates) == 1
        assert result.candidates[0].source_protocol == "tool_call_envelope"


# ─── Regression: generic fallback must not overlap this tier ──────────────


class TestGenericFallbackDoesNotRecoverFencedContent:
    def test_generic_balanced_json_alone_does_not_recover_fenced_call(self):
        """The review's required regression: a generic_balanced_json
        fallback strategy must NOT execute a fenced example unless
        whole_message_json is ALSO explicitly listed — the two tiers are
        independently gated, ordered entries in the same list."""
        registry = ExtractorRegistry()
        text = '```json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n```'
        result = registry.extract(
            [CanonicalTextBlock(text=text)],
            extractor_id="qwen",  # a primary extractor that finds nothing here
            tools=_TOOLS,
            envelope="tool_call",
            fallback_strategies=[ExtractionStrategy(parser_id="generic_balanced_json")],
            tool_choice=CanonicalToolChoice.auto(),
        )
        assert result.candidates == ()

    def test_both_strategies_together_recovers(self):
        registry = ExtractorRegistry()
        text = '```json\n{"name":"read_file","arguments":{"path":"/tmp/x"},"interop_call_id":"nonce-abc"}\n```'
        result = registry.extract(
            [CanonicalTextBlock(text=text)],
            extractor_id="qwen",
            tools=_TOOLS,
            envelope="tool_call",
            fallback_strategies=[
                ExtractionStrategy(parser_id="generic_balanced_json"),
                ExtractionStrategy(parser_id="whole_message_json"),
            ],
            tool_choice=CanonicalToolChoice.auto(),
            expected_execution_nonce="nonce-abc",
        )
        assert len(result.candidates) == 1


# ─── Layer 3: full gateway, non-streaming, fake transport ─────────────────


class TestGatewayEndToEnd:
    """Proves the profile -> ResolvedModelProfile -> InvocationPlan ->
    ExtractorRegistry wiring is correct end-to-end, not just at the unit
    level. Both the non-streaming and buffered-streaming paths converge
    on Gateway._extract_tool_candidates, so this exercises the same code
    the streaming path uses."""

    def _config(self, *, whole_message_json: bool):
        import json

        from agent_interop.config import (
            InteropServerConfig,
            ModelRoute,
            RepairConfig,
            ToolMode,
            TranslationMode,
            UpstreamConfig,
            UpstreamKind,
            UpstreamProtocol,
        )
        from agent_interop.model.profiles_v2 import ModelProfile, ProfileIndex, ToolBehaviorProfile
        from agent_interop.model.registry import ModelProfileRegistry

        profile = ModelProfile(
            id="test-fenced-json",
            tool_behavior=ToolBehaviorProfile(
                presentation_mode="prompted",
                extractor_id="tool_call_envelope",
                output_envelope="tool_call",
                fallback_strategies=(
                    tuple(_WHOLE_MESSAGE_JSON_STRATEGY) if whole_message_json else ()
                ),
            ),
        )
        index = ProfileIndex()
        index.add_profile(profile, {"match": {"model_patterns": [".*"]}})
        registry = ModelProfileRegistry(profiles=index)

        config = InteropServerConfig(
            probe_on_startup=False,
            default_route_id="r",
            routes={
                "r": ModelRoute(
                    id="r",
                    client_model_aliases=["test-model"],
                    upstream_model="test-model",
                    upstream=UpstreamConfig(
                        kind=UpstreamKind.OLLAMA,
                        base_url="http://x",
                        wire_protocol=UpstreamProtocol.OLLAMA_CHAT,
                    ),
                    tool_mode=ToolMode.AUTO,
                    translation_mode=TranslationMode.CANONICAL,
                    repair=RepairConfig(),
                ),
            },
        )
        return config, registry, json

    async def _run(self, *, whole_message_json: bool, model_text: str):
        import re

        from agent_interop.abi import (
            CanonicalMessage,
            CanonicalModelReference,
            CanonicalRequest,
        )
        from agent_interop.context import RequestContext
        from agent_interop.gateway import Gateway
        from agent_interop.transport.http import UpstreamResponse, UpstreamTransport

        config, profile_registry, json = self._config(whole_message_json=whole_message_json)

        class FakeTransport(UpstreamTransport):
            async def send(self, request):
                # When the ambiguous-auto nonce guard is active, the live
                # nonce is embedded in the rendered prompt contract sent
                # upstream — recover it here so a test can echo it back in
                # model_text (via the __NONCE__ placeholder) the same way a
                # real model reading its own prompt would.
                def _iter_strings(obj):
                    if isinstance(obj, str):
                        yield obj
                    elif isinstance(obj, dict):
                        for v in obj.values():
                            yield from _iter_strings(v)
                    elif isinstance(obj, list):
                        for v in obj:
                            yield from _iter_strings(v)

                nonce = ""
                for s in _iter_strings(request.body):
                    m = re.search(r'"interop_call_id":\s*"([0-9a-f]+)"', s)
                    if m:
                        nonce = m.group(1)
                        break
                body = {
                    "model": "test-model",
                    "message": {"role": "assistant", "content": model_text.replace("__NONCE__", nonce)},
                    "done": True,
                    "done_reason": "stop",
                }
                return UpstreamResponse(status_code=200, headers={}, body=json.dumps(body).encode())

        gw = Gateway(config, transport=FakeTransport(), profile_registry=profile_registry)
        req = CanonicalRequest(
            model=CanonicalModelReference(requested_name="test-model"),
            messages=[CanonicalMessage(role="user", content=[CanonicalTextBlock(text="read it")])],
            tools=_TOOLS,
            tool_choice=CanonicalToolChoice.auto(),
        )
        resp = await gw.handle_request(req, RequestContext())
        await gw.close()
        return resp

    @pytest.mark.asyncio
    async def test_fenced_call_recovered_end_to_end_when_enabled(self):
        resp = await self._run(
            whole_message_json=True,
            model_text='```json\n{"name":"read_file","arguments":{"path":"/tmp/x"},'
                       '"interop_call_id":"__NONCE__"}\n```',
        )
        assert resp.error is None
        tool_calls = [b for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "read_file"
        assert tool_calls[0].arguments == {"path": "/tmp/x"}

    @pytest.mark.asyncio
    async def test_fenced_call_not_recovered_end_to_end_without_matching_nonce(self):
        """The actual end-to-end P0-3 safety proof: opted in, correct
        shape, but the response doesn't carry the live nonce the gateway
        embedded in its own prompt — must NOT become a tool call."""
        resp = await self._run(
            whole_message_json=True,
            model_text='```json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n```',
        )
        assert resp.error is None
        tool_calls = [b for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_fenced_call_not_recovered_end_to_end_when_disabled(self):
        """Same input, opt-in off — must remain plain text, not a tool call."""
        resp = await self._run(
            whole_message_json=False,
            model_text='```json\n{"name":"read_file","arguments":{"path":"/tmp/x"}}\n```',
        )
        assert resp.error is None
        tool_calls = [b for b in resp.content if isinstance(b, CanonicalToolCallBlock)]
        assert tool_calls == []
