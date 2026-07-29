"""Tests for the tag-envelope scanner (parsing/envelope_scan.py).

The scanner exists to close a real coverage gap: prompted-mode models
(the common case even for first-class profiles like qwen3-coder) are
taught an envelope format via prompt injection, and small/local models
frequently drift from the exact taught shape — dropped closing tags,
alternate tag spellings. Every test here either proves a specific
recoverable defect IS recovered, or proves the scanner's core safety
property: it never treats unanchored text as a candidate, so it cannot
turn a JSON code example or quoted config into an executed tool call.
Fenced code blocks are deliberately never unwrapped, even when a message
consists of nothing else — that is an existing, deliberate invariant
elsewhere in the extraction pipeline (see TestFencedCodeNeverRecovered).
"""

from __future__ import annotations

from agent_interop.extraction import _mask_fenced_code
from agent_interop.parsing.envelope_scan import recover_envelope, scan_envelopes


def _envs(text: str, canonical=("tool_call",), alias=("tool_calls", "toolcall")):
    return scan_envelopes(text, canonical_names=canonical, alias_names=alias)


class TestPairedMatching:
    def test_simple_paired_call(self):
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "x"}}</tool_call>'
        results = _envs(text)
        assert len(results) == 1
        assert results[0].rule_id == "paired"
        assert '"name": "read_file"' in results[0].payload

    def test_case_variant_tag(self):
        text = '<Tool_Call>{"name": "x"}</TOOL_CALL>'
        results = _envs(text)
        assert len(results) == 1
        assert results[0].rule_id == "paired"

    def test_multiple_independent_calls(self):
        text = (
            '<tool_call>{"name": "a"}</tool_call> some text '
            '<tool_call>{"name": "b"}</tool_call>'
        )
        results = _envs(text)
        assert len(results) == 2
        assert results[0].payload == '{"name": "a"}'
        assert results[1].payload == '{"name": "b"}'

    def test_prose_immediately_touching_tag(self):
        text = 'Sure, calling it now.<tool_call>{"name": "x"}</tool_call> done.'
        results = _envs(text)
        assert len(results) == 1


class TestAliasPairing:
    def test_alias_open_and_close(self):
        text = '<tool_calls>{"name": "x"}</tool_calls>'
        results = _envs(text)
        assert len(results) == 1
        assert results[0].rule_id == "tag_alias"

    def test_mixed_canonical_and_alias(self):
        text = '<tool_call>{"name": "x"}</toolcall>'
        results = _envs(text)
        assert len(results) == 1
        assert results[0].rule_id == "tag_alias"

    def test_unknown_tag_not_in_family_ignored(self):
        text = '<some_other_tag>{"name": "x"}</some_other_tag>'
        results = _envs(text)
        assert results == []


class TestDroppedCloseTag:
    def test_unclosed_but_balanced_json_recovered(self):
        text = 'Sure.<tool_call>{"name": "read_file", "arguments": {"path": "x"}}'
        results = _envs(text)
        assert len(results) == 1
        assert results[0].rule_id == "unclosed_but_balanced"
        assert results[0].payload == '{"name": "read_file", "arguments": {"path": "x"}}'

    def test_unclosed_and_truncated_json_not_recovered(self):
        """Genuinely truncated output (mid-argument) must NOT be recovered —
        that's a correctness/safety property, not a gap. Executing a call
        built from truncated arguments would be actively wrong."""
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "x'
        results = _envs(text)
        assert results == []

    def test_two_independent_unclosed_calls_each_recovered(self):
        text = (
            '<tool_call>{"name": "a"}<tool_call>{"name": "b"}'
        )
        results = _envs(text)
        assert len(results) == 2
        assert {r.payload for r in results} == {'{"name": "a"}', '{"name": "b"}'}

    def test_trailing_junk_after_json_blocks_recovery(self):
        """If there's non-whitespace content after the JSON besides a
        proper close tag, this isn't the "forgot to close" shape — don't
        guess."""
        text = '<tool_call>{"name": "a"} and then some more unrelated text'
        results = _envs(text)
        assert results == []


class TestDroppedOpenTag:
    def test_missing_open_but_balanced_json_before_close_recovered(self):
        text = '{"name": "read_file", "arguments": {"path": "x"}}</tool_call>'
        results = _envs(text)
        assert len(results) == 1
        assert results[0].rule_id == "missing_open_tag"

    def test_junk_between_json_and_close_tag_blocks_recovery(self):
        text = '{"name": "a"} some unrelated prose </tool_call>'
        results = _envs(text)
        assert results == []


class TestSafetyProperty:
    def test_bare_json_with_no_tag_never_matches(self):
        """The core safety property: no tag token anywhere means zero
        candidates, full stop — this is what keeps the scanner from
        reinterpreting ordinary JSON output as a tool call."""
        text = 'Here is the config: {"name": "read_file", "arguments": {"path": "x"}}'
        results = _envs(text)
        assert results == []

    def test_json_code_example_in_prose_not_matched(self):
        text = (
            'You can call it like this: {"name": "read_file"} — '
            'but I am not calling it right now.'
        )
        results = _envs(text)
        assert results == []

    def test_empty_text(self):
        assert _envs("") == []

    def test_ordinary_html_like_prose_not_matched(self):
        text = "Use <div> tags for layout and <span> for inline text."
        results = _envs(text)
        assert results == []


class TestFencedCodeNeverRecovered:
    """Fenced code is never extracted, under any shape, no exceptions — an
    existing, deliberate invariant elsewhere in the extraction pipeline
    (see test_deep_integration.py::test_fenced_code_not_extracted). This
    module does not carry a fence-unwrapping rule at all: `recover_envelope`
    only ever sees `masked_text` (fenced code already blanked out by the
    caller before either the primary regex or this recovery path runs), so
    there is nothing for it to find inside a fence regardless of the
    message's shape."""

    def _recovered(self, raw: str, canonical=("tool_call",), alias=("tool_calls", "toolcall")):
        masked = _mask_fenced_code(raw)
        return recover_envelope(masked_text=masked, canonical_names=canonical, alias_names=alias)

    def test_whole_message_is_one_fenced_call(self):
        assert self._recovered('```json\n<tool_call>{"name": "read_file"}</tool_call>\n```') == []

    def test_prose_followed_by_fenced_block(self):
        text = (
            'Here is an example:\n```json\n<tool_call>{"name": "x"}</tool_call>\n```\n'
            'That is how you would call it, but I will not call it now.'
        )
        assert self._recovered(text) == []

    def test_tilde_fence(self):
        assert self._recovered('~~~json\n<tool_call>{"name": "x"}</tool_call>\n~~~') == []

    def test_language_tagged_fence(self):
        assert self._recovered('```xml\n<tool_call>{"name": "x"}</tool_call>\n```') == []

    def test_multiple_examples_with_one_real_unfenced_call(self):
        raw = (
            'Example:\n```json\n<tool_call>{"name": "example"}</tool_call>\n```\n'
            'Now for real: <tool_call>{"name": "real"}</tool_call>'
        )
        recovered = self._recovered(raw)
        # The real, unfenced call is found by the PRIMARY regex (paired,
        # well-formed) before recovery ever runs — recovery only sees
        # masked_text, which still has that clean match in it. Prove the
        # example fence contributes nothing to whatever is found.
        direct = _envs(_mask_fenced_code(raw))
        assert len(direct) == 1
        assert direct[0].payload == '{"name": "real"}'
        assert recovered == direct  # recover_envelope defers to the same scan
