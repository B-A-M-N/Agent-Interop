"""String-aware tag-envelope scanner.

Finds tool-call envelopes such as ``<tool_call>...</tool_call>`` in model
output using a single-pass tag tokenizer plus stack-based pairing, rather
than a set of independent regexes each hand-matching one failure shape.

This mirrors ``json_scan.BalancedJsonScanner`` one level up: a regex is the
right tool for finding flat, non-recursive tokens (tag markers), but
reconstructing PAIRING structure from those tokens — which open matches
which close, what's left dangling — is a stack problem, not something a
regular expression can do in general. A regex that tries to hand-match each
defect shape (missing close, missing open, mixed alias pairing, multiple
independent calls) as a separate pattern misses combinations by
construction; a stack walk handles all of them uniformly in one pass.

Every recovered envelope is still anchored to at least one real tag token
found by the tokenizer — this scanner never treats unanchored text as a
candidate. That is what keeps it safe: ordinary prose containing no tag
produces no tokens at all, so it cannot manufacture a tool call out of a
JSON code example or a quoted config snippet. The only place free-floating
JSON is ever considered is immediately adjacent to a real tag token
(right after a dangling open, or right before a dangling close), never
scanned across the whole message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from agent_interop.parsing.json_scan import BalancedJsonScanner

# Tokenizes any XML-ish tag: <name ...>, </name>, <name/>. Deliberately
# permissive on internal whitespace/attributes (`[^>]*?`) since that
# variance is not where the ambiguity risk lives — the risk is entirely in
# which NAMES are treated as belonging to a given dialect's tag family,
# which callers control via `canonical_names`/`alias_names`.
_TAG_TOKEN_RE = re.compile(r"<\s*(?P<close>/?)\s*(?P<name>[A-Za-z_][\w-]*)\s*[^>]*?(?P<selfclose>/?)\s*>")


@dataclass(frozen=True)
class TagToken:
    start: int
    end: int
    name: str  # lowercased
    is_close: bool
    is_self_closing: bool


@dataclass(frozen=True)
class EnvelopeMatch:
    """A recovered tool-call envelope."""

    payload: str
    start: int  # start of the open tag, or of the JSON if the open tag is missing
    end: int    # end of the close tag, or of the JSON if the close tag is missing
    rule_id: str  # "paired" | "tag_alias" | "unclosed_but_balanced" | "missing_open_tag"


def _scan_tags(text: str) -> list[TagToken]:
    tokens = []
    for m in _TAG_TOKEN_RE.finditer(text):
        tokens.append(TagToken(
            start=m.start(),
            end=m.end(),
            name=m.group("name").lower(),
            is_close=bool(m.group("close")),
            is_self_closing=bool(m.group("selfclose")),
        ))
    return tokens


def _first_balanced_payload(scanner: BalancedJsonScanner, text: str) -> tuple[str, int, int] | None:
    """First complete (non-truncated) balanced JSON span in `text`, or None."""
    for span in scanner.scan(text):
        if span.parse() is not None:
            return span.text, span.start, span.end
    return None


def scan_envelopes(
    text: str,
    *,
    canonical_names: tuple[str, ...],
    alias_names: tuple[str, ...] = (),
) -> list[EnvelopeMatch]:
    """Find every tool-call envelope belonging to a tag family in `text`.

    ``canonical_names`` is the dialect's own tag(s) (e.g. ``("tool_call",)``
    for Hermes/generic, ``("tool",)`` for Qwen). ``alias_names`` are other
    tag spellings seen in real model output for the same intent (e.g.
    "tool_calls", "toolcall") — recovered, but tagged with a distinct
    ``rule_id`` so callers can see when a model is drifting off-dialect.

    Handles, uniformly, via one tag scan + stack pairing:
    - case variation in tag names (tokenizer lowercases)
    - mixed canonical/alias open+close pairing
    - a dropped close tag (recovered iff the trailing content is a
      complete, non-truncated JSON object — never a truncated one)
    - a dropped open tag (recovered iff the preceding content is a
      complete JSON object ending exactly at the close tag)
    - multiple independent envelopes in the same message, each resolved
      against its own neighboring tokens rather than the whole message

    Never matches unanchored text: every result requires at least one real
    tag token from the tokenizer.
    """
    family = frozenset(n.lower() for n in (*canonical_names, *alias_names))
    canonical = frozenset(n.lower() for n in canonical_names)
    tokens = [t for t in _scan_tags(text) if t.name in family]
    if not tokens:
        return []

    scanner = BalancedJsonScanner()
    results: list[EnvelopeMatch] = []
    stack: list[TagToken] = []
    cursor = 0  # text already claimed by a resolved close-with-no-open pairing

    for tok in tokens:
        if not tok.is_close:
            stack.append(tok)
            continue

        if stack:
            open_tok = stack.pop()
            payload = text[open_tok.end:tok.start].strip()
            rule = "paired" if (open_tok.name in canonical and tok.name in canonical) else "tag_alias"
            results.append(EnvelopeMatch(payload, open_tok.start, tok.end, rule))
            cursor = tok.end
            continue

        # Close tag with nothing open on the stack: the model may have
        # dropped the opening tag. Only recover if the region since the
        # last resolved position is a complete balanced JSON object ending
        # exactly where this close tag starts — not merely "some JSON
        # appears somewhere in there".
        region = text[cursor:tok.start]
        found = None
        for span in scanner.scan(region):
            if span.parse() is None:
                continue
            if region[span.end:].strip():
                continue
            found = span
        if found is not None:
            results.append(EnvelopeMatch(
                found.text, cursor + found.start, tok.end, "missing_open_tag",
            ))
            cursor = tok.end

    # Opens left on the stack never got a close. Recover each
    # independently, bounded by the NEXT tag token (or end of text) so
    # multiple unclosed calls in one message don't bleed into each other.
    for open_tok in stack:
        idx = tokens.index(open_tok)
        boundary = tokens[idx + 1].start if idx + 1 < len(tokens) else len(text)
        after = text[open_tok.end:boundary]
        balanced = _first_balanced_payload(scanner, after)
        if balanced is None:
            continue
        payload, rel_start, rel_end = balanced
        if after[:rel_start].strip():
            continue  # something other than whitespace precedes the JSON
        if after[rel_end:].strip():
            continue  # something other than whitespace follows the JSON —
            # ambiguous whether trailing content belonged inside the call;
            # don't guess.
        results.append(EnvelopeMatch(
            payload, open_tok.start, open_tok.end + rel_end, "unclosed_but_balanced",
        ))

    results.sort(key=lambda r: r.start)
    return results


def recover_envelope(
    *,
    masked_text: str,
    canonical_names: tuple[str, ...],
    alias_names: tuple[str, ...] = (),
) -> list[EnvelopeMatch]:
    """Envelope recovery for use when a dialect's strict primary regex found
    no match: structural repair (alias pairing, dropped open/close tag) on
    ``masked_text`` — fenced-code-masked, exactly like the primary regex
    the caller already ran, so a literal example elsewhere in a longer
    message can never be recovered either way.

    Deliberately does NOT unwrap fenced code blocks under any
    circumstances, including a message that is nothing but one fence
    wrapping one clean envelope: that policy ("fenced code is never
    extracted") is an existing, deliberate invariant elsewhere in the
    extraction pipeline, and this module does not override it.
    """
    return scan_envelopes(masked_text, canonical_names=canonical_names, alias_names=alias_names)
