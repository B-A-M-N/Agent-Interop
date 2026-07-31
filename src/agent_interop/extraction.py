"""Tool candidate extraction — transport-independent candidate extraction.

Extracts structured tool-call candidates from model output content blocks,
regardless of whether they arrived as native structured output or as text
emitted in a declared envelope.

The extraction layer sits between upstream codec decoding and the universal
tool transaction service. It is selected by the model profile, NOT by the
upstream codec, because an OpenAI-compatible upstream can serve Qwen,
DeepSeek, Llama or any other model dialect.

Critical rule: extractors preserve raw payload evidence so the transaction
service can repair malformed arguments. Never parse arguments during
extraction — that is the repair pipeline's job.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agent_interop.abi import (
    CanonicalContentBlock,
    CanonicalTextBlock,
    CanonicalTool,
    CanonicalToolChoice,
    RawToolCallCandidate,
    ToolChoiceMode,
)
from agent_interop.parsing.envelope_scan import EnvelopeMatch, recover_envelope


@dataclass(frozen=True)
class ExtractionDiagnostic:
    """A diagnostic note about the extraction process."""

    level: str = "info"  # info | warning | error
    message: str = ""
    envelope: str = ""


@dataclass(frozen=True)
class ExtractionResult:
    """Result of extracting tool candidates from content blocks.

    Carries the extracted candidates, any content that was not consumed
    as tool calls (including reasoning text), diagnostics, and a
    confidence score indicating how certain the extractor is about
    the candidates it produced.
    """

    candidates: tuple[RawToolCallCandidate, ...] = ()
    remaining_content: tuple[CanonicalContentBlock, ...] = ()
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()
    reasoning_content_remainder: str = ""
    consumed_spans: tuple[tuple[int, int], ...] = ()
    confidence: float = 1.0


class ToolCandidateExtractor(Protocol):
    """Protocol for tool candidate extractors.

    Each extractor handles one dialect or envelope format.
    """

    id: str

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        """Extract tool candidates from content blocks.

        Must preserve raw argument evidence for repair — never parse
        arguments during extraction.
        """
        ...


# ─── Confidence calculation (item 58) ──────────────────────────────────────


def compute_extraction_confidence(
    candidate_count: int,
    *,
    tool_names: set[str],
    candidate_names: list[str],
    envelope: str | None,
    from_fallback: bool = False,
) -> float:
    """Compute extraction confidence based on structural evidence.

    Factors:
    - High (0.95): Name matches a declared tool in a structured envelope
    - Medium (0.7): Name matches but envelope is generic or non-standard
    - Low (0.4): Name doesn't match any declared tool (rejection candidate)
    - Fallback: 30% penalty on top of whatever the base score would be
    """
    if not candidate_names:
        return 1.0  # No candidates extracted — no uncertainty

    base_confidence = 1.0
    for name in candidate_names:
        if name in tool_names:
            # Named tool — high confidence
            tool_conf = 0.95
        else:
            # Unknown tool — much lower, pipeline will reject
            tool_conf = 0.4
        base_confidence = min(base_confidence, tool_conf)

    # Generic envelope (no structured markers) reduces confidence
    if envelope is None:
        base_confidence = min(base_confidence, 0.7)

    # Fallback penalty (item 60)
    if from_fallback:
        base_confidence *= 0.7

    return round(base_confidence, 2)


# ─── Generic <tool_call> envelope extractor ─────────────────────────────────

# Matches <tool_call>...</tool_call> with robust handling of malformed closing.
# Case-insensitive: models occasionally vary tag case (<Tool_Call>, <TOOL_CALL>)
# — the tag name is specific and deliberate enough that case variance is not
# a plausible source of false positives from ordinary prose.
_TOOL_CALL_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
# Fallback for unclosed envelopes
_TOOL_CALL_UNCLOSED_RE = re.compile(r"<tool_call\b[^>]*>(.*?)$", re.DOTALL | re.IGNORECASE)
_TOOL_CALL_OPENING_RE = re.compile(r"<tool_call\b(?P<attributes>[^>]*)>", re.IGNORECASE)
_XML_NAME_ATTRIBUTE_RE = re.compile(
    r"\bname\s*=\s*(?:\"(?P<double>[^\"]*)\"|'(?P<single>[^']*)'|(?P<bare>[^\s>]+))",
    re.IGNORECASE,
)
_XML_ARGUMENTS_ATTRIBUTE_RE = re.compile(r"\barguments\s*=\s*", re.IGNORECASE)
# Matches fenced code blocks (```...``` or ~~~...~~~).  These
# regions are excluded from tool-call extraction to avoid matching
# example tool invocations in literal code.
_FENCED_CODE_RE = re.compile(
    r"(?P<fence>```[^\n]*\n.*?```|~~~[^\n]*\n.*?~~~)",
    re.DOTALL,
)
# An UNCLOSED fence: an opening ``` / ~~~ marker with no matching close
# before the end of the text. This is not "not a fence" — it's a fence
# whose closing marker hasn't arrived yet (or never will, if generation was
# truncated). Masking only CLOSED fences would leave a literal example
# inside an incomplete one fully visible to extraction, letting it become
# an executable tool call.
_UNCLOSED_FENCE_RE = re.compile(r"(?:```|~~~)[^\n]*\n.*$", re.DOTALL)


def _mask_fenced_code(text: str) -> str:
    """Replace fenced code blocks — closed or unclosed — with spaces to
    prevent false extraction."""
    closed_masked = _FENCED_CODE_RE.sub(lambda m: " " * len(m.group(0)), text)
    return _UNCLOSED_FENCE_RE.sub(lambda m: " " * len(m.group(0)), closed_masked)


# Known dialect-drift tag spellings, by canonical tag. A local model taught
# one exact envelope via prompt injection sometimes drifts toward a nearby
# spelling of ITS OWN tag (typo, pluralization, underscore/no-underscore).
# Deliberately does NOT cross into another extractor's canonical tag (e.g.
# Qwen's family must never include "tool_call" — that belongs to
# Hermes/generic) — which dialect a model is speaking is decided by profile
# resolution and extractor SELECTION, not guessed here by absorbing another
# extractor's namespace as an "alias". Blurring that boundary would make
# extraction results depend on which extractor happened to run first rather
# than on the model's actual declared dialect.
_ENVELOPE_TAG_ALIASES: dict[str, tuple[str, ...]] = {
    "tool_call": ("tool_calls", "toolcall", "function_call"),
    "tool": ("tools", "toolcall"),
}


def _recover_first_envelope(
    masked_text: str,
    *,
    canonical_names: tuple[str, ...],
) -> EnvelopeMatch | None:
    """Attempt bounded envelope recovery when a dialect's strict primary
    regex found no match in this block. Returns the first recovered
    envelope, if any — recovering every independent envelope in a single
    defective block is out of scope for now; the primary regex already
    handles the common multi-call case (well-formed tags), so this only
    needs to cover the rarer "at least one call in this block is
    defective" case.
    """
    alias_names = _ENVELOPE_TAG_ALIASES.get(canonical_names[0], ())
    matches = recover_envelope(
        masked_text=masked_text,
        canonical_names=canonical_names,
        alias_names=alias_names,
    )
    return matches[0] if matches else None


class ToolCallEnvelopeExtractor:
    """Extracts tool calls from <tool_call>...</tool_call> envelopes.

    Preserves raw payload evidence for repair — does NOT parse arguments.
    Even a malformed envelope that clearly intends a tool call produces a
    candidate with raw_arguments preserved.
    """

    id = "tool_call_envelope"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []
        diagnostics: list[ExtractionDiagnostic] = []

        tool_names = {t.name for t in tools}

        consumed_spans: list[tuple[int, int]] = []
        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            # Mask fenced code blocks (closed AND unclosed — see
            # _mask_fenced_code) so literal tool-call examples inside
            # example code are not extracted as candidates.
            masked_text = _mask_fenced_code(text)
            # Collect the code-fence spans (closed and unclosed) so they
            # are advertised in the extraction result for downstream
            # debugging.
            for m in _FENCED_CODE_RE.finditer(text):
                consumed_spans.append((m.start(), m.end()))
            for m in _UNCLOSED_FENCE_RE.finditer(_FENCED_CODE_RE.sub(
                lambda fm: " " * len(fm.group(0)), text,
            )):
                consumed_spans.append((m.start(), m.end()))

            matches = list(_TOOL_CALL_RE.finditer(masked_text))

            if not matches:
                recovered = _recover_first_envelope(
                    masked_text, canonical_names=("tool_call",),
                )
                if recovered is not None:
                    raw_payload = recovered.payload.strip()
                    name, arguments = _extract_name_and_args_from_raw(raw_payload)
                    if name or raw_payload:
                        prefix = text[:recovered.start].strip()
                        if prefix:
                            remaining.append(CanonicalTextBlock(text=prefix))
                        candidates.append(RawToolCallCandidate(
                            name=name,
                            raw_arguments=arguments if arguments is not None else raw_payload,
                            source_protocol="tool_call_envelope",
                            source_index=block_idx,
                            source_text=raw_payload,
                            raw_name=name,
                            provenance=_make_provenance(
                                "model_output", "tool_call_envelope", name,
                                arguments if arguments is not None else raw_payload,
                            ),
                        ))
                        diagnostics.append(ExtractionDiagnostic(
                            level="warning",
                            message=f"Recovered <tool_call> envelope at block {block_idx} "
                                    f"via '{recovered.rule_id}' repair",
                            envelope="tool_call",
                        ))
                        if name and tool_names and name not in tool_names:
                            diagnostics.append(ExtractionDiagnostic(
                                level="warning",
                                message=f"Unknown tool '{name}' in envelope at block {block_idx}",
                                envelope="tool_call",
                            ))
                        suffix = text[recovered.end:].strip()
                        if suffix:
                            remaining.append(CanonicalTextBlock(text=suffix))
                        consumed_spans.append((recovered.start, recovered.end))
                        continue

                # Check for unclosed envelope — reject with diagnostic (item 55).
                # Incomplete envelopes must not be silently accepted because
                # the arguments are almost certainly truncated.
                unclosed = _TOOL_CALL_UNCLOSED_RE.search(masked_text)
                if unclosed:
                    diagnostics.append(ExtractionDiagnostic(
                        level="error",
                        message=f"Rejected unclosed <tool_call> envelope at block {block_idx}: "
                                f"incomplete arguments cannot be safely executed",
                        envelope="tool_call",
                    ))
                remaining.append(block)
                continue

            # Process each match
            last_end = 0
            for match in matches:
                # Preserve text before this envelope
                if match.start() > last_end:
                    prefix = text[last_end:match.start()].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))

                raw_payload = match.group(1).strip()

                # Extract name and arguments from the raw payload. Llama 3.2
                # can put both on an otherwise-empty opening XML tag.
                name, arguments = _extract_name_and_args_from_raw(raw_payload)
                source_text = raw_payload
                if not raw_payload and not name:
                    name, arguments = _extract_name_and_args_from_tag_attributes(match.group(0))
                    source_text = match.group(0)

                if name and tool_names and name not in tool_names:
                    # Unknown tool — emit candidate anyway for rejection with diagnostics
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="tool_call_envelope",
                        source_index=block_idx,
                        source_text=source_text,
                        raw_name=name,
                        provenance=_make_provenance(
                            "model_output",
                            "tool_call_envelope",
                            name,
                            arguments if arguments is not None else raw_payload,
                        ),
                    ))
                    diagnostics.append(ExtractionDiagnostic(
                        level="warning",
                        message=f"Unknown tool '{name}' in envelope at block {block_idx}",
                        envelope="tool_call",
                    ))
                elif name or raw_payload:
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="tool_call_envelope",
                        source_index=block_idx,
                        source_text=source_text,
                        raw_name=name,
                        provenance=_make_provenance(
                            "model_output",
                            "tool_call_envelope",
                            name,
                            arguments if arguments is not None else raw_payload,
                        ),
                    ))

                last_end = match.end()

            # Preserve text after last envelope
            if last_end < len(text):
                suffix = text[last_end:].strip()
                if suffix:
                    remaining.append(CanonicalTextBlock(text=suffix))
            # Record the consumed envelope spans
            for m in matches:
                consumed_spans.append((m.start(), m.end()))

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            diagnostics=tuple(diagnostics),
            consumed_spans=tuple(consumed_spans),
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope="tool_call",
            ),
        )


def _extract_name_from_raw(raw_payload: str) -> str | None:
    """Try to extract a tool name from raw envelope payload without full parse."""
    name, _ = _extract_name_and_args_from_raw(raw_payload)
    return name


def _extract_name_and_args_from_raw(raw_payload: str) -> tuple[str | None, Any]:
    """Extract name and arguments from raw envelope payload.

    Returns (name, arguments) where arguments is the raw argument value
    (always a string, never a parsed dict). Handles malformed JSON gracefully.

    Uses the BalancedJsonScanner to identify exact field-value spans in
    the wrapper object so that the argument value is preserved verbatim
    without requiring it to be valid JSON.
    """
    from agent_interop.parsing.json_scan import BalancedJsonScanner

    # Use field-span scanner to identify exact value spans.
    # This preserves the raw_arguments value verbatim (including trailing
    # commas, unclosed braces, etc.) without attempting to parse it.
    scanner = BalancedJsonScanner()
    fields = scanner.extract_field_spans(raw_payload)

    name: str | None = None
    arguments_raw: str | None = None

    for field in fields:
        if field.key == "name":
            # Strip surrounding quotes from string value
            if field.raw_value.startswith('"') and field.raw_value.endswith('"'):
                extracted = field.raw_value[1:-1]
                # Unescape JSON escapes for the name value
                name = json.loads(f'"{extracted}"') if extracted else extracted
            else:
                name = field.raw_value  # non-string (shouldn't happen, but be safe)
        elif field.key == "arguments":
            arguments_raw = field.raw_value

    if arguments_raw is not None:
        # Return the exact raw text of the arguments value — no parsing.
        return name, arguments_raw

    return name, raw_payload


def _extract_name_and_args_from_tag_attributes(opening_tag: str) -> tuple[str | None, str | None]:
    """Recover a complete call encoded on an empty ``<tool_call>`` tag."""
    opening = _TOOL_CALL_OPENING_RE.match(opening_tag)
    if opening is None:
        return None, None
    attributes = opening.group("attributes")
    name_match = _XML_NAME_ATTRIBUTE_RE.search(attributes)
    name = None
    if name_match is not None:
        name = next(
            (value for value in name_match.group("double", "single", "bare") if value is not None),
            None,
        )
    arguments_match = _XML_ARGUMENTS_ATTRIBUTE_RE.search(attributes)
    if arguments_match is None:
        return name, None

    from agent_interop.parsing.json_scan import BalancedJsonScanner

    trailing = attributes[arguments_match.end():].lstrip()
    spans = BalancedJsonScanner().scan(trailing)
    if spans and spans[0].start == 0:
        return name, spans[0].text
    return name, None


# ─── Native structured candidate passthrough ───────────────────────────────


class NativeStructuredExtractor:
    """Passes through candidates already decoded by an upstream codec."""

    id = "native_structured"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        # Native candidates are handled by the upstream codec's decode_response.
        # This extractor is a passthrough for content that has no textual candidates.
        # Confidence is 1.0 (no extraction uncertainty — codec already structured).
        return ExtractionResult(remaining_content=tuple(content), confidence=1.0)


# ─── Hermes extractor ───────────────────────────────────────────────────────

_HERMES_RE = re.compile(r"<tool_call\b[^>]*>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


class HermesExtractor:
    """Extracts tool calls from Hermes format: <tool_call>JSON</tool_call>.

    Hermes models (NousResearch) emit tool calls inside <tool_call> XML tags.
    The content between tags is a JSON object with "name" and "arguments" keys.

    Outer markers: <tool_call>...</tool_call>
    Bare JSON allowed: No
    Name location: JSON "name" key
    Arguments location: JSON "arguments" key
    Multiple calls: Yes, via repeated <tool_call> blocks
    Confidence requirement: Must have both "name" and "arguments"
    False-positive protection: Only matches explicit <tool_call> XML tags
    """

    id = "hermes"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []
        diagnostics: list[ExtractionDiagnostic] = []

        tool_names = {t.name for t in tools}

        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            masked_text = _mask_fenced_code(text)
            matches = list(_HERMES_RE.finditer(masked_text))

            if not matches:
                recovered = _recover_first_envelope(
                    masked_text, canonical_names=("tool_call",),
                )
                if recovered is None:
                    remaining.append(block)
                    continue
                raw_payload = recovered.payload.strip()
                name, arguments = _normalize_name_and_args_from_json(raw_payload)
                if name or raw_payload:
                    prefix = text[:recovered.start].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="hermes",
                        source_index=block_idx,
                        source_text=raw_payload,
                        raw_name=name or "",
                        provenance=_make_provenance("model_output", "hermes", name, arguments),
                    ))
                    diagnostics.append(ExtractionDiagnostic(
                        level="warning",
                        message=f"Recovered Hermes envelope at block {block_idx} "
                                f"via '{recovered.rule_id}' repair",
                        envelope="hermes",
                    ))
                    if name and tool_names and name not in tool_names:
                        diagnostics.append(ExtractionDiagnostic(
                            level="warning",
                            message=f"Unknown tool '{name}' in Hermes block {block_idx}",
                            envelope="hermes",
                        ))
                    suffix = text[recovered.end:].strip()
                    if suffix:
                        remaining.append(CanonicalTextBlock(text=suffix))
                    continue
                remaining.append(block)
                continue

            last_end = 0
            for match in matches:
                if match.start() > last_end:
                    prefix = text[last_end:match.start()].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))

                raw_payload = match.group(1).strip()
                name, arguments = _normalize_name_and_args_from_json(raw_payload)

                if name or raw_payload:
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="hermes",
                        source_index=block_idx,
                        source_text=raw_payload,
                        raw_name=name or "",
                        provenance=_make_provenance("model_output", "hermes", name, arguments),
                    ))
                    if name and tool_names and name not in tool_names:
                        diagnostics.append(ExtractionDiagnostic(
                            level="warning",
                            message=f"Unknown tool '{name}' in Hermes block {block_idx}",
                            envelope="hermes",
                        ))

                last_end = match.end()

            if last_end < len(text):
                suffix = text[last_end:].strip()
                if suffix:
                    remaining.append(CanonicalTextBlock(text=suffix))

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            diagnostics=tuple(diagnostics),
            consumed_spans=(),  # Could track match spans if needed
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope="native_structured",
            ),
        )


# ─── Qwen extractor ─────────────────────────────────────────────────────────

_QWEN_TOOL_RE = re.compile(
    r"<tool>\s*(.*?)\s*</tool>",
    re.DOTALL | re.IGNORECASE,
)


class QwenExtractor:
    """Extracts tool calls from Qwen XML format: <tool>JSON</tool>.

    Qwen models (Qwen2.5, Qwen3) emit tool calls inside <tool> XML tags.
    The content between tags is a JSON object with name and arguments.

    Outer markers: <tool>...</tool>
    Bare JSON allowed: No
    Name location: JSON "name" or "tool" key
    Arguments location: JSON "arguments", "parameters", or "input" key
    Multiple calls: Yes, via repeated <tool> blocks
    Confidence requirement: Must have a name field
    False-positive protection: Only matches explicit <tool> XML tags with JSON content
    """

    id = "qwen"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []
        diagnostics: list[ExtractionDiagnostic] = []

        tool_names = {t.name for t in tools}

        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            # Mask fenced code before matching — a Qwen-format tool-call
            # example inside a literal code block must not be executed.
            # (Previously this extractor matched raw, unmasked text; the
            # other two XML-tag extractors already masked.)
            masked_text = _mask_fenced_code(text)
            matches = list(_QWEN_TOOL_RE.finditer(masked_text))

            if not matches:
                recovered = _recover_first_envelope(
                    masked_text, canonical_names=("tool",),
                )
                if recovered is None:
                    remaining.append(block)
                    continue
                raw_payload = recovered.payload.strip()
                name, arguments = _normalize_name_and_args_from_json(raw_payload)
                if name or raw_payload:
                    prefix = text[:recovered.start].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="qwen",
                        source_index=block_idx,
                        source_text=raw_payload,
                        raw_name=name or "",
                        provenance=_make_provenance("model_output", "qwen", name, arguments),
                    ))
                    diagnostics.append(ExtractionDiagnostic(
                        level="warning",
                        message=f"Recovered Qwen envelope at block {block_idx} "
                                f"via '{recovered.rule_id}' repair",
                        envelope="qwen",
                    ))
                    if name and tool_names and name not in tool_names:
                        diagnostics.append(ExtractionDiagnostic(
                            level="warning",
                            message=f"Unknown tool '{name}' in Qwen block {block_idx}",
                            envelope="qwen",
                        ))
                    suffix = text[recovered.end:].strip()
                    if suffix:
                        remaining.append(CanonicalTextBlock(text=suffix))
                    continue
                remaining.append(block)
                continue

            last_end = 0
            for match in matches:
                if match.start() > last_end:
                    prefix = text[last_end:match.start()].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))

                raw_payload = match.group(1).strip()
                name, arguments = _normalize_name_and_args_from_json(raw_payload)

                if name or raw_payload:
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="qwen",
                        source_index=block_idx,
                        source_text=raw_payload,
                        raw_name=name or "",
                        provenance=_make_provenance("model_output", "qwen", name, arguments),
                    ))
                    if name and tool_names and name not in tool_names:
                        diagnostics.append(ExtractionDiagnostic(
                            level="warning",
                            message=f"Unknown tool '{name}' in Qwen block {block_idx}",
                            envelope="qwen",
                        ))

                last_end = match.end()

            if last_end < len(text):
                suffix = text[last_end:].strip()
                if suffix:
                    remaining.append(CanonicalTextBlock(text=suffix))

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            diagnostics=tuple(diagnostics),
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope="qwen",
            ),
        )


# ─── Mistral extractor ──────────────────────────────────────────────────────

_MISTRAL_TOOL_RE = re.compile(
    r"\[TOOL_CALLS\]\s*(.*?)(?:\n|$)",
    re.DOTALL,
)


class MistralExtractor:
    """Extracts tool calls from Mistral format: [TOOL_CALLS]{...} or [TOOL_CALLS][{...}, {...}].

    Mistral models emit tool calls starting with the [TOOL_CALLS] marker followed
    by a JSON array of tool call objects or a single JSON object.

    Outer markers: [TOOL_CALLS] prefix
    Bare JSON allowed: No (requires [TOOL_CALLS] prefix)
    Name location: JSON "name" or "function" key
    Arguments location: JSON "arguments" key
    Multiple calls: Yes, via JSON array after [TOOL_CALLS]
    Confidence requirement: Must have [TOOL_CALLS] prefix and valid JSON
    False-positive protection: Requires [TOOL_CALLS] prefix, unlikely in prose
    """

    id = "mistral"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []
        diagnostics: list[ExtractionDiagnostic] = []

        tool_names = {t.name for t in tools}

        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            masked_text = _mask_fenced_code(text)
            matches = list(_MISTRAL_TOOL_RE.finditer(masked_text))

            if not matches:
                remaining.append(block)
                continue

            last_end = 0
            for match in matches:
                if match.start() > last_end:
                    prefix = text[last_end:match.start()].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))

                raw_payload = match.group(1).strip()
                # Mistral: must look like JSON ({ or [) — reject prose
                if not raw_payload or not (raw_payload.startswith(("{", "["))):
                    if match.start() > last_end:
                        prefix = text[last_end:match.start()].strip()
                        if prefix:
                            remaining.append(CanonicalTextBlock(text=prefix))
                    last_end = match.end()
                    continue

                # Mistral can emit a single object or an array of objects
                payloads = _unpack_json_payload(raw_payload)

                for item_payload in payloads:
                    name, arguments = _normalize_name_and_args_from_json(item_payload)

                    if name or item_payload:
                        candidates.append(RawToolCallCandidate(
                            name=name,
                            raw_arguments=arguments if arguments is not None else item_payload,
                            source_protocol="mistral",
                            source_index=block_idx,
                            source_text=item_payload,
                            raw_name=name or "",
                            provenance=_make_provenance("model_output", "mistral", name, arguments),
                        ))
                        if name and tool_names and name not in tool_names:
                            diagnostics.append(ExtractionDiagnostic(
                                level="warning",
                                message=f"Unknown tool '{name}' in Mistral block {block_idx}",
                                envelope="mistral",
                            ))

                last_end = match.end()

            if last_end < len(text):
                suffix = text[last_end:].strip()
                if suffix:
                    remaining.append(CanonicalTextBlock(text=suffix))

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            diagnostics=tuple(diagnostics),
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope="mistral",
            ),
        )


# ─── DeepSeek extractor ─────────────────────────────────────────────────────

_DEEPSEEK_TOOL_RE = re.compile(
    r"\x14(.*?)\x14",
    re.DOTALL,
)


class DeepSeekExtractor:
    """Extracts tool calls from DeepSeek format: \\x14{...}\\x14.

    DeepSeek models emit tool calls wrapped in the STX control character (\\x14).
    The content between markers is a JSON object with function call details.

    Outer markers: \\x14 (STX control character) delimiters
    Bare JSON allowed: No (requires \\x14 delimiters)
    Name location: JSON wrapped in {"name": ..., "arguments": ...} or
                   {"function": {"name": ..., "arguments": ...}}
    Arguments location: JSON "arguments" key (possibly nested)
    Multiple calls: Yes, via multiple \\x14...\\x14 pairs
    Confidence requirement: Must have a name field
    False-positive protection: \\x14 control chars are unlikely in normal prose
    """

    id = "deepseek"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []
        diagnostics: list[ExtractionDiagnostic] = []

        tool_names = {t.name for t in tools}

        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            masked_text = _mask_fenced_code(text)
            matches = list(_DEEPSEEK_TOOL_RE.finditer(masked_text))

            if not matches:
                remaining.append(block)
                continue

            last_end = 0
            for match in matches:
                if match.start() > last_end:
                    prefix = text[last_end:match.start()].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))

                raw_payload = match.group(1).strip()
                name, arguments = _normalize_name_and_args_from_json(raw_payload)

                if name or raw_payload:
                    candidates.append(RawToolCallCandidate(
                        name=name,
                        raw_arguments=arguments if arguments is not None else raw_payload,
                        source_protocol="deepseek",
                        source_index=block_idx,
                        source_text=raw_payload,
                        raw_name=name or "",
                        provenance=_make_provenance("model_output", "deepseek", name, arguments),
                    ))
                    if name and tool_names and name not in tool_names:
                        diagnostics.append(ExtractionDiagnostic(
                            level="warning",
                            message=f"Unknown tool '{name}' in DeepSeek block {block_idx}",
                            envelope="deepseek",
                        ))

                last_end = match.end()

            if last_end < len(text):
                suffix = text[last_end:].strip()
                if suffix:
                    remaining.append(CanonicalTextBlock(text=suffix))

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            diagnostics=tuple(diagnostics),
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope="deepseek",
            ),
        )


# ─── Llama extractor ────────────────────────────────────────────────────────

_LLAMA_PYTHON_TAG_RE = re.compile(
    r"<\|python_tag\|>(.*?)(?:\n|$)",
    re.DOTALL,
)


class LlamaExtractor:
    """Extracts tool calls from Llama format: <|python_tag|>{...}

    Llama models (Meta) may emit tool calls prefixed with <|python_tag|> when
    using the built-in API tool_calls field as fallback. The content is a JSON
    object with function call details.

    Outer markers: <|python_tag|> prefix
    Bare JSON allowed: No (requires <|python_tag|> prefix)
    Name location: JSON "name" or "function" key
    Arguments location: JSON "arguments" key
    Multiple calls: Yes, via multiple <|python_tag|> blocks
    Confidence requirement: Must have a name field
    False-positive protection: Only matches explicit <|python_tag|> markers
    """

    id = "llama"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []
        diagnostics: list[ExtractionDiagnostic] = []

        tool_names = {t.name for t in tools}

        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            masked_text = _mask_fenced_code(text)
            matches = list(_LLAMA_PYTHON_TAG_RE.finditer(masked_text))

            if not matches:
                remaining.append(block)
                continue

            last_end = 0
            for match in matches:
                if match.start() > last_end:
                    prefix = text[last_end:match.start()].strip()
                    if prefix:
                        remaining.append(CanonicalTextBlock(text=prefix))

                raw_payload = match.group(1).strip()
                # Try to extract a JSON object from the payload
                payloads = _unpack_json_payload(raw_payload)

                for item_payload in payloads:
                    name, arguments = _normalize_name_and_args_from_json(item_payload)

                    if name or item_payload:
                        candidates.append(RawToolCallCandidate(
                            name=name,
                            raw_arguments=arguments if arguments is not None else item_payload,
                            source_protocol="llama",
                            source_index=block_idx,
                            source_text=item_payload,
                            raw_name=name or "",
                            provenance=_make_provenance("model_output", "llama", name, arguments),
                        ))
                        if name and tool_names and name not in tool_names:
                            diagnostics.append(ExtractionDiagnostic(
                                level="warning",
                                message=f"Unknown tool '{name}' in Llama block {block_idx}",
                                envelope="llama",
                            ))

                last_end = match.end()

            if last_end < len(text):
                suffix = text[last_end:].strip()
                if suffix:
                    remaining.append(CanonicalTextBlock(text=suffix))

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            diagnostics=tuple(diagnostics),
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope="llama",
            ),
        )


# ─── Normalization helpers ──────────────────────────────────────────────────


def _make_provenance(
    source: str,
    dialect: str,
    raw_name: str | None,
    raw_arguments: Any,
) -> Any:
    """Create a ToolCallProvenance from extracted values."""
    from agent_interop.abi import ToolCallProvenance

    return ToolCallProvenance(
        source=source,
        dialect=dialect,
        raw_name=raw_name or "",
        raw_arguments=str(raw_arguments) if not isinstance(raw_arguments, str) else raw_arguments,
    )


def _normalize_name_and_args_from_json(raw_payload: str) -> tuple[str | None, Any]:
    """Extract name and arguments from a JSON tool-call payload.

    Handles multiple JSON shapes ported from profiles.py _normalize_tool_json:
      - {"name": "x", "arguments": {...} or "..."}
      - {"function": "x", "arguments": {...} or "..."}
      - {"function": {"name": "x", "arguments": {...}}}
      - {"tool": "x", "input": {...}}
      - {"name": "x", "parameters": {...}}

    Returns (name, raw_arguments) where raw_arguments preserves the exact
    text if it was originally a JSON value (substring extraction via scanner),
    or the parsed value if already a dict/list.
    """
    import json

    try:
        data = json.loads(raw_payload)
    except (json.JSONDecodeError, ValueError):
        return None, raw_payload

    if not isinstance(data, dict):
        return None, raw_payload

    name: str | None = None
    arguments: Any = raw_payload

    # {"function": {"name": "x", "arguments": {...}}}
    if "function" in data and isinstance(data["function"], dict):
        fn = data["function"]
        name = fn.get("name")
        args_val = fn.get("arguments", {})
        arguments = args_val
        if isinstance(arguments, str):
            # Try to extract the raw arguments substring from the wrapper
            name, arguments = _try_extract_name_and_args(raw_payload)
            if arguments is None:
                arguments = args_val  # fallback
        return name, arguments

    # {"function": "x", "arguments": {...}}
    if "function" in data and isinstance(data.get("function"), str):
        name = data["function"]
        args_val = _get_first_of(data, ("arguments", "parameters", "input"))
        arguments = args_val if args_val is not None else raw_payload
        return name, arguments

    # {"name": "x", ...} — try arguments, parameters, input, params keys
    if "name" in data and isinstance(data["name"], str):
        name = data["name"]
        args_val = _get_first_of(data, ("arguments", "parameters", "input", "params"))
        if args_val is not None:
            arguments = args_val
        else:
            # No args key found — preserve whole payload for repair
            arguments = raw_payload
        return name, arguments

    # {"tool": "x", "input": {...}}  — Qwen style
    if "tool" in data and isinstance(data["tool"], str):
        name = data["tool"]
        args_val = _get_first_of(data, ("input", "arguments", "parameters"))
        arguments = args_val if args_val is not None else raw_payload
        return name, arguments

    return name, arguments


def _get_first_of(d: dict, keys: tuple[str, ...]) -> Any:
    """Return the value of the first key found in d, or None."""
    for k in keys:
        if k in d:
            return d[k]
    return None


def _try_extract_name_and_args(raw_payload: str) -> tuple[str | None, Any]:
    """Try to extract name and raw arguments text using field-span scanner."""
    from agent_interop.parsing.json_scan import BalancedJsonScanner

    scanner = BalancedJsonScanner()
    fields = scanner.extract_field_spans(raw_payload)

    name: str | None = None
    arguments_raw: str | None = None

    for field in fields:
        if field.key == "name" and name is None:
            if field.raw_value.startswith('"') and field.raw_value.endswith('"'):
                import json
                extracted = field.raw_value[1:-1]
                try:
                    name = json.loads(f'"{extracted}"') if extracted else extracted
                except json.JSONDecodeError:
                    name = extracted
        elif field.key in ("arguments", "parameters", "input") and arguments_raw is None:
            arguments_raw = field.raw_value

    return name, arguments_raw


def _unpack_json_payload(raw_payload: str) -> list[str]:
    """Unpack a JSON payload that may be a single object or an array of objects.

    Returns a list of individual JSON object strings.
    """
    import json

    text = raw_payload.strip()
    if not text:
        return []

    # If it starts with [, it's an array — split into individual objects
    if text.startswith("["):
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return [text]
        if isinstance(data, list):
            return [json.dumps(item) if isinstance(item, dict) else str(item) for item in data]
        return [text]

    # Single object
    if text.startswith("{"):
        return [text]

    return [text]


# ─── Generic balanced JSON extractor ───────────────────────────────────────



class GenericBalancedJsonExtractor:
    """Extracts balanced JSON objects that look like tool calls.

    Conservative: only matches when a "name" field is present and the name
    matches a declared tool. Never converts prose into a call.

    Uses BalancedJsonScanner with field-span extraction so raw_arguments
    preserves the exact argument value text (not the wrapper object).
    Fenced code is masked before scanning, same as every other extractor
    — this is an UNFENCED bare-JSON fallback only; a fenced JSON object
    is a different, separately-gated tier (see
    extraction.allow_whole_message_json / WholeMessageJsonExtractor).

    Bare JSON allowed: Yes, but NOT inside a fenced code block
    Name location: JSON "name", "tool", or "function" key
    Arguments location: JSON "arguments", "input", or "parameters" key
    Multiple calls: Yes, via adjacent or separated JSON objects
    Confidence requirement: "name" field must match a declared tool
    False-positive protection: Only matches when name is in tool_names;
        does not create candidates from plain prose without a name
    """

    id = "generic_balanced_json"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        from agent_interop.parsing.json_scan import BalancedJsonScanner

        candidates: list[RawToolCallCandidate] = []
        remaining: list[CanonicalContentBlock] = []

        tool_names = {t.name for t in tools}
        scanner = BalancedJsonScanner()

        for block_idx, block in enumerate(content):
            if not isinstance(block, CanonicalTextBlock) or not block.text:
                remaining.append(block)
                continue

            text = block.text
            # Mask fenced code before scanning — same invariant every other
            # extractor already enforces (a literal example inside a code
            # fence must never become an executable candidate). _mask_fenced_code
            # replaces fence characters with equal-length spaces, so span
            # positions computed against masked_text are still valid
            # positions in the original text; a masked (fenced) region can
            # never contain a balanced {...} span since it's all whitespace.
            masked_text = _mask_fenced_code(text)
            spans = scanner.scan(masked_text)

            if not spans:
                remaining.append(block)
                continue

            # Per-block state to prevent inter-block cross-contamination
            block_candidates: list[RawToolCallCandidate] = []
            consumed_span_indices: set[int] = set()
            object_spans = [s for s in spans if s.kind == "object"]

            for span_idx, span in enumerate(object_spans):
                if span_idx in consumed_span_indices:
                    continue

                raw_json = span.text

                # Try to parse — skip if not valid JSON
                parsed = span.parse()
                if not isinstance(parsed, dict):
                    continue

                # Use normalized extraction to handle multiple JSON shapes
                name, _ = _normalize_name_and_args_from_json(raw_json)

                if not isinstance(name, str) or not name:
                    continue

                if tool_names and name not in tool_names:
                    continue

                # Always extract raw arguments value as a string using field-span scanner
                raw_arguments: Any = raw_json  # fallback: entire wrapper
                fields = scanner.extract_field_spans(raw_json)
                for field in fields:
                    if field.key in ("arguments", "parameters", "input"):
                        raw_arguments = field.raw_value
                        break

                block_candidates.append(RawToolCallCandidate(
                    name=name,
                    raw_arguments=raw_arguments,
                    source_protocol="generic_balanced_json",
                    source_index=block_idx,
                    source_text=raw_json,
                    raw_name=name,
                    provenance=_make_provenance(
                        "model_output", "generic_balanced_json", name, raw_arguments,
                    ),
                ))
                consumed_span_indices.add(span_idx)

            # Build remaining content from non-consumed text regions
            if block_candidates:
                consumed_sorted = sorted(consumed_span_indices)
                last_end = 0
                for span_idx in consumed_sorted:
                    span = object_spans[span_idx]
                    if span.start > last_end:
                        prefix = text[last_end:span.start].strip()
                        if prefix:
                            remaining.append(CanonicalTextBlock(text=prefix))
                    last_end = span.end
                if last_end < len(text):
                    suffix = text[last_end:].strip()
                    if suffix:
                        remaining.append(CanonicalTextBlock(text=suffix))
                candidates.extend(block_candidates)
            else:
                remaining.append(block)

        candidate_names = [c.name for c in candidates if c.name]
        return ExtractionResult(
            candidates=tuple(candidates),
            remaining_content=tuple(remaining),
            confidence=compute_extraction_confidence(
                len(candidates),
                tool_names=tool_names,
                candidate_names=candidate_names,
                envelope=None,  # Generic JSON has no structured envelope
            ),
        )


# ─── Whole-message JSON extractor (profile-gated dialect, not repair) ──────

# The ENTIRE trimmed message must be exactly one closed fence (backtick or
# tilde) with an empty or "json" language tag — anchored at both ends, so
# any prose before/after (including a second fence) fails the match and
# falls through to the stricter "starts with { and ends with }" check below,
# which rejects it too.
#
# The opening delimiter is captured into the "fence" group and
# backreferenced (?P=fence) for the close — NOT a second independent
# ```|~~~ alternation. Two separate alternations would accept a message
# that opens with backticks and closes with tildes (or vice versa) as a
# single valid fence, which markdown never treats as one block.
_WHOLE_MESSAGE_FENCE_RE = re.compile(
    r"^(?P<fence>```|~~~)(?P<lang>[A-Za-z0-9_+-]*)\n(?P<body>.*)\n(?P=fence)$",
    re.DOTALL,
)

# Deliberately narrow: only these three keys are ever accepted at the top
# level. Restricting the shape this tightly (vs. GenericBalancedJsonExtractor's
# broader "name"/"tool"/"function" + "arguments"/"input"/"parameters"/"args"
# key sets) means ordinary JSON data that merely happens to have a "name"
# field is far less likely to be mistaken for a tool call.
_WHOLE_MESSAGE_JSON_ALLOWED_KEYS = frozenset({"name", "arguments", "id", "interop_call_id"})


def _is_nonempty_content_block(block: CanonicalContentBlock) -> bool:
    """True if this block represents real, visible content.

    A CanonicalTextBlock counts only when it has non-whitespace text —
    other block types (image, tool_call, tool_result, reasoning, refusal,
    unknown) always count as "present" regardless of their own internal
    state, since none of them has an "empty" concept comparable to text,
    and their mere presence means the response isn't purely the JSON
    object under consideration.
    """
    if isinstance(block, CanonicalTextBlock):
        return bool(block.text and block.text.strip())
    return True


class WholeMessageJsonExtractor:
    """Recognizes a bare or single-fenced JSON tool-call object when it is
    the model's ENTIRE response.

    This is a distinct, profile-approved output DIALECT (see
    ToolBehaviorProfile.allow_whole_message_json) for models that default
    to plain/fenced JSON instead of the profile's configured tag envelope
    — not a repair of a damaged envelope. envelope_scan.py's tag-defect
    recovery answers "did the model attempt the configured envelope but
    damage or alias its tags?"; this answers "did the model use an
    alternate, profile-approved representation with no envelope at all?"
    Those are different questions and stay in different layers.

    Deliberately much narrower than GenericBalancedJsonExtractor: the JSON
    must be the WHOLE message (no surrounding prose, no other content
    blocks of any kind), only "name"/"arguments"/"id" top-level keys are
    accepted, and the CALLER (ExtractorRegistry.extract) is responsible
    for the cross-cutting policy decisions this extractor's fixed
    protocol signature has no room for: only invoking it when the
    profile opts in, only when tool_choice != none, rejecting a
    named-choice mismatch, and skipping it entirely whenever a native
    tool call is already present (native evidence outranks this weaker
    fallback).

    Bare JSON allowed: Yes, but ONLY as the entire response
    Name location: JSON "name" key only
    Arguments location: JSON "arguments" key only
    Multiple calls: No — a second object anywhere voids the whole match
    Confidence requirement: name must match a declared tool
    False-positive protection: whole-response shape + narrow key set +
        declared-tool-name match + (via the caller) tool_choice gating
    """

    id = "whole_message_json"

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
    ) -> ExtractionResult:
        from agent_interop.parsing.json_scan import BalancedJsonScanner

        _ = envelope  # no envelope concept for bare/fenced JSON

        no_match = ExtractionResult(remaining_content=tuple(content))

        if not tools:
            return no_match

        non_empty = [b for b in content if _is_nonempty_content_block(b)]
        if len(non_empty) != 1 or not isinstance(non_empty[0], CanonicalTextBlock):
            return no_match

        trimmed = non_empty[0].text.strip()
        if not trimmed:
            return no_match

        is_fenced = False
        fence_match = _WHOLE_MESSAGE_FENCE_RE.match(trimmed)
        if fence_match is not None:
            lang = fence_match.group("lang").strip().lower()
            if lang not in ("", "json"):
                return no_match
            inner = fence_match.group("body").strip()
            is_fenced = True
        else:
            inner = trimmed

        if not inner.startswith("{") or not inner.endswith("}"):
            return no_match

        scanner = BalancedJsonScanner()
        object_spans = [s for s in scanner.scan(inner) if s.kind == "object"]
        # Exactly one balanced object, spanning the ENTIRE inner text — no
        # leading/trailing data, no second object anywhere in it.
        if len(object_spans) != 1:
            return no_match
        span = object_spans[0]
        if span.start != 0 or span.end != len(inner):
            return no_match

        parsed = span.parse()
        if not isinstance(parsed, dict):
            # The overall object doesn't parse — but the model may still
            # have unambiguously INTENDED a whole-message tool call with
            # malformed argument JSON (the classic small-model failure:
            # an unescaped quote inside a string value breaks the
            # "arguments" value's own JSON, e.g. malformed_call_repair's
            # scenario in testing/runner.py). Recognizing "whole-message
            # intent" only needs a clean, valid "name" key naming a
            # declared tool plus a separate "arguments" key being
            # present — it does NOT require the "arguments" VALUE itself
            # to parse. extract_field_spans() scans structurally without
            # requiring overall JSON validity, so it can still locate
            # "name" and hand back "arguments"'s raw (possibly malformed)
            # text for the repair pipeline to attempt bounded recovery
            # on, instead of this extractor discarding an unambiguous
            # whole-message call outright just because one field failed
            # to parse.
            return self._recover_with_malformed_arguments(inner, scanner, tools, no_match)
        if not set(parsed.keys()) <= _WHOLE_MESSAGE_JSON_ALLOWED_KEYS:
            return no_match

        name = parsed.get("name")
        if not isinstance(name, str) or not name:
            return no_match

        tool_names = {t.name for t in tools}
        if name not in tool_names:
            return no_match

        if "arguments" not in parsed:
            return no_match

        raw_arguments: Any = inner
        for field in scanner.extract_field_spans(inner):
            if field.key == "arguments":
                raw_arguments = field.raw_value
                break

        # "id" is an accepted top-level key (_WHOLE_MESSAGE_JSON_ALLOWED_KEYS)
        # but was previously parsed and then silently discarded — any model
        # that emits one for correlation (or for downstream dedup, see
        # Gateway._dedup_tool_candidates) never had it survive extraction.
        raw_id = parsed.get("id")
        raw_nonce = parsed.get("interop_call_id")
        candidate = RawToolCallCandidate(
            id=raw_id if isinstance(raw_id, str) and raw_id else None,
            name=name,
            raw_arguments=raw_arguments,
            source_protocol="whole_message_json",
            source_index=0,
            source_text=inner,
            raw_name=name,
            provenance=_make_provenance(
                "model_output", "whole_message_json", name, raw_arguments,
            ),
            execution_nonce=raw_nonce if isinstance(raw_nonce, str) and raw_nonce else None,
        )
        diagnostics = (ExtractionDiagnostic(
            level="warning",
            message=(
                f"Recovered a whole-message {'fenced ' if is_fenced else ''}"
                "JSON tool call — the entire response was one JSON object "
                "with no surrounding content (profile-approved "
                "whole_message_json dialect, not envelope repair)"
            ),
            envelope="whole_message_json",
        ),)

        return ExtractionResult(
            candidates=(candidate,),
            remaining_content=(),
            diagnostics=diagnostics,
            confidence=compute_extraction_confidence(
                1, tool_names=tool_names, candidate_names=[name],
                envelope=None, from_fallback=True,
            ),
        )

    @staticmethod
    def _recover_with_malformed_arguments(
        inner: str,
        scanner: Any,  # interop.parsing.json_scan.BalancedJsonScanner — kept local-imported (see extract())
        tools: Sequence[CanonicalTool],
        no_match: ExtractionResult,
    ) -> ExtractionResult:
        """Bounded recovery when the outer whole-message object fails to
        parse as JSON, but the message still unambiguously declares a
        whole-message tool call: a clean, valid ``"name"`` key naming a
        declared tool, plus a separate ``"arguments"`` key — regardless of
        whether the "arguments" VALUE itself is valid JSON.

        Returns the same ``no_match`` a caller would get from the strict
        path when even this bounded recognition fails (e.g. no declared
        tool name found), so callers don't need two different empty
        results to check for.
        """
        field_spans = scanner.extract_field_spans(inner)
        if not field_spans:
            return no_match
        if not {f.key for f in field_spans} <= _WHOLE_MESSAGE_JSON_ALLOWED_KEYS:
            return no_match

        span_by_key = {f.key: f for f in field_spans}
        name_field = span_by_key.get("name")
        args_field = span_by_key.get("arguments")
        if name_field is None or args_field is None:
            return no_match

        try:
            # The "name" key itself must still parse cleanly — if even the
            # tool name is ambiguous, there's nothing safe to recover.
            recovered_name = json.loads(name_field.raw_value)
        except (json.JSONDecodeError, ValueError):
            return no_match
        if not isinstance(recovered_name, str) or not recovered_name:
            return no_match

        tool_names = {t.name for t in tools}
        if recovered_name not in tool_names:
            return no_match

        recovered_id: str | None = None
        id_field = span_by_key.get("id")
        if id_field is not None:
            try:
                parsed_id = json.loads(id_field.raw_value)
                recovered_id = parsed_id if isinstance(parsed_id, str) and parsed_id else None
            except (json.JSONDecodeError, ValueError):
                recovered_id = None

        recovered_nonce: str | None = None
        nonce_field = span_by_key.get("interop_call_id")
        if nonce_field is not None:
            try:
                parsed_nonce = json.loads(nonce_field.raw_value)
                recovered_nonce = parsed_nonce if isinstance(parsed_nonce, str) and parsed_nonce else None
            except (json.JSONDecodeError, ValueError):
                recovered_nonce = None

        raw_arguments = args_field.raw_value
        candidate = RawToolCallCandidate(
            id=recovered_id,
            name=recovered_name,
            raw_arguments=raw_arguments,
            source_protocol="whole_message_json",
            source_index=0,
            source_text=inner,
            raw_name=recovered_name,
            provenance=_make_provenance(
                "model_output", "whole_message_json", recovered_name, raw_arguments,
            ),
            execution_nonce=recovered_nonce,
        )
        diagnostics = (ExtractionDiagnostic(
            level="warning",
            message=(
                "Recovered a whole-message JSON tool call whose 'arguments' "
                "value did not parse as valid JSON on its own — the raw "
                "argument text is passed through for the repair pipeline "
                "to attempt bounded recovery, rather than discarding an "
                "unambiguous whole-message call outright because one field "
                "failed to parse"
            ),
            envelope="whole_message_json",
        ),)

        return ExtractionResult(
            candidates=(candidate,),
            remaining_content=(),
            diagnostics=diagnostics,
            confidence=compute_extraction_confidence(
                1, tool_names=tool_names, candidate_names=[recovered_name],
                envelope=None, from_fallback=True,
            ),
        )


# ─── Extractor registry ────────────────────────────────────────────────────


class ExtractorRegistry:
    """Registry of tool candidate extractors keyed by extractor ID."""

    def __init__(self) -> None:
        self._extractors: dict[str, ToolCandidateExtractor] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(ToolCallEnvelopeExtractor())
        self.register(NativeStructuredExtractor())
        self.register(HermesExtractor())
        self.register(QwenExtractor())
        self.register(MistralExtractor())
        self.register(DeepSeekExtractor())
        self.register(LlamaExtractor())
        self.register(GenericBalancedJsonExtractor())
        self.register(WholeMessageJsonExtractor())

    def register(self, extractor: ToolCandidateExtractor) -> None:
        self._extractors[extractor.id] = extractor

    def get(self, extractor_id: str) -> ToolCandidateExtractor:
        if extractor_id not in self._extractors:
            raise ValueError(
                f"Unknown extractor: '{extractor_id}'. "
                f"Available: {sorted(self._extractors)}"
            )
        return self._extractors[extractor_id]

    def extract(
        self,
        content: Sequence[CanonicalContentBlock],
        *,
        extractor_id: str,
        tools: Sequence[CanonicalTool],
        envelope: str | None,
        fallback_strategies: Sequence[Any] = (),
        tool_choice: CanonicalToolChoice | None = None,
        native_candidates_present: bool = False,
        expected_execution_nonce: str | None = None,
    ) -> ExtractionResult:
        """Run the primary extractor, then each profile-configured
        fallback strategy in order until one finds a candidate.

        ``fallback_strategies`` generalizes what were two independent,
        hardcoded fallback tiers (a "bare JSON anywhere in the text" tier
        and a "the whole message is one JSON object" tier) into a single
        ordered mechanism: each entry (model.profiles_v2.ExtractionStrategy)
        just names a registered extractor id and the conditions under
        which it's eligible to run. A new fallback shape a model needs no
        longer requires its own bespoke profile field and call-site
        change — it's another entry in the profile's list.

        Every fallback strategy is gated uniformly:
          - only when nothing was found yet (primary, or an earlier
            strategy in the list)
          - never under a tool_choice mode the strategy didn't opt into
            (the ExtractionStrategy default excludes "none" — a fallback
            recovering a call the client explicitly said not to use tools
            for is almost certainly wrong)
          - skipped once native candidates are already present, unless the
            strategy explicitly opts out of that (skip_when_native_present):
            native evidence outranks a weaker textual fallback, and bare
            textual JSON alongside a native call is most likely an echo
            of it — unlike an explicit configured envelope, which Interop
            deliberately does treat as a possible distinct hybrid call
            (see _extract_tool_candidates's docstring)
          - under tool_choice=named, a recovered call for a DIFFERENT tool
            than the one requested is not silently substituted
        """
        extractor = self.get(extractor_id)
        result = extractor.extract(content, tools=tools, envelope=envelope)

        # native_structured never falls through to any text-scanning
        # strategy: it means "tool calls come from the codec directly,
        # there is no text to scan" — an empty result from it says "no
        # native call was made", not "extraction failed, try text now".
        # Falling back from it would let ordinary prose be reinterpreted
        # as a tool call on every native-mode response that happens to
        # discuss JSON.
        if extractor_id == "native_structured":
            return result

        effective_tool_choice = tool_choice or CanonicalToolChoice()
        mode_name = (
            effective_tool_choice.mode.value
            if hasattr(effective_tool_choice.mode, "value")
            else str(effective_tool_choice.mode)
        )

        for strategy in fallback_strategies:
            if result.candidates:
                break
            if strategy.parser_id == extractor_id:
                continue  # don't fall back to the primary parser itself
            if strategy.skip_when_native_present and native_candidates_present:
                continue
            if mode_name not in strategy.allowed_tool_choice_modes:
                continue

            fallback_extractor = self._extractors.get(strategy.parser_id)
            if fallback_extractor is None:
                continue
            fallback = fallback_extractor.extract(content, tools=tools, envelope=None)
            if not fallback.candidates:
                continue
            if (
                effective_tool_choice.mode == ToolChoiceMode.NAMED
                and fallback.candidates[0].name != effective_tool_choice.name
            ):
                continue

            # Ambiguous-auto guard: whole_message_json recovers a bare/fenced
            # JSON object with no taught envelope — under tool_choice=auto
            # that shape is indistinguishable from demonstration content
            # (see the P0-3 audit finding). This branch is only reachable at
            # all when a project/user-tier profile override explicitly
            # re-enabled "auto" for this dialect (builtin profiles are
            # rejected at load time — see profiles_v2.py). Even then, only a
            # candidate carrying the exact live per-request nonce
            # build_invocation_plan() issued is trusted.
            is_ambiguous_auto = strategy.parser_id == "whole_message_json" and mode_name == "auto"
            if is_ambiguous_auto:
                candidate_nonce = getattr(fallback.candidates[0], "execution_nonce", None)
                if not expected_execution_nonce or candidate_nonce != expected_execution_nonce:
                    result = ExtractionResult(
                        candidates=(),
                        remaining_content=result.remaining_content,
                        diagnostics=result.diagnostics + (ExtractionDiagnostic(
                            level="warning",
                            message=(
                                "Discarded a bare/fenced JSON candidate under "
                                "tool_choice=auto: whole_message_json ambiguous-auto "
                                "recovery requires a matching per-request execution "
                                "nonce, which was missing or did not match"
                            ),
                            envelope="whole_message_json",
                        ),),
                    )
                    continue

            # Apply confidence penalty for fallback extraction (item 60):
            # any fallback tier is less certain than a primary dialect match.
            result = ExtractionResult(
                candidates=fallback.candidates,
                remaining_content=fallback.remaining_content,
                diagnostics=fallback.diagnostics + (ExtractionDiagnostic(
                    level="warning",
                    message=f"Primary extractor '{extractor_id}' found no candidates; "
                            f"used '{strategy.parser_id}' fallback strategy "
                            f"(confidence penalty applied)"
                            + (
                                " [ambiguous-intent recovery, nonce-verified]"
                                if is_ambiguous_auto else ""
                            ),
                    envelope=strategy.parser_id,
                ),),
                reasoning_content_remainder=fallback.reasoning_content_remainder,
                consumed_spans=fallback.consumed_spans,
                confidence=fallback.confidence * 0.7,
            )

        return result


# ─── Module-level default registry ─────────────────────────────────────────

_default_registry = ExtractorRegistry()


def get_default_registry() -> ExtractorRegistry:
    return _default_registry
