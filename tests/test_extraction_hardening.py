"""Tests for Phase 2 extraction hardening: confidence, envelope rejection, raw evidence."""

from __future__ import annotations

from agent_interop.abi import CanonicalTextBlock
from agent_interop.extraction import (
    GenericBalancedJsonExtractor,
    ToolCallEnvelopeExtractor,
    compute_extraction_confidence,
    get_default_registry,
)

# ─── Confidence calculation (item 58) ──────────────────────────────────────


class TestExtractionConfidence:
    def test_known_tool_in_envelope(self):
        conf = compute_extraction_confidence(
            1, tool_names={"read_file", "edit_file"},
            candidate_names=["read_file"], envelope="tool_call",
        )
        assert conf == 0.95

    def test_unknown_tool(self):
        conf = compute_extraction_confidence(
            1, tool_names={"read_file"},
            candidate_names=["unknown_tool"], envelope="tool_call",
        )
        assert conf == 0.4

    def test_no_candidates(self):
        conf = compute_extraction_confidence(
            0, tool_names={"read_file"},
            candidate_names=[], envelope="tool_call",
        )
        assert conf == 1.0

    def test_generic_envelope_reduces_confidence(self):
        conf = compute_extraction_confidence(
            1, tool_names={"read_file"},
            candidate_names=["read_file"], envelope=None,
        )
        assert conf == 0.7

    def test_fallback_penalty(self):
        conf = compute_extraction_confidence(
            1, tool_names={"read_file"},
            candidate_names=["read_file"], envelope="tool_call",
            from_fallback=True,
        )
        assert conf < 0.95
        assert conf >= 0.6

    def test_multiple_candidates_min_confidence(self):
        conf = compute_extraction_confidence(
            2, tool_names={"read_file"},
            candidate_names=["read_file", "nonexistent"], envelope="tool_call",
        )
        assert conf == 0.4


# ─── All extractors use computed confidence (item 58) ───────────────────────


class TestAllExtractorsUseComputedConfidence:
    """Every extractor must use compute_extraction_confidence, not hardcode 0.95."""

    def test_qwen_extractor_computes_confidence(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.extraction import QwenExtractor
        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        content = [CanonicalTextBlock(text='<tool>{"name": "search", "arguments": {"q": "test"}}</tool>')]
        result = QwenExtractor().extract(content, tools=[tool], envelope="qwen")
        assert result.confidence == 0.95

    def test_mistral_extractor_computes_confidence(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.extraction import MistralExtractor
        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        content = [CanonicalTextBlock(text='[TOOL_CALLS][{"name": "search", "arguments": {"q": "test"}}]')]
        result = MistralExtractor().extract(content, tools=[tool], envelope="mistral")
        assert result.confidence == 0.95

    def test_deepseek_extractor_computes_confidence(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.extraction import DeepSeekExtractor
        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        content = [CanonicalTextBlock(text='\x14{"name": "search", "arguments": {"q": "test"}}\x14')]
        result = DeepSeekExtractor().extract(content, tools=[tool], envelope="deepseek")
        assert result.confidence == 0.95

    def test_llama_extractor_computes_confidence(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.extraction import LlamaExtractor
        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        content = [CanonicalTextBlock(text='<|python_tag|>{"name": "search", "arguments": {"q": "test"}}')]
        result = LlamaExtractor().extract(content, tools=[tool], envelope="llama")
        assert result.confidence == 0.95

    def test_generic_extractor_has_lower_confidence(self):
        """Generic balanced JSON extractor should have lower confidence (no envelope)."""
        from agent_interop.abi import CanonicalTool
        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        content = [CanonicalTextBlock(text='{"name": "search", "arguments": {"q": "test"}}')]
        result = GenericBalancedJsonExtractor().extract(content, tools=[tool], envelope=None)
        assert result.confidence == 0.7

    def test_unknown_tool_reduces_confidence(self):
        from agent_interop.abi import CanonicalTool
        from agent_interop.extraction import QwenExtractor
        tool = CanonicalTool(name="search", description="", input_schema={"type": "object"})
        content = [CanonicalTextBlock(text='<tool>{"name": "unknown_tool", "arguments": {}}</tool>')]
        result = QwenExtractor().extract(content, tools=[tool], envelope="qwen")
        assert result.confidence == 0.4


# ─── Incomplete envelope rejection (item 55) ────────────────────────────────


class TestEnvelopeRejection:
    def test_rejects_unclosed_envelope(self):
        extractor = ToolCallEnvelopeExtractor()
        content = [CanonicalTextBlock(text='Here is the call:\n<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"')]
        result = extractor.extract(content, tools=[], envelope="tool_call")
        assert len(result.candidates) == 0
        assert any(d.level == "error" for d in result.diagnostics)
        assert any("unclosed" in d.message.lower() or "incomplete" in d.message.lower() for d in result.diagnostics)

    def test_accepts_closed_envelope(self):
        extractor = ToolCallEnvelopeExtractor()
        content = [CanonicalTextBlock(text='<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>')]
        result = extractor.extract(content, tools=[], envelope="tool_call")
        assert len(result.candidates) == 1
        assert result.candidates[0].name == "read_file"

    def test_accepts_closed_envelope_with_tools(self):
        from agent_interop.abi import CanonicalTool
        extractor = ToolCallEnvelopeExtractor()
        tool = CanonicalTool(
            name="read_file",
            description="Read a file",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
        content = [CanonicalTextBlock(text='<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>')]
        result = extractor.extract(content, tools=[tool], envelope="tool_call")
        assert len(result.candidates) == 1


# ─── Generic fallback confidence penalty (item 60) ─────────────────────────


class TestGenericFallbackPenalty:
    def test_fallback_gets_confidence_penalty(self):
        registry = get_default_registry()
        from agent_interop.abi import CanonicalTool
        tool = CanonicalTool(
            name="search",
            description="Search",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        from agent_interop.model.profiles_v2 import ExtractionStrategy

        content = [CanonicalTextBlock(text='{"name": "search", "arguments": {"q": "test"}}')]
        result = registry.extract(
            content, extractor_id="qwen", tools=[tool], envelope="qwen",
            fallback_strategies=[ExtractionStrategy(parser_id="generic_balanced_json")],
        )
        if result.candidates:
            assert result.confidence < 0.95


class TestGenericFallbackDisabledByDefault:
    """MVP-07: bare JSON extraction must be opt-in, not opt-out.

    Without an explicit allow_generic_fallback=True, ordinary JSON in model
    output (config examples, quoted data, documentation) must never be
    reinterpreted as a tool call.
    """

    TOOL = None  # set in setup

    def _tool(self):
        from agent_interop.abi import CanonicalTool
        return CanonicalTool(
            name="edit_file",
            description="Edit a file",
            input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        )

    def _extract(self, text: str):
        registry = get_default_registry()
        content = [CanonicalTextBlock(text=text)]
        return registry.extract(
            content, extractor_id="qwen", tools=[self._tool()], envelope="qwen",
        )

    def test_default_is_no_fallback(self):
        registry = get_default_registry()
        import inspect
        sig = inspect.signature(registry.extract)
        assert sig.parameters["fallback_strategies"].default == ()

    def test_json_in_fenced_block_not_extracted(self):
        text = (
            "Here is an example:\n"
            "```json\n"
            '{"name": "edit_file", "arguments": {"path": "a.py"}}\n'
            "```\n"
        )
        result = self._extract(text)
        assert result.candidates == ()

    def test_json_config_example_not_extracted(self):
        text = 'Set this in your config: {"name": "edit_file", "arguments": {"path": "a.py"}}'
        result = self._extract(text)
        assert result.candidates == ()

    def test_user_quoted_tool_shaped_json_not_extracted(self):
        text = 'The user asked me to quote: \'{"name": "edit_file", "arguments": {"path": "a.py"}}\''
        result = self._extract(text)
        assert result.candidates == ()

    def test_json_returned_as_data_not_extracted(self):
        text = 'The API returned: {"name": "edit_file", "arguments": {"path": "a.py"}, "status": "archived"}'
        result = self._extract(text)
        assert result.candidates == ()

    def test_explanatory_prose_followed_by_json_not_extracted(self):
        text = (
            "I looked at the file and here's what its metadata looks like as JSON:\n"
            '{"name": "edit_file", "arguments": {"path": "a.py"}}'
        )
        result = self._extract(text)
        assert result.candidates == ()


# ─── Raw evidence preservation (item 59) ───────────────────────────────────


class TestRawEvidencePreservation:
    def test_generic_extract_preserves_raw_json(self):
        from agent_interop.abi import CanonicalTool
        tool = CanonicalTool(
            name="echo",
            description="Echo",
            input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
        )
        content = [CanonicalTextBlock(text='{"name": "echo", "arguments": {"msg": "hello"}}')]
        extractor = GenericBalancedJsonExtractor()
        result = extractor.extract(content, tools=[tool], envelope=None)
        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.raw_arguments is not None
        assert "hello" in str(candidate.raw_arguments)
        assert "echo" in candidate.source_text

    def test_field_spans_extract_value_verbatim(self):
        from agent_interop.parsing.json_scan import BalancedJsonScanner
        scanner = BalancedJsonScanner()
        text = '{"name": "test", "arguments": "{malformed json"}'
        fields = scanner.extract_field_spans(text)
        arg_fields = [f for f in fields if f.key == "arguments"]
        assert len(arg_fields) == 1
        assert "{malformed json" in arg_fields[0].raw_value


# ─── Fenced-code masking (item 57) ─────────────────────────────────────────


class TestFencedCodeMasking:
    def test_fenced_code_not_extracted(self):
        """Tool calls inside fenced code blocks must not be extracted."""
        extractor = ToolCallEnvelopeExtractor()
        content = [CanonicalTextBlock(text='Example:\n```\n<tool_call>{"name": "hack", "arguments": {}}\n```\n')]
        result = extractor.extract(content, tools=[], envelope="tool_call")
        assert len(result.candidates) == 0

    def test_real_call_outside_fence_extracted(self):
        """Tool calls outside fenced code should still be extracted."""
        extractor = ToolCallEnvelopeExtractor()
        content = [CanonicalTextBlock(text='Run this:\n<tool_call>{"name": "read", "arguments": {"p": "/x"}}</tool_call>\n')]
        result = extractor.extract(content, tools=[], envelope="tool_call")
        assert len(result.candidates) == 1
