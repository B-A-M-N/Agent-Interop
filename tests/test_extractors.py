"""Tests for dialect-specific extractors (P0 #3)."""

from agent_interop.abi import CanonicalContentBlock, CanonicalTextBlock, CanonicalTool
from agent_interop.extraction import (
    DeepSeekExtractor,
    GenericBalancedJsonExtractor,
    HermesExtractor,
    LlamaExtractor,
    MistralExtractor,
    QwenExtractor,
    ToolCallEnvelopeExtractor,
)


def _make_tool(name: str, props: dict | None = None) -> CanonicalTool:
    return CanonicalTool(
        name=name,
        description=f"Tool {name}",
        input_schema=props or {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    )


_READ_FILE_TOOL = _make_tool("read_file")
_SEARCH_TOOL = _make_tool("search")
_WRITE_FILE_TOOL = _make_tool("write_file")


# ─── Hermes: <tool_call>JSON</tool_call> ────────────────────────────────────


def test_hermes_extracts_simple_call():
    extractor = HermesExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


def test_hermes_preserves_surrounding_text():
    extractor = HermesExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='Before <tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call> After'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    remaining_texts = [b.text for b in result.remaining_content if hasattr(b, 'text') and b.text]
    assert any("Before" in t for t in remaining_texts)
    assert any("After" in t for t in remaining_texts)


def test_hermes_does_not_match_no_tags():
    extractor = HermesExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


def test_hermes_multiple_calls():
    extractor = HermesExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='First <tool_call>{"name": "read_file", "arguments": {"path": "/a"}}</tool_call>'),
        CanonicalTextBlock(text='Second <tool_call>{"name": "search", "arguments": {"query": "x"}}</tool_call>'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL, _SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 2


# ─── Qwen: <tool>JSON</tool> ──────────────────────────────────────────────


def test_qwen_extracts_simple_call():
    extractor = QwenExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='<tool>\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n</tool>'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


def test_qwen_not_tool_call_tag():
    """Qwen uses <tool> not <tool_call>. Ensure <tool_call> is NOT detected by QwenExtractor."""
    extractor = QwenExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x"}}</tool_call>'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


def test_qwen_single_line_tag():
    extractor = QwenExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='<tool>{"name": "search", "arguments": {"query": "hello"}}</tool>'),
    ]
    result = extractor.extract(blocks, tools=[_SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "search"


def test_qwen_preserves_surrounding_text():
    extractor = QwenExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='Let me check.<tool>\n{"name": "read_file", "arguments": {"path": "/x"}}\n</tool>Done.'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    remaining_texts = [b.text for b in result.remaining_content if hasattr(b, 'text') and b.text]
    assert any("Let me check" in t for t in remaining_texts)
    assert any("Done" in t for t in remaining_texts)


# ─── Mistral: [TOOL_CALLS]{...} ────────────────────────────────────────────


def test_mistral_extracts_single_call():
    extractor = MistralExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='[TOOL_CALLS]{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


def test_mistral_extracts_array_calls():
    extractor = MistralExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='[TOOL_CALLS][{"name": "read_file", "arguments": {"path": "/a"}}, {"name": "search", "arguments": {"query": "x"}}]'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL, _SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 2


def test_mistral_requires_prefix():
    extractor = MistralExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


def test_mistral_no_false_positive_from_prose():
    extractor = MistralExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='I think [TOOL_CALLS] is not something we need here'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


# ─── DeepSeek: \x14{...}\x14 ────────────────────────────────────────────────


def test_deepseek_extracts_simple_call():
    extractor = DeepSeekExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='\x14{"name": "read_file", "arguments": {"path": "/tmp/x"}}\x14'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


def test_deepseek_multiple_calls():
    extractor = DeepSeekExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='\x14{"name": "read_file", "arguments": {"path": "/a"}}\x14\x14{"name": "search", "arguments": {"query": "x"}}\x14'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL, _SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 2


def test_deepseek_preserves_surrounding_text():
    extractor = DeepSeekExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='Let me look.\x14{"name": "read_file", "arguments": {"path": "/x"}}\x14There.'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    remaining_texts = [b.text for b in result.remaining_content if hasattr(b, 'text') and b.text]
    assert any("Let me look" in t for t in remaining_texts)
    assert any("There" in t for t in remaining_texts)


def test_deepseek_no_false_positive_without_markers():
    extractor = DeepSeekExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


# ─── Llama: <|python_tag|>{...} ──────────────────────────────────────────────


def test_llama_extracts_simple_call():
    extractor = LlamaExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='<|python_tag|>{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


def test_llama_multiple_calls():
    extractor = LlamaExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='<|python_tag|>{"name": "read_file", "arguments": {"path": "/a"}}'),
        CanonicalTextBlock(text='<|python_tag|>{"name": "search", "arguments": {"query": "x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL, _SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 2


def test_llama_requires_tag():
    extractor = LlamaExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


def test_llama_no_false_positive():
    extractor = LlamaExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='Some text with no tool calls'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


# ─── GenericBalancedJsonExtractor ────────────────────────────────────────────


def test_generic_balanced_extracts_by_name():
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


def test_generic_balanced_raw_arguments_is_substring():
    """raw_arguments should be the arguments value, not the entire wrapper."""
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    raw = result.candidates[0].raw_arguments
    assert isinstance(raw, str)
    assert raw == '{"path": "/tmp/x"}'


def test_generic_balanced_does_not_match_unknown_tool():
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "unknown_tool", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


def test_generic_balanced_multiple_calls():
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/a"}} {"name": "search", "arguments": {"query": "x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL, _SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 2


def test_generic_balanced_per_block_state():
    """A candidate in one block should not suppress spans in a later block."""
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/a"}}'),
        CanonicalTextBlock(text='{"name": "read_file", "arguments": {"path": "/b"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 2


def test_generic_balanced_no_false_positive_plain_text():
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='Hello, how are you?'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 0


def test_generic_balanced_function_key():
    """Should handle {"function": "read_file", "arguments": {...}}"""
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"function": "read_file", "arguments": {"path": "/tmp/x"}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"
    raw = result.candidates[0].raw_arguments
    assert isinstance(raw, str)
    assert "path" in raw


def test_generic_balanced_tool_key():
    """Should handle {"tool": "search", "input": {"query": "hello"}}"""
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"tool": "search", "input": {"query": "hello"}}'),
    ]
    result = extractor.extract(blocks, tools=[_SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "search"


def test_generic_balanced_parameters_key():
    """Should handle {"name": "search", "parameters": {"query": "hello"}}"""
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"name": "search", "parameters": {"query": "hello"}}'),
    ]
    result = extractor.extract(blocks, tools=[_SEARCH_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "search"


def test_generic_balanced_nested_function_object():
    """Should handle {"function": {"name": "read_file", "arguments": {"path": "/x"}}}"""
    extractor = GenericBalancedJsonExtractor()
    blocks: list[CanonicalContentBlock] = [
        CanonicalTextBlock(text='{"function": {"name": "read_file", "arguments": {"path": "/x"}}}'),
    ]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "read_file"


# ─── EnvelopeExtractor ────────────────────────────────────────────────────────


def test_envelope_extractor_malformed_arguments_trailing_comma():
    """Trailing comma inside arguments must be preserved as raw_arguments."""
    extractor = ToolCallEnvelopeExtractor()
    text = '<tool_call>{"name":"read_file","arguments":{"path":"x",}}</tool_call>'
    blocks: list[CanonicalContentBlock] = [CanonicalTextBlock(text=text)]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    raw = result.candidates[0].raw_arguments
    assert isinstance(raw, str)
    assert raw == '{"path":"x",}'


def test_envelope_extractor_accepts_empty_xml_attribute_call_from_llama():
    extractor = ToolCallEnvelopeExtractor()
    workflow = _make_tool(
        "Workflow",
        {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "string"}}},
    )
    result = extractor.extract(
        [CanonicalTextBlock(
            text='<tool_call name="Workflow" arguments={"key":"read_file","value":"/tmp/x"}></tool_call>',
        )],
        tools=[workflow],
        envelope="tool_call",
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].name == "Workflow"
    assert result.candidates[0].raw_arguments == '{"key":"read_file","value":"/tmp/x"}'


def test_envelope_extractor_malformed_wrapper_valid_args():
    """Wrapper malformed but arguments recoverable."""
    extractor = ToolCallEnvelopeExtractor()
    text = '<tool_call>{"name":"read_file","arguments":{"path":"x"}</tool_call>'  # missing closing }
    blocks: list[CanonicalContentBlock] = [CanonicalTextBlock(text=text)]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) >= 1


def test_envelope_extractor_raw_arguments_is_substring():
    """raw_arguments should be the arguments value ONLY, not entire wrapper."""
    extractor = ToolCallEnvelopeExtractor()
    text = '<tool_call>{"name":"read_file","arguments":{"path":"x"}}</tool_call>'
    blocks: list[CanonicalContentBlock] = [CanonicalTextBlock(text=text)]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    raw = result.candidates[0].raw_arguments
    assert isinstance(raw, str)
    assert raw == '{"path":"x"}'
    # Must NOT include the outer wrapper's closing brace
    assert raw.count("}") == 1


def test_envelope_extractor_arguments_with_nested_objects():
    extractor = ToolCallEnvelopeExtractor()
    text = '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x","nested":{"a":1,"b":[2,3]}}}</tool_call>'
    blocks: list[CanonicalContentBlock] = [CanonicalTextBlock(text=text)]
    result = extractor.extract(blocks, tools=[_READ_FILE_TOOL], envelope=None)
    assert len(result.candidates) == 1
    raw = result.candidates[0].raw_arguments
    assert isinstance(raw, str)
    assert raw.startswith('{"path"')
    assert raw.endswith('[2,3]}}}')
