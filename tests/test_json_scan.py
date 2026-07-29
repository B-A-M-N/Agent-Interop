"""Tests for the balanced JSON scanner."""
from agent_interop.parsing.json_scan import BalancedJsonScanner


def test_balanced_object():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"name": "read_file", "arguments": {"path": "/tmp/x"}}')
    assert len(spans) == 1
    assert spans[0].kind == "object"


def test_balanced_object_nested():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"name": "read_file", "arguments": {"path": "/tmp/x", "nested": {"a": 1}}}')
    assert len(spans) == 1
    assert spans[0].kind == "object"


def test_balanced_object_with_string_braces():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"name": "read_file", "text": "hello {world}"}')
    assert len(spans) == 1
    assert spans[0].kind == "object"


def test_balanced_object_escaped_quotes():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"name": "read_file", "text": "he said \\"hello\\""}')
    assert len(spans) == 1
    assert spans[0].kind == "object"


def test_balanced_array():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('[1, 2, 3]')
    assert len(spans) == 1
    assert spans[0].kind == "array"


def test_multiple_adjacent_objects():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"a":1}{"b":2}')
    assert len(spans) == 2


def test_truncated_object():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"name": "read_file", "arguments": {"path":')
    assert len(spans) >= 1


def test_extract_tool_calls():
    candidates = BalancedJsonScanner.extract_tool_calls(
        'Here is the result: {"name": "read_file", "arguments": {"path": "/tmp/x"}}'
    )
    assert len(candidates) == 1
    assert candidates[0].name == "read_file"
    assert candidates[0].raw_arguments == {"path": "/tmp/x"}


def test_extract_tool_calls_multiple():
    candidates = BalancedJsonScanner.extract_tool_calls(
        '{"name": "read_file", "input": {"path": "/a"}} and {"name": "write", "arguments": {"path": "/b"}}'
    )
    assert len(candidates) >= 2


def test_extract_no_tool_calls_plain_text():
    candidates = BalancedJsonScanner.extract_tool_calls("Hello, how are you?")
    assert len(candidates) == 0


def test_extract_no_tool_calls_missing_keys():
    candidates = BalancedJsonScanner.extract_tool_calls('{"key": "value", "foo": "bar"}')
    assert len(candidates) == 0


def test_classify_tool_call_function_key():
    candidates = BalancedJsonScanner.extract_tool_calls(
        '{"function": "read_file", "arguments": {"path": "/x"}}'
    )
    assert len(candidates) == 1
    assert candidates[0].name == "read_file"


def test_classify_tool_call_parameters_key():
    candidates = BalancedJsonScanner.extract_tool_calls(
        '{"tool": "search", "parameters": {"query": "hello"}}'
    )
    assert len(candidates) == 1
    assert candidates[0].name == "search"


def test_balanced_object_array_inside():
    scanner = BalancedJsonScanner()
    spans = scanner.scan('{"items": [1, 2, {"nested": 3}]}')
    assert len(spans) == 1
    result = spans[0].parse()
    assert result is not None
    assert result["items"] == [1, 2, {"nested": 3}]