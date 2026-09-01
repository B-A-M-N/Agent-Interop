import re

_TOOL_CALL_RE = re.compile(r'<tool_call\b[^>]*>(.*?)<\/tool_call>', re.DOTALL | re.IGNORECASE)

# Test with proper closing tags
text = (
    '<tool_call>{"name":"test_tool","arguments":{"key":"value"}}'
    '<tool_call>{"name":"test_tool","arguments":{"key":"other"}}'
)
print('Text:', repr(text))

matches = list(_TOOL_CALL_RE.finditer(text))
print(f'Found {len(matches)} matches')
for i, m in enumerate(matches):
    print(f'Match {i}: start={m.start()}, end={m.end()}, group(1)={m.group(1)}')

# Now test without closing tags (current test format)
text2 = (
    '{"name":"test_tool","arguments":{"key":"value"}}'
    '{"name":"test_tool","arguments":{"key":"other"}}'
)
print('\nText2 (no closing tags):', repr(text2))

matches2 = list(_TOOL_CALL_RE.finditer(text2))
print(f'Found {len(matches2)} matches')
for i, m in enumerate(matches2):
    print(f'Match {i}: start={m.start()}, end={m.end()}, group(1)={m.group(1)}')