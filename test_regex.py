import re

_TOOL_CALL_RE = re.compile(r'<tool_call\b[^>]*>(.*?)<\/tool_call>', re.DOTALL | re.IGNORECASE)

# Test with actual Unicode character from the test file
text = '<tool_call>{"name":"test_tool","arguments":{"key":"value"}}'
print('Text:', repr(text))

matches = list(_TOOL_CALL_RE.finditer(text))
print(f'Found {len(matches)} matches')
for i, m in enumerate(matches):
    print(f'Match {i}: start={m.start()}, end={m.end()}, group(1)={m.group(1)}')

# Now test with two envelopes
text2 = (
    ''
    ''
)
print('\nText2:', repr(text2))

matches2 = list(_TOOL_CALL_RE.finditer(text2))
print(f'Found {len(matches2)} matches')
for i, m in enumerate(matches2):
    print(f'Match {i}: start={m.start()}, end={m.end()}, group(1)={m.group(1)}')