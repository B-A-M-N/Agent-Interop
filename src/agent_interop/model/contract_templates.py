"""Contract-template registry — the rendered instruction text a PROMPTED-
mode model receives describing how to call tools.

Different model families respond to different phrasing. Concretely: a
live benchmark against qwen2.5-coder:7b found it defaults to wrapping its
tool-call JSON in a markdown code fence instead of emitting the taught
``<tool_call>`` tag directly — see ``extraction.WholeMessageJsonExtractor``,
the recovery tier built for that exact shape. A profile can select a
template by ID (``tool_calling.presentation.contract_template``) instead
of always getting the universal default. Previously this field was parsed
from YAML but never consulted by ``build_invocation_plan()``, so every
profile silently got identical instructions regardless of what its own
YAML claimed (e.g. ``qwen-coder-ollama.yaml`` declared
``contract_template: qwen-tool-v1`` while runtime always rendered the
generic template) — this registry is what makes that field real.
"""

from __future__ import annotations

from collections.abc import Callable

DEFAULT_CONTRACT_TEMPLATE_ID = "interop-tool-v1"

_CONTRACT_VERSION = "1"


def _render_interop_tool_v1(tool_descriptions: str, choice_instructions: str) -> str:
    """The universal default contract — bare ``<tool_call>`` tag envelope,
    no dialect-specific guidance."""
    return f"""\
<interop_tool_contract version="{_CONTRACT_VERSION}">
To call a tool, emit exactly:

<tool_call>{{"name":"tool_name","arguments":{{"key":"value"}}}}</tool_call>

Text outside a <tool_call> block is ordinary assistant text.
A <tool_call> block means the tool is intended to execute.

Available tools:

{tool_descriptions}
{choice_instructions}
</interop_tool_contract>
"""


def _render_qwen_tool_v1(tool_descriptions: str, choice_instructions: str) -> str:
    """Qwen-family variant: same envelope, plus an explicit instruction
    against the specific defect this family is prone to. This reduces how
    often the whole_message_json fallback tier needs to fire — it doesn't
    replace it; a profile should keep that fallback enabled regardless,
    since a prompt instruction is guidance, not a guarantee."""
    return f"""\
<interop_tool_contract version="{_CONTRACT_VERSION}">
To call a tool, emit exactly:

<tool_call>{{"name":"tool_name","arguments":{{"key":"value"}}}}</tool_call>

Do NOT wrap the <tool_call> block in Markdown or code fences (no triple
backticks before or after it). Emit the tag directly, with no surrounding
formatting.

Text outside a <tool_call> block is ordinary assistant text.
A <tool_call> block means the tool is intended to execute.

Available tools:

{tool_descriptions}
{choice_instructions}
</interop_tool_contract>
"""


CONTRACT_TEMPLATES: dict[str, Callable[[str, str], str]] = {
    "interop-tool-v1": _render_interop_tool_v1,
    "qwen-tool-v1": _render_qwen_tool_v1,
}


def render_contract(
    template_id: str | None,
    *,
    tool_descriptions: str,
    choice_instructions: str,
) -> str:
    """Render the named contract template.

    None falls back to the default (a profile that doesn't opt into a
    specific template) — an unknown non-None ID is a bug, not an expected
    default case, since validate_profile_schema already rejects unknown
    template IDs at profile-load time.
    """
    render_fn = CONTRACT_TEMPLATES[template_id] if template_id else _render_interop_tool_v1
    return render_fn(tool_descriptions, choice_instructions).strip()
