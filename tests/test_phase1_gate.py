"""Phase 1 completion gate: verify production code doesn't import v1 compat types.

New code MUST use canonical ABI types from agent_interop.abi, not the backward-compat
v1 types re-exported via interop.types. This static-analysis test enforces
that rule for the production protocol adapters, upstream codecs, and server.
"""

from __future__ import annotations

import ast
from importlib import import_module
from pathlib import Path

import pytest

PRODUCTION_MODULES = [
    "agent_interop.protocols.anthropic_messages",
    "agent_interop.protocols.openai_chat",
    "agent_interop.protocols.openai_responses",
    "agent_interop.gateway",
    "agent_interop.upstreams.openai_chat",
    "agent_interop.upstreams.ollama_chat",
    "agent_interop.upstreams.anthropic",
    "agent_interop.upstreams.openai_responses",
    "agent_interop.server.app",
]

V1_TYPE_NAMES = [
    "AgentMessage",
    "ContentBlock",
    "ToolCall",
    "ToolResult",
]

EXCLUDED_MODULES = {
    "agent_interop.types",
    "src.interop.repair.validate",
    "agent_interop.repair.validate",
    "src.interop.plugin.adapter",
    "agent_interop.plugin.adapter",
    "src.interop.testing.conformance",
    "agent_interop.testing.conformance",
}


def test_no_v1_imports_in_production() -> None:
    """Fail if any production module imports a v1 backward-compat type."""
    production_files = list(Path("src/agent_interop").rglob("*.py"))

    for file_path in production_files:
        module_path = str(file_path).replace("/", ".").replace(".py", "").replace("src/", "")

        if module_path in EXCLUDED_MODULES or module_path == "src.interop.types":
            continue

        try:
            source = file_path.read_text()
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module in {"agent_interop.types", "agent_interop.compat"}:
                    imported = {alias.name for alias in node.names}
                    forbidden = imported & set(V1_TYPE_NAMES)
                    if forbidden:
                        pytest.fail(f"{module_path} imports deprecated {forbidden}")


@pytest.mark.parametrize("mod_name", PRODUCTION_MODULES)
@pytest.mark.parametrize("v1_type", V1_TYPE_NAMES)
def test_no_v1_imports_in_modules(mod_name: str, v1_type: str) -> None:
    """Fail if any production module imports a v1 backward-compat type."""
    import inspect

    mod = import_module(mod_name)
    source = inspect.getsource(mod)

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module in {"agent_interop.types", "agent_interop.compat"}:
                imported = {alias.name for alias in node.names}
                forbidden = imported & {v1_type}
                assert not forbidden, f"{mod_name} imports deprecated {v1_type}"