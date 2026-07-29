"""P0.5 gate: production code must not import competing legacy implementations.

These legacy modules contain unsafe behaviors (fuzzy tool-name guessing,
prefix/substring matching, mutating calls, coercing strings to booleans).
The v2 pipeline (interop.config.RepairPolicy, interop.repair.pipeline,
interop.transaction) must be the only execution path.
"""

from __future__ import annotations

import ast
import inspect
from importlib import import_module

# Modules that must NOT be imported by any production code
FORBIDDEN_IMPORTS = {
    "agent_interop.repair.validate": {
        "reason": "Contains fuzzy-name guessing, call mutation, and string-to-bool coercion",
        "replacement": "agent_interop.repair.pipeline + interop.config.RepairPolicy",
    },
}

# Files allowed to import from these modules (tests, compat shims, migration tools)
ALLOWLIST = {
    "agent_interop.types": "Temporary import facade — allowed",
    "agent_interop.testing.conformance": "Test harness — allowed",
    "agent_interop.plugin.adapter": "Plugin adapter — allowed",
    "agent_interop.repair.validate": "Self-import allowed within the module",
    "agent_interop.repair.parse": "Self-import allowed within the module",
}

PRODUCTION_MODULES = [
    "agent_interop.gateway",
    "agent_interop.server.app",
    "agent_interop.upstreams.openai_chat",
    "agent_interop.upstreams.ollama_chat",
    "agent_interop.upstreams.anthropic",
    "agent_interop.upstreams.openai_responses",
    "agent_interop.protocols.anthropic_messages",
    "agent_interop.protocols.openai_chat",
    "agent_interop.protocols.openai_responses",
    "agent_interop.transport.http",
    "agent_interop.transport.sse",
    "agent_interop.streaming.coordinator",
    "agent_interop.evidence.store",
    "agent_interop.evidence.key",
    "agent_interop.model.registry",
    "agent_interop.repair.pipeline",
    "agent_interop.repair.invocation",
    "agent_interop.repair.rules",
    "agent_interop.repair.paths",
    "agent_interop.transaction",
    "agent_interop.extraction",
    "agent_interop.session",
    "agent_interop.history.reconcile",
]


def _is_allowlisted(_module_path: str, imported_module: str) -> bool:
    """Check if an import is in the allowlist."""
    if imported_module in ALLOWLIST:
        return True
    for pattern in ALLOWLIST:
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if imported_module.startswith(prefix):
                return True
    return False


def test_production_does_not_import_legacy_repair_validate() -> None:
    """Production code must not import from agent_interop.repair.validate."""
    violations: list[str] = []

    for mod_name in PRODUCTION_MODULES:
        try:
            mod = import_module(mod_name)
            source = inspect.getsource(mod)
        except (ImportError, OSError):
            continue

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "agent_interop.repair.validate":
                    if not _is_allowlisted(mod_name, node.module):
                        violations.append(
                            f"{mod_name} imports from {node.module}: "
                            f"{[a.name for a in node.names]}"
                        )

    assert not violations, (
        "Production modules must not import legacy repair.validate:\n"
        + "\n".join("  ✗ " + v for v in violations)
    )


def test_gateway_uses_only_v2_repair_pipeline() -> None:
    """Gateway must use only interop.repair.pipeline and interop.repair.invocation."""
    mod = import_module("agent_interop.gateway")
    source = inspect.getsource(mod)

    tree = ast.parse(source)

    allowed_repair_imports = {
        "agent_interop.repair.pipeline",
        "agent_interop.repair.invocation",
    }

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("agent_interop.repair.") and node.module not in allowed_repair_imports:
                violations.append(
                    f"agent_interop.gateway imports from {node.module}: "
                    f"{[a.name for a in node.names]}"
                )

    assert not violations, (
        "Gateway must only import from v2 repair modules:\n"
        + "\n".join("  ✗ " + v for v in violations)
    )


def test_server_app_does_not_import_legacy_backends() -> None:
    """Server app must not import legacy interop.backends."""
    mod = import_module("agent_interop.server.app")
    source = inspect.getsource(mod)

    tree = ast.parse(source)

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("agent_interop.backends."):
                violations.append(
                    f"agent_interop.server.app imports from {node.module}: "
                    f"{[a.name for a in node.names]}"
                )

    assert not violations, (
        "Server app must not import legacy backends:\n"
        + "\n".join("  ✗ " + v for v in violations)
    )


def test_no_production_import_of_legacy_backend_types() -> None:
    """Production code must not import BackendKind or Backend from legacy backends."""
    forbidden_names = {"BackendKind", "Backend", "BackendAdapter"}
    violations: list[str] = []

    for mod_name in PRODUCTION_MODULES:
        try:
            mod = import_module(mod_name)
            source = inspect.getsource(mod)
        except (ImportError, OSError):
            continue

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and "backends" in node.module:
                    names = {a.name for a in node.names}
                    hit = names & forbidden_names
                    if hit:
                        violations.append(f"{mod_name} imports {hit} from {node.module}")

    assert not violations, (
        "Production modules must not import legacy backend types:\n"
        + "\n".join("  ✗ " + v for v in violations)
    )
