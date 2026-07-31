"""Side-effect-free qualification probe descriptions."""

from __future__ import annotations

from dataclasses import dataclass

from agent_interop.abi import CanonicalTool

SYNTHETIC_TOOL = CanonicalTool(
    name="interop_probe",
    description="Return the supplied marker. This has no side effects.",
    input_schema={
        "type": "object",
        "properties": {"marker": {"type": "string"}},
        "required": ["marker"],
    },
)


@dataclass(frozen=True)
class BootstrapProbe:
    name: str
    prompt: str
    requires_tools: bool = False


def fast_bootstrap_battery() -> tuple[BootstrapProbe, ...]:
    return (
        BootstrapProbe("exact_text", "Reply with exactly: INTEROP_PROBE_OK"),
        BootstrapProbe("native_forced_tool", "Call interop_probe with marker native", True),
        BootstrapProbe("prompted_forced_tool", "Call interop_probe with marker prompted", True),
        BootstrapProbe("no_tool", "Reply with exactly: no tool needed", True),
        BootstrapProbe("tool_result_continuation", "The tool returned marker=done. Reply with exactly: continued", True),
    )
