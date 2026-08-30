"""Side-effect-free qualification probe descriptions."""

from __future__ import annotations

from dataclasses import dataclass

from agent_interop.abi import CanonicalTool
from agent_interop.config import ToolMode

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
    """P0.32: Added presentation field to force a specific tool mode for the probe."""

    name: str
    prompt: str
    requires_tools: bool = False
    presentation: ToolMode | None = None  # P0.32: force NATIVE or PROMPTED mode
    # Review #22: exact substring the response MUST contain for the probe to
    # pass.  When set, the gateway checks this INSTEAD of any name-based
    # special case, so probe semantics live in the probe contract (and feed
    # the battery-revision digest) rather than in gateway if-chains.
    expected_text: str = ""
    # P0-55: forced-tool probes must call the synthetic tool EXACTLY ONCE
    # with this marker as arguments.marker.  Empty for non-tool probes.
    expected_marker: str = ""


def fast_bootstrap_battery() -> tuple[BootstrapProbe, ...]:
    return (
        BootstrapProbe(
            "exact_text", "Reply with exactly: INTEROP_PROBE_OK", expected_text="INTEROP_PROBE_OK"
        ),
        # P0.32: Force NATIVE mode for native_forced_tool probe
        BootstrapProbe(
            "native_forced_tool", "Call interop_probe with marker native", True,
            presentation=ToolMode.NATIVE, expected_marker="native",
        ),
        # P0.32: Force PROMPTED mode for prompted_forced_tool probe
        BootstrapProbe(
            "prompted_forced_tool", "Call interop_probe with marker prompted", True,
            presentation=ToolMode.PROMPTED, expected_marker="prompted",
        ),
        BootstrapProbe(
            "no_tool", "Reply with exactly: no tool needed", True, expected_text="no tool needed"
        ),
        BootstrapProbe(
            "tool_result_continuation",
            "The tool returned marker=done. Reply with exactly: continued",
            True,
            expected_text="continued",
        ),
    )
