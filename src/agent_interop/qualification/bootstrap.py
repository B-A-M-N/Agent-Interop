"""Qualification coordinator contract.

Execution is injected so the gateway can use its normal authenticated
transport and never grants probes filesystem or shell tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from agent_interop.qualification.probes import BootstrapProbe, fast_bootstrap_battery
from agent_interop.qualification.promotion import promote
from agent_interop.qualification.state import QualificationRecord

ProbeExecutor = Callable[[BootstrapProbe], Awaitable[bool]]


class BootstrapQualifier:
    async def qualify(self, model_digest: str, execute: ProbeExecutor) -> QualificationRecord:
        outcomes = {probe.name: await execute(probe) for probe in fast_bootstrap_battery()}
        record = QualificationRecord(
            model_digest=model_digest,
            native_forced_tool=outcomes["native_forced_tool"],
            prompted_forced_tool=outcomes["prompted_forced_tool"],
            no_tool_compliant=outcomes["no_tool"],
            continuation=outcomes["tool_result_continuation"],
        )
        return QualificationRecord(**{**record.__dict__, "state": promote(record)})
