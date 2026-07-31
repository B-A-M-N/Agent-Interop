"""Compatibility-controller contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_interop.abi import CanonicalToolCallBlock


class ControllerAction(str, Enum):
    FINAL_TEXT = "final_text"
    TOOL_CALL = "tool_call"
    TOOL_BATCH = "tool_batch"
    DELEGATE_PRIMARY = "delegate_primary"
    FAIL = "fail"


@dataclass(frozen=True)
class ControllerDecision:
    action: ControllerAction
    text: str = ""
    tool_calls: tuple[CanonicalToolCallBlock, ...] = ()
    primary_prompt: str = ""
    diagnostics: tuple[str, ...] = ()


@dataclass
class ControllerSessionState:
    session_id: str
    route_id: str
    client_id: str
    controller_route_id: str
    primary_route_id: str
    phase: str = "receive_client_request"
    visible_tool_fingerprint: str = ""
    pending_tool_call_ids: tuple[str, ...] = ()
    primary_turn_count: int = 0
    controller_turn_count: int = 0
