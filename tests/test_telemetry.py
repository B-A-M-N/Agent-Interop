"""Tests for repair telemetry."""
from agent_interop.repair.telemetry import (
    RepairTelemetry,
    ResponseCompletedEvent,
    ToolInputRejectedEvent,
    ToolInputRepairedEvent,
)


def test_telemetry_emit_request_started():
    tel = RepairTelemetry()
    event = tel.emit_request_started("req-1", "sess-1", route_id="qwen-local")
    assert event.event_type == "request_started"
    assert event.request_id == "req-1"
    assert tel.summary["request_started"] == 1


def test_telemetry_emit_repaired():
    tel = RepairTelemetry()
    event = tel.emit_repaired("req-1", "read_file", rules=["rename_aliased_fields"], paths=["$.path"])
    assert isinstance(event, ToolInputRepairedEvent)
    assert event.tool_name == "read_file"
    assert event.repair_rules == ["rename_aliased_fields"]


def test_telemetry_emit_rejected():
    tel = RepairTelemetry()
    event = tel.emit_rejected("req-1", "read_file", paths=["$.path"])
    assert isinstance(event, ToolInputRejectedEvent)
    assert event.tool_name == "read_file"


def test_telemetry_emit_response_completed():
    tel = RepairTelemetry()
    event = tel.emit_response_completed("req-1", tool_call_count=3, accepted_count=2, rejected_count=1)
    assert isinstance(event, ResponseCompletedEvent)
    assert event.tool_call_count == 3
    assert event.accepted_count == 2


def test_telemetry_summary():
    tel = RepairTelemetry()
    tel.emit_request_started("req-1", "sess-1")
    tel.emit_request_started("req-2", "sess-2")
    tel.emit_tool_candidate("req-2", "read_file")
    assert tel.summary["request_started"] == 2
    assert tel.summary["tool_candidate_detected"] == 1


def test_telemetry_get_events():
    tel = RepairTelemetry()
    tel.emit_request_started("req-1", "sess-1")
    tel.emit_repaired("req-1", "read_file", rules=[], paths=[])
    events = tel.get_events("request_started")
    assert len(events) == 1
    assert events[0].event_type == "request_started"


def test_telemetry_clear():
    tel = RepairTelemetry()
    tel.emit_request_started("req-1", "sess-1")
    tel.clear()
    assert len(tel.get_events()) == 0
    assert tel.summary == {}


def test_telemetry_session_id_hash():
    tel = RepairTelemetry()
    event = tel.emit_request_started("req-1", "my-secret-session-123")
    # Should be hashed, not raw
    assert event.session_id_hash != "my-secret-session-123"
    assert len(event.session_id_hash) == 16


def test_telemetry_no_sensitive_data():
    tel = RepairTelemetry()
    # Telemetry should never contain sensitive content
    event = tel.emit_repaired("req-1", "read_file", rules=["drop_null"], paths=["$.path"])
    assert "arguments" not in str(event.__dict__)
    assert "content" not in str(event.__dict__) or event.event_type == "tool_call_detected"