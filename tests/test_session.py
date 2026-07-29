"""Tests for bounded session state."""

from agent_interop.session import LoopState, SessionManager, SessionState


def test_session_state_create():
    state = SessionState(session_id="test-session-1")
    assert state.session_id == "test-session-1"
    assert state.request_count == 0
    assert state.repair_count == 0
    assert not state.flagged


def test_session_state_touch():
    state = SessionState(session_id="test-session-1")
    state.touch()
    assert state.request_count == 1


def test_session_state_record_repair():
    state = SessionState(session_id="test-session-1")
    state.record_repair("read_file", 2, "repaired")
    assert state.repair_count == 1
    assert "read_file" in state.recent_repairs
    assert state.recent_repairs["read_file"].issue_count == 2


def test_session_manager_create():
    mgr = SessionManager()
    state = mgr.begin_request("sess-1", route_id="qwen-local")
    assert state.session_id == "sess-1"
    assert state.route_id == "qwen-local"
    assert mgr.active_count == 1


def test_session_manager_reuse():
    mgr = SessionManager()
    s1 = mgr.begin_request("sess-1")
    s2 = mgr.begin_request("sess-1")
    assert s1 is s2
    assert mgr.active_count == 1


def test_session_manager_record_repair():
    mgr = SessionManager()
    mgr.begin_request("sess-1")
    mgr.record_repair("sess-1", "", "read_file", 2, "repaired")
    state = mgr.get("sess-1")
    assert state is not None
    assert state.repair_count == 1


def test_session_ttl_eviction():
    mgr = SessionManager(ttl_seconds=0)  # instant expiry
    mgr.begin_request("sess-1")
    # Force eviction check
    mgr._evict_expired()
    assert mgr.active_count == 0


def test_session_max_sessions():
    mgr = SessionManager(max_sessions=2)
    mgr.begin_request("sess-1")
    mgr.begin_request("sess-2")
    mgr.begin_request("sess-3")  # should evict one
    assert mgr.active_count == 2


def test_loop_state_not_flagged():
    loop = LoopState()
    loop.record_tool("read_file", False)
    assert not loop.flagged


def test_loop_state_same_tool_repeat():
    loop = LoopState()
    for _ in range(6):
        loop.record_tool("read_file", False)
    assert loop.flagged


def test_loop_state_consecutive_repairs():
    loop = LoopState()
    for _ in range(4):
        loop.record_tool("read_file", True)
    assert loop.flagged


def test_loop_state_reset():
    loop = LoopState()
    for _ in range(6):
        loop.record_tool("read_file", True)
    assert loop.flagged
    loop.reset()
    assert not loop.flagged


def test_session_flagged_repair_loop():
    state = SessionState(session_id="sess-1")
    for _ in range(4):
        state.record_repair("read_file", 2, "repaired")
    assert state.flagged


def test_session_manager_flag_status():
    mgr = SessionManager()
    mgr.begin_request("sess-1")
    mgr.record_repair("sess-1", "", "read_file", 2, "repaired")
    # Not enough to flag yet
    assert len(mgr.get_flag_status()) == 0


def test_session_manager_flag_after_loop():
    mgr = SessionManager()
    mgr.begin_request("sess-1")
    for _ in range(6):
        mgr.record_repair("sess-1", "", "read_file", 2, "repaired")
        state = mgr.get("sess-1")
        if state and state.flagged:
            break
    flagged = mgr.get_flag_status()
    assert "sess-1" in flagged or not flagged  # at least not error
    # Note: consecutive_repairs check triggers at 4
    state = mgr.get("sess-1")
    if state:
        assert state.flagged


# ─── MVP-14: route-scoped sessions, single-touch request counting ─────────


def test_same_session_id_different_routes_are_isolated():
    """The same client session ID talking to two different routes must not
    share loop-detection state between them."""
    mgr = SessionManager()
    mgr.begin_request("sess-1", route_id="route-a")
    for _ in range(6):
        mgr.record_repair("sess-1", "route-a", "read_file", 2, "repaired")

    mgr.begin_request("sess-1", route_id="route-b")
    state_a = mgr.get("sess-1", "route-a")
    state_b = mgr.get("sess-1", "route-b")
    assert state_a is not state_b
    assert state_a.flagged
    assert not state_b.flagged
    assert state_b.repair_count == 0


def test_begin_request_is_the_only_thing_that_touches():
    """get() and record_repair() must never increment request_count —
    only begin_request() may. A prior bug called get_or_create() a second
    time inside repair recording, double-counting request_count for every
    request that repaired at least one tool call."""
    mgr = SessionManager()
    mgr.begin_request("sess-1", route_id="route-a")
    assert mgr.get("sess-1", "route-a").request_count == 1

    mgr.get("sess-1", "route-a")
    mgr.record_repair("sess-1", "route-a", "read_file", 2, "repaired")
    mgr.record_repair("sess-1", "route-a", "edit_file", 1, "rejected")
    assert mgr.get("sess-1", "route-a").request_count == 1

    mgr.begin_request("sess-1", route_id="route-a")
    assert mgr.get("sess-1", "route-a").request_count == 2
