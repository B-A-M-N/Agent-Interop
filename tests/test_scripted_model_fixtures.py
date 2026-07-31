"""Acceptance matrix coverage for scripted model behaviours."""

from agent_interop.planning import CompatibilityPath
from agent_interop.testing.scripted_fixtures import scripted_model_fixtures


def test_scripted_fixture_catalog_covers_required_model_behaviors() -> None:
    fixtures = {fixture.name: fixture for fixture in scripted_model_fixtures()}
    assert set(fixtures) == {
        "native_tool", "prompted_envelope", "bare_json", "forced_only",
        "automatic_selection_failure", "chat_only", "malformed_arguments",
        "duplicate_id", "low_context", "streaming_text_only",
        "continuation_failure", "looping",
    }
    assert fixtures["native_tool"].expected_path is CompatibilityPath.DIRECT
    assert fixtures["chat_only"].expected_path is CompatibilityPath.CONTROLLED
    assert fixtures["malformed_arguments"].expected_path is CompatibilityPath.ADAPTED
