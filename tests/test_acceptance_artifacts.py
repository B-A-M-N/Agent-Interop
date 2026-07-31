"""Acceptance artifact provenance is complete even for opt-in client runs."""

from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _harness_module():
    path = Path(__file__).parent / "acceptance" / "_harness.py"
    spec = spec_from_file_location("interop_acceptance_harness", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_acceptance_artifact_contains_required_provenance(tmp_path, monkeypatch) -> None:
    harness = _harness_module()
    monkeypatch.setattr(harness, "RESULTS_DIR", tmp_path)
    path = harness.write_acceptance_result(
        "Example Client",
        "1.2.3",
        passed=True,
        scenario="single_tool_round_trip",
        argv=["example", "run"],
        configuration_strategy="generated_config",
        protocol="openai_chat",
        compatibility_path="adapted",
        controller_used=False,
        model_digest="sha256:example",
        verification={"read_test": True, "multi_turn_continuation": True},
    )
    artifact = json.loads(path.read_text())
    assert artifact["argv"] == ["example", "run"]
    assert artifact["build"]["planner_revision"]
    assert artifact["compatibility_path"] == "adapted"
    assert artifact["model_digest"] == "sha256:example"
    assert artifact["verification"] == {
        "read_test": True,
        "edit_test": False,
        "tool_error_recovery": False,
        "multi_turn_continuation": True,
        "cleanup_verification": False,
    }
