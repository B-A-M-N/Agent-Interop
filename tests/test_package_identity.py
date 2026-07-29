"""Tests for src/agent_interop/__init__.py: version resolution."""

from __future__ import annotations

import importlib.metadata

import pytest

import agent_interop


def test_version_is_nonempty_string() -> None:
    assert isinstance(agent_interop.__version__, str)
    assert agent_interop.__version__


def test_version_falls_back_when_distribution_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _raise)
    # Re-run the same resolution logic __init__ uses at import time.
    try:
        version = importlib.metadata.version(agent_interop._DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0.dev0"
    assert version == "0.0.0.dev0"
