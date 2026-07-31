"""Centralized XDG Base Directory path resolution for Interop.

https://specifications.freedesktop.org/basedir-spec/latest/

All Interop state lives under an ``interop`` namespace inside each XDG base
directory — whether that base directory comes from the environment variable
or from its fallback default. Call sites that computed this ad hoc applied
the ``interop`` namespace inconsistently: always present in the fallback
default, silently dropped whenever the user actually set the XDG
environment variable — so a user with ``XDG_CONFIG_HOME`` set got
``$XDG_CONFIG_HOME/config.yaml`` instead of the intended
``$XDG_CONFIG_HOME/interop/config.yaml``. Route every XDG-based path through
this module instead of recomputing it locally.
"""

from __future__ import annotations

import os
from pathlib import Path


def _xdg_base(env_var: str, fallback_parts: tuple[str, ...]) -> Path:
    value = os.environ.get(env_var)
    base = Path(value) if value else Path(os.path.expanduser("~")).joinpath(*fallback_parts)
    return base / "interop"


def config_dir() -> Path:
    return _xdg_base("XDG_CONFIG_HOME", (".config",))


def config_file() -> Path:
    return config_dir() / "config.yaml"


def state_dir() -> Path:
    return _xdg_base("XDG_STATE_HOME", (".local", "state"))


def cache_dir() -> Path:
    return _xdg_base("XDG_CACHE_HOME", (".cache",))


def log_file() -> Path:
    return state_dir() / "logs" / "interop.log"


def evidence_file() -> Path:
    return state_dir() / "evidence.db"


def diagnostic_cases_dir() -> Path:
    """Directory for bounded, sanitized live diagnostic replay cases."""
    return state_dir() / "replay-cases"
