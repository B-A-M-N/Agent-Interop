"""Model profile loader — loads YAML profiles from directories.

Profile precedence (highest first):
1. Session override
2. Project configuration (.interop.yaml)
3. User configuration (~/.config/interop/config.yaml)
4. Verified conformance result (cached)
5. Built-in profile (from profiles/ directory)
6. Backend metadata
7. Generic fallback profile
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ─── Default profile directories ────────────────────────────────────────────

DEFAULT_PROFILE_DIRS = [
    Path(__file__).resolve().parent.parent.parent.parent / "profiles",  # project profiles/
]


@dataclass
class ModelProfile:
    """Resolved model profile loaded from YAML."""

    schema_version: str = "interop.model-profile.v1"
    id: str = ""
    matched_by: str = ""

    # Tool settings
    tool_parser: str = "generic"
    tool_automatic: bool = False
    tool_required: bool = True
    tool_named: bool = False
    tool_parallel: bool = False
    tool_strict_arguments: bool = False

    # Reasoning
    reasoning_supported: bool = False
    reasoning_parser: str = ""
    reasoning_send_to_client: str = "suppress"

    # Context
    declared_tokens: int = 4096
    safe_tokens: int = 2048
    reserve_output_tokens: int = 1024

    # Streaming
    streaming_supported: bool = True
    tool_arguments_incremental: bool = False

    # Repair
    repair_malformed_json: str = "reparse"
    repair_unknown_tool: str = "reject"
    repair_missing_tool_id: str = "synthesize"
    repair_max_attempts: int = 1

    # Template
    template_source: str = "backend"
    template_fallback: str = ""

    # Profile revision tracking
    profile_revision: str = ""

    @classmethod
    def from_yaml(cls, data: dict[str, Any], matched_by: str = "") -> ModelProfile:
        """Load from a parsed YAML dict."""
        tc = data.get("tool_calling", {})
        r = data.get("reasoning", {})
        ctx = data.get("context", {})
        st = data.get("streaming", {})
        rp = data.get("repair", {})
        tmpl = data.get("template", {})

        return cls(
            schema_version=data.get("schema_version", cls.schema_version),
            id=data.get("id", ""),
            matched_by=matched_by,
            tool_parser=tc.get("parser", cls.tool_parser),
            tool_automatic=tc.get("automatic", cls.tool_automatic),
            tool_required=tc.get("required", cls.tool_required),
            tool_named=tc.get("named", cls.tool_named),
            tool_parallel=tc.get("parallel", cls.tool_parallel),
            tool_strict_arguments=tc.get("strict_arguments", cls.tool_strict_arguments) in ("preferred", True),
            reasoning_supported=r.get("supported", cls.reasoning_supported),
            reasoning_parser=r.get("parser", cls.reasoning_parser),
            reasoning_send_to_client=r.get("send_to_client", cls.reasoning_send_to_client),
            declared_tokens=ctx.get("declared_tokens", cls.declared_tokens),
            safe_tokens=ctx.get("safe_tokens", cls.safe_tokens),
            reserve_output_tokens=ctx.get("reserve_output_tokens", cls.reserve_output_tokens),
            streaming_supported=st.get("supported", cls.streaming_supported),
            tool_arguments_incremental=st.get("tool_arguments_incremental", cls.tool_arguments_incremental),
            repair_malformed_json=rp.get("malformed_json", cls.repair_malformed_json),
            repair_unknown_tool=rp.get("unknown_tool", cls.repair_unknown_tool),
            repair_missing_tool_id=rp.get("missing_tool_id", cls.repair_missing_tool_id),
            repair_max_attempts=rp.get("max_attempts", cls.repair_max_attempts),
            template_source=tmpl.get("source", cls.template_source),
            template_fallback=tmpl.get("fallback", cls.template_fallback),
            profile_revision=data.get("revision", ""),
        )


# ─── Profile Index ──────────────────────────────────────────────────────────


class ProfileIndex:
    """Index of all loaded profiles with match-based resolution."""

    def __init__(self) -> None:
        self._profiles: list[tuple[ModelProfile, list[re.Pattern], list[str]]] = []

    def add_profile(self, profile: ModelProfile, data: dict[str, Any]) -> None:
        """Add a profile with its match rules."""
        match_section = data.get("match", {})
        patterns = [re.compile(p) for p in match_section.get("model_patterns", [])]
        backends = match_section.get("backends", ["ollama", "openai-compatible"])
        self._profiles.append((profile, patterns, backends))

        # Sort: most specific (longest non-wildcard patterns, specific backends) first
        def specificity(item):
            p, pats, backends = item
            # Reward specific backends (smaller set = more specific)
            backend_score = len(backends)
            # Penalize wildcards heavily; reward specific chars
            pattern_score = 0
            for pat in pats:
                s = pat.pattern
                if s == ".*":
                    pattern_score -= 1000  # generic catch-all
                elif s.startswith("^") and s.endswith("$"):
                    pattern_score += 500 + len(s)  # anchored exact
                elif s.startswith("^"):
                    pattern_score += 200 + len(s)  # anchored prefix
                else:
                    pattern_score += len(s)
            # Lower sort_key = more specific
            return (pattern_score, backend_score)  # high pattern_score = more specific

        self._profiles.sort(key=specificity, reverse=True)

    def resolve(self, model_name: str, backend: str = "ollama") -> ModelProfile | None:
        """Find the best matching profile for a model/backend pair."""
        for profile, patterns, backends in self._profiles:
            if backend not in backends and "*" not in backends:
                continue
            for pattern in patterns:
                if pattern.search(model_name):
                    profile.matched_by = pattern.pattern
                    return profile
        return None

    def get_by_id(self, profile_id: str) -> ModelProfile | None:
        """Look up a profile by its ID."""
        for profile, _, _ in self._profiles:
            if profile.id == profile_id:
                return profile
        return None

    def list_ids(self) -> list[str]:
        """List all loaded profile IDs."""
        return [p.id for p, _, _ in self._profiles]

    def __len__(self) -> int:
        return len(self._profiles)


# ─── Loader ──────────────────────────────────────────────────────────────────


def load_profiles(dirs: list[Path] | None = None) -> ProfileIndex:
    """Load all YAML profiles from the given directories."""
    index = ProfileIndex()
    search_dirs = dirs or DEFAULT_PROFILE_DIRS

    for d in search_dirs:
        if not d.exists():
            continue
        for fpath in sorted(d.glob("*.yaml")):
            try:
                raw = fpath.read_text()
                data = yaml.safe_load(raw)
                if not isinstance(data, dict):
                    continue
                if data.get("schema_version", "").startswith("interop.model-profile"):
                    profile = ModelProfile.from_yaml(data, matched_by=fpath.stem)
                    index.add_profile(profile, data)
            except (yaml.YAMLError, OSError, ValueError) as exc:
                # Log and skip bad profiles
                import logging
                logging.getLogger("interop.profiles").warning(
                    "Failed to load profile %s: %s", fpath, exc
                )

    return index


def get_fallback_profile() -> ModelProfile:
    """Return the generic fallback profile used when nothing matches."""
    return ModelProfile(
        id="generic-fallback",
        tool_automatic=False,
        tool_required=True,
        tool_named=False,
        tool_parallel=False,
        tool_strict_arguments=False,
        reasoning_supported=False,
        declared_tokens=4096,
        safe_tokens=2048,
        reserve_output_tokens=1024,
        repair_malformed_json="reparse",
    )