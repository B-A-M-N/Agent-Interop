"""Shared model-name normalization for tag-aware comparison.

Ollama (and similar registries) name models with an optional ``:tag``
suffix (e.g. ``qwen3-coder:latest``). A configured model name often omits
the tag while the backend's inventory lists it explicitly, or vice versa —
naive exact-string comparison then reports a model as "missing" even
though it's actually present. Both the managed launcher (checking whether
a model needs pulling) and gateway readiness probing (checking whether a
route's configured model is in the backend's inventory) need the exact
same tag-aware comparison; having two independent implementations risks
them silently drifting apart on edge cases.
"""

from __future__ import annotations


def model_names_match(requested: str, available: str) -> bool:
    """True if ``available`` (a name from a backend's model inventory)
    satisfies ``requested`` (a configured/desired model name), accounting
    for an optional ``:tag`` suffix on either side."""
    req = requested.strip().lower()
    avail = available.strip().lower()

    if avail == req:
        return True
    if ":" in avail and avail.split(":", 1)[0] == req:
        return True
    return ":" in req and req.split(":", 1)[0] == avail
