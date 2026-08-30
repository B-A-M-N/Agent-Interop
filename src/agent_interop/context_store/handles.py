"""Handle generation and validation for stored context."""

from __future__ import annotations

import re

# Refs are url-safe base64, 24 random bytes -> ~32 chars.
_REF_RE = re.compile(r"^[A-Za-z0-9_\-]{16,}$")


def is_valid_ref(ref: str) -> bool:
    """Return True if the string looks like a valid opaque handle."""
    return bool(_REF_RE.match(ref))
