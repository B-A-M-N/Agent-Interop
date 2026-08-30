"""Battery revision tracking for qualification records.

The revision is DERIVED from the battery definition itself (probe names,
prompts, tool requirements, forced presentations, expected texts) so any
change to the probe contracts automatically invalidates every cached
qualification record stamped by an older battery.
"""

from __future__ import annotations

import hashlib
import json

from agent_interop.qualification.probes import fast_bootstrap_battery
from agent_interop.qualification.state import QualificationRecord


def _battery_digest() -> str:
    """Stable 16-hex digest over the current bootstrap battery contracts.

    P0-56: the synthetic probe tool's schema is part of the battery — a
    schema change alters what the model was asked to do, so evidence gathered
    against the old schema must not survive.
    """
    from agent_interop.qualification.probes import SYNTHETIC_TOOL

    payload = {
        "synthetic_tool": {
            "name": SYNTHETIC_TOOL.name,
            "schema": SYNTHETIC_TOOL.input_schema,
        },
        "probes": [
            {
                "name": probe.name,
                "prompt": probe.prompt,
                "requires_tools": probe.requires_tools,
                "presentation": probe.presentation.value if probe.presentation else None,
                "expected_text": probe.expected_text,
                "expected_marker": probe.expected_marker,
            }
            for probe in fast_bootstrap_battery()
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# Stable 16-hex revision for the current qualification battery definition.
# Computed from the probe contracts, not hard-coded: editing a probe prompt,
# forcing a different presentation, or changing an expected_text bumps this
# digest and thereby invalidates all previously persisted records.
QUALIFICATION_BATTERY_REVISION: str = _battery_digest()


def battery_revision(record: QualificationRecord) -> str:
    """Derive a per-record battery revision digest.

    The revision is a 16-hex digest of the record's model digest plus the
    canonical battery hash so that two records with the same battery but
    different models (or vice-versa) get distinct revisions.  This lets
    gateway logic compare whether a stored record was produced by the same
    battery definition without importing the literal constant.
    """
    data = f"{record.model_digest}:{QUALIFICATION_BATTERY_REVISION}".encode()
    return hashlib.sha256(data).hexdigest()[:16]
