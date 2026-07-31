"""Small-sample-safe evidence confidence helpers."""

from __future__ import annotations

import math


def wilson_lower_bound(successes: int, samples: int, *, z: float = 1.96) -> float:
    """Return a conservative binomial lower confidence bound.

    A raw rate is not enough to promote a runtime path: one success out of
    one is useful telemetry but is not operational proof.
    """
    if samples <= 0 or successes < 0 or successes > samples:
        return 0.0
    proportion = successes / samples
    denominator = 1.0 + (z * z / samples)
    centre = proportion + z * z / (2.0 * samples)
    margin = z * math.sqrt((proportion * (1.0 - proportion) + z * z / (4.0 * samples)) / samples)
    return max(0.0, (centre - margin) / denominator)


def has_confident_capability(
    successes: int,
    samples: int,
    *,
    minimum_samples: int = 5,
    threshold: float = 0.8,
) -> bool:
    """Whether observed evidence safely clears a capability promotion bar."""
    return samples >= minimum_samples and wilson_lower_bound(successes, samples) >= threshold
