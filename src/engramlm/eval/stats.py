"""Bootstrap confidence intervals for seed- or document-level results."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float

    def format(self, digits: int = 3) -> str:
        return f"{self.mean:.{digits}f} [{self.low:.{digits}f}, {self.high:.{digits}f}]"


def bootstrap_ci(
    values: Sequence[float], *, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Percentile bootstrap interval for the mean."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("bootstrap_ci needs at least one value")
    if data.size == 1:
        return Interval(float(data[0]), float(data[0]), float(data[0]))
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(n_resamples, data.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return Interval(float(data.mean()), float(low), float(high))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval for ``mean(a - b)`` over paired observations (same seeds or documents)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diffs = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bootstrap_ci(diffs.tolist(), n_resamples=n_resamples, confidence=confidence, seed=seed)
