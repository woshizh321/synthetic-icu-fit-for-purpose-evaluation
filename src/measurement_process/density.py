"""Observed-hour density semantics locked after the U0-R1 correction."""
from __future__ import annotations

import numpy as np


def observed_hour_density(offset_minutes) -> float:
    """Fraction of 24 landmark hours with >=1 event, using floor-hour bins."""
    offsets = np.asarray(offset_minutes, dtype=float)
    eligible = offsets[(offsets >= 0.0) & (offsets < 1440.0)]
    return float(np.unique(np.floor(eligible / 60.0)).size / 24.0)
