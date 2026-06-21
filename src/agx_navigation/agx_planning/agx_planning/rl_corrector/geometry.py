"""Small geometry helpers for the RL corrector env. Pure (numpy only)."""

from typing import Optional, Tuple

import numpy as np


def cumulative_arclength(xy) -> np.ndarray:
    """Cumulative arc length along a polyline (N,2). Returns (N,) with [0]=0."""
    xy = np.asarray(xy, dtype=float)
    if xy.shape[0] < 2:
        return np.zeros(xy.shape[0])
    seg = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def project_arclength(point, xy, cum: Optional[np.ndarray] = None) -> Tuple[float, int]:
    """Nearest-vertex projection of `point` onto polyline `xy` (N,2).

    Returns (arc_length_at_nearest_vertex, vertex_index). Vertex projection is
    adequate because nominals are densely sampled at control_dt.
    """
    xy = np.asarray(xy, dtype=float)
    p = np.asarray(point, dtype=float)
    if cum is None:
        cum = cumulative_arclength(xy)
    d2 = np.sum((xy - p) ** 2, axis=1)
    i = int(np.argmin(d2))
    return float(cum[i]), i
