"""One invalid-cell preparation rule for planner fields and transport."""

import numpy as np


def prepare_travel_time(values, invalid_mask=None, max_finite=None):
    """Return owned float64 values using the deployed 4*max_finite+1 sentinel.

    Raw input arrays keep their invalid values. A supplied mask marks extra
    no-go cells. With no finite values, the sentinel is one.
    """
    raw = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(raw)
    if max_finite is None:
        maximum = float(raw[finite].max()) if finite.any() else 0.0
    else:
        maximum = float(max_finite)
        if not np.isfinite(maximum):
            raise ValueError("max_finite must be finite")
    sentinel = 4.0 * maximum + 1.0
    if not np.isfinite(sentinel):
        raise ValueError("travel-time sentinel is not finite")
    invalid = ~finite
    if invalid_mask is not None:
        mask = np.asarray(invalid_mask, dtype=bool)
        if mask.shape != raw.shape:
            raise ValueError("invalid mask must match travel time")
        invalid = invalid | mask
    return np.where(invalid, sentinel, raw)
