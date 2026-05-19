"""Pure-Python deviation detection logic for the trajectory corrector.

No ROS2 imports. Operates on plain Python types and numpy arrays so it can
be unit-tested outside a ROS2 workspace.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from agx_planning.runtime_corrector.config import CorrectorConfig
from agx_planning.runtime_corrector.geometry import nearest_projection_on_path


def _path_tangents(path_xy: np.ndarray) -> np.ndarray:
    """Unit tangent vectors for each point in path_xy (N, 2).

    Uses forward differences for all points, repeating the last tangent
    at the endpoint. Zero-length segments produce a zero vector (the
    direction filter will treat them as invalid matches, which is safe).
    """
    diffs = np.diff(path_xy, axis=0)  # (N-1, 2)
    norms = np.hypot(diffs[:, 0], diffs[:, 1])
    # Avoid division by zero; zero-norm tangents are left as (0, 0).
    safe = norms > 1e-9
    unit = np.where(
        safe[:, np.newaxis], diffs / np.where(safe, norms, 1.0)[:, np.newaxis], 0.0
    )
    # Repeat last tangent for the final point.
    return np.vstack([unit, unit[-1:]])  # (N, 2)


def _windowed_max_deviation(
    deviations: np.ndarray, arc_s: np.ndarray, window_size: float
) -> float:
    """Largest per-point deviation found within any arc-length window.

    Uses a two-pointer sweep so the right boundary is never moved backward;
    total pointer travel is O(N). np.max over the (typically short) window
    slice is fast enough for N <= 2000 at 5 Hz.

    inf values in deviations (points with no valid directional match)
    propagate naturally through np.max, so a window containing a genuinely
    unmatched point always reports inf and triggers a replan.
    """
    n = len(deviations)
    if n == 0:
        return 0.0
    best = 0.0
    right = 0
    for left in range(n):
        if right < left:
            right = left
        while right + 1 < n and arc_s[right + 1] - arc_s[left] <= window_size:
            right += 1
        w_max = float(np.max(deviations[left : right + 1]))
        if w_max > best:
            best = w_max
    return best


class DeviationDetector:
    """Detects when the robot has deviated from the planned trajectory.

    All methods are pure Python / numpy — no ROS2 types involved.
    """

    def __init__(self, cfg: CorrectorConfig) -> None:
        self._cfg = cfg

    def check_corridor(
        self,
        pose: Tuple[float, float, float],
        path: List[Tuple[float, float, float]],
    ) -> Tuple[bool, float]:
        """Return (deviated, dist).

        deviated is True if the robot's perpendicular distance to the path
        exceeds corridor_epsilon. dist is always returned so the caller can
        publish it as a metric without recomputing.

        pose -- (x, y, theta) of the robot in the planning frame
        path -- ordered (x, y, theta) list from _build_path_polyline()
        """
        rx, ry, _ = pose
        proj_x, proj_y, _, _, _ = nearest_projection_on_path(rx, ry, path)
        dist = math.hypot(rx - proj_x, ry - proj_y)
        return dist > self._cfg.corridor_epsilon, dist

    def check_gradient_path(
        self,
        future_grad: np.ndarray,
        future_plan: np.ndarray,
    ) -> Tuple[bool, float, float]:
        """Return (should_replan, pct_val, win_val).

        Compares the FM2 gradient path against the buffered planned trajectory
        using a percentile check and an optional sliding-window max check.
        Cooldown gating is handled by the caller.

        future_grad -- (N, 2) array of (x, y), already trimmed and arc-truncated
        future_plan -- (M, 2) array of (x, y), already trimmed and arc-truncated
        """
        # Squared-distance matrix: (N, M) where N = gradient points, M = plan points.
        diff = future_grad[:, np.newaxis, :] - future_plan[np.newaxis, :, :]
        sq_dist = (diff**2).sum(axis=2)  # (N, M)

        if self._cfg.path_diff_min_tangent_dot > -1.0:
            # Direction-aware matching: reject plan points whose travel
            # direction is more than acos(min_tangent_dot) from the gradient
            # direction at the query point. This prevents a U-turn's return
            # leg from absorbing gradient path points on the outbound leg and
            # reporting a spuriously small distance.
            grad_tan = _path_tangents(future_grad)  # (N, 2)
            plan_tan = _path_tangents(future_plan)  # (M, 2)
            dot = grad_tan @ plan_tan.T  # (N, M)
            invalid = dot < self._cfg.path_diff_min_tangent_dot
            sq_dist = np.where(invalid, np.inf, sq_dist)

        min_sq = sq_dist.min(axis=1)  # (N,)
        min_dist = np.where(np.isfinite(min_sq), np.sqrt(min_sq), np.inf)

        # --- Check 1: global percentile over finite deviations ---
        # inf values (no valid directional match) are excluded from the
        # percentile; the windowed check handles them.
        finite_mask = np.isfinite(min_dist)
        pct_val = (
            float(np.percentile(min_dist[finite_mask], self._cfg.path_diff_percentile))
            if finite_mask.any()
            else 0.0
        )
        pct_triggered = pct_val > self._cfg.path_diff_threshold

        # --- Check 2: sliding window maximum ---
        # Catches short but severe detours that the global percentile dilutes.
        # inf values propagate naturally through np.max, so a window of
        # topologically-unmatched points (e.g. the apex of a U-turn the plan
        # has no segment for) always fires.
        win_val = 0.0
        win_triggered = False
        if self._cfg.path_diff_window_size > 0.0:
            segs = np.hypot(np.diff(future_grad[:, 0]), np.diff(future_grad[:, 1]))
            arc_s = np.concatenate([[0.0], np.cumsum(segs)])
            win_val = _windowed_max_deviation(
                min_dist, arc_s, self._cfg.path_diff_window_size
            )
            win_triggered = win_val > self._cfg.path_diff_threshold

        should_replan = pct_triggered or win_triggered
        return should_replan, pct_val, win_val
