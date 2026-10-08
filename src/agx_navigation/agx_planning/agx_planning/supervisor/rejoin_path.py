"""Re-plans that keep the PMP plan, for the Nav2 same-plan arms. ROS-free.

supervisor_node's core.replan_source picks what a follow_path re-plan sends:

  planner  ComputePathToPose from the current pose to the goal
  plan     the PMP plan's remainder from the re-join target tgt (rejoin_t_min
           seconds past the robot's progress), prefixed by the current pose
  pmp      a PMP re-join (rejoin_service) to the plan state at tgt, then the
           remainder; falls back to plan if the solve fails or hits a wall

The reference is the stamped PMP plan on /supervisor/pmp_plan; time base and
wheel speeds are recovered from it by finite differences.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def resample_path(poses, ds: float = 0.05) -> np.ndarray:
    """(x, y, yaw) rows at ~ds arc-length spacing; same contract as
    tools/compare_run.plan_to_path (yaw interpolated from the plan's own
    heading, zero-length steps dropped, final pose kept)."""
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    xy = poses[:, :2]
    yaw = np.unwrap(poses[:, 2])
    seg = np.hypot(*np.diff(xy, axis=0).T)
    keep = np.concatenate([[True], seg > 1e-6])
    xy, yaw = xy[keep], yaw[keep]
    yaw[-1] = np.unwrap(poses[:, 2])[-1]
    if len(xy) < 2:
        return np.array([[xy[0, 0], xy[0, 1], poses[-1, 2]]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))])
    n = max(2, int(math.ceil(s[-1] / ds)) + 1)
    q = np.linspace(0.0, s[-1], n)
    out = np.column_stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1]),
                           np.interp(q, s, yaw)])
    out[:, 2] = (out[:, 2] + math.pi) % (2 * math.pi) - math.pi
    return out


def wheel_speeds(poses, dt: float, cfg) -> np.ndarray:
    """(N, 2) wheel speeds (w_l, w_r) that drive ``poses`` at spacing ``dt``
    under the planner's model (``cfg``: PlannerConfig.body_to_wheels). Forward
    differences; the last sample repeats the one before it."""
    p = np.asarray(poses, dtype=float).reshape(-1, 3)
    if len(p) < 2:
        return np.zeros((len(p), 2))
    d = np.diff(p, axis=0)
    th = p[:-1, 2]
    v = (d[:, 0] * np.cos(th) + d[:, 1] * np.sin(th)) / dt
    w = ((d[:, 2] + math.pi) % (2 * math.pi) - math.pi) / dt
    wl, wr = cfg.body_to_wheels(v, w)
    out = np.column_stack([wl, wr])
    return np.vstack([out, out[-1:]])


def progress(poses, x: float, y: float, k0: int, window: int) -> int:
    """Nearest plan index to (x, y) in [k0, k0 + window]: the robot's
    progress on the reference, never moving backwards."""
    p = np.asarray(poses)
    hi = min(len(p), k0 + window + 1)
    d = np.hypot(p[k0:hi, 0] - x, p[k0:hi, 1] - y)
    return k0 + int(np.argmin(d))


def join_path(pose, ref_poses, tgt: int, rejoin_poses=None, ds: float = 0.05) -> np.ndarray:
    """The FollowPath path for a ``plan`` / ``pmp`` re-plan: the re-join's
    poses (``pmp``) or just the current pose (``plan``), then the reference
    from ``tgt`` to its end, resampled to ``ds``."""
    ref = np.asarray(ref_poses, dtype=float).reshape(-1, 3)
    head = (np.asarray(rejoin_poses, dtype=float).reshape(-1, 3)
            if rejoin_poses is not None and len(rejoin_poses)
            else np.asarray(pose, dtype=float).reshape(1, 3))
    tail = ref[tgt + 1:] if rejoin_poses is not None and len(rejoin_poses) else ref[tgt:]
    return resample_path(np.vstack([head, tail]), ds)


def stop_thresholds(core_cfg, c_plan, k0: int, tgt: int, n: int) -> Optional[np.ndarray]:
    """The core's plan-relative stop_at (SupervisorCore.wall_thresholds) at
    the reference index each of n re-join samples stands in for, linear over
    k0..tgt -- the runtime corrector's wall gate, applied to the reference."""
    if c_plan is None:
        return None
    c = np.asarray(c_plan, dtype=float)
    ks = np.minimum(np.linspace(k0, tgt, n).round().astype(int), len(c) - 1)
    return np.maximum(core_cfg.contact_dist,
                      np.minimum(core_cfg.stop_dist, c[ks] - core_cfg.stop_margin))
