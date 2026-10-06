"""ROS-free adapter that makes the upstream GeometricMPC track our plans.

PlanTracker subclasses GeometricMPC and overrides only set_ref_traj; solve and
the rest of the controller are upstream. A plan is a list of (x, y, yaw) poses,
either timed (the PMP plan's dt_sample grid) or untimed (an arc-length path
driven at a constant cruise speed). Poses are resampled onto the controller's
dt and the reference twist is the SE(2) log of consecutive relative poses over
dt. The last reference velocity is zero, so after the plan ends the controller
regulates to the final pose. The lateral twist component is discarded.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np


def _gmpc_root() -> str:
    """Directory holding upstream's ``controller/``, ``planner/``, ``utils/``.

    Installed copy (share/agx_baselines/third_party/gmpc) first, the source
    submodule second, so a symlink-install or a plain ``python -m`` both work.
    """
    override = os.environ.get("AGX_GMPC_SRC")
    if override:
        return override
    try:
        from ament_index_python.packages import get_package_share_directory

        share = os.path.join(get_package_share_directory("agx_baselines"),
                             "third_party", "gmpc")
        if os.path.isdir(os.path.join(share, "controller")):
            return share
    except Exception:  # noqa: BLE001 -- not installed / no ament: fall through
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "third_party",
                                         "GMPC-Tracking-Control"))


def _import_upstream():
    root = _gmpc_root()
    if not os.path.isdir(os.path.join(root, "controller")):
        raise ImportError(
            f"GMPC upstream sources not found under {root}; init the submodule "
            "(git submodule update --init src/agx_navigation/agx_baselines/third_party/"
            "GMPC-Tracking-Control) and rebuild, or set AGX_GMPC_SRC")
    if root not in sys.path:
        sys.path.insert(0, root)
    # matplotlib is imported at module level upstream; keep it off any display.
    os.environ.setdefault("MPLBACKEND", "Agg")
    from controller.geometric_mpc import GeometricMPC  # noqa: E402

    return GeometricMPC


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def se2_log(dx: float, dy: float, dth: float) -> np.ndarray:
    """Log of the SE(2) element with translation (dx, dy) and rotation dth.

    Returns tangent coeffs (rho_x, rho_y, theta) in manif's ordering. Used only
    to build the reference twists; the controller's own log is manif's.
    """
    th = wrap_angle(dth)
    if abs(th) < 1e-9:
        return np.array([dx, dy, th])
    a = math.sin(th) / th
    b = (1.0 - math.cos(th)) / th
    det = a * a + b * b
    vinv = np.array([[a, b], [-b, a]]) / det
    rho = vinv @ np.array([dx, dy])
    return np.array([rho[0], rho[1], th])


def build_reference(poses: np.ndarray, times: np.ndarray | None, dt: float,
                    cruise_speed: float = 0.4, min_speed: float = 0.05):
    """Resample a plan onto a uniform ``dt`` grid and derive (v, w) per sample.

    Pure. Returns ``(ref_state 3xM, ref_control 2xM)``. ``times`` may be None
    (arc-length time law at ``cruise_speed``) or an (N,) array of seconds
    measured from the first pose; it must be non-decreasing.
    """
    poses = np.asarray(poses, dtype=float).reshape(-1, 3)
    if poses.shape[0] == 1:
        poses = np.vstack([poses, poses])
    yaw = np.unwrap(poses[:, 2])
    if times is None or len(times) != len(poses) or float(np.ptp(times)) <= 0.0:
        seg = np.hypot(*np.diff(poses[:, :2], axis=0).T)
        # Turn-in-place samples have no arc length; charge them the time the
        # heading change takes at a modest rate so they are not collapsed.
        turn = np.abs(np.diff(yaw)) / 1.0
        times = np.concatenate([[0.0], np.cumsum(np.maximum(
            seg / max(cruise_speed, min_speed), turn))])
    else:
        times = np.asarray(times, dtype=float) - float(times[0])
        times = np.maximum.accumulate(times)
    # Collapse duplicate timestamps so interp is well defined.
    keep = np.concatenate([[True], np.diff(times) > 1e-9])
    times, xy, yaw = times[keep], poses[keep, :2], yaw[keep]
    if len(times) < 2:
        times = np.array([0.0, dt])
        xy = np.vstack([xy, xy])
        yaw = np.concatenate([yaw, yaw])
    n = int(math.floor(times[-1] / dt)) + 1
    grid = np.arange(n) * dt
    rx = np.interp(grid, times, xy[:, 0])
    ry = np.interp(grid, times, xy[:, 1])
    ryaw = np.interp(grid, times, yaw)
    # Always end exactly on the plan's final pose.
    if grid[-1] < times[-1] - 1e-9:
        rx = np.append(rx, xy[-1, 0])
        ry = np.append(ry, xy[-1, 1])
        ryaw = np.append(ryaw, yaw[-1])
        n += 1
    ref_state = np.vstack([rx, ry, np.array([wrap_angle(a) for a in ryaw])])
    ref_control = np.zeros((2, n))
    for k in range(n - 1):
        c, s = math.cos(ryaw[k]), math.sin(ryaw[k])
        dxw, dyw = rx[k + 1] - rx[k], ry[k + 1] - ry[k]
        # World displacement into the body frame of sample k.
        dx, dy = c * dxw + s * dyw, -s * dxw + c * dyw
        xi = se2_log(dx, dy, ryaw[k + 1] - ryaw[k]) / dt
        ref_control[0, k] = xi[0]
        ref_control[1, k] = xi[2]
    return ref_state, ref_control


class PlanTracker:
    """Upstream ``GeometricMPC`` driven by one of our plans.

    ``horizon``, ``q`` (3 weights), ``r`` default to upstream's values
    (N=10, Q=diag(20000, 20000, 2000), R=0.3) -- the paper's, not ours.
    """

    def __init__(self, poses, times, dt: float, *, horizon: int = 10,
                 q=(20000.0, 20000.0, 2000.0), r: float = 0.3,
                 v_bounds=(-0.5, 0.5), w_bounds=(-1.5, 1.5),
                 cruise_speed: float = 0.4):
        GeometricMPC = _import_upstream()
        ref_state, ref_control = build_reference(poses, times, dt, cruise_speed)
        self.dt = float(dt)
        self.ref_state = ref_state
        self.ref_control = ref_control
        self.n = ref_state.shape[1]
        self.duration = (self.n - 1) * self.dt

        class _Ours(GeometricMPC):
            # The one hook: upstream builds its reference from a TrajGenerator
            # config; we hand it the arrays directly.
            def set_ref_traj(inner, cfg):
                inner.ref_state, inner.ref_control, inner.dt = cfg
                inner.nTraj = inner.ref_state.shape[1]

        self.mpc = _Ours((ref_state, ref_control, self.dt))
        self.mpc.setup_solver(Q=list(q), R=float(r), N=int(horizon))
        self.mpc.set_control_bound(v_min=v_bounds[0], v_max=v_bounds[1],
                                   w_min=w_bounds[0], w_max=w_bounds[1])

    def reference_at(self, t: float) -> np.ndarray:
        k = min(max(int(round(t / self.dt)), 0), self.n - 1)
        return self.ref_state[:, k]

    def final_pose(self) -> np.ndarray:
        return self.ref_state[:, -1]

    def solve(self, pose, t: float):
        """(v, w) for the robot at ``pose=(x, y, yaw)`` and plan time ``t``.

        ``t`` is clamped to the reference, which turns the tail into pose
        regulation; upstream indexes ``ref_state[:, round(t/dt)]`` unclamped
        and would raise past the end.
        """
        t = min(max(float(t), 0.0), self.duration)
        u = np.asarray(self.mpc.solve(np.asarray(pose, dtype=float), t),
                       dtype=float).reshape(-1)
        return float(u[0]), float(u[1]), float(self.mpc.get_solve_time())
