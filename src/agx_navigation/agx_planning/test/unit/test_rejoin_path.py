"""Tests for supervisor/rejoin_path.py, the Nav2 arms' plan|pmp re-plans."""
import math
import sys

import numpy as np

# conftest's rclpy MagicMock has no __spec__ (see test_rejoin.py): hide it for
# the solver imports, which need no ROS.
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.pmp_planner import PlannerConfig
    from agx_planning.pmp_planner.rejoin_service import RejoinSolver
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock
from agx_planning.supervisor import rejoin_path as rp
from agx_planning.supervisor.core import SupervisorConfig, SupervisorCore


def _arc(n=200, dt=0.05, v=0.4, w=0.3):
    th = np.arange(n) * dt * w
    x = np.cumsum(np.r_[0.0, v * dt * np.cos(th[:-1])])
    y = np.cumsum(np.r_[0.0, v * dt * np.sin(th[:-1])])
    return np.c_[x, y, th], dt, v, w


def test_wheel_speeds_recover_the_twist():
    poses, dt, v, w = _arc()
    cfg = PlannerConfig(mode="offline")
    wh = rp.wheel_speeds(poses, dt, cfg)
    vv, ww = cfg.wheels_to_body(wh[:, 0], wh[:, 1])
    assert np.allclose(vv, v, atol=1e-3) and np.allclose(ww, w, atol=1e-6)


def test_progress_is_monotone_within_window():
    poses, *_ = _arc()
    assert rp.progress(poses, *poses[50, :2], k0=0, window=400) == 50
    # the start is nearest overall, but progress never moves back past k0
    assert rp.progress(poses, *poses[0, :2], k0=30, window=400) == 30


def test_join_path_plan_vs_pmp_head():
    poses, *_ = _arc()
    p = rp.join_path((0.0, 0.5, 0.0), poses, 80)
    assert np.allclose(p[0], (0.0, 0.5, 0.0)) and np.allclose(p[-1], poses[-1])
    head = poses[40:81].copy()
    head[:, 1] += np.linspace(0.3, 0.0, len(head))
    q = rp.join_path(None, poses, 80, head)
    assert np.allclose(q[0], head[0]) and np.allclose(q[-1], poses[-1])
    assert np.all(np.hypot(*np.diff(q[:, :2], axis=0).T) < 0.06)


def test_stop_thresholds_match_the_core():
    cfg = SupervisorConfig()
    poses, *_ = _arc()
    c = np.linspace(0.05, 0.6, len(poses))
    core = SupervisorCore(cfg)
    core.set_plan(poses, np.arange(len(poses)) * 0.05, clearance=c)
    st = rp.stop_thresholds(cfg, c, 10, 70, 13)
    ks = np.linspace(10, 70, 13).round().astype(int)
    assert np.allclose(st, [core.wall_thresholds(int(k))[1] for k in ks])
    assert rp.stop_thresholds(cfg, None, 0, 5, 3) is None


def test_pmp_rejoin_onto_a_reference_rebuilt_from_poses():
    """The node's path: wheel speeds from poses, RejoinSolver, join_path."""
    from agx_planning.runtime_corrector.trajectory_buffer import PlaybackSample
    poses, dt, v, w = _arc(n=300)
    cfg = PlannerConfig(mode="offline")
    wh = rp.wheel_speeds(poses, dt, cfg)
    smp = [PlaybackSample(left=float(a), right=float(b), pose=tuple(map(float, p)))
           for p, (a, b) in zip(poses, wh)]
    rs = RejoinSolver(smp, dt=dt, T_w_min=3.0)
    k0 = 60
    tgt = k0 + int(math.ceil(3.0 / dt))
    x0 = np.r_[poses[k0, :2] + [0.0, 0.2], poses[k0, 2], wh[k0]]
    out = rs.solve(x0, tgt, k0)
    assert out is not None
    assert np.hypot(out[-1].pose[0] - poses[tgt, 0], out[-1].pose[1] - poses[tgt, 1]) < 0.02
    path = rp.join_path(None, poses, tgt, [s.pose for s in out])
    assert np.allclose(path[0, :2], x0[:2], atol=1e-6)
