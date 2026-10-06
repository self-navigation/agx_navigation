"""Closed-loop sanity check of the GMPC adapter on a kinematic unicycle.

Needs casadi + manifpy (the GMPC venv) and the vendored submodule; skips
otherwise. Pure python, no ROS. Run on the VM with
``~/.venvs/gmpc/bin/python -m pytest src/agx_navigation/agx_baselines/test -v``.
"""

import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from agx_baselines.gmpc.tracker import build_reference, se2_log  # noqa: E402

pytest.importorskip("casadi")
pytest.importorskip("manifpy")


def _plan(n=120, dt=0.1, v=0.4, w=0.3):
    poses = np.zeros((n, 3))
    for k in range(1, n):
        x, y, th = poses[k - 1]
        poses[k] = (x + v * math.cos(th) * dt, y + v * math.sin(th) * dt, th + w * dt)
    return poses, np.arange(n) * dt


def test_se2_log_roundtrip():
    xi = se2_log(0.1, 0.02, 0.3)
    th = xi[2]
    a, b = math.sin(th) / th, (1 - math.cos(th)) / th
    V = np.array([[a, -b], [b, a]])
    assert np.allclose(V @ xi[:2], [0.1, 0.02])


def test_reference_velocities_match_plan():
    poses, t = _plan()
    rs, rc = build_reference(poses, t, 0.05)
    assert rs.shape[1] == rc.shape[1]
    assert np.allclose(rs[:2, -1], poses[-1, :2])
    assert abs(math.remainder(rs[2, -1] - poses[-1, 2], 2 * math.pi)) < 1e-9
    mid = slice(10, rc.shape[1] - 10)
    assert np.allclose(rc[0, mid], 0.4, atol=0.02)
    assert np.allclose(rc[1, mid], 0.3, atol=0.02)
    assert rc[0, -1] == 0.0 and rc[1, -1] == 0.0


def test_unstamped_path_gets_cruise_time_law():
    poses, _ = _plan()
    rs, rc = build_reference(poses, None, 0.05, cruise_speed=0.25)
    assert np.allclose(rc[0, 10:-10], 0.25, atol=0.02)


def test_closed_loop_tracks_from_offset():
    from agx_baselines.gmpc.tracker import PlanTracker

    poses, t = _plan()
    dt = 0.05
    tr = PlanTracker(poses, t, dt)
    x = np.array([0.15, -0.1, 0.2])  # start off-plan
    errs, solves = [], []
    for k in range(tr.n + 20):
        v, w, s = tr.solve(x, k * dt)
        solves.append(s)
        assert abs(v) <= 0.5 + 1e-6 and abs(w) <= 1.5 + 1e-6
        x = x + dt * np.array([v * math.cos(x[2]), v * math.sin(x[2]), w])
        ref = tr.reference_at(k * dt)
        errs.append(math.hypot(ref[0] - x[0], ref[1] - x[1]))
    assert errs[-1] < 0.03, errs[-1]
    assert max(errs[tr.n // 2:]) < 0.05
    assert np.median(solves) < 0.05
