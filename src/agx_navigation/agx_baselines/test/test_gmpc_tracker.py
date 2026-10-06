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

from agx_baselines.gmpc.tracker import (  # noqa: E402
    build_reference, plan_times, reference_index, retime_t0, se2_log)


def _needs_gmpc():
    # Only the closed-loop test needs the upstream solver; the reference and
    # retime arithmetic is pure numpy and runs anywhere.
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
    _needs_gmpc()
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


def test_plan_times_match_build_reference_grid():
    poses, t = _plan()
    assert np.allclose(plan_times(poses, t), t - t[0])
    # Unstamped: arc length over the cruise speed (no turn-in-place here).
    pt = plan_times(poses, None, cruise_speed=0.25)
    assert np.allclose(np.diff(pt), 0.4 * 0.1 / 0.25)


def test_retime_jumps_reference_to_plan_index():
    """/supervisor/retime (#42): after retiming at plan pose i, the reference
    the controller reads is the grid sample at plan_t[i], whatever the clock
    said before -- behind or ahead."""
    poses, t = _plan(n=120, dt=0.1)
    dt = 0.05
    rs, _ = build_reference(poses, t, dt)
    n = rs.shape[1]
    pt = plan_times(poses, t)
    now = 37.3
    for i in (40, 0, 119):
        t0 = retime_t0(now, pt, i)
        k = reference_index(now - t0, dt, n)
        assert k == round(pt[i] / dt)
        assert np.allclose(rs[:2, k], poses[i, :2], atol=1e-9)
        # The clock then runs on from there.
        assert reference_index(now + 1.0 - t0, dt, n) == min(k + 20, n - 1)
    # Out-of-range indices clamp to the plan.
    assert retime_t0(now, pt, 10_000) == now - pt[-1]
    assert retime_t0(now, pt, -5) == now
