"""Tests for the re-join TPBVP (PMPShootingSolver.solve_rejoin), Phase 0 of the
re-join re-planner (docs/corrector-design.md).

Phase 0 MEASURES a failure rate, so the failure path matters as much as the
success path: a solve that reports success without reaching its pins would make
the rate look better than it is, and one that raises would crash the sweep
instead of counting as a failure.

Pure: a synthetic travel-time field, no map, no skfmm, no ROS.
"""

import sys

import numpy as np
import pytest

# conftest.py mocks rclpy with a MagicMock, which has no __spec__, so the
# packages' `find_spec("rclpy")` guard raises instead of returning False. Hide
# the mock for this import only: the solver itself needs no ROS.
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.vector_field import VectorFieldGrid
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def _straight_field(goal_x=20.0):
    """T = distance to x = goal_x along +x: F points +x everywhere."""
    res = 0.05
    xs = np.arange(0.0, 25.0, res)
    ys = np.arange(-5.0, 5.0, res)
    T = np.tile(np.abs(goal_x - xs) / 0.5, (ys.size, 1))
    grid = VectorFieldGrid()
    grid.update(T, 0.0, -5.0, res)
    return grid


def _solver():
    cfg = PlannerConfig(mode="offline")
    return PMPShootingSolver(cfg, _straight_field()), cfg


def _cruise(cfg, x, y=0.0, v=0.3):
    wl, wr = cfg.body_to_wheels(v, 0.0)
    return np.array([x, y, 0.0, wl, wr])


GOAL = np.array([20.0, 0.0, 0.0])


@pytest.mark.parametrize("cost", ["field", "effort"])
def test_on_plan_start_reaches_all_five_pins(cost):
    solver, cfg = _solver()
    T_w = 2.0
    x0 = _cruise(cfg, 5.0)
    x_t = _cruise(cfg, 5.0 + 0.3 * T_w)
    r = solver.solve_rejoin(x0, x_t, T_w, GOAL, cost=cost)
    assert r["success"], r["message"]
    yT = r["sol"](T_w)
    np.testing.assert_allclose(yT[0:5], x_t, atol=1e-3)
    np.testing.assert_allclose(r["sol"](0.0)[0:5], x0, atol=1e-3)


@pytest.mark.parametrize("cost", ["field", "effort"])
def test_lateral_offset_rejoins(cost):
    """The case the network exists for: 0.3 m off to the side, 3 s to fix it."""
    solver, cfg = _solver()
    T_w = 3.0
    x0 = _cruise(cfg, 5.0, y=0.3)
    x_t = _cruise(cfg, 5.0 + 0.3 * T_w)
    r = solver.solve_rejoin(x0, x_t, T_w, GOAL, cost=cost)
    assert r["success"], r["message"]
    np.testing.assert_allclose(r["sol"](T_w)[0:5], x_t, atol=1e-3)
    assert r["max_accel"] <= cfg.a_wheel_max + 1e-9


def test_effort_cost_on_plan_needs_no_acceleration():
    """Min-effort from a state already on a constant-speed line to its own
    future point: the optimum is zero acceleration throughout."""
    solver, cfg = _solver()
    T_w = 1.5
    x0 = _cruise(cfg, 5.0)
    r = solver.solve_rejoin(x0, _cruise(cfg, 5.0 + 0.3 * T_w), T_w, GOAL, cost="effort")
    assert r["success"]
    assert r["max_accel"] < 1e-2


def test_target_heading_is_unwrapped():
    """A target heading 2*pi away is the SAME heading: no full spin requested."""
    solver, cfg = _solver()
    T_w = 2.0
    x0 = _cruise(cfg, 5.0)
    x_t = _cruise(cfg, 5.0 + 0.3 * T_w)
    x_t[2] += 2.0 * np.pi
    r = solver.solve_rejoin(x0, x_t, T_w, GOAL, cost="effort")
    assert r["success"]
    assert abs(r["sol"](T_w)[2]) < 1e-3


def test_infeasible_pin_is_reported_not_faked():
    """1 m sideways in 0.1 s is beyond the acceleration bound. The measurement
    must see a failure (or a solution that visibly breaks the bound), never a
    clean success."""
    solver, cfg = _solver()
    x0 = _cruise(cfg, 5.0, y=1.0)
    r = solver.solve_rejoin(x0, _cruise(cfg, 5.03), 0.1, GOAL, cost="effort")
    assert r["status"] in ("ok", "fail", "exception")
    if r["success"]:
        yT = r["sol"](0.1)
        assert np.max(np.abs(yT[0:5] - _cruise(cfg, 5.03))) > 1e-3 or r["max_accel"] > 0.99 * cfg.a_wheel_max


def test_does_not_disturb_the_main_solve_state():
    solver, cfg = _solver()
    sentinel = object()
    solver._prev_sol = sentinel
    solver._theta_pursuit = 0.123
    solver.solve_rejoin(_cruise(cfg, 5.0), _cruise(cfg, 5.6), 2.0, GOAL, cost="field")
    assert solver._prev_sol is sentinel
    assert solver._theta_pursuit == 0.123


def _plan_samples(cfg, n=80, v=0.3, dt=0.1):
    from agx_planning.runtime_corrector.trajectory_buffer import PlaybackSample
    wl, wr = cfg.body_to_wheels(v, 0.0)
    return [PlaybackSample(left=wl, right=wr, pose=(1.0 + v * dt * i, 0.0, 0.0))
            for i in range(n)]


def _rejoin_solver(**kw):
    from agx_planning.pmp_planner.rejoin_service import RejoinSolver
    cfg = PlannerConfig(mode="offline")
    return RejoinSolver(_plan_samples(cfg), None, cfg, dt=0.1, **kw), cfg


def test_rejoin_service_pins_first_and_last_sample():
    rs, cfg = _rejoin_solver()
    plan = rs.plan
    x0 = plan[10] + np.array([0.0, 0.25, 0.1, 0.3, -0.2])
    out = rs.solve(x0, 45, k_now=10)
    assert out is not None, rs.last
    assert len(out) == 36
    np.testing.assert_allclose([*out[0].pose, out[0].left, out[0].right], x0, atol=1e-3)
    np.testing.assert_allclose([*out[-1].pose, out[-1].left, out[-1].right], plan[45], atol=1e-3)
    assert all(s.costates is None for s in out)


def test_rejoin_service_stretches_short_window_to_T_w_min():
    rs, cfg = _rejoin_solver()
    x0 = rs.plan[10] + np.array([0.0, 0.1, 0.0, 0.0, 0.0])
    out = rs.solve(x0, 15, k_now=10)  # 0.5 s away -> 3.0 s window
    assert out is not None
    assert rs.last["stretched"] and abs(rs.last["T_w"] - 3.0) < 1e-9
    assert len(out) == 31
    np.testing.assert_allclose(out[-1].pose, rs.plan[15, 0:3], atol=1e-3)


def test_rejoin_service_failure_returns_none():
    # max_nodes below the initial mesh: solve_bvp cannot converge -> None.
    rs, cfg = _rejoin_solver(max_nodes=5)
    x0 = rs.plan[10] + np.array([0.0, 1.5, 2.5, 5.0, -5.0])
    assert rs.solve(x0, 45, k_now=10) is None
    assert rs.last["status"] != "ok"


def test_label_knots_reproduce_solution():
    from agx_planning.pmp_planner.rejoin_service import body_knots
    rs, cfg = _rejoin_solver()
    x0 = rs.plan[10] + np.array([0.0, 0.2, 0.0, 0.0, 0.0])
    out = rs.solve(x0, 45, k_now=10)
    assert out is not None
    T_w = rs.last["T_w"]
    kn = body_knots(rs.last["sol"], T_w, cfg).reshape(6, 2)
    # knots at t = 0, 0.7, ..., 3.5 s land on samples 0, 7, ..., 35
    for j, i in enumerate(range(0, 36, 7)):
        v, om = cfg.wheels_to_body(out[i].left, out[i].right)
        np.testing.assert_allclose(kn[j], [v, om], atol=1e-3)


def test_unknown_cost_raises():
    solver, cfg = _solver()
    with pytest.raises(ValueError):
        solver.solve_rejoin(_cruise(cfg, 5.0), _cruise(cfg, 5.6), 2.0, GOAL, cost="nope")
