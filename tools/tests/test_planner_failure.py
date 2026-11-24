"""Failure and ownership gates around the numerical BVP call.

The controlled splines below are test doubles for the BVP result. The baseline
tests exercise real converged BVPs and the same acceptance path.
"""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from scipy.interpolate import PPoly

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.pmp_planner import (
        PlannerConfig, PMPShootingSolver, goal_reached, goal_state_reached,
        rollout_generator,
    )
    from agx_planning.pmp_planner import shooting_solver
    from agx_planning.vector_field import VectorFieldGrid
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def field(offset=0.0):
    grid = VectorFieldGrid()
    grid.update(np.tile(100.0 - np.arange(100), (20, 1)) + offset, 0.0, 0.0, 0.1)
    return grid


@pytest.fixture
def problem():
    cfg = PlannerConfig(N=8, T_horizon=1.0, control_rate=10.0,
                        dt_segment=0.2, max_rollout_sim_time=2.0,
                        goal_tolerance_xy=0.05)
    return PMPShootingSolver(cfg, field()), np.array([1., 1., 0., 0., 0.]), np.array([4., 1., 0.])


def candidate(x0, horizon, *, speed=0.0, wheel_accel=0.0, pin_error=0.0, costate=0.0):
    knots = np.array([0., horizon])
    coefficients = np.zeros((2, 1, 10))
    coefficients[1, 0, :5] = x0
    coefficients[1, 0, 0] += pin_error
    coefficients[1, 0, 8:10] = costate
    coefficients[0, 0, 0] = speed
    coefficients[0, 0, 3:5] = wheel_accel
    spline = PPoly(coefficients.transpose(2, 0, 1), knots, axis=1)
    return SimpleNamespace(success=True, status=0, message="Controlled test result",
                           sol=spline, x=knots, y=spline(knots), niter=1,
                           rms_residuals=np.zeros(1))


def patch_candidate(monkeypatch, **kwargs):
    def solve(_ode, _bc, knots, guess, **_options):
        return candidate(guess[:5, 0], knots[-1], **kwargs)
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    return solve


def collect(generator):
    chunks = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as terminal:
            return chunks, terminal.value


@pytest.mark.parametrize("failure", ["exception", "mesh_exhausted", "singular_jacobian"])
def test_failed_call_cannot_sample_previous_success(problem, monkeypatch, failure):
    solver, x0, goal = problem
    patch_candidate(monkeypatch)
    assert solver.sample_committed_segment(x0, goal, 0.1, 2) is not None
    previous = solver._prev_sol

    def fail(_ode, _bc, knots, guess, **_options):
        if failure == "exception":
            raise RuntimeError("injected failure")
        result = candidate(guess[:5, 0], knots[-1])
        result.success = False
        result.status = 1 if failure == "mesh_exhausted" else 2
        return result

    monkeypatch.setattr(shooting_solver, "solve_bvp", fail)
    assert solver.sample_committed_segment(x0, goal, 0.1, 2) is None
    assert solver.last_outcome.status == failure
    assert not solver.last_outcome.success
    assert solver.last_outcome.solution is None
    assert solver.last_outcome.command is None
    assert solver._last_state is solver._last_costate is solver._last_segment_states is None
    # A valid cache may seed another attempt, but cannot become a current result.
    assert solver._prev_sol is previous
    assert solver.solve(x0, goal) is None


@pytest.mark.parametrize("which", ["start", "goal", "shape"])
def test_invalid_input_does_not_call_bvp(problem, monkeypatch, which):
    solver, x0, goal = problem
    monkeypatch.setattr(shooting_solver, "solve_bvp", lambda *a, **k: pytest.fail("BVP must not run"))
    if which == "start":
        x0[3] = np.nan
    elif which == "goal":
        goal[0] = np.inf
    else:
        x0 = x0[:3]
    result = solver.solve_outcome(x0, goal)
    assert result.status == "invalid_input"
    assert result.solution is None


@pytest.mark.parametrize("bad_result", ["pin", "mesh", "spline", "residual", "command"])
def test_success_flag_does_not_bypass_acceptance(problem, monkeypatch, bad_result):
    solver, x0, goal = problem
    def solve(_ode, _bc, knots, guess, **_options):
        result = candidate(guess[:5, 0], knots[-1], pin_error=0.01 if bad_result == "pin" else 0.)
        if bad_result == "mesh":
            result.y[0, 1] = np.nan
        elif bad_result == "spline":
            result.sol.c[-1, 0, 0] = np.inf
        elif bad_result == "residual":
            result.rms_residuals[0] = np.nan
        return result
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    if bad_result == "command":
        monkeypatch.setattr(solver, "_predistort", lambda *a, **k: (np.nan, 0.))
    result = solver.solve_outcome(x0, goal)
    expected = {"pin": "invalid_pin", "command": "invalid_controls"}.get(bad_result, "nonfinite_solution")
    assert result.status == expected
    assert not result.success and result.solution is None and result.command is None
    assert solver._prev_sol is None


def test_finite_controls_obey_acceleration_bound(problem, monkeypatch):
    solver, x0, goal = problem
    patch_candidate(monkeypatch, costate=1e6)
    segment = solver.sample_committed_segment(x0, goal, 0.1, 2)
    assert segment is not None
    assert np.isfinite(segment[1]).all()
    assert np.max(np.abs(segment[1])) <= solver.cfg.a_wheel_max
    assert solver.last_outcome.diagnostics.max_accel == solver.cfg.a_wheel_max


def test_cancel_during_bvp_rejects_result(problem, monkeypatch):
    solver, x0, goal = problem
    stopped = False
    def solve(_ode, _bc, knots, guess, **_options):
        nonlocal stopped
        stopped = True
        return candidate(guess[:5, 0], knots[-1])
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    result = solver.solve_outcome(x0, goal, stop_fn=lambda: "cancelled" if stopped else None)
    assert result.status == "cancelled" and result.solution is None
    assert solver._prev_sol is None


def test_deadline_during_bvp_rejects_result(problem, monkeypatch):
    solver, x0, goal = problem
    now = 0.0
    monkeypatch.setattr(shooting_solver.time, "monotonic", lambda: now)
    def solve(_ode, _bc, knots, guess, **_options):
        nonlocal now
        now = 2.0
        return candidate(guess[:5, 0], knots[-1])
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    result = solver.solve_outcome(x0, goal, deadline=1.0)
    assert result.status == "deadline_exceeded" and result.solution is None
    assert solver._prev_sol is None


def test_rollout_cancel_during_solve_yields_no_chunk(problem, monkeypatch):
    solver, x0, goal = problem
    stopped = False
    original = solver.field
    def solve(_ode, _bc, knots, guess, **_options):
        nonlocal stopped
        stopped = True
        return candidate(guess[:5, 0], knots[-1])
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    chunks, result = collect(rollout_generator(
        solver, solver.cfg, x0, goal, lambda: "cancelled" if stopped else None))
    assert chunks == [] and result.status == "cancelled"
    assert solver.last_outcome.solution is None and solver._prev_sol is None
    assert solver.field is original


def test_rollout_field_replacement_during_solve_yields_no_chunk(problem, monkeypatch):
    solver, x0, goal = problem
    replacement = field(1.0)
    def solve(_ode, _bc, knots, guess, **_options):
        solver.field = replacement
        return candidate(guess[:5, 0], knots[-1])
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert chunks == [] and result.solve_status == "field_changed"
    assert solver.field is replacement
    assert solver.last_outcome.solution is None


def test_rollout_has_raw_boundary_states_and_full_spline(problem, monkeypatch):
    solver, x0, goal = problem
    goal[0] = 1.3
    patch_candidate(monkeypatch, speed=1.0)
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert result.status == "success"
    assert len(chunks) == 2
    assert len(chunks[-1].wheel_cmds) == 1
    for chunk in chunks:
        n = len(chunk.wheel_cmds)
        assert chunk.raw_states.shape == (n + 1, 5)
        np.testing.assert_allclose(chunk.raw_states[:-1, :3], chunk.poses)
        np.testing.assert_allclose(chunk.raw_states,
                                   chunk.solution(np.arange(n + 1) * chunk.dt_sample)[:5].T)
        assert chunk.solution.x[-1] == solver.cfg.T_horizon
        assert not chunk.raw_states.flags.writeable
    for left, right in zip(chunks, chunks[1:]):
        np.testing.assert_allclose(left.raw_states[-1], right.raw_states[0], atol=1e-6)
    np.testing.assert_array_equal(x0, [1., 1., 0., 0., 0.])


def test_near_goal_stagnation_is_not_success(problem, monkeypatch):
    solver, x0, goal = problem
    goal[0] = x0[0] + 0.1
    patch_candidate(monkeypatch)
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert len(chunks) >= 5
    assert result.status == "stagnated"


@pytest.mark.parametrize("wheels, accepted", [
    ([0., 0.], True), ([0.5, -0.5], True),
    ([0.500001, 0.], False), ([0., -0.500001], False),
    ([1., -1.], False), ([np.nan, 0.], False), ([0., np.inf], False),
])
def test_goal_state_requires_both_raw_wheel_speeds(problem, wheels, accepted):
    solver, _, goal = problem
    state = np.r_[goal, wheels]
    assert goal_reached(state, goal, solver.cfg)  # Pose remains a separate check.
    assert goal_state_reached(state, goal, solver.cfg) is accepted
    assert not goal_state_reached(state[:3], goal, solver.cfg)
    state[0] += 0.1
    assert not goal_state_reached(state, goal, solver.cfg)


def test_rollout_at_goal_still_plans_until_wheels_slow(problem, monkeypatch):
    solver, x0, goal = problem
    goal[:] = x0[:3]
    x0[3:] = 1.0
    patch_candidate(monkeypatch, wheel_accel=-2.0)
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert result.status == "success"
    assert [len(chunk.wheel_cmds) for chunk in chunks] == [2, 1]
    assert np.max(np.abs(chunks[0].raw_states[-1, 3:])) > 0.5
    np.testing.assert_allclose(chunks[-1].raw_states[-1, 3:], [0.4, 0.4])


def test_rollout_stopped_at_goal_needs_no_solve(problem, monkeypatch):
    solver, x0, goal = problem
    goal[:] = x0[:3]
    monkeypatch.setattr(shooting_solver, "solve_bvp", lambda *a, **k: pytest.fail("No solve needed"))
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert chunks == [] and result.status == "success"


def test_rollout_skips_fast_pose_hit_and_truncates_at_stopped_boundary(problem, monkeypatch):
    solver, x0, goal = problem
    solver.cfg.dt_segment = 0.5
    x0[3:] = [0.8, -0.8]
    goal[0] = x0[0] + 0.06
    patch_candidate(monkeypatch, speed=0.2, wheel_accel=[-1., 1.])
    # Published commands cannot stand in for raw state: force a zero command.
    monkeypatch.setattr(solver, "_predistort", lambda wl_state, wr_state, lam_wl, lam_wr: (
        np.zeros_like(wl_state), np.zeros_like(wr_state)))
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert result.status == "success" and len(chunks) == 1
    chunk = chunks[0]
    assert len(chunk.wheel_cmds) == 3
    assert chunk.raw_states.shape == (4, 5)
    assert len(chunk.accels) == len(chunk.poses) == len(chunk.costates) == 3
    assert goal_reached(chunk.raw_states[1], goal, solver.cfg)
    assert not goal_state_reached(chunk.raw_states[1], goal, solver.cfg)
    assert goal_state_reached(chunk.raw_states[-1], goal, solver.cfg)
    assert not chunk.wheel_cmds.any()


def test_rollout_accepts_final_boundary_at_time_cap_without_extra_solve(problem, monkeypatch):
    solver, x0, goal = problem
    solver.cfg.max_rollout_sim_time = solver.cfg.dt_segment
    goal[:] = x0[:3]
    x0[3:] = 0.75
    calls = []
    def solve(_ode, _bc, knots, guess, **_options):
        calls.append(1)
        return candidate(guess[:5, 0], knots[-1], wheel_accel=-1.25)
    monkeypatch.setattr(shooting_solver, "solve_bvp", solve)
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert result.status == "success" and len(calls) == 1
    assert len(chunks) == 1 and len(chunks[0].wheel_cmds) == 2
    np.testing.assert_allclose(chunks[0].raw_states[-1, 3:], [0.5, 0.5])


def test_rollout_fast_goal_state_at_time_cap_is_not_success(problem, monkeypatch):
    solver, x0, goal = problem
    solver.cfg.max_rollout_sim_time = solver.cfg.dt_segment
    goal[:] = x0[:3]
    x0[3:] = 1.0
    patch_candidate(monkeypatch)
    chunks, result = collect(rollout_generator(solver, solver.cfg, x0, goal))
    assert len(chunks) == 1 and result.status == "failed"


def test_warm_start_requires_field_identity_and_goal(problem, monkeypatch):
    solver, x0, goal = problem
    patch_candidate(monkeypatch)
    assert solver.solve_outcome(x0, goal).success
    assert not solver.last_outcome.diagnostics.warm_start
    assert solver.solve_outcome(x0, goal).diagnostics.warm_start
    version = solver.field.version
    solver.field = field(1.0)
    assert solver.field.version == version
    assert not solver.solve_outcome(x0, goal).diagnostics.warm_start
    goal[2] += 0.1
    assert not solver.solve_outcome(x0, goal).diagnostics.warm_start


def test_real_bvp_success_then_exception_has_no_stale_segment(monkeypatch):
    cfg = PlannerConfig(N=20, T_horizon=1.0, control_rate=10.0, dt_segment=0.2)
    grid = VectorFieldGrid()
    travel_time = np.tile(np.abs(4.025 - (np.arange(100) + 0.5) * 0.05) / 0.5, (60, 1))
    grid.update(travel_time, 0., 0., 0.05)
    solver = PMPShootingSolver(cfg, grid)
    x0, goal = np.array([1.025, 1.025, 0., 0., 0.]), np.array([4.025, 1.025, 0.])
    segment = solver.sample_committed_segment(x0, goal, 0.1, 2)
    assert segment is not None, solver.last_outcome.message
    assert solver.last_outcome.success and solver.last_outcome.diagnostics.nodes > 0
    np.testing.assert_allclose(solver._last_segment_states[0], x0, atol=1e-6)
    np.testing.assert_allclose(solver._last_segment_states[-1], segment[-1])
    previous = solver.last_outcome.solution

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected failure after real convergence")
    monkeypatch.setattr(shooting_solver, "solve_bvp", fail)
    assert solver.sample_committed_segment(segment[-1], goal, 0.1, 2) is None
    assert solver.last_outcome.status == "exception"
    assert solver.last_outcome.solution is None
    assert solver._last_segment_states is None
    assert solver._prev_sol is previous
