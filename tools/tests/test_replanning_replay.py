"""Check declared execution models and complete bounded replay intervals."""

from dataclasses import replace
from pathlib import Path
import math
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.interpolate import PPoly

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.maps import GridGeometry, MapSnapshot, OccupancyPolicy
    from agx_planning.vector_field.contracts import FieldKey
    from agx_planning.pmp_planner.plan_store import PlanRecord, SegmentSolution
    from agx_planning.replanning.replay import command_step, replay_plan
    from agx_planning.replanning.validity import Footprint, GridCollisionChecker
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def plan_record(coefficients=None):
    key = FieldKey(map_session="execution-map", map_revision=1,
                   geometry=GridGeometry((100, 100), 0.1), goal_cell=(60, 50),
                   recipe_id="test", numerical_version="test-v1",
                   occupancy_policy_id="occupied-65")
    if coefficients is None:
        coefficients = np.zeros((2, 1, 10))
        coefficients[0, 0, 0] = 1.0
        coefficients[1, 0, :5] = [1.5, 1.5, 0, 1, 1]
    polynomial = PPoly(coefficients, [0., 1.])
    times = np.array([0., 0.1, 0.2, 0.3])
    values = polynomial(times)
    return PlanRecord(session_id="execution-session", plan_id="p1", parent_id=None,
                      field_key=key, start_epoch=0., times=times,
                      raw_states=values[:, :5], commands=np.full((3, 2), 2.),
                      accels=np.zeros((3, 2)), costates=values[:-1, 5:],
                      segments=(SegmentSolution.from_ppoly(polynomial, key),))


def test_constant_wheel_step_matches_straight_and_half_circle():
    start = [1., 2., 0., 0., 0.]
    straight = command_step(start, [3., 3.], 0.5, 0.5, 0.5)
    np.testing.assert_allclose(straight, [2.5, 2., 0., 3., 3.], atol=1e-12)
    arc = command_step(start, [0., 2.], math.pi, 0.5, 0.5)
    np.testing.assert_allclose(arc, [1., 4., math.pi, 0., 2.], atol=1e-12)


def test_command_and_nominal_replay_keep_distinct_wheel_states_and_full_duration():
    plan = plan_record()
    command = replay_plan(plan, 0.5, 0.5, mode="command", max_step=0.07)
    nominal = replay_plan(plan, 0.5, 0.5, mode="nominal", max_step=0.07)
    assert command.times[-1] == nominal.times[-1] == plan.times[-1]
    assert len(command.times) == len(nominal.times) == 7
    np.testing.assert_allclose(command.states[-1], [2.1, 1.5, 0, 2, 2], atol=1e-12)
    np.testing.assert_allclose(nominal.states[-1], plan.raw_states[-1], atol=1e-12)
    assert nominal.states[-1, 0] == pytest.approx(1.8)
    # The full one-second BVP is retained, but only its 0.3 s prefix is replayed.
    assert plan.segments[0].duration == 1.0
    assert nominal.times[-1] == 0.3
    for value in (command.times, command.states, command.motion_bounds):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_nominal_motion_bounds_cover_polynomial_translation_and_rotation():
    coefficients = np.zeros((4, 1, 10))
    coefficients[:, 0, 0] = [12., -2., 1., 1.5]
    coefficients[:, 0, 1] = [-5., 1., 0., 1.5]
    coefficients[:, 0, 2] = [15., -3., 2., 0.]
    polynomial = PPoly(coefficients, [0., 1.])
    footprint = Footprint(0.2, 0.1)
    replay = replay_plan(plan_record(coefficients), 0.5, 0.5, mode="nominal",
                         max_step=0.08, footprint=footprint)
    derivative = polynomial.derivative()
    for left, right, bound in zip(replay.times[:-1], replay.times[1:], replay.motion_bounds):
        actual, _ = quad(lambda t: math.hypot(*derivative(t)[:2])
                         + footprint.radius * abs(derivative(t)[2]), left, right)
        assert bound >= actual - 1e-12


def test_command_replay_cannot_skip_a_wall_between_clear_samples():
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 18] = True
    footprint = Footprint(0.05, 0.05)
    checker = GridCollisionChecker(mask, 0.1, map_revision="execution-map/1", footprint=footprint)
    replay = replay_plan(plan_record(), 0.5, 0.5, mode="command", max_step=1., footprint=footprint)
    result = checker.check_path(replay.states[:, :3], replay.times,
                                interval_motion_bounds=replay.motion_bounds)
    assert not result.valid
    assert result.first_unsafe_interval is not None


def test_nominal_replay_rejects_missing_horizon_and_changed_start():
    plan = plan_record()
    with pytest.raises(ValueError, match="matching raw states"):
        replay_plan(plan, 0.5, 0.5, mode="nominal", initial_state=[1.6, 1.5, 0, 1, 1])
    with pytest.raises(ValueError, match="stored splines"):
        replay_plan(replace(plan, segments=()), 0.5, 0.5, mode="nominal")
