"""Geometry and braking checks use filled cells and bounded motion."""

from pathlib import Path
from dataclasses import FrozenInstanceError
import math
import sys

import numpy as np
import pytest
from scipy.integrate import quad

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

from agx_planning.replanning.validity import Footprint, GridCollisionChecker
from agx_planning.replanning.braking import BrakeConfig, build_brake, reserve_brake, reserve_prefix_brake


def checker(mask=None, footprint=Footprint(0.1, 0.1), resolution=0.1):
    if mask is None:
        mask = np.zeros((60, 60), dtype=bool)
    return GridCollisionChecker(mask, resolution, map_revision="map/session/7", footprint=footprint)


def clip_polygon(poly, axis, value, keep_lower):
    """Independent polygon clipping oracle; no separating-axis tests."""
    if not len(poly):
        return poly
    result = []
    for start, end in zip(poly, np.roll(poly, -1, axis=0)):
        inside_start = start[axis] <= value if keep_lower else start[axis] >= value
        inside_end = end[axis] <= value if keep_lower else end[axis] >= value
        if inside_start != inside_end:
            amount = (value - start[axis]) / (end[axis] - start[axis])
            result.append(start + amount * (end - start))
        if inside_end:
            result.append(end)
    return np.asarray(result)


def oracle_overlap(pose, footprint, row, col, resolution):
    corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [footprint.half_length, footprint.half_width]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    poly = corners @ np.array([[c, s], [-s, c]]) + pose[:2]
    for axis, lower in ((0, col * resolution), (1, row * resolution)):
        poly = clip_polygon(poly, axis, lower, False)
        poly = clip_polygon(poly, axis, lower + resolution, True)
    if len(poly) < 3:
        return False
    area = abs(np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - poly[:, 1] * np.roll(poly[:, 0], -1))) / 2
    return area > 1e-12


def test_filled_body_detects_cell_wholly_inside_and_keeps_map_identity():
    mask = np.zeros((60, 60), dtype=bool)
    mask[29, 29] = True
    c = checker(mask, Footprint(0.6, 0.4))
    result = c.check_pose([3, 3, 0])
    assert not result.valid and result.reason == "occupied"
    assert result.map_revision == "map/session/7"
    assert result.clearance_lower_bound < 0
    mask[:] = False
    assert not c.check_pose([3, 3, 0]).valid
    assert not c.occupied.flags.writeable
    with pytest.raises(ValueError):
        c.occupied.flags.writeable = True
    with pytest.raises(FrozenInstanceError):
        c.map_revision = "different"


def test_rotated_rectangles_match_independent_polygon_cell_oracle():
    rng = np.random.default_rng(412)
    mask = rng.random((9, 11)) < 0.18
    footprint = Footprint(0.27, 0.11)
    c = checker(mask, footprint, resolution=0.5)
    cells = np.argwhere(mask)
    for _ in range(250):
        pose = np.array([rng.uniform(0.6, 4.9), rng.uniform(0.6, 3.9), rng.uniform(-math.pi, math.pi)])
        expected = any(oracle_overlap(pose, footprint, row, col, 0.5) for row, col in cells)
        assert c.check_pose(pose).valid == (not expected)


def test_contact_and_out_of_map_geometry_are_invalid():
    c = checker()
    assert c.check_pose([0.1, 1, 0]).reason == "out_of_bounds"
    assert c.check_pose([-0.1, 1, 0]).reason == "out_of_bounds"
    assert c.check_pose([0.1001, 1, 0]).valid
    mask = np.zeros((60, 60), dtype=bool)
    mask[:, 20] = True
    c = checker(mask)
    assert not c.check_pose([1.9, 2, 0]).valid
    assert c.check_pose([1.89, 2, 0]).valid


def test_one_cell_wall_cannot_be_skipped_between_endpoints():
    mask = np.zeros((60, 60), dtype=bool)
    mask[:, 20] = True
    c = checker(mask)
    poses = np.array([[1, 2, 0], [3, 2, 0]])
    assert all(c.check_pose(pose).valid for pose in poses)
    result = c.check_path(poses, [4, 5])
    assert not result.valid and result.first_unsafe_interval == 0
    assert 4 <= result.valid_until < 5


def test_rotation_near_jamb_is_checked_between_clear_endpoints():
    mask = np.zeros((100, 100), dtype=bool)
    mask[55, 55] = True
    c = checker(mask, Footprint(0.7, 0.1), resolution=0.05)
    poses = np.array([[2.5, 2.5, 0], [2.5, 2.5, math.pi / 2]])
    assert all(c.check_pose(pose).valid for pose in poses)
    assert not c.check_path(poses, [0, 1]).valid


def test_curved_motion_requires_declared_bounds_and_conservative_envelopes():
    mask = np.zeros((60, 60), dtype=bool)
    mask[23, 23] = True
    c = checker(mask)
    poses = np.array([[2, 2, 0], [2.1, 2, 0]])
    assert c.check_path(poses).valid
    assert not c.check_path(poses, interval_motion_bounds=[1.0]).valid
    with pytest.raises(ValueError, match="smaller"):
        c.check_path(poses, interval_motion_bounds=[0.01])


def test_clearance_is_a_positive_capped_lower_bound_on_a_clear_path():
    c = checker()
    result = c.check_path([[1, 2, 0], [3, 2, 0]], [5, 7])
    assert result.valid and result.valid_until == 7
    assert 0 < result.clearance_lower_bound <= c.clearance_cap


@pytest.mark.parametrize("kind", ["empty", "nonfinite", "times", "errors", "bounds"])
def test_path_rejects_invalid_arrays(kind):
    c = checker()
    poses, options = np.array([[1, 2, 0], [2, 2, 0]], dtype=float), {}
    if kind == "empty":
        poses = np.empty((0, 3))
    elif kind == "nonfinite":
        poses[0, 0] = np.nan
    elif kind == "times":
        options["times"] = [1, 1]
    elif kind == "errors":
        options["pose_error_bounds"] = [0, -1]
    else:
        options["interval_motion_bounds"] = [np.inf]
    with pytest.raises(ValueError):
        c.check_path(poses, **options)


def test_straight_brake_includes_delay_and_stopping_distance_without_teleport():
    config = BrakeConfig(deceleration=5, delay_s=0.3, max_step_s=0.07)
    state = np.array([1, 2, 0, 10, 10], dtype=float)
    brake = build_brake(state, 0.04, 0.14, config)
    assert brake.times[-1] == pytest.approx(2.3)
    assert brake.states[-1, 0] == pytest.approx(1 + 0.8 * 0.3 + 0.8 * 2 / 2)
    np.testing.assert_allclose(brake.states[:, 1:3], np.tile([2, 0], (len(brake.times), 1)))
    np.testing.assert_array_equal(brake.states[0], state)
    np.testing.assert_array_equal(brake.states[-1, 3:], [0, 0])
    assert np.max(np.abs(np.diff(brake.states[:, 3:], axis=0) / np.diff(brake.times)[:, None])) <= 5 + 1e-10
    assert np.max(np.diff(brake.times)) <= config.max_step_s + 1e-12
    assert not brake.states.flags.writeable
    with pytest.raises(ValueError):
        brake.states.flags.writeable = True


def test_unequal_wheels_stop_separately_and_quadrature_error_bounds_hold():
    config = BrakeConfig(deceleration=3, delay_s=0.12, max_step_s=0.08)
    state = np.array([2, 2, 0.4, 2, -5], dtype=float)
    cv, cw = 0.04, 0.14
    brake = build_brake(state, cv, cw, config)
    assert np.any((brake.states[:, 3] == 0) & (brake.states[:, 4] != 0))

    def wheels(t):
        return np.sign(state[3:]) * np.maximum(0, np.abs(state[3:]) - config.deceleration * max(0, t - config.delay_s))

    def wheel_integrals(t):
        elapsed = np.minimum(max(0, t - config.delay_s), np.abs(state[3:]) / config.deceleration)
        return state[3:] * min(t, config.delay_s) + np.sign(state[3:]) * (np.abs(state[3:]) * elapsed - config.deceleration * elapsed**2 / 2)

    def yaw(t):
        integrals = wheel_integrals(t)
        return state[2] + cw * (integrals[1] - integrals[0])

    breaks = [config.delay_s, *list(config.delay_s + np.abs(state[3:]) / config.deceleration)]
    for i, t in enumerate(brake.times):
        points = [point for point in breaks if 0 < point < t]
        x = state[0] + quad(lambda q: cv * wheels(q).sum() * math.cos(yaw(q)), 0, t, points=points, epsabs=1e-12)[0]
        y = state[1] + quad(lambda q: cv * wheels(q).sum() * math.sin(yaw(q)), 0, t, points=points, epsabs=1e-12)[0]
        assert np.linalg.norm(brake.states[i, :2] - [x, y]) <= brake.position_error_bounds[i] + 1e-10
        assert brake.states[i, 2] == pytest.approx(yaw(t), abs=1e-12)


def test_requested_brake_samples_have_exact_times_and_analytic_straight_states():
    config = BrakeConfig(deceleration=3, delay_s=0.17, max_step_s=0.08)
    state = np.array([1, 2, 0.4, 6, 6], dtype=float)
    samples = np.array([0.63, 0.03, 2.17, 0.17, 0.63, 0.0])
    reservation = reserve_brake(checker(), state, 0.04, 0.14, config, sample_times=samples)
    assert reservation.feasible
    brake = reservation.trajectory
    for time in samples:
        indices = np.flatnonzero(brake.times == time)
        assert len(indices) == 1
        elapsed = max(0.0, time - config.delay_s)
        distance = 0.48 * min(time, config.delay_s) + 0.08 * (6 * elapsed - 1.5 * elapsed**2)
        expected = [state[0] + distance * math.cos(state[2]),
                    state[1] + distance * math.sin(state[2]), state[2],
                    max(0, 6 - 3 * elapsed), max(0, 6 - 3 * elapsed)]
        np.testing.assert_allclose(brake.states[indices[0]], expected, rtol=0, atol=1e-13)
    assert np.max(np.diff(brake.times)) <= config.max_step_s + 1e-12


@pytest.mark.parametrize("samples", [[-0.01], [1.01], [np.nan], [np.inf], [[0.2]], 0.2])
def test_invalid_brake_sample_times_are_rejected(samples):
    state, config = [1, 2, 0, 5, 5], BrakeConfig(5)
    with pytest.raises(ValueError, match="sample_times"):
        build_brake(state, 0.04, 0.14, config, sample_times=samples)
    result = reserve_brake(checker(), state, 0.04, 0.14, config, sample_times=samples)
    assert not result.feasible and result.status == "invalid_input"


def test_empty_requested_brake_samples_preserve_default_trajectory():
    args = ([1, 2, 0.4, 2, -5], 0.04, 0.14, BrakeConfig(3, delay_s=0.12))
    default, empty = build_brake(*args), build_brake(*args, sample_times=[])
    for name in ("times", "states", "motion_bounds", "position_error_bounds"):
        np.testing.assert_array_equal(getattr(default, name), getattr(empty, name))


def test_reserve_checks_delay_and_reports_initial_collision_separately():
    mask = np.zeros((60, 60), dtype=bool)
    mask[:, 16] = True
    c = checker(mask)
    state = [1, 2, 0, 10, 10]
    clear = reserve_brake(c, state, 0.04, 0.14, BrakeConfig(10))
    delayed = reserve_brake(c, state, 0.04, 0.14, BrakeConfig(10, delay_s=0.25))
    assert clear.feasible and clear.status == "reserved"
    assert not delayed.feasible and delayed.status == "unsafe_brake"
    assert delayed.validity.map_revision == c.map_revision
    invalid = reserve_brake(c, [1.65, 2, 0, 0, 0], 0.04, 0.14, BrakeConfig(10))
    assert not invalid.feasible and invalid.status == "initial_state_invalid"


def test_uncertainty_can_remove_a_braking_reserve():
    mask = np.zeros((60, 60), dtype=bool)
    mask[:, 16] = True
    c = checker(mask)
    state = [1, 2, 0, 10, 10]
    result = reserve_brake(c, state, 0.04, 0.14, BrakeConfig(10, position_uncertainty=0.11))
    assert not result.feasible and result.status == "unsafe_brake"


def test_heading_uncertainty_includes_growth_in_center_position_error():
    mask = np.zeros((60, 60), dtype=bool)
    mask[23, 27] = True
    c = checker(mask)
    state = [1, 2, 0, 10, 10]
    assert reserve_brake(c, state, 0.1, 0.14, BrakeConfig(5)).feasible
    result = reserve_brake(c, state, 0.1, 0.14, BrakeConfig(5, heading_uncertainty=0.15))
    assert not result.feasible and result.status == "unsafe_brake"


def test_prefix_reserve_selects_latest_declared_safe_anchor():
    states = np.array([[1, 2, 0, 5, 5], [1.2, 2, 0, 5, 5], [1.4, 2, 0, 5, 5]])
    result = reserve_prefix_brake(checker(), states, [10, 11, 12], 0.04, 0.14,
                                  BrakeConfig(10), interval_motion_bounds=[0.2, 0.2],
                                  candidate_indices=np.array([0, 1]))
    assert result.feasible and result.anchor_index == 1 and result.decision_time == 11
    assert result.prefix_validity.valid_until == 11
    assert result.prefix_validity.map_revision == result.brake.validity.map_revision
    assert result.brake.trajectory.states[0, 0] == 1.2


def test_prefix_deadline_moves_earlier_when_late_brakes_hit_wall():
    mask = np.zeros((60, 60), dtype=bool)
    mask[:, 20] = True
    states = np.array([[1, 2, 0, 10, 10], [1.2, 2, 0, 10, 10], [1.6, 2, 0, 10, 10]])
    result = reserve_prefix_brake(checker(mask), states, [0, 0.25, 0.75], 0.04, 0.14,
                                  BrakeConfig(10), interval_motion_bounds=[0.2, 0.4])
    assert result.feasible and result.anchor_index == 1 and result.decision_time == 0.25
    assert result.brake.feasible
    assert result.brake.trajectory.states[-1, 0] == pytest.approx(1.6)


def test_prefix_cannot_skip_invalid_interval_to_reach_a_clear_brake_anchor():
    mask = np.zeros((60, 60), dtype=bool)
    mask[:, 20] = True
    states = np.array([[1, 2, 0, 0, 0], [3, 2, 0, 0, 0]])
    c = checker(mask)
    assert reserve_brake(c, states[1], 0.04, 0.14, BrakeConfig(10)).feasible
    result = reserve_prefix_brake(c, states, [0, 1], 0.04, 0.14, BrakeConfig(10),
                                  interval_motion_bounds=[2.0], candidate_indices=[1])
    assert not result.feasible and result.status == "no_safe_anchor"
    assert result.anchor_index is None and result.decision_time is None
    assert result.prefix_validity.first_unsafe_interval == 0


def test_prefix_reserve_rejects_bad_bound_and_candidate_inputs():
    states = np.array([[1, 2, 0, 0, 0], [1.1, 2, 0, 0, 0]])
    for options in ({"interval_motion_bounds": []},
                    {"interval_motion_bounds": [0.1], "candidate_indices": [1, 1]},
                    {"interval_motion_bounds": [0.1], "candidate_indices": [True]},
                    {"interval_motion_bounds": [0.1], "candidate_indices": [2]}):
        with pytest.raises(ValueError):
            reserve_prefix_brake(checker(), states, [0, 1], 0.04, 0.14, BrakeConfig(10), **options)


def test_stopped_state_and_invalid_model_states_have_explicit_results():
    result = reserve_brake(checker(), [1, 2, 0, 0, 0], 0.04, 0.14, BrakeConfig(10))
    assert result.feasible and result.status == "already_stopped"
    assert result.trajectory.times.tolist() == [0.0]
    invalid = reserve_brake(checker(), [1, 2, 0, 21, 0], 0.04, 0.14, BrakeConfig(10))
    assert not invalid.feasible and invalid.status == "invalid_input"
    with pytest.raises(ValueError, match="duration"):
        build_brake([1, 2, 0, 10, 0], 0.04, 0.14, BrakeConfig(0.01))


@pytest.mark.parametrize("kwargs", [{"deceleration": 0}, {"deceleration": 1, "delay_s": -1},
                                    {"deceleration": 1, "max_step_s": np.inf}])
def test_invalid_brake_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        BrakeConfig(**kwargs)
