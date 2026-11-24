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
    from agx_planning.replanning.braking import BrakeConfig
    from agx_planning.replanning.execution import ExecutionConfig, FullStopRunner, PlanAttempt
    from agx_planning.replanning.reveal import HiddenObject, ObjectRevealer
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def execution_map(objects=()):
    snapshot = MapSnapshot("execution-map", 0, GridGeometry((60, 100), 0.1),
                           np.zeros((60, 100), dtype=np.int8), OccupancyPolicy())
    return ObjectRevealer(snapshot, objects, max_range=0.05)


def test_production_planner_reports_disconnected_start_without_a_plan():
    from agx_planning.profiles import load_profile
    from agx_planning.replanning.execution import ProductionPlanner
    profile = load_profile(REPO / "config/replanning/baseline_profile.yaml")
    profile["field_path"] = "raw_float64"
    occupancy = np.zeros((40, 60), np.int8)
    occupancy[:, 30] = 100
    snapshot = MapSnapshot("disconnected", 0, GridGeometry(occupancy.shape, 0.1),
                           occupancy, OccupancyPolicy())
    attempt = ProductionPlanner(profile, "test-build")(
        snapshot, [1.5, 2., 0., 0., 0.], [4.5, 2., 0.], 0.)
    assert attempt.status == "unreachable"
    assert attempt.plan is None


def test_production_planner_identifies_a_stopped_zero_command_plan():
    from agx_planning.profiles import load_profile
    from agx_planning.replanning.execution import ProductionPlanner
    profile = load_profile(REPO / "config/replanning/baseline_profile.yaml")
    profile["field_path"] = "raw_float64"
    snapshot = execution_map().current
    attempt = ProductionPlanner(profile, "test-build")(
        snapshot, [1.5, 1.5, 0., 0., 0.], [1.5, 1.5, 0.], 0.)
    assert attempt.status == "success"
    assert attempt.plan.field_key.map_revision == snapshot.revision
    assert attempt.plan.field_key.occupancy_policy_id == snapshot.policy.identity
    assert attempt.plan.raw_states.shape == (1, 5)
    assert attempt.plan.commands.shape == (0, 2)


def straight_attempt(snapshot, state, goal, epoch, parent_id, *, number,
                     solver_status="success", end_offset=0.0):
    """Controlled planner callback; execution physics is checked independently."""
    duration = float(goal[0] + end_offset - state[0])
    assert duration > 0
    times = np.array([0., duration, duration + 0.1])
    commands = np.array([[1., 1.], [0., 0.]])
    middle = command_step(state, commands[0], duration, 0.5, 0.5)
    final = command_step(middle, commands[1], 0.1, 0.5, 0.5)
    raw = np.array([state, middle, final])
    coefficients = np.zeros((2, 2, 10))
    coefficients[0, :, :5] = np.diff(raw, axis=0) / np.diff(times)[:, None]
    coefficients[1, :, :5] = raw[:-1]
    polynomial = PPoly(coefficients, times)
    key = FieldKey.for_map(snapshot, snapshot.geometry.world_to_cell(*goal[:2]),
                           {"test": "controlled_execution"}, "test-v1")
    plan = PlanRecord(session_id="execution-session", plan_id=f"plan-{number}",
                      parent_id=parent_id, field_key=key, start_epoch=epoch,
                      times=times, raw_states=raw, commands=commands,
                      accels=np.zeros((2, 2)), costates=np.zeros((2, 5)),
                      segments=(SegmentSolution.from_ppoly(polynomial, key),),
                      status=solver_status)
    return PlanAttempt(solver_status, "controlled planner result", plan)


def runner(revealer, planner, **config):
    config = {"collision_step": 0.02, **config}
    return FullStopRunner(revealer, planner, c_v=0.5, c_w=0.5,
                          brake=BrakeConfig(deceleration=10., max_step_s=0.02),
                          footprint=Footprint(0.05, 0.05),
                          config=ExecutionConfig(**config))


def test_delayed_replacement_rejects_stale_map_without_true_map_leakage():
    objects = [HiddenObject("first", [[70, 40, 71, 41, 100]]),
               HiddenObject("second", [[75, 40, 76, 41, 100]])]
    revealer = execution_map(objects)
    seen = []

    def planner(snapshot, state, goal, epoch, parent_id):
        seen.append((snapshot, state.copy(), epoch, parent_id))
        return straight_attempt(snapshot, state, goal, epoch, parent_id, number=len(seen))

    result = runner(revealer, planner, latency_override_s=0.5).run(
        [1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0],
        scheduled_reveals=[(0.2, ["first"]), (0.4, ["second"])])
    assert result.status == "success", result.message
    assert [entry[0].revision for entry in seen] == [0, 1, 2]
    assert [int(entry[0].obstacles.sum()) for entry in seen] == [0, 1, 2]
    assert all(entry[0].session != revealer.true_map.session for entry in seen)
    assert all(np.max(np.abs(entry[1][3:])) == 0 for entry in seen)
    np.testing.assert_array_equal(seen[1][1], seen[2][1])
    assert seen[2][2] - seen[1][2] == pytest.approx(0.5)
    assert [plan.plan_id for plan in result.plans] == ["plan-1", "plan-3"]
    rejected = [event for event in result.events if event["kind"] == "candidate_rejected"]
    assert len(rejected) == 1 and rejected[0]["reason"] == "stale_map"
    assert result.metadata["stopped_compute_s"] == 1.0
    assert np.all(np.diff(result.times) > 0)
    np.testing.assert_allclose(result.states[-1], [2.5, 1.5, 0, 0, 0], atol=1e-10)

    reservation = next(event for event in result.events
                       if event["kind"] == "brake_reservation" and event["reason"] == "map_change")
    stopped = next(event for event in result.events
                   if event["kind"] == "stopped" and event["reason"] == "map_change")
    assert stopped["time"] - reservation["time"] == pytest.approx(0.1)
    selected = (result.times >= reservation["time"] - 1e-12) & (result.times <= stopped["time"] + 1e-12)
    wheel_accels = np.diff(result.states[selected, 3:], axis=0) / np.diff(result.times[selected])[:, None]
    assert np.max(np.abs(wheel_accels)) <= 10.0 + 1e-10


@pytest.mark.parametrize("solver_status", ["success", "stagnated"])
def test_final_success_uses_same_execution_checks_for_each_solver_status(solver_status):
    def planner(snapshot, state, goal, epoch, parent_id):
        return straight_attempt(snapshot, state, goal, epoch, parent_id,
                                number=1, solver_status=solver_status)
    result = runner(execution_map(), planner).run([1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0])
    assert result.status == "success"
    final = next(event for event in result.events if event["kind"] == "final_check")
    assert final["success"] and final["solver_status"] == solver_status
    assert final["max_wheel_speed"] == 0


@pytest.mark.parametrize("offset, heading", [(-0.3, 0.0), (0.0, math.pi / 2)])
def test_solver_success_cannot_override_final_position_or_heading_miss(offset, heading):
    def planner(snapshot, state, goal, epoch, parent_id):
        return straight_attempt(snapshot, state, goal, epoch, parent_id,
                                number=1, end_offset=offset)
    result = runner(execution_map(), planner).run([1.5, 1.5, 0, 0, 0], [2.5, 1.5, heading])
    assert result.status == "goal_miss"
    assert not next(event for event in result.events if event["kind"] == "final_check")["success"]


def test_failed_replan_leaves_robot_stopped_with_no_replacement_activation():
    revealer = execution_map([HiddenObject("first", [[70, 40, 71, 41, 100]])])
    seen = []
    def planner(snapshot, state, goal, epoch, parent_id):
        seen.append(state.copy())
        if snapshot.revision:
            return PlanAttempt("solver_failed", "injected solve failure")
        return straight_attempt(snapshot, state, goal, epoch, parent_id, number=1)
    result = runner(revealer, planner, latency_override_s=0.2).run(
        [1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0], scheduled_reveals=[(0.2, ["first"])])
    assert result.status == "solver_failed"
    assert len(result.plans) == 1 and len(seen) == 2
    np.testing.assert_array_equal(result.states[-1, 3:], [0, 0])
    np.testing.assert_array_equal(result.states[-1], seen[-1])


def test_unseen_true_obstacle_fails_collision_without_entering_planner_map():
    revealer = execution_map([HiddenObject("hidden", [[20, 14, 21, 17, 100]])])
    seen = []
    def planner(snapshot, state, goal, epoch, parent_id):
        seen.append(snapshot)
        return straight_attempt(snapshot, state, goal, epoch, parent_id, number=1)
    result = runner(revealer, planner).run([1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0])
    assert result.status == "collision"
    assert len(seen) == 1 and not seen[0].obstacles.any()
    assert result.states[-1, 0] < 2.0
    assert not any(event["kind"] == "final_check" for event in result.events)


@pytest.mark.parametrize("collision_step", [0.02, 0.03, 0.05])
def test_controlled_observations_follow_sample_ticks_during_motion_and_wait(collision_step):
    objects = [HiddenObject("first", [[70, 40, 71, 41, 100]]),
               HiddenObject("second", [[75, 40, 76, 41, 100]])]
    calls = 0
    def planner(snapshot, state, goal, epoch, parent_id):
        nonlocal calls
        calls += 1
        return straight_attempt(snapshot, state, goal, epoch, parent_id, number=calls)
    result = runner(execution_map(objects), planner, sample_period=0.1,
                    collision_step=collision_step, latency_override_s=0.5).run(
        [1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0],
        scheduled_reveals=[(0.15, ["first"]), (0.35, ["second"])])
    assert result.status == "success"
    reveals = [event for event in result.events if event["kind"] == "map_reveal"]
    np.testing.assert_allclose([event["time"] for event in reveals], [0.2, 0.4], atol=1e-10)
    assert [event["requested_time"] for event in reveals] == [0.15, 0.35]


def test_failed_attempt_with_populated_plan_cannot_activate_it():
    def planner(snapshot, state, goal, epoch, parent_id):
        result = straight_attempt(snapshot, state, goal, epoch, parent_id, number=1)
        return replace(result, status="solver_failed")
    result = runner(execution_map(), planner).run([1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0])
    assert result.status == "invalid_plan"
    assert result.plans == ()
    assert not any(event["kind"] == "plan_activated" for event in result.events)


def test_observation_ticks_remain_absolute_after_non_aligned_compute_delay():
    objects = [HiddenObject("first", [[70, 40, 71, 41, 100]]),
               HiddenObject("second", [[75, 40, 76, 41, 100]])]
    calls = 0
    def planner(snapshot, state, goal, epoch, parent_id):
        nonlocal calls
        calls += 1
        return straight_attempt(snapshot, state, goal, epoch, parent_id, number=calls)
    result = runner(execution_map(objects), planner, sample_period=0.1,
                    collision_step=0.03, latency_override_s=0.515).run(
        [1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0],
        scheduled_reveals=[(0.15, ["first"]), (0.95, ["second"])])
    assert result.status == "success"
    reveals = [event for event in result.events if event["kind"] == "map_reveal"]
    np.testing.assert_allclose([event["time"] for event in reveals], [0.2, 1.0], atol=1e-10)


def test_callback_exception_is_terminal_failure():
    def planner(*_args):
        raise RuntimeError("injected planner exception")
    result = runner(execution_map(), planner).run([1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0])
    assert result.status == "planner_exception"
    assert result.plans == ()
    assert "injected planner exception" in result.message


def test_nominal_replay_failure_is_invalid_plan():
    def planner(snapshot, state, goal, epoch, parent_id):
        result = straight_attempt(snapshot, state, goal, epoch, parent_id, number=1)
        return replace(result, plan=replace(result.plan, segments=()))
    result = runner(execution_map(), planner, mode="nominal").run(
        [1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0])
    assert result.status == "invalid_plan"
    assert result.plans == ()


def test_compute_past_execution_limit_cannot_activate_replacement():
    calls = 0
    def planner(snapshot, state, goal, epoch, parent_id):
        nonlocal calls
        calls += 1
        return straight_attempt(snapshot, state, goal, epoch, parent_id, number=calls)
    result = runner(execution_map([HiddenObject("first", [[70, 40, 71, 41, 100]])]),
                    planner, latency_override_s=2., max_execution_s=1.).run(
        [1.5, 1.5, 0, 0, 0], [2.5, 1.5, 0], scheduled_reveals=[(0.2, ["first"])])
    assert result.status == "time_limit"
    assert len(result.plans) == 1
    assert np.max(np.abs(result.states[-1, 3:])) == 0
