"""Replacements must preserve a checked stop until a valid plan is committed."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.interpolate import PPoly

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.maps import GridGeometry, MapSnapshot
    from agx_planning.vector_field.contracts import FieldKey
    from agx_planning.pmp_planner.plan_store import PlanRecord, PlanStore, SegmentSolution
    from agx_planning.replanning.braking import BrakeConfig
    from agx_planning.replanning.coordinator import ReplanningCoordinator
    from agx_planning.replanning.validity import Footprint
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def snapshot(revision=0, wall=False):
    occupancy = np.zeros((100, 100), dtype=np.int8)
    if wall:
        occupancy[:, 30] = 100
    return MapSnapshot("test-map", revision, GridGeometry((100, 100), 0.1), occupancy)


def key_for(value):
    return FieldKey.for_map(value, (90, 20), {"method": "test"}, "test-v1")


def plan(value, *, plan_id="active", parent_id=None, epoch=10.0,
         x0=1.0, stopping=False):
    coefficients = np.zeros((3, 1, 10))
    coefficients[1, 0, 0] = 1.0
    coefficients[2, 0, :5] = [x0, 2.0, 0.0, 1.0, 1.0]
    if stopping:
        coefficients[0, 0, 0] = -1.25
        coefficients[1, 0, 3:5] = -2.5
        times = np.array([0.0, 0.2, 0.4])
        horizon = 0.4
    else:
        times = np.arange(21) * 0.2
        horizon = 5.0
    polynomial = PPoly(coefficients, [0.0, horizon], extrapolate=False)
    states = polynomial(times)
    field_key = key_for(value)
    segments = []
    for start in ([0.0] if stopping else [0.0, 1.0, 2.0, 3.0]):
        local = coefficients.copy()
        local[2, 0, 0] += start
        segments.append(SegmentSolution.from_ppoly(
            PPoly(local, [0.0, horizon], extrapolate=False), field_key, start_time=start,
        ))
    return PlanRecord(
        session_id="test-planner", plan_id=plan_id, parent_id=parent_id,
        field_key=field_key, start_epoch=epoch, times=times,
        raw_states=states[:, :5], commands=polynomial((times[:-1] + times[1:]) / 2)[:, 3:5],
        accels=np.tile([-2.5, -2.5] if stopping else [0.0, 0.0], (len(times) - 1, 1)),
        costates=states[:-1, 5:],
        segments=tuple(segments),
    )


def motion_bounds(value):
    return np.diff(value.raw_states[:, 0])


def setup():
    active = plan(snapshot())
    store = PlanStore()
    store.install_initial(active)
    coordinator = ReplanningCoordinator(
        store, 0.5, 0.5, BrakeConfig(5.0), footprint=Footprint(0.05, 0.05),
    )
    return coordinator, store, active


def assess(coordinator, active, value, **overrides):
    arguments = dict(now=active.start_epoch, current_index=0,
                     measured_state=active.raw_states[0],
                     interval_motion_bounds=motion_bounds(active))
    arguments.update(overrides)
    return coordinator.assess(value, **arguments)


def candidate(active, value, anchor_index=5, **overrides):
    arguments = dict(plan_id="replacement", parent_id=active.plan_id,
                     epoch=active.start_epoch + active.times[anchor_index],
                     x0=active.raw_states[anchor_index, 0], stopping=True)
    arguments.update(overrides)
    return plan(value, **arguments)


def stage(coordinator, value, *, action="SPLICE", anchor_index=5, now=10.0, **overrides):
    arguments = dict(action=action, anchor_index=anchor_index, now=now,
                     interval_motion_bounds=motion_bounds(value), limits_ok=True)
    arguments.update(overrides)
    return coordinator.stage_candidate(value, **arguments)


def repair(active, value):
    middle = candidate(active, value, stopping=False, plan_id="middle")
    middle = replace(middle, times=middle.times[:6], raw_states=middle.raw_states[:6],
                     commands=middle.commands[:5], accels=middle.accels[:5],
                     costates=middle.costates[:5], segments=middle.segments[:1])
    return PlanRecord.compose_repair(active, middle, departure_index=5,
                                    landing_index=10, plan_id="repair")


def test_clear_suffix_keeps_the_active_plan_without_staging_a_replacement():
    coordinator, store, active = setup()
    decision = assess(coordinator, active, snapshot(revision=1))
    assert decision.action == "KEEP"
    assert decision.status == "remaining_valid"
    assert decision.validity.valid
    assert store.active is active and store.pending is None


def test_invalid_suffix_keeps_a_checked_future_braking_reserve():
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    decision = assess(coordinator, active, changed)
    assert decision.action == "STOP_REPLAN"
    assert decision.status == "replacement_required"
    assert decision.validity.valid
    assert decision.validity.valid_until == decision.execute_at
    assert active.start_epoch < decision.execute_at < active.start_epoch + active.times[-1]
    assert decision.anchor_index > 5
    assert decision.brake.feasible
    assert decision.brake.validity.valid
    assert store.active is active and store.pending is None


@pytest.mark.parametrize("failure", ["parent", "field", "late", "path", "limits"])
def test_rejected_candidate_preserves_the_active_plan_and_brake_reserve(failure):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    fallback = assess(coordinator, active, changed)
    value = candidate(active, changed)
    options = {}
    if failure == "parent":
        value = replace(value, parent_id="stale-parent")
    elif failure == "field":
        value = candidate(active, snapshot(revision=2, wall=True))
    elif failure == "late":
        options["now"] = value.start_epoch + 0.01
    elif failure == "path":
        value = candidate(active, changed, stopping=False)
    else:
        options["limits_ok"] = False
    decision = stage(coordinator, value, **options)
    assert decision.action == "STOP_REPLAN" and decision.status == "candidate_rejected"
    assert store.active is active and store.pending is None
    due = coordinator.deadline_decision(now=fallback.execute_at, snapshot=changed)
    assert due.action == "STOP_REPLAN" and due.status == "brake_due"
    assert due.execute_at == fallback.execute_at
    np.testing.assert_array_equal(due.brake.trajectory.states, fallback.brake.trajectory.states)


def test_timely_splice_activates_only_at_its_checked_raw_boundary():
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    assess(coordinator, active, changed)
    value = candidate(active, changed)
    staged = stage(coordinator, value)
    assert staged.action == "SPLICE"
    assert store.pending is value and store.active is active
    committed = coordinator.commit(now=value.start_epoch, measured_state=value.raw_states[0],
                                   snapshot=changed)
    assert committed.action == "SPLICE" and committed.status == "committed"
    assert store.active is value and store.pending is None
    assert store.previous == (active,)
    reserve = coordinator.reservation
    assert reserve.feasible
    assert reserve.decision_time == value.start_epoch + value.times[-1]
    due = coordinator.deadline_decision(now=reserve.decision_time, snapshot=changed)
    assert due.action == "STOP_REPLAN" and due.status == "brake_due"
    np.testing.assert_array_equal(due.brake.trajectory.states[0], value.raw_states[-1])


@pytest.mark.parametrize("same_revision", [False, True])
def test_changed_map_blocks_commit_even_when_the_candidate_path_still_looks_clear(same_revision):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    assess(coordinator, active, changed)
    value = candidate(active, changed)
    stage(coordinator, value)
    newer = snapshot(revision=2, wall=True)
    if same_revision:
        occupancy = changed.occupancy.copy()
        occupancy[90, 90] = 100
        newer = replace(changed, occupancy=occupancy)
    decision = coordinator.commit(now=value.start_epoch, measured_state=value.raw_states[0],
                                  snapshot=newer)
    assert decision.action == "FAILED" and decision.status == "map_changed"
    assert store.active is active


def test_quality_trigger_reserves_a_stop_while_waiting_on_a_clear_path():
    coordinator, store, active = setup()
    changed = snapshot(revision=1)
    decision = assess(coordinator, active, changed, quality_trigger=True)
    assert decision.action == "STOP_REPLAN" and decision.status == "replacement_required"
    assert decision.validity.valid and decision.brake.feasible
    waiting = coordinator.deadline_decision(now=active.start_epoch, snapshot=changed)
    assert waiting.action == "STOP_REPLAN" and waiting.status == "waiting"
    assert store.active is active


def test_missed_braking_deadline_does_not_claim_an_old_reserve_is_executable():
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    fallback = assess(coordinator, active, changed)
    decision = coordinator.deadline_decision(now=fallback.execute_at + 0.01, snapshot=changed)
    assert decision.action == "FAILED" and decision.status == "deadline_missed"
    assert store.active is active


@pytest.mark.parametrize("operation", ["stage", "commit"])
def test_late_replacement_callback_cannot_return_a_braking_reserve_in_the_past(operation):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    fallback = assess(coordinator, active, changed)
    value = candidate(active, changed)
    late = fallback.execute_at + 0.01
    if operation == "stage":
        decision = stage(coordinator, value, now=late)
    else:
        stage(coordinator, value)
        decision = coordinator.commit(now=late, measured_state=value.raw_states[0], snapshot=changed)
    assert decision.action == "FAILED" and decision.status == "deadline_missed"
    assert store.active is active


@pytest.mark.parametrize("failure", ["state", "early_time"])
def test_failed_activation_preserves_active_and_reuses_reserve_only_if_state_still_matches(failure):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    fallback = assess(coordinator, active, changed)
    value = candidate(active, changed)
    stage(coordinator, value)
    state = value.raw_states[0].copy()
    now = value.start_epoch
    if failure == "state":
        state[3] += 0.1
    else:
        now -= 0.01
    rejected = coordinator.commit(now=now, measured_state=state, snapshot=changed)
    if failure == "state":
        assert rejected.action == "FAILED" and rejected.status == "state_changed"
        assert coordinator.reservation is None
    else:
        assert rejected.action == "STOP_REPLAN" and rejected.status == "candidate_rejected"
        assert rejected.execute_at == fallback.execute_at
    assert store.active is active


def test_typed_repair_activates_with_original_suffix_field_provenance():
    coordinator, store, active = setup()
    changed = snapshot(revision=1)
    assess(coordinator, active, changed, quality_trigger=True)
    value = repair(active, changed)
    assert value.segments[0].field_key == key_for(changed)
    assert all(segment.field_key == active.field_key for segment in value.segments[1:])
    staged = stage(coordinator, value, action="REPAIR")
    assert staged.action == "REPAIR" and staged.status == "staged"
    committed = coordinator.commit(now=value.start_epoch, measured_state=value.raw_states[0],
                                   snapshot=changed)
    assert committed.action == "REPAIR" and committed.status == "committed"
    assert store.active is value


def test_repair_rechecks_the_complete_retained_suffix_against_the_new_map():
    coordinator, store, active = setup()
    occupancy = np.zeros((100, 100), dtype=np.int8)
    occupancy[:, 45] = 100
    changed = replace(snapshot(revision=1), occupancy=occupancy)
    fallback = assess(coordinator, active, changed)
    value = repair(active, changed)
    decision = stage(coordinator, value, action="REPAIR")
    assert decision.action == "STOP_REPLAN" and decision.status == "candidate_rejected"
    assert decision.execute_at == fallback.execute_at
    assert store.active is active and store.pending is None
