"""Activation must use the exact validated plan and current state assumptions."""

from dataclasses import replace

import numpy as np
import pytest

from test_replanning_coordinator import assess, candidate, repair, setup, snapshot, stage


@pytest.mark.parametrize("state_change", ["wheel", "position", "nonfinite"])
def test_changed_measured_state_invalidates_the_nominal_braking_certificate(state_change):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    assess(coordinator, active, changed)
    value = candidate(active, changed)
    assert stage(coordinator, value).accepted
    measured = value.raw_states[0].copy()
    if state_change == "wheel":
        measured[3] += 0.1
    elif state_change == "position":
        measured[0] += 0.1
    else:
        measured[0] = np.nan

    decision = coordinator.commit(now=value.start_epoch, measured_state=measured,
                                  snapshot=changed)

    assert decision.action == "FAILED"
    assert decision.brake is None
    assert coordinator.reservation is None
    assert store.active is active


@pytest.mark.parametrize("reuse_plan_id", [False, True])
def test_pending_substitution_cannot_bypass_the_coordinator_path_check(reuse_plan_id):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    assess(coordinator, active, changed)
    value = candidate(active, changed)
    assert stage(coordinator, value).accepted

    # This complete record crosses the new wall. A store checks identity and
    # joints only; it cannot replace the coordinator's geometric validation.
    unchecked = candidate(active, changed, stopping=False,
                          plan_id=value.plan_id if reuse_plan_id else "unchecked")
    store.stage(unchecked, anchor_index=5, validation_field_key=unchecked.field_key,
                deadline=coordinator.reservation.decision_time)
    decision = coordinator.commit(now=unchecked.start_epoch,
                                  measured_state=unchecked.raw_states[0], snapshot=changed)

    assert not decision.accepted
    assert store.active is active


@pytest.mark.parametrize("mutation", ["landing", "retained_commands"])
def test_repair_provenance_must_match_the_actual_retained_parent_suffix(mutation):
    coordinator, store, active = setup()
    changed = snapshot(revision=1)
    assess(coordinator, active, changed, quality_trigger=True)
    value = repair(active, changed)
    if mutation == "landing":
        value = replace(value, composition=replace(value.composition, landing_index=15))
    else:
        commands = value.commands.copy()
        join = np.flatnonzero(np.isclose(value.times, value.composition.join_time))[0]
        commands[join:, 0] += 0.1
        value = replace(value, commands=commands)

    decision = stage(coordinator, value, action="REPAIR")

    assert not decision.accepted
    assert store.active is active
    assert store.pending is None


@pytest.mark.parametrize("now", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("operation", ["stage", "commit"])
def test_nonfinite_execution_time_cannot_reuse_an_old_braking_certificate(now, operation):
    coordinator, store, active = setup()
    changed = snapshot(revision=1, wall=True)
    assess(coordinator, active, changed)
    value = candidate(active, changed)
    if operation == "stage":
        decision = stage(coordinator, value, now=now)
    else:
        assert stage(coordinator, value).accepted
        decision = coordinator.commit(now=now, measured_state=value.raw_states[0],
                                      snapshot=changed)

    assert decision.action == "FAILED"
    assert decision.brake is None
    assert store.active is active
