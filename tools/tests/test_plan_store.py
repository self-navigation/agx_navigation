"""Preserve raw plans and full splines without accepting stale replacements."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import sys

import numpy as np
import pytest
from scipy.interpolate import PPoly

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.maps import GridGeometry
    from agx_planning.vector_field.contracts import FieldKey
    from agx_planning.pmp_planner.plan_store import (
        LegacyPlan, PlanRecord, PlanStore, SegmentSolution, load_plan, save_plan,
    )
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def field_key(revision=1):
    return FieldKey(
        map_session="map-session", map_revision=revision,
        geometry=GridGeometry((21, 31), 0.1), goal_cell=(20, 10),
        recipe_id="recipe", numerical_version="test-solver-v1",
        occupancy_policy_id="occupied-65",
    )


def spline(x0=0.0):
    knots = np.array([0.0, 0.2, 0.5])
    coefficients = np.zeros((2, 2, 10))
    coefficients[0, :, 0] = 1.0
    coefficients[1, :, 0] = x0 + knots[:-1]
    coefficients[1, :, 3:5] = 1.0
    coefficients[1, :, 5:] = np.arange(1.0, 6.0)
    return PPoly(coefficients, knots, extrapolate=False)


def record(plan_id="plan-1", parent_id=None, key=None, epoch=10.0, x0=0.0,
           segments=None, status="success"):
    times = np.array([0.0, 0.1, 0.2, 0.3])
    values = spline(x0)(times)
    key = field_key() if key is None else key
    if segments is None:
        segments = (SegmentSolution.from_ppoly(spline(x0), key),)
    return PlanRecord(
        session_id="planner-session", plan_id=plan_id, parent_id=parent_id,
        field_key=key, start_epoch=epoch,
        times=times, raw_states=values[:, :5], commands=np.full((3, 2), 2.0),
        accels=np.zeros((3, 2)), costates=values[:-1, 5:],
        segments=segments, status=status, metadata={"scenario": {"seed": 7}},
    )


def replacement(active, plan_id="plan-2", revision=2, anchor_index=1):
    return record(
        plan_id=plan_id, parent_id=active.plan_id, key=field_key(revision),
        epoch=active.start_epoch + active.times[anchor_index],
        x0=active.raw_states[anchor_index, 0],
    )


def stage(store, candidate, anchor_index=1):
    store.stage(candidate, anchor_index=anchor_index,
                validation_field_key=candidate.field_key,
                deadline=candidate.start_epoch + 0.05)


def commit(store, candidate, **overrides):
    arguments = dict(now=candidate.start_epoch,
                     measured_state=candidate.raw_states[0],
                     current_field_key=candidate.field_key,
                     session_id=candidate.session_id)
    arguments.update(overrides)
    return store.commit_pending(**arguments)


def test_spline_owns_coefficients_and_evaluates_all_channels_exactly():
    original = PPoly(np.random.default_rng(71).normal(size=(4, 3, 10)),
                     np.array([0.0, 0.2, 0.6, 1.0]), extrapolate=False)
    queries = np.linspace(0.0, 1.0, 17)
    expected = original(queries).T.copy()
    segment = SegmentSolution.from_ppoly(original, field_key(), start_time=1.5)
    original.c[:] = 999
    np.testing.assert_array_equal(segment.evaluate(queries), expected)
    np.testing.assert_array_equal(segment.evaluate(0.0), expected[:, 0])
    for array in (segment.knots, segment.coefficients):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_plan_owns_arrays_metadata_and_pickle_roundtrip_keeps_them_immutable():
    plan = record()
    raw = plan.raw_states.copy()
    commands = plan.commands.copy()
    metadata = {"nested": {"values": [1, 2]}}
    owned = replace(plan, raw_states=raw, commands=commands, metadata=metadata)
    raw[:] = -9
    commands[:] = -7
    metadata["nested"]["values"][0] = 999
    np.testing.assert_array_equal(owned.raw_states, plan.raw_states)
    np.testing.assert_array_equal(owned.commands, plan.commands)
    assert owned.metadata["nested"]["values"][0] == 1
    restored = pickle.loads(pickle.dumps(owned))
    for item in (owned, restored):
        for array in (item.times, item.raw_states, item.commands, item.accels, item.costates):
            with pytest.raises(ValueError):
                array.setflags(write=True)
        with pytest.raises(TypeError):
            item.metadata["nested"]["new"] = 3


def test_versioned_roundtrip_preserves_raw_wheels_full_horizon_and_provenance(tmp_path):
    segment = SegmentSolution.from_ppoly(
        spline(), field_key(), metadata={"boundary": {"start": [0, 0, 0, 1, 1]}},
    )
    plan = record(segments=(segment,))
    npz_path, json_path = save_plan(plan, tmp_path / "saved")
    with np.load(npz_path, allow_pickle=False) as data:
        assert all(data[name].dtype != object for name in data.files)
    assert isinstance(json.loads(Path(json_path).read_text())["schema_version"], int)
    loaded = load_plan(tmp_path / "saved")
    assert isinstance(loaded, PlanRecord)
    assert loaded.field_key == plan.field_key
    assert loaded.session_id == plan.session_id and loaded.plan_id == plan.plan_id
    assert loaded.start_epoch == plan.start_epoch
    for name in ("times", "raw_states", "commands", "accels", "costates"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(plan, name))
    assert not np.array_equal(loaded.raw_states[:-1, 3:], loaded.commands)
    assert loaded.segments[0].knots[-1] > loaded.times[-1]
    queries = np.linspace(0.0, 0.5, 15)
    np.testing.assert_array_equal(loaded.segments[0].evaluate(queries),
                                  segment.evaluate(queries))
    assert loaded.array_bytes == plan.array_bytes


@pytest.mark.parametrize("change", [
    {"raw_states": np.zeros((3, 5))},
    {"commands": np.zeros((4, 2))},
    {"times": [0.0, 0.1, 0.1, 0.3]},
    {"times": [1.0, 1.1, 1.2, 1.3]},
    {"accels": np.full((3, 2), np.nan)},
    {"start_epoch": float("inf")},
])
def test_invalid_shape_time_and_nonfinite_plan_data_are_rejected(change):
    with pytest.raises((TypeError, ValueError)):
        replace(record(), **change)


def chunk(index, x0):
    values = spline(x0)(np.array([0.0, 0.1, 0.2]))
    return SimpleNamespace(
        chunk_idx=index, dt_sample=0.1, raw_states=values[:, :5], field_key=field_key(),
        wheel_cmds=np.full((2, 2), 2.0), accels=np.zeros((2, 2)),
        poses=values[:-1, :3], costates=values[:-1, 5:], solution=spline(x0),
    )


def from_chunks(chunks):
    return PlanRecord.from_chunks(
        chunks, field_key=field_key(), session_id="planner-session",
        plan_id="joined", parent_id=None, start_epoch=10.0, status="success",
    )


def repair_parts():
    parent = from_chunks([chunk(0, 0.0), chunk(1, 0.2), chunk(2, 0.4)])
    middle = record(plan_id="middle", parent_id=parent.plan_id,
                    key=field_key(2), epoch=10.2, x0=0.2)
    coefficients = middle.segments[0].coefficients.copy()
    coefficients[0, :, 0] = 2.0 / 3.0
    coefficients[1, :, 0] = 0.2 + middle.segments[0].knots[:-1] * (2.0 / 3.0)
    segment = SegmentSolution(field_key(2), 0.0, middle.segments[0].knots, coefficients)
    middle = replace(middle, raw_states=segment.evaluate(middle.times)[:5].T,
                     segments=(segment,))
    return parent, middle


def test_repair_keeps_source_keys_and_explicitly_retimes_the_whole_suffix(tmp_path):
    parent, middle = repair_parts()
    composed = PlanRecord.compose_repair(parent, middle, departure_index=2,
                                         landing_index=4, plan_id="repaired")
    assert composed.field_key == middle.field_key
    assert [part.field_key for part in composed.segments] == [field_key(2), field_key(1)]
    assert composed.composition.parent_plan_id == parent.plan_id
    assert composed.composition.repair_plan_id == middle.plan_id
    assert composed.composition.join_time == 0.3
    np.testing.assert_allclose(composed.times, [0, 0.1, 0.2, 0.3, 0.4, 0.5])
    np.testing.assert_array_equal(composed.raw_states[4:], parent.raw_states[5:])
    np.testing.assert_array_equal(composed.commands[3:], parent.commands[4:])
    save_plan(composed, tmp_path / "composed")
    loaded = load_plan(tmp_path / "composed")
    for result in (loaded, pickle.loads(pickle.dumps(composed))):
        assert result.composition == composed.composition
        assert [part.field_key for part in result.segments] == [field_key(2), field_key(1)]
        np.testing.assert_array_equal(result.segments[-1].evaluate([0.0, 0.2]),
                                      parent.segments[-1].evaluate([0.0, 0.2]))


def test_mixed_source_keys_require_typed_repair_provenance():
    parent, middle = repair_parts()
    composed = PlanRecord.compose_repair(parent, middle, departure_index=2,
                                         landing_index=4, plan_id="repaired")
    with pytest.raises(ValueError, match="field key"):
        replace(composed, composition=None)


@pytest.mark.parametrize("failure", ["departure", "landing", "wheel_joint", "epoch"])
def test_repair_rejects_nonsegment_anchors_and_full_state_joint_errors(failure):
    parent, middle = repair_parts()
    departure, landing = (1 if failure == "departure" else 2), (5 if failure == "landing" else 4)
    if failure == "wheel_joint":
        coefficients = middle.segments[0].coefficients.copy()
        coefficients[1, :, 3] += 0.1
        segment = replace(middle.segments[0], coefficients=coefficients)
        middle = replace(middle, segments=(segment,), raw_states=segment.evaluate(middle.times)[:5].T)
    if failure == "epoch":
        middle = replace(middle, start_epoch=11.0)
    with pytest.raises(ValueError):
        PlanRecord.compose_repair(parent, middle, departure_index=departure,
                                  landing_index=landing, plan_id="bad-repair")


def test_chunk_conversion_joins_raw_boundaries_and_retains_each_full_solution():
    plan = from_chunks([chunk(0, 0.0), chunk(1, 0.2)])
    assert plan.commands.shape == (4, 2)
    assert plan.raw_states.shape == (5, 5)
    np.testing.assert_allclose(plan.times, np.arange(5) * 0.1, rtol=0, atol=1e-15)
    np.testing.assert_allclose(plan.raw_states[:, 0], plan.times, rtol=0, atol=1e-15)
    assert [segment.start_time for segment in plan.segments] == [0.0, 0.2]
    assert all(segment.knots[-1] == 0.5 for segment in plan.segments)


@pytest.mark.parametrize("failure", ["missing_raw", "discontinuous", "out_of_order"])
def test_chunk_conversion_rejects_legacy_or_discontinuous_chunks(failure):
    first, second = chunk(0, 0.0), chunk(1, 0.2)
    if failure == "missing_raw":
        first.raw_states = None
    elif failure == "discontinuous":
        second.raw_states[0, 0] += 0.01
    else:
        second.chunk_idx = 3
    with pytest.raises((TypeError, ValueError)):
        from_chunks([first, second])


@pytest.mark.parametrize("missing", [True, False])
def test_chunk_conversion_requires_each_chunks_exact_field_identity(missing):
    first, second = chunk(0, 0.0), chunk(1, 0.2)
    if missing:
        del second.field_key
    else:
        second.field_key = field_key(2)
    with pytest.raises((TypeError, ValueError)):
        from_chunks([first, second])


def test_legacy_npz_is_readable_but_cannot_be_activated(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, wheel_cmds=np.ones((3, 2)), poses=np.zeros((3, 3)))
    legacy = load_plan(path)
    assert isinstance(legacy, LegacyPlan)
    assert legacy.capabilities["raw_states"] is False
    assert legacy.capabilities["full_horizons"] is False
    with pytest.raises((TypeError, ValueError)):
        PlanStore().install_initial(legacy)


def test_legacy_sidecar_metadata_remains_readable_and_immutable_after_pickle(tmp_path):
    path = tmp_path / "legacy.npz"
    np.savez(path, wheel_cmds=np.ones((3, 2)), poses=np.zeros((3, 3)))
    path.with_suffix(".json").write_text(json.dumps({"scenario": {"seed": 7}}))
    legacy = load_plan(path)
    restored = pickle.loads(pickle.dumps(legacy))
    assert isinstance(restored, LegacyPlan)
    assert restored.metadata["scenario"]["seed"] == 7
    assert restored.capabilities["raw_states"] is False
    with pytest.raises(TypeError):
        restored.metadata["scenario"]["seed"] = 9
    for array in restored.arrays.values():
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_load_rejects_unknown_schema_and_object_arrays(tmp_path):
    _, metadata_path = save_plan(record(), tmp_path / "versioned")
    metadata = json.loads(Path(metadata_path).read_text())
    metadata["schema_version"] = 999
    Path(metadata_path).write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_plan(tmp_path / "versioned")
    objects = tmp_path / "objects.npz"
    np.savez(objects, poses=np.array([{"not": "numeric"}], dtype=object))
    with pytest.raises(ValueError):
        load_plan(objects)


def test_warm_seed_uses_only_valid_tail_and_pins_all_five_initial_states():
    segment = SegmentSolution.from_ppoly(spline(), field_key())
    times = np.array([0.0, 0.1, 0.2])
    initial = np.array([0.25, -0.1, 0.2, 0.8, 1.1])
    seed = segment.warm_seed(times, shift=0.2, initial_state=initial)
    np.testing.assert_array_equal(seed[:5, 0], initial)
    np.testing.assert_array_equal(seed[[0, 1, 3, 4, 5, 6, 7, 8, 9], 1:],
                                  spline()(times[1:] + 0.2).T[[0, 1, 3, 4, 5, 6, 7, 8, 9]])
    state_seed = segment.warm_seed(times, shift=0.2, initial_state=initial, mode="states")
    np.testing.assert_array_equal(state_seed[5:], np.zeros((5, 3)))
    with pytest.raises(ValueError):
        segment.warm_seed(times, shift=0.4, initial_state=initial)
    with pytest.raises(ValueError):
        segment.evaluate([0.0, 0.6])


def test_warm_seed_requires_a_finite_explicit_extension_for_uncovered_tail():
    segment = SegmentSolution.from_ppoly(spline(), field_key())
    times = np.array([0.0, 0.1, 0.2, 0.3])
    observed = []

    def tail(query):
        observed.append(query.copy())
        result = np.zeros((10, len(query)))
        result[0] = query
        result[3:5] = 1.0
        return result

    seed = segment.warm_seed(times, shift=0.4,
                             initial_state=np.array([0.4, 0, 0, 1, 1]), tail=tail)
    np.testing.assert_allclose(observed[0], [0.6, 0.7])
    np.testing.assert_array_equal(seed[0], times + 0.4)
    with pytest.raises(ValueError):
        segment.warm_seed(times, shift=0.4, initial_state=np.zeros(5),
                          tail=lambda query: np.full((10, len(query)), np.nan))


def test_pending_plan_leaves_active_intact_until_checked_activation():
    active = record()
    candidate = replacement(active)
    store = PlanStore()
    store.install_initial(active)
    stage(store, candidate)
    assert store.active is active
    commit(store, candidate)
    assert store.active is candidate
    assert store.pending is None
    assert active in store.previous


@pytest.mark.parametrize("failure", ["field", "session", "state", "time", "deadline"])
def test_stale_or_mismatched_pending_plan_never_replaces_active(failure):
    active = record()
    candidate = replacement(active)
    store = PlanStore()
    store.install_initial(active)
    stage(store, candidate)
    overrides = {}
    if failure == "field":
        overrides["current_field_key"] = field_key(3)
    elif failure == "session":
        overrides["session_id"] = "restarted-session"
    elif failure == "state":
        overrides["measured_state"] = candidate.raw_states[0] + np.array([0.01, 0, 0, 0, 0])
    elif failure == "time":
        overrides["now"] = candidate.start_epoch - 0.01
    else:
        overrides["now"] = candidate.start_epoch + 0.1
    with pytest.raises(ValueError):
        commit(store, candidate, **overrides)
    assert store.active is active


@pytest.mark.parametrize("change", [
    {"parent_id": "another-plan"},
    {"session_id": "another-session"},
    {"start_epoch": 100.0},
    {"status": "failed"},
])
def test_staging_rejects_wrong_parent_epoch_session_and_failed_plans(change):
    active = record()
    candidate = replace(replacement(active), **change)
    store = PlanStore()
    store.install_initial(active)
    with pytest.raises(ValueError):
        stage(store, candidate)
    assert store.active is active and store.pending is None


def test_budget_rejects_candidate_without_dropping_active():
    active = record()
    candidate = replacement(active)
    store = PlanStore(max_records=2, max_bytes=active.array_bytes + candidate.array_bytes - 1)
    store.install_initial(active)
    with pytest.raises(MemoryError):
        stage(store, candidate)
    assert store.active is active and store.pending is None


def test_staging_requires_matching_validation_and_raw_anchor_state():
    active = record()
    candidate = replacement(active)
    store = PlanStore()
    store.install_initial(active)
    with pytest.raises(ValueError):
        store.stage(candidate, anchor_index=1, validation_field_key=field_key(3),
                    deadline=candidate.start_epoch + 0.05)
    shifted = record(plan_id=candidate.plan_id, parent_id=candidate.parent_id,
                     key=candidate.field_key, epoch=candidate.start_epoch,
                     x0=candidate.raw_states[0, 0] + 0.02)
    with pytest.raises(ValueError):
        stage(store, shifted)
    assert store.active is active and store.pending is None


def test_record_limit_evicts_previous_plans_but_preserves_active_and_pending():
    first = record()
    second = replacement(first)
    third = replacement(second, plan_id="plan-3", revision=3)
    store = PlanStore(max_records=2)
    store.install_initial(first)
    stage(store, second)
    commit(store, second)
    assert store.active is second and first in store.previous
    stage(store, third)
    assert store.active is second and not store.previous
    commit(store, third)
    assert store.active is third and second in store.previous
    store.clear()
    assert store.active is None and store.pending is None and not store.previous
