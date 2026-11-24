"""Check unread input edits that retain a complete march without snapshots."""

from dataclasses import fields
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
import skfmm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
    from agx_planning.vector_field.incremental import IncrementalEikonal
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def inputs(random_speed=False):
    mask = np.zeros((41, 81), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, 30, -1]] = True
    speed = (np.random.default_rng(611).uniform(0.8, 1.2, mask.shape)
             if random_speed else np.ones(mask.shape))
    return mask, speed, 10, 20, 0.1


def controller():
    return IncrementalEikonal(checkpoint_interval=10, max_checkpoints=4,
                              cache_bytes=4 * 1024**2)


def assert_complete_reuse(solver, total):
    stats = solver.last_stats
    assert stats["status"] == "replay"
    assert stats["reason"] == "complete_state_reused"
    assert stats["changed_cells"] > 0
    assert stats["first_affected_batch"] == -1
    assert stats["reused_batches"] == stats["total_batches"] == total
    assert stats["computed_batches"] == 0
    assert stats["checkpoint_count"] == stats["checkpoint_copies"] == 0
    assert stats["checkpoint_copy_bytes"] == 0
    assert stats["checkpoint_batches"] == []
    assert solver._checkpoints == []
    assert solver._state.stats["complete"]
    assert stats["cache_array_bytes"] == solver._state.stats["working_array_bytes"]


def solve_without_marching_or_copying(solver, request):
    with patch.object(solver, "_finish", side_effect=AssertionError("unexpected finish")), \
            patch.object(skfmm.TravelTimeState, "checkpoint", side_effect=AssertionError("unexpected snapshot")), \
            patch.object(skfmm.TravelTimeState, "advance", side_effect=AssertionError("unexpected march")):
        return solver.solve(*request)


@pytest.mark.parametrize("random_speed", [False, True])
@pytest.mark.parametrize("edit", ["faster", "slower", "add", "remove", "zero", "positive"])
def test_unread_inputs_reuse_current_state_and_invalidate_old_checkpoints(edit, random_speed):
    request = inputs(random_speed)
    mask, speed = request[:2]
    cell = (20, 55)
    if edit == "remove":
        mask[cell] = True
    elif edit == "positive":
        speed[cell] = 0.0
    solver = controller()
    solver.solve(*request)
    total = solver.last_stats["total_batches"]
    assert max(solver.last_stats["checkpoint_batches"]) < total
    state, saved = solver._state, list(solver._checkpoints)
    if edit == "faster":
        speed[cell] = 2.0
    elif edit == "slower":
        speed[cell] = 0.3
    elif edit in ("add", "remove"):
        mask[cell] = not mask[cell]
    elif edit == "zero":
        speed[cell] = 0.0
    else:
        speed[cell] = 1.0
    actual = solve_without_marching_or_copying(solver, request)
    np.testing.assert_array_equal(actual, vf.solve_eikonal(*request))
    assert solver._state is state
    assert_complete_reuse(solver, total)
    for checkpoint, _, _ in saved:
        with pytest.raises(ValueError):
            state.restore(checkpoint)


def test_repeated_unread_edits_and_unchanged_inputs_need_no_checkpoints():
    request = inputs()
    mask, speed = request[:2]
    solver = controller()
    solver.solve(*request)
    total = solver.last_stats["total_batches"]
    for step in range(4):
        speed[20, 55] = 2.0 if step % 2 == 0 else 0.5
        mask[25, 65] = not mask[25, 65]
        actual = solve_without_marching_or_copying(solver, request)
        expected = vf.solve_eikonal(*request)
        np.testing.assert_array_equal(actual, expected)
        assert_complete_reuse(solver, total)
        actual[:] = -100
        np.testing.assert_array_equal(solve_without_marching_or_copying(solver, request), expected)
        assert solver.last_stats["status"] == "unchanged"
        assert solver.last_stats["checkpoint_count"] == 0
        assert solver.last_stats["checkpoint_copies"] == 0


@pytest.mark.parametrize("edit", ["speed", "open_connection"])
def test_later_read_edit_falls_back_without_old_checkpoints(edit):
    request = inputs()
    mask, speed = request[:2]
    solver = controller()
    solver.solve(*request)
    speed[20, 55] = 2.0
    solver.solve(*request)
    state = solver._state
    assert solver.last_stats["reason"] == "complete_state_reused"
    if edit == "speed":
        speed[20, 20] = 0.3
    else:
        mask[20, 30] = False
    actual = solver.solve(*request)
    np.testing.assert_array_equal(actual, vf.solve_eikonal(*request))
    assert solver._state is not state
    assert solver.last_stats["status"] == "full_recompute"
    assert solver.last_stats["reason"] == "no_safe_checkpoint"
    assert solver.last_stats["checkpoint_copies"] > 0
    if edit == "open_connection":
        assert np.isfinite(actual[20, 55])


@pytest.mark.parametrize("reset", ["clear", "invalid_speed"])
def test_complete_reuse_state_can_be_reset(reset):
    request = inputs()
    solver = controller()
    solver.solve(*request)
    request[1][20, 55] = 2.0
    solver.solve(*request)
    assert solver.last_stats["reason"] == "complete_state_reused"
    if reset == "clear":
        solver.clear()
    else:
        request[1][20, 55] = np.nan
        with pytest.raises(ValueError):
            solver.solve(*request)
        request[1][20, 55] = 2.0
    assert solver._state is None and solver._checkpoints == []
    assert solver.last_stats == {}
    np.testing.assert_array_equal(solver.solve(*request), vf.solve_eikonal(*request))
    assert solver.last_stats["status"] == "full_recompute"
    assert solver.last_stats["reason"] == "new_field"


@pytest.mark.parametrize("fallback", ["no_slots", "no_bytes", "early_exit"])
def test_stateless_fallback_policy_is_unchanged(fallback):
    request = inputs()
    options = {"max_checkpoints": 0} if fallback == "no_slots" else {}
    if fallback == "no_bytes":
        options["cache_bytes"] = 0
    solver = IncrementalEikonal(**options)
    kwargs = {"early_exit_target": (20, 20)} if fallback == "early_exit" else {}
    for value in (1.0, 2.0):
        request[1][20, 55] = value
        actual = solver.solve(*request, **kwargs)
        np.testing.assert_array_equal(actual, vf.solve_eikonal(*request, **kwargs))
        assert solver.last_stats["status"] == "full_recompute"
        assert solver.last_stats["reason"] == ("early_exit" if fallback == "early_exit" else "cache_budget")
        assert solver.last_stats["checkpoint_copies"] == 0
        assert solver._state is None


def test_stateless_fallback_clears_a_previously_reused_complete_state():
    request = inputs()
    solver = controller()
    solver.solve(*request)
    request[1][20, 55] = 2.0
    solver.solve(*request)
    assert solver.last_stats["reason"] == "complete_state_reused"
    actual = solver.solve(*request, early_exit_target=(20, 20))
    expected = vf.solve_eikonal(*request, early_exit_target=(20, 20))
    np.testing.assert_array_equal(actual, expected)
    assert solver._state is None
    assert solver.last_stats["reason"] == "early_exit"


def assert_field(actual, expected):
    assert actual is not None and expected is not None
    assert actual[1] == expected[1]
    for item in fields(vf.VectorFieldResult):
        value, reference = getattr(actual[0], item.name), getattr(expected[0], item.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, reference, err_msg=item.name)
        else:
            assert value == reference, item.name


@pytest.mark.parametrize("centring", [False, True])
def test_unread_map_edits_preserve_every_field_channel(centring):
    mask, _, col, row, resolution = inputs()
    occupancy = np.where(mask, 100, 0).astype(np.int8)
    cfg = vf.VectorFieldConfig(early_exit_enable=False, centring=centring,
                                centring_cap=0.4, inflation_radius=0.2,
                                smooth_T_before_grad=True)
    request = dict(map_array=occupancy, goal_col=col, goal_row=row,
                   resolution=resolution, origin_x=-3.0, origin_y=-2.0, cfg=cfg)
    updater = FieldUpdater(incremental=True, checkpoint_interval=10)
    assert_field(updater.update(**request), vf.compute_field(**request))
    total = updater.last_fmm["total_batches"]
    for value in (100, 0, 100):
        occupancy[16:24, 55] = value
        with patch.object(updater._eikonal, "_finish", side_effect=AssertionError("unexpected finish")), \
                patch.object(skfmm.TravelTimeState, "checkpoint", side_effect=AssertionError("unexpected snapshot")):
            actual = updater.update(**request)
        assert_field(actual, vf.compute_field(**request))
        assert updater.last_update == "fmm_replay"
        assert_complete_reuse(updater._eikonal, total)
        actual[0].wall_dist[:] = -100
    assert_field(updater.update(**request), vf.compute_field(**request))
    assert updater.last_update == "field_reuse" and updater.last_fmm == {}
    occupancy[20, 30] = 0
    assert_field(updater.update(**request), vf.compute_field(**request))
    assert updater.last_fmm["status"] == "full_recompute"
