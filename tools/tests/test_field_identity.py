"""Field identities, owned snapshots, and one field per solver request."""

from dataclasses import replace
from collections import deque
import importlib
import pickle
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.maps.grid import GridGeometry, MapSnapshot, OccupancyPolicy
    from agx_planning.vector_field.contracts import FieldKey, FieldSnapshot
    from agx_planning.vector_field.grid import VectorFieldGrid
    from agx_planning.vector_field.preparation import prepare_travel_time
    from agx_planning.vector_field.field import (
        VectorFieldResult, pack_field_array, field_result_to_grid, world_to_grid, grid_to_world,
    )
    from agx_planning.pmp_planner.config import PlannerConfig
    from agx_planning.pmp_planner.rollout import parse_field_array, RolloutResult
    from agx_planning.pmp_planner.solver_process import SolverProcess, _child_main
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


def snapshot(revision=1, offset=0.0):
    geometry = GridGeometry((5, 7), 0.25, (-0.5, -0.25, 0.0))
    key = FieldKey("session", revision, geometry, (1, 2), "recipe", "native-revision", "policy")
    tt = np.arange(35, dtype=float).reshape(5, 7) / 8 + offset
    obstacles = np.zeros(tt.shape, dtype=bool)
    obstacles[0, 0] = True
    tt[0, 0] = np.nan
    return FieldSnapshot(key, tt, obstacles, np.isfinite(tt), np.ones(tt.shape),
                         np.ones(tt.shape) * 0.25)


def test_field_key_roundtrip_and_complete_identity():
    key = snapshot().key
    assert FieldKey.from_dict(key.to_dict()) == key
    assert hash(FieldKey.from_dict(key.to_dict())) == hash(key)
    changes = [dict(map_session="new"), dict(map_revision=2), dict(goal_cell=(2, 2)),
               dict(recipe_id="other"), dict(numerical_version="other"),
               dict(occupancy_policy_id="other"), dict(domain="partial"),
               dict(geometry=replace(key.geometry, frame="other")),
               dict(source_definition="other")]
    for change in changes:
        assert replace(key, **change).digest != key.digest
    occupancy = np.zeros(key.geometry.shape, dtype=np.int8)
    source = MapSnapshot("session", 1, key.geometry, occupancy, OccupancyPolicy())
    generated = FieldKey.for_map(source, key.goal_cell, {"centring": True}, "native-revision")
    assert generated.recipe_id == FieldKey.for_map(source, key.goal_cell,
                                                   {"centring": True}, "native-revision").recipe_id
    assert generated.map_session == source.session and generated.map_revision == source.revision
    assert generated.occupancy_policy_id == source.policy.identity
    conservative = replace(source, policy=OccupancyPolicy(65, False))
    assert FieldKey.for_map(conservative, key.goal_cell, {"centring": True},
                            "native-revision").occupancy_policy_id != generated.occupancy_policy_id


def test_snapshot_owns_immutable_arrays_even_after_pickle():
    value = snapshot()
    raw = np.array(value.travel_time)
    copied = replace(value, travel_time=raw)
    raw[1, 1] = 900
    assert copied.travel_time[1, 1] != 900
    for candidate in (copied, pickle.loads(pickle.dumps(copied))):
        for array in (candidate.travel_time, candidate.obstacle_mask, candidate.reached_mask,
                      candidate.speed, candidate.wall_distance, candidate.uncomputed_mask):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
        assert candidate.key == value.key


@pytest.mark.parametrize("change", ["mask", "source", "infinite", "partial", "wall"])
def test_snapshot_rejects_inconsistent_channels(change):
    value = snapshot()
    kwargs = {}
    if change == "mask":
        kwargs["reached_mask"] = np.ones(value.key.geometry.shape, dtype=bool)
    elif change == "source":
        kwargs["key"] = replace(value.key, goal_cell=(0, 0))
    elif change == "infinite":
        tt = np.array(value.travel_time)
        tt[0, 0] = np.inf
        kwargs["travel_time"] = tt
    elif change == "partial":
        kwargs["uncomputed_mask"] = np.ones(value.key.geometry.shape, dtype=bool)
    else:
        kwargs["wall_distance"] = np.full(value.key.geometry.shape, np.nan)
    with pytest.raises(ValueError):
        replace(value, **kwargs)


def test_identified_grid_rejects_wrong_goal_and_legacy_is_unidentified():
    value = snapshot()
    grid = value.build_grid(PlannerConfig())
    x, y = value.key.geometry.cell_center(*value.key.goal_cell)
    grid.require_goal([x, y, 1.3])
    with pytest.raises(ValueError, match="source"):
        grid.require_goal([x + value.key.geometry.resolution, y, 0])
    with pytest.raises(ValueError):
        grid.require_goal([-1, y, 0])
    assert grid.identified and grid.field_key == value.key
    assert grid.frozen_copy() is grid
    with pytest.raises(ValueError, match="frozen"):
        grid.update(value.travel_time, -0.5, -0.25, 0.25)
    legacy = VectorFieldGrid()
    legacy.update(value.travel_time, -0.5, -0.25, 0.25)
    assert not legacy.identified and legacy.field_key is None


def test_legacy_content_identity_replaces_local_version_equality():
    first, second = VectorFieldGrid(), VectorFieldGrid()
    tt = np.arange(20.0).reshape(4, 5)
    direction = (np.ones(tt.shape), np.zeros(tt.shape))
    first.update(tt, 0, 0, 1, direction=direction)
    second.update(tt + 1, 0, 0, 1, direction=direction)
    assert first.version == second.version == 1
    assert first.identity != second.identity
    same = VectorFieldGrid()
    same.update(tt, 0, 0, 1, direction=direction)
    assert same.identity == first.identity
    frozen = first.frozen_copy()
    expected = frozen.query_scalar(1.5, 1.5)
    direction[0][:] = 0
    tt[:] = 100
    assert frozen.query_scalar(1.5, 1.5) == expected
    first.update(tt, 0, 0, 1)
    assert frozen.identity != first.identity
    assert frozen.query_scalar(1.5, 1.5) == expected


def test_cell_centres_and_outer_half_cells_use_one_geometry():
    # Each row/column value is sampled at its cell centre.
    tt = np.array([[0, 1, 2], [10, 11, 12], [20, 21, 22]], dtype=float)
    grid = VectorFieldGrid()
    grid.update(tt, -1, -2, 0.5, wall_dist=tt, wall_sigma=0)
    for row in range(3):
        for col in range(3):
            x, y = -1 + (col + 0.5) * 0.5, -2 + (row + 0.5) * 0.5
            assert grid.query_scalar(x, y)[0] == tt[row, col]
            assert grid.query_dist(np.array([x]), np.array([y]))[0][0] == tt[row, col]
    assert grid.query_scalar(-1, -2)[0] == 0
    assert grid.query_scalar(0.49, -0.51)[0] == 22
    np.testing.assert_array_equal(grid.in_bounds(np.array([-1, 0.49, 0.5, -1.01]),
                                                np.array([-2, -0.51, -1, -1])),
                                  [True, True, False, False])
    assert grid.query_scalar(0.5, -1)[1:] == (0, 0, 0, 0)
    assert grid.query_scalar(np.nan, -1)[1:] == (0, 0, 0, 0)
    assert grid.query_dist(np.array([np.nan]), np.array([-1]))[0][0] == -1


def test_direct_and_packed_fields_share_invalid_preparation_at_declared_precision():
    value = snapshot()
    tt = np.array(value.travel_time)
    zeros = np.zeros_like(tt)
    result = VectorFieldResult(travel_time=tt, grad_x=zeros, grad_y=zeros, grad_mag=zeros,
                               free_max_T=float(np.nanmax(tt)), origin_x=-0.5, origin_y=-0.25,
                               resolution=0.25, wall_dist=np.array(value.wall_distance))
    direct = field_result_to_grid(result, wall_sigma=0)
    packed = parse_field_array(pack_field_array(result), PlannerConfig(fp_smooth_sigma=0))
    typed = value.build_grid(PlannerConfig(fp_smooth_sigma=0))
    for name in ("_tt", "_dT_dx", "_dT_dy", "_wd"):
        np.testing.assert_array_equal(getattr(direct, name), getattr(packed, name))
        np.testing.assert_array_equal(getattr(direct, name), getattr(typed, name))
    assert direct._tt[0, 0] == 4 * np.nanmax(tt) + 1
    np.testing.assert_array_equal(prepare_travel_time(np.full((2, 2), np.nan)), np.ones((2, 2)))


def test_world_bounds_match_geometry_despite_normalized_rounding():
    geometry = GridGeometry((3, 3), 0.03, (-4.1, -4.1, 0.0))
    tt = np.arange(9, dtype=float).reshape(3, 3)
    grid = VectorFieldGrid()
    grid.update(tt, *geometry.origin[:2], geometry.resolution)
    upper = geometry.origin[0] + geometry.width * geometry.resolution
    middle = geometry.cell_center(1, 1)[1]
    # The normalized coordinate rounds below 3 even at the exact upper edge.
    assert (upper - geometry.origin[0]) / geometry.resolution < geometry.width
    assert not grid.in_bounds(np.array([upper]), np.array([middle]))[0]
    with pytest.raises(ValueError, match="outside"):
        geometry.world_to_cell(upper, middle)
    result = VectorFieldResult(tt, tt + 1, tt + 2, tt + 3, float(tt.max()),
                               *geometry.origin[:2], geometry.resolution)
    assert result.query_vec(upper, middle) is None
    assert result.query_vec(np.nan, middle) is None
    for row in range(3):
        for col in range(3):
            x, y = geometry.cell_center(col, row)
            np.testing.assert_allclose(result.query_vec(x, y),
                                       [tt[row, col] + 1, tt[row, col] + 2, tt[row, col]],
                                       atol=1e-12)
    np.testing.assert_array_equal(result.query_vec(*geometry.origin[:2]), [1, 2, 0])
    np.testing.assert_allclose(result.query_vec(upper - 0.01, upper - 0.01), [9, 10, 8])


def test_empty_result_adapters_use_the_same_declared_sentinel():
    tt = np.full((3, 3), np.nan)
    zeros = np.zeros(tt.shape)
    result = VectorFieldResult(tt, zeros, zeros, zeros, 1.0, 0.0, 0.0, 1.0)
    direct = field_result_to_grid(result)
    packed = parse_field_array(pack_field_array(result), PlannerConfig())
    np.testing.assert_array_equal(direct._tt, np.full(tt.shape, 5.0))
    np.testing.assert_array_equal(direct._tt, packed._tt)


def test_legacy_world_cell_helpers_match_canonical_geometry():
    geometry = GridGeometry((3, 3), 0.03, (-4.1, -4.1, 0.0))
    args = (*geometry.origin[:2], geometry.resolution, geometry.width, geometry.height)
    ox, oy = geometry.origin[:2]
    upper = ox + geometry.width * geometry.resolution
    for x, y in ((ox, oy), geometry.cell_center(2, 2), (upper - 0.01, oy)):
        assert world_to_grid(x, y, *args) == geometry.world_to_cell(x, y)
    for x, y in ((ox - 0.01, oy), (ox, oy - 0.01), (upper, oy), (np.nan, oy)):
        assert world_to_grid(x, y, *args) is None
    assert grid_to_world(2, 2, ox, oy, geometry.resolution) == geometry.cell_center(2, 2)


class FakeConnection:
    def __init__(self, messages):
        self.messages = deque(messages)
        self.during = deque()
        self.sent = []

    def recv(self):
        if self.during:
            return self.during.popleft()
        if not self.messages:
            raise EOFError
        return self.messages.popleft()

    def poll(self, *args):
        return bool(self.during)

    def send(self, value):
        self.sent.append(value)


def install_fake_rollout(monkeypatch, conn, during=()):
    seen = []
    solver_module = importlib.import_module("agx_planning.pmp_planner.shooting_solver")
    rollout_module = importlib.import_module("agx_planning.pmp_planner.rollout")

    class FakeSolver:
        def __init__(self, cfg, field):
            self.field = field

        def reset_warm_start(self):
            pass

    def generate(solver, cfg, x0, goal, stop_fn):
        initial = solver.field
        seen.append(initial.field_key)
        if len(seen) == 1:
            conn.during.extend(during)
        assert stop_fn() is None
        assert solver.field is initial
        yield "one chunk"
        assert stop_fn() is None
        assert solver.field is initial
        return RolloutResult("success", "done")

    monkeypatch.setattr(solver_module, "PMPShootingSolver", FakeSolver)
    monkeypatch.setattr(rollout_module, "rollout_generator", generate)
    return seen


def test_worker_keeps_one_field_and_coalesces_later_updates(monkeypatch):
    first, middle, latest = snapshot(1), snapshot(2, 1), snapshot(3, 2)
    x, y = first.key.geometry.cell_center(*first.key.goal_cell)
    goal = np.array([x, y, 0])
    state = np.array([x, y, 0, 0, 0])
    conn = FakeConnection([("field", first), ("plan", state, goal, first.key),
                           ("plan", state, goal, latest.key), ("quit",)])
    seen = install_fake_rollout(monkeypatch, conn, [("field", middle), ("field", latest)])
    _child_main(conn, SimpleNamespace(is_set=lambda: False), PlannerConfig())
    assert seen == [first.key, latest.key]
    assert [payload.status for kind, payload in conn.sent if kind == "done"] == ["success", "success"]


def test_bad_pending_field_cannot_leave_a_stale_valid_field(monkeypatch):
    first = snapshot()
    x, y = first.key.geometry.cell_center(*first.key.goal_cell)
    state, goal = np.array([x, y, 0, 0, 0]), np.array([x, y, 0])
    conn = FakeConnection([("field", first), ("plan", state, goal, first.key),
                           ("plan", state, goal, first.key), ("quit",)])
    seen = install_fake_rollout(monkeypatch, conn, [("field", np.array([1, 2]))])
    _child_main(conn, SimpleNamespace(is_set=lambda: False), PlannerConfig())
    assert seen == [first.key]
    done = [payload for kind, payload in conn.sent if kind == "done"]
    assert [result.status for result in done] == ["success", "failed"]
    assert "Invalid field" in done[-1].message


def test_worker_rejects_a_plan_for_a_different_key(monkeypatch):
    value = snapshot()
    x, y = value.key.geometry.cell_center(*value.key.goal_cell)
    conn = FakeConnection([("field", value),
                           ("plan", np.array([x, y, 0, 0, 0]), np.array([x, y, 0]),
                            replace(value.key, map_revision=4)), ("quit",)])
    seen = install_fake_rollout(monkeypatch, conn)
    _child_main(conn, SimpleNamespace(is_set=lambda: False), PlannerConfig())
    assert seen == []
    done = [payload for kind, payload in conn.sent if kind == "done"]
    assert done[0].status == "failed" and "key" in done[0].message


def test_parent_owns_legacy_payload_and_rejects_use_after_close():
    process = SolverProcess.__new__(SolverProcess)
    process._send_lock = threading.Lock()
    process._closed = False
    process._conn = FakeConnection([])
    raw = np.arange(9, dtype=float)
    process.set_field(raw)
    raw[:] = -1
    np.testing.assert_array_equal(process._field, np.arange(9))
    process._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        process.set_field(raw)


def test_spawned_worker_checks_missing_field_source_and_close():
    process = SolverProcess(PlannerConfig())

    def finish(state, goal):
        deadline = time.monotonic() + 10
        generator = process.rollout(state, goal, lambda: None)
        while True:
            try:
                next(generator)
            except StopIteration as stop:
                return stop.value
            if time.monotonic() >= deadline:
                pytest.fail("spawned solver did not answer")
            time.sleep(0.01)

    try:
        value = snapshot()
        x, y = value.key.geometry.cell_center(*value.key.goal_cell)
        state, goal = np.array([x, y, 0, 0, 0]), np.array([x, y, 0])
        missing = finish(state, goal)
        assert missing.status == "failed" and "No field" in missing.message
        process.set_field(value)
        assert finish(state, goal).status == "success"
        wrong_goal = goal + np.array([value.key.geometry.resolution, 0, 0])
        mismatch = finish(state, wrong_goal)
        assert mismatch.status == "failed" and "source cell" in mismatch.message
    finally:
        process.close()
    process.close()
    with pytest.raises(RuntimeError, match="closed"):
        next(process.rollout(state, goal, lambda: None))
