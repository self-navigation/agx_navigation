"""Map revisions, owned occupancy, and canonical geometry without ROS."""

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

from agx_planning.maps import GridGeometry, MapDelta, MapSnapshot, OccupancyPolicy
from agx_planning.maps.io import apply_rectangles, load_map, load_snapshot


def snapshot(values=None, *, session="session-a", revision=0):
    geometry = GridGeometry((3, 4), 0.25, (-1, -2, 0))
    return MapSnapshot(session, revision, geometry,
                       np.zeros(geometry.shape, dtype=np.int8) if values is None else values)


def test_floor_cells_centres_and_half_open_map_boundaries():
    geometry = GridGeometry([3, 4], 0.25, [-1, -2, 0])
    assert geometry.world_to_cell(-1, -2) == (0, 0)
    assert geometry.world_to_cell(-0.75, -1.75) == (1, 1)
    assert geometry.cell_center(0, 0) == (-0.875, -1.875)
    for row in range(3):
        for col in range(4):
            assert geometry.world_to_cell(*geometry.cell_center(col, row)) == (col, row)
    for point in [(-1.001, -1.8), (-0.9, -2.001), (0, -1.8), (-0.9, -1.25)]:
        with pytest.raises(ValueError, match="outside"):
            geometry.world_to_cell(*point)
    with pytest.raises(ValueError, match="finite"):
        geometry.world_to_cell(np.nan, 0)
    with pytest.raises(ValueError):
        geometry.cell_center(-1, 0)
    with pytest.raises(ValueError):
        geometry.cell_center(0, 3)
    assert GridGeometry.from_dict(geometry.to_dict()) == geometry
    assert hash(GridGeometry.from_dict(geometry.to_dict())) == hash(geometry)
    assert geometry.to_dict(include_metadata=False) == {
        "shape": [3, 4], "resolution": 0.25, "origin": [-1.0, -2.0, 0.0],
    }


@pytest.mark.parametrize("options", [
    {"shape": (0, 4)}, {"shape": (3.0, 4)}, {"shape": (True, 4)},
    {"resolution": 0}, {"resolution": float("inf")},
    {"origin": (0, 0, 0.1)}, {"origin": (0, float("nan"), 0)},
    {"convention_version": 2}, {"frame": ""},
])
def test_invalid_geometry_is_rejected(options):
    with pytest.raises(ValueError):
        GridGeometry(**{"shape": (3, 4), "resolution": 0.25, **options})


def test_occupancy_threshold_and_unknown_policy_are_explicit():
    occupancy = np.array([[-1, 0, 64, 65, 100]], dtype=np.int16)
    free_unknown = OccupancyPolicy(65, True)
    blocked_unknown = OccupancyPolicy(65, False)
    np.testing.assert_array_equal(free_unknown.mask(occupancy), [[False, False, False, True, True]])
    np.testing.assert_array_equal(blocked_unknown.mask(occupancy), [[True, False, False, True, True]])
    assert free_unknown.identity != blocked_unknown.identity
    with pytest.raises(ValueError):
        OccupancyPolicy(True, True)
    with pytest.raises(ValueError):
        OccupancyPolicy(65, 1)


def test_extreme_geometry_and_outside_coordinates_fail_without_overflow():
    geometry = GridGeometry((3, 4), 1e-308)
    with pytest.raises(ValueError, match="outside"):
        geometry.world_to_cell(1e308, 0)
    with pytest.raises(ValueError, match="extent"):
        GridGeometry((3, 4), 1e308)
    with pytest.raises(ValueError, match="extent"):
        GridGeometry((3, 4), 0.1, (1e308, 0, 0))


def test_snapshot_arrays_cannot_alias_or_be_made_writeable():
    original = np.zeros((3, 4), dtype=np.int16)
    original[0, 0] = -1
    result = snapshot(original)
    identity = result.identity
    original[:] = 100
    assert result.occupancy[0, 0] == -1 and not result.obstacles.any()
    for array in (result.occupancy, result.obstacles):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 1
    assert result.identity == identity
    same = snapshot(result.occupancy.copy())
    assert result == same and result.identity == same.identity
    changed_revision = result.updated(result.occupancy)
    assert changed_revision.revision == 1 and changed_revision.identity != result.identity
    changed_session = snapshot(result.occupancy, session="session-b")
    assert changed_session.identity != result.identity
    changed_content = result.occupancy.copy()
    changed_content[2, 3] = 100
    assert snapshot(changed_content).identity != result.identity


@pytest.mark.parametrize("occupancy", [
    np.zeros((3, 4), dtype=float), np.zeros((3, 4), dtype=bool),
    np.zeros((4, 3), dtype=np.int8), np.full((3, 4), -2), np.full((3, 4), 101),
])
def test_invalid_snapshot_occupancy_is_rejected(occupancy):
    with pytest.raises(ValueError):
        snapshot(occupancy)


def test_delta_round_trip_and_parent_rejection():
    before = snapshot()
    changed = before.occupancy.copy()
    changed[0, 2], changed[2, 3] = 100, -1
    after = before.updated(changed, revision=3)
    delta = MapDelta.between(before, after)
    changed[:] = 50
    observed = delta.apply(before)
    assert observed == after
    np.testing.assert_array_equal(observed.occupancy, after.occupancy)
    for parent in [snapshot(session="wrong"), snapshot(revision=1),
                   snapshot(np.full((3, 4), 5, dtype=np.int8)),
                   MapSnapshot("session-a", 0, GridGeometry((3, 4), 0.5), before.occupancy)]:
        with pytest.raises(ValueError, match="parent"):
            delta.apply(parent)
    for array in (delta.indices, delta.values):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(ValueError, match="greater"):
        before.updated(before.occupancy, revision=0)


def test_delta_validation_and_empty_revision():
    before = snapshot()
    unchanged = before.updated(before.occupancy)
    delta = MapDelta.between(before, unchanged)
    assert delta.indices.size == 0 and delta.apply(before) == unchanged
    assert MapDelta("session-a", 0, 1, before.geometry, [], []).apply(before) == unchanged
    for indices, values in [([1, 1], [100, 0]), ([-1], [100]), ([12], [100]),
                            ([1.0], [100]), ([1], [101]), ([1, 2], [100])]:
        with pytest.raises(ValueError):
            MapDelta("session-a", 0, 1, before.geometry, indices, values)
    with pytest.raises(ValueError):
        MapDelta("session-a", 1, 1, before.geometry, [1], [100])


def test_yaml_loader_keeps_pixel_semantics_and_records_source(tmp_path):
    # Exact threshold boundary pixels: 0, 0.2, 0.6, and 1 occupancy probability.
    image = np.array([[255, 204], [102, 0]], dtype=np.uint8)
    Image.fromarray(image).save(tmp_path / "map.png")
    path = tmp_path / "map.yaml"
    path.write_text("image: map.png\nresolution: 0.25\norigin: [-1,-2,0]\n"
                    "free_thresh: 0.2\noccupied_thresh: 0.6\nnegate: 0\n")
    occupancy, geometry, source = load_map({"yaml": "map.yaml"}, tmp_path)
    np.testing.assert_array_equal(occupancy, [[100, 100], [0, 0]])
    assert geometry == {"shape": [2, 2], "resolution": 0.25, "origin": [-1.0, -2.0, 0.0]}
    assert source["yaml"] == str(path)
    assert source["image"] == str(tmp_path / "map.png")
    assert len(source["yaml_sha256"]) == len(source["image_sha256"]) == 64
    loaded = load_snapshot(path, session="image-map", revision=7,
                           policy=OccupancyPolicy(65, False))
    np.testing.assert_array_equal(loaded.occupancy, occupancy)
    assert loaded.session == "image-map" and loaded.revision == 7
    path.write_text(path.read_text().replace("negate: 0", "negate: 1"))
    reversed_grid, _, reversed_source = load_map({"yaml": "map.yaml"}, tmp_path)
    np.testing.assert_array_equal(reversed_grid, [[-1, 0], [100, 100]])
    assert reversed_source["image_sha256"] == source["image_sha256"]
    assert reversed_source["yaml_sha256"] != source["yaml_sha256"]


def test_synthetic_loader_and_rectangle_inputs_are_owned():
    source = np.zeros((3, 4), dtype=np.int8)
    changed = apply_rectangles(source, [[1, 1, 3, 2, 100]])
    assert not source.any()
    np.testing.assert_array_equal(changed[1], [0, 100, 100, 0])
    spec = {"grid": {"shape": [3, 4], "resolution": 0.25, "origin": [0, 0, 0],
                     "rectangles": [[1, 1, 3, 2, 100]]}}
    occupancy, _, provenance = load_map(spec, Path("."))
    np.testing.assert_array_equal(occupancy, changed)
    assert provenance == {"synthetic": spec}
