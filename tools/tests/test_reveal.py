"""Whole-object observations stay deterministic and separate from true maps."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/agx_navigation/agx_planning"))

from agx_planning.maps import GridGeometry, MapSnapshot, OccupancyPolicy
from agx_planning.replanning.reveal import HiddenObject, ObjectRevealer


def make_map(values=None, *, allow_unknown=True):
    occupancy = np.zeros((9, 17), dtype=np.int8) if values is None else values
    return MapSnapshot("revealed", 0, GridGeometry(occupancy.shape, 1.0), occupancy,
                       OccupancyPolicy(65, allow_unknown))


def test_first_visible_cell_reveals_the_whole_object_and_preserves_old_snapshot():
    base = make_map()
    rectangles = [[5, 3, 6, 7, 100]]
    obj = HiddenObject("leaf", rectangles)
    model = ObjectRevealer(base, [obj], max_range=3.0, fov=math.pi / 2)
    rectangles[0][0] = 12
    assert not model.current.obstacles.any()
    assert model.true_map.obstacles.sum() == 4
    assert model.true_map.session != base.session
    initial, names = model.scan([0.5, 4.5, 0])
    assert initial is base and names == ()
    current, names = model.scan([3.5, 4.5, 0])
    assert names == ("leaf",)
    assert current.revision == 1
    assert current.obstacles.sum() == 4
    assert not base.obstacles.any()
    assert model.scan([3.5, 4.5, 0]) == (current, ())
    assert model.revealed_ids == ("leaf",)
    with pytest.raises(ValueError):
        model.true_map.occupancy.setflags(write=True)


def test_range_and_heading_limit_revelation():
    model = ObjectRevealer(make_map(), [HiddenObject("leaf", [[5, 4, 6, 5, 100]])],
                           max_range=2.0, fov=math.pi / 2)
    assert model.scan([2.5, 4.5, 0])[1] == ()
    assert model.scan([3.5, 4.5, math.pi])[1] == ()
    assert model.scan([3.5, 4.5, 2 * math.pi])[1] == ("leaf",)


def test_known_wall_occludes_a_hidden_object():
    occupancy = np.zeros((9, 17), dtype=np.int8)
    occupancy[:, 6] = 100
    model = ObjectRevealer(make_map(occupancy),
                           [HiddenObject("leaf", [[9, 4, 10, 5, 100]])], max_range=20)
    assert model.scan([2.5, 4.5, 0])[1] == ()
    assert model.scan([7.5, 4.5, 0])[1] == ("leaf",)


def test_true_hidden_object_can_occlude_another_without_revealing_its_shape():
    model = ObjectRevealer(make_map(),
                           [HiddenObject("front", [[5, 0, 6, 9, 100]]),
                            HiddenObject("back", [[9, 4, 10, 5, 100]])], max_range=20)
    observed, names = model.scan([2.5, 4.5, 0])
    assert names == ("front",)
    assert observed.occupancy[4, 9] == 0
    assert model.true_map.occupancy[4, 9] == 100
    assert model.scan([2.5, 4.5, 0])[1] == ()


def test_corner_contact_is_conservatively_occluded():
    occupancy = np.zeros((5, 5), dtype=np.int8)
    occupancy[0, 1] = 100
    model = ObjectRevealer(make_map(occupancy),
                           [HiddenObject("corner", [[3, 3, 4, 4, 100]])], max_range=10)
    assert model.scan([0.5, 0.5, math.pi / 4])[1] == ()


@pytest.mark.parametrize("allow_unknown, expected", [(True, ("leaf",)), (False, ())])
def test_unknown_cells_follow_explicit_occlusion_policy(allow_unknown, expected):
    occupancy = np.zeros((9, 17), dtype=np.int8)
    occupancy[:, 4] = -1
    model = ObjectRevealer(make_map(occupancy, allow_unknown=allow_unknown),
                           [HiddenObject("leaf", [[7, 4, 8, 5, 100]])], max_range=20)
    assert model.scan([2.5, 4.5, 0])[1] == expected


def test_known_base_wall_overlap_cannot_reveal_hidden_extension():
    occupancy = np.zeros((9, 17), dtype=np.int8)
    occupancy[:, 6] = 100
    model = ObjectRevealer(make_map(occupancy),
                           [HiddenObject("leaf", [[6, 4, 9, 5, 100]])], max_range=20)
    assert model.scan([2.5, 4.5, 0])[1] == ()
    assert model.scan([10.5, 4.5, math.pi])[1] == ("leaf",)


def test_controlled_reveal_is_atomic_ordered_and_idempotent():
    objects = [HiddenObject("first", [[5, 3, 6, 4, 100]]),
               HiddenObject("second", [[9, 3, 10, 4, 100]])]
    model = ObjectRevealer(make_map(), objects, max_range=1)
    before = model.current
    with pytest.raises(ValueError, match="unknown"):
        model.reveal_ids(["first", "bad"])
    assert model.current is before
    observed, names = model.reveal_ids(["second", "first", "second"])
    assert names == ("first", "second")
    assert observed.revision == 1
    assert observed.obstacles.sum() == 2
    assert model.reveal_ids(["first"]) == (observed, ())
    assert model.reveal_ids([]) == (observed, ())


def test_invalid_objects_overlap_and_sensor_inputs_are_rejected():
    base = make_map()
    first = HiddenObject("first", [[5, 3, 6, 4, 100]])
    for objects in [[first, first],
                    [first, HiddenObject("overlap", [[5, 3, 7, 5, 100]])],
                    [HiddenObject("outside", [[16, 3, 18, 4, 100]])],
                    [HiddenObject("free", [[5, 3, 6, 4, 0]])]]:
        with pytest.raises(ValueError):
            ObjectRevealer(base, objects, max_range=10)
    with pytest.raises(ValueError):
        ObjectRevealer(base, [], max_range=0)
    with pytest.raises(ValueError):
        ObjectRevealer(base, [], max_range=10, fov=7)
    model = ObjectRevealer(base, [first], max_range=10)
    for pose in [[1, 2], [1, 2, np.nan], [-0.01, 2, 0]]:
        with pytest.raises(ValueError):
            model.scan(pose)
    with pytest.raises(ValueError):
        model.reveal_ids("first")
