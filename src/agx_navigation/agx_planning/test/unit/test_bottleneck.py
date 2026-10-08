"""Bottleneck detection and the stop-and-aim direction field (vector_field/bottleneck.py)."""
import sys

import numpy as np

# conftest.py's rclpy MagicMock has no __spec__ (see test_rejoin.py); hide it
# for this import only.
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field.bottleneck import find_bottlenecks, shape_direction
    from agx_planning.vector_field.field import (VectorFieldConfig, compute_field,
                                                 field_result_to_grid)
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock

RES = 0.05


def two_rooms(door_w=0.9, wall_x=4.0):
    """8 x 6 m box split by a wall at x = wall_x with one door centred at y = 3."""
    h, w = int(6 / RES), int(8 / RES)
    occ = np.zeros((h, w), dtype=np.int8)
    occ[:2, :] = occ[-2:, :] = 100
    occ[:, :2] = occ[:, -2:] = 100
    c = int(wall_x / RES)
    occ[:, c - 2:c + 2] = 100
    half = int(round(door_w / 2 / RES))
    mid = int(3.0 / RES)
    occ[mid - half:mid + half, c - 2:c + 2] = 0
    return occ


def test_finds_the_door_and_its_axis():
    occ = two_rooms()
    b = find_bottlenecks(occ >= 65, RES, 0.0, 0.0)
    assert len(b) == 1
    b = b[0]
    assert abs(b.x - 4.0) < 0.15 and abs(b.y - 3.0) < 0.1
    assert abs(abs(b.ax) - 1.0) < 1e-6 and abs(b.ay) < 1e-6   # passage runs along x
    assert 0.8 <= b.width <= 0.95


def test_wide_opening_is_not_a_bottleneck():
    occ = two_rooms(door_w=2.0)
    assert find_bottlenecks(occ >= 65, RES, 0.0, 0.0) == []


def test_shaped_field_aims_along_the_axis_and_onto_it():
    occ = two_rooms()
    res, _ = compute_field(occ, int(7.0 / RES), int(1.0 / RES), RES, 0.0, 0.0,
                           VectorFieldConfig(bottleneck_align=True, early_exit_enable=False))
    assert len(res.bottlenecks) == 1
    b, sgn = res.bottlenecks[0]
    assert sgn * b.ax > 0                     # oriented towards the goal room (+x)

    def d(x, y):
        r, c = int(y / RES), int(x / RES)
        return res.dir_x[r, c], res.dir_y[r, c]

    # On the axis, 0.5 m before the door: straight through.
    fx, fy = d(b.x - 0.5, b.y)
    assert fx > 0.99
    # 0.4 m above the axis, before the door: points down onto the axis, steeply.
    fx, fy = d(b.x - 0.5, b.y + 0.4)
    assert fy < -0.8
    # Far from the door the field is FM2's.
    r, c = int(1.0 / RES), int(1.0 / RES)
    assert np.isclose(res.dir_x[r, c], res.grad_x[r, c]) and np.isclose(res.dir_y[r, c], res.grad_y[r, c])


def test_zone_does_not_leak_through_walls():
    """Cells inside the zone box but behind the wall from the opening are untouched."""
    occ = two_rooms()
    obst = occ >= 65
    b = find_bottlenecks(obst, RES, 0.0, 0.0)
    h, w = occ.shape
    gx, gy = np.zeros((h, w)), np.ones((h, w))   # base field: +y everywhere ...
    r0, c0 = int(b[0].y / RES), int(b[0].x / RES)
    gx[r0, c0], gy[r0, c0] = 1.0, 0.0             # ... except +x through the door
    fx, fy, oriented = shape_direction(gx, gy, obst, b, RES, 0.0, 0.0, lateral=1.5)
    assert len(oriented) == 1
    changed = (fx != gx) | (fy != gy)
    # Free cells hugging the wall's far face, beyond the jambs: in the box, out of sight.
    c = int(4.15 / RES)
    hidden = [(r, c) for r in range(int(0.3 / RES), int(1.6 / RES)) if not obst[r, c]]
    assert hidden and not any(changed[r, c] for r, c in hidden)
    # ... while the axis past the door is shaped.
    assert changed[r0, int((b[0].x + 0.4) / RES)] or fx[r0, int((b[0].x + 0.4) / RES)] > 0.99


def test_grid_direction_channel_keeps_the_goal_fade():
    occ = two_rooms()
    res, _ = compute_field(occ, int(7.0 / RES), int(1.0 / RES), RES, 0.0, 0.0,
                           VectorFieldConfig(bottleneck_align=True, early_exit_enable=False))
    g = field_result_to_grid(res)
    b, _ = res.bottlenecks[0]
    _, _, _, fx, fy = g.query_scalar(b.x - 0.5, b.y + 0.4)
    assert fy < -0.8                               # F_unit reads the shaped direction
    # Same magnitude as FM2's own F_unit everywhere: the eps fade is kept.
    res.dir_x = res.dir_y = None
    g0 = field_result_to_grid(res)
    xs, ys = np.meshgrid(np.linspace(0.3, 7.7, 40), np.linspace(0.3, 5.7, 30))
    f1 = g.query_vec(xs.ravel(), ys.ravel())
    f0 = g0.query_vec(xs.ravel(), ys.ravel())
    assert np.allclose(np.hypot(f1[3], f1[4]), np.hypot(f0[3], f0[4]), atol=1e-9)
    assert np.allclose(f1[1], f0[1]) and np.allclose(f1[2], f0[2])   # grad T untouched


def test_off_by_default():
    occ = two_rooms()
    res, _ = compute_field(occ, int(7.0 / RES), int(1.0 / RES), RES, 0.0, 0.0,
                           VectorFieldConfig(early_exit_enable=False))
    assert res.dir_x is None and res.bottlenecks is None


def test_goal_inside_a_zone_keeps_the_fm2_direction():
    """A goal just past the door: near it the field must point at the goal, not along the axis."""
    occ = two_rooms()
    gx_, gy_ = 4.5, 2.6                      # 0.5 m past the door, 0.4 m off its axis
    res, _ = compute_field(occ, int(gx_ / RES), int(gy_ / RES), RES, 0.0, 0.0,
                           VectorFieldConfig(bottleneck_align=True, early_exit_enable=False))
    r, c = int(2.75 / RES), int(4.45 / RES)  # 0.16 m from the goal
    assert np.isclose(res.dir_x[r, c], res.grad_x[r, c]) and np.isclose(res.dir_y[r, c], res.grad_y[r, c])
