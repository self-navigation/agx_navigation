"""Doorway-preserving centring of the FM2 speed profile."""

import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

# conftest.py mocks rclpy; hide the mock for this import only (see test_rejoin).
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field.field import (
        VectorFieldConfig,
        build_speed_field,
        local_half_width,
        normalized_clearance,
    )
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock

RES = 0.05


def _corridor(width_cells, length=60, wall=3):
    """Horizontal corridor; rows [wall, wall + width) free."""
    obs = np.ones((width_cells + 2 * wall, length), dtype=bool)
    obs[wall:wall + width_cells, :] = False
    return obs


def _edt(obs):
    return distance_transform_edt(~obs) * RES


def test_half_width_is_constant_across_a_corridor():
    obs = _corridor(20)  # 1.0 m wide
    edt = _edt(obs)
    hw = local_half_width(edt, RES, cap=0.8)
    col = hw[3:23, 30]
    # Largest inscribed disk: radius = max EDT across the section.
    assert np.allclose(col, edt[3:23, 30].max())


def test_half_width_capped_in_a_wide_room():
    obs = _corridor(80)  # 4 m wide
    hw = local_half_width(_edt(obs), RES, cap=0.8)
    assert np.allclose(hw[3:83, 30], 0.8)


def test_half_width_does_not_leak_across_a_thin_wall():
    # A narrow free sliver (a thin wall's "unknown" interior) next to a corridor.
    obs = np.ones((40, 60), dtype=bool)
    obs[3:23, :] = False      # 1 m corridor
    obs[24:26, :] = False     # 2-cell sliver behind a 1-cell wall at row 23
    hw = local_half_width(_edt(obs), RES, cap=0.8)
    assert hw[22, 30] > 0.4   # corridor cell next to the wall keeps its width


def test_normalized_clearance_doorway_centre_equals_corridor_centre():
    narrow, wide = _corridor(15), _corridor(30)
    n_values = []
    for obs, w in ((narrow, 15), (wide, 30)):
        edt = _edt(obs)
        n = normalized_clearance(edt, local_half_width(edt, RES, 0.8), 0.8)
        n_values.append(n[3 + w // 2, 30])
        assert n[3, 30] < 0.25  # next to the wall: low
    assert np.allclose(n_values, 1.0)


def test_normalized_clearance_zero_in_walls_and_bounded():
    obs = _corridor(20)
    edt = _edt(obs)
    n = normalized_clearance(edt, local_half_width(edt, RES, 0.8), 0.8)
    assert np.all(n[obs] == 0.0)
    assert n.min() >= 0.0 and n.max() <= 1.0


def test_speed_field_default_unchanged_and_centring_keeps_door_open():
    obs = _corridor(16)  # 0.8 m doorway-like passage
    edt = _edt(obs)
    off = build_speed_field(edt, obs, VectorFieldConfig(speed_profile="exponential"), RES)
    legacy = build_speed_field(edt, obs, VectorFieldConfig(speed_profile="exponential"))
    assert np.array_equal(off, legacy)
    on = build_speed_field(edt, obs, VectorFieldConfig(speed_profile="exponential",
                                                       centring=True), RES)
    # Inflation alone slows the whole 0.8 m passage (EDT <= 0.4 < R = 0.5);
    # centring runs its centreline at full speed.
    assert off[3 + 8, 30] < 0.9
    assert np.isclose(on[3 + 8, 30], 1.0)
