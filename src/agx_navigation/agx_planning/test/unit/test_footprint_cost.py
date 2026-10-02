"""Tests for the PMP footprint barrier (cfg.w_fp, #34).

The barrier's analytic gradient enters the costate ODEs by hand, so a sign or
chain-rule slip would not crash anything: it would push the plan INTO walls.
These check the gradient against finite differences of the cost itself, and
that _ode wires it in as -dL/dx with the term fully off at w_fp = 0.

Pure: a synthetic travel-time field and wall distance, no map, no ROS.
"""

import sys

import numpy as np
import pytest

# conftest.py mocks rclpy; hide the mock for this import only (see test_rejoin).
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner.shooting_solver import footprint_cost
    from agx_planning.vector_field import VectorFieldGrid
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock

RES = 0.05


def _doorway_grid(width=0.7, with_wall=True, extra=0.0, sigma=0.05):
    """A 0.2 m wall (+2*extra) along x = 5 with a doorway of `width` centred on y = 0.

    T is the distance to x = 10 (F points +x, through the door), and
    wall_dist is the true signed distance to the wall cells.
    """
    from scipy.ndimage import distance_transform_edt

    xs = np.arange(0.0, 10.0, RES)
    ys = np.arange(-3.0, 3.0, RES)
    X, Y = np.meshgrid(xs, ys)
    wall = (np.abs(X - 5.0) < 0.1 + extra) & (np.abs(Y) > width / 2)
    T = np.abs(10.0 - X) / 0.5
    signed = (distance_transform_edt(~wall) - distance_transform_edt(wall)) * RES
    grid = VectorFieldGrid()
    grid.update(T, 0.0, -3.0, RES, wall_dist=signed if with_wall else None,
                wall_sigma=sigma)
    return grid


def _cfg(**kw):
    return PlannerConfig(mode="offline", w_fp=kw.pop("w_fp", 20.0), **kw)


# Poses where some body points are inside the margin: entering the door
# skewed, square in the door, and a corner clipping a jamb.
POSES = [(4.5, 0.05, 0.4), (5.0, 0.0, 0.0), (4.7, 0.25, -0.6), (5.2, -0.1, 0.9)]


@pytest.mark.parametrize("pose", POSES)
def test_gradient_matches_finite_difference(pose):
    # Thick wall, wider blur: the gradient is bilinear(np.gradient(d)), which
    # matches d/dx of bilinear(d) only where d is smooth on the cell scale.
    # A thin wall's EDT ridge is a kink no interpolation agrees on.
    grid, cfg = _doorway_grid(extra=0.3, sigma=0.15), _cfg(fp_margin=0.1)
    p = np.array(pose, dtype=float)
    L, gx, gy, gth = footprint_cost(grid, p[[0]], p[[1]], p[[2]], cfg)
    assert L[0] > 0.0, "pose should engage the barrier"
    h = 1e-5
    fd = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = h
        up = footprint_cost(grid, *[np.array([v]) for v in p + e], cfg)[0][0]
        dn = footprint_cost(grid, *[np.array([v]) for v in p - e], cfg)[0][0]
        fd.append((up - dn) / (2 * h))
    # The analytic gradient interpolates np.gradient(d) (C^0, kinder to
    # solve_bvp, and how the beta*T term does it too) rather than
    # differentiating the interpolant, so the two differ by a few-to-20%
    # where d curves on the cell scale. A sign or chain-rule slip flips or
    # rotates the vector; that is what this catches.
    an, fd = np.array([gx[0], gy[0], gth[0]]), np.array(fd)
    cos = an @ fd / (np.linalg.norm(an) * np.linalg.norm(fd))
    assert cos > 0.98, (an, fd)
    assert np.linalg.norm(an) == pytest.approx(np.linalg.norm(fd), rel=0.15)


def test_square_in_door_is_cheaper_than_skewed():
    grid, cfg = _doorway_grid(), _cfg()
    one = lambda th: footprint_cost(grid, np.array([5.0]), np.array([0.0]),  # noqa: E731
                                    np.array([th]), cfg)[0][0]
    assert one(0.0) < one(0.3) < one(0.6)
    # and the rotational gradient points back toward square
    _, _, _, gth = footprint_cost(grid, np.array([5.0]), np.array([0.0]),
                                  np.array([0.3]), cfg)
    assert gth[0] > 0.0


def _ode_rows(grid, cfg, y):
    solver = PMPShootingSolver(cfg, grid)
    solver._goal = np.array([10.0, 0.0, 0.0])
    return solver._ode(0.0, y)


def _mesh():
    rng = np.random.default_rng(0)
    m = 7
    y = np.zeros((10, m))
    y[0] = rng.uniform(4.4, 5.4, m)
    y[1] = rng.uniform(-0.3, 0.3, m)
    y[2] = rng.uniform(-0.8, 0.8, m)
    y[3:5] = rng.uniform(0.0, 2.0, (2, m))
    y[5:] = rng.normal(0.0, 1.0, (5, m))
    return y


def test_ode_adds_minus_gradient_to_pose_costates():
    grid, y = _doorway_grid(), _mesh()
    on = _ode_rows(grid, _cfg(w_fp=20.0), y)
    off = _ode_rows(grid, _cfg(w_fp=0.0), y)
    _, gx, gy, gth = footprint_cost(grid, y[0], y[1], y[2], _cfg(w_fp=20.0))
    np.testing.assert_allclose(on[5] - off[5], -gx, atol=1e-12)
    np.testing.assert_allclose(on[6] - off[6], -gy, atol=1e-12)
    np.testing.assert_allclose(on[7] - off[7], -gth, atol=1e-12)
    # states and the wheel costates' direct terms are untouched
    np.testing.assert_array_equal(on[:5], off[:5])


def test_off_without_wall_channel():
    """A field without wall_dist (old publisher) silently disables the term."""
    y = _mesh()
    on = _ode_rows(_doorway_grid(with_wall=False), _cfg(w_fp=20.0), y)
    off = _ode_rows(_doorway_grid(with_wall=False), _cfg(w_fp=0.0), y)
    np.testing.assert_array_equal(on, off)


def test_pack_parse_roundtrip_carries_wall_dist():
    from agx_planning.pmp_planner.rollout import parse_field_array
    from agx_planning.vector_field.field import VectorFieldResult, pack_field_array

    h, w = 20, 30
    wd = np.linspace(-1, 1, h * w).reshape(h, w)
    res = VectorFieldResult(travel_time=np.ones((h, w)), grad_x=np.zeros((h, w)),
                            grad_y=np.zeros((h, w)), grad_mag=np.ones((h, w)),
                            free_max_T=1.0, origin_x=0.0, origin_y=0.0,
                            resolution=RES, wall_dist=wd)
    grid = parse_field_array(pack_field_array(res), PlannerConfig(fp_smooth_sigma=0.0))
    assert grid.has_wall_dist
    d, _, _ = grid.query_dist(np.array([10 * RES]), np.array([5 * RES]))
    assert d[0] == pytest.approx(wd[5, 10], abs=1e-6)
    res.wall_dist = None
    assert not parse_field_array(pack_field_array(res), PlannerConfig()).has_wall_dist
