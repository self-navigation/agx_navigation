"""Hull clearance to the baked map's walls. ROS-free.

Computes the signed distance from the robot's footprint rectangle, sampled
along its perimeter, to the nearest wall pixel of the baked occupancy PNG (254
free / 0 wall / 205 unknown; unknown counts as free). It is positive in free
space and negative once a perimeter point sits on a wall pixel; walls are often
one cell thick, so negative values do not rank depth. The geometry matches
``tools/compare_run.py::wall_contact``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

# Scout Mini footprint half-extents [m] and perimeter sample spacing, as in
# compare_run (FP_HALF_LENGTH, FP_HALF_WIDTH, FP_SPACING).
HALF_LENGTH, HALF_WIDTH, SPACING = 0.315, 0.2925, 0.02


def _perimeter(half_length: float, half_width: float, spacing: float) -> np.ndarray:
    nl = int(2 * half_length / spacing) + 1
    nw = int(2 * half_width / spacing) + 1
    lx = np.linspace(-half_length, half_length, nl)
    wy = np.linspace(-half_width, half_width, nw)
    return np.vstack([np.c_[lx, np.full(nl, half_width)], np.c_[lx, np.full(nl, -half_width)],
                      np.c_[np.full(nw, half_length), wy], np.c_[np.full(nw, -half_length), wy]])


@dataclass
class SignedWallDistance:
    """Signed distance field [m] over the map grid; row 0 is the origin's y."""

    dist: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    body: np.ndarray = None  # (M, 2) perimeter points in the body frame

    def __post_init__(self):
        if self.body is None:
            self.body = _perimeter(HALF_LENGTH, HALF_WIDTH, SPACING)

    @classmethod
    def from_wall_mask(cls, wall: np.ndarray, resolution: float, origin_x: float,
                       origin_y: float) -> "SignedWallDistance":
        """``wall`` is a bool grid with row 0 at ``origin_y``."""
        from scipy.ndimage import distance_transform_edt
        wall = np.asarray(wall, dtype=bool)
        d = (distance_transform_edt(~wall) - distance_transform_edt(wall)) * resolution
        return cls(d, float(resolution), float(origin_x), float(origin_y))

    @classmethod
    def from_map_yaml(cls, yaml_path: str) -> "SignedWallDistance":
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(yaml_path))
        img_path = os.path.join(os.path.dirname(yaml_path), meta["image"])
        img = np.asarray(Image.open(img_path).convert("L"))[::-1]  # row 0 = origin y
        return cls.from_wall_mask(img == 0, meta["resolution"], meta["origin"][0],
                                  meta["origin"][1])

    @classmethod
    def from_occupancy(cls, data, width: int, height: int, resolution: float,
                       origin_x: float, origin_y: float) -> "SignedWallDistance":
        """From an OccupancyGrid's fields (row-major, row 0 at origin y).

        Occupied (>= 50) counts as wall, unknown (-1) as free, matching the
        baked-PNG convention above. Assumes an unrotated map origin.
        """
        grid = np.asarray(data, dtype=np.int16).reshape(height, width)
        return cls.from_wall_mask(grid >= 50, resolution, origin_x, origin_y)

    def point_clearance(self, x: float, y: float) -> float:
        col = int(np.clip((x - self.origin_x) / self.resolution, 0, self.dist.shape[1] - 1))
        row = int(np.clip((y - self.origin_y) / self.resolution, 0, self.dist.shape[0] - 1))
        return float(self.dist[row, col])

    def body_clearance(self, x: float, y: float, yaw: float) -> float:
        """Minimum signed clearance over the hull perimeter at pose (x, y, yaw)."""
        c, s = np.cos(yaw), np.sin(yaw)
        qx = x + self.body[:, 0] * c - self.body[:, 1] * s
        qy = y + self.body[:, 0] * s + self.body[:, 1] * c
        col = np.clip(((qx - self.origin_x) / self.resolution).astype(int), 0, self.dist.shape[1] - 1)
        row = np.clip(((qy - self.origin_y) / self.resolution).astype(int), 0, self.dist.shape[0] - 1)
        return float(self.dist[row, col].min())


def rejoin_clear(samples, swd: "SignedWallDistance", c_start: float,
                 stop_dist: float, margin: float = 0.01) -> bool:
    """Whether a re-join may be spliced in, by hull clearance along it.

    The re-join's effort cost knows nothing about walls (figures/2026-10-07/
    s5_rejoin), so the supervisor gates it here. Every sample must keep at
    least ``min(stop_dist, c_start) - margin``: a re-join may not take the
    body closer to a wall than the stop distance, or than it already is when
    it starts inside that distance (a robot stopped at a wall must be allowed
    to back away). The last sample must be clear of the stop distance.
    """
    floor = min(stop_dist, c_start) - margin
    cs = [swd.body_clearance(*s.pose) for s in samples]
    return bool(cs) and min(cs) >= floor and cs[-1] > stop_dist
