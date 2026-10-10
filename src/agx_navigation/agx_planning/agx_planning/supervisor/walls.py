"""Hull clearance to the baked map's walls. ROS-free.

Computes the signed distance from the robot's footprint rectangle, sampled
along its perimeter, to the nearest wall pixel of the baked occupancy PNG (254
free / 0 wall / 205 unknown; unknown counts as free). It is positive in free
space and negative once a perimeter point sits on a wall pixel; walls are often
one cell thick, so negative values do not rank depth. The geometry matches
``tools/compare_run.py::wall_contact``. ``hull_penetration`` and
``crosses_wall`` measure how deep a wall pixel lies inside the filled hull.
"""

from __future__ import annotations

import math
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

    def hull_penetration(self, x: float, y: float, yaw: float) -> float:
        """How deep [m] the deepest wall pixel sits inside the body at a pose.

        0 when no wall pixel overlaps the hull's FILLED rectangle (the
        perimeter clearance above cannot tell a wall on the hull's edge from
        one passing through its middle: both read -1 cell). A pixel counts
        when its centre lies inside the rectangle grown by half a cell; its
        depth is the distance from that centre to the nearest side, plus half
        a cell, so a pixel just grazing an edge has a depth near 0+ and a wall
        under the body's middle has a depth near the half-width.
        """
        hl = float(self.body[:, 0].max())
        hw = float(self.body[:, 1].max())
        res = self.resolution
        reach = math.hypot(hl, hw) + res
        h, w = self.dist.shape
        c0 = max(int(math.floor((x - reach - self.origin_x) / res)), 0)
        c1 = min(int(math.floor((x + reach - self.origin_x) / res)) + 1, w)
        r0 = max(int(math.floor((y - reach - self.origin_y) / res)), 0)
        r1 = min(int(math.floor((y + reach - self.origin_y) / res)) + 1, h)
        if c0 >= c1 or r0 >= r1:
            return 0.0
        rr, cc = np.nonzero(self.dist[r0:r1, c0:c1] < 0.0)
        if not len(rr):
            return 0.0
        dx = self.origin_x + (cc + c0 + 0.5) * res - x
        dy = self.origin_y + (rr + r0 + 0.5) * res - y
        c, s = math.cos(yaw), math.sin(yaw)
        bx = c * dx + s * dy
        by = -s * dx + c * dy
        depth = np.minimum(hl - np.abs(bx), hw - np.abs(by)) + 0.5 * res
        return float(max(depth.max(), 0.0))

    def interpolate(self, poses) -> np.ndarray:
        """Insert poses between consecutive ``poses`` (N, 3) so that no hull
        point moves more than half a map cell from one to the next: a one-cell
        wall cannot then fall between two checked poses ("Crossing" in
        rejoin_clear). Yaw is interpolated along the short way round."""
        p = np.asarray(poses, dtype=float).reshape(-1, 3)
        if len(p) < 2:
            return p
        r_max = float(np.hypot(self.body[:, 0], self.body[:, 1]).max())
        step = 0.5 * self.resolution
        out = [p[:1]]
        for a, b in zip(p[:-1], p[1:]):
            dyaw = math.atan2(math.sin(b[2] - a[2]), math.cos(b[2] - a[2]))
            move = math.hypot(b[0] - a[0], b[1] - a[1]) + abs(dyaw) * r_max
            n = max(1, int(math.ceil(move / step)))
            f = np.arange(1, n + 1)[:, None] / n
            seg = np.hstack([a[:2] + f * (b[:2] - a[:2]), a[2] + f * dyaw])
            out.append(seg)
        return np.vstack(out)


def crosses_wall(poses, swd: "SignedWallDistance", start_touching: bool) -> bool:
    """Whether a path's body overlaps a wall it should not ("Crossing" in
    rejoin_clear). ``poses`` is (N, 3); the path is interpolated first.

    From a clear start any overlap is a crossing. From a touching start
    (``start_touching``) the leading run of overlapping poses is allowed while
    the wall gets no deeper into the body than at the first pose (plus half a
    cell of discretization slack): that is backing or turning away. A wall
    going deeper means the body is driving into or through it. Once a pose is
    clear, any later overlap is a crossing.
    """
    fine = swd.interpolate(poses)
    if not len(fine):
        return False
    pen = np.array([swd.hull_penetration(*p) for p in fine])
    over = pen > 0.0
    if not start_touching:
        return bool(over.any())
    clear = np.flatnonzero(~over)
    first_clear = int(clear[0]) if len(clear) else len(pen)
    if over[first_clear:].any():
        return True
    return bool(np.any(pen[:first_clear] > pen[0] + 0.5 * swd.resolution))


def rejoin_clear(samples, swd: "SignedWallDistance", c_start: float,
                 stop_dist, margin: float = 0.01) -> bool:
    """Whether a re-join may be spliced in, by hull clearance along it.

    The re-join's effort cost knows nothing about walls (figures/2026-10-07/
    s5_rejoin), so the supervisor gates it here. ``stop_dist`` is a scalar or
    one stop threshold per sample (the core's plan-relative ``stop_at`` at the
    plan index nearest each sample). Every sample must keep at least
    ``min(stop_i, c_start) - margin``: a re-join may not take the body inside
    the stop threshold, or closer than it already is when it starts inside it
    (a robot stopped at a wall must be allowed to back away). The last sample
    must be clear of its stop threshold.

    Crossing (2026-10-10). The relative rule alone is vacuous for a robot
    already on a wall: walls are one cell thick and unknown counts as free, so
    the clearance never reads below -1 cell, and with ``c_start`` = -1 cell
    every candidate passed (job 244: 39 spliced vs 0 blocked from a wall
    contact; one re-join drove through a wall slab,
    figures/2026-10-10/rejoin_runs). So a re-join is also rejected when the
    body crosses a wall (``crosses_wall``): the path is interpolated to half a
    cell of hull motion per step, so a wall cannot be stepped over between
    samples; from a clear start (``c_start`` >= 0) any overlap of the filled
    hull with a wall pixel rejects; from a touching start the robot may stay
    on the wall while it backs away (the wall getting no deeper into the
    body), and once it is clear any later overlap rejects.
    """
    n = len(samples)
    if not n:
        return False
    stops = np.broadcast_to(np.asarray(stop_dist, dtype=float), (n,))
    cs = np.array([swd.body_clearance(*s.pose) for s in samples])
    if not (np.all(cs >= np.minimum(stops, c_start) - margin) and cs[-1] > stops[-1]):
        return False
    return not crosses_wall([s.pose for s in samples], swd, start_touching=c_start < 0.0)
