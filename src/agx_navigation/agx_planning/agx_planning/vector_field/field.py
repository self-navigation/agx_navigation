"""Pure-Python FM2 (Fast Marching Square) vector field computation.

No ROS 2 imports -- usable from scripts, notebooks, and tests without
pulling in rclpy.

Public API
----------
  VectorFieldConfig   -- full generation parameters
  world_to_grid       -- world (x, y) -> (col, row), returns None if OOB
  grid_to_world       -- (col, row) -> world (x, y) cell-centre
  compute_field       -- full FM2 pipeline; returns VectorFieldResult or None
  pack_field_array    -- serialise VectorFieldResult to a flat float32 array
                         (same layout as /vector_field/planner_data)
  field_result_to_grid -- build a VectorFieldGrid directly from a result,
                          bypassing the Float32MultiArray round-trip; use
                          this when the generator and planner share a process
  local_half_width    -- passage half-width per cell (largest wall-free disk)
  normalized_clearance -- EDT / min(half-width, cap); drives cfg.centring

Internal helpers (not part of the public API):
  build_speed_field, solve_eikonal, field_from_T, _gaussian_with_nan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, gaussian_filter

from agx_planning.maps.grid import GridGeometry
from agx_planning.vector_field import VectorFieldGrid
from agx_planning.vector_field.preparation import prepare_travel_time


@dataclass
class VectorFieldConfig:
    # Clearance band: cells with EDT < inflation_radius see reduced speed.
    inflation_radius: float = 0.5  # [m]
    # Speed at the wall surface. Strictly > 0; eikonal diverges as v -> 0.
    speed_v_min: float = 0.1
    # Speed in open space (>= R from any obstacle).
    speed_v_max: float = 1.0
    # "linear" -> sharp band; "exponential" -> long-tailed, pulls paths to
    # corridor centrelines even in wide spaces.
    speed_profile: str = "linear"
    # Decay rate for the exponential profile only; ignored otherwise.
    speed_decay_rate: float = 2.5

    # Doorway-preserving centring (#48). Default OFF, so existing fields and
    # plans are unchanged. When on, the profile above is driven by the
    # NORMALISED clearance n = EDT / min(half_width, centring_cap) instead
    # of EDT / inflation_radius, where half_width is the radius of the
    # largest wall-free disk containing the cell (see `local_half_width`).
    # n = 1 on the centreline of ANY passage, so a doorway centre is as fast
    # as a corridor centre and no doorway closes, while every passage still
    # pulls the path to its middle. inflation_radius is unused when on.
    # Measured 2026-10-07 on the broad 40 (figures/2026-10-07/p48_clearance):
    # with speed_profile="exponential" use speed_decay_rate=1.5 -- same solve
    # rate as the stack (34/40), hull time under 0.15 m 4.8% -> 0.4%, no
    # overlaps. The default 2.5 centres harder but loses 7 solves.
    # centring_cap [m]: spaces wider than 2 * cap are treated as 2 * cap wide,
    #    so open rooms push the path ~cap off a wall instead of to the room's
    #    middle. Plays the role inflation_radius plays without centring.
    # centring_smooth [m]: NaN-aware Gaussian blur of n over free cells. The
    #    half-width is piecewise constant (it jumps where a corridor opens
    #    into a room), and a step in v puts a kink in grad T that the BVP
    #    meshes to exhaustion. 0 disables. Measured: smoothing LOSES (it
    #    lifts n next to walls, and did not recover solves); leave at 0.
    centring: bool = False
    centring_cap: float = 0.8  # [m]
    centring_smooth: float = 0.0  # [m]

    # Bottleneck alignment, i.e. stop-and-aim at doorways (#50, bottleneck.py).
    # Default OFF. When on, the result carries a shaped unit DIRECTION field
    # (dir_x, dir_y): near every narrow passage it is a line-following field on
    # the passage axis, so the planner stops, turns in place and drives straight
    # through instead of skidding through a turn taken on the move. T, and hence
    # the position pull, is unchanged.
    bottleneck_align: bool = False
    bottleneck_max_width: float = 1.2      # [m] openings narrower than this
    bottleneck_approach: float = 1.0       # [m] zone length before the opening
    bottleneck_exit: float = 0.6           # [m] zone length past it
    bottleneck_lateral: float = 0.8        # [m] zone half-width
    bottleneck_fade: float = 0.3           # [m] blend into the FM2 direction
    bottleneck_converge_deg: float = 80.0  # max approach angle onto the axis
    bottleneck_converge_dist: float = 0.15  # [m] lateral scale of that approach

    # Smooth T before differentiating. This regularises FMM ridges, where
    # the central-difference gradient otherwise averages two opposing sides
    # and produces an unstable unit direction. Default OFF: with the
    # confidence-weighted planner cost, the planner can de-weight cut-locus
    # cells without needing the smoothing pass. Re-enable if you observe
    # zig-zagging across corridor centrelines and confidence weighting is
    # disabled in the planner.
    smooth_T_before_grad: bool = False
    smooth_T_sigma: float = 0.10  # [m]

    early_exit_enable: bool = True
    # Travel-time buffer: ceil(margin / resolution) * resolution.
    # This does not guarantee a physical radius around the robot.
    early_exit_margin: float = 0.5


@dataclass
class VectorFieldResult:
    """Output of one FM2 solve.

    Arrays are (H, W) float64; all share the same grid layout as the
    source OccupancyGrid. Metadata fields carry the origin and resolution
    needed to convert between world and grid coordinates.

    travel_time  -- T(x): eikonal solution; NaN on unreachable / obstacle cells.
    grad_x       -- unit vector field x-component (-grad T direction).
    grad_y       -- unit vector field y-component.
    grad_mag     -- speed-corrected confidence = |grad T| * v(x).
                    ~1 in smooth free space, ~0 at cut loci and goal sink.
    free_max_T   -- max finite T over reachable free cells; used to scale
                    the cost-to-go colour map and the packed array sentinel.
    origin_x     -- world x of the cell-corner of cell (col=0, row=0) [m].
    origin_y     -- world y of the cell-corner of cell (col=0, row=0) [m].
    resolution   -- cell size [m/cell].
    wall_dist    -- SIGNED distance to the nearest obstacle cell [m]:
                    positive in free space, negative inside walls, so
                    the planner's footprint cost has an outward gradient
                    even for a body point already in a wall. Optional so
                    hand-built results (tests) need not supply it.
    dir_x, dir_y -- shaped unit direction (cfg.bottleneck_align), or None.
                    grad_x/grad_y stay FM2's own -grad T direction.
    bottlenecks  -- the (Bottleneck, sign) pairs the shaping used.
    """

    travel_time: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    grad_mag: np.ndarray
    free_max_T: float
    origin_x: float
    origin_y: float
    resolution: float
    wall_dist: Optional[np.ndarray] = None
    dir_x: Optional[np.ndarray] = None
    dir_y: Optional[np.ndarray] = None
    bottlenecks: Optional[list] = None

    def query_vec(self, wx: float, wy: float) -> Optional[np.ndarray]:
        """Bilinear interpolation of [vx, vy, T] at world position (wx, wy).

        Returns a shape-(3,) array so callers can slice by component index,
        e.g. result[0], result[:2], etc.
        Samples are at cell centres. The outer half-cell band holds the
        nearest centre value. Returns None outside the physical map bounds.
        """
        gx = (wx - self.origin_x) / self.resolution
        gy = (wy - self.origin_y) / self.resolution

        h, w = self.grad_x.shape
        if not (h >= 2 and w >= 2
                and self.origin_x <= wx < self.origin_x + w * self.resolution
                and self.origin_y <= wy < self.origin_y + h * self.resolution
                and 0 <= gx < w and 0 <= gy < h):
            return None

        u, v = np.clip(gx - 0.5, 0.0, w - 1), np.clip(gy - 0.5, 0.0, h - 1)
        x0, y0 = min(int(u), w - 2), min(int(v), h - 2)
        fx, fy = u - x0, v - y0

        # Bilinear weights for the four surrounding corners
        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy

        def _bilerp(arr: np.ndarray) -> float:
            return float(
                arr[y0, x0] * w00
                + arr[y0, x0 + 1] * w10
                + arr[y0 + 1, x0] * w01
                + arr[y0 + 1, x0 + 1] * w11
            )

        return np.array(
            [
                _bilerp(self.grad_x),
                _bilerp(self.grad_y),
                _bilerp(self.travel_time),
            ]
        )

    def gradient_path(
        self,
        origin_x: float,
        origin_y: float,
        goal_x: float,
        goal_y: float,
        step: float,
        max_iter: int,
        stop_radius_cells: float,
    ) -> np.ndarray:
        """
        Trace a path through a vector field from origin toward goal.

        Returns a (N, 2) array of world-space waypoints, columns are [x, y].
        stop_radius_cells is multiplied by field.resolution for the stopping distance.
        """
        stop_radius_sq = (stop_radius_cells * self.resolution) ** 2
        stall_threshold_sq = (0.1 * step) ** 2

        wx, wy = origin_x, origin_y
        points: list[tuple[float, float]] = []
        prev = (wx, wy)
        stalled = 0

        for _ in range(max_iter):
            points.append((wx, wy))

            dx, dy = goal_x - wx, goal_y - wy
            if dx * dx + dy * dy < stop_radius_sq:
                break

            q = self.query_vec(wx, wy)
            if q is None:
                break
            vx, vy = q[0], q[1]
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                break

            wx += step * vx
            wy += step * vy

            ddx, ddy = wx - prev[0], wy - prev[1]
            if ddx * ddx + ddy * ddy < stall_threshold_sq:
                stalled += 1
                if stalled >= 3:
                    break
            else:
                stalled = 0
            prev = (wx, wy)

        return np.array(points, dtype=np.float64)


def world_to_grid(
    wx: float,
    wy: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Convert world (wx, wy) to (col, row). Returns None if out of bounds."""
    try:
        geometry = GridGeometry((height, width), resolution, (origin_x, origin_y, 0.0))
        return geometry.world_to_cell(wx, wy)
    except ValueError:
        return None


def grid_to_world(
    col: int,
    row: int,
    origin_x: float,
    origin_y: float,
    resolution: float,
) -> Tuple[float, float]:
    """Convert (col, row) to world centre of that cell."""
    return (
        origin_x + (col + 0.5) * resolution,
        origin_y + (row + 0.5) * resolution,
    )


def local_half_width(
    edt_free: np.ndarray,
    resolution: float,
    cap: float,
) -> np.ndarray:
    """Local passage half-width [m], capped at ``cap`` (#48).

    The radius of the largest wall-free disk that contains the cell (the
    "local thickness"). In a corridor or a doorway this is half its width at
    every cell across it; next to a wall of a wide room it is >= cap. A disk
    is wall-free by construction, so the measure never leaks across a wall
    (a nearest-medial-axis lookup does, into the unknown interior of a thin
    baked wall). Computed level by level: at radius r, the disks of every
    centre with EDT >= r cover the cells within r of such a centre. Cells
    in walls get 0.
    """
    free = edt_free > 0.0
    hw = np.where(free, np.minimum(edt_free, cap), 0.0)
    n_levels = int(np.floor(cap / resolution + 1e-9))
    for k in range(1, n_levels + 1):
        r = k * resolution
        centres = edt_free >= r - 1e-9
        if not centres.any():
            break
        covered = free & (distance_transform_edt(~centres) * resolution <= r + 1e-9)
        hw[covered] = np.maximum(hw[covered], r)
    return hw


def _local_half_width_tiled(edt_free, resolution, cap):
    """Evaluate radius levels only where they can still raise half-width."""
    if (edt_free.ndim != 2 or not np.isfinite(resolution) or resolution <= 0
            or not np.isfinite(cap) or cap <= 0):
        return local_half_width(edt_free, resolution, cap)
    free = edt_free > 0.0
    hw = np.where(free, np.minimum(edt_free, cap), 0.0)
    n_levels = int(np.floor(cap / resolution + 1e-9))
    if not n_levels or not free.any():
        return hw
    # Keep the input dtype for the same scalar comparison as the full array.
    maximum = np.max(edt_free, where=free, initial=0, keepdims=True)
    low, high = 0, n_levels
    while low < high:
        middle = (low + high + 1) // 2
        if (maximum >= middle * resolution - 1e-9).item():
            low = middle
        else:
            high = middle - 1
    h, w = edt_free.shape
    tile_size = 128
    whole = ((0, h, 0, w), (0, h, 0, w))
    for k in range(low, 0, -1):
        r = k * resolution
        pending = free & (hw < r)
        if not pending.any():
            break
        centres = edt_free >= r - 1e-9
        if not centres.any():
            continue
        regions = []
        if edt_free.size > tile_size**2:
            halo = int(np.ceil((r + 1e-9) / resolution))
            for tr0 in range(0, h, tile_size):
                for tc0 in range(0, w, tile_size):
                    selected = pending[tr0:tr0 + tile_size, tc0:tc0 + tile_size]
                    if not selected.any():
                        continue
                    rows = np.flatnonzero(np.any(selected, axis=1))
                    cols = np.flatnonzero(np.any(selected, axis=0))
                    r0, r1 = tr0 + int(rows[0]), tr0 + int(rows[-1]) + 1
                    c0, c1 = tc0 + int(cols[0]), tc0 + int(cols[-1]) + 1
                    source = (max(0, r0 - halo), min(h, r1 + halo),
                              max(0, c0 - halo), min(w, c1 + halo))
                    regions.append(((r0, r1, c0, c1), source))
        source_cells = sum((r1 - r0) * (c1 - c0)
                           for _, (r0, r1, c0, c1) in regions)
        if not regions or source_cells >= edt_free.size:
            regions = [whole]
        for (r0, r1, c0, c1), (sr0, sr1, sc0, sc1) in regions:
            local_centres = centres[sr0:sr1, sc0:sc1]
            if not local_centres.any():
                continue
            distance = distance_transform_edt(~local_centres) * resolution
            covered = pending[r0:r1, c0:c1] & (
                distance[r0 - sr0:r1 - sr0, c0 - sc0:c1 - sc0] <= r + 1e-9)
            # The first covering radius is the largest possible level.
            hw[r0:r1, c0:c1][covered] = r
    return hw


def normalized_clearance(
    edt_free: np.ndarray,
    half_width: np.ndarray,
    cap: float,
) -> np.ndarray:
    """EDT / min(half_width, cap), clipped to [0, 1] (#48).

    1 on the centreline of any passage up to 2 * cap wide, and at >= cap
    from a wall in wider spaces; 0 at a wall.
    """
    denom = np.minimum(half_width, cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.where(denom > 0.0, edt_free / denom, 0.0)
    return np.clip(n, 0.0, 1.0)


def build_speed_field(
    edt_free: np.ndarray,
    obstacle_mask: np.ndarray,
    cfg: VectorFieldConfig,
    resolution: Optional[float] = None,
    *,
    tiled_half_width: bool = False,
) -> np.ndarray:
    """Map EDT distance to wave speed v(x) for the eikonal solve.

    With cfg.centring the profile's argument is the normalised clearance
    (needs ``resolution``); otherwise EDT / inflation_radius.
    """
    R = cfg.inflation_radius
    v = np.full_like(edt_free, cfg.speed_v_max, dtype=np.float64)
    if cfg.centring:
        if resolution is None:
            raise ValueError("centring needs the grid resolution")
        half_width = _local_half_width_tiled if tiled_half_width else local_half_width
        hw = half_width(edt_free, resolution, cfg.centring_cap)
        n_all = normalized_clearance(edt_free, hw, cfg.centring_cap)
        if cfg.centring_smooth > 0.0:
            n_all = _gaussian_with_nan(np.where(obstacle_mask, np.nan, n_all),
                                       cfg.centring_smooth / resolution)
            n_all = np.where(obstacle_mask, 0.0, np.clip(n_all, 0.0, 1.0))
        near = ~obstacle_mask & (n_all < 1.0)
        norm = n_all[near]
    else:
        if R <= 0.0:
            return v
        near = ~obstacle_mask & (edt_free < R)
        norm = edt_free[near] / R  # 0 at wall, 1 at boundary
    if not near.any():
        return v

    if cfg.speed_profile == "exponential":
        v[near] = np.clip(
            cfg.speed_v_max * np.exp(-cfg.speed_decay_rate * (1.0 - norm)),
            cfg.speed_v_min,
            cfg.speed_v_max,
        )
    else:
        v[near] = np.clip(
            cfg.speed_v_min + (cfg.speed_v_max - cfg.speed_v_min) * norm,
            cfg.speed_v_min,
            cfg.speed_v_max,
        )
    return v


_MAX_SPEED_REGIONS = 4


def _speed_update_regions(dirty, halo):
    h, w = dirty.shape

    def bounds(box):
        r0, r1, c0, c1 = box
        selected = dirty[r0:r1, c0:c1]
        rows = np.flatnonzero(np.any(selected, axis=1))
        cols = np.flatnonzero(np.any(selected, axis=0))
        return (r0 + int(rows[0]), r0 + int(rows[-1]) + 1,
                c0 + int(cols[0]), c0 + int(cols[-1]) + 1)

    def expand(box, padding):
        r0, r1, c0, c1 = box
        return (max(0, r0 - padding), min(h, r1 + padding),
                max(0, c0 - padding), min(w, c1 + padding))

    def source_area(box):
        r0, r1, c0, c1 = expand(box, 2 * halo)
        return (r1 - r0) * (c1 - c0)

    regions = [bounds((0, h, 0, w))]
    while len(regions) < _MAX_SPEED_REGIONS:
        best, best_saving = None, 0
        for index, box in enumerate(regions):
            r0, r1, c0, c1 = box
            selected = dirty[r0:r1, c0:c1]
            for axis in (0, 1):
                occupied = np.flatnonzero(np.any(selected, axis=1 - axis))
                gaps = np.diff(occupied) - 1
                if not gaps.size:
                    continue
                gap = int(np.argmax(gaps))
                # Keep output regions disjoint; source halos may overlap.
                if gaps[gap] < max(1, 2 * halo):
                    continue
                cut = box[2 * axis] + int(occupied[gap]) + 1
                before, after = list(box), list(box)
                before[2 * axis + 1], after[2 * axis] = cut, cut
                children = [bounds(before), bounds(after)]
                saving = source_area(box) - sum(source_area(child) for child in children)
                if saving > best_saving:
                    best, best_saving = (index, children), saving
        if best is None:
            break
        index, children = best
        regions[index:index + 1] = children
    return [(expand(box, halo), expand(box, 2 * halo)) for box in regions]


def _update_speed_field(edt_free, obstacles, cfg, resolution, previous):
    total = obstacles.size
    info = dict(status="full_recompute", reason="new_map", changed_cells=None,
                updated_cells=total, computed_cells=total, total_cells=total,
                region_count=1)

    def full(reason):
        info["reason"] = reason
        return build_speed_field(edt_free, obstacles, cfg, resolution,
                                 tiled_half_width=True), info

    if previous is None or previous.obstacle_mask.shape != obstacles.shape:
        return full("new_map")
    if not cfg.centring:
        return full("centring_disabled")
    if cfg.centring_smooth != 0.0:
        return full("smoothing")
    cap = cfg.centring_cap
    if not np.isfinite(resolution) or resolution <= 0 or not np.isfinite(cap) or cap <= 0:
        return full("unsupported_geometry")

    levels = int(np.floor(cap / resolution + 1e-9))
    radius = levels * resolution
    # Include the final level even when rounding places its threshold above cap.
    cutoff = max(cap, radius - 1e-9)
    old_edt = np.maximum(previous.wall_dist, 0.0)
    dirty = ((np.minimum(old_edt, cutoff) != np.minimum(edt_free, cutoff))
             | (previous.obstacle_mask != obstacles))
    info["changed_cells"] = int(np.count_nonzero(dirty))
    if not info["changed_cells"]:
        info.update(status="unchanged", reason="inputs_unchanged",
                    updated_cells=0, computed_cells=0, region_count=0)
        return previous.speed.copy(), info

    halo = int(np.ceil((radius + 1e-9) / resolution)) if levels else 0
    regions = _speed_update_regions(dirty, halo)
    computed_cells = sum((r1 - r0) * (c1 - c0)
                         for _, (r0, r1, c0, c1) in regions)
    if computed_cells >= total:
        return full("full_region")
    speed = previous.speed.copy()
    for (r0, r1, c0, c1), (sr0, sr1, sc0, sc1) in regions:
        source = np.s_[sr0:sr1, sc0:sc1]
        local_speed = build_speed_field(edt_free[source], obstacles[source], cfg, resolution,
                                        tiled_half_width=True)
        speed[r0:r1, c0:c1] = local_speed[r0 - sr0:r1 - sr0, c0 - sc0:c1 - sc0]
    info.update(status="local_update", reason="changed_region",
                updated_cells=sum((r1 - r0) * (c1 - c0)
                                  for (r0, r1, c0, c1), _ in regions),
                computed_cells=computed_cells, region_count=len(regions))
    return speed, info


def _update_interior_distance(obstacles, resolution, previous):
    """Update distances inside obstacles using certified source crops."""
    total = obstacles.size
    info = dict(status="full_recompute", reason="new_map", changed_cells=None,
                updated_cells=total, computed_cells=0, total_cells=total,
                region_count=1, transform_calls=0)

    def full(reason):
        info.update(status="full_recompute", reason=reason,
                    updated_cells=total, region_count=1)
        source, output = obstacles, None
        if obstacles.ndim == 2 and np.isfinite(resolution) and resolution > 0:
            rows = np.flatnonzero(obstacles.any(axis=1))
            if rows.size:
                cols = np.flatnonzero(obstacles.any(axis=0))
                h, w = obstacles.shape
                r0, r1 = max(0, int(rows[0]) - 1), min(h, int(rows[-1]) + 2)
                c0, c1 = max(0, int(cols[0]) - 1), min(w, int(cols[-1]) + 2)
                if (r0, r1, c0, c1) != (0, h, 0, w):
                    # The halo outside the occupied box is free. It is nearer
                    # to every obstacle than any excluded source in that direction.
                    output = np.s_[r0:r1, c0:c1]
                    source = obstacles[output]
        info["computed_cells"] += source.size
        info["transform_calls"] += 1
        distance = distance_transform_edt(source) * resolution
        if output is not None:
            complete = np.zeros(obstacles.shape, dtype=distance.dtype)
            complete[output] = distance
            distance = complete
        return distance, info

    if previous is None or previous.obstacle_mask.shape != obstacles.shape:
        return full("new_map")
    if obstacles.ndim != 2 or not np.isfinite(resolution) or resolution <= 0:
        return full("unsupported_geometry")
    changed = obstacles != previous.obstacle_mask
    info["changed_cells"] = int(np.count_nonzero(changed))
    # SciPy uses an implicit outside source when the whole map is blocked.
    if previous.obstacle_mask.all() or obstacles.all():
        return full("no_free_source")
    if not info["changed_cells"]:
        info.update(status="unchanged", reason="inputs_unchanged",
                    updated_cells=0, region_count=0)
        return np.maximum(-previous.wall_dist, 0.0), info

    old_min = float(previous.wall_dist.min())
    if not np.isfinite(old_min):
        return full("unsupported_geometry")
    old_max = max(0.0, -old_min) / resolution
    if not np.isfinite(old_max):
        return full("unsupported_geometry")
    # The extra cell protects the bound after world-to-pixel conversion.
    padding = int(np.ceil(old_max)) + 1
    regions = _speed_update_regions(changed, padding)
    if sum((r1 - r0) * (c1 - c0) for _, (r0, r1, c0, c1) in regions) >= total:
        return full("full_region")

    interior = np.maximum(-previous.wall_dist, 0.0)
    h, w = obstacles.shape
    for (r0, r1, c0, c1), _ in regions:
        halo = padding
        for _ in range(3):
            sr0, sr1 = max(0, r0 - halo), min(h, r1 + halo)
            sc0, sc1 = max(0, c0 - halo), min(w, c1 + halo)
            source = obstacles[sr0:sr1, sc0:sc1]
            if source.all():
                halo *= 2
                continue
            if info["computed_cells"] + source.size >= total:
                return full("work_limit")
            distance = distance_transform_edt(source)
            info["computed_cells"] += source.size
            info["transform_calls"] += 1
            local = distance[r0 - sr0:r1 - sr0, c0 - sc0:c1 - sc0]
            rows, cols = np.arange(r0, r1), np.arange(c0, c1)
            row_bound = np.minimum(rows - sr0 + 1 if sr0 else np.inf,
                                   sr1 - rows if sr1 < h else np.inf)
            col_bound = np.minimum(cols - sc0 + 1 if sc0 else np.inf,
                                   sc1 - cols if sc1 < w else np.inf)
            bound = np.minimum(np.broadcast_to(row_bound, rows.shape)[:, None],
                               np.broadcast_to(col_bound, cols.shape)[None, :])
            if np.all(local <= bound):
                interior[r0:r1, c0:c1] = local * resolution
                break
            halo *= 2
        else:
            return full("uncertified_region")
    info.update(status="local_update", reason="changed_region",
                updated_cells=sum((r1 - r0) * (c1 - c0)
                                  for (r0, r1, c0, c1), _ in regions),
                region_count=len(regions))
    return interior, info


def solve_eikonal(
    obstacle_mask: np.ndarray,
    speed: np.ndarray,
    goal_col: int,
    goal_row: int,
    resolution: float,
    early_exit_target: Optional[Tuple[int, int]] = None,
    early_exit_margin_cells: int = 0,
    updater=None,
) -> np.ndarray:
    """Run FMM via skfmm.

    With ``early_exit_target=(row, col)`` set, the wavefront stops shortly
    after that cell is frozen (``early_exit_margin_cells`` past it). Cells
    outside the marched region come back as NaN, same as obstacle cells --
    the gradient pipeline treats both as no-go.

    Both early_exit_* kwargs require the forked scikit-fmm with the
    early-exit patch. If not present, raises a TypeError at the
    skfmm.travel_time call.
    """
    if updater is not None:
        return updater.solve(obstacle_mask, speed, goal_col, goal_row, resolution,
                             early_exit_target, early_exit_margin_cells)
    h, w = obstacle_mask.shape
    phi = np.ones((h, w), dtype=np.float64)
    phi[goal_row, goal_col] = -1.0
    phi_m = np.ma.MaskedArray(phi, mask=obstacle_mask)
    spd_m = np.ma.MaskedArray(speed, mask=obstacle_mask)

    kwargs = {"dx": resolution}
    if early_exit_target is not None:
        kwargs["early_exit_target"] = early_exit_target
        kwargs["early_exit_margin"] = int(early_exit_margin_cells)

    raw = skfmm.travel_time(phi_m, spd_m, **kwargs)
    tt = np.array(raw, dtype=np.float64)
    if np.ma.is_masked(raw):
        tt[raw.mask] = np.nan
    return tt


def _gaussian_with_nan(arr: np.ndarray, sigma_cells: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing (Knutsson-Westin normalised convolution)."""
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(valid > 0, arr, 0.0)
    num = gaussian_filter(filled, sigma_cells)
    den = gaussian_filter(valid, sigma_cells)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    out[~np.isfinite(arr)] = np.nan
    return out


def field_from_T(
    tt: np.ndarray,
    obstacle_mask: np.ndarray,
    speed: np.ndarray,
    resolution: float,
    smooth_sigma_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate (optionally smoothed) T to produce a unit vector field.

    Smoothing, when enabled, is applied to T BEFORE the gradient operator.
    Smoothing T preserves the scalar-potential structure; smoothing the
    gradient components afterwards does not -- it introduces curl and
    creates regions where the renormalised direction is unstable.

    No-go cells -- obstacles OR any cell the FMM did not reach (early-exit
    cutoff, disconnected pocket behind walls) -- are substituted with a
    large finite value before differentiation. The central-difference
    gradient at the boundary between reached cells and no-go cells then
    points outward, providing a valid recovery direction if the robot
    ever strays out of the computed region.

    Returns
    -------
    gx, gy  unit vector field components, zeroed at cells where the gradient
            is undefined.
    mag     speed-corrected confidence = |grad T| * v.

            In FM2, |grad T| = 1/v in smooth regions, so the raw magnitude
            varies by v_max/v_min across the inflation band as a NORMAL
            feature of the field. Multiplying by v cancels this variation:
            the product is ~1 in every smooth region regardless of profile,
            and collapses to 0 at FMM cut loci and at the goal where the
            direction genuinely is unreliable.

            At the no-go boundary, raw_mag is artificially huge (T jumps
            from ~free_max_T to ~4*free_max_T over one cell). The downstream
            planner is expected to clip or threshold mag for the alignment
            cost; this is the same behaviour the field has always exhibited
            at obstacle boundaries.
    """
    if smooth_sigma_m > 0.0:
        sigma_cells = smooth_sigma_m / resolution
        tt_for_grad = _gaussian_with_nan(tt, sigma_cells)
    else:
        tt_for_grad = tt.copy()

    # No-go = obstacle OR unreached. Treating these uniformly with the
    # same big-T sentinel does double duty: it gives the boundary cells
    # a recovery gradient and it gives the cost-to-go (T itself) a high
    # value everywhere the robot shouldn't be.
    tt_diff = prepare_travel_time(tt_for_grad, invalid_mask=obstacle_mask)

    d_row, d_col = np.gradient(tt_diff, resolution)
    raw_gx = -d_col
    raw_gy = -d_row

    raw_mag = np.sqrt(raw_gx * raw_gx + raw_gy * raw_gy)
    safe = np.where(raw_mag > 1e-8, raw_mag, 1.0)
    gx = raw_gx / safe
    gy = raw_gy / safe
    mag = raw_mag * speed

    bad = ~np.isfinite(gx) | ~np.isfinite(gy)
    gx[bad] = 0.0
    gy[bad] = 0.0
    mag[bad] = 0.0
    return gx, gy, mag


@dataclass
class _MapData:
    obstacle_mask: np.ndarray
    speed: np.ndarray
    wall_dist: np.ndarray
    speed_update: Optional[dict] = None
    interior_update: Optional[dict] = None


def _obstacle_mask(map_array, occupancy_threshold, allow_unknown):
    obstacles = map_array >= occupancy_threshold
    if not allow_unknown:
        obstacles |= map_array < 0
    return obstacles


def _valid_cells(obstacles, goal_col, goal_row, cfg, robot_col, robot_row):
    h, w = obstacles.shape
    if not (0 <= goal_col < w and 0 <= goal_row < h):
        return False
    if obstacles[goal_row, goal_col]:
        return False
    if cfg.early_exit_enable:
        if robot_col is None or robot_row is None:
            return False
        if not (0 <= robot_col < w and 0 <= robot_row < h):
            return False
        if obstacles[robot_row, robot_col]:
            return False
    return True


def _prepare_map(obstacles, resolution, cfg, previous=None, incremental=False):
    edt_free = distance_transform_edt(~obstacles) * resolution
    speed_update = None
    if incremental:
        speed, speed_update = _update_speed_field(edt_free, obstacles, cfg, resolution, previous)
    else:
        speed = build_speed_field(edt_free, obstacles, cfg, resolution)
    interior_update = None
    if incremental:
        interior, interior_update = _update_interior_distance(obstacles, resolution, previous)
        wall_dist = edt_free - interior
    else:
        wall_dist = edt_free - distance_transform_edt(obstacles) * resolution
    return _MapData(obstacles, speed, wall_dist, speed_update, interior_update)


def compute_field(
    map_array: np.ndarray,
    goal_col: int,
    goal_row: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    cfg: VectorFieldConfig,
    robot_col: Optional[int] = None,
    robot_row: Optional[int] = None,
    occupancy_threshold: int = 65,
    allow_unknown: bool = False,
) -> Optional[Tuple[VectorFieldResult, str]]:
    """Run the full FM2 pipeline and return (result, log_message).

    Returns None on unrecoverable error:
      - goal cell is an obstacle or outside the grid
      - early exit requested but no robot position supplied
      - robot position is out of bounds or inside an obstacle

    Parameters
    ----------
    map_array           (H, W) int8 occupancy grid data (values 0-100, -1=unknown).
    goal_col, goal_row  target cell in grid coordinates.
    resolution          [m/cell].
    origin_x, origin_y  world coordinates of the cell-corner of cell (0, 0).
    cfg                 field config.
    robot_col, robot_row  robot's cell coordinates. Only consulted when
                        early exit is enabled; otherwise ignored.
    occupancy_threshold cells with value >= this are treated as obstacles.
    allow_unknown       if False, unknown cells (value -1) are also obstacles.

    Returns
    -------
    (VectorFieldResult, message_str) on success,
    None                             on failure.
    """
    obstacles = _obstacle_mask(map_array, occupancy_threshold, allow_unknown)
    if not _valid_cells(obstacles, goal_col, goal_row, cfg, robot_col, robot_row):
        return None
    prepared = _prepare_map(obstacles, resolution, cfg)
    return _compute_prepared_field(prepared, goal_col, goal_row, resolution,
                                   origin_x, origin_y, cfg, robot_col, robot_row)


def _compute_prepared_field(prepared, goal_col, goal_row, resolution,
                            origin_x, origin_y, cfg, robot_col, robot_row,
                            eikonal_updater=None):
    obstacle_mask, speed = prepared.obstacle_mask, prepared.speed
    h, w = obstacle_mask.shape
    early_exit_target: Optional[Tuple[int, int]] = None
    early_exit_margin_cells = 0
    use_early_exit = cfg.early_exit_enable

    if use_early_exit:
        early_exit_target = (int(robot_row), int(robot_col))
        # Round up the configured buffer to a whole number of cells.
        early_exit_margin_cells = int(
            np.ceil(max(0.0, cfg.early_exit_margin) / resolution)
        )

    tt = solve_eikonal(
        obstacle_mask,
        speed,
        goal_col,
        goal_row,
        resolution,
        early_exit_target=early_exit_target,
        early_exit_margin_cells=early_exit_margin_cells,
        updater=eikonal_updater,
    )

    free_tt = tt[np.isfinite(tt)]
    free_max_T = float(free_tt.max()) if free_tt.size else 1.0

    sigma = cfg.smooth_T_sigma if cfg.smooth_T_before_grad else 0.0
    gx, gy, mag = field_from_T(tt, obstacle_mask, speed, resolution, sigma)

    n_free = int((~obstacle_mask).sum())
    n_reached = int(np.isfinite(tt).sum())
    pct = 100.0 * n_reached / max(n_free, 1)

    if use_early_exit:
        ee_desc = (
            f"on, robot=({robot_row},{robot_col}), "
            f"margin={cfg.early_exit_margin:.2f}m={early_exit_margin_cells}c"
        )
    else:
        ee_desc = "off"

    message = (
        f"Field computed: {w}x{h} cells, "
        f"{n_reached}/{n_free} free cells reached ({pct:.1f}%) "
        f"[smooth_T={cfg.smooth_T_before_grad}, "
        f"profile={cfg.speed_profile}, "
        f"R={cfg.inflation_radius:.2f}m, "
        f"early_exit={ee_desc}]"
    )

    result = VectorFieldResult(
        travel_time=tt,
        grad_x=gx,
        grad_y=gy,
        grad_mag=mag,
        free_max_T=free_max_T,
        origin_x=origin_x,
        origin_y=origin_y,
        resolution=resolution,
        wall_dist=prepared.wall_dist,
    )
    if cfg.bottleneck_align:
        from agx_planning.vector_field.bottleneck import find_bottlenecks, shape_direction
        bns = find_bottlenecks(obstacle_mask, resolution, origin_x, origin_y,
                               max_width=cfg.bottleneck_max_width,
                               approach=cfg.bottleneck_approach)
        result.dir_x, result.dir_y, result.bottlenecks = shape_direction(
            gx, gy, obstacle_mask, bns, resolution, origin_x, origin_y,
            approach=cfg.bottleneck_approach, exit=cfg.bottleneck_exit,
            lateral=cfg.bottleneck_lateral, fade=cfg.bottleneck_fade,
            converge_angle=np.deg2rad(cfg.bottleneck_converge_deg),
            converge_dist=cfg.bottleneck_converge_dist,
            goal_xy=(origin_x + (goal_col + 0.5) * resolution,
                     origin_y + (goal_row + 0.5) * resolution))
        message += f" [bottleneck_align: {len(result.bottlenecks)}/{len(bns)} oriented]"
    return result, message


def pack_field_array(result: VectorFieldResult) -> np.ndarray:
    """Pack a VectorFieldResult into a flat float32 array.

    Layout (same as /vector_field/planner_data):
      [h, w, origin_x, origin_y, resolution,
       travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W),
       wall_dist(H*W)]

    wall_dist is appended last (and only when present) so a parser that
    reads the first four channels by offset is unaffected.

    NaN cells in T are replaced with (free_max_T * 4 + 1) so the planner
    can compare without special-casing NaN. This sentinel matches the
    `big` value used in field_from_T, so the cost-to-go and gradient
    fields are consistent: cells the planner sees as "infinitely costly"
    in T are exactly the cells whose -grad T points back into safe space.
    Corresponding grad_mag entries are already zero from field_from_T().
    """
    h, w = result.travel_time.shape
    tt_out = prepare_travel_time(result.travel_time, max_finite=result.free_max_T)
    mag_out = np.where(np.isfinite(result.grad_mag), result.grad_mag, 0.0)

    header = np.array(
        [h, w, result.origin_x, result.origin_y, result.resolution],
        dtype=np.float32,
    )
    parts = [
        header,
        tt_out.astype(np.float32).ravel(),
        result.grad_x.astype(np.float32).ravel(),
        result.grad_y.astype(np.float32).ravel(),
        mag_out.astype(np.float32).ravel(),
    ]
    if result.wall_dist is not None:
        parts.append(result.wall_dist.astype(np.float32).ravel())
    return np.concatenate(parts)


def field_result_to_grid(
    result: VectorFieldResult,
    field_eps: float = 1e-2,
    wall_sigma: float = 0.05,
) -> VectorFieldGrid:
    """Build a VectorFieldGrid directly from a VectorFieldResult.

    Bypasses the Float32MultiArray serialisation round-trip; use this
    when the generator and the planner share a Python process (sim,
    offline batch planning, unit tests).

    Invalid cells use the same result.free_max_T sentinel as pack_field_array.
    This path retains float64 values; the packed path rounds to float32.
    """
    grid = VectorFieldGrid()
    grid.update(
        prepare_travel_time(result.travel_time, max_finite=result.free_max_T),
        result.origin_x,
        result.origin_y,
        result.resolution,
        field_eps=field_eps,
        wall_dist=result.wall_dist,
        wall_sigma=wall_sigma,
        direction=(None if result.dir_x is None else (result.dir_x, result.dir_y)),
    )
    return grid
