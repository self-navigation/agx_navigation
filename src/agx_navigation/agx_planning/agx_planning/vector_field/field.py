"""Pure-Python FM2 (Fast Marching Square) vector field computation.

No ROS 2 imports -- usable from scripts, notebooks, and tests without
pulling in rclpy.

Public API
----------
  SpeedConfig         -- EDT-to-speed profile parameters
  CutLocusConfig      -- optional T-smoothing for FMM cut-locus fix
  VectorFieldResult   -- output of compute_field(); holds T, grad, metadata
  world_to_grid       -- world (x, y) -> (col, row), returns None if OOB
  grid_to_world       -- (col, row) -> world (x, y) cell-centre
  compute_field       -- full FM2 pipeline; returns VectorFieldResult or None
  pack_field_array    -- serialise VectorFieldResult to a flat float32 array
                         (same layout as /vector_field/planner_data)
  field_result_to_grid -- build a VectorFieldGrid directly from a result,
                          bypassing the Float32MultiArray round-trip; use
                          this when the generator and planner share a process

Internal helpers (not part of the public API):
  build_speed_field, solve_eikonal_full, field_from_T, _gaussian_with_nan
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, gaussian_filter

from agx_planning.vector_field import VectorFieldGrid


@dataclass
class SpeedConfig:
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


@dataclass
class CutLocusConfig:
    # Smooth T before differentiating. This regularises FMM ridges, where
    # the central-difference gradient otherwise averages two opposing sides
    # and produces an unstable unit direction. Default OFF: with the
    # confidence-weighted planner cost, the planner can de-weight cut-locus
    # cells without needing the smoothing pass. Re-enable if you observe
    # zig-zagging across corridor centrelines and confidence weighting is
    # disabled in the planner.
    smooth_T_before_grad: bool = False
    smooth_T_sigma: float = 0.10  # [m]


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
    """

    travel_time: np.ndarray
    grad_x: np.ndarray
    grad_y: np.ndarray
    grad_mag: np.ndarray
    free_max_T: float
    origin_x: float
    origin_y: float
    resolution: float


def world_to_grid(
    wx: float,
    wy: float,
    origin_x: float,
    origin_y: float,
    resolution: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    """Convert world (wx, wy) to (col, row). Returns None if out of bounds.

    Cell-corner convention: the origin is the corner of cell (0, 0), so
      col = int((wx - origin_x) / resolution)
      row = int((wy - origin_y) / resolution)
    consistent with OccupancyGrid.info.origin.
    """
    col = int((wx - origin_x) / resolution)
    row = int((wy - origin_y) / resolution)
    if 0 <= col < width and 0 <= row < height:
        return col, row
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


def build_speed_field(
    edt_free: np.ndarray,
    obstacle_mask: np.ndarray,
    cfg: SpeedConfig,
) -> np.ndarray:
    """Map EDT distance to wave speed v(x) for the eikonal solve."""
    R = cfg.inflation_radius
    v = np.full_like(edt_free, cfg.speed_v_max, dtype=np.float64)
    if R <= 0.0:
        return v

    near = ~obstacle_mask & (edt_free < R)
    if not near.any():
        return v

    norm = edt_free[near] / R  # 0 at wall, 1 at boundary
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


def solve_eikonal_full(
    obstacle_mask: np.ndarray,
    speed: np.ndarray,
    goal_col: int,
    goal_row: int,
    resolution: float,
) -> np.ndarray:
    """Full-grid FMM via skfmm. Returns T with NaN on obstacle cells."""
    h, w = obstacle_mask.shape
    phi = np.ones((h, w), dtype=np.float64)
    phi[goal_row, goal_col] = -1.0
    phi_m = np.ma.MaskedArray(phi, mask=obstacle_mask)
    spd_m = np.ma.MaskedArray(speed, mask=obstacle_mask)
    raw = skfmm.travel_time(phi_m, spd_m, dx=resolution)
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

    Optional smoothing is applied to T BEFORE the gradient operator. This
    is the cut-locus fix: smoothing T preserves the scalar-potential
    structure of the field, while smoothing the gradient components
    afterwards does not -- it introduces curl and creates regions where the
    renormalised direction is unstable.

    Obstacle cells (NaN in T) are substituted with a large finite value
    before differentiation. The central-difference gradient at wall-adjacent
    free cells then points outward, providing a valid recovery direction if
    the robot ever enters one.

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
    """
    if smooth_sigma_m > 0.0:
        sigma_cells = smooth_sigma_m / resolution
        tt_for_grad = _gaussian_with_nan(tt, sigma_cells)
    else:
        tt_for_grad = tt.copy()

    finite = tt_for_grad[np.isfinite(tt_for_grad)]
    big = float(finite.max()) * 4.0 + 1.0 if finite.size else 1.0
    tt_diff = np.where(obstacle_mask, big, tt_for_grad)

    d_row, d_col = np.gradient(tt_diff, resolution)
    raw_gx = -d_col
    raw_gy = -d_row

    raw_mag = np.sqrt(raw_gx * raw_gx + raw_gy * raw_gy)
    safe = np.where(raw_mag > 1e-8, raw_mag, 1.0)
    gx = raw_gx / safe
    gy = raw_gy / safe
    mag = raw_mag * speed

    # Cells with non-finite gradient -> zero direction and zero confidence.
    bad = ~np.isfinite(gx) | ~np.isfinite(gy)
    gx[bad] = 0.0
    gy[bad] = 0.0
    mag[bad] = 0.0
    return gx, gy, mag


def compute_field(
    map_array: np.ndarray,
    goal_col: int,
    goal_row: int,
    resolution: float,
    origin_x: float,
    origin_y: float,
    speed_cfg: SpeedConfig,
    cutlocus_cfg: CutLocusConfig,
    occupancy_threshold: int = 65,
    allow_unknown: bool = False,
) -> Optional[Tuple[VectorFieldResult, str]]:
    """Run the full FM2 pipeline and return (result, log_message).

    Returns None on unrecoverable error (goal in obstacle, goal outside
    grid). The caller is responsible for logging -- this function has no
    logger dependency.

    Parameters
    ----------
    map_array           (H, W) int8 occupancy grid data (values 0-100, -1=unknown).
    goal_col, goal_row  target cell in grid coordinates.
    resolution          [m/cell].
    origin_x, origin_y  world coordinates of the cell-corner of cell (0, 0).
    speed_cfg           EDT-to-speed profile.
    cutlocus_cfg        optional T-smoothing.
    occupancy_threshold cells with value >= this are treated as obstacles.
    allow_unknown       if False, unknown cells (value -1) are also obstacles.

    Returns
    -------
    (VectorFieldResult, message_str) on success,
    None                             on failure.
    """
    h, w = map_array.shape

    obstacle_mask = map_array >= occupancy_threshold
    if not allow_unknown:
        obstacle_mask = obstacle_mask | (map_array < 0)

    if obstacle_mask[goal_row, goal_col]:
        return None

    edt_free = distance_transform_edt(~obstacle_mask) * resolution
    speed = build_speed_field(edt_free, obstacle_mask, speed_cfg)
    tt = solve_eikonal_full(obstacle_mask, speed, goal_col, goal_row, resolution)

    free_tt = tt[np.isfinite(tt)]
    free_max_T = float(free_tt.max()) if free_tt.size else 1.0

    sigma = cutlocus_cfg.smooth_T_sigma if cutlocus_cfg.smooth_T_before_grad else 0.0
    gx, gy, mag = field_from_T(tt, obstacle_mask, speed, resolution, sigma)

    n_finite = int(np.isfinite(tt).sum())
    message = (
        f"Field computed: {w}x{h} cells, {n_finite} reached "
        f"({100.0 * n_finite / (h * w):.1f}%) "
        f"[smooth_T={cutlocus_cfg.smooth_T_before_grad}, "
        f"profile={speed_cfg.speed_profile}, "
        f"R={speed_cfg.inflation_radius:.2f} m]"
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
    )
    return result, message


def pack_field_array(result: VectorFieldResult) -> np.ndarray:
    """Pack a VectorFieldResult into a flat float32 array.

    Layout (same as /vector_field/planner_data):
      [h, w, origin_x, origin_y, resolution,
       travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W)]

    NaN cells in T are replaced with (free_max_T * 4 + 1) so the planner
    can compare without special-casing NaN. Corresponding grad_mag entries
    are already zero from field_from_T().
    """
    h, w = result.travel_time.shape
    big_T = result.free_max_T * 4.0 + 1.0
    tt_out = np.where(np.isfinite(result.travel_time), result.travel_time, big_T)
    mag_out = np.where(np.isfinite(result.grad_mag), result.grad_mag, 0.0)

    header = np.array(
        [h, w, result.origin_x, result.origin_y, result.resolution],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            header,
            tt_out.astype(np.float32).ravel(),
            result.grad_x.astype(np.float32).ravel(),
            result.grad_y.astype(np.float32).ravel(),
            mag_out.astype(np.float32).ravel(),
        ]
    )


def field_result_to_grid(
    result: VectorFieldResult,
    field_eps: float = 1e-2,
    align_smooth_sigma: float = 0.0,
) -> VectorFieldGrid:
    """Build a VectorFieldGrid directly from a VectorFieldResult.

    This bypasses the Float32MultiArray serialization/deserialization
    round-trip (pack_field_array -> ROS topic -> parse_field_array) and
    is the preferred path when the field generator and the planner share
    a Python process (e.g. in simulation, offline batch planning, or
    unit tests).

    Parameters
    ----------
    result              output of compute_field().
    field_eps           normalisation regulariser passed to VectorFieldGrid.update();
                        controls how fast the alignment cost fades near the goal.
    align_smooth_sigma  if > 0, VectorFieldGrid re-derives F from a smoothed T
                        internally; set to 0 to use grad_x/grad_y as-is.
    """
    grid = VectorFieldGrid()
    grid.update(
        result.travel_time,
        result.grad_x,
        result.grad_y,
        result.origin_x,
        result.origin_y,
        result.resolution,
        field_eps=field_eps,
        align_smooth_sigma=align_smooth_sigma,
    )
    return grid
