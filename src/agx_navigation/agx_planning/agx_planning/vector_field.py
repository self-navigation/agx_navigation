import math
import time
from typing import Optional

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, gaussian_filter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException


def fast_sweep_eikonal(
    T: np.ndarray,
    speed: np.ndarray,
    fixed: np.ndarray,
    dx: float,
    max_sweeps: int = 8,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve the Eikonal equation on a 2D patch using the Fast Sweeping Method
    (Zhao 2005) with fixed Dirichlet boundary cells.

    Sweeps in all 4 diagonal directions; converges in 2-4 sweeps for convex domains.
    """
    H, W = T.shape

    for _ in range(max_sweeps):
        max_change = 0.0

        for row_order in [range(H), range(H - 1, -1, -1)]:
            for col_order in [range(W), range(W - 1, -1, -1)]:
                for r in row_order:
                    for c in col_order:
                        if fixed[r, c] or speed[r, c] <= 0.0:
                            continue

                        t_x = min(
                            T[r, c - 1] if c > 0 else np.inf,
                            T[r, c + 1] if c < W - 1 else np.inf,
                        )
                        t_y = min(
                            T[r - 1, c] if r > 0 else np.inf,
                            T[r + 1, c] if r < H - 1 else np.inf,
                        )

                        slowness = dx / speed[r, c]
                        t_a, t_b = (t_x, t_y) if t_x <= t_y else (t_y, t_x)

                        # 1D update; upgrade to 2D if the result exceeds t_b
                        t_new = t_a + slowness
                        if t_new > t_b:
                            disc = 2.0 * slowness**2 - (t_a - t_b) ** 2
                            if disc >= 0.0:
                                t_new = (t_a + t_b + math.sqrt(disc)) / 2.0

                        if t_new < T[r, c]:
                            max_change = max(max_change, T[r, c] - t_new)
                            T[r, c] = t_new

        if max_change < tol:
            break

    return T


# Numba-accelerated inner loop (~50-100x faster). Falls back to pure Python
# if numba is not installed.
try:
    from numba import njit

    @njit(cache=True)
    def _fast_sweep_inner(T, speed, fixed, dx, max_sweeps, tol):
        H, W = T.shape
        for _ in range(max_sweeps):
            max_change = 0.0
            for r in range(H):
                for c in range(W):
                    if fixed[r, c] or speed[r, c] <= 0.0:
                        continue
                    t_x = min(T[r, c-1] if c > 0 else np.inf, T[r, c+1] if c < W-1 else np.inf)
                    t_y = min(T[r-1, c] if r > 0 else np.inf, T[r+1, c] if r < H-1 else np.inf)
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y); t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            for r in range(H):
                for c in range(W - 1, -1, -1):
                    if fixed[r, c] or speed[r, c] <= 0.0:
                        continue
                    t_x = min(T[r, c-1] if c > 0 else np.inf, T[r, c+1] if c < W-1 else np.inf)
                    t_y = min(T[r-1, c] if r > 0 else np.inf, T[r+1, c] if r < H-1 else np.inf)
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y); t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            for r in range(H - 1, -1, -1):
                for c in range(W):
                    if fixed[r, c] or speed[r, c] <= 0.0:
                        continue
                    t_x = min(T[r, c-1] if c > 0 else np.inf, T[r, c+1] if c < W-1 else np.inf)
                    t_y = min(T[r-1, c] if r > 0 else np.inf, T[r+1, c] if r < H-1 else np.inf)
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y); t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1):
                    if fixed[r, c] or speed[r, c] <= 0.0:
                        continue
                    t_x = min(T[r, c-1] if c > 0 else np.inf, T[r, c+1] if c < W-1 else np.inf)
                    t_y = min(T[r-1, c] if r > 0 else np.inf, T[r+1, c] if r < H-1 else np.inf)
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y); t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            if max_change < tol:
                break
        return T

    def fast_sweep_eikonal_numba(T, speed, fixed, dx, max_sweeps=8, tol=1e-6):
        return _fast_sweep_inner(T, speed, fixed, dx, max_sweeps, tol)

    _USE_NUMBA = True

except ImportError:
    _USE_NUMBA = False


def eikonal_local_solve(T, speed, fixed, dx, max_sweeps=8, tol=1e-6):
    if _USE_NUMBA:
        return fast_sweep_eikonal_numba(T, speed, fixed, dx, max_sweeps, tol)
    return fast_sweep_eikonal(T, speed, fixed, dx, max_sweeps, tol)


class FMMVectorFieldNode(Node):
    def __init__(self):
        super().__init__("fmm_vector_field")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        # Must match global_frame in your Nav2 local costmap config (default: odom).
        self.declare_parameter("local_costmap_frame", "odom")
        self.declare_parameter("inflation_weight", 0.8)
        self.declare_parameter("lethal_cost", 99)
        self.declare_parameter("allow_unknown", False)
        self.declare_parameter("viz_subsample", 4)
        self.declare_parameter("viz_arrow_length", 0.3)
        self.declare_parameter("viz_rate", 5.0)
        # "max": take the higher cost from either costmap. "replace": local overrides global.
        self.declare_parameter("local_merge_mode", "max")
        # Cells of padding around the local costmap footprint included in the patch re-solve.
        # 1-2 cells is usually sufficient; the boundary ring is pinned to T_global anyway.
        self.declare_parameter("local_patch_padding", 2)
        self.declare_parameter("local_max_sweeps", 6)
        # Scale factor for the EDT depth penalty inside lethal cells.
        # penalty = obstacle_slope_factor * max_T * depth_m
        # Must make crossing the inflation depth more expensive than any free-space detour.
        # Rule of thumb: factor >= map_diameter / inflation_depth (e.g. 10m / 0.23m ~ 43).
        self.declare_parameter("obstacle_slope_factor", 400.0)
        # Adds an EDT-based repulsive potential to free cells within this radius of a
        # lethal cell, decaying quadratically to zero at the boundary. Eliminates the
        # inflation-zone travel-time "plateau" that makes cutting through it look cheaper
        # than retreating. Set to your Nav2 inflation_radius or slightly above.
        self.declare_parameter("wall_repulsion_radius", 0.5)
        # Peak of the repulsive potential as a multiple of max free-space T.
        # Raise if the robot still cuts corners; lower if detours become too wide.
        self.declare_parameter("wall_repulsion_strength", 3.0)
        # Gaussian blur sigma in cells, applied to (gx, gy) before normalisation.
        # The sharp direction jump at the wall_repulsion_radius boundary excites
        # the MPC heading controller, causing wave-like oscillation along walls.
        # Blurring spreads the transition without changing repulsion strength
        # (magnitude is restored by re-normalisation). 0.0 disables; 2.5 is a good start.
        self.declare_parameter("field_smooth_sigma", 2.5)
        # FMM speed function: speed = exp(-rate * normalised_cost).
        # Exponential mapping gives much steeper gradients near lethal cells than
        # the old linear formula (1 - inflation_weight * norm).
        # rate=0.0 falls back to linear. rate=2.5 gives speed range [1.0, 0.08].
        self.declare_parameter("inflation_decay_rate", 2.5)

        self.map_frame: str = self.get_parameter("map_frame").value
        self.robot_frame: str = self.get_parameter("robot_frame").value
        self.local_costmap_frame: str = self.get_parameter("local_costmap_frame").value
        # Cached odom->map translation applied to local costmap cell coordinates.
        self._local_to_map_tx: float = 0.0
        self._local_to_map_ty: float = 0.0
        self.inflation_weight: float = self.get_parameter("inflation_weight").value
        self.lethal_cost: int = self.get_parameter("lethal_cost").value
        self.allow_unknown: bool = self.get_parameter("allow_unknown").value
        self.viz_subsample: int = self.get_parameter("viz_subsample").value
        self.viz_arrow_length: float = self.get_parameter("viz_arrow_length").value
        viz_rate: float = self.get_parameter("viz_rate").value
        self.local_merge_mode: str = self.get_parameter("local_merge_mode").value
        self.local_patch_padding: int = self.get_parameter("local_patch_padding").value
        self.local_max_sweeps: int = self.get_parameter("local_max_sweeps").value
        self.obstacle_slope_factor: float = self.get_parameter("obstacle_slope_factor").value
        self.wall_repulsion_radius: float = self.get_parameter("wall_repulsion_radius").value
        self.wall_repulsion_strength: float = self.get_parameter("wall_repulsion_strength").value
        self.inflation_decay_rate: float = self.get_parameter("inflation_decay_rate").value
        self.field_smooth_sigma: float = self.get_parameter("field_smooth_sigma").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.costmap_msg: Optional[OccupancyGrid] = None
        self.costmap_array: Optional[np.ndarray] = None
        self.current_goal: Optional[PoseStamped] = None

        # Live field arrays: global solve writes these; local patch overwrites a sub-region.
        self.travel_time: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_y: Optional[np.ndarray] = None

        # Pristine copies from the last global solve; never modified by local patches.
        self._global_travel_time: Optional[np.ndarray] = None
        self._global_grad_x: Optional[np.ndarray] = None
        self._global_grad_y: Optional[np.ndarray] = None
        self._global_speed: Optional[np.ndarray] = None
        self._free_max_T: float = 1.0

        # Accumulates lethal cells from local observations. Cleared only when the sensor
        # explicitly free-spaces a cell (cost == 0), preventing obstacle erasure at the
        # rolling window boundary. Reset to the global mask on each global FMM re-solve.
        self._persistent_obstacle_mask: Optional[np.ndarray] = None

        self.field_dirty: bool = False
        self._local_costmap_msg: Optional[OccupancyGrid] = None
        self._local_costmap_array: Optional[np.ndarray] = None
        self._local_dirty: bool = False

        costmap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, "/global_costmap/costmap", self._costmap_cb, costmap_qos,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_cb, 10
        )
        local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self._local_costmap_cb, local_qos,
        )

        self.lines_pub = self.create_publisher(Marker, "/vector_field/lines", 10)
        self.cost_to_go_pub = self.create_publisher(
            OccupancyGrid, "/vector_field/cost_to_go", 10
        )
        self.planner_data_pub = self.create_publisher(
            Float32MultiArray, "/vector_field/planner_data", 1
        )
        self.viz_timer = self.create_timer(1.0 / viz_rate, self._viz_timer_cb)

        self.get_logger().info(
            f"FMMVectorFieldNode ready (numba={'yes' if _USE_NUMBA else 'no'}). "
            "Waiting for costmap + goal..."
        )

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_msg = msg
        self.costmap_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.field_dirty = True
        self.get_logger().info(
            f"Global costmap: {msg.info.width}x{msg.info.height} "
            f"res={msg.info.resolution:.3f} m/cell",
            throttle_duration_sec=10.0,
        )

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = msg
        self.field_dirty = True
        self.get_logger().info(
            f"Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )
        self._maybe_recompute_global()

    def _local_costmap_cb(self, msg: OccupancyGrid):
        """The local costmap is published in the odom frame; we look up the odom->map
        TF so local cell coordinates are correctly placed in the global grid.
        """
        self._local_costmap_msg = msg
        self._local_costmap_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )

        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.local_costmap_frame,
                msg.header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            self._local_to_map_tx = t.transform.translation.x
            self._local_to_map_ty = t.transform.translation.y
        except Exception:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.map_frame, self.local_costmap_frame, rclpy.time.Time(),
                )
                self._local_to_map_tx = t.transform.translation.x
                self._local_to_map_ty = t.transform.translation.y
            except Exception as e:
                self.get_logger().warn(
                    f"TF {self.local_costmap_frame}->{self.map_frame} unavailable: {e}",
                    throttle_duration_sec=2.0,
                )

        self._local_dirty = True
        self._maybe_patch_local()

    def _viz_timer_cb(self):
        self._maybe_recompute_global()
        self._publish_visualization()

    def _world_to_grid(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        """World (m) -> global grid (col, row). Returns None if out of bounds."""
        info = self.costmap_msg.info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if 0 <= col < info.width and 0 <= row < info.height:
            return (col, row)
        return None

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        info = self.costmap_msg.info
        wx = info.origin.position.x + (col + 0.5) * info.resolution
        wy = info.origin.position.y + (row + 0.5) * info.resolution
        return (wx, wy)

    def _local_footprint_in_global(self) -> Optional[tuple[int, int, int, int]]:
        """Bounding box (row_min, row_max, col_min, col_max) of the local costmap
        in global grid coordinates with padding, or None if no overlap.
        """
        if self._local_costmap_msg is None or self.costmap_msg is None:
            return None

        g_info = self.costmap_msg.info
        l_info = self._local_costmap_msg.info

        local_origin_x_map = l_info.origin.position.x + self._local_to_map_tx
        local_origin_y_map = l_info.origin.position.y + self._local_to_map_ty

        col_start = int((local_origin_x_map - g_info.origin.position.x) / g_info.resolution)
        row_start = int((local_origin_y_map - g_info.origin.position.y) / g_info.resolution)

        scale = l_info.resolution / g_info.resolution
        col_span = int(l_info.width * scale)
        row_span = int(l_info.height * scale)

        pad = self.local_patch_padding
        r_min = max(row_start - pad, 0)
        r_max = min(row_start + row_span + pad, g_info.height)
        c_min = max(col_start - pad, 0)
        c_max = min(col_start + col_span + pad, g_info.width)

        if r_min >= r_max or c_min >= c_max:
            return None
        return (r_min, r_max, c_min, c_max)

    def _cost_to_speed(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert int8 costmap values to (speed [0.01,1], obstacle_mask).

        Exponential mapping (inflation_decay_rate > 0) gives a wider speed range
        near the lethal boundary than the linear formula, creating steeper FMM
        gradients and stronger pre-obstacle repulsion.
        """
        fraw = raw.astype(np.float64)
        mask = fraw >= self.lethal_cost
        if not self.allow_unknown:
            mask |= fraw == -1

        clamped = np.clip(fraw, 0.0, self.lethal_cost - 1)
        normalized = clamped / (self.lethal_cost - 1)

        if self.inflation_decay_rate > 0.0:
            speed = np.clip(np.exp(-self.inflation_decay_rate * normalized), 0.01, 1.0)
        else:
            speed = np.clip(1.0 - self.inflation_weight * normalized, 0.01, 1.0)

        return speed, mask

    def _smooth_gradient(
        self, gx: np.ndarray, gy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gaussian-blur (gx, gy) then re-normalise to unit vectors.

        The direction field has a sharp change at the wall_repulsion_radius boundary,
        which excites the MPC heading controller each crossing and causes wave-like
        oscillation along walls. Blurring before normalisation smooths the transition
        without changing repulsion strength.
        """
        if self.field_smooth_sigma <= 0.0:
            return gx, gy

        gx = gaussian_filter(gx, sigma=self.field_smooth_sigma)
        gy = gaussian_filter(gy, sigma=self.field_smooth_sigma)

        mag = np.sqrt(gx**2 + gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        gx /= safe_mag
        gy /= safe_mag
        return gx, gy

    def _apply_wall_repulsion(
        self,
        tt: np.ndarray,
        obstacle_mask: np.ndarray,
        free_max_T: float,
        resolution: float,
    ) -> np.ndarray:
        """Add a quadratic EDT-based repulsive potential to free cells near obstacles.

        FMM gradients point toward the goal, not away from walls. In the inflation zone
        all cells carry high but nearly uniform travel-times, so the MPC has no
        directional signal to escape (the "plateau"). For each free cell within
        wall_repulsion_radius of a lethal cell, adds:
            penalty = wall_repulsion_strength * max_T * ((R - d) / R)^2
        Quadratic (not linear) so the field is C1 at the boundary, avoiding a gradient
        kink. scipy EDT convention: distance_transform_edt(~obstacle_mask) gives each
        free cell its distance to the nearest obstacle cell.
        """
        free_mask = ~obstacle_mask
        if not (obstacle_mask.any() and free_mask.any()):
            return tt

        edt_to_wall = distance_transform_edt(free_mask) * resolution
        R = self.wall_repulsion_radius
        peak_penalty = self.wall_repulsion_strength * free_max_T

        near_wall = free_mask & (edt_to_wall < R)
        if not near_wall.any():
            return tt

        ratio = (R - edt_to_wall[near_wall]) / R
        tt[near_wall] += peak_penalty * ratio * ratio
        return tt

    def _compute_global_field(self):
        """Full FMM solve on the global costmap."""
        if self.costmap_msg is None or self.current_goal is None:
            return

        goal_cell = self._world_to_grid(
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
        )
        if goal_cell is None:
            self.get_logger().error("Goal is outside costmap bounds.")
            return

        gc, gr = goal_cell
        height, width = self.costmap_array.shape
        resolution = self.costmap_msg.info.resolution

        speed, mask = self._cost_to_speed(self.costmap_array)
        speed_masked = np.ma.MaskedArray(speed, mask=mask)

        if mask[gr, gc]:
            self.get_logger().error("Goal cell is inside a lethal obstacle.")
            return

        phi = np.ones((height, width), dtype=np.float64)
        phi[gr, gc] = -1.0
        phi = np.ma.MaskedArray(phi, mask=mask)

        t0 = time.monotonic()
        try:
            travel_time = skfmm.travel_time(phi, speed_masked, dx=resolution)
        except Exception as e:
            self.get_logger().error(f"FMM failed: {e}")
            return
        elapsed = time.monotonic() - t0

        tt = np.array(travel_time, dtype=np.float64)
        if np.ma.is_masked(travel_time):
            tt[travel_time.mask] = np.nan

        # Replace NaN obstacle cells with a depth-proportional penalty via EDT.
        # A flat fill has zero gradient, leaving the robot with no recovery direction
        # if it enters a lethal cell. Depth-proportional fill gives every interior
        # obstacle cell a gradient pointing toward the nearest free cell.
        obstacle_mask = np.isnan(tt)
        max_T = float(np.nanmax(tt)) if not np.all(np.isnan(tt)) else 1.0
        self._free_max_T = max_T

        if obstacle_mask.any() and (~obstacle_mask).any():
            edt = distance_transform_edt(obstacle_mask) * resolution
            slope = self.obstacle_slope_factor * max_T
            tt[obstacle_mask] = max_T + slope * edt[obstacle_mask]
        else:
            slope = self.obstacle_slope_factor * max_T
            tt[obstacle_mask] = max_T * (1.0 + self.obstacle_slope_factor * resolution)

        tt = self._apply_wall_repulsion(tt, obstacle_mask, max_T, resolution)

        d_row, d_col = np.gradient(tt, resolution)
        gx = -d_col
        gy = -d_row

        bad = np.isnan(gx) | np.isnan(gy) | np.isinf(gx) | np.isinf(gy)
        gx[bad] = 0.0
        gy[bad] = 0.0

        # Smooth before normalisation so the blur acts on raw (non-unit) vectors.
        gx, gy = self._smooth_gradient(gx, gy)

        mag = np.sqrt(gx**2 + gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        gx /= safe_mag
        gy /= safe_mag

        self._global_travel_time = tt.copy()
        self._global_grad_x = gx.copy()
        self._global_grad_y = gy.copy()
        self._global_speed = speed.copy()
        self._persistent_obstacle_mask = mask.copy()

        self.travel_time = tt
        self.grad_x = gx
        self.grad_y = gy
        self.field_dirty = False

        self.get_logger().info(
            f"Global FMM: {width}x{height} in {elapsed:.3f}s "
            f"(max_T={max_T:.1f}, slope={slope:.1f}/m, "
            f"repulsion R={self.wall_repulsion_radius}m "
            f"peak={self.wall_repulsion_strength:.1f}xT, "
            f"decay={self.inflation_decay_rate:.1f})"
        )

        if self._local_costmap_array is not None:
            self._local_dirty = True
            self._maybe_patch_local()

    def _maybe_recompute_global(self):
        if self.field_dirty:
            self._compute_global_field()

    def _maybe_patch_local(self):
        """Re-solve a local patch of the Eikonal field with boundary cells pinned
        to the global travel-time.
        """
        if not self._local_dirty or self._global_travel_time is None:
            return

        bbox = self._local_footprint_in_global()
        if bbox is None:
            return

        r_min, r_max, c_min, c_max = bbox
        patch_h = r_max - r_min
        patch_w = c_max - c_min
        resolution = self.costmap_msg.info.resolution
        g_info = self.costmap_msg.info
        l_info = self._local_costmap_msg.info

        t0 = time.monotonic()

        # Start from the pristine global field. The persistent mask accumulates lethal
        # cells and only clears them when the sensor free-spaces them, preventing the
        # "thin strip" clearing effect at the rolling window boundary.
        T_patch = self._global_travel_time[r_min:r_max, c_min:c_max].copy()
        speed_patch = self._global_speed[r_min:r_max, c_min:c_max].copy()

        if self._persistent_obstacle_mask is not None:
            pers_patch = self._persistent_obstacle_mask[r_min:r_max, c_min:c_max]
            speed_patch[pers_patch] = 0.0
            T_patch[pers_patch] = 1e18

        l_h, l_w = self._local_costmap_array.shape
        for lr in range(l_h):
            for lc in range(l_w):
                # l_info.origin is in the odom frame; add the cached TF offset.
                world_x = (
                    l_info.origin.position.x + self._local_to_map_tx
                    + (lc + 0.5) * l_info.resolution
                )
                world_y = (
                    l_info.origin.position.y + self._local_to_map_ty
                    + (lr + 0.5) * l_info.resolution
                )
                gc = int((world_x - g_info.origin.position.x) / g_info.resolution)
                gr = int((world_y - g_info.origin.position.y) / g_info.resolution)
                pr = gr - r_min
                pc = gc - c_min
                if not (0 <= pr < patch_h and 0 <= pc < patch_w):
                    continue

                local_val = int(self._local_costmap_array[lr, lc])
                local_speed, local_mask = self._cost_to_speed(
                    np.array([[local_val]], dtype=np.int8)
                )

                if self._persistent_obstacle_mask is not None:
                    if local_mask[0, 0]:
                        self._persistent_obstacle_mask[gr, gc] = True
                    elif local_val == 0:
                        self._persistent_obstacle_mask[gr, gc] = False

                if self.local_merge_mode == "max":
                    speed_patch[pr, pc] = min(speed_patch[pr, pc], local_speed[0, 0])
                else:
                    speed_patch[pr, pc] = local_speed[0, 0]

                if local_mask[0, 0]:
                    speed_patch[pr, pc] = 0.0
                    T_patch[pr, pc] = 1e18

        # Fixed cells: boundary ring (pinned to T_global), walls, lethal sentinels.
        fixed = np.zeros((patch_h, patch_w), dtype=np.bool_)
        fixed[0, :] = True
        fixed[-1, :] = True
        fixed[:, 0] = True
        fixed[:, -1] = True
        fixed[speed_patch <= 0.0] = True
        fixed[T_patch >= 1e18] = True

        T_patch = eikonal_local_solve(
            T_patch, speed_patch, fixed, resolution,
            max_sweeps=self.local_max_sweeps,
        )

        patch_obstacle = T_patch >= 1e18
        free_max_T = (
            float(np.nanmax(np.where(patch_obstacle, np.nan, T_patch)))
            if (~patch_obstacle).any()
            else float(np.max(self._global_travel_time))
        )
        if patch_obstacle.any() and (~patch_obstacle).any():
            edt_patch = distance_transform_edt(patch_obstacle) * resolution
            slope = self.obstacle_slope_factor * free_max_T
            T_patch[patch_obstacle] = free_max_T + slope * edt_patch[patch_obstacle]
        else:
            T_patch[patch_obstacle] = free_max_T * (
                1.0 + self.obstacle_slope_factor * resolution
            )

        T_patch = self._apply_wall_repulsion(
            T_patch, patch_obstacle, free_max_T, resolution
        )

        d_row, d_col = np.gradient(T_patch, resolution)
        local_gx = -d_col
        local_gy = -d_row

        bad = (
            np.isnan(local_gx) | np.isnan(local_gy)
            | np.isinf(local_gx) | np.isinf(local_gy)
        )
        local_gx[bad] = 0.0
        local_gy[bad] = 0.0

        local_gx, local_gy = self._smooth_gradient(local_gx, local_gy)

        mag = np.sqrt(local_gx**2 + local_gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        local_gx /= safe_mag
        local_gy /= safe_mag

        # Exclude the outermost ring: it is pinned to global values and
        # np.gradient is unreliable at grid edges.
        m = 1
        self.travel_time[r_min + m : r_max - m, c_min + m : c_max - m] = T_patch[m:-m, m:-m]
        self.grad_x[r_min + m : r_max - m, c_min + m : c_max - m] = local_gx[m:-m, m:-m]
        self.grad_y[r_min + m : r_max - m, c_min + m : c_max - m] = local_gy[m:-m, m:-m]

        self._local_dirty = False
        self.get_logger().debug(
            f"Local patch: {patch_w}x{patch_h} in {(time.monotonic()-t0)*1000:.1f}ms",
            throttle_duration_sec=2.0,
        )

    def query_vector(
        self, wx: float, wy: float
    ) -> Optional[tuple[float, float, float]]:
        """Bilinear interpolation of (vx, vy, cost_to_go) at world position (wx, wy).

        Returns None if out of bounds or before the first solve.
        """
        if self.costmap_msg is None or self.grad_x is None or self.travel_time is None:
            return None

        info = self.costmap_msg.info
        gx = (wx - info.origin.position.x) / info.resolution - 0.5
        gy = (wy - info.origin.position.y) / info.resolution - 0.5

        height, width = self.grad_x.shape
        if not (0 <= gx < width - 1 and 0 <= gy < height - 1):
            return None

        x0, y0 = int(gx), int(gy)
        x1, y1 = x0 + 1, y0 + 1
        fx, fy = gx - x0, gy - y0

        def _bilerp(arr: np.ndarray) -> float:
            return float(
                arr[y0, x0] * (1 - fx) * (1 - fy)
                + arr[y0, x1] * fx * (1 - fy)
                + arr[y1, x0] * (1 - fx) * fy
                + arr[y1, x1] * fx * fy
            )

        return (_bilerp(self.grad_x), _bilerp(self.grad_y), _bilerp(self.travel_time))

    def get_robot_pose(self) -> Optional[PoseStamped]:
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = t.header.stamp
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.position.z = t.transform.translation.z
        pose.pose.orientation = t.transform.rotation
        return pose

    def _publish_visualization(self):
        if self.travel_time is None or self.grad_x is None:
            return
        self._publish_line_list()
        self._publish_cost_to_go_grid()
        self._publish_planner_data()

    def _cost_to_color(self, t_val: float, t_max: float) -> ColorRGBA:
        if t_max < 1e-8:
            return ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)
        ratio = min(t_val / t_max, 1.0)
        return ColorRGBA(
            r=float(ratio), g=float(0.2 * (1.0 - ratio)),
            b=float(1.0 - ratio), a=0.8,
        )

    def _publish_line_list(self):
        height, width = self.grad_x.shape
        step = self.viz_subsample
        arrow_len = self.viz_arrow_length
        tt = np.array(self.travel_time, dtype=np.float64)
        t_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vector_field"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(sec=0, nanosec=0)

        points = []
        colors = []
        for row in range(0, height, step):
            for col in range(0, width, step):
                vx = self.grad_x[row, col]
                vy = self.grad_y[row, col]
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                if abs(vx) < 1e-8 and abs(vy) < 1e-8:
                    continue
                t_val = tt[row, col]
                wx, wy = self._grid_to_world(col, row)
                color = self._cost_to_color(t_val, t_max)
                points.append(Point(x=wx, y=wy, z=0.05))
                points.append(Point(x=wx + arrow_len * vx, y=wy + arrow_len * vy, z=0.05))
                colors.append(color)
                colors.append(color)

        marker.points = points
        marker.colors = colors
        self.lines_pub.publish(marker)

    def _publish_cost_to_go_grid(self):
        if self.costmap_msg is None or self.travel_time is None:
            return
        # Normalise by free-space max, not tt.max(). The EDT obstacle fill makes
        # tt.max() ~400x larger, which would compress the useful gradient range
        # into 1% of the colormap.
        free_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0
        tt = np.array(self.travel_time, dtype=np.float64)
        grid_data = np.clip(tt / free_max * 99.0, 0.0, 100.0).astype(np.int8).flatten().tolist()

        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.costmap_msg.info
        msg.data = grid_data
        self.cost_to_go_pub.publish(msg)

    def _publish_planner_data(self):
        """Pack field data for the NMPC planner node.

        Layout (all float32): [height, width, origin_x, origin_y, resolution,
        travel_time (H*W), grad_x (H*W), grad_y (H*W)]
        """
        if self.costmap_msg is None or self.travel_time is None:
            return

        info = self.costmap_msg.info
        h, w = self.grad_x.shape
        tt = np.array(self.travel_time, dtype=np.float32)
        header = np.array(
            [h, w, info.origin.position.x, info.origin.position.y, info.resolution],
            dtype=np.float32,
        )

        msg = Float32MultiArray()
        msg.data = np.concatenate([
            header,
            tt.ravel(),
            self.grad_x.astype(np.float32).ravel(),
            self.grad_y.astype(np.float32).ravel(),
        ]).tolist()
        self.planner_data_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FMMVectorFieldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
