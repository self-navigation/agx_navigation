#!/usr/bin/env python3
"""
FMM Vector Field Planner with Local Patch Updates - ROS2 Jazzy / Nav2

Extends the base FMM vector field planner with fast local costmap integration.

Architecture:
  GLOBAL path (slow, ~0.5-2 Hz):
    Global costmap update -> full FMM re-solve -> T_global, grad_global

  LOCAL path (fast, ~5-10 Hz):
    Local costmap update -> extract patch around robot -> merge local
    obstacles into speed function -> fast-sweep re-solve on patch only
    (boundary pinned to T_global) -> overwrite gradients in patch region

The Fast Sweeping Method (FSM) is used for the local patch because it
natively supports fixed Dirichlet boundary conditions - each boundary
cell keeps its T_global value while interior cells are re-solved with
the updated speed function.  FSM converges in O(n) for a small convex
domain and typically needs only 2-4 full sweeps.

Subscribes:
  /global_costmap/costmap        (nav_msgs/OccupancyGrid)
  /local_costmap/costmap         (nav_msgs/OccupancyGrid)
  /goal_pose                     (geometry_msgs/PoseStamped)

Publishes:
  /vector_field/lines            (visualization_msgs/Marker)
  /vector_field/cost_to_go       (nav_msgs/OccupancyGrid)

Robot pose: TF  map -> base_link

Depends: pip install scikit-fmm
"""

import math
import time
from typing import Optional

import numpy as np
import skfmm

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException


# ---------------------------------------------------------------------------
#  Fast Sweeping Method - local Eikonal solver with fixed boundary values
# ---------------------------------------------------------------------------

def fast_sweep_eikonal(
    T: np.ndarray,
    speed: np.ndarray,
    fixed: np.ndarray,
    dx: float,
    max_sweeps: int = 8,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve the Eikonal equation |∇T| = 1/F on a small 2D grid using
    the Fast Sweeping Method with fixed (Dirichlet) boundary cells.

    This is the Zhao (2005) algorithm: sweep in all 4 diagonal directions,
    updating each non-fixed cell from its two axis-aligned neighbours.
    Convergence is guaranteed and typically takes 2-4 sweeps for a
    convex domain.

    Parameters
    ----------
    T : (H, W) float64
        Initial travel-time values.  Fixed cells must already contain
        their correct T values; interior cells should be initialized to
        a large value (e.g. np.inf or the global T value).
    speed : (H, W) float64
        Speed function F > 0.  Cells with F <= 0 are treated as walls
        (never updated).
    fixed : (H, W) bool
        True for cells whose T value must not change (boundary cells
        and obstacle cells).
    dx : float
        Grid spacing in meters (same in both axes).
    max_sweeps : int
        Safety cap on the number of full 4-directional sweeps.
    tol : float
        Convergence threshold on max absolute change per sweep.

    Returns
    -------
    T : (H, W) float64  - updated in-place and returned for convenience.
    """
    H, W = T.shape

    for sweep in range(max_sweeps):
        max_change = 0.0

        # Four diagonal sweep orders cover all characteristic directions.
        for row_order in [range(H), range(H - 1, -1, -1)]:
            for col_order in [range(W), range(W - 1, -1, -1)]:
                for r in row_order:
                    for c in col_order:
                        if fixed[r, c] or speed[r, c] <= 0.0:
                            continue

                        # Smallest neighbor along each axis
                        t_x = min(
                            T[r, c - 1] if c > 0 else np.inf,
                            T[r, c + 1] if c < W - 1 else np.inf,
                        )
                        t_y = min(
                            T[r - 1, c] if r > 0 else np.inf,
                            T[r + 1, c] if r < H - 1 else np.inf,
                        )

                        # Solve the 2D Eikonal update:
                        #   (T - t_x)^2 + (T - t_y)^2 = (dx/F)^2
                        slowness = dx / speed[r, c]

                        # Sort so t_a <= t_b
                        t_a, t_b = (t_x, t_y) if t_x <= t_y else (t_y, t_x)

                        # Try 2D update first
                        t_new = t_a + slowness
                        if t_new > t_b:
                            # Two-sided quadratic:
                            # 2*T^2 - 2*(t_a+t_b)*T + (t_a^2+t_b^2 - s^2) = 0
                            disc = (
                                2.0 * slowness * slowness
                                - (t_a - t_b) ** 2
                            )
                            if disc >= 0.0:
                                t_new = (t_a + t_b + math.sqrt(disc)) / 2.0

                        # Only accept if it decreases T (causality)
                        if t_new < T[r, c]:
                            max_change = max(max_change, T[r, c] - t_new)
                            T[r, c] = t_new

        if max_change < tol:
            break

    return T


# Optional: Numba-accelerated version for real-time performance.
# If numba is available, the inner loop runs ~50-100x faster.
try:
    from numba import njit

    @njit(cache=True)
    def _fast_sweep_inner(T, speed, fixed, dx, max_sweeps, tol):
        """Numba-compiled fast sweeping inner loop."""
        H, W = T.shape
        for sweep in range(max_sweeps):
            max_change = 0.0
            # Sweep 1: top-left to bottom-right
            for r in range(H):
                for c in range(W):
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
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y)
                    t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            # Sweep 2: top-right to bottom-left
            for r in range(H):
                for c in range(W - 1, -1, -1):
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
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y)
                    t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            # Sweep 3: bottom-left to top-right
            for r in range(H - 1, -1, -1):
                for c in range(W):
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
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y)
                    t_b = max(t_x, t_y)
                    t_new = t_a + s
                    if t_new > t_b:
                        disc = 2.0 * s * s - (t_a - t_b) ** 2
                        if disc >= 0.0:
                            t_new = (t_a + t_b + math.sqrt(disc)) / 2.0
                    if t_new < T[r, c]:
                        max_change = max(max_change, T[r, c] - t_new)
                        T[r, c] = t_new
            # Sweep 4: bottom-right to top-left
            for r in range(H - 1, -1, -1):
                for c in range(W - 1, -1, -1):
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
                    s = dx / speed[r, c]
                    t_a = min(t_x, t_y)
                    t_b = max(t_x, t_y)
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
        """Wrapper that calls the numba-compiled version."""
        return _fast_sweep_inner(T, speed, fixed, dx, max_sweeps, tol)

    _USE_NUMBA = True

except ImportError:
    _USE_NUMBA = False


def eikonal_local_solve(T, speed, fixed, dx, max_sweeps=8, tol=1e-6):
    """Dispatch to numba version if available, else pure-Python."""
    if _USE_NUMBA:
        return fast_sweep_eikonal_numba(T, speed, fixed, dx, max_sweeps, tol)
    return fast_sweep_eikonal(T, speed, fixed, dx, max_sweeps, tol)


class FMMVectorFieldNode(Node):
    def __init__(self):
        super().__init__("fmm_vector_field")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("inflation_weight", 0.8)
        self.declare_parameter("lethal_cost", 99)
        self.declare_parameter("allow_unknown", False)
        self.declare_parameter("viz_subsample", 4)
        self.declare_parameter("viz_arrow_length", 0.3)
        self.declare_parameter("viz_rate", 5.0)

        # How to merge local costmap cells with global ones.
        # "max" = take the worse (higher) cost from either costmap.
        # "replace" = local costmap fully overrides the global in its footprint.
        self.declare_parameter("local_merge_mode", "max")

        # Padding (in cells) around the local costmap footprint to include
        # in the patch re-solve.  Larger = smoother blending with the global
        # field, but slower.  1-2 cells is usually enough since the boundary
        # ring is pinned to T_global anyway.
        self.declare_parameter("local_patch_padding", 2)

        # Maximum number of fast-sweeping iterations for the local patch.
        self.declare_parameter("local_max_sweeps", 6)

        self.map_frame: str = self.get_parameter("map_frame").value
        self.robot_frame: str = self.get_parameter("robot_frame").value
        self.inflation_weight: float = self.get_parameter("inflation_weight").value
        self.lethal_cost: int = self.get_parameter("lethal_cost").value
        self.allow_unknown: bool = self.get_parameter("allow_unknown").value
        self.viz_subsample: int = self.get_parameter("viz_subsample").value
        self.viz_arrow_length: float = self.get_parameter("viz_arrow_length").value
        viz_rate: float = self.get_parameter("viz_rate").value
        self.local_merge_mode: str = self.get_parameter("local_merge_mode").value
        self.local_patch_padding: int = self.get_parameter("local_patch_padding").value
        self.local_max_sweeps: int = self.get_parameter("local_max_sweeps").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.costmap_msg: Optional[OccupancyGrid] = None
        self.costmap_array: Optional[np.ndarray] = None
        self.current_goal: Optional[PoseStamped] = None

        # The "master" copies - global FMM writes these, local patch
        # overwrites a sub-region.  Visualization and query_vector()
        # always read from these.
        self.travel_time: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_y: Optional[np.ndarray] = None

        # Stashed pristine global solve (never touched by local patches)
        # so we can re-patch from clean state on each local update.
        self._global_travel_time: Optional[np.ndarray] = None
        self._global_grad_x: Optional[np.ndarray] = None
        self._global_grad_y: Optional[np.ndarray] = None
        self._global_speed: Optional[np.ndarray] = None  # speed function

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
            OccupancyGrid,
            "/global_costmap/costmap",
            self._costmap_cb,
            costmap_qos,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_cb, 10
        )

        # Local costmap - typically published at sensor rate with
        # RELIABLE / VOLATILE QoS (Nav2 default for local costmap).
        local_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid,
            "/local_costmap/costmap",
            self._local_costmap_cb,
            local_qos,
        )

        self.lines_pub = self.create_publisher(Marker, "/vector_field/lines", 10)
        self.cost_to_go_pub = self.create_publisher(
            OccupancyGrid, "/vector_field/cost_to_go", 10
        )

        self.viz_timer = self.create_timer(1.0 / viz_rate, self._viz_timer_cb)

        self.get_logger().info(
            f"FMMVectorFieldNode ready (numba={'yes' if _USE_NUMBA else 'no'}).  "
            f"Waiting for costmap + goal..."
        )

    def _costmap_cb(self, msg: OccupancyGrid):
        """Global costmap update - triggers a full FMM re-solve."""
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
        """Local costmap update - triggers a fast patch re-solve."""
        self._local_costmap_msg = msg
        self._local_costmap_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self._local_dirty = True
        # Run the local patch immediately (it's fast)
        self._maybe_patch_local()

    def _viz_timer_cb(self):
        self._maybe_recompute_global()
        self._publish_visualization()

    def _world_to_grid(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        """World (m) -> global grid (col, row).  None if out of bounds."""
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

    def _local_footprint_in_global(
        self,
    ) -> Optional[tuple[int, int, int, int]]:
        """Compute the bounding box of the local costmap in global grid
        coordinates, with padding.

        Returns (row_min, row_max, col_min, col_max) clipped to the
        global grid, or None if no overlap.
        """
        if self._local_costmap_msg is None or self.costmap_msg is None:
            return None

        g_info = self.costmap_msg.info
        l_info = self._local_costmap_msg.info

        # Local costmap origin in global grid coordinates
        col_start = int(
            (l_info.origin.position.x - g_info.origin.position.x)
            / g_info.resolution
        )
        row_start = int(
            (l_info.origin.position.y - g_info.origin.position.y)
            / g_info.resolution
        )

        # Account for possible resolution mismatch: scale local dimensions
        # into global-cell counts.
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
        """Convert int8 costmap values to (speed, mask) arrays.

        Returns
        -------
        speed : float64 array, values in [0.01, 1.0]
        mask  : bool array, True = impassable
        """
        fraw = raw.astype(np.float64)
        mask = fraw >= self.lethal_cost
        if not self.allow_unknown:
            mask |= fraw == -1

        clamped = np.clip(fraw, 0.0, self.lethal_cost - 1)
        normalized = clamped / (self.lethal_cost - 1)
        speed = np.clip(1.0 - self.inflation_weight * normalized, 0.01, 1.0)
        return speed, mask

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

        # Seed the goal
        phi = np.ones((height, width), dtype=np.float64)
        phi[gr, gc] = -1.0
        phi = np.ma.MaskedArray(phi, mask=mask)

        t0 = time.monotonic()
        try:
            travel_time = skfmm.travel_time(phi, speed_masked, dx=resolution)
        except Exception as e:
            self.get_logger().error(f"FMM failed: {e}")
            return
        dt = time.monotonic() - t0

        # Convert to plain array, NaN for unreachable
        tt = np.array(travel_time, dtype=np.float64)
        if np.ma.is_masked(travel_time):
            tt[travel_time.mask] = np.nan

        # Gradient -> vector field (negate to point toward goal)
        d_row, d_col = np.gradient(tt, resolution)
        gx = -d_col
        gy = -d_row

        # Clean up NaN / inf in gradients
        bad = np.isnan(tt) | np.isnan(gx) | np.isnan(gy) | np.isinf(gx) | np.isinf(gy)
        gx[bad] = 0.0
        gy[bad] = 0.0

        # Normalize to unit vectors
        mag = np.sqrt(gx**2 + gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        gx /= safe_mag
        gy /= safe_mag

        # Stash pristine global copies (local patches read from these)
        self._global_travel_time = tt.copy()
        self._global_grad_x = gx.copy()
        self._global_grad_y = gy.copy()
        self._global_speed = speed.copy()

        # Set the "live" arrays that visualization and query read from
        self.travel_time = tt
        self.grad_x = gx
        self.grad_y = gy

        self.field_dirty = False
        self.get_logger().info(
            f"Global FMM: {width}x{height} in {dt:.3f}s "
            f"(max T = {np.nanmax(tt):.1f})"
        )

        # If we already have a local costmap, immediately re-patch
        if self._local_costmap_array is not None:
            self._local_dirty = True
            self._maybe_patch_local()

    def _maybe_recompute_global(self):
        if self.field_dirty:
            self._compute_global_field()

    def _maybe_patch_local(self):
        """Re-solve a local patch of the Eikonal field using the local
        costmap, pinning boundary cells to the global travel-time."""
        if not self._local_dirty:
            return
        if self._global_travel_time is None:
            return  # No global solve yet - nothing to patch

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

        # 1. Start from the pristine global T in this patch region.
        #    We always patch from the clean global state so that
        #    obstacles that disappeared in the local costmap are
        #    correctly "un-blocked" rather than persisting from a
        #    previous local patch.
        T_patch = self._global_travel_time[r_min:r_max, c_min:c_max].copy()
        speed_patch = self._global_speed[r_min:r_max, c_min:c_max].copy()

        # 2. Merge local costmap into the speed function for this patch.
        #    We iterate over local costmap cells, map them into the
        #    global grid, and update the speed function.
        l_h, l_w = self._local_costmap_array.shape

        for lr in range(l_h):
            for lc in range(l_w):
                # Map local cell -> global grid coordinates
                world_x = l_info.origin.position.x + (lc + 0.5) * l_info.resolution
                world_y = l_info.origin.position.y + (lr + 0.5) * l_info.resolution
                gc = int((world_x - g_info.origin.position.x) / g_info.resolution)
                gr = int((world_y - g_info.origin.position.y) / g_info.resolution)

                # Map to patch-local coordinates
                pr = gr - r_min
                pc = gc - c_min
                if not (0 <= pr < patch_h and 0 <= pc < patch_w):
                    continue

                local_val = int(self._local_costmap_array[lr, lc])
                local_speed, local_mask = self._cost_to_speed(
                    np.array([[local_val]], dtype=np.int8)
                )

                if self.local_merge_mode == "max":
                    # Take the worse (slower) speed from either costmap
                    speed_patch[pr, pc] = min(speed_patch[pr, pc], local_speed[0, 0])
                else:
                    # "replace" - local fully overrides
                    speed_patch[pr, pc] = local_speed[0, 0]

                # If local says lethal, mark as wall
                if local_mask[0, 0]:
                    speed_patch[pr, pc] = 0.0
                    T_patch[pr, pc] = np.inf

        # 3. Build the fixed-cell mask for fast sweeping.
        #    Fixed cells = boundary ring + walls + unreachable.
        fixed = np.zeros((patch_h, patch_w), dtype=np.bool_)

        # Boundary ring: first/last row and column are pinned to T_global
        fixed[0, :] = True
        fixed[-1, :] = True
        fixed[:, 0] = True
        fixed[:, -1] = True

        # Walls (speed == 0) are also fixed (they stay at inf)
        fixed[speed_patch <= 0.0] = True

        # Cells with NaN T (unreachable in global solve) stay fixed
        nan_mask = np.isnan(T_patch)
        T_patch[nan_mask] = np.inf
        fixed[nan_mask] = True

        # 4. Run fast sweeping on the patch.
        T_patch = eikonal_local_solve(
            T_patch, speed_patch, fixed, resolution,
            max_sweeps=self.local_max_sweeps,
        )

        # 5. Recompute gradients for the patch interior (excluding the
        #    1-cell boundary ring, where np.gradient would look outside
        #    the patch).
        # Temporarily put inf back to NaN for gradient computation
        T_for_grad = T_patch.copy()
        T_for_grad[T_patch >= 1e18] = np.nan

        d_row, d_col = np.gradient(T_for_grad, resolution)
        local_gx = -d_col
        local_gy = -d_row

        bad = np.isnan(local_gx) | np.isnan(local_gy) | np.isinf(local_gx) | np.isinf(local_gy)
        local_gx[bad] = 0.0
        local_gy[bad] = 0.0

        mag = np.sqrt(local_gx**2 + local_gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        local_gx /= safe_mag
        local_gy /= safe_mag

        # 6. Write results back into the live arrays.
        #    We skip the outermost ring (it's pinned to global values
        #    anyway, and np.gradient is unreliable at edges).
        m = 1  # margin: don't overwrite the boundary ring
        self.travel_time[r_min + m : r_max - m, c_min + m : c_max - m] = (
            T_patch[m:-m, m:-m]
        )
        # Restore NaN for truly unreachable cells
        inf_cells = self.travel_time >= 1e18
        self.travel_time[inf_cells] = np.nan

        self.grad_x[r_min + m : r_max - m, c_min + m : c_max - m] = (
            local_gx[m:-m, m:-m]
        )
        self.grad_y[r_min + m : r_max - m, c_min + m : c_max - m] = (
            local_gy[m:-m, m:-m]
        )

        self._local_dirty = False
        dt = time.monotonic() - t0
        self.get_logger().debug(
            f"Local patch: {patch_w}x{patch_h} in {dt*1000:.1f}ms",
            throttle_duration_sec=2.0,
        )

    def query_vector(
        self, wx: float, wy: float
    ) -> Optional[tuple[float, float, float]]:
        """Look up the vector field at a world coordinate.

        Returns (vx, vy, cost_to_go) or None if out of bounds / unreachable.
        Uses bilinear interpolation for smooth lookup between cells.
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

        tt = self.travel_time
        for r, c in [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]:
            if np.isnan(tt[r, c]):
                return None

        return (_bilerp(self.grad_x), _bilerp(self.grad_y), _bilerp(tt))

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
        t_max = float(np.nanmax(tt)) if not np.all(np.isnan(tt)) else 1.0

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
                if not math.isfinite(t_val):
                    continue

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
        tt = np.array(self.travel_time, dtype=np.float64)
        t_max = float(np.nanmax(tt))
        if t_max < 1e-8:
            return
        normalized = tt / t_max * 100.0
        normalized = np.where(np.isnan(normalized), -1.0, normalized)
        grid_data = normalized.astype(np.int8).flatten().tolist()

        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.costmap_msg.info
        msg.data = grid_data
        self.cost_to_go_pub.publish(msg)


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
