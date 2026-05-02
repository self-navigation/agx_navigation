"""Fast Marching Square (FM2) vector field generator.

This node publishes the same Float32MultiArray layout as the previous
implementation -- [h, w, origin_x, origin_y, resolution, T(H*W), gx(H*W),
gy(H*W)] -- so the NMPC planner is unchanged. Internally it follows
Garrido-Moreno-Abderrahim-Blanco (2007): the obstacle-distance field is
folded into the wave-propagation speed of a single eikonal solve, instead
of being added on top as a separate repulsive potential. There is no
obstacle fill, no gradient-domination tuning, and no additive term.

See vector_field_fm2_notes.md for the design rationale.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import skfmm
from scipy.ndimage import distance_transform_edt, gaussian_filter

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException

# ---------------------------------------------------------------------------
# Speed profile (EDT -> v(x))
# ---------------------------------------------------------------------------


@dataclass
class SpeedConfig:
    inflation_radius: float  # [m]; clearance band beyond which v = v_max
    v_max: float  # speed in open space (>= R from any obstacle)
    v_min: float  # floor at the wall surface (must be > 0)
    profile: str  # "linear" | "exponential"
    decay_rate: float  # only used by "exponential"


def build_speed_field(
    edt_free: np.ndarray,
    obstacle_mask: np.ndarray,
    cfg: SpeedConfig,
) -> np.ndarray:
    """Map EDT distance to wave speed v(x) used by the second eikonal solve.

    "Linear" (recommended default) gives a sharp clearance band: v is at
    its floor on the wall, ramps linearly to v_max at d = R, and equals
    v_max past R. "Exponential" matches the previous code's profile and
    decays smoothly throughout the inflation zone, attracting paths to
    the centreline of corridors even in wide spaces.

    The v_min floor is essential: the eikonal solution diverges as v -> 0,
    and skfmm clips internally anyway, so making the clip explicit here
    keeps behaviour predictable and the cut-locus pattern consistent
    across solver implementations.
    """
    R = cfg.inflation_radius
    v = np.full_like(edt_free, cfg.v_max, dtype=np.float64)
    if R <= 0.0:
        return v

    near = ~obstacle_mask & (edt_free < R)
    if not near.any():
        return v

    norm = edt_free[near] / R  # 0 at wall, 1 at boundary
    if cfg.profile == "exponential":
        # Reproduces the previous node's behaviour: v(0) = exp(-k), v(R) = 1.
        v[near] = np.clip(
            cfg.v_max * np.exp(-cfg.decay_rate * (1.0 - norm)),
            cfg.v_min,
            cfg.v_max,
        )
    else:
        # Linear: sharp band, v(0) = v_min, v(R) = v_max.
        v[near] = np.clip(
            cfg.v_min + (cfg.v_max - cfg.v_min) * norm,
            cfg.v_min,
            cfg.v_max,
        )
    return v


# ---------------------------------------------------------------------------
# Eikonal solver
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Gradient and field assembly
# ---------------------------------------------------------------------------


def field_from_T(
    tt: np.ndarray,
    obstacle_mask: np.ndarray,
    resolution: float,
    smooth_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate (optionally smoothed) T to produce a unit vector field.

    The optional Gaussian smoothing is applied to T BEFORE the gradient
    operator. This is the cut-locus fix: smoothing T preserves the
    scalar-potential structure of the field, while smoothing the gradient
    components afterwards (as the previous node did) does not -- it
    introduces curl and creates regions where the renormalised direction
    is unstable.

    Obstacle cells (NaN in T) are substituted with a large finite value
    before differentiation. The central-difference gradient at wall-
    adjacent free cells then points outward, providing a valid recovery
    direction if the robot ever enters one. Leaving NaN in T would
    propagate NaN into the gradient at every wall-adjacent free cell.

    Returns:
        gx, gy : unit vector field, components ready for publishing.
                 Zeroed at cells where the gradient is undefined.
        mag    : magnitude of -grad(T_smooth) BEFORE renormalisation,
                 useful for diagnostics and arrow scaling.
    """
    if smooth_sigma_m > 0.0:
        sigma_cells = smooth_sigma_m / resolution
        tt_for_grad = _gaussian_with_nan(tt, sigma_cells)
    else:
        tt_for_grad = tt.copy()

    # Substitute a large value at obstacle cells. This gives a clean
    # outward-pointing gradient at wall-adjacent free cells.
    finite = tt_for_grad[np.isfinite(tt_for_grad)]
    big = float(finite.max()) * 4.0 + 1.0 if finite.size else 1.0
    tt_diff = np.where(obstacle_mask, big, tt_for_grad)

    # np.gradient on tt_diff. Obstacle cells now hold a large value so
    # the gradient at wall-adjacent free cells points outward. Any
    # remaining NaN in tt_diff (none expected after the substitution
    # above, but defensive) propagates through the central-difference
    # stencil and is zeroed below.
    d_row, d_col = np.gradient(tt_diff, resolution)
    raw_gx = -d_col
    raw_gy = -d_row

    mag = np.sqrt(raw_gx * raw_gx + raw_gy * raw_gy)
    safe = np.where(mag > 1e-8, mag, 1.0)
    gx = raw_gx / safe
    gy = raw_gy / safe

    # Defensive: cells with non-finite gradient -> zero direction. The
    # planner treats (0, 0) as "off-field, brake".
    bad = ~np.isfinite(gx) | ~np.isfinite(gy)
    gx[bad] = 0.0
    gy[bad] = 0.0
    mag[bad] = np.nan
    return gx, gy, mag


def _gaussian_with_nan(arr: np.ndarray, sigma_cells: float) -> np.ndarray:
    """Gaussian-smooth a field that contains NaN by normalising the kernel.

    Standard Gaussian filtering propagates NaN to every cell within the
    kernel radius. Replacing NaN with 0 first and dividing by the smoothed
    validity mask gives a NaN-aware smoothing that preserves T values near
    the obstacle boundary.
    """
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(valid > 0, arr, 0.0)
    num = gaussian_filter(filled, sigma_cells)
    den = gaussian_filter(valid, sigma_cells)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    # Restore NaN where the original was NaN (smoothing inside is fine,
    # but cells that started as NaN should stay NaN to mark unreachable).
    out[~np.isfinite(arr)] = np.nan
    return out


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


class FMMVectorFieldNode(Node):

    def __init__(self):
        super().__init__("fmm_vector_field")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("occupancy_threshold", 65)
        self.declare_parameter("allow_unknown", False)

        # FM2 speed-field parameters. inflation_radius and v_min set the
        # clearance band; v_max sets open-space speed and is normally 1.0.
        # The "linear" profile gives a crisp band useful for NMPC tracking;
        # "exponential" reproduces the previous node's behaviour and pulls
        # paths toward corridor centrelines even in wide spaces.
        self.declare_parameter("inflation_radius", 0.5)  # [m]
        self.declare_parameter("speed_v_min", 0.1)  # [m/s] (relative)
        self.declare_parameter("speed_v_max", 1.0)  # [m/s] (relative)
        self.declare_parameter("speed_profile", "linear")
        self.declare_parameter("speed_decay_rate", 2.5)  # exp profile only

        # Cut-locus fix: smooth T (the scalar) BEFORE differentiation.
        # This is independent of FM2 itself and addresses the FMM ridge
        # problem where np.gradient averages the two sides of a shock,
        # producing a tiny vector that becomes an unstable direction
        # after renormalisation.  Set sigma to 0 to disable.
        self.declare_parameter("smooth_T_before_grad", True)
        self.declare_parameter("smooth_T_sigma", 0.10)  # [m]

        # Visualisation.
        self.declare_parameter("viz_subsample", 4)
        self.declare_parameter("viz_arrow_length", 0.3)  # max arrow length [m]
        self.declare_parameter("viz_scale_arrows", True)  # scale by |grad T|
        self.declare_parameter("viz_path_step", 0.5)  # [cells]; <1 sub-cell
        self.declare_parameter("viz_path_max_iter", 2000)
        self.declare_parameter("viz_rate", 5.0)

        self._cache_params()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_array: Optional[np.ndarray] = None
        self.current_goal: Optional[PoseStamped] = None
        self.field_dirty = False

        self.travel_time: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_y: Optional[np.ndarray] = None
        self.grad_mag: Optional[np.ndarray] = None  # pre-norm |grad T|, for viz
        self._free_max_T: float = 1.0

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.create_subscription(PoseStamped, "/goal_pose", self._goal_cb, 10)

        self.lines_pub = self.create_publisher(Marker, "/vector_field/lines", 10)
        self.path_pub = self.create_publisher(Path, "/vector_field/optimal_path", 10)
        self.cost_to_go_pub = self.create_publisher(
            OccupancyGrid, "/vector_field/cost_to_go", 10
        )
        self.planner_data_pub = self.create_publisher(
            Float32MultiArray, "/vector_field/planner_data", 1
        )

        self.viz_timer = self.create_timer(1.0 / self.viz_rate, self._viz_timer_cb)

        # NOTE on caching. EDT and the speed field depend only on the
        # static obstacle map; they could be cached and reused across
        # cycles when /map has not changed. We deliberately do not cache
        # because the robot operates in unknown environments where /map
        # is updated frequently. To enable caching, hash the obstacle
        # mask in _map_cb and reuse the EDT and speed arrays in
        # _recompute_field whenever the hash matches.

        self.get_logger().info(
            "FMMVectorFieldNode (FM2) ready. Waiting for /map and /goal_pose..."
        )

    # ------------------------------------------------------------------
    # Parameter caching
    # ------------------------------------------------------------------

    def _cache_params(self):
        gp = self.get_parameter
        self.map_frame = gp("map_frame").value
        self.robot_frame = gp("robot_frame").value
        self.occupancy_threshold = gp("occupancy_threshold").value
        self.allow_unknown = gp("allow_unknown").value

        self.speed_cfg = SpeedConfig(
            inflation_radius=gp("inflation_radius").value,
            v_min=gp("speed_v_min").value,
            v_max=gp("speed_v_max").value,
            profile=gp("speed_profile").value,
            decay_rate=gp("speed_decay_rate").value,
        )
        if self.speed_cfg.profile not in ("linear", "exponential"):
            self.get_logger().warn(
                f"Unknown speed_profile '{self.speed_cfg.profile}', falling back to 'linear'."
            )
            self.speed_cfg.profile = "linear"
        if self.speed_cfg.v_min <= 0.0:
            raise ValueError("speed_v_min must be > 0 (eikonal diverges as v -> 0)")

        self.smooth_T = gp("smooth_T_before_grad").value
        self.smooth_T_sigma = gp("smooth_T_sigma").value

        self.viz_subsample = int(gp("viz_subsample").value)
        self.viz_arrow_length = float(gp("viz_arrow_length").value)
        self.viz_scale_arrows = gp("viz_scale_arrows").value
        self.viz_path_step = float(gp("viz_path_step").value)
        self.viz_path_max_iter = int(gp("viz_path_max_iter").value)
        self.viz_rate = float(gp("viz_rate").value)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.field_dirty = True
        self.get_logger().info(
            f"Map received: {msg.info.width}x{msg.info.height} "
            f"res={msg.info.resolution:.3f} m/cell",
            throttle_duration_sec=10.0,
        )
        if self.current_goal is not None:
            self._recompute_field()

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = msg
        self.field_dirty = True
        self.get_logger().info(
            f"Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )
        self._recompute_field()

    def _viz_timer_cb(self):
        if self.field_dirty:
            self._recompute_field()
        self._publish_visualization()

    # ------------------------------------------------------------------
    # World <-> grid
    # ------------------------------------------------------------------

    def _world_to_grid(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        info = self.map_msg.info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if 0 <= col < info.width and 0 <= row < info.height:
            return col, row
        return None

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        info = self.map_msg.info
        return (
            info.origin.position.x + (col + 0.5) * info.resolution,
            info.origin.position.y + (row + 0.5) * info.resolution,
        )

    # ------------------------------------------------------------------
    # Field recomputation (the core FM2 pipeline)
    # ------------------------------------------------------------------

    def _recompute_field(self):
        """FM2 pipeline: EDT -> speed -> single eikonal solve -> grad T.

        No additive repulsion, no obstacle fill, no gradient smoothing.
        The optional smoothing happens on T (scalar) before differentiation
        when smooth_T_before_grad is True. Compare against False to see
        the cut-locus instabilities the smoothing fixes.
        """
        if self.map_msg is None or self.current_goal is None:
            return

        goal = self._world_to_grid(
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
        )
        if goal is None:
            self.get_logger().error("Goal is outside map bounds.")
            return
        goal_col, goal_row = goal

        resolution = self.map_msg.info.resolution
        raw = self.map_array

        obstacle_mask = raw >= self.occupancy_threshold
        if not self.allow_unknown:
            obstacle_mask = obstacle_mask | (raw < 0)

        if obstacle_mask[goal_row, goal_col]:
            self.get_logger().error("Goal cell is inside an obstacle.")
            return

        t0 = time.monotonic()

        # Step 1 of FM2: distance to nearest obstacle [m]. The "first FMM"
        # of the literature is mathematically equivalent to the EDT (see
        # vector_field_fm2_notes.md sec. "Constant-speed eikonal = EDT");
        # we compute it directly via scipy for an order-of-magnitude
        # speedup over running FMM with constant speed.
        edt_free = distance_transform_edt(~obstacle_mask) * resolution

        # Step 2 of FM2: build the slowness/speed field from the EDT.
        speed = build_speed_field(edt_free, obstacle_mask, self.speed_cfg)

        # Step 3 of FM2: single eikonal solve in the slowness medium.
        # The wave-speed encoding handles wall avoidance; no separate
        # repulsive potential is added.
        tt = solve_eikonal_full(
            obstacle_mask,
            speed,
            goal_col,
            goal_row,
            resolution,
        )

        free_tt = tt[np.isfinite(tt)]
        self._free_max_T = float(free_tt.max()) if free_tt.size else 1.0

        # Step 4 (cut-locus fix, optional): smooth T then differentiate.
        sigma = self.smooth_T_sigma if self.smooth_T else 0.0
        gx, gy, mag = field_from_T(tt, obstacle_mask, resolution, sigma)

        self.travel_time = tt
        self.grad_x = gx
        self.grad_y = gy
        self.grad_mag = mag
        self.field_dirty = False

        h, w = obstacle_mask.shape
        n_finite = int(np.isfinite(tt).sum())
        self.get_logger().info(
            f"Field recomputed: {w}x{h} cells, {n_finite} reached "
            f"({100.0*n_finite/(h*w):.1f}%), {(time.monotonic()-t0)*1000:.1f} ms "
            f"[smooth_T={self.smooth_T}, "
            f"profile={self.speed_cfg.profile}, R={self.speed_cfg.inflation_radius:.2f} m]"
        )

    # ------------------------------------------------------------------
    # Public query interface
    # ------------------------------------------------------------------

    def query_vector(
        self, wx: float, wy: float
    ) -> Optional[tuple[float, float, float]]:
        """Bilinear interp of (vx, vy, T) at world (wx, wy)."""
        if self.map_msg is None or self.grad_x is None or self.travel_time is None:
            return None

        info = self.map_msg.info
        gx = (wx - info.origin.position.x) / info.resolution - 0.5
        gy = (wy - info.origin.position.y) / info.resolution - 0.5

        h, w = self.grad_x.shape
        if not (0 <= gx < w - 1 and 0 <= gy < h - 1):
            return None

        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0

        def _bilerp(arr: np.ndarray) -> float:
            v00 = arr[y0, x0]
            v01 = arr[y0, x0 + 1]
            v10 = arr[y0 + 1, x0]
            v11 = arr[y0 + 1, x0 + 1]
            return float(
                v00 * (1.0 - fx) * (1.0 - fy)
                + v01 * fx * (1.0 - fy)
                + v10 * (1.0 - fx) * fy
                + v11 * fx * fy
            )

        return _bilerp(self.grad_x), _bilerp(self.grad_y), _bilerp(self.travel_time)

    def get_robot_pose(self) -> Optional[PoseStamped]:
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
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

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _publish_visualization(self):
        if self.travel_time is None or self.grad_x is None:
            return
        self._publish_arrows()
        self._publish_optimal_path()
        self._publish_cost_to_go_grid()
        self._publish_planner_data()

    def _cost_to_color(self, t_val: float, t_max: float) -> ColorRGBA:
        if not math.isfinite(t_val) or t_max < 1e-8:
            return ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)
        ratio = min(max(t_val / t_max, 0.0), 1.0)
        return ColorRGBA(
            r=float(ratio),
            g=float(0.2 * (1.0 - ratio)),
            b=float(1.0 - ratio),
            a=0.85,
        )

    def _publish_arrows(self):
        """Arrows whose length encodes |grad T| (pre-renormalisation).

        Cells on the FMM cut locus or in the goal neighbourhood produce
        small |grad T|, so their arrows are short -- this makes the cut
        locus visible at a glance and is the most informative single
        diagnostic for "where will the planner have trouble". When
        viz_scale_arrows is False the previous fixed-length behaviour
        is used.
        """
        h, w = self.grad_x.shape
        step = self.viz_subsample
        max_len = self.viz_arrow_length
        scale = self.viz_scale_arrows
        t_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0

        # Normalise the visible magnitude range. Use the 95th percentile of
        # finite |grad T| values so a few large outliers (boundary cells)
        # do not compress the rest of the field into invisible stubs.
        if scale and self.grad_mag is not None:
            valid_mag = self.grad_mag[np.isfinite(self.grad_mag)]
            if valid_mag.size > 0:
                mag_ref = float(np.percentile(valid_mag, 95))
            else:
                mag_ref = 1.0
            if mag_ref < 1e-8:
                mag_ref = 1.0
        else:
            mag_ref = 1.0

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

        points: list = []
        colors: list = []
        for row in range(0, h, step):
            for col in range(0, w, step):
                vx = self.grad_x[row, col]
                vy = self.grad_y[row, col]
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                if abs(vx) < 1e-8 and abs(vy) < 1e-8:
                    continue
                if scale and self.grad_mag is not None:
                    raw = float(self.grad_mag[row, col])
                    if not math.isfinite(raw):
                        continue
                    arrow = max_len * min(raw / mag_ref, 1.0)
                else:
                    arrow = max_len

                wx, wy = self._grid_to_world(col, row)
                T_val = float(self.travel_time[row, col])
                color = self._cost_to_color(T_val, t_max)
                points.append(Point(x=wx, y=wy, z=0.05))
                points.append(
                    Point(
                        x=wx + arrow * vx,
                        y=wy + arrow * vy,
                        z=0.05,
                    )
                )
                colors.extend([color, color])

        marker.points = points
        marker.colors = colors
        self.lines_pub.publish(marker)

    def _publish_optimal_path(self):
        """Trace gradient descent from the robot pose to the goal.

        This is the path the planner *would* follow under perfect
        tracking of the published vector field. Useful for spotting:
          - obvious detours through corridors,
          - oscillation across cut loci (path zig-zags),
          - dead-ends caused by an unreachable goal cell.
        """
        if self.travel_time is None:
            self._publish_empty_path()
            return
        pose = self.get_robot_pose()
        if pose is None:
            self._publish_empty_path()
            return

        info = self.map_msg.info
        res = info.resolution
        h, w = self.grad_x.shape
        wx = pose.pose.position.x
        wy = pose.pose.position.y
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        step_world = self.viz_path_step * res
        max_iter = self.viz_path_max_iter
        # Stop within 1.5 cells of the goal to avoid hovering in the
        # near-zero-gradient neighbourhood where the discrete field
        # becomes unreliable.
        stop_radius_sq = (1.5 * res) ** 2

        path = Path()
        path.header.frame_id = self.map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        prev = (wx, wy)
        stalled = 0
        for _ in range(max_iter):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            path.poses.append(ps)

            dx = gx - wx
            dy = gy - wy
            if dx * dx + dy * dy < stop_radius_sq:
                break

            q = self.query_vector(wx, wy)
            if q is None:
                break
            vx, vy, _ = q
            if abs(vx) < 1e-6 and abs(vy) < 1e-6:
                break

            wx += step_world * vx
            wy += step_world * vy

            # Detect stalls: the path has stopped advancing but is not at
            # the goal. Three consecutive non-moves -> abort.
            ddx = wx - prev[0]
            ddy = wy - prev[1]
            if ddx * ddx + ddy * ddy < (0.1 * step_world) ** 2:
                stalled += 1
                if stalled >= 3:
                    break
            else:
                stalled = 0
            prev = (wx, wy)

        self.path_pub.publish(path)

    def _publish_empty_path(self):
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(msg)

    def _publish_cost_to_go_grid(self):
        if self.map_msg is None or self.travel_time is None:
            return
        free_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0
        tt = np.array(self.travel_time, dtype=np.float64)
        # Cells with NaN (obstacles or unreached) saturate at 100.
        ratio = np.where(
            np.isfinite(tt),
            np.clip(tt / free_max * 99.0, 0.0, 99.0),
            100.0,
        )
        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_msg.info
        msg.data = ratio.astype(np.int8).flatten().tolist()
        self.cost_to_go_pub.publish(msg)

    def _publish_planner_data(self):
        """Pack the field for the NMPC.

        Layout (float32, unchanged from the previous node):
          [h, w, origin_x, origin_y, resolution,
           travel_time(H*W), grad_x(H*W), grad_y(H*W)]

        NaN-valued T cells are converted to a large finite value
        (free_max_T * 4) so downstream code can use comparisons without
        special-casing NaN. The grad arrays carry zeros at those cells.
        """
        if self.map_msg is None or self.travel_time is None:
            return

        info = self.map_msg.info
        h, w = self.grad_x.shape
        big_T = self._free_max_T * 4.0 + 1.0
        tt_out = np.where(np.isfinite(self.travel_time), self.travel_time, big_T)

        header = np.array(
            [
                h,
                w,
                info.origin.position.x,
                info.origin.position.y,
                info.resolution,
            ],
            dtype=np.float32,
        )

        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [
                header,
                tt_out.astype(np.float32).ravel(),
                self.grad_x.astype(np.float32).ravel(),
                self.grad_y.astype(np.float32).ravel(),
            ]
        ).tolist()
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
