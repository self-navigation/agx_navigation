"""Fast Marching Square (FM2) vector field generator.

Publishes:
  /vector_field/planner_data    Float32MultiArray with the full field
                                packed as [h, w, ox, oy, res, T(H*W),
                                gx(H*W), gy(H*W), |grad T|(H*W)].

  /vector_field/lines           Marker (LINE_LIST), arrows for RViz.
  /vector_field/optimal_path    Path traced by gradient descent.
  /vector_field/cost_to_go      OccupancyGrid colour map of T.

Pipeline (FM2):
  1. EDT(obstacle_mask) * dx                 -> distance from each
                                                free cell to the
                                                nearest obstacle.
  2. v(d) = clip(...) according to the chosen profile (linear or exp).
  3. skfmm.travel_time with v as wave speed -> T(x).
  4. (Optional) Gaussian-smooth T (cut-locus fix).
  5. -grad(T) -> raw vector field; renormalise to unit length.

The unit field, T, AND the pre-renormalisation gradient magnitude are
all published. The magnitude doubles as a confidence signal for the
planner (low |grad T| means cut locus or near-goal -> field unreliable).
"""

import math
import time
from dataclasses import dataclass, fields, replace
from typing import Any, Optional

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
# Generic dataclass-driven ROS2 parameter loader
# ---------------------------------------------------------------------------


def declare_and_load_dataclass(node: Node, instance: Any, prefix: str = "") -> Any:
    """Declare every dataclass field as a ROS2 parameter and return a new
    instance populated from the parameter values.

    The dataclass instance's current values become the parameter defaults.
    Field types are inferred from the runtime values (Python types of the
    defaults), so the dataclass should use concrete types -- str, float,
    int, bool, list -- rather than typing-module aliases.

    A new instance is returned via dataclasses.replace; the input is left
    unmodified. This is the same idea as Pydantic BaseSettings or hydra's
    config dataclasses, scoped to ROS2's flat parameter API.
    """
    updates: dict = {}
    for f in fields(instance):
        name = prefix + f.name
        default = getattr(instance, f.name)
        # ROS2 declare_parameter infers descriptor type from the default's
        # Python type. The dataclass default has the correct type, so this
        # round-trip is safe.
        node.declare_parameter(name, default)
        updates[f.name] = node.get_parameter(name).value
    return replace(instance, **updates)


# ---------------------------------------------------------------------------
# Speed profile (EDT -> v(x))
# ---------------------------------------------------------------------------


@dataclass
class SpeedConfig:
    # Clearance band: cells with EDT < inflation_radius see reduced speed.
    inflation_radius: float = 0.5  # [m]
    # Speed at the wall surface. Strictly > 0; eikonal diverges as v -> 0.
    speed_v_min: float = 0.1
    # Speed in open space (>= R from any obstacle).
    speed_v_max: float = 1.0
    # "linear" -> sharp band, recommended; "exponential" -> long-tailed,
    # pulls paths to centrelines even in wide spaces.
    speed_profile: str = "linear"
    # Decay rate for the exponential profile only; ignored otherwise.
    speed_decay_rate: float = 2.5


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
    speed: np.ndarray,
    resolution: float,
    smooth_sigma_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Differentiate (optionally smoothed) T to produce a unit vector field.

    Optional smoothing is applied to T BEFORE the gradient operator. This
    is the cut-locus fix: smoothing T preserves the scalar-potential
    structure of the field, while smoothing the gradient components
    afterwards (as the original additive code did) does not -- it
    introduces curl and creates regions where the renormalised direction
    is unstable.

    Obstacle cells (NaN in T) are substituted with a large finite value
    before differentiation. The central-difference gradient at wall-
    adjacent free cells then points outward, providing a valid recovery
    direction if the robot ever enters one.

    Returns:
        gx, gy : unit vector field, components ready for publishing.
                 Zeroed at cells where the gradient is undefined.
        mag    : confidence magnitude = |grad T| * v.

                 In FM2, |grad T| = 1/v in smooth regions, so the raw
                 magnitude varies by v_max/v_min across the inflation
                 band as a NORMAL feature of the field -- not as a
                 signal of unreliability. Multiplying by v cancels this
                 variation: the product is ~1 in every smooth region
                 regardless of profile, and collapses to 0 at FMM cut
                 loci and at the goal where the direction genuinely is
                 unreliable. This is what the planner consumes as its
                 confidence signal.
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

    # Speed-corrected confidence: |grad T| * v ~= 1 in smooth regions of
    # the FM2 field regardless of how v varies across the inflation band.
    # Drops to 0 at cut loci and at the goal where direction is genuinely
    # unreliable. See the docstring for the rationale.
    mag = raw_mag * speed

    # Defensive: cells with non-finite gradient -> zero direction.
    bad = ~np.isfinite(gx) | ~np.isfinite(gy)
    gx[bad] = 0.0
    gy[bad] = 0.0
    mag[bad] = 0.0  # publish a finite zero, not NaN; planner treats
    # zero magnitude as zero confidence -> ignore
    return gx, gy, mag


def _gaussian_with_nan(arr: np.ndarray, sigma_cells: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing (Knutsson-Westin normalised convolution)."""
    valid = np.isfinite(arr).astype(np.float64)
    filled = np.where(valid > 0, arr, 0.0)
    num = gaussian_filter(filled, sigma_cells)
    den = gaussian_filter(valid, sigma_cells)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan)
    out[~np.isfinite(arr)] = np.nan
    return out


# ---------------------------------------------------------------------------
# Node configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FrameConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    occupancy_threshold: int = 65
    allow_unknown: bool = False


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
class VizConfig:
    viz_subsample: int = 4
    viz_arrow_length: float = 0.3  # [m]; max arrow length
    viz_scale_arrows: bool = True  # scale by |grad T|
    viz_path_step: float = 0.5  # [cells]; <1 -> sub-cell
    viz_path_max_iter: int = 2000
    viz_rate: float = 5.0  # [Hz]


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------


class FMMVectorFieldNode(Node):

    def __init__(self):
        super().__init__("fmm_vector_field")

        # All parameters come from dataclass defaults via the generic
        # loader. Adding a new parameter is now a one-line change to the
        # appropriate dataclass.
        self.frame_cfg = declare_and_load_dataclass(self, FrameConfig())
        self.speed_cfg = declare_and_load_dataclass(self, SpeedConfig())
        self.cutlocus_cfg = declare_and_load_dataclass(self, CutLocusConfig())
        self.viz_cfg = declare_and_load_dataclass(self, VizConfig())

        # Validate the parts that have non-trivial constraints.
        if self.speed_cfg.speed_profile not in ("linear", "exponential"):
            self.get_logger().warn(
                f"Unknown speed_profile '{self.speed_cfg.speed_profile}', "
                "falling back to 'linear'."
            )
            self.speed_cfg = replace(self.speed_cfg, speed_profile="linear")
        if self.speed_cfg.speed_v_min <= 0.0:
            raise ValueError("speed_v_min must be > 0 (eikonal diverges as v -> 0)")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_array: Optional[np.ndarray] = None
        self.current_goal: Optional[PoseStamped] = None
        self.field_dirty = False

        self.travel_time: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_y: Optional[np.ndarray] = None
        self.grad_mag: Optional[np.ndarray] = None
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

        self.viz_timer = self.create_timer(
            1.0 / self.viz_cfg.viz_rate,
            self._viz_timer_cb,
        )

        # NOTE on caching. EDT and the speed field depend only on the
        # static obstacle map and could be cached when /map has not
        # changed. Not implemented here: the deployment runs in unknown
        # environments where /map updates frequently. If the deployment
        # ever switches to a static map, hash the obstacle mask in
        # _map_cb and reuse the EDT and speed arrays in _recompute_field
        # whenever the hash matches.

        self.get_logger().info(
            "FMMVectorFieldNode (FM2) ready. Waiting for /map and /goal_pose..."
        )

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
        if msg.header.frame_id == "":
            self.current_goal = None
            self.get_logger().info(f"Goal was removed")
            return

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

        obstacle_mask = raw >= self.frame_cfg.occupancy_threshold
        if not self.frame_cfg.allow_unknown:
            obstacle_mask = obstacle_mask | (raw < 0)

        if obstacle_mask[goal_row, goal_col]:
            self.get_logger().error("Goal cell is inside an obstacle.")
            return

        t0 = time.monotonic()

        # FM2 pipeline. The "first FMM" of the literature is mathematically
        # the EDT; we compute it directly via scipy.
        edt_free = distance_transform_edt(~obstacle_mask) * resolution
        speed = build_speed_field(edt_free, obstacle_mask, self.speed_cfg)
        tt = solve_eikonal_full(
            obstacle_mask,
            speed,
            goal_col,
            goal_row,
            resolution,
        )

        free_tt = tt[np.isfinite(tt)]
        self._free_max_T = float(free_tt.max()) if free_tt.size else 1.0

        sigma = (
            self.cutlocus_cfg.smooth_T_sigma
            if self.cutlocus_cfg.smooth_T_before_grad
            else 0.0
        )
        gx, gy, mag = field_from_T(tt, obstacle_mask, speed, resolution, sigma)

        self.travel_time = tt
        self.grad_x = gx
        self.grad_y = gy
        self.grad_mag = mag
        self.field_dirty = False

        h, w = obstacle_mask.shape
        n_finite = int(np.isfinite(tt).sum())
        self.get_logger().info(
            f"Field recomputed: {w}x{h} cells, {n_finite} reached "
            f"({100.0*n_finite/(h*w):.1f}%), "
            f"{(time.monotonic()-t0)*1000:.1f} ms "
            f"[smooth_T={self.cutlocus_cfg.smooth_T_before_grad}, "
            f"profile={self.speed_cfg.speed_profile}, "
            f"R={self.speed_cfg.inflation_radius:.2f} m]"
        )

    # ------------------------------------------------------------------
    # Public query interface (used by the path-trace viz)
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
                self.frame_cfg.map_frame,
                self.frame_cfg.robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return None
        pose = PoseStamped()
        pose.header.frame_id = self.frame_cfg.map_frame
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
        if self.travel_time is None or self.grad_x is None or self.current_goal is None:
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
        h, w = self.grad_x.shape
        step = self.viz_cfg.viz_subsample
        max_len = self.viz_cfg.viz_arrow_length
        scale = self.viz_cfg.viz_scale_arrows
        t_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0

        if scale and self.grad_mag is not None:
            valid = self.grad_mag[(self.grad_mag > 0) & np.isfinite(self.grad_mag)]
            mag_ref = float(np.percentile(valid, 95)) if valid.size else 1.0
            if mag_ref < 1e-8:
                mag_ref = 1.0
        else:
            mag_ref = 1.0

        marker = Marker()
        marker.header.frame_id = self.frame_cfg.map_frame
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
                    raw_m = float(self.grad_mag[row, col])
                    if not math.isfinite(raw_m):
                        continue
                    arrow = max_len * min(raw_m / mag_ref, 1.0)
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
        if self.travel_time is None:
            self._publish_empty_path()
            return
        pose = self.get_robot_pose()
        if pose is None:
            self._publish_empty_path()
            return

        info = self.map_msg.info
        res = info.resolution
        wx = pose.pose.position.x
        wy = pose.pose.position.y
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        step_world = self.viz_cfg.viz_path_step * res
        max_iter = self.viz_cfg.viz_path_max_iter
        stop_radius_sq = (1.5 * res) ** 2

        path = Path()
        path.header.frame_id = self.frame_cfg.map_frame
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
        msg.header.frame_id = self.frame_cfg.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(msg)

    def _publish_cost_to_go_grid(self):
        if self.map_msg is None or self.travel_time is None:
            return
        free_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0
        tt = np.array(self.travel_time, dtype=np.float64)
        ratio = np.where(
            np.isfinite(tt),
            np.clip(tt / free_max * 99.0, 0.0, 99.0),
            100.0,
        )
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_cfg.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_msg.info
        msg.data = ratio.astype(np.int8).flatten().tolist()
        self.cost_to_go_pub.publish(msg)

    def _publish_planner_data(self):
        """Pack the field for the NMPC.

        Layout (float32):
          [h, w, origin_x, origin_y, resolution,
           travel_time(H*W), grad_x(H*W), grad_y(H*W), grad_mag(H*W)]

        The grad_mag channel is new: the planner uses it to compute a
        per-stage confidence weight (larger |grad T| -> field is more
        reliable -> trust the published direction more heavily). NaN
        cells in T are converted to a large finite value so the planner
        can compare without special-casing NaN; the corresponding grad
        and grad_mag entries are zero.
        """
        if (
            self.map_msg is None
            or self.travel_time is None
            or self.current_goal is None
        ):
            return

        info = self.map_msg.info
        h, w = self.grad_x.shape
        big_T = self._free_max_T * 4.0 + 1.0
        tt_out = np.where(np.isfinite(self.travel_time), self.travel_time, big_T)
        mag_out = np.where(np.isfinite(self.grad_mag), self.grad_mag, 0.0)

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
                mag_out.astype(np.float32).ravel(),
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
