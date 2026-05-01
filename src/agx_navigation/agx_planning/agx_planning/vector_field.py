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


class FMMVectorFieldNode(Node):
    def __init__(self):
        super().__init__("fmm_vector_field")

        # --- Parameters ---
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")

        # Cells with occupancy >= this are treated as obstacles.
        # Nav2/ROS convention: 0=free, 100=occupied, -1=unknown.
        # Default 65 matches Nav2's occ_thresh=0.65.
        self.declare_parameter("occupancy_threshold", 65)

        # When False, unknown (-1) cells are treated as obstacles.
        self.declare_parameter("allow_unknown", False)

        # EDT inflation radius [m].  Free cells within this distance of an
        # obstacle receive a speed penalty in FMM AND an additive repulsive
        # potential.  Set this to your robot's safety margin.
        self.declare_parameter("inflation_radius", 0.5)

        # Speed decay exponent inside the inflation zone:
        #   speed(d) = exp(-rate * (1 - d/R))  for 0 <= d < R
        # rate=0 -> linear fallback.  rate=2.5 -> speed in [0.08, 1.0].
        self.declare_parameter("inflation_decay_rate", 2.5)

        # Peak additive repulsion [metres of equivalent travel time] - absolute,
        # NOT a multiple of max_T.
        #
        # The no-local-minima condition requires that the repulsion gradient never
        # exceeds the FMM gradient at the same distance from the wall:
        #
        #   |grad P(d)| = 2*P*(R-d) / R^2
        #   |grad T_fmm(d)| = exp(k*(1-d/R))
        #
        # Setting u = 1 - d/R, the ratio |grad P| / |grad T_fmm| = (2P*u/R) / exp(k*u)
        # is maximised at u* = 1/k  (i.e. d = R*(1-1/k)), NOT at the wall surface.
        #
        # Balance condition (max ratio < 1):
        #   k >= 1:  P  <  k * R * e / 2       (maximum at interior point u=1/k)
        #   k  < 1:  P  <  exp(k) * R / 2      (maximum at endpoint u=1, d=0)
        #   unified: P  <  R/2 * min(k*e, exp(k))
        #
        # For k=2.5, R=0.5 m:  balance = 2.5*0.5*e/2 = 1.70 m.
        # Default 1.0 m gives a ~1.7x safety margin.
        # Raise if the robot clips corners; never exceed the balance point.
        self.declare_parameter("wall_repulsion_strength", 1.0)

        # Scales the constant obstacle fill above max_T:
        #   T(obstacle) = T_max * (1 + obstacle_slope_factor * resolution)
        # This sets all obstacle cells to the same high value, producing a steep
        # gradient at the boundary via np.gradient.  Just needs to be large enough
        # that T_obstacle >> max_T_free for any realistic resolution.
        # Rule of thumb: obstacle_slope_factor * resolution >= 1  (i.e. factor >= 1/res).
        # At 0.05 m/cell: factor >= 20.  Default 400 gives 20x headroom.
        self.declare_parameter("obstacle_slope_factor", 400.0)

        # Gaussian blur radius [m] applied to (grad_x, grad_y) before normalisation.
        # Converted to cells at solve time using the map resolution, so behaviour
        # is independent of resolution.  0 disables smoothing.
        self.declare_parameter("field_smooth_sigma", 0.15)

        self.declare_parameter("viz_subsample", 4)
        self.declare_parameter("viz_arrow_length", 0.3)
        self.declare_parameter("viz_rate", 5.0)

        # --- Cache parameter values ---
        self.map_frame: str = self.get_parameter("map_frame").value
        self.robot_frame: str = self.get_parameter("robot_frame").value
        self.occupancy_threshold: int = self.get_parameter("occupancy_threshold").value
        self.allow_unknown: bool = self.get_parameter("allow_unknown").value
        self.inflation_radius: float = self.get_parameter("inflation_radius").value
        self.inflation_decay_rate: float = self.get_parameter(
            "inflation_decay_rate"
        ).value
        self.wall_repulsion_strength: float = self.get_parameter(
            "wall_repulsion_strength"
        ).value
        self.obstacle_slope_factor: float = self.get_parameter(
            "obstacle_slope_factor"
        ).value
        self.field_smooth_sigma: float = self.get_parameter("field_smooth_sigma").value
        self.viz_subsample: int = self.get_parameter("viz_subsample").value
        self.viz_arrow_length: float = self.get_parameter("viz_arrow_length").value
        viz_rate: float = self.get_parameter("viz_rate").value

        # Check once at startup - parameters are fixed after init.
        self._check_no_local_minima_condition()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_msg: Optional[OccupancyGrid] = None
        self.map_array: Optional[np.ndarray] = None
        self.current_goal: Optional[PoseStamped] = None
        self.field_dirty: bool = False

        # Output field arrays written by _recompute_field.
        self.travel_time: Optional[np.ndarray] = None
        self.grad_x: Optional[np.ndarray] = None
        self.grad_y: Optional[np.ndarray] = None
        # Max travel time over free cells; used to normalise all penalty terms
        # and visualisation colour scale.
        self._free_max_T: float = 1.0

        # /map uses TRANSIENT_LOCAL durability so late-joining subscribers
        # receive the last published map without waiting for a new one.
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self._map_cb, map_qos
        )
        self.goal_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self._goal_cb, 10
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
            "FMMVectorFieldNode ready. Waiting for /map and /goal_pose..."
        )

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
        # Re-solve immediately if we already have a goal.
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
        # Retry recomputation in case a callback fired before map/goal were ready.
        if self.field_dirty:
            self._recompute_field()
        self._publish_visualization()

    def _world_to_grid(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        """World [m] -> (col, row).  Returns None if out of bounds."""
        info = self.map_msg.info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if 0 <= col < info.width and 0 <= row < info.height:
            return (col, row)
        return None

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Cell centre -> world [m]."""
        info = self.map_msg.info
        wx = info.origin.position.x + (col + 0.5) * info.resolution
        wy = info.origin.position.y + (row + 0.5) * info.resolution
        return (wx, wy)

    def _compute_edt(self, obstacle_mask: np.ndarray, resolution: float) -> np.ndarray:
        """
        Compute the EDT from every free cell to the nearest obstacle.

        scipy.ndimage.distance_transform_edt(X) gives each non-zero (foreground)
        cell its distance to the nearest zero (background) cell.  Zero cells
        receive 0.  Free cells must be foreground so they receive a distance:

          distance_transform_edt(~obstacle_mask)
            -> free cells (True): distance to nearest obstacle [m]  = edt_free
            -> obstacle cells (False): 0  (ignored)

        Returns:
            edt_free : distance [m] from each free cell to the nearest obstacle.
                       Values at obstacle cells are 0 and should be ignored.
        """
        return distance_transform_edt(~obstacle_mask) * resolution

    def _compute_fmm(
        self,
        obstacle_mask: np.ndarray,
        edt_free: np.ndarray,
        goal_col: int,
        goal_row: int,
        resolution: float,
    ) -> np.ndarray:
        """
        Solve the Eikonal equation from the goal with an EDT-modulated speed field.

        Free cells within inflation_radius of an obstacle are assigned reduced
        speed, making it more expensive for the wavefront to propagate through
        the inflation zone.  This creates travel-time gradients that already steer
        away from walls globally, before any additive repulsion is applied.

        Speed model (d = edt_free, R = inflation_radius, k = inflation_decay_rate):
            v(d) = exp(-k * (1 - d/R))   for d < R   (inflation zone)
            v(d) = 1.0                   for d >= R   (open space)

        Returns travel_time with NaN at obstacle cells.
        """
        height, width = obstacle_mask.shape
        R = self.inflation_radius

        speed = np.ones((height, width), dtype=np.float64)
        if R > 0.0:
            # Select free cells inside the inflation zone.
            near_obstacle = ~obstacle_mask & (edt_free < R)
            if near_obstacle.any():
                # norm: 0.0 at the obstacle surface, 1.0 at the radius boundary.
                norm = edt_free[near_obstacle] / R
                if self.inflation_decay_rate > 0.0:
                    speed[near_obstacle] = np.clip(
                        np.exp(-self.inflation_decay_rate * (1.0 - norm)), 0.01, 1.0
                    )
                else:
                    # Linear fallback when decay rate is 0.
                    speed[near_obstacle] = np.clip(norm, 0.01, 1.0)

        # Mask obstacle cells so skfmm treats them as unreachable.
        phi = np.ones((height, width), dtype=np.float64)
        phi[goal_row, goal_col] = -1.0
        phi_masked = np.ma.MaskedArray(phi, mask=obstacle_mask)
        speed_masked = np.ma.MaskedArray(speed, mask=obstacle_mask)

        try:
            raw_tt = skfmm.travel_time(phi_masked, speed_masked, dx=resolution)
        except Exception as exc:
            self.get_logger().error(f"FMM failed: {exc}")
            return np.full((height, width), np.nan)

        tt = np.array(raw_tt, dtype=np.float64)
        if np.ma.is_masked(raw_tt):
            tt[raw_tt.mask] = np.nan
        return tt

    def _recompute_field(self):
        """
        Run the full solve: EDT -> FMM -> obstacle fill -> wall repulsion -> gradient.

        Combined potential field:

          Free cells, inflation zone (d < R):
            T_total = T_fmm + P * ((R - d) / R)^2

          Free cells, open space (d >= R):
            T_total = T_fmm

          Obstacle cells (constant boundary fill):
            T_total = T_max * (1 + S * resolution)

          Output gradient:
            F = -normalize( Gaussian_sigma * grad(T_total) )

          Symbols:
            T_fmm      -- FMM travel time with EDT-modulated speed
            d          -- edt_free: distance from free cell to nearest obstacle [m]
            R          -- inflation_radius [m]
            P          -- wall_repulsion_strength [m], absolute
            S          -- obstacle_slope_factor (sets obstacle T >> max free T)
            T_max      -- max(T_fmm) over free cells [m]
            resolution -- grid cell size [m]

        Local-minima condition: max over d in (0,R) of |grad P| / |grad T_fmm| < 1.
        Worst case is at d = R*(1-1/k) for k>=1, giving condition P < k*R*e/2.
        For k<1 worst case is at d=0, giving P < exp(k)*R/2.
        Unified: P < R/2 * min(k*e, exp(k)).  Default P=1.0m gives ~1.7x margin at k=2.5, R=0.5m.
        """
        if self.map_msg is None or self.current_goal is None:
            return

        goal_cell = self._world_to_grid(
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
        )
        if goal_cell is None:
            self.get_logger().error("Goal is outside map bounds.")
            return

        goal_col, goal_row = goal_cell
        resolution = self.map_msg.info.resolution
        raw = self.map_array

        # Build binary obstacle mask from raw int8 occupancy values.
        obstacle_mask = raw >= self.occupancy_threshold
        if not self.allow_unknown:
            obstacle_mask = obstacle_mask | (raw < 0)  # -1 = unknown -> obstacle

        if obstacle_mask[goal_row, goal_col]:
            self.get_logger().error("Goal cell is inside an obstacle.")
            return

        t0 = time.monotonic()

        edt_free = self._compute_edt(obstacle_mask, resolution)
        tt = self._compute_fmm(obstacle_mask, edt_free, goal_col, goal_row, resolution)

        # FMM max over free cells - used to scale obstacle fill and as penalty reference.
        free_tt = tt[~obstacle_mask]
        max_T = (
            float(np.nanmax(free_tt))
            if free_tt.size > 0 and not np.all(np.isnan(free_tt))
            else 1.0
        )

        # --- Obstacle interior fill ---
        # Set all obstacle cells to a value well above max_T so np.gradient sees a
        # steep outward ramp at the boundary and produces a valid recovery gradient.
        # A depth-proportional fill (second EDT call) is not worth the cost: the
        # robot should never enter a lethal cell, and the boundary gradient alone
        # provides a one-cell-wide recovery signal if it does.
        if obstacle_mask.any():
            tt[obstacle_mask] = max_T * (1.0 + self.obstacle_slope_factor * resolution)

        # --- Additive wall repulsion (EDT-based quadratic potential) ---
        # For each free cell within inflation_radius, add:
        #   P(d) = wall_repulsion_strength * ((R - d) / R)^2
        # wall_repulsion_strength is in metres of travel time (absolute, not x max_T).
        # Keeping the peak small relative to exp(inflation_decay_rate) ensures the
        # FMM goal direction dominates even in tight passages.
        # Combined field:
        #   T_total = T_fmm + wall_repulsion_strength * max(0, (R - d) / R)^2
        R = self.inflation_radius
        if R > 0.0:
            near_wall = ~obstacle_mask & (edt_free < R)
            if near_wall.any():
                peak = self.wall_repulsion_strength  # absolute [m], not * max_T
                ratio = (R - edt_free[near_wall]) / R
                tt[near_wall] += peak * ratio * ratio

        # Compute _free_max_T AFTER repulsion so the colour scale spans the actual
        # post-repulsion range.  Normalising by the pre-repulsion max_T would make
        # any cell with repulsion added (T > max_T_fmm) clip to ratio=1 (all red).
        free_tt_final = tt[~obstacle_mask]
        self._free_max_T = (
            float(np.nanmax(free_tt_final))
            if free_tt_final.size > 0 and not np.all(np.isnan(free_tt_final))
            else 1.0
        )

        # --- Gradient ---
        # np.gradient(arr, dx) returns (d/d_row, d/d_col) in row-major order.
        # Column axis maps to world-x, row axis maps to world-y.
        # Negating gives -grad(T), which points toward decreasing T (toward goal).
        d_row, d_col = np.gradient(tt, resolution)
        gx = -d_col
        gy = -d_row

        bad = np.isnan(gx) | np.isnan(gy) | np.isinf(gx) | np.isinf(gy)
        gx[bad] = 0.0
        gy[bad] = 0.0

        gx, gy = self._smooth_gradient(gx, gy, resolution)

        mag = np.sqrt(gx**2 + gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        gx /= safe_mag
        gy /= safe_mag

        self.travel_time = tt
        self.grad_x = gx
        self.grad_y = gy
        self.field_dirty = False

        h, w = obstacle_mask.shape
        self.get_logger().info(
            f"Field recomputed: {w}x{h} cells in {(time.monotonic()-t0)*1000:.1f} ms "
            f"(max_T={max_T:.1f}, R={R} m, W={self.wall_repulsion_strength:.1f}, "
            f"S={self.obstacle_slope_factor:.0f})"
        )

    def _smooth_gradient(
        self, gx: np.ndarray, gy: np.ndarray, resolution: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Gaussian-smooth the gradient field then re-normalise to unit vectors.

        sigma is specified in metres and converted to cells here, so the smoothing
        radius is independent of map resolution.

        Blurring before normalisation weights each cell by its gradient magnitude.
        Strong, coherent gradients survive; noisy low-magnitude ones are suppressed.
        """
        if self.field_smooth_sigma <= 0.0:
            return gx, gy
        sigma_cells = self.field_smooth_sigma / resolution
        gx = gaussian_filter(gx, sigma=sigma_cells)
        gy = gaussian_filter(gy, sigma=sigma_cells)
        mag = np.sqrt(gx**2 + gy**2)
        safe_mag = np.where(mag > 1e-8, mag, 1.0)
        return gx / safe_mag, gy / safe_mag

    def _check_no_local_minima_condition(self):
        """
        Verifies that the repulsion gradient never exceeds the FMM gradient
        anywhere in the inflation zone, which is the sufficient condition for
        the combined field to have no spurious local minima.

        With u = 1 - d/R, the ratio |grad P| / |grad T_fmm| = (2Pu/R)*exp(-ku)
        is maximised at u* = min(1/k, 1), giving:
          k >= 1:  max ratio = 2P/(kRe)  ->  balance at P = kRe/2
          k  < 1:  max ratio = 2P*exp(-k)/R  ->  balance at P = exp(k)*R/2
        The condition fails at the wall surface (u=1) only for k < 1; for
        k >= 1 the worst case is at d = R*(1-1/k), inside the inflation zone.
        Logs a warning if violated; does not clamp, since in practice FMM
        paths never align anti-parallel to the repulsion gradient, so the
        field may remain local-minima-free even slightly above the balance.
        """
        k = self.inflation_decay_rate
        R = self.inflation_radius
        P = self.wall_repulsion_strength

        if R <= 0.0:
            return

        balance = R / 2.0 * min(k * math.e, math.exp(k))

        # Worst-case d where the ratio is maximised.
        if k >= 1.0:
            d_worst = R * (1.0 - 1.0 / k)
            case = f"interior d={d_worst:.3f} m  (k>=1 case)"
        else:
            d_worst = 0.0
            case = "wall surface d=0  (k<1 case)"

        ratio_at_worst = (
            (2.0 * P / R) * (1.0 - d_worst / R) * math.exp(-k * (1.0 - d_worst / R))
        )

        if P >= balance:
            self.get_logger().warn(
                f"wall_repulsion_strength={P:.3f} m exceeds no-local-minima balance "
                f"{balance:.3f} m  [k={k}, R={R} m, worst case at {case}, "
                f"ratio={ratio_at_worst:.3f}>=1.0]. "
                f"Spurious local minima may appear in the inflation zone."
            )
        else:
            margin = balance / P
            self.get_logger().info(
                f"Local-minima condition satisfied: P={P:.3f} m < balance={balance:.3f} m "
                f"({margin:.2f}x margin, worst-case ratio={ratio_at_worst:.3f} at {case})."
            )

    def query_vector(
        self, wx: float, wy: float
    ) -> Optional[tuple[float, float, float]]:
        """
        Bilinear interpolation of (vx, vy, cost_to_go) at world position (wx, wy).

        Returns None if out of bounds or before the first solve.
        """
        if self.map_msg is None or self.grad_x is None or self.travel_time is None:
            return None

        info = self.map_msg.info
        # Cell centres are at integer + 0.5; subtract 0.5 to get fractional index.
        gx = (wx - info.origin.position.x) / info.resolution - 0.5
        gy = (wy - info.origin.position.y) / info.resolution - 0.5

        height, width = self.grad_x.shape
        if not (0 <= gx < width - 1 and 0 <= gy < height - 1):
            return None

        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0

        def _bilerp(arr: np.ndarray) -> float:
            return float(
                arr[y0, x0] * (1 - fx) * (1 - fy)
                + arr[y0, x0 + 1] * fx * (1 - fy)
                + arr[y0 + 1, x0] * (1 - fx) * fy
                + arr[y0 + 1, x0 + 1] * fx * fy
            )

        return (_bilerp(self.grad_x), _bilerp(self.grad_y), _bilerp(self.travel_time))

    def get_robot_pose(self) -> Optional[PoseStamped]:
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
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
            r=float(ratio),
            g=float(0.2 * (1.0 - ratio)),
            b=float(1.0 - ratio),
            a=0.8,
        )

    def _publish_line_list(self):
        """
        Publish the gradient field as a LINE_LIST marker.

        Each line segment represents one cell: it starts at the cell centre and
        ends at (centre + arrow_length * (grad_x, grad_y)).  Because grad_x/grad_y
        are unit vectors equal to -normalise(grad(T_total)), the arrow points
        directly in the direction the robot should move.

        Colour encodes travel time normalised by the free-space max (blue = near
        goal, red = far).  Obstacle-interior cells clip to red and are shown only
        if their gradient is non-zero (recovery direction).
        """
        height, width = self.grad_x.shape
        step = self.viz_subsample
        arrow_len = self.viz_arrow_length
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

        points: list = []
        colors: list = []
        for row in range(0, height, step):
            for col in range(0, width, step):
                vx = self.grad_x[row, col]
                vy = self.grad_y[row, col]
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                if abs(vx) < 1e-8 and abs(vy) < 1e-8:
                    continue
                wx, wy = self._grid_to_world(col, row)
                color = self._cost_to_color(float(self.travel_time[row, col]), t_max)
                points.append(Point(x=wx, y=wy, z=0.05))
                points.append(
                    Point(x=wx + arrow_len * vx, y=wy + arrow_len * vy, z=0.05)
                )
                colors.extend([color, color])

        marker.points = points
        marker.colors = colors
        self.lines_pub.publish(marker)

    def _publish_cost_to_go_grid(self):
        """
        Publish travel time as an OccupancyGrid, normalised by free-space max_T.

        Obstacle cells have T >> max_T and saturate at 100 (appear lethal in
        RViz).  Free cells span [0, 99] proportional to their cost-to-go.
        Using free-space max_T rather than the global max preserves the useful
        gradient detail that would otherwise be compressed into <1% of the range.
        """
        if self.map_msg is None or self.travel_time is None:
            return
        free_max = self._free_max_T if self._free_max_T > 1e-8 else 1.0
        tt = np.array(self.travel_time, dtype=np.float64)
        grid_data = (
            np.clip(tt / free_max * 99.0, 0.0, 100.0).astype(np.int8).flatten().tolist()
        )

        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.map_msg.info
        msg.data = grid_data
        self.cost_to_go_pub.publish(msg)

    def _publish_planner_data(self):
        """
        Pack field data for the NMPC planner node.

        Layout (float32): [height, width, origin_x, origin_y, resolution,
                           travel_time (H*W), grad_x (H*W), grad_y (H*W)]
        """
        if self.map_msg is None or self.travel_time is None:
            return

        info = self.map_msg.info
        h, w = self.grad_x.shape
        header = np.array(
            [h, w, info.origin.position.x, info.origin.position.y, info.resolution],
            dtype=np.float32,
        )

        msg = Float32MultiArray()
        msg.data = np.concatenate(
            [
                header,
                np.array(self.travel_time, dtype=np.float32).ravel(),
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
