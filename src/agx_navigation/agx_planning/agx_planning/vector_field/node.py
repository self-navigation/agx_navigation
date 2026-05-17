"""Fast Marching Square (FM2) vector field generator -- ROS 2 node.

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
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA, Float32MultiArray
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException

from agx_planning.utils import declare_and_load_dataclass
from agx_planning.vector_field import (
    SpeedConfig,
    CutLocusConfig,
    VectorFieldResult,
    world_to_grid,
    grid_to_world,
    compute_field,
    pack_field_array,
)


@dataclass
class FrameConfig:
    map_frame: str = "map"
    robot_frame: str = "base_link"
    occupancy_threshold: int = 65
    allow_unknown: bool = False


@dataclass
class VizConfig:
    viz_subsample: int = 4
    viz_arrow_length: float = 0.3  # [m]; max arrow length
    viz_scale_arrows: bool = True  # scale by |grad T|
    # Step size for the path-trace gradient descent, expressed as a cell
    # multiplier (<1 -> sub-cell steps). Multiplied by resolution to get
    # the world-space step: step_world = viz_path_step * resolution.
    viz_path_step: float = 0.5
    viz_path_max_iter: int = 2000
    viz_path_rate: float = 5.0  # [Hz] -- path only; field data is event-driven


class VectorFieldNode(Node):

    def __init__(self):
        super().__init__("vector_field")

        self.frame_cfg = declare_and_load_dataclass(self, FrameConfig())
        self.speed_cfg = declare_and_load_dataclass(self, SpeedConfig())
        self.cutlocus_cfg = declare_and_load_dataclass(self, CutLocusConfig())
        self.viz_cfg = declare_and_load_dataclass(self, VizConfig())

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
        self._field_result: Optional[VectorFieldResult] = None

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

        # Only the path needs a timer: it traces from the live robot pose,
        # so it should refresh even when the field itself has not changed.
        self.path_timer = self.create_timer(
            1.0 / self.viz_cfg.viz_path_rate,
            self._publish_optimal_path,
        )

        # NOTE on caching. EDT and the speed field depend only on the
        # static obstacle map and could be cached when /map has not
        # changed. Not implemented here: the deployment runs in unknown
        # environments where /map updates frequently. If the deployment
        # ever switches to a static map, hash the obstacle mask in
        # _map_cb and reuse the EDT and speed arrays in _recompute_field
        # whenever the hash matches.

        self.get_logger().info(
            "VectorFieldNode (FM2) ready. Waiting for /map and /goal_pose..."
        )

    def _map_cb(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.get_logger().info(
            f"Map received: {msg.info.width}x{msg.info.height} "
            f"res={msg.info.resolution:.3f} m/cell",
            throttle_duration_sec=10.0,
        )
        # Recompute and publish immediately; no need to wait for the timer.
        self._recompute_and_publish()

    def _goal_cb(self, msg: PoseStamped):
        if msg.header.frame_id == "":
            self.current_goal = None
            self.get_logger().info("Goal was removed")
            return

        self.current_goal = msg
        self.get_logger().info(
            f"Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )
        self._recompute_and_publish()

    def _recompute_and_publish(self):
        """Recompute the field and immediately push all field-derived topics.

        Called directly from the map/goal callbacks so field data is always
        fresh on the subscriber side without waiting for a timer tick.
        The path is excluded here because it depends on the live robot pose
        and is handled separately by the path timer.
        """
        if not self._recompute_field():
            return

        self._publish_arrows()
        self._publish_cost_to_go_grid()
        self._publish_planner_data()

    def _recompute_field(self) -> bool:
        """Run FM2 and store the result.  Returns True on success."""
        if self.map_msg is None or self.current_goal is None:
            return False

        info = self.map_msg.info
        goal = world_to_grid(
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
            info.origin.position.x,
            info.origin.position.y,
            info.resolution,
            info.width,
            info.height,
        )
        if goal is None:
            self.get_logger().error("Goal is outside map bounds.")
            return False
        goal_col, goal_row = goal

        if self.map_array[goal_row, goal_col] >= self.frame_cfg.occupancy_threshold:
            self.get_logger().error("Goal cell is inside an obstacle.")
            return False

        t0 = time.monotonic()
        outcome = compute_field(
            self.map_array,
            goal_col,
            goal_row,
            info.resolution,
            info.origin.position.x,
            info.origin.position.y,
            self.speed_cfg,
            self.cutlocus_cfg,
            occupancy_threshold=self.frame_cfg.occupancy_threshold,
            allow_unknown=self.frame_cfg.allow_unknown,
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if outcome is None:
            self.get_logger().error(
                "compute_field() failed (goal in obstacle or outside grid)."
            )
            return False

        self._field_result, message = outcome
        self.get_logger().info(f"{message}, {elapsed_ms:.1f} ms")
        return True

    def query_vector(
        self, wx: float, wy: float
    ) -> Optional[tuple[float, float, float]]:
        """Bilinear interpolation of (vx, vy, T) at world (wx, wy).

        Uses cell-corner convention, consistent with world_to_grid() and
        VectorFieldGrid.query_vec(): u = (wx - origin_x) / resolution.
        """
        if self._field_result is None:
            return None

        r = self._field_result
        gx = (wx - r.origin_x) / r.resolution
        gy = (wy - r.origin_y) / r.resolution

        h, w = r.grad_x.shape
        if not (0 <= gx < w - 1 and 0 <= gy < h - 1):
            return None

        x0, y0 = int(gx), int(gy)
        fx, fy = gx - x0, gy - y0

        def _bilerp(arr: np.ndarray) -> float:
            return float(
                arr[y0, x0] * (1.0 - fx) * (1.0 - fy)
                + arr[y0, x0 + 1] * fx * (1.0 - fy)
                + arr[y0 + 1, x0] * (1.0 - fx) * fy
                + arr[y0 + 1, x0 + 1] * fx * fy
            )

        return _bilerp(r.grad_x), _bilerp(r.grad_y), _bilerp(r.travel_time)

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

    def _publish_arrows(self):
        r = self._field_result
        h, w = r.grad_x.shape
        step = self.viz_cfg.viz_subsample
        max_len = self.viz_cfg.viz_arrow_length
        scale = self.viz_cfg.viz_scale_arrows
        t_max = r.free_max_T if r.free_max_T > 1e-8 else 1.0

        if scale:
            valid = r.grad_mag[(r.grad_mag > 0) & np.isfinite(r.grad_mag)]
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
                vx = r.grad_x[row, col]
                vy = r.grad_y[row, col]
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                if abs(vx) < 1e-8 and abs(vy) < 1e-8:
                    continue
                if scale:
                    raw_m = float(r.grad_mag[row, col])
                    if not math.isfinite(raw_m):
                        continue
                    arrow = max_len * min(raw_m / mag_ref, 1.0)
                else:
                    arrow = max_len

                wx, wy = grid_to_world(col, row, r.origin_x, r.origin_y, r.resolution)
                T_val = float(r.travel_time[row, col])
                color = self._cost_to_color(T_val, t_max)
                points.append(Point(x=wx, y=wy, z=0.05))
                points.append(Point(x=wx + arrow * vx, y=wy + arrow * vy, z=0.05))
                colors.extend([color, color])

        marker.points = points
        marker.colors = colors
        self.lines_pub.publish(marker)

    def _publish_optimal_path(self):
        """Timer callback -- traces from the current robot pose each tick."""
        if self._field_result is None or self.current_goal is None:
            self._publish_empty_path()
            return
        pose = self.get_robot_pose()
        if pose is None:
            self._publish_empty_path()
            return

        r = self._field_result
        wx = pose.pose.position.x
        wy = pose.pose.position.y
        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y
        step_world = self.viz_cfg.viz_path_step * r.resolution
        max_iter = self.viz_cfg.viz_path_max_iter
        stop_radius_sq = (1.5 * r.resolution) ** 2

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
        if self.map_msg is None or self._field_result is None:
            return
        r = self._field_result
        free_max = r.free_max_T if r.free_max_T > 1e-8 else 1.0
        tt = np.array(r.travel_time, dtype=np.float64)
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
        """Pack the field as a Float32MultiArray and publish it.

        The serialisation layout is owned by pack_field_array() in
        vf_compute.py; the planner's parse_field_array() in pmp_compute.py
        is the corresponding deserialiser. When generator and planner share
        a process, use field_result_to_grid() instead to skip this step.
        """
        if self._field_result is None:
            return
        msg = Float32MultiArray()
        msg.data = pack_field_array(self._field_result).tolist()
        self.planner_data_pub.publish(msg)

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
