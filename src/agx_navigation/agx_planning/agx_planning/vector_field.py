#!/usr/bin/env python3
"""
FMM Vector Field Planner - ROS2 Jazzy / Nav2

Computes a cost-to-go scalar field from the goal using the Fast Marching
Method (Eikonal equation) on the global costmap.  The negative gradient of
that field gives a smooth, obstacle-respecting vector field that everywhere
points "toward the goal" through free space.

The costmap inflation layer feeds directly into the FMM speed function:
  - free space     -> speed = 1.0  (wavefront travels fast)
  - inflated cells -> speed in (0, 1)  (wavefront slows -> field curves away)
  - lethal cells   -> masked out  (impassable)

Subscribes:
  /global_costmap/costmap   (nav2_msgs OccupancyGrid)
  /goal_pose                (geometry_msgs/PoseStamped)

Publishes:
  /vector_field/lines       (visualization_msgs/Marker) - LINE_LIST visualization
  /vector_field/cost_to_go  (nav_msgs/OccupancyGrid)    - cost-to-go as grid

Robot pose: TF  map -> base_link

Depends: pip install scikit-fmm
"""

import math
import time
from typing import Optional

import numpy as np

# scikit-fmm solves the Eikonal equation |∇T| = 1/F  via the
# Fast Marching Method.  T is arrival time (our cost-to-go),
# F is the speed function derived from the costmap.
import skfmm

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Point
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener, TransformException


class FMMVectorFieldNode(Node):
    def __init__(self):
        super().__init__("fmm_vector_field")

        self.declare_parameter("map_frame", "map")
        self.declare_parameter("robot_frame", "base_link")

        # How strongly the costmap inflation reduces wavefront speed.
        # 0.0 = inflation ignored (only lethal blocks), 1.0 = even mildly
        # inflated cells are very slow -> field strongly avoids walls.
        self.declare_parameter("inflation_weight", 0.8)

        # Cells with OccupancyGrid value >= this are impassable.
        self.declare_parameter("lethal_cost", 99)

        # Whether unknown (-1) cells are traversable.
        self.declare_parameter("allow_unknown", False)

        # Visualization: sample every Nth cell in each axis for LINE_LIST.
        self.declare_parameter("viz_subsample", 4)

        # Arrow scale for the LINE_LIST visualization (meters per unit
        # of normalized gradient - purely visual, doesn't affect planning).
        self.declare_parameter("viz_arrow_length", 0.3)

        # How often to republish the visualization (Hz).
        self.declare_parameter("viz_rate", 5.0)

        self.map_frame: str = self.get_parameter("map_frame").value
        self.robot_frame: str = self.get_parameter("robot_frame").value
        self.inflation_weight: float = self.get_parameter("inflation_weight").value
        self.lethal_cost: int = self.get_parameter("lethal_cost").value
        self.allow_unknown: bool = self.get_parameter("allow_unknown").value
        self.viz_subsample: int = self.get_parameter("viz_subsample").value
        self.viz_arrow_length: float = self.get_parameter("viz_arrow_length").value
        viz_rate: float = self.get_parameter("viz_rate").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.costmap_msg: Optional[OccupancyGrid] = None
        self.costmap_array: Optional[np.ndarray] = None  # (H, W) int8
        self.current_goal: Optional[PoseStamped] = None

        # Computed fields (recomputed on new goal or costmap)
        self.travel_time: Optional[np.ndarray] = None  # (H, W) float64
        self.grad_x: Optional[np.ndarray] = None  # (H, W) float64
        self.grad_y: Optional[np.ndarray] = None  # (H, W) float64
        self.field_dirty: bool = False  # needs recompute?

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
            PoseStamped,
            "/goal_pose",
            self._goal_cb,
            10,
        )

        self.lines_pub = self.create_publisher(Marker, "/vector_field/lines", 10)
        self.cost_to_go_pub = self.create_publisher(
            OccupancyGrid, "/vector_field/cost_to_go", 10
        )

        self.viz_timer = self.create_timer(1.0 / viz_rate, self._viz_timer_cb)

        self.get_logger().info(
            "FMMVectorFieldNode ready. Waiting for costmap + goal..."
        )

    def _costmap_cb(self, msg: OccupancyGrid):
        self.costmap_msg = msg
        self.costmap_array = np.array(msg.data, dtype=np.int8).reshape(
            (msg.info.height, msg.info.width)
        )
        self.field_dirty = True
        self.get_logger().info(
            f"Costmap: {msg.info.width}x{msg.info.height} "
            f"res={msg.info.resolution:.3f} m/cell",
            throttle_duration_sec=10.0,
        )

    def _goal_cb(self, msg: PoseStamped):
        self.current_goal = msg
        self.field_dirty = True
        self.get_logger().info(
            f"Goal: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )
        # Compute immediately on new goal
        self._maybe_recompute()

    def _viz_timer_cb(self):
        """Recompute if needed, then publish visualization."""
        self._maybe_recompute()
        self._publish_visualization()

    def _world_to_grid(self, wx: float, wy: float) -> Optional[tuple[int, int]]:
        """World (m) -> grid (col, row).  None if out of bounds."""
        info = self.costmap_msg.info
        col = int((wx - info.origin.position.x) / info.resolution)
        row = int((wy - info.origin.position.y) / info.resolution)
        if 0 <= col < info.width and 0 <= row < info.height:
            return (col, row)
        return None

    def _grid_to_world(self, col: int, row: int) -> tuple[float, float]:
        """Grid (col, row) -> world (m), returning cell centre."""
        info = self.costmap_msg.info
        wx = info.origin.position.x + (col + 0.5) * info.resolution
        wy = info.origin.position.y + (row + 0.5) * info.resolution
        return (wx, wy)

    def _build_speed_function(self) -> np.ma.MaskedArray:
        """Convert the OccupancyGrid into a speed function for FMM.

        Returns a masked array where:
          - masked cells   = impassable (lethal / unknown)
          - speed ∈ (0, 1] = traversable, reduced near obstacles
        """
        raw = self.costmap_array.astype(np.float64)
        height, width = raw.shape

        # Build mask: True = impassable
        mask = raw >= self.lethal_cost
        if not self.allow_unknown:
            # OccupancyGrid stores unknown as -1 (int8), which is 255 unsigned
            mask |= raw == -1

        # Build speed: free space = 1.0, inflated cells slow down.
        # Costmap values 0..98 map to speed 1.0 .. (1 - inflation_weight).
        # Clamp negative (unknown) to 0 before computing - they're masked anyway.
        clamped = np.clip(raw, 0.0, self.lethal_cost - 1)
        normalized_cost = clamped / (self.lethal_cost - 1)  # 0..1

        speed = 1.0 - self.inflation_weight * normalized_cost
        # Ensure a small positive floor so FMM doesn't get numerical issues
        # on cells that aren't quite lethal but have high cost.
        speed = np.clip(speed, 0.01, 1.0)

        return np.ma.MaskedArray(speed, mask=mask)

    def _compute_field(self):
        """Run FMM from the goal and compute the gradient (vector field)."""
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

        # Speed function
        speed = self._build_speed_function()

        # Check that the goal cell itself is traversable
        if speed.mask[gr, gc]:
            self.get_logger().error("Goal cell is inside a lethal obstacle.")
            return

        # Initial condition (phi)
        # skfmm convention: the zero-level-set is the starting front.
        # Negative values = "already reached", positive = "not yet reached".
        # We set the goal cell to -1 and everything else to +1.
        phi = np.ones((height, width), dtype=np.float64)
        phi[gr, gc] = -1.0

        # Apply the same mask so FMM doesn't try to propagate into obstacles.
        phi = np.ma.MaskedArray(phi, mask=speed.mask)

        # Solve Eikonal equation
        t0 = time.monotonic()
        try:
            self.travel_time = skfmm.travel_time(phi, speed, dx=resolution)
        except Exception as e:
            self.get_logger().error(f"FMM failed: {e}")
            return
        dt = time.monotonic() - t0

        # Replace masked (unreachable) cells with a large sentinel so that
        # np.gradient doesn't produce nonsense at boundaries.
        tt = np.array(self.travel_time, dtype=np.float64)
        if np.ma.is_masked(self.travel_time):
            tt[self.travel_time.mask] = np.nan

        # Gradient (vector field)
        # np.gradient returns (d/drow, d/dcol).  We negate to get vectors
        # that point TOWARD the goal (downhill on the cost-to-go surface).
        d_row, d_col = np.gradient(tt, resolution)
        self.grad_x = -d_col  # world x aligns with grid columns
        self.grad_y = -d_row  # world y aligns with grid rows

        # np.gradient produces NaN for any cell adjacent to a NaN cell,
        # not just unreachable cells themselves.  Catch ALL of them.
        bad_mask = (
            np.isnan(tt)
            | np.isnan(self.grad_x)
            | np.isnan(self.grad_y)
            | np.isinf(self.grad_x)
            | np.isinf(self.grad_y)
        )
        self.grad_x[bad_mask] = 0.0
        self.grad_y[bad_mask] = 0.0

        # Normalize to unit vectors (keep magnitude in travel_time).
        magnitude = np.sqrt(self.grad_x**2 + self.grad_y**2)
        safe_mag = np.where(magnitude > 1e-8, magnitude, 1.0)
        self.grad_x /= safe_mag
        self.grad_y /= safe_mag

        self.field_dirty = False
        self.get_logger().info(
            f"FMM computed: {width}x{height} in {dt:.3f}s  "
            f"(max T = {np.nanmax(tt):.1f})"
        )

    def _maybe_recompute(self):
        if self.field_dirty:
            self._compute_field()

    def _publish_visualization(self):
        """Publish the vector field as lines and cost-to-go grid."""
        if self.travel_time is None or self.grad_x is None:
            return

        self._publish_line_list()
        self._publish_cost_to_go_grid()

    def _cost_to_color(self, t_val: float, t_max: float) -> ColorRGBA:
        """Map a cost-to-go value to a blue(low) -> red(high) colour."""
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
        """Publish the vector field as a single LINE_LIST marker.

        Each vector is a line from the cell centre to cell + arrow_length * dir.
        Per-vertex colouring encodes cost-to-go."""
        height, width = self.grad_x.shape
        step = self.viz_subsample
        arrow_len = self.viz_arrow_length

        tt = np.array(self.travel_time, dtype=np.float64)
        if np.ma.is_masked(self.travel_time):
            tt[self.travel_time.mask] = np.nan
        t_max = float(np.nanmax(tt)) if not np.all(np.isnan(tt)) else 1.0

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "vector_field"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.02  # line width in meters
        marker.pose.orientation.w = 1.0
        marker.lifetime = Duration(sec=0, nanosec=0)  # persistent

        points = []
        colors = []

        for row in range(0, height, step):
            for col in range(0, width, step):
                vx = self.grad_x[row, col]
                vy = self.grad_y[row, col]

                # Skip zero, NaN, or inf vectors
                if not (math.isfinite(vx) and math.isfinite(vy)):
                    continue
                if abs(vx) < 1e-8 and abs(vy) < 1e-8:
                    continue

                t_val = tt[row, col]
                if not math.isfinite(t_val):
                    continue

                wx, wy = self._grid_to_world(col, row)
                color = self._cost_to_color(t_val, t_max)

                # Start point
                p_start = Point(x=wx, y=wy, z=0.05)
                # End point: offset along the direction vector
                p_end = Point(
                    x=wx + arrow_len * vx,
                    y=wy + arrow_len * vy,
                    z=0.05,
                )

                points.append(p_start)
                points.append(p_end)
                # LINE_LIST needs a colour per vertex (both same)
                colors.append(color)
                colors.append(color)

        marker.points = points
        marker.colors = colors

        self.lines_pub.publish(marker)
        self.get_logger().debug(
            f"Published LINE_LIST with {len(points) // 2} vectors",
            throttle_duration_sec=5.0,
        )

    def _publish_cost_to_go_grid(self):
        """Re-publish the travel-time field as an OccupancyGrid so it
        can be viewed with RViz's Map display (useful for debugging)."""
        if self.costmap_msg is None or self.travel_time is None:
            return

        tt = np.array(self.travel_time, dtype=np.float64)
        if np.ma.is_masked(self.travel_time):
            tt[self.travel_time.mask] = np.nan

        t_max = float(np.nanmax(tt))
        if t_max < 1e-8:
            return

        # Normalize to 0..100 range for OccupancyGrid
        normalized = tt / t_max * 100.0
        # NaN (unreachable) -> -1 (unknown in OccupancyGrid convention)
        normalized = np.where(np.isnan(normalized), -1.0, normalized)
        grid_data = normalized.astype(np.int8).flatten().tolist()

        msg = OccupancyGrid()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info = self.costmap_msg.info
        msg.data = grid_data

        self.cost_to_go_pub.publish(msg)

    def query_vector(
        self, wx: float, wy: float
    ) -> Optional[tuple[float, float, float]]:
        """Look up the vector field at a world coordinate.

        Returns (vx, vy, cost_to_go) or None if out of bounds / unreachable.
        Uses bilinear interpolation for smooth lookup between cells."""
        if self.costmap_msg is None or self.grad_x is None or self.travel_time is None:
            return None

        info = self.costmap_msg.info
        # Continuous grid coordinates (not snapped to integers)
        gx = (wx - info.origin.position.x) / info.resolution - 0.5
        gy = (wy - info.origin.position.y) / info.resolution - 0.5

        height, width = self.grad_x.shape
        if not (0 <= gx < width - 1 and 0 <= gy < height - 1):
            return None

        # Bilinear interpolation indices
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

        tt = np.array(self.travel_time, dtype=np.float64)
        if np.ma.is_masked(self.travel_time):
            tt[self.travel_time.mask] = np.nan

        # If any of the 4 corners are unreachable, bail
        for r, c in [(y0, x0), (y0, x1), (y1, x0), (y1, x1)]:
            if np.isnan(tt[r, c]):
                return None

        vx = _bilerp(self.grad_x)
        vy = _bilerp(self.grad_y)
        cost = _bilerp(tt)

        return (vx, vy, cost)

    def get_robot_pose(self) -> Optional[PoseStamped]:
        """Look up the robot pose via TF."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
        except TransformException as e:
            self.get_logger().warn(
                f"TF lookup failed: {e}",
                throttle_duration_sec=2.0,
            )
            return None

        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = t.header.stamp
        pose.pose.position.x = t.transform.translation.x
        pose.pose.position.y = t.transform.translation.y
        pose.pose.position.z = t.transform.translation.z
        pose.pose.orientation = t.transform.rotation
        return pose


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
