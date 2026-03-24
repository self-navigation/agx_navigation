import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Empty, ColorRGBA
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.ndimage import label, binary_dilation
import math

# ====================================================================== #
#  Scoring heuristics
#
#  Each heuristic is a standalone function:
#      (segment, context) -> float
#
#  - Returns a score *component* (lower = better).
#  - `segment` is a dict with at least: goal_wx, goal_wy, size, length.
#  - `context` is a dict with: rx, ry, robot_yaw, and any params.
#
#  To add a new heuristic:
#    1. Write a function below following the same signature.
#    2. Add it to HEURISTICS with a name, a default weight, and
#       a default enabled flag.
#    3. That's it - the node picks it up automatically.
#       Weight and enabled are exposed as ROS parameters:
#         heuristic.<name>.enabled  (bool)
#         heuristic.<name>.weight   (float)
# ====================================================================== #


def heuristic_distance(segment, context):
    """Prefer closer frontiers.  Returns Euclidean distance in metres."""
    dx = segment["goal_wx"] - context["rx"]
    dy = segment["goal_wy"] - context["ry"]
    return math.sqrt(dx * dx + dy * dy)


def heuristic_heading(segment, context):
    """Penalise frontiers behind the robot (maintains corridor momentum).
    Returns a value in [1.0, 2.0]:  1.0 = straight ahead, 2.0 = directly behind."""
    dx = segment["goal_wx"] - context["rx"]
    dy = segment["goal_wy"] - context["ry"]
    angle_to_goal = math.atan2(dy, dx)
    angle_diff = abs(
        math.atan2(
            math.sin(angle_to_goal - context["robot_yaw"]),
            math.cos(angle_to_goal - context["robot_yaw"]),
        )
    )
    return 1.0 + (1.0 - math.cos(angle_diff))


def heuristic_segment_size(segment, context):
    """Prefer larger frontier segments (wider openings / more info gain).
    Returns a divisor-style value: 1/log2(length+1), so larger = lower = better."""
    return 1.0 / math.log2(segment["length"] + 1.0)


# Registry: (name, function, default_weight, default_enabled)
HEURISTICS = [
    ("distance", heuristic_distance, 1.0, True),
    ("heading", heuristic_heading, 0.3, True),
    ("segment_size", heuristic_segment_size, 1.0, True),
]


# ====================================================================== #
#  Node
# ====================================================================== #


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        # --- State ---
        self.active = False
        self.current_goal_handle = None
        self.costmap_data = None
        self.robot_pose = None  # (x, y) in map frame
        self.robot_yaw = 0.0
        self.replan_timer = None
        self.current_goal_xy = None
        self._debug_reachable = None

        # --- Toggle service ---
        self.toggle_srv = self.create_service(
            SetBool, "/toggle_exploration", self.toggle_service_callback
        )

        # --- Costmap subscription ---
        self.declare_parameter("costmap_topic", "/global_costmap/costmap")
        costmap_topic = (
            self.get_parameter("costmap_topic").get_parameter_value().string_value
        )
        self.costmap_sub = self.create_subscription(
            OccupancyGrid, costmap_topic, self.costmap_callback, 10
        )

        # --- Nav2 action client ---
        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        # --- Core parameters ---
        self.declare_parameter("traversable_cost_threshold", 50)
        self.declare_parameter("replan_frequency", 0.5)  # Hz
        self.declare_parameter(
            "known_space_dilation", 3
        )  # cells – fills inter-ray gaps
        self.declare_parameter("goal_switch_threshold", 0.3)

        self.traversable_threshold = (
            self.get_parameter("traversable_cost_threshold")
            .get_parameter_value()
            .integer_value
        )
        self.replan_frequency = (
            self.get_parameter("replan_frequency").get_parameter_value().double_value
        )
        self.known_space_dilation = (
            self.get_parameter("known_space_dilation")
            .get_parameter_value()
            .integer_value
        )
        self.goal_switch_threshold = (
            self.get_parameter("goal_switch_threshold")
            .get_parameter_value()
            .double_value
        )

        # --- Heuristic parameters (auto-registered from HEURISTICS) ---
        self.scoring_heuristics = []
        for name, func, default_weight, default_enabled in HEURISTICS:
            param_enabled = f"heuristic.{name}.enabled"
            param_weight = f"heuristic.{name}.weight"
            self.declare_parameter(param_enabled, default_enabled)
            self.declare_parameter(param_weight, default_weight)
            enabled = self.get_parameter(param_enabled).get_parameter_value().bool_value
            weight = self.get_parameter(param_weight).get_parameter_value().double_value
            self.scoring_heuristics.append(
                {
                    "name": name,
                    "func": func,
                    "weight": weight,
                    "enabled": enabled,
                }
            )
            status = "ON" if enabled else "OFF"
            self.get_logger().info(
                f'  Heuristic "{name}": {status}, weight={weight:.2f}'
            )

        # --- Visualisation publishers ---
        self.goal_pub = self.create_publisher(PoseStamped, "/exploration_goal", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/frontier_debug", 10)

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"Frontier explorer ready (dormant), listening on {costmap_topic}. "
            "Call /toggle_exploration service to start/stop."
        )

    # ------------------------------------------------------------------ #
    #  Scoring
    # ------------------------------------------------------------------ #
    def score_segment(self, segment, context):
        """Compute a composite score for a frontier segment.
        Lower = better.  Each enabled heuristic contributes multiplicatively."""
        score = 1.0
        for h in self.scoring_heuristics:
            if h["enabled"]:
                value = h["func"](segment, context)
                score *= value ** h["weight"]
        return score

    # ------------------------------------------------------------------ #
    #  Toggle
    # ------------------------------------------------------------------ #
    def toggle_service_callback(self, request, response):
        self.active = request.data
        if self.active:
            self.get_logger().info("Exploration ACTIVATED.")
            replan_period = 1.0 / self.replan_frequency
            self.replan_timer = self.create_timer(replan_period, self.replan_callback)
            self.replan_callback()
            response.success = True
            response.message = "Exploration activated"
        else:
            self.get_logger().info("Exploration DEACTIVATED – stopping robot.")
            self.stop()
            response.success = True
            response.message = "Exploration deactivated"
        return response

    def stop(self):
        if self.replan_timer is not None:
            self.replan_timer.cancel()
            self.destroy_timer(self.replan_timer)
            self.replan_timer = None

        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling current Nav2 goal…")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None

        self.current_goal_xy = None
        self.publish_debug_markers([], None)

    # ------------------------------------------------------------------ #
    #  Costmap callback
    # ------------------------------------------------------------------ #
    def costmap_callback(self, msg):
        self.costmap_data = msg

    # ------------------------------------------------------------------ #
    #  Periodic replan
    # ------------------------------------------------------------------ #
    def replan_callback(self):
        if not self.active or self.costmap_data is None:
            return
        self.update_robot_pose()
        if self.robot_pose is None:
            return
        self.explore()

    # ------------------------------------------------------------------ #
    #  Robot pose
    # ------------------------------------------------------------------ #
    def update_robot_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            t = trans.transform.translation
            self.robot_pose = (t.x, t.y)

            q = trans.transform.rotation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        except Exception as e:
            self.get_logger().warn(f"Could not get TF: {e}")
            self.robot_pose = None

    # ------------------------------------------------------------------ #
    #  Exploration
    # ------------------------------------------------------------------ #
    def explore(self):
        if not self.costmap_data or not self.active:
            return

        info = self.costmap_data.info
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        h, w = info.height, info.width

        grid = np.array(self.costmap_data.data, dtype=np.int8).reshape((h, w))

        occupied = grid > self.traversable_threshold
        traversable = (grid >= 0) & (grid <= self.traversable_threshold)

        # ---- 1. Build smoothed "known" region ----
        if self.known_space_dilation > 0:
            k = 2 * self.known_space_dilation + 1
            known = binary_dilation(traversable, structure=np.ones((k, k)))
            known = known & ~occupied
        else:
            known = traversable

        # ---- 2. Find robot cell ----
        rx, ry = self.robot_pose
        robot_col = int((rx - ox) / res)
        robot_row = int((ry - oy) / res)
        robot_col = np.clip(robot_col, 0, w - 1)
        robot_row = np.clip(robot_row, 0, h - 1)

        if not known[robot_row, robot_col]:
            snap_r = 10
            r_lo = max(0, robot_row - snap_r)
            r_hi = min(h, robot_row + snap_r + 1)
            c_lo = max(0, robot_col - snap_r)
            c_hi = min(w, robot_col + snap_r + 1)
            local = known[r_lo:r_hi, c_lo:c_hi]
            if not np.any(local):
                self.get_logger().warn("Robot not on known cell.")
                return
            local_pts = np.argwhere(local)
            dists = (local_pts[:, 0] - (robot_row - r_lo)) ** 2 + (
                local_pts[:, 1] - (robot_col - c_lo)
            ) ** 2
            best = local_pts[np.argmin(dists)]
            robot_row = best[0] + r_lo
            robot_col = best[1] + c_lo

        # ---- 3. Flood fill on smoothed known region ----
        labeled, _ = label(known)
        robot_label = labeled[robot_row, robot_col]
        if robot_label == 0:
            self.get_logger().warn("Robot cell has no known component.")
            return
        reachable = labeled == robot_label

        self._debug_reachable = reachable

        # ---- 4. Derive synthetic unknown ----
        unknown = ~reachable & ~occupied

        self.get_logger().debug(
            f"Grid: {h}x{w}, occupied={int(np.sum(occupied))}, "
            f"known={int(np.sum(known))}, reachable={int(np.sum(reachable))}, "
            f"synthetic_unknown={int(np.sum(unknown))}"
        )

        # ---- 5. Frontier detection ----
        frontier_mask = (
            (np.roll(reachable, 1, axis=0) & unknown)
            | (np.roll(reachable, -1, axis=0) & unknown)
            | (np.roll(reachable, 1, axis=1) & unknown)
            | (np.roll(reachable, -1, axis=1) & unknown)
        )
        frontier_mask[0, :] = False
        frontier_mask[-1, :] = False
        frontier_mask[:, 0] = False
        frontier_mask[:, -1] = False

        if not np.any(frontier_mask):
            self.get_logger().info("No more frontiers! Exploration complete.")
            self.publish_debug_markers([], None)
            return

        # ---- 6. Segment extraction (8-connectivity) ----
        frontier_labeled, num_segments = label(frontier_mask, structure=np.ones((3, 3)))

        # ---- 7. Build segment list with metadata ----
        context = {
            "rx": rx,
            "ry": ry,
            "robot_yaw": self.robot_yaw,
        }

        segments = []
        for i in range(1, num_segments + 1):
            points = np.argwhere(frontier_labeled == i)
            size = len(points)

            mid_row = points[:, 0].mean()
            mid_col = points[:, 1].mean()

            # Snap midpoint to nearest frontier cell
            d2 = (points[:, 1] - mid_col) ** 2 + (points[:, 0] - mid_row) ** 2
            snap_idx = int(np.argmin(d2))
            snap_row, snap_col = points[snap_idx]

            segment = {
                "goal_wx": ox + snap_col * res,
                "goal_wy": oy + snap_row * res,
                "size": size,
                "length": size * res,
                "points": points,
            }

            segment["score"] = self.score_segment(segment, context)
            segments.append(segment)

        if not segments:
            self.get_logger().info("No reachable frontier segments.")
            self.current_goal_xy = None
            self.publish_debug_markers([], None)
            return

        # ---- 8. Select best with hysteresis ----
        segments.sort(key=lambda s: s["score"])
        best_new = segments[0]
        chosen = best_new

        if self.current_goal_xy is not None:
            cgx, cgy = self.current_goal_xy
            current_seg = None
            best_match_dist = float("inf")
            for seg in segments:
                d = math.sqrt((seg["goal_wx"] - cgx) ** 2 + (seg["goal_wy"] - cgy) ** 2)
                if d < best_match_dist:
                    best_match_dist = d
                    current_seg = seg

            if current_seg is not None and best_match_dist < 1.0:
                improvement = 1.0 - best_new["score"] / current_seg["score"]
                if improvement < self.goal_switch_threshold:
                    chosen = current_seg

        goal_x, goal_y = chosen["goal_wx"], chosen["goal_wy"]
        self.current_goal_xy = (goal_x, goal_y)

        self.get_logger().info(
            f"Selected border: ({goal_x:.2f}, {goal_y:.2f}), "
            f'score={chosen["score"]:.2f}, length={chosen["length"]:.2f}m, '
            f'cells={chosen["size"]}'
        )

        # ---- 9. Goal with orientation toward target ----
        yaw = math.atan2(goal_y - ry, goal_x - rx)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # ---- 10. Debug markers ----
        self.publish_debug_markers(segments, (goal_x, goal_y))
        self.send_nav_goal(goal)

    # ------------------------------------------------------------------ #
    #  RViz debug markers
    # ------------------------------------------------------------------ #
    # Distinct colour palette (no yellow - reserved for laserscan)
    SEGMENT_COLORS = [
        ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9),  # cyan
        ColorRGBA(r=1.0, g=0.5, b=0.0, a=0.9),  # orange
        ColorRGBA(r=0.6, g=0.2, b=1.0, a=0.9),  # purple
        ColorRGBA(r=1.0, g=0.0, b=0.5, a=0.9),  # pink
        ColorRGBA(r=0.0, g=0.7, b=1.0, a=0.9),  # sky blue
        ColorRGBA(r=0.0, g=1.0, b=0.4, a=0.9),  # mint
        ColorRGBA(r=1.0, g=0.2, b=0.2, a=0.9),  # red
        ColorRGBA(r=0.5, g=1.0, b=0.0, a=0.9),  # lime
        ColorRGBA(r=1.0, g=0.0, b=1.0, a=0.9),  # magenta
        ColorRGBA(r=0.3, g=0.5, b=1.0, a=0.9),  # cornflower
    ]

    def publish_debug_markers(self, segments, goal_xy):
        info = self.costmap_data.info if self.costmap_data else None
        if info is None:
            return
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # -- Clean up stale markers from previous versions --
        for ns in (
            "smoothing_debug",
            "frontier_border",
            "frontier_raw",
            "frontier_navigable",
        ):
            m = Marker()
            m.header.frame_id = "map"
            m.header.stamp = stamp
            m.ns = ns
            m.id = 0
            m.action = Marker.DELETE
            markers.markers.append(m)

        # -- Known space debug: full reachable region (faint blue) --
        m_known = Marker()
        m_known.header.frame_id = "map"
        m_known.header.stamp = stamp
        m_known.ns = "dilation_debug"
        m_known.id = 0
        m_known.type = Marker.POINTS
        m_known.action = Marker.ADD
        m_known.scale.x = res
        m_known.scale.y = res
        m_known.pose.orientation.w = 1.0
        m_known.color = ColorRGBA(r=0.2, g=0.3, b=1.0, a=0.3)

        if self._debug_reachable is not None and np.any(self._debug_reachable):
            reachable_cells = np.argwhere(self._debug_reachable)
            for row, col in reachable_cells:
                p = Point()
                p.x = ox + col * res
                p.y = oy + row * res
                p.z = 0.01
                m_known.points.append(p)
        markers.markers.append(m_known)

        # -- Frontier segments: per-point colour --
        m_front = Marker()
        m_front.header.frame_id = "map"
        m_front.header.stamp = stamp
        m_front.ns = "frontier_segments"
        m_front.id = 0
        m_front.type = Marker.POINTS
        m_front.action = Marker.ADD
        m_front.scale.x = res * 1.2
        m_front.scale.y = res * 1.2
        m_front.pose.orientation.w = 1.0

        palette = self.SEGMENT_COLORS
        for seg_idx, seg in enumerate(segments):
            color = palette[seg_idx % len(palette)]
            for row, col in seg["points"]:
                p = Point()
                p.x = ox + col * res
                p.y = oy + row * res
                p.z = 0.03
                m_front.points.append(p)
                m_front.colors.append(color)
        markers.markers.append(m_front)

        # -- Selected goal: green sphere --
        m_goal = Marker()
        m_goal.header.frame_id = "map"
        m_goal.header.stamp = stamp
        m_goal.ns = "frontier_goal"
        m_goal.id = 0
        m_goal.type = Marker.SPHERE
        m_goal.action = Marker.ADD if goal_xy else Marker.DELETE
        m_goal.scale.x = 0.25
        m_goal.scale.y = 0.25
        m_goal.scale.z = 0.25
        m_goal.color = ColorRGBA(r=0.2, g=1.0, b=0.2, a=1.0)
        m_goal.pose.orientation.w = 1.0
        if goal_xy:
            m_goal.pose.position.x = goal_xy[0]
            m_goal.pose.position.y = goal_xy[1]
            m_goal.pose.position.z = 0.15
        markers.markers.append(m_goal)

        self.marker_pub.publish(markers)

    # ------------------------------------------------------------------ #
    #  Navigation
    # ------------------------------------------------------------------ #
    def send_nav_goal(self, goal_pose):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.goal_pub.publish(goal_pose)
        self.get_logger().info(
            f"Sending goal: ({goal_pose.pose.position.x:.2f}, "
            f"{goal_pose.pose.position.y:.2f})"
        )

        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.nav_goal_callback)

    def nav_goal_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("Nav2 goal rejected!")
            return

        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        self.current_goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
