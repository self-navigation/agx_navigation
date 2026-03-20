import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Empty, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.ndimage import label
import math


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
        self.blacklisted_goals = []

        # --- Toggle subscription ---
        self.toggle_sub = self.create_subscription(
            Empty, "/toggle_exploration", self.toggle_callback, 10
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

        # --- Parameters ---
        self.declare_parameter("traversable_cost_threshold", 50)  # 0–100
        self.declare_parameter("min_goal_distance", 1.0)  # metres
        self.declare_parameter("blacklist_radius", 0.5)  # metres
        self.declare_parameter("heading_bias_weight", 0.3)  # 0 = off
        self.declare_parameter("replan_interval", 2.0)  # seconds
        self.declare_parameter(
            "unknown_filter_size", 50
        )  # cells – discard unknown patches smaller than this
        self.declare_parameter("debug_markers", True)

        self.traversable_threshold = (
            self.get_parameter("traversable_cost_threshold")
            .get_parameter_value()
            .integer_value
        )
        self.min_goal_distance = (
            self.get_parameter("min_goal_distance").get_parameter_value().double_value
        )
        self.blacklist_radius = (
            self.get_parameter("blacklist_radius").get_parameter_value().double_value
        )
        self.heading_bias_weight = (
            self.get_parameter("heading_bias_weight").get_parameter_value().double_value
        )
        self.replan_interval = (
            self.get_parameter("replan_interval").get_parameter_value().double_value
        )
        self.unknown_filter_size = (
            self.get_parameter("unknown_filter_size")
            .get_parameter_value()
            .integer_value
        )
        self.debug_markers = (
            self.get_parameter("debug_markers").get_parameter_value().bool_value
        )

        # --- Visualisation publishers ---
        self.goal_pub = self.create_publisher(PoseStamped, "/exploration_goal", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/frontier_debug", 10)

        # --- TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f"Frontier explorer ready (dormant), listening on {costmap_topic}. "
            "Publish to /toggle_exploration to start/stop."
        )

    # ------------------------------------------------------------------ #
    #  Toggle
    # ------------------------------------------------------------------ #
    def toggle_callback(self, msg):
        self.active = not self.active
        if self.active:
            self.get_logger().info("Exploration ACTIVATED.")
            self.blacklisted_goals.clear()
            self.replan_timer = self.create_timer(
                self.replan_interval, self.replan_callback
            )
            # Immediate first replan
            self.replan_callback()
        else:
            self.get_logger().info("Exploration DEACTIVATED – stopping robot.")
            self.stop()

    def stop(self):
        """Cancel the active Nav2 goal and stop the replan timer."""
        if self.replan_timer is not None:
            self.replan_timer.cancel()
            self.destroy_timer(self.replan_timer)
            self.replan_timer = None

        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling current Nav2 goal…")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None

        self.publish_debug_markers(
            np.empty((0, 2), dtype=int), np.empty((0, 2), dtype=int), None
        )

    # ------------------------------------------------------------------ #
    #  Costmap callback – just store data
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
    #  Robot pose (position + yaw)
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
    #  Exploration – flood-fill on costmap
    # ------------------------------------------------------------------ #
    def explore(self):
        if not self.costmap_data or not self.active:
            return

        info = self.costmap_data.info
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        h, w = info.height, info.width

        grid = np.array(self.costmap_data.data, dtype=np.int8).reshape((h, w))

        unknown = grid == -1
        traversable = (grid >= 0) & (grid <= self.traversable_threshold)

        # ---- 1. Find robot cell, snap to nearest traversable if needed ----
        rx, ry = self.robot_pose
        robot_col = int((rx - ox) / res)
        robot_row = int((ry - oy) / res)

        robot_col = np.clip(robot_col, 0, w - 1)
        robot_row = np.clip(robot_row, 0, h - 1)

        if not traversable[robot_row, robot_col]:
            # Snap to nearest traversable cell in a small search window
            snap_r = 10  # cells
            r_lo = max(0, robot_row - snap_r)
            r_hi = min(h, robot_row + snap_r + 1)
            c_lo = max(0, robot_col - snap_r)
            c_hi = min(w, robot_col + snap_r + 1)
            local = traversable[r_lo:r_hi, c_lo:c_hi]
            if not np.any(local):
                self.get_logger().warn("Robot not on traversable cell, " "cannot plan.")
                return
            local_pts = np.argwhere(local)
            dists = (local_pts[:, 0] - (robot_row - r_lo)) ** 2 + (
                local_pts[:, 1] - (robot_col - c_lo)
            ) ** 2
            best = local_pts[np.argmin(dists)]
            robot_row = best[0] + r_lo
            robot_col = best[1] + c_lo

        # ---- 2. Flood fill: find the connected traversable region ----
        labeled, num_components = label(traversable)
        robot_label = labeled[robot_row, robot_col]

        if robot_label == 0:
            self.get_logger().warn("Robot cell has no traversable component.")
            return

        reachable = labeled == robot_label

        # ---- 3. Filter unknown: discard small interior patches ----
        # RTABmap produces dithered unknown patches inside explored space.
        # Label connected components of the unknown mask and keep only
        # those large enough to be real unexplored regions.
        if self.unknown_filter_size > 0:
            unknown_labeled, num_unknown = label(unknown)
            # Count pixels per component using bincount (fast)
            sizes = np.bincount(unknown_labeled.ravel())
            # sizes[0] is background (not unknown), skip it
            keep = sizes >= self.unknown_filter_size
            keep[0] = False  # background is never unknown
            unknown = keep[unknown_labeled]

        # ---- 4. Frontier: unknown cells adjacent to the reachable region ----
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
            self.publish_debug_markers(
                np.argwhere(reachable), np.empty((0, 2), dtype=int), None
            )
            return

        frontier_cells = np.argwhere(frontier_mask)

        # ---- 5. Cluster frontier cells into segments ----
        frontier_labeled, num_segments = label(frontier_mask)

        # ---- 6. Score each segment ----
        segments = []
        for i in range(1, num_segments + 1):
            points = np.argwhere(frontier_labeled == i)
            size = len(points)

            mid_row = points[:, 0].mean()
            mid_col = points[:, 1].mean()
            wx = ox + mid_col * res
            wy = oy + mid_row * res

            dist = math.sqrt((wx - rx) ** 2 + (wy - ry) ** 2)
            if dist < self.min_goal_distance:
                continue

            if any(
                math.sqrt((wx - bx) ** 2 + (wy - by) ** 2) < self.blacklist_radius
                for bx, by in self.blacklisted_goals
            ):
                continue

            # Snap midpoint to nearest frontier cell in this segment
            d2 = (points[:, 1] - mid_col) ** 2 + (points[:, 0] - mid_row) ** 2
            snap_idx = int(np.argmin(d2))
            snap_row, snap_col = points[snap_idx]
            goal_wx = ox + snap_col * res
            goal_wy = oy + snap_row * res

            seg_length = size * res

            # Heading bias
            angle_to_goal = math.atan2(goal_wy - ry, goal_wx - rx)
            angle_diff = abs(
                math.atan2(
                    math.sin(angle_to_goal - self.robot_yaw),
                    math.cos(angle_to_goal - self.robot_yaw),
                )
            )
            heading_penalty = 1.0 + self.heading_bias_weight * (
                1.0 - math.cos(angle_diff)
            )

            goal_dist = math.sqrt((goal_wx - rx) ** 2 + (goal_wy - ry) ** 2)
            score = goal_dist * heading_penalty / math.log2(seg_length + 1.0)

            segments.append(
                {
                    "goal_wx": goal_wx,
                    "goal_wy": goal_wy,
                    "score": score,
                    "length": seg_length,
                    "size": size,
                }
            )

        if not segments:
            self.get_logger().info("No reachable frontier segments.")
            self.publish_debug_markers(
                frontier_cells, np.empty((0, 2), dtype=int), None
            )
            return

        # ---- 7. Select best ----
        segments.sort(key=lambda s: s["score"])
        best = segments[0]

        goal_x, goal_y = best["goal_wx"], best["goal_wy"]
        self.get_logger().info(
            f"Selected border: ({goal_x:.2f}, {goal_y:.2f}), "
            f'score={best["score"]:.2f}, length={best["length"]:.2f}m, '
            f'cells={best["size"]}'
        )

        # ---- 8. Goal with orientation toward target ----
        yaw = math.atan2(goal_y - ry, goal_x - rx)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # ---- 9. Debug markers ----
        self.publish_debug_markers(frontier_cells, frontier_cells, (goal_x, goal_y))

        self.send_nav_goal(goal)

    # ------------------------------------------------------------------ #
    #  RViz debug markers
    # ------------------------------------------------------------------ #
    def publish_debug_markers(self, frontier_cells, segment_cells, goal_xy):
        """Publish two marker layers to /frontier_debug.

        - Cyan points:  reachable frontier border
        - Green sphere:  selected goal
        """
        if not self.debug_markers:
            return

        info = self.costmap_data.info if self.costmap_data else None
        if info is None:
            return
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # -- Frontier border: cyan --
        m_front = Marker()
        m_front.header.frame_id = "map"
        m_front.header.stamp = stamp
        m_front.ns = "frontier_border"
        m_front.id = 0
        m_front.type = Marker.POINTS
        m_front.action = Marker.ADD
        m_front.scale.x = res * 1.2
        m_front.scale.y = res * 1.2
        m_front.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9)
        m_front.pose.orientation.w = 1.0
        for row, col in frontier_cells:
            p = Point()
            p.x = ox + col * res
            p.y = oy + row * res
            p.z = 0.03
            m_front.points.append(p)
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

        self.last_goal_pose = goal_pose

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
        """Blacklist goals that Nav2 failed to reach."""
        result = future.result().result
        self.current_goal_handle = None

        if not result and hasattr(self, "last_goal_pose"):
            bx = self.last_goal_pose.pose.position.x
            by = self.last_goal_pose.pose.position.y
            self.get_logger().warn(
                f"Nav2 failed to reach ({bx:.2f}, {by:.2f}) – blacklisting."
            )
            self.blacklisted_goals.append((bx, by))
            if len(self.blacklisted_goals) > 50:
                self.blacklisted_goals.pop(0)


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
