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
        self.exploring = False
        self.current_goal_handle = None
        self.costmap_data = None
        self.robot_pose = None  # (x, y) in map frame
        self.robot_yaw = 0.0

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
        self.declare_parameter("min_goal_distance", 1.0)  # metres
        self.declare_parameter("min_segment_length", 0.3)  # metres
        self.declare_parameter("blacklist_radius", 0.5)  # metres
        self.declare_parameter("max_navigable_cost", 50)  # 0–100 costmap scale
        self.declare_parameter("heading_bias_weight", 0.3)  # 0 = off
        self.declare_parameter("debug_markers", True)

        self.min_goal_distance = (
            self.get_parameter("min_goal_distance").get_parameter_value().double_value
        )
        self.min_segment_length = (
            self.get_parameter("min_segment_length").get_parameter_value().double_value
        )
        self.blacklist_radius = (
            self.get_parameter("blacklist_radius").get_parameter_value().double_value
        )
        self.max_navigable_cost = (
            self.get_parameter("max_navigable_cost").get_parameter_value().integer_value
        )
        self.heading_bias_weight = (
            self.get_parameter("heading_bias_weight").get_parameter_value().double_value
        )
        self.debug_markers = (
            self.get_parameter("debug_markers").get_parameter_value().bool_value
        )

        self.blacklisted_goals = []

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
            if self.costmap_data:
                self.update_robot_pose()
                if self.robot_pose:
                    self.explore()
        else:
            self.get_logger().info("Exploration DEACTIVATED – stopping robot.")
            self.stop_robot()

    def stop_robot(self):
        """Cancel any active Nav2 goal – Nav2 handles stopping the robot."""
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling current Nav2 goal…")
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            self.exploring = False

    def cancel_done_callback(self, future):
        self.get_logger().info("Nav2 goal cancel acknowledged.")
        self.current_goal_handle = None
        self.exploring = False

    # ------------------------------------------------------------------ #
    #  Costmap callback
    # ------------------------------------------------------------------ #
    def costmap_callback(self, msg):
        self.costmap_data = msg
        if not self.active or self.exploring:
            return
        self.update_robot_pose()
        if self.robot_pose:
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
    #  Exploration – border-segment on costmap
    # ------------------------------------------------------------------ #
    def explore(self):
        if not self.costmap_data or not self.active:
            return

        self.exploring = True

        info = self.costmap_data.info
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        h, w = info.height, info.width

        # Costmap OccupancyGrid: -1 = unknown, 0 = free, 100 = lethal
        grid = np.array(self.costmap_data.data, dtype=np.int8).reshape((h, w))

        unknown = grid == -1

        # Navigable = not unknown AND cost below threshold
        navigable = (grid >= 0) & (grid <= self.max_navigable_cost)

        # ---- 1. Raw frontier: ANY non-unknown cell adjacent to unknown ----
        #         (for debug visualisation only)
        any_known = grid != -1
        raw_frontier_mask = (
            (np.roll(unknown, 1, axis=0) & any_known)
            | (np.roll(unknown, -1, axis=0) & any_known)
            | (np.roll(unknown, 1, axis=1) & any_known)
            | (np.roll(unknown, -1, axis=1) & any_known)
        )
        raw_frontier_mask[0, :] = False
        raw_frontier_mask[-1, :] = False
        raw_frontier_mask[:, 0] = False
        raw_frontier_mask[:, -1] = False
        raw_frontier_cells = np.argwhere(raw_frontier_mask)

        # ---- 2. Navigable frontier: navigable cells adjacent to unknown ----
        frontier_mask = (
            (np.roll(unknown, 1, axis=0) & navigable)
            | (np.roll(unknown, -1, axis=0) & navigable)
            | (np.roll(unknown, 1, axis=1) & navigable)
            | (np.roll(unknown, -1, axis=1) & navigable)
        )
        frontier_mask[0, :] = False
        frontier_mask[-1, :] = False
        frontier_mask[:, 0] = False
        frontier_mask[:, -1] = False

        if not np.any(frontier_mask):
            self.get_logger().info("No navigable frontiers! Exploration complete.")
            self.publish_debug_markers(
                raw_frontier_cells, np.empty((0, 2), dtype=int), None, res, ox, oy
            )
            self.exploring = False
            return

        # ---- 3. Segment extraction ----
        labeled, num_segments = label(frontier_mask)

        rx, ry = self.robot_pose
        min_cells = max(1, int(self.min_segment_length / res))

        # ---- 4. Score each border segment ----
        segments = []
        for i in range(1, num_segments + 1):
            points = np.argwhere(labeled == i)  # (row, col)
            size = len(points)
            if size < min_cells:
                continue

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

            # Snap midpoint to nearest actual frontier cell
            d2 = (points[:, 1] - mid_col) ** 2 + (points[:, 0] - mid_row) ** 2
            snap_idx = int(np.argmin(d2))
            snap_row, snap_col = points[snap_idx]
            snap_wx = ox + snap_col * res
            snap_wy = oy + snap_row * res

            seg_length = size * res

            # Heading bias
            angle_to_goal = math.atan2(snap_wy - ry, snap_wx - rx)
            angle_diff = abs(
                math.atan2(
                    math.sin(angle_to_goal - self.robot_yaw),
                    math.cos(angle_to_goal - self.robot_yaw),
                )
            )
            heading_penalty = 1.0 + self.heading_bias_weight * (
                1.0 - math.cos(angle_diff)
            )

            snap_dist = math.sqrt((snap_wx - rx) ** 2 + (snap_wy - ry) ** 2)
            score = snap_dist * heading_penalty / math.log2(seg_length + 1.0)

            segments.append(
                {
                    "points": points,
                    "snap_wx": snap_wx,
                    "snap_wy": snap_wy,
                    "score": score,
                    "length": seg_length,
                    "size": size,
                }
            )

        if not segments:
            self.get_logger().info("No reachable frontier segments.")
            self.publish_debug_markers(
                raw_frontier_cells, np.empty((0, 2), dtype=int), None, res, ox, oy
            )
            self.exploring = False
            return

        # ---- 5. Select best ----
        segments.sort(key=lambda s: s["score"])
        best = segments[0]

        goal_x, goal_y = best["snap_wx"], best["snap_wy"]
        self.get_logger().info(
            f"Selected border: ({goal_x:.2f}, {goal_y:.2f}), "
            f'score={best["score"]:.2f}, length={best["length"]:.2f}m, '
            f'cells={best["size"]}'
        )

        # ---- 6. Goal with orientation toward target ----
        yaw = math.atan2(goal_y - ry, goal_x - rx)
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.orientation.w = math.cos(yaw / 2.0)

        # ---- 7. Debug markers ----
        all_seg_cells = (
            np.vstack([s["points"] for s in segments]) if segments else np.empty((0, 2))
        )
        self.publish_debug_markers(
            raw_frontier_cells, all_seg_cells, (goal_x, goal_y), res, ox, oy
        )

        self.send_nav_goal(goal)

    # ------------------------------------------------------------------ #
    #  RViz debug markers
    # ------------------------------------------------------------------ #
    def publish_debug_markers(self, raw_cells, navigable_cells, goal_xy, res, ox, oy):
        """Publish three marker layers to /frontier_debug.

        - Red points:  raw boundary (all known cells touching unknown)
        - Cyan points: navigable frontier segments (cost < threshold)
        - Green sphere: selected goal
        """
        if not self.debug_markers:
            return

        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # -- Raw boundary: faint red --
        m_raw = Marker()
        m_raw.header.frame_id = "map"
        m_raw.header.stamp = stamp
        m_raw.ns = "frontier_raw"
        m_raw.id = 0
        m_raw.type = Marker.POINTS
        m_raw.action = Marker.ADD
        m_raw.scale.x = res * 0.8
        m_raw.scale.y = res * 0.8
        m_raw.color = ColorRGBA(r=1.0, g=0.3, b=0.3, a=0.35)
        m_raw.pose.orientation.w = 1.0
        for row, col in raw_cells:
            p = Point()
            p.x = ox + col * res
            p.y = oy + row * res
            p.z = 0.02
            m_raw.points.append(p)
        markers.markers.append(m_raw)

        # -- Navigable frontier segments: bright cyan --
        m_nav = Marker()
        m_nav.header.frame_id = "map"
        m_nav.header.stamp = stamp
        m_nav.ns = "frontier_navigable"
        m_nav.id = 0
        m_nav.type = Marker.POINTS
        m_nav.action = Marker.ADD
        m_nav.scale.x = res * 1.2
        m_nav.scale.y = res * 1.2
        m_nav.color = ColorRGBA(r=0.0, g=1.0, b=1.0, a=0.9)
        m_nav.pose.orientation.w = 1.0
        for row, col in navigable_cells:
            p = Point()
            p.x = ox + col * res
            p.y = oy + row * res
            p.z = 0.03
            m_nav.points.append(p)
        markers.markers.append(m_nav)

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
            self.exploring = False
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
            self.current_goal_handle = None
            self.exploring = False
            return

        self.get_logger().info("Nav2 goal accepted.")
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        result = future.result().result
        self.current_goal_handle = None

        if result:
            self.get_logger().info("Reached frontier! Checking for new ones.")
        else:
            self.get_logger().warn("Failed to reach frontier – blacklisting.")
            if hasattr(self, "last_goal_pose"):
                bx = self.last_goal_pose.pose.position.x
                by = self.last_goal_pose.pose.position.y
                self.blacklisted_goals.append((bx, by))
                if len(self.blacklisted_goals) > 50:
                    self.blacklisted_goals.pop(0)

        self.exploring = False


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
