import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from nav2_msgs.action import NavigateToPose
from std_msgs.msg import Empty
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.ndimage import label
import math
import random


class FrontierExplorer(Node):
    def __init__(self):
        super().__init__("frontier_explorer")

        self.active = False  # Toggled on/off; starts dormant
        self.exploring = False  # True while a nav goal is in-flight
        self.current_goal_handle = None  # Track the active Nav2 goal handle
        self.map_data = None
        self.robot_pose = None

        self.toggle_sub = self.create_subscription(
            Empty, "/toggle_exploration", self.toggle_callback, 10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, 10
        )

        self.nav_client = ActionClient(self, NavigateToPose, "/navigate_to_pose")

        self.declare_parameter("use_stamped_cmd_vel", False)
        self.use_stamped = (
            self.get_parameter("use_stamped_cmd_vel").get_parameter_value().bool_value
        )

        if self.use_stamped:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        else:
            self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.goal_pub = self.create_publisher(PoseStamped, "/exploration_goal", 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            "Frontier explorer ready (dormant). "
            "Publish to /toggle_exploration to start/stop."
        )

    def toggle_callback(self, msg):
        """Toggle exploration on/off."""
        self.active = not self.active

        if self.active:
            self.get_logger().info("Exploration ACTIVATED.")
            # Kick off an immediate exploration cycle if we already have a map
            if self.map_data:
                self.update_robot_pose()
                if self.robot_pose:
                    self.explore()
        else:
            self.get_logger().info("Exploration DEACTIVATED – stopping robot.")
            self.stop_robot()

    def stop_robot(self):
        """Cancel any active Nav2 goal and publish zero-velocity."""
        # 1. Cancel the in-flight Nav2 goal
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling current Nav2 goal…")
            cancel_future = self.current_goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        else:
            self.exploring = False

        # 2. Immediately command zero velocity so the robot halts
        if self.use_stamped:
            stop_msg = TwistStamped()
            stop_msg.header.stamp = self.get_clock().now().to_msg()
            stop_msg.header.frame_id = "base_link"
            # stop_msg.twist fields default to 0.0
        else:
            stop_msg = Twist()  # all fields default to 0.0
        self.cmd_vel_pub.publish(stop_msg)

    def cancel_done_callback(self, future):
        """Called when the Nav2 cancel request completes."""
        self.get_logger().info("Nav2 goal cancel acknowledged.")
        self.current_goal_handle = None
        self.exploring = False

    def map_callback(self, msg):
        """Callback for new map data – only acts when active & idle."""
        self.map_data = msg
        if not self.active or self.exploring:
            return
        self.update_robot_pose()
        if self.robot_pose:
            self.explore()

    def update_robot_pose(self):
        """Get robot's current pose in map frame via TF."""
        try:
            trans = self.tf_buffer.lookup_transform(
                "map", "base_link", rclpy.time.Time()
            )
            self.robot_pose = trans.transform.translation
        except Exception as e:
            self.get_logger().warn(f"Could not get TF: {e}")
            self.robot_pose = None

    def explore(self):
        """Find frontiers and send the closest cluster centroid as a goal."""
        if not self.map_data or not self.active:
            return

        self.exploring = True

        grid = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width)
        )

        # Find frontiers: free cells (0) adjacent to unknown (-1)
        frontiers = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 0:
                    neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
                    if any(
                        0 <= ny < grid.shape[0]
                        and 0 <= nx < grid.shape[1]
                        and grid[ny, nx] == -1
                        for ny, nx in neighbors
                    ):
                        frontiers.append((x, y))

        if not frontiers:
            self.get_logger().info("No more frontiers! Exploration complete.")
            self.exploring = False
            return

        # Cluster frontiers
        frontier_array = np.zeros_like(grid)
        for fx, fy in frontiers:
            frontier_array[fy, fx] = 1
        labeled, num_clusters = label(frontier_array)

        min_size = 5
        centroids = []
        for i in range(1, num_clusters + 1):
            cluster_points = np.argwhere(labeled == i)
            if len(cluster_points) >= min_size:
                centroid = cluster_points.mean(axis=0)  # [row, col]
                centroids.append((centroid[1], centroid[0]))  # (x, y)

        if not centroids:
            self.get_logger().info("No valid frontier clusters found.")
            self.exploring = False
            return

        # Nearest centroid (grid coords)
        rx, ry = self.robot_pose.x, self.robot_pose.y
        res = self.map_data.info.resolution
        ox = self.map_data.info.origin.position.x
        oy = self.map_data.info.origin.position.y
        grid_rx = int((rx - ox) / res)
        grid_ry = int((ry - oy) / res)

        distances = [
            math.sqrt((cx - grid_rx) ** 2 + (cy - grid_ry) ** 2) + random.random()
            for cx, cy in centroids
        ]
        distances = [d if d > 0.5 else float("inf") for d in distances]

        best_idx = int(np.argmin(distances))
        best_cx, best_cy = centroids[best_idx]

        goal_x = ox + best_cx * res
        goal_y = oy + best_cy * res

        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.w = 1.0

        self.send_nav_goal(goal)

    def send_nav_goal(self, goal_pose):
        """Send a navigation goal to Nav2."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            self.exploring = False
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
        """Handle Nav2 goal acceptance/rejection."""
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
        """Handle navigation result – triggers next exploration cycle."""
        result = future.result().result
        self.current_goal_handle = None

        if result:
            self.get_logger().info("Reached frontier! Checking for new ones.")
        else:
            self.get_logger().warn("Failed to reach frontier.")

        self.exploring = False
        # If still active, the next map_callback will trigger a new cycle.


def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

