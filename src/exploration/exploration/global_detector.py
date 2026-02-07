import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, PointStamped
from nav2_msgs.action import NavigateToPose
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.ndimage import label  # For clustering; install scipy if needed
import math
import random

class FrontierExplorer(Node):
    def __init__(self):
        super().__init__('frontier_explorer')
        # Subscribe to the map topic
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        # Action client for Nav2 navigation
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        # TF listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.map_data = None  # Store latest map
        self.robot_pose = None  # Current robot pose in map frame
        self.exploring = False  # Flag to prevent concurrent exploration

        self.goal_pub = self.create_publisher(
            PoseStamped, '/exploration_goal', 10)  # QoS 10 is reliable for visualization

    def map_callback(self, msg):
        """Callback for new map data."""
        if not self.exploring:
            self.map_data = msg
            self.update_robot_pose()
            if self.robot_pose:
                self.explore()

    def update_robot_pose(self):
        """Get robot's current pose in map frame via TF."""
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            self.robot_pose = trans.transform.translation  # x, y
        except Exception as e:
            self.get_logger().warn(f'Could not get TF: {e}')
            self.robot_pose = None

    def explore(self):
        """Main exploration logic."""
        if not self.map_data:
            return

        self.exploring = True
        # Convert map to NumPy array (height x width)
        grid = np.array(self.map_data.data).reshape(
            (self.map_data.info.height, self.map_data.info.width))
        
        # Find frontiers: free cells (0) adjacent to unknown (-1)
        frontiers = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] == 0:  # Free
                    # Check 4-neighbors for unknown
                    neighbors = [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]
                    if any(0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1] and grid[ny, nx] == -1 for ny, nx in neighbors):
                        frontiers.append((x, y))  # Note: (x, y) is (col, row)

        if not frontiers:
            self.get_logger().info('No more frontiers! Exploration complete.')
            self.exploring = False
            return

        # Cluster frontiers using connected components (labeling)
        frontier_array = np.zeros_like(grid)
        for fx, fy in frontiers:
            frontier_array[fy, fx] = 1
        labeled, num_clusters = label(frontier_array)
        
        # Find centroids of clusters larger than min_size (e.g., 5 cells)
        min_size = 5
        centroids = []
        for i in range(1, num_clusters + 1):
            cluster_points = np.argwhere(labeled == i)
            if len(cluster_points) >= min_size:
                centroid = cluster_points.mean(axis=0)  # [row, col] -> [y, x]
                centroids.append((centroid[1], centroid[0]))  # (x, y)

        if not centroids:
            self.get_logger().info('No valid frontier clusters found.')
            self.exploring = False
            return

        # Select closest centroid to robot
        rx, ry = self.robot_pose.x, self.robot_pose.y
        # Convert robot pose to grid coords
        grid_rx = int((rx - self.map_data.info.origin.position.x) / self.map_data.info.resolution)
        grid_ry = int((ry - self.map_data.info.origin.position.y) / self.map_data.info.resolution)
        
        distances = [math.sqrt((cx - grid_rx)**2 + (cy - grid_ry)**2) + random.random() for cx, cy in centroids]
        distances = [i if i>0.5 else float('inf') for i in distances]

        best_idx = np.argmin(distances)
        best_cx, best_cy = centroids[best_idx]

        # Convert back to world coords
        goal_x = self.map_data.info.origin.position.x + best_cx * self.map_data.info.resolution
        goal_y = self.map_data.info.origin.position.y + best_cy * self.map_data.info.resolution

        # Create PoseStamped goal
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = goal_x
        goal.pose.position.y = goal_y
        goal.pose.orientation.w = 1.0  # Face forward; adjust as needed

        # Send to Nav2
        self.send_nav_goal(goal)

    def send_nav_goal(self, goal_pose):
        """Send navigation goal and handle feedback."""
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Nav2 action server not available!')
            self.exploring = False
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        self.goal_pub.publish(goal_pose)
        self.get_logger().info(f'sending goal pose: {goal_pose}')


        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.nav_goal_callback)

    def nav_goal_callback(self, future):
        """Callback for Nav2 goal response."""
        goal_handle = future.result()
        self.get_logger().info(f'Nav2 goal response: {goal_handle}')
        if not goal_handle.accepted:
            self.get_logger().warn('Nav2 goal rejected!')
            self.exploring = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.nav_result_callback)

    def nav_result_callback(self, future):
        """Handle navigation result."""
        result = future.result().result
        print(f'Nav2 navigation result: {result}')
        if result:
            self.get_logger().info('Reached frontier! Checking for new ones.')
        else:
            self.get_logger().warn('Failed to reach frontier.')
        self.exploring = False  # Ready for next cycle

def main(args=None):
    rclpy.init(args=args)
    node = FrontierExplorer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()