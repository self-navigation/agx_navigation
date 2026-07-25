"""Publish randomly sampled, reachable goals to drive the fixture.

Goals are sampled from the baked map the planner uses: occupied cells are
inflated by `clearance` metres, unknown cells count as blocked, and only the
connected component containing the robot is sampled. Goals go out one at a
time, each held for `dwell` seconds. The corrector's empty-frame_id sentinel on
/goal_pose advances to the next goal early; an expired dwell is logged as a
failed goal. The node has no ground-truth input and cannot detect collisions.
"""

import os
import random

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from rudn_ordjo_building.map_publisher import load_occupancy_grid


def reachable_mask(data, width, height, resolution, origin, seed_xy, clearance):
    """Cells that are free, clear of walls by `clearance`, and reachable.

    Returns a (height, width) bool array in the OccupancyGrid's row-major
    layout, where row 0 is y_min.
    """
    from scipy import ndimage

    grid = np.asarray(data, dtype=np.int8).reshape(height, width)

    # Unknown counts as blocked -- see the module docstring.
    blocked = grid != 0

    radius_cells = int(np.ceil(clearance / resolution))
    if radius_cells > 0:
        r = radius_cells
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        blocked = ndimage.binary_dilation(blocked, structure=disk)

    free = ~blocked

    seed_col = int((seed_xy[0] - origin[0]) / resolution)
    seed_row = int((seed_xy[1] - origin[1]) / resolution)
    if not (0 <= seed_row < height and 0 <= seed_col < width):
        raise RuntimeError(f"seed {seed_xy} is outside the map")
    if not free[seed_row, seed_col]:
        raise RuntimeError(
            f"seed {seed_xy} is blocked or within {clearance} m of a wall; "
            "the robot cannot be where it says it is"
        )

    labels, _ = ndimage.label(free)
    return labels == labels[seed_row, seed_col]


class RandomGoalDriver(Node):
    def __init__(self):
        super().__init__("random_goals")

        self.declare_parameter("map_yaml", "")
        self.declare_parameter("floor_number", 3)
        self.declare_parameter("count", 10)
        self.declare_parameter("dwell", 60.0)
        self.declare_parameter("clearance", 0.45)
        self.declare_parameter("min_range", 2.0)
        self.declare_parameter("max_range", 15.0)
        self.declare_parameter("seed_x", 0.0)
        self.declare_parameter("seed_y", 0.0)
        self.declare_parameter("rng_seed", 0)
        # Publishing before every subscriber has been matched silently drops the
        # goal for whoever was slow -- /goal_pose has two consumers (the vector
        # field and the corrector) and losing either wedges the run with no error.
        self.declare_parameter("expected_subscribers", 2)

        map_yaml = self.get_parameter("map_yaml").value
        if not map_yaml:
            from ament_index_python.packages import get_package_share_directory

            floor = self.get_parameter("floor_number").value
            map_yaml = os.path.join(
                get_package_share_directory("rudn_ordjo_building"),
                "maps",
                f"floor_{floor}.yaml",
            )

        self._count = int(self.get_parameter("count").value)
        self._dwell = float(self.get_parameter("dwell").value)
        self._expected_subs = int(self.get_parameter("expected_subscribers").value)
        min_range = float(self.get_parameter("min_range").value)
        max_range = float(self.get_parameter("max_range").value)
        seed_xy = (
            float(self.get_parameter("seed_x").value),
            float(self.get_parameter("seed_y").value),
        )

        data, width, height, meta = load_occupancy_grid(map_yaml)
        res = meta["resolution"]
        origin = meta["origin"]

        mask = reachable_mask(
            data, width, height, res, origin, seed_xy,
            float(self.get_parameter("clearance").value),
        )

        rows, cols = np.nonzero(mask)
        xs = origin[0] + (cols + 0.5) * res
        ys = origin[1] + (rows + 0.5) * res
        dist = np.hypot(xs - seed_xy[0], ys - seed_xy[1])
        keep = (dist >= min_range) & (dist <= max_range)
        self._candidates = list(zip(xs[keep], ys[keep]))

        if not self._candidates:
            raise RuntimeError(
                f"no reachable cell between {min_range} and {max_range} m of "
                f"{seed_xy}; widen the range or check the seed pose"
            )

        self.get_logger().info(
            f"{mask.sum()} reachable cells, {len(self._candidates)} within "
            f"[{min_range}, {max_range}] m of {seed_xy} -- sampling {self._count}"
        )

        self._rng = random.Random(int(self.get_parameter("rng_seed").value))
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=1,
        )
        self._pub = self.create_publisher(PoseStamped, "/goal_pose", qos)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal, qos)

        self._sent = 0
        self._done = False
        self._deadline = None
        self.create_timer(0.5, self._tick)

    def _on_goal(self, msg: PoseStamped):
        # Only the completion sentinel is of interest; our own goals echo back.
        if msg.header.frame_id == "":
            self._done = True

    def _tick(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        if self._deadline is not None:
            if self._done:
                self.get_logger().info(f"goal {self._sent} reached")
            elif now < self._deadline:
                return
            else:
                self.get_logger().warn(
                    f"goal {self._sent} did not finish within {self._dwell}s"
                )

        if self._sent >= self._count:
            self.get_logger().info("sequence complete")
            raise SystemExit(0)

        if self._pub.get_subscription_count() < self._expected_subs:
            self.get_logger().warn(
                f"only {self._pub.get_subscription_count()} of "
                f"{self._expected_subs} subscribers matched; waiting",
                throttle_duration_sec=5.0,
            )
            return

        x, y = self._rng.choice(self._candidates)
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.w = 1.0
        self._pub.publish(msg)

        self._sent += 1
        self._done = False
        self._deadline = now + self._dwell
        self.get_logger().info(f"goal {self._sent}/{self._count}: ({x:.2f}, {y:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RandomGoalDriver()
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
