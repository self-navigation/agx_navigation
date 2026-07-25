"""Drive the fixture with randomly sampled, guaranteed-reachable goals.

Exists because single hand-picked goals are a poor way to characterise the
planner: one goal exercises one corridor, and picking them by hand biases
towards the geometry you already suspect. This samples uniformly from the free
space the robot can actually reach, so failures turn up where they live rather
than where we looked.

Reachability is decided from the SAME baked map the planner sees, not from the
meshes, which matters: a goal the planner considers unreachable produces a
useless run, and a goal on the far side of a wall produces a misleading one.
The sampler therefore reproduces the two constraints the planner is subject to:

  1. Clearance -- occupied cells are inflated by `clearance` metres before
     sampling, so no goal sits closer to a wall than the robot's half-width.
     Unknown (-1) cells are treated as blocked; the baked map marks everything
     outside the building envelope unknown, and a goal out there would send the
     Fast Marching front off into space.
  2. Connectivity -- only the connected component containing the robot is
     sampled. Euclidean distance is not reachability: a point 2 m away through
     a wall is a 30 m drive, or none at all.

Goals go out one at a time, each held for `dwell` seconds. There is no attempt
to detect arrival: the runtime corrector already publishes the empty-frame_id
sentinel on /goal_pose when a trajectory finishes, and this node listens for it
to advance early. A dwell that expires first is treated as a failed goal, which
is the interesting case and is logged as such.

NOTE on collisions: the fixture pins map->odom to identity, so the robot trusts
its own odometry. Once it strikes a wall, odometry diverges from truth and every
subsequent goal in the sequence is measured against a pose that is already
wrong. Treat the run as ending at the first collision; this node cannot detect
one (it has no ground-truth subscription) and will keep going regardless.
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
        # Cap, not an expected duration. The corrector publishes its completion
        # sentinel ONLY on success (_finish in runtime_corrector/node.py), so a
        # goal that ends in a reported failure -- including the false failure
        # the stagnation detector raises when the robot stops because it has
        # ARRIVED -- sends no signal at all, and the full dwell elapses. Keep it
        # a little above the slowest expected goal rather than generous.
        self.declare_parameter("dwell", 90.0)
        self.declare_parameter("clearance", 0.45)
        self.declare_parameter("min_range", 2.0)
        self.declare_parameter("max_range", 15.0)
        self.declare_parameter("seed_x", 0.0)
        self.declare_parameter("seed_y", 0.0)
        self.declare_parameter("rng_seed", 0)
        # Publishing before every subscriber has been matched silently drops the
        # goal for whoever was slow, and each loss wedges the run differently
        # with no error naming the cause: losing the corrector means nothing
        # drives, losing vector_field means the planner sits until it reports
        # 'Timeout waiting for vector field'.
        #
        # COUNTS THIS NODE'S OWN SUBSCRIPTION. We subscribe to /goal_pose
        # ourselves to catch the completion sentinel, and rclpy counts that too.
        # So the default 3 means {vector_field, wheel_corrector, self} -- add one
        # for each extra listener, e.g. 4 when run_recorder is up. Setting this
        # too low is not benign: it lets the goal go out early and the run then
        # fails in a way that looks like a planner bug.
        self.declare_parameter("expected_subscribers", 3)
        # Seconds to keep waiting AFTER the count is satisfied. The count goes
        # green when DDS reports the endpoints matched, which is earlier than
        # the reliable channel being ready to carry data -- publish on that edge
        # and the first (only) message can still be dropped by every peer at
        # once. Observed exactly that: the sampler logged the goal, and the
        # corrector, vector_field and planner all recorded zero goals received.
        self.declare_parameter("settle", 3.0)

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

        self._settle = float(self.get_parameter("settle").value)
        self._matched_since = None
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

        matched = self._pub.get_subscription_count()
        if matched < self._expected_subs:
            self._matched_since = None
            self.get_logger().warn(
                f"only {matched} of {self._expected_subs} subscribers matched; "
                f"waiting",
                throttle_duration_sec=5.0,
            )
            return

        # Count satisfied -- now let the channel settle before publishing. See
        # the `settle` parameter: matched != ready, and this goal is sent once.
        if self._matched_since is None:
            self._matched_since = now
            self.get_logger().info(
                f"{matched} subscribers matched; settling {self._settle}s "
                f"before publishing")
            return
        if now - self._matched_since < self._settle:
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
