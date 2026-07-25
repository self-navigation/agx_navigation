"""Publish map->odom from Gazebo ground truth: perfect localization, sim only.

This is the `localization:=truth` mode. It stands in the same place SLAM or AMCL
would -- it is the *only* writer of map->odom -- but it takes the answer from the
simulator instead of estimating it from sensors. Nothing downstream can tell the
difference: the corrector, the planner and RViz all just see a map frame that
happens to be exact.

WHY THIS EXISTS
---------------
The static-map fixture used to pin map->odom to identity, which meant the robot
navigated on raw wheel odometry. Wheel odometry cannot observe slip, so it
over-reports distance travelled -- measured at 0.6-0.7 m over one fixture run on
this world. An open-loop controller ignores pose and is unharmed by that, but a
pose-feedback corrector *drives to the bias*: TVLQR stopped 0.74 m short of the
goal while believing it had arrived within 5 cm. The fixture was measuring
odometry drift and reporting it as corrector error.

So this mode is a measurement instrument, not a deployment mode. It gives the
corrector's PERFORMANCE CEILING -- what it achieves when localization is not the
limiting term -- which is what separates "the corrector is wrong" from
"localization is the constraint". Compare against `localization:=amcl` (the same
baked map, localized honestly off the lidar) to price localization in, and
against `localization:=slam` for the real thing.

Ground truth must never reach the control path by any other route. It enters
here, as a transform, exactly where a real estimator's output would.

FRAMES
------
The map frame is taken to BE the Gazebo world frame. That is not an assumption
this node can verify, but it is what the rest of the fixture already assumes:
the robot spawns at world (0, 0) (gz_sim.launch.py; the floor mesh is offset to
(23, 5) so that origin lands inside it), odometry starts at its own zero, and the
baked map's YAML origin is expressed in world coordinates. `run_recorder`
compares raw Gazebo poses against /odom on the same basis. If the robot's spawn
pose ever moves, `map_offset_*` below is the knob, and the baked map's origin
has to move with it.

The published transform is planar (x, y, yaw). Ground truth carries roll and
pitch too, but the odom frame is planar, the map is a 2-D grid, and the identity
publisher this replaces was planar -- feeding a tilt in here would tilt the
whole map frame for a robot that is merely driving over a bump.
"""

import math
import threading
from typing import Optional, Tuple

import gz.transport13 as gz_transport
import rclpy
from geometry_msgs.msg import TransformStamped
from gz.msgs10.pose_v_pb2 import Pose_V
from rclpy.node import Node
from tf2_ros import Buffer, TransformBroadcaster, TransformListener


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TruthLocalization(Node):

    def __init__(self):
        super().__init__("truth_localization")

        self.declare_parameter("model_name", "scout_mini")
        self.declare_parameter("world_name", "ordjo_world")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_rate", 50.0)
        # Stamped slightly ahead so consumers looking up "now" do not have to
        # extrapolate -- the same trick AMCL uses, for the same reason.
        self.declare_parameter("transform_tolerance", 0.1)
        # World->map offset, should the robot ever stop spawning at the origin.
        self.declare_parameter("map_offset_x", 0.0)
        self.declare_parameter("map_offset_y", 0.0)
        self.declare_parameter("map_offset_yaw", 0.0)

        self.model_name = str(self.get_parameter("model_name").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tolerance = float(self.get_parameter("transform_tolerance").value)
        self._offset = (
            float(self.get_parameter("map_offset_x").value),
            float(self.get_parameter("map_offset_y").value),
            float(self.get_parameter("map_offset_yaw").value),
        )

        # Written from a gz-transport thread, read from the rclpy timer.
        self._lock = threading.Lock()
        self._truth: Optional[Tuple[float, float, float]] = None
        self._warned = False

        self._buffer = Buffer()
        self._listener = TransformListener(self._buffer, self)
        self._br = TransformBroadcaster(self)

        # gz-transport rather than a ros_gz_bridge topic: the Pose_V -> TFMessage
        # bridge drops the entity names, leaving no way to pick the robot out of
        # the hundreds of entities. pose/info (not dynamic_pose/info) carries
        # every entity at a steady rate, so the pose stays fresh while the robot
        # is stopped. Same reasoning as run_recorder.
        world = str(self.get_parameter("world_name").value)
        topic = f"/world/{world}/pose/info"
        self._gz = gz_transport.Node()
        if not self._gz.subscribe(Pose_V, topic, self._on_truth):
            raise RuntimeError(f"could not subscribe to {topic}")

        rate = float(self.get_parameter("publish_rate").value)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info(
            f"ground-truth localization: {topic} [{self.model_name}] "
            f"-> {self.map_frame}->{self.odom_frame}")

    def _on_truth(self, msg: Pose_V):
        """Called on a gz-transport thread, not the rclpy executor."""
        for p in msg.pose:
            if p.name != self.model_name:
                continue
            q = p.orientation
            with self._lock:
                self._truth = (p.position.x, p.position.y,
                               yaw_from_quat(q.x, q.y, q.z, q.w))
            return

    def _publish(self):
        with self._lock:
            truth = self._truth
        if truth is None:
            return

        try:
            tf = self._buffer.lookup_transform(
                self.odom_frame, self.base_frame, rclpy.time.Time())
        except Exception as exc:  # noqa: BLE001 -- tf2 raises several types
            # Expected for the first moments, while the EKF spins up. Warn once
            # so a *persistent* failure is still visible: without odom->base
            # there is no map->odom, and nothing localizes at all.
            if not self._warned:
                self._warned = True
                self.get_logger().warn(
                    f"no {self.odom_frame}->{self.base_frame} yet ({exc}); "
                    "map->odom is not being published")
            return

        ox = tf.transform.translation.x
        oy = tf.transform.translation.y
        q = tf.transform.rotation
        oyaw = yaw_from_quat(q.x, q.y, q.z, q.w)

        # map->odom = (map->base) o (odom->base)^-1, in the plane.
        tx, ty, tyaw = truth
        dx, dy, dyaw = self._offset
        tx, ty, tyaw = tx - dx, ty - dy, tyaw - dyaw

        yaw = tyaw - oyaw
        c, s = math.cos(yaw), math.sin(yaw)
        x = tx - (c * ox - s * oy)
        y = ty - (s * ox + c * oy)

        stamp = self.get_clock().now() + rclpy.duration.Duration(
            seconds=self.tolerance)
        t = TransformStamped()
        t.header.stamp = stamp.to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self._br.sendTransform(t)


def main():
    rclpy.init()
    node = TruthLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
