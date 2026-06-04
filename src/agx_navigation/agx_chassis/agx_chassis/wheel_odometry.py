import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster

# These are the *joint* names from the ros2_control block (sim.gazebo), which is
# what joint_state_broadcaster publishes on /joint_states. They are not the link
# names (which carry the _link suffix).
_WHEEL_NAMES = (
    "front_left_wheel",
    "rear_left_wheel",
    "front_right_wheel",
    "rear_right_wheel",
)

_POSE_COV_DIAG = [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]
_TWIST_COV_DIAG = [0.001, 0.001, 0.001, 0.001, 0.001, 0.01]


def _diag_cov(diag: list[float]) -> list[float]:
    cov = [0.0] * 36
    for i, v in enumerate(diag):
        cov[i * 7] = v
    return cov


def _yaw_to_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw / 2.0)
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    return q


def _shortest_angle(delta: float) -> float:
    # Wrap an angular delta into [-pi, pi]. This keeps integration correct
    # regardless of whether the controller reports wheel angle as an unbounded
    # accumulator or wrapped; at 100 Hz the per-tick delta is tiny either way.
    return math.atan2(math.sin(delta), math.cos(delta))


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__("wheel_odometry")

        self.declare_parameter("wheel_radius", 0.08)
        self.declare_parameter("track", 0.416503)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        # Default off: a downstream robot_localization EKF is expected to own the
        # odom -> base_link transform. This mirrors enable_odom_tf: false in the
        # old diff_drive config. Two nodes publishing this same transform is what
        # makes sensor data spin with the robot and the body detach from the
        # wheels in RViz. Set true only if nothing else publishes it.
        self.declare_parameter("publish_tf", False)

        self._radius = self.get_parameter("wheel_radius").value
        self._track = self.get_parameter("track").value
        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value
        self._publish_tf = self.get_parameter("publish_tf").value

        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        self._prev_stamp: float | None = None
        self._prev_pos: dict[str, float] | None = None

        self._pose_cov = _diag_cov(_POSE_COV_DIAG)
        self._twist_cov = _diag_cov(_TWIST_COV_DIAG)

        self._tf_broadcaster = TransformBroadcaster(self) if self._publish_tf else None
        self._pub = self.create_publisher(Odometry, "/odom", 10)

        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)

    def _on_joints(self, msg: JointState) -> None:
        # Integrate wheel *position* deltas rather than velocities. The position
        # signal from gz_ros2_control is much cleaner than the velocity signal,
        # which is the main reason DiffDriveController odometry looked better.
        pos = {}
        for name, position in zip(msg.name, msg.position):
            if name in _WHEEL_NAMES:
                pos[name] = position

        if len(pos) < len(_WHEEL_NAMES):
            return

        now = self._stamp_to_sec(msg.header.stamp)

        if self._prev_pos is None:
            # First tick: seed state only, nothing to integrate yet.
            self._prev_pos = pos
            self._prev_stamp = now
            return

        dt = now - self._prev_stamp
        if dt <= 0.0:
            return

        # Per-wheel angular delta -> linear distance travelled by that wheel.
        d_wheel = {
            name: _shortest_angle(pos[name] - self._prev_pos[name]) * self._radius
            for name in _WHEEL_NAMES
        }
        self._prev_pos = pos
        self._prev_stamp = now

        d_left = (d_wheel["front_left_wheel"] + d_wheel["rear_left_wheel"]) / 2.0
        d_right = (d_wheel["front_right_wheel"] + d_wheel["rear_right_wheel"]) / 2.0

        d = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self._track

        # Midpoint integration for better accuracy on curves.
        mid_theta = self._theta + d_theta / 2.0
        self._x += d * math.cos(mid_theta)
        self._y += d * math.sin(mid_theta)
        self._theta = _shortest_angle(self._theta + d_theta)

        # Twist (base_link frame) is derived from the same deltas so pose and
        # twist always agree. If you prefer a smoother instantaneous twist you
        # can swap these two for the reported msg.velocity values instead.
        self._publish(msg.header.stamp, d / dt, d_theta / dt)

    def _publish(self, stamp, v: float, omega: float) -> None:
        q = _yaw_to_quaternion(self._theta)

        # Only broadcast TF if explicitly enabled. When an EKF is running it is
        # the sole owner of odom -> base_link; broadcasting here as well creates
        # a conflicting transform.
        if self._tf_broadcaster is not None:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self._odom_frame
            t.child_frame_id = self._base_frame
            t.transform.translation.x = self._x
            t.transform.translation.y = self._y
            t.transform.translation.z = 0.0
            t.transform.rotation = q
            self._tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame

        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q
        odom.pose.covariance = self._pose_cov

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = omega
        odom.twist.covariance = self._twist_cov

        self._pub.publish(odom)

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = WheelOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
