import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion

_WHEEL_NAMES = (
    "front_left_wheel",
    "rear_left_wheel",
    "front_right_wheel",
    "rear_right_wheel",
)

# 6×6 covariance diagonal indices (row-major, [x,y,z,roll,pitch,yaw])
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


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__("wheel_odometry")

        self.declare_parameter("wheel_radius", 0.08)
        self.declare_parameter("track", 0.416503)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")

        self._radius = self.get_parameter("wheel_radius").value
        self._track = self.get_parameter("track").value
        self._odom_frame = self.get_parameter("odom_frame").value
        self._base_frame = self.get_parameter("base_frame").value

        # Integrated pose
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0

        # Previous joint positions (by name); None until first message
        self._prev: dict[str, float] | None = None

        self._prev_stamp: float | None = None

        self._pose_cov = _diag_cov(_POSE_COV_DIAG)
        self._twist_cov = _diag_cov(_TWIST_COV_DIAG)

        self._pub = self.create_publisher(Odometry, "/odom", 10)

        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)

    # ------------------------------------------------------------------
    def _on_joints(self, msg: JointState) -> None:
        # Build name→position map for our four wheels
        pos = {}
        for name, position in zip(msg.name, msg.position):
            if name in _WHEEL_NAMES:
                pos[name] = position

        if len(pos) < 4:
            return  # not all wheels present yet

        if self._prev is None:
            # Seed previous positions without accumulating a delta
            self._prev = pos
            self._prev_stamp = self._stamp_to_sec(msg.header.stamp)
            return

        now = self._stamp_to_sec(msg.header.stamp)
        dt = now - self._prev_stamp if self._prev_stamp is not None else 0.0
        self._prev_stamp = now

        # Average left and right encoder deltas
        delta_fl = pos["front_left_wheel"] - self._prev["front_left_wheel"]
        delta_rl = pos["rear_left_wheel"] - self._prev["rear_left_wheel"]
        delta_fr = pos["front_right_wheel"] - self._prev["front_right_wheel"]
        delta_rr = pos["rear_right_wheel"] - self._prev["rear_right_wheel"]

        self._prev = pos

        delta_theta_l = (delta_fl + delta_rl) / 2.0
        delta_theta_r = (delta_fr + delta_rr) / 2.0

        d_l = delta_theta_l * self._radius
        d_r = delta_theta_r * self._radius

        d = (d_l + d_r) / 2.0
        d_theta = (d_r - d_l) / self._track

        # Midpoint integration for better accuracy on curves
        mid_theta = self._theta + d_theta / 2.0
        self._x += d * math.cos(mid_theta)
        self._y += d * math.sin(mid_theta)
        self._theta += d_theta

        v = d / dt if dt > 0.0 else 0.0
        omega = d_theta / dt if dt > 0.0 else 0.0

        self._publish_odom(msg.header.stamp, v, omega)

    # ------------------------------------------------------------------
    def _publish_odom(self, stamp, v: float, omega: float) -> None:
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame

        odom.pose.pose.position.x = self._x
        odom.pose.pose.position.y = self._y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _yaw_to_quaternion(self._theta)
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
