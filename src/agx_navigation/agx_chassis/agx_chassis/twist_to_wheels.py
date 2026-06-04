import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Float64MultiArray


# Joint order expected by wheel_velocity_controller.yaml
_JOINT_ORDER = ["front_left", "rear_left", "front_right", "rear_right"]


class TwistToWheelsNode(Node):
    def __init__(self):
        super().__init__("twist_to_wheels")

        self.declare_parameter("wheel_radius", 0.08)
        self.declare_parameter("track", 0.416503)
        self.declare_parameter("max_wheel_speed", 20.0)
        # How long a CMD_WHEEL message keeps raw mode active before falling
        # back to the Twist path.
        self.declare_parameter("raw_timeout", 0.5)

        self._radius = self.get_parameter("wheel_radius").value
        self._track = self.get_parameter("track").value
        self._max_speed = self.get_parameter("max_wheel_speed").value
        self._raw_timeout = self.get_parameter("raw_timeout").value

        self._latest_twist: TwistStamped | None = None
        self._latest_raw: list[float] | None = None
        self._raw_stamp: float = 0.0  # wall time of last CMD_WHEEL message

        self._pub = self.create_publisher(
            Float64MultiArray, "/wheel_velocity_controller/commands", 10
        )

        self.create_subscription(TwistStamped, "/cmd_vel", self._on_twist, 10)
        self.create_subscription(
            Float64MultiArray, "/cmd_wheel", self._on_raw, 10
        )

        # Publish at the controller's update rate.
        self.create_timer(1.0 / 100.0, self._publish)

    # ------------------------------------------------------------------
    def _on_twist(self, msg: TwistStamped) -> None:
        self._latest_twist = msg

    def _on_raw(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 4:
            self.get_logger().warn(
                f"CMD_WHEEL expects 4 values, got {len(msg.data)} — ignoring"
            )
            return
        self._latest_raw = list(msg.data)
        self._raw_stamp = time.monotonic()

    def _publish(self) -> None:
        raw_active = (time.monotonic() - self._raw_stamp) < self._raw_timeout

        if raw_active and self._latest_raw is not None:
            speeds = [self._clamp(v) for v in self._latest_raw]
        elif self._latest_twist is not None:
            speeds = self._twist_to_speeds(self._latest_twist)
        else:
            return  # nothing to send yet

        msg = Float64MultiArray()
        msg.data = speeds
        self._pub.publish(msg)

    # ------------------------------------------------------------------
    def _twist_to_speeds(self, twist: TwistStamped) -> list[float]:
        v = twist.twist.linear.x
        omega = twist.twist.angular.z
        half_track = self._track / 2.0

        v_left = (v - omega * half_track) / self._radius
        v_right = (v + omega * half_track) / self._radius

        return [
            self._clamp(v_left),   # front_left
            self._clamp(v_left),   # rear_left
            self._clamp(v_right),  # front_right
            self._clamp(v_right),  # rear_right
        ]

    def _clamp(self, value: float) -> float:
        return max(-self._max_speed, min(self._max_speed, value))


def main(args=None):
    rclpy.init(args=args)
    node = TwistToWheelsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
