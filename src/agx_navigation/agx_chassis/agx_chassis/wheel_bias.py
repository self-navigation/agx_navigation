"""Sim-only actuator fault: scale each wheel command by a fixed factor.

Sits between the only writer of wheel commands (the runtime corrector) and
the JointGroupVelocityController, so the corrector cannot see the fault
except through the pose -- exactly like a sagging battery or a weak motor
on the real robot (issue #27).

    ~/in  (Float64MultiArray, 4 wheels)  ->  ~/out = in * scale

scale is in controller joint order [front_left, rear_left, front_right,
rear_right]. Every message is relayed, zeros included: the controller
latches its last command, so dropping the corrector's terminal zero would
keep the wheels spinning.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class WheelBiasNode(Node):
    def __init__(self):
        super().__init__("wheel_bias")
        self.declare_parameter("scale", [1.0, 1.0, 1.0, 1.0])
        self._scale = [float(s) for s in self.get_parameter("scale").value]
        if len(self._scale) != 4:
            raise ValueError(f"scale needs 4 values, got {self._scale}")
        self._pub = self.create_publisher(Float64MultiArray, "~/out", 10)
        self.create_subscription(Float64MultiArray, "~/in", self._on_cmd, 10)
        self.get_logger().info(f"wheel_bias active, scale={self._scale}")

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != 4:
            self.get_logger().warn(f"expected 4 wheel commands, got {len(msg.data)}")
            return
        out = Float64MultiArray()
        out.data = [v * s for v, s in zip(msg.data, self._scale)]
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = WheelBiasNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
