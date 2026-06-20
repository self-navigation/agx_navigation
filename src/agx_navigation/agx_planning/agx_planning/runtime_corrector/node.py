"""Pass-through wheel-command corrector (placeholder) with debug plan markers.

This node sits between the planner's wheel-velocity output and the
velocity_controllers/JointGroupVelocityController input. Right now it is a
pure relay: every Float64MultiArray it receives is forwarded unchanged.

It exists so the full wiring

    pmp_planner --(wheel_cmd_in)--> wheel_corrector --(wheel_cmd_out)--> controller

is in place and testable before the real corrector lands. The real corrector
(the "optimal control with small errors" stage) will replace _correct() with a
per-wheel residual model -- heuristic or learned -- that nudges each of the four
wheel setpoints to cancel execution error on bad terrain (ice, mud, oil). The
plumbing here does not change when that happens; only _correct() does.

The command array is in the controller's joint order
[front_left, rear_left, front_right, rear_right] = [w_l, w_l, w_r, w_r].

Debug visualization mirrors the runtime_corrector's TrajectoryVisualizer
(same marker ids/namespaces/colours), restricted to the plan-only markers a
pass-through can produce: the centerline, the corridor tube, and the state
text. The recovery-only markers (projection/carrot/look-ahead) are not drawn
because there is no correction logic yet. The class itself is not imported to
avoid pulling the whole runtime_corrector package into this placeholder.
"""

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from builtin_interfaces.msg import Duration as BuiltinDuration
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import (
    Buffer,
    TransformListener,
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)

_MARKER_NS = "pmp_wheel_corrector"
_MARKER_LIFETIME = BuiltinDuration(sec=1, nanosec=0)


class WheelCorrectorNode(Node):
    """Relays JointGroupVelocityController commands, correcting nothing (yet)."""

    def __init__(self) -> None:
        super().__init__("wheel_corrector")

        # Number of wheel setpoints we expect per message. Mismatches are
        # logged once (throttled) but still forwarded -- a dummy must never
        # silently drop a command, since the controller latches the last one.
        self.declare_parameter("expected_size", 4)
        # Frames + corridor width are debug-only; corridor_epsilon just sets
        # the visual tube half-width, it gates nothing here.
        self.declare_parameter("planning_frame", "map")
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("corridor_epsilon", 0.2)
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("debug_marker_rate", 5.0)
        # Seconds since the last relayed command before the state text flips
        # from PLAY back to IDLE.
        self.declare_parameter("idle_after", 0.5)

        self._expected_size = int(self.get_parameter("expected_size").value)
        self._planning_frame = str(self.get_parameter("planning_frame").value)
        self._robot_frame = str(self.get_parameter("robot_frame").value)
        self._corridor_epsilon = float(self.get_parameter("corridor_epsilon").value)
        self._publish_debug = bool(self.get_parameter("publish_debug").value)
        self._idle_after = float(self.get_parameter("idle_after").value)
        marker_rate = float(self.get_parameter("debug_marker_rate").value)

        # The forward command controller subscribes with the rclcpp default
        # (reliable, volatile, keep-last). Match it on the output side so no
        # command is dropped; mirror it on the input side too.
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # Topics are private so the launch file can remap them explicitly:
        #   ~/wheel_cmd_in  <- planner / interpreter output
        #   ~/wheel_cmd_out -> /wheel_velocity_controller/commands
        #   ~/plan          <- planner nav_msgs/Path (for debug viz only)
        #   ~/debug_markers -> RViz MarkerArray
        self._pub = self.create_publisher(Float64MultiArray, "~/wheel_cmd_out", cmd_qos)
        self._sub = self.create_subscription(
            Float64MultiArray, "~/wheel_cmd_in", self._on_cmd, cmd_qos
        )

        # Cached plan polyline (x, y) and the last-command timestamp drive the
        # debug markers. TF gives the robot pose for the state-text anchor.
        self._plan_xy: List[Tuple[float, float]] = []
        self._last_cmd_time: Optional[rclpy.time.Time] = None

        if self._publish_debug:
            self._marker_pub = self.create_publisher(
                MarkerArray, "~/debug_markers", 10
            )
            self._plan_sub = self.create_subscription(
                Path, "~/plan", self._on_plan, 10
            )
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)
            period = 1.0 / marker_rate if marker_rate > 0.0 else 0.2
            self._marker_timer = self.create_timer(period, self._publish_debug_markers)

        self.get_logger().info(
            "wheel_corrector up (pass-through, expected_size=%d, debug=%s)"
            % (self._expected_size, self._publish_debug)
        )

    # ----------------------- Command relay ---------------------------------

    def _on_cmd(self, msg: Float64MultiArray) -> None:
        if len(msg.data) != self._expected_size:
            # Throttled so a persistently wrong upstream does not spam the log.
            self.get_logger().warn(
                "expected %d wheel setpoints, got %d -- forwarding as-is"
                % (self._expected_size, len(msg.data)),
                throttle_duration_sec=2.0,
            )

        corrected = self._correct(list(msg.data))

        # Reuse the incoming layout so any MultiArrayLayout metadata survives
        # the hop; only the data vector is (potentially) modified.
        out = Float64MultiArray()
        out.layout = msg.layout
        out.data = corrected
        self._pub.publish(out)
        self._last_cmd_time = self.get_clock().now()

    def _correct(self, cmd: List[float]) -> List[float]:
        """Apply the per-wheel correction. Identity for now.

        The real implementation will map (planned wheel cmd, deviation state,
        local context) -> corrected wheel cmd, one factor per wheel. Keeping
        the signature list-in/list-out means swapping it in touches nothing
        else in this node.
        """
        return cmd

    # ----------------------- Debug visualization ---------------------------

    def _on_plan(self, msg: Path) -> None:
        self._plan_xy = [
            (ps.pose.position.x, ps.pose.position.y) for ps in msg.poses
        ]

    def _robot_xy(self) -> Optional[Tuple[float, float]]:
        try:
            t = self._tf_buffer.lookup_transform(
                self._planning_frame, self._robot_frame, rclpy.time.Time()
            )
            return t.transform.translation.x, t.transform.translation.y
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _make_marker(self, mid: int, mtype: int, ns: str, action: int = Marker.ADD,
                     stamp=None) -> Marker:
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = self._planning_frame
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = action
        m.lifetime = _MARKER_LIFETIME
        m.pose.orientation.w = 1.0
        return m

    def _publish_debug_markers(self) -> None:
        """Plan-only subset of the runtime_corrector debug markers."""
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        # -- 0: plan centerline (thin blue LINE_STRIP) --
        if self._plan_xy:
            m = self._make_marker(0, Marker.LINE_STRIP, "centerline", stamp=stamp)
            m.scale.x = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 1.0
            m.points = [Point(x=x, y=y, z=0.0) for x, y in self._plan_xy]
            markers.markers.append(m)

            # -- 1: corridor tube (thick semi-transparent LINE_STRIP) --
            c = self._make_marker(1, Marker.LINE_STRIP, "corridor", stamp=stamp)
            c.scale.x = 2.0 * self._corridor_epsilon
            c.color.r, c.color.g, c.color.b, c.color.a = 0.2, 0.4, 1.0, 0.2
            c.points = [Point(x=x, y=y, z=0.0) for x, y in self._plan_xy]
            markers.markers.append(c)
        else:
            markers.markers.append(
                self._make_marker(0, Marker.LINE_STRIP, "centerline", Marker.DELETE, stamp)
            )
            markers.markers.append(
                self._make_marker(1, Marker.LINE_STRIP, "corridor", Marker.DELETE, stamp)
            )

        # -- 5: state text above the robot (PLAY while relaying, else IDLE) --
        t = self._make_marker(5, Marker.TEXT_VIEW_FACING, "state", stamp=stamp)
        robot = self._robot_xy()
        if robot is not None:
            t.pose.position.x, t.pose.position.y, t.pose.position.z = robot[0], robot[1], 0.5
        t.scale.z = 0.3
        if self._relaying():
            t.text = "PLAY (passthru)"
            t.color.r, t.color.g, t.color.b, t.color.a = 0.2, 1.0, 0.2, 1.0
        else:
            t.text = "IDLE"
            t.color.r, t.color.g, t.color.b, t.color.a = 0.7, 0.7, 0.7, 1.0
        markers.markers.append(t)

        self._marker_pub.publish(markers)

    def _relaying(self) -> bool:
        if self._last_cmd_time is None:
            return False
        dt = (self.get_clock().now() - self._last_cmd_time).nanoseconds * 1e-9
        return dt <= self._idle_after


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelCorrector()
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
