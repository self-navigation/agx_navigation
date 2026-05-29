"""RViz debug marker publishing for the trajectory corrector."""

from __future__ import annotations

from typing import Optional

from builtin_interfaces.msg import Duration as BuiltinDuration
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from agx_planning.runtime_corrector.config import CorrectorConfig
from agx_planning.runtime_corrector.strategies import State

_MARKER_NS = "pmp_trajectory_corrector"
_MARKER_LIFETIME = BuiltinDuration(sec=1, nanosec=0)


class TrajectoryVisualizer:
    """Builds and publishes RViz debug markers for the trajectory corrector.

    Accepts a ROS2 publisher and the algorithm config; the node calls
    publish() on each tick, passing in the data it already has.
    """

    def __init__(self, publisher, cfg: CorrectorConfig) -> None:
        self._pub = publisher
        self._cfg = cfg

    def publish(
        self,
        path: list[tuple[float, float, float]],
        frame: str,
        proj: Optional[tuple[float, float]],
        carrot: Optional[tuple[float, float]],
        state: State,
        robot_pose: Optional[tuple[float, float, float]],
        strat_name: Optional[str],
        stamp,
    ) -> None:
        """Publish all debug markers for one tick.

        path       -- ordered (x, y, theta) polyline, or empty list
        frame      -- TF frame id for the marker header
        proj       -- (x, y) nearest projection point, or None
        carrot     -- (x, y) look-ahead carrot point, or None
        state_name -- string name of the current _State enum value
        robot_pose -- (x, y, theta) of the robot, or None if TF failed
        strat_name -- Strategy name for correcting
        stamp      -- ROS2 timestamp (from node.get_clock().now().to_msg())
        """
        markers = MarkerArray()

        def _make(
            mid: int, mtype: int, action: int = Marker.ADD, namespace: str = _MARKER_NS
        ) -> Marker:
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = frame
            m.ns = namespace
            m.id = mid
            m.type = mtype
            m.action = action
            m.lifetime = _MARKER_LIFETIME
            m.pose.orientation.w = 1.0
            return m

        # -- 0: path centerline (thin blue LINE_STRIP) --
        if path:
            m = _make(0, Marker.LINE_STRIP, namespace="centerline")
            m.scale.x = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 1.0
            m.points = [Point(x=x, y=y, z=0.0) for x, y, _ in path]
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(0, Marker.LINE_STRIP, Marker.DELETE, namespace="centerline")
            )

        # -- 1: corridor tube (thick semi-transparent LINE_STRIP) --
        if path:
            m = _make(1, Marker.LINE_STRIP, namespace="corridor")
            m.scale.x = 2.0 * self._cfg.corridor_epsilon
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.4, 1.0, 0.2
            m.points = [Point(x=x, y=y, z=0.0) for x, y, _ in path]
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(1, Marker.LINE_STRIP, Marker.DELETE, namespace="corridor")
            )

        # -- 2: nearest projection point (yellow sphere) --
        if proj is not None:
            m = _make(2, Marker.SPHERE, namespace="projection")
            m.pose.position.x, m.pose.position.y = proj[0], proj[1]
            m.scale.x = m.scale.y = m.scale.z = 0.15
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 1.0, 0.0, 1.0
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(2, Marker.SPHERE, Marker.DELETE, namespace="projection")
            )

        # -- 3: look-ahead carrot (green sphere) --
        if carrot is not None:
            m = _make(3, Marker.SPHERE, namespace="carrot")
            m.pose.position.x, m.pose.position.y = carrot[0], carrot[1]
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 1.0
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(3, Marker.SPHERE, Marker.DELETE, namespace="carrot")
            )

        # -- 4: look-ahead arrow (projection → carrot) --
        if proj is not None and carrot is not None:
            m = _make(4, Marker.ARROW, namespace="lookahead")
            m.points = [
                Point(x=proj[0], y=proj[1], z=0.0),
                Point(x=carrot[0], y=carrot[1], z=0.0),
            ]
            m.scale.x = 0.04
            m.scale.y = 0.08
            m.scale.z = 0.10
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 0.4, 1.0
            markers.markers.append(m)
        else:
            markers.markers.append(
                _make(4, Marker.ARROW, Marker.DELETE, namespace="lookahead")
            )

        # -- 5: state text above robot (colour-coded) --
        m = _make(5, Marker.TEXT_VIEW_FACING, namespace="state")
        if robot_pose is not None:
            m.pose.position.x = robot_pose[0]
            m.pose.position.y = robot_pose[1]
            m.pose.position.z = 0.5
        m.scale.z = 0.3
        if state == State.IDLE:
            m.text = "IDLE"
            m.color.r, m.color.g, m.color.b, m.color.a = 0.7, 0.7, 0.7, 1.0
        elif state == State.PLAYING:
            m.text = "PLAY"
            m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 1.0, 0.2, 1.0
        else:
            m.text = "FIX " + str(strat_name)
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.5, 0.0, 1.0
        markers.markers.append(m)

        self._pub.publish(markers)
