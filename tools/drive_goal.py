#!/usr/bin/env python3
"""Publish one goal to a running fixture, wait for it to finish, and report.

WHY. Driving the fixture by hand is a three-trap operation, all three of which
fail SILENTLY, and all three are documented in CLAUDE.md as things that have
already cost measurements:

  1. Publishing a goal RACES DISCOVERY. /goal_pose has several subscribers and
     a single message reaches only those already matched. Lose vector_field and
     you get 'Timeout waiting for vector field'; lose runtime_corrector and
     nothing drives -- and the publisher sees a successful publish either way.
     This waits for the subscriber count and then settles, which is what
     `ros2 topic pub -w <n>` plus random_goals' `settle` delay do together.

  2. The completion sentinel on /goal_pose means "nobody is pursuing a goal",
     NOT "arrived". It fires on any terminal outcome. So this reports the final
     distance to the goal as the outcome, and calls it a success only against
     an explicit tolerance -- never on the sentinel alone.

  3. Ground truth, never /odom. Wheel odometry over-reports distance by
     0.6-0.7 m over one fixture run, so scoring arrival from /odom is scoring
     the error being measured. Pose comes from the map->base_link TRANSFORM,
     which under `localization:=truth` is Gazebo's own pose -- the same signal
     the corrector is fed, arriving exactly where a real estimator's would.

Usage (on the VM, inside the stack's partition):

    tools/with-worker 5 python3 tools/drive_goal.py --x 12.5 --y 3.0
    tools/with-worker 5 python3 tools/drive_goal.py --x 12.5 --y 3.0 --json

Exit 0 = arrived within tolerance, 1 = finished but short/off, 2 = timeout or
never started. `--json` prints one object for an automated caller.

NOTE ON TIME. Every timeout here is SIM time, taken from /clock. `make fixture`
runs at whatever real-time factor the box can manage, and a wall-clock timeout
would abort a healthy slow run and pass a fast broken one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from rosgraph_msgs.msg import Clock
from tf2_msgs.msg import TFMessage


# MUST match the stack's own /goal_pose profile: RELIABLE + VOLATILE, depth 1
# (runtime_corrector/node.py and random_goals.py both declare exactly this).
#
# This is not a detail. A TRANSIENT_LOCAL subscriber is INCOMPATIBLE with a
# VOLATILE publisher, so it receives nothing at all -- rclpy logs one warning
# and then behaves exactly like a goal that never completed. The first run of
# this tool did that: it published the goal successfully, the robot drove, and
# the tool sat waiting for a sentinel it could not physically receive. A wrong
# QoS here does not look like a QoS problem, it looks like a stack problem.
GOAL_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
)


class GoalDriver(Node):
    def __init__(self, goal_xy, tolerance: float):
        super().__init__("drive_goal")
        self.goal_xy = goal_xy
        self.tolerance = tolerance

        self._sim_t: float | None = None
        self._pose: tuple[float, float] | None = None
        self._map_to_odom: tuple[float, float] | None = None
        self._odom_to_base: tuple[float, float] | None = None
        self._finished = False
        self._track: list[tuple[float, float, float]] = []

        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 100)
        self.create_subscription(PoseStamped, "/goal_pose", self._on_goal_echo, GOAL_QOS)
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", GOAL_QOS)

    def _on_clock(self, msg: Clock) -> None:
        self._sim_t = msg.clock.sec + msg.clock.nanosec * 1e-9

    def _on_tf(self, msg: TFMessage) -> None:
        # Compose map->odom with odom->base rather than requiring a single
        # map->base transform: the fixture publishes them as two links (the
        # localization provides the first, wheel odometry the second) and no
        # publisher ever emits the composition.
        for tr in msg.transforms:
            parent = tr.header.frame_id.lstrip("/")
            child = tr.child_frame_id.lstrip("/")
            t = tr.transform.translation
            q = tr.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                             1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            if parent == "map" and child == "odom":
                self._map_to_odom = (t.x, t.y, yaw)
            elif parent == "odom" and child in ("base_link", "base_footprint"):
                self._odom_to_base = (t.x, t.y, yaw)
        if self._map_to_odom and self._odom_to_base:
            # Compose with ROTATION, not by adding translations.
            #
            # The first version added them, on the reasoning that map->odom is
            # "identity or truth" and a distance report does not need the
            # heading. That is wrong twice: under `localization:=truth` map->odom
            # carries whatever rotation corrects accumulated odometry drift, and
            # the odom->base translation must be rotated INTO the map frame
            # before it means anything. The error grows with both the drift and
            # the distance from the odom origin, so it is smallest exactly where
            # it is checked by hand (near the start) and largest at the goal.
            # Its visible symptom was a reported path length of 15 m over 4.6
            # sim-seconds -- 3.3 m/s, well above what the chassis can do.
            mx, my, myaw = self._map_to_odom
            ox, oy, _ = self._odom_to_base
            c, s = math.cos(myaw), math.sin(myaw)
            self._pose = (mx + c * ox - s * oy, my + s * ox + c * oy)
            if self._sim_t is not None:
                self._track.append((self._sim_t, *self._pose))

    def _on_goal_echo(self, msg: PoseStamped) -> None:
        # The empty-frame_id sentinel: some terminal outcome was reached.
        if msg.header.frame_id == "":
            self._finished = True

    def wait_for_subscribers(self, want: int, timeout_s: float) -> int:
        import time

        deadline = time.monotonic() + timeout_s
        n = 0
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            n = self.count_subscribers("/goal_pose")
            if n >= want:
                # Matched is not the same as ready -- random_goals carries a
                # `settle` delay for exactly this reason.
                settle_end = time.monotonic() + 1.5
                while time.monotonic() < settle_end:
                    rclpy.spin_once(self, timeout_sec=0.1)
                return n
        return n

    def publish_goal(self, frame: str) -> None:
        msg = PoseStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(self.goal_xy[0])
        msg.pose.position.y = float(self.goal_xy[1])
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def distance_to_goal(self) -> float | None:
        if self._pose is None:
            return None
        return math.hypot(self._pose[0] - self.goal_xy[0], self._pose[1] - self.goal_xy[1])

    def run(self, sim_timeout: float) -> dict:
        import time

        start_sim = self._sim_t
        wall0 = time.monotonic()
        while True:
            rclpy.spin_once(self, timeout_sec=0.1)
            if self._finished:
                # Let the last few transforms land before scoring.
                end = time.monotonic() + 1.0
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.05)
                break
            if start_sim is None:
                start_sim = self._sim_t
            elif self._sim_t is not None and self._sim_t - start_sim > sim_timeout:
                return self._result("timeout", start_sim, wall0)
            # A wall-clock backstop, generous, for the case /clock itself stops.
            if time.monotonic() - wall0 > max(600.0, sim_timeout * 20):
                return self._result("clock stalled", start_sim, wall0)
        return self._result("finished", start_sim, wall0)

    def _result(self, outcome: str, start_sim, wall0) -> dict:
        import time

        dist = self.distance_to_goal()
        travelled = 0.0
        for a, b in zip(self._track, self._track[1:]):
            travelled += math.hypot(b[1] - a[1], b[2] - a[2])
        return {
            "outcome": outcome,
            "arrived": bool(dist is not None and dist <= self.tolerance),
            "final_err": None if dist is None else round(dist, 4),
            "goal": list(self.goal_xy),
            "end_pose": None if self._pose is None else [round(v, 4) for v in self._pose],
            "path_length": round(travelled, 3),
            "sim_time_s": None
            if (start_sim is None or self._sim_t is None)
            else round(self._sim_t - start_sim, 2),
            "wall_time_s": round(time.monotonic() - wall0, 1),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--x", type=float, required=True)
    ap.add_argument("--y", type=float, required=True)
    ap.add_argument("--frame", default="map")
    ap.add_argument("--tolerance", type=float, default=0.2,
                    help="metres; arrival is looser than goal_tolerance_xy by design")
    ap.add_argument("--sim-timeout", type=float, default=180.0, help="SIM seconds, not wall")
    ap.add_argument("--subscribers", type=int, default=2,
                    help="wait for this many /goal_pose subscribers before publishing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rclpy.init()
    node = GoalDriver((args.x, args.y), args.tolerance)
    try:
        n = node.wait_for_subscribers(args.subscribers, timeout_s=30.0)
        if n < args.subscribers:
            out = {"outcome": "no subscribers", "arrived": False, "goal_pose_subscribers": n}
            print(json.dumps(out) if args.json else
                  f"only {n} subscriber(s) on /goal_pose (wanted {args.subscribers}) -- "
                  "the stack is not ready; run tools/stack_ready.py")
            return 2
        node.publish_goal(args.frame)
        result = node.run(args.sim_timeout)
        result["goal_pose_subscribers"] = n
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{result['outcome']}: final_err={result['final_err']} m "
              f"(tolerance {args.tolerance}), path {result['path_length']} m, "
              f"{result['sim_time_s']} sim-s / {result['wall_time_s']} wall-s")
    if result["outcome"] in ("timeout", "clock stalled"):
        return 2
    return 0 if result["arrived"] else 1


if __name__ == "__main__":
    sys.exit(main())
