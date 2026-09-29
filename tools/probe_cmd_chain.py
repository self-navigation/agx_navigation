#!/usr/bin/env python3
"""Sample each hop of the nav2 command chain at 2 Hz (sim time), for tuning.

    tools/with-worker 1 python3 tools/probe_cmd_chain.py --secs 120 > /tmp/chain.tsv

Columns: sim t, then (v, w) at cmd_vel_nav (controller), cmd_vel_smoothed
(velocity smoother), cmd_vel (collision monitor -> twist_to_wheels), odom twist,
and collision_monitor's action. Answers "which hop eats the speed".
"""
import argparse
import sys

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


class Probe(Node):
    def __init__(self):
        super().__init__("probe_cmd_chain", parameter_overrides=[
            rclpy.parameter.Parameter("use_sim_time", value=True)])
        self.last = {k: (float("nan"), float("nan"))
                     for k in ("nav", "smooth", "out", "odom")}
        self.state = "-"
        for key, topic in (("nav", "/cmd_vel_nav"), ("smooth", "/cmd_vel_smoothed"),
                           ("out", "/cmd_vel")):
            self.create_subscription(TwistStamped, topic,
                                     lambda m, k=key: self._tw(k, m.twist), 10)
        self.create_subscription(Odometry, "/odom",
                                 lambda m: self._tw("odom", m.twist.twist), 10)
        try:
            from nav2_msgs.msg import CollisionMonitorState
            self.create_subscription(CollisionMonitorState, "/collision_monitor_state",
                                     self._cm, 10)
        except ImportError:
            pass

    def _tw(self, key, t):
        self.last[key] = (t.linear.x, t.angular.z)

    def _cm(self, m):
        self.state = f"{m.action_type}:{m.polygon_name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=float, default=120.0)
    a = ap.parse_args()
    rclpy.init()
    n = Probe()
    print("t\tnav_v\tnav_w\tsm_v\tsm_w\tout_v\tout_w\todom_v\todom_w\tcm", flush=True)
    t0 = None
    nxt = 0.0
    while rclpy.ok():
        rclpy.spin_once(n, timeout_sec=0.05)
        now = n.get_clock().now().nanoseconds * 1e-9
        if now == 0.0:
            continue
        t0 = now if t0 is None else t0
        if now - t0 >= a.secs:
            break
        if now - t0 >= nxt:
            nxt += 0.5
            vals = [f"{x:+.3f}" for k in ("nav", "smooth", "out", "odom") for x in n.last[k]]
            print(f"{now - t0:.1f}\t" + "\t".join(vals) + f"\t{n.state}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
