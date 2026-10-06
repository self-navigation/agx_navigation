#!/usr/bin/env python3
"""Serve a STORED library plan over the PlanToGoal action (compare_run `ours-lib`).

Evaluation tool, not part of the stack. It stands in for pmp_planner (offline
mode) so that the unchanged runtime_corrector plays back and corrects the SAME
library trajectory the Nav2 Type-A arms (pmp-mppi / pmp-rpp) follow -- a
controllers-only comparison. Without it our arm re-solved the plan live with
the stack's planner settings, and 5/40 broad pairs planner-failed (#39).

Start the stack with `use_server:=true` (vec_pmp.launch.py's existing switch:
pmp_planner is not launched), then run this in the same partition:

    python3 tools/library_plan_server.py --plan X.npz [--status out.json]

It serves `/pmp_planner/plan_to_goal`, the corrector's default action name.
On each goal it streams the npz as feedback chunks exactly shaped like
pmp_planner's `_publish_chunk_feedback` (poses, wheel_left/right, accels,
costates), publishes the cumulative Path on /optimal_trajectory (what the
planner's remap feeds), and succeeds. The goal's start/target are NOT used to
plan -- the plan is what it is; their distance to the plan's start/goal is
logged and written to --status so a start jump is visible in the row.

npz keys used: poses (N,3), wheel_cmds (N,2) [rad/s, the published
setpoints], costates (N,5), dt_sample. Accelerations are not stored; they are
reconstructed as the forward difference of wheel_cmds (last = 0). The
corrector uses only poses, wheel speeds and costates.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import rclpy
from rclpy.action import ActionServer, GoalResponse
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from agx_planning_msgs.action import PlanToGoal


def load(path: str) -> dict:
    with np.load(path) as f:
        poses = np.asarray(f["poses"], dtype=float)
        wheels = np.asarray(f["wheel_cmds"], dtype=float)
        lam = (np.asarray(f["costates"], dtype=float) if "costates" in f
               else np.zeros((poses.shape[0], 5)))
        dt = float(f["dt_sample"]) if "dt_sample" in f else 0.1
    acc = np.zeros_like(wheels)
    acc[:-1] = np.diff(wheels, axis=0) / dt
    return {"poses": poses, "wheels": wheels, "acc": acc, "lam": lam, "dt": dt}


class LibraryPlanServer(Node):
    def __init__(self, plan_path: str, chunk: int, frame: str, status: str | None):
        super().__init__("library_plan_server")
        self.plan = load(plan_path)
        self.plan_path = plan_path
        self.chunk = max(1, int(chunk))
        self.frame = frame
        self.status = status
        self.traj_id = 0
        fb_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=64,
                            reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.VOLATILE)
        self.path_pub = self.create_publisher(Path, "/optimal_trajectory", 5)
        self.server = ActionServer(
            self, PlanToGoal, "/pmp_planner/plan_to_goal",
            execute_callback=self._execute,
            goal_callback=lambda _g: GoalResponse.ACCEPT,
            feedback_pub_qos_profile=fb_qos)
        n = self.plan["poses"].shape[0]
        self.get_logger().info(f"serving {plan_path}: {n} samples, dt={self.plan['dt']}")

    def _execute(self, gh):
        g = gh.request
        p = self.plan
        res = PlanToGoal.Result()
        if g.frame_id != self.frame:
            gh.abort()
            res.success, res.message, res.trajectory_id = False, f"frame {g.frame_id!r} != {self.frame!r}", 0
            return res
        self.traj_id += 1
        tid = self.traj_id
        p0, pN = p["poses"][0], p["poses"][-1]
        start_off = math.hypot(g.start_x - p0[0], g.start_y - p0[1])
        start_dyaw = math.atan2(math.sin(g.start_theta - p0[2]), math.cos(g.start_theta - p0[2]))
        target_off = math.hypot(g.target_x - pN[0], g.target_y - pN[1])
        self.get_logger().info(
            f"goal traj {tid}: start offset {start_off:.3f} m / {math.degrees(start_dyaw):.1f} deg "
            f"from plan start, target offset {target_off:.3f} m")
        if self.status:
            with open(self.status, "w") as fh:
                json.dump({"traj_id": tid, "start_offset_m": start_off,
                           "start_dyaw_rad": start_dyaw, "target_offset_m": target_off,
                           "n_samples": int(p["poses"].shape[0]),
                           "plan_wheel0": [float(v) for v in p["wheels"][0]]}, fh)
        n = p["poses"].shape[0]
        f32 = lambda a: [float(v) for v in a]  # noqa: E731
        for ci, s in enumerate(range(0, n, self.chunk)):
            e = min(n, s + self.chunk)
            fb = PlanToGoal.Feedback()
            fb.trajectory_id, fb.chunk_index, fb.dt = tid, ci, float(p["dt"])
            fb.pose_x, fb.pose_y, fb.pose_theta = (f32(p["poses"][s:e, k]) for k in range(3))
            fb.wheel_left, fb.wheel_right = f32(p["wheels"][s:e, 0]), f32(p["wheels"][s:e, 1])
            fb.accel_left, fb.accel_right = f32(p["acc"][s:e, 0]), f32(p["acc"][s:e, 1])
            (fb.lam_x, fb.lam_y, fb.lam_theta,
             fb.lam_wheel_left, fb.lam_wheel_right) = (f32(p["lam"][s:e, k]) for k in range(5))
            gh.publish_feedback(fb)
            time.sleep(0.01)  # do not burst past the subscriber's queue
        path = Path()
        path.header.frame_id = self.frame
        path.header.stamp = self.get_clock().now().to_msg()
        for x, y, th in p["poses"]:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x, ps.pose.position.y = float(x), float(y)
            ps.pose.orientation.z, ps.pose.orientation.w = math.sin(th / 2), math.cos(th / 2)
            path.poses.append(ps)
        self.path_pub.publish(path)
        time.sleep(0.2)  # let the last feedback land before the result
        gh.succeed()
        res.success, res.message, res.trajectory_id = True, "Library plan served", tid
        return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--chunk", type=int, default=5, help="samples per feedback chunk")
    ap.add_argument("--frame", default="map")
    ap.add_argument("--status", help="write the last goal's start/target offsets here (JSON)")
    args = ap.parse_args()
    rclpy.init()
    node = LibraryPlanServer(args.plan, args.chunk, args.frame, args.status)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
