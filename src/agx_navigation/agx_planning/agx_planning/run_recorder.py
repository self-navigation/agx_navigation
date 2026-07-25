#!/usr/bin/env python3
"""Record a run to CSV: planned path, true path, and tracking error over time.

Produces the numbers and the plot data for "how well did the controller hold the
trajectory" -- one row per ground-truth pose sample, plus a one-line summary and
a copy of the planned path. Intended for the static-map fixture, where the map
is identical every run and two runs are therefore comparable.

WHY GROUND TRUTH, AND WHY THAT IS ACCEPTABLE HERE
-------------------------------------------------
Tracking error is measured against Gazebo ground truth, never /odom. Wheel
odometry integrates the ideal differential-drive relation and cannot observe
slip, so scoring against it measures the controller against the same model the
controller already believes -- it reported 7 mm of cross-track error on a run
that ended metres off course. Ground truth is the only honest referee.

That makes this node SIM-ONLY, deliberately. It is an instrument, not part of
the control path: nothing it publishes feeds back into planning or control, so
it cannot create a dependency that fails to exist on the real robot. Keep it
that way -- if a controller ever needs this data, that controller is not
deployable.

WHAT CROSS-TRACK MEANS HERE
---------------------------
Distance from the true position to the nearest point on the planned polyline,
signed by which side of the path the robot is on (positive = left of the
direction of travel). Nearest-point rather than same-timestamp comparison is
deliberate: a robot that drives the right path one second late is doing
something quite different from one that drives the wrong path on time, and the
signed nearest-point distance separates them. `along_track` records progress
along the path, so lag is still visible as along_track falling behind time.

OUTPUT
  <output_dir>/<run_name>_track.csv    per-sample: t, true pose, nearest planned
                                       point, signed cross-track, along-track
  <output_dir>/<run_name>_plan.csv     the planned path as x,y (for plotting)
  <output_dir>/<run_name>_summary.txt  rms/max/final error and run metadata

USAGE
  ros2 run agx_planning run_recorder --ros-args -p use_sim_time:=true \\
      -p run_name:=tvlqr_goal1

Start it before sending the goal. It writes on completion (the corrector's
cleared-goal sentinel) and on shutdown, so Ctrl-C still yields a usable file.
"""

import csv
import math
import os
from typing import List, Optional, Tuple

import gz.transport13 as gz_transport
import rclpy
from geometry_msgs.msg import PoseStamped
from gz.msgs10.pose_v_pb2 import Pose_V
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def nearest_on_polyline(
    pt: Tuple[float, float], path: List[Tuple[float, float]]
) -> Tuple[float, float, float, float]:
    """Nearest point on a polyline to pt.

    Returns (nx, ny, signed_cross_track, along_track). The sign is positive when
    pt lies to the LEFT of the segment's direction of travel, so a sign flip in
    the log is a real side change rather than noise around zero.
    """
    best = (float("inf"), 0.0, 0.0, 0.0, 0.0)  # d2, nx, ny, cross, along
    cum = 0.0
    for i in range(len(path) - 1):
        ax, ay = path[i]
        bx, by = path[i + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 < 1e-12:
            continue
        t = ((pt[0] - ax) * dx + (pt[1] - ay) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        nx, ny = ax + t * dx, ay + t * dy
        ex, ey = pt[0] - nx, pt[1] - ny
        d2 = ex * ex + ey * ey
        if d2 < best[0]:
            seg_len = math.sqrt(seg_len2)
            # z of (segment direction) x (error vector): >0 means left.
            cross = math.copysign(math.sqrt(d2), dx * ey - dy * ex)
            best = (d2, nx, ny, cross, cum + t * seg_len)
        cum += math.sqrt(seg_len2)
    return best[1], best[2], best[3], best[4]


class RunRecorder(Node):

    def __init__(self):
        super().__init__("run_recorder")

        self.declare_parameter("run_name", "run")
        self.declare_parameter("output_dir", "/tmp/runs")
        self.declare_parameter("model_name", "scout_mini")
        self.declare_parameter("plan_topic", "/optimal_trajectory")
        self.declare_parameter("world_name", "ordjo_world")
        self.declare_parameter("goal_topic", "/goal_pose")
        # Ground truth arrives at the physics rate, which is far denser than
        # anything a plot needs and makes the CSV awkward to open.
        self.declare_parameter("sample_period", 0.05)

        self.run_name = str(self.get_parameter("run_name").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.sample_period = float(self.get_parameter("sample_period").value)

        self._plan: List[Tuple[float, float]] = []
        self._rows: List[Tuple] = []
        self._goal: Optional[Tuple[float, float]] = None
        self._t0: Optional[float] = None
        self._last_sample_t = -1e9
        self._written = False

        self.create_subscription(
            Path, str(self.get_parameter("plan_topic").value), self._on_plan, 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            PoseStamped, str(self.get_parameter("goal_topic").value),
            self._on_goal, qos)

        # Ground truth over gz-transport rather than a ros_gz_bridge topic.
        # The Pose_V -> TFMessage bridge drops the entity names (every
        # frame_id/child_frame_id comes through empty), so there is no way to
        # pick the robot out of the ~hundreds of entities. Matching on
        # Pose_V.pose[].name is what GazeboBridge already does.
        #
        # pose/info, NOT dynamic_pose/info: it carries every entity at a steady
        # rate, so the pose stays fresh while the robot is stopped, and it
        # preserves the names. dynamic_pose only carries entities that moved.
        world = str(self.get_parameter("world_name").value)
        self._topic_pose = f"/world/{world}/pose/info"
        self._gz = gz_transport.Node()
        if not self._gz.subscribe(Pose_V, self._topic_pose, self._on_truth):
            raise RuntimeError(f"could not subscribe to {self._topic_pose}")

        os.makedirs(self.output_dir, exist_ok=True)
        self.get_logger().info(
            f"recording '{self.run_name}' -> {self.output_dir} "
            f"(truth={self._topic_pose})")

    # ---- inputs ----------------------------------------------------------

    def _on_plan(self, msg: Path):
        # The planner republishes a growing path as chunks are solved; keep the
        # longest, which is the complete trajectory.
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if len(pts) > len(self._plan):
            self._plan = pts

    def _on_goal(self, msg: PoseStamped):
        if msg.header.frame_id == "":
            # Completion sentinel -- fires on any terminal outcome.
            self._write()
            return
        self._goal = (msg.pose.position.x, msg.pose.position.y)

    def _on_truth(self, msg: Pose_V):
        """Called on a gz-transport thread, not the rclpy executor.

        pose/info lists every entity (links, visuals, the ground, the sun); the
        top-level model pose is the one whose name matches exactly.
        """
        for p in msg.pose:
            if p.name != self.model_name:
                continue

            t = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
            if self._t0 is None:
                self._t0 = t
            rel_t = t - self._t0
            if rel_t - self._last_sample_t < self.sample_period:
                return
            self._last_sample_t = rel_t

            x = p.position.x
            y = p.position.y
            q = p.orientation
            yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

            if len(self._plan) >= 2:
                nx, ny, cross, along = nearest_on_polyline((x, y), self._plan)
            else:
                nx = ny = cross = along = float("nan")

            self._rows.append((rel_t, x, y, yaw, nx, ny, cross, along))
            return

    # ---- output ----------------------------------------------------------

    def _write(self):
        if self._written:
            return
        if not self._rows:
            self.get_logger().warn(
                "nothing recorded -- did ground truth arrive? Check that the "
                "dynamic_pose/info bridge is running and model_name matches.")
            return
        self._written = True

        base = os.path.join(self.output_dir, self.run_name)

        with open(base + "_track.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "true_x", "true_y", "true_yaw",
                        "plan_x", "plan_y", "cross_track", "along_track"])
            for r in self._rows:
                w.writerow([f"{r[0]:.3f}"] + [f"{v:.5f}" for v in r[1:]])

        with open(base + "_plan.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plan_x", "plan_y"])
            for x, y in self._plan:
                w.writerow([f"{x:.5f}", f"{y:.5f}"])

        crosses = [abs(r[6]) for r in self._rows if not math.isnan(r[6])]
        rms = (sum(c * c for c in crosses) / len(crosses)) ** 0.5 if crosses else float("nan")
        peak = max(crosses) if crosses else float("nan")
        last = self._rows[-1]
        final_err = (math.hypot(last[1] - self._goal[0], last[2] - self._goal[1])
                     if self._goal else float("nan"))

        summary = (
            f"run:             {self.run_name}\n"
            f"samples:         {len(self._rows)}\n"
            f"duration:        {last[0]:.2f} s (sim)\n"
            f"planned points:  {len(self._plan)}\n"
            f"goal:            {self._goal}\n"
            f"final true pose: ({last[1]:.3f}, {last[2]:.3f}) yaw {last[3]:.3f}\n"
            f"final error:     {final_err:.3f} m\n"
            f"cross-track rms: {rms:.4f} m\n"
            f"cross-track max: {peak:.4f} m\n"
        )
        with open(base + "_summary.txt", "w") as f:
            f.write(summary)

        self.get_logger().info("\n" + summary + f"written to {base}_*.csv")

    def destroy_node(self):
        self._write()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RunRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
