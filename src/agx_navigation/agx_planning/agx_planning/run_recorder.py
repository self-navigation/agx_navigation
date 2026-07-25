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
the control path: the only thing it publishes is an RViz marker of the true
pose, which nothing consumes. Keep it that way -- if a controller ever needs
this data, that controller is not deployable.

WHY IT DRAWS THE TRUE POSE
--------------------------
RViz places the robot at its TF pose, which is only as good as whatever provides
map->odom. Under `localization:=none` that is dead reckoning: odometry cannot
observe slip, so on a run that ended 0.71 m short of the plan the screen showed
the robot arriving exactly on target -- the display and the controller share one
belief, and neither can see the error. The green marker is ground truth; when it
separates from the robot model, that gap IS the localization error, and it is
the thing every screenshot was previously hiding. Under `localization:=truth`
the two coincide by construction, which is itself the check that the mode works.

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
  <output_dir>/<run_name>_track.csv    per-sample: t, true pose, odom pose,
                                       nearest planned point, signed
                                       cross-track, along-track
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
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from gz.msgs10.pose_v_pb2 import Pose_V
from nav_msgs.msg import Odometry, Path
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
        # The pose the CONTROLLER believes, for comparison against truth. This
        # is what RViz draws the robot at, and what the corrector corrects on.
        self.declare_parameter("odom_topic", "/odom/filtered")

        self.run_name = str(self.get_parameter("run_name").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.sample_period = float(self.get_parameter("sample_period").value)

        self._plan: List[Tuple[float, float]] = []
        self._rows: List[Tuple] = []
        self._goal: Optional[Tuple[float, float, float]] = None
        self._t0: Optional[float] = None
        self._last_sample_t = -1e9
        self._written = False
        self._odom: Optional[Tuple[float, float, float]] = None
        self._truth: Optional[Tuple[float, float, float]] = None

        self.create_subscription(
            Path, str(self.get_parameter("plan_topic").value), self._on_plan, 10)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.VOLATILE)
        self.create_subscription(
            PoseStamped, str(self.get_parameter("goal_topic").value),
            self._on_goal, qos)

        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 10)

        # Viz only -- see the module docstring. Nothing subscribes to this in
        # the control path, and nothing may.
        self._marker_pub = self.create_publisher(MarkerArray, "~/truth_markers", 10)
        self.create_timer(0.1, self._publish_truth_marker)

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

    def _on_odom(self, msg: Odometry):
        q = msg.pose.pose.orientation
        self._odom = (msg.pose.pose.position.x, msg.pose.pose.position.y,
                      yaw_from_quat(q.x, q.y, q.z, q.w))

    def _on_goal(self, msg: PoseStamped):
        if msg.header.frame_id == "":
            # Completion sentinel -- fires on any terminal outcome.
            self._write()
            return
        q = msg.pose.orientation
        self._goal = (msg.pose.position.x, msg.pose.position.y,
                      yaw_from_quat(q.x, q.y, q.z, q.w))

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

            self._truth = (x, y, yaw)
            ox, oy, oyaw = self._odom if self._odom is not None else (
                float("nan"), float("nan"), float("nan"))

            if len(self._plan) >= 2:
                nx, ny, cross, along = nearest_on_polyline((x, y), self._plan)
            else:
                nx = ny = cross = along = float("nan")

            self._rows.append((rel_t, x, y, yaw, ox, oy, oyaw, nx, ny, cross, along))
            return

    def _publish_truth_marker(self):
        """Draw ground truth in green. The gap to the robot model is the
        odometry error -- the thing a screenshot otherwise cannot show."""
        if self._truth is None:
            return
        x, y, yaw = self._truth
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "ground_truth"
        m.id = 0
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.points = [Point(x=x, y=y, z=0.05),
                    Point(x=x + 0.4 * math.cos(yaw), y=y + 0.4 * math.sin(yaw), z=0.05)]
        m.scale.x, m.scale.y, m.scale.z = 0.06, 0.12, 0.0
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 1.0, 0.2, 1.0
        arr.markers.append(m)

        t = Marker()
        t.header.frame_id = "map"
        t.header.stamp = m.header.stamp
        t.ns = "ground_truth"
        t.id = 1
        t.type = Marker.TEXT_VIEW_FACING
        t.action = Marker.ADD
        # Offset in the GROUND PLANE, not in z: the fixture is watched from
        # directly overhead, where a label raised in z lands on the very point
        # it annotates. Each label gets its own y offset so they cannot stack.
        t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y - 0.6, 0.35
        t.scale.z = 0.16
        t.color.r, t.color.g, t.color.b, t.color.a = 0.1, 1.0, 0.2, 1.0
        if self._odom is not None:
            t.text = "truth d%.2f" % math.hypot(
                x - self._odom[0], y - self._odom[1])
        else:
            t.text = "truth"
        arr.markers.append(t)

        # The GOAL that was asked for, and -- separately -- where the PLAN ends.
        # They are not the same point: the planner terminates inside a goal ball
        # and its last sample sits ~0.05 m short, consistently across every run
        # measured so far. Drawing both splits a miss into the part the planner
        # conceded and the part the controller lost, which one marker cannot do.
        if self._goal is not None:
            gx, gy, gyaw = self._goal
            g = Marker()
            g.header.frame_id = "map"
            g.header.stamp = m.header.stamp
            g.ns = "goal"
            g.id = 2
            g.type = Marker.ARROW
            g.action = Marker.ADD
            g.points = [
                Point(x=gx, y=gy, z=0.05),
                Point(x=gx + 0.5 * math.cos(gyaw),
                      y=gy + 0.5 * math.sin(gyaw), z=0.05),
            ]
            g.scale.x, g.scale.y, g.scale.z = 0.08, 0.16, 0.0
            g.color.r, g.color.g, g.color.b, g.color.a = 1.0, 0.1, 0.1, 1.0
            arr.markers.append(g)

            gt = Marker()
            gt.header.frame_id = "map"
            gt.header.stamp = m.header.stamp
            gt.ns = "goal"
            gt.id = 3
            gt.type = Marker.TEXT_VIEW_FACING
            gt.action = Marker.ADD
            gt.pose.position.x, gt.pose.position.y = gx, gy + 0.6
            gt.pose.position.z = 0.35
            gt.scale.z = 0.16
            gt.color.r, gt.color.g, gt.color.b, gt.color.a = 1.0, 0.1, 0.1, 1.0
            # Distance from GROUND TRUTH, not from the robot model: under
            # localization:=none those differ by however far odometry has
            # drifted, and the true number is the one worth reading.
            gt.text = "goal d%.2f" % math.hypot(x - gx, y - gy)
            arr.markers.append(gt)

        if len(self._plan) >= 1:
            px, py = self._plan[-1]
            p = Marker()
            p.header.frame_id = "map"
            p.header.stamp = m.header.stamp
            p.ns = "plan_end"
            p.id = 4
            p.type = Marker.SPHERE
            p.action = Marker.ADD
            p.pose.position.x, p.pose.position.y, p.pose.position.z = px, py, 0.05
            p.scale.x = p.scale.y = p.scale.z = 0.18
            p.color.r, p.color.g, p.color.b, p.color.a = 1.0, 0.6, 0.0, 0.9
            arr.markers.append(p)

        self._marker_pub.publish(arr)

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
                        "odom_x", "odom_y", "odom_yaw",
                        "plan_x", "plan_y", "cross_track", "along_track"])
            for r in self._rows:
                w.writerow([f"{r[0]:.3f}"] + [f"{v:.5f}" for v in r[1:]])

        with open(base + "_plan.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["plan_x", "plan_y"])
            for x, y in self._plan:
                w.writerow([f"{x:.5f}", f"{y:.5f}"])

        crosses = [abs(r[9]) for r in self._rows if not math.isnan(r[9])]
        rms = (sum(c * c for c in crosses) / len(crosses)) ** 0.5 if crosses else float("nan")
        peak = max(crosses) if crosses else float("nan")
        last = self._rows[-1]
        final_err = (math.hypot(last[1] - self._goal[0], last[2] - self._goal[1])
                     if self._goal else float("nan"))

        # The odometry error is the gap between what the controller (and RViz)
        # believe and what actually happened. When it is the same size as the
        # final error, no amount of feedback ON odometry can close the miss.
        odom_err = (math.hypot(last[1] - last[4], last[2] - last[5])
                    if not math.isnan(last[4]) else float("nan"))
        odom_final_err = (math.hypot(last[4] - self._goal[0], last[5] - self._goal[1])
                          if self._goal and not math.isnan(last[4]) else float("nan"))

        summary = (
            f"run:             {self.run_name}\n"
            f"samples:         {len(self._rows)}\n"
            f"duration:        {last[0]:.2f} s (sim)\n"
            f"planned points:  {len(self._plan)}\n"
            f"goal:            {self._goal}\n"
            f"final true pose: ({last[1]:.3f}, {last[2]:.3f}) yaw {last[3]:.3f}\n"
            f"final odom pose: ({last[4]:.3f}, {last[5]:.3f}) yaw {last[6]:.3f}\n"
            f"final error:     {final_err:.3f} m\n"
            f"final err(odom): {odom_final_err:.3f} m  <- what the robot believes\n"
            f"odom drift:      {odom_err:.3f} m  <- truth vs odom\n"
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
