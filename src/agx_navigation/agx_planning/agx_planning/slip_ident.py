#!/usr/bin/env python3
"""Identify skid-steer slip (chi) and the command gains, in sim OR on the robot.

Companion to calibrator.py, which identifies the chassis step response
(chassis_gain_*, chassis_tau_*, the BVP bounds) by comparing COMMANDS against
/odom. That comparison has a blind spot this script exists to cover.

WHY A SECOND SCRIPT
-------------------
A skid-steer loses part of its ideal yaw to lateral tyre scrub: turning the
wheels at a given differential rotates the body LESS than the textbook
differential-drive relation predicts. The planner models this as

    omega_actual = omega_ideal / chi,        omega_ideal = r*(w_r - w_l)/track

with chi >= 1 (PlannerConfig.slip_chi, mirrored in planner.launch.py).

agx_chassis/wheel_odometry, however, integrates heading with the IDEAL relation
and no slip term. So /odom's yaw is not a measurement of rotation -- it is a
PREDICTION of rotation from wheel speeds. Comparing commands against it cannot
see chi, because the same missing slip term sits on both sides and cancels.
calibrator.py is therefore structurally unable to identify chi, and a chi fitted
that way would come out at 1.0 no matter what the robot physically did.

THE REFERENCE
-------------
The gyro. It measures body rotation directly, owes nothing to the wheels, and
-- unlike Gazebo ground truth -- exists on the real robot. So

    chi = (yaw change reported by wheel odometry) / (yaw change measured by gyro)

is identifiable with only sensors the deployed robot carries, which is what lets
the same script run in both places and produce comparable numbers. Divergence
between the sim value and the real value is then a statement about the sim's
friction, not about the method.

Two consequences worth stating, because they are easy to get wrong:

  - Subscribe to RAW /odom, never /odometry/filtered. The EKF fuses the IMU, so
    filtered yaw is already part gyro; measuring against it would cancel a
    varying fraction of the very bias being measured and yield chi -> 1.
  - Drive ARCS, not spins in place. A pure spin is the worst-conditioned case
    for a skid-steer (all four contact patches scrubbing, chi strongly
    load-dependent) and is not what the planner spends its time doing. Arcs at
    several radii show whether chi is usefully constant or needs a schedule.

COMMAND MODES
-------------
  twist  (default) -- publishes (v, omega) on cmd_topic. Works in sim AND on the
                      robot, whose chassis accepts only (v, omega).
  wheels           -- publishes the 4 wheel velocities directly. SIM ONLY, and
                      the only mode that isolates chi from the chassis's own
                      internal twist->wheel conversion. Prefer it in sim.

In twist mode the measured ratio folds chi together with whatever the chassis
controller does internally, so it is reported as `yaw_gain` rather than chi.
That number is still the one that matters for control -- it is what the robot
actually does with a commanded yaw rate -- but only wheels mode yields chi as
the planner defines it.

TIME
----
Everything here runs on the ROS clock, never the wall clock. `make rl-sim` is
headless and unthrottled -- it runs at roughly 30x realtime -- so a phase timed
with time.monotonic() would last ~30x longer in the world than intended, while
the gyro (integrated on message timestamps, i.e. sim time) measured the real
duration. The first run of this script did exactly that and reported chi ~= 0.01
with a yaw_gain of ~21: not slip, just the realtime factor. Pass
use_sim_time:=true in simulation.

USAGE
  ros2 run agx_planning slip_ident --ros-args -p use_sim_time:=true \\
      -p cmd_mode:=wheels -p odom_topic:=/odom -p imu_topic:=/imu/data

Clear a few metres around the robot. Every test returns to its starting pose by
mirroring the command, so net drift stays small, but the arcs do travel.
"""

import csv
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray


@dataclass
class ArcTest:
    label: str
    v: float          # body linear velocity [m/s]
    omega: float      # body yaw rate [rad/s]
    hold_s: float


# Arcs of decreasing radius, each in both directions so that a constant gyro
# bias cancels in the mean (it flips sign with the turn, the true yaw does not).
# radius = v / omega:      0.75 m,      0.37 m,      0.22 m,     spin
DEFAULT_TESTS = [
    ArcTest("arc_wide_left",   0.30, +0.40, 4.0),
    ArcTest("arc_wide_right",  0.30, -0.40, 4.0),
    ArcTest("arc_mid_left",    0.30, +0.80, 3.0),
    ArcTest("arc_mid_right",   0.30, -0.80, 3.0),
    ArcTest("arc_tight_left",  0.25, +1.10, 2.5),
    ArcTest("arc_tight_right", 0.25, -1.10, 2.5),
    # Kept last and reported separately: worst-conditioned, most load-dependent.
    ArcTest("spin_left",       0.00, +0.80, 3.0),
    ArcTest("spin_right",      0.00, -0.80, 3.0),
]


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angle b -> a in [-pi, pi]."""
    return ((a - b + math.pi) % (2.0 * math.pi)) - math.pi


class SlipIdent(Node):

    def __init__(self):
        super().__init__("slip_ident")

        self.declare_parameter("cmd_mode", "twist")     # twist | wheels
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("wheel_cmd_topic",
                               "/wheel_velocity_controller/commands")
        self.declare_parameter("use_stamped", True)
        # RAW wheel odometry -- see the module docstring on why filtered is wrong.
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("output_dir", "/tmp/slip_ident")
        # Kinematics, for wheels mode. Must match wheel_odometry / the planner.
        self.declare_parameter("wheel_radius", 0.08)
        self.declare_parameter("track", 0.416503)
        self.declare_parameter("settle_s", 1.0)

        self.cmd_mode = str(self.get_parameter("cmd_mode").value)
        self.use_stamped = bool(self.get_parameter("use_stamped").value)
        self.pub_dt = 1.0 / float(self.get_parameter("publish_rate_hz").value)
        self.frame = str(self.get_parameter("robot_frame").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        self.track = float(self.get_parameter("track").value)
        self.settle_s = float(self.get_parameter("settle_s").value)

        if self.cmd_mode not in ("twist", "wheels"):
            raise RuntimeError(f"cmd_mode must be twist|wheels, got {self.cmd_mode}")

        if self.cmd_mode == "wheels":
            self._pub = self.create_publisher(
                Float64MultiArray,
                str(self.get_parameter("wheel_cmd_topic").value), 10)
        elif self.use_stamped:
            self._pub = self.create_publisher(
                TwistStamped, str(self.get_parameter("cmd_topic").value), 10)
        else:
            self._pub = self.create_publisher(
                Twist, str(self.get_parameter("cmd_topic").value), 10)

        sensor_qos = QoSProfile(depth=50,
                                reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value),
                                 self._on_odom, sensor_qos)
        self.create_subscription(Imu, str(self.get_parameter("imu_topic").value),
                                 self._on_imu, sensor_qos)

        self._cur_v = 0.0
        self._cur_w = 0.0
        self.create_timer(self.pub_dt, self._publish_tick)

        self._px = self._py = self._odom_yaw = 0.0
        # Accumulated (unwrapped) odom heading. Differencing two wrapped yaws
        # aliases any rotation beyond +-pi, and these tests deliberately turn
        # further than that -- the first run reported -1.37 rad for a turn that
        # was really several revolutions.
        self._odom_yaw_acc = 0.0
        self._odom_yaw_prev: Optional[float] = None
        self._odom_count = 0
        self._have_odom = False

        # The gyro is integrated here rather than read as an angle: an IMU's
        # absolute yaw may be a driver-side integration of this same rate (or
        # magnetometer-derived), and we want the raw measurement.
        self._gyro_yaw = 0.0
        self._gyro_z = 0.0
        self._imu_count = 0
        self._imu_t: Optional[float] = None

        self._samples: List[Tuple] = []
        self._recording = False
        self._t0: Optional[float] = None

    # ---- I/O -------------------------------------------------------------

    def _wheel_msg(self, v: float, w: float) -> Float64MultiArray:
        """Body twist -> the controller's joint order.

        Deliberately the IDEAL inverse, with no chi: we are commanding a known
        wheel differential and measuring what the body actually does. Applying
        a slip correction here would presuppose the answer.
        """
        w_l = (v - w * self.track / 2.0) / self.wheel_radius
        w_r = (v + w * self.track / 2.0) / self.wheel_radius
        msg = Float64MultiArray()
        # [front_left, rear_left, front_right, rear_right]
        msg.data = [w_l, w_l, w_r, w_r]
        return msg

    def _publish_tick(self):
        if self.cmd_mode == "wheels":
            self._pub.publish(self._wheel_msg(self._cur_v, self._cur_w))
        elif self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame
            msg.twist.linear.x = float(self._cur_v)
            msg.twist.angular.z = float(self._cur_w)
            self._pub.publish(msg)
        else:
            msg = Twist()
            msg.linear.x = float(self._cur_v)
            msg.angular.z = float(self._cur_w)
            self._pub.publish(msg)

    def _on_odom(self, msg: Odometry):
        self._odom_count += 1
        self._px = msg.pose.pose.position.x
        self._py = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self._odom_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        if self._odom_yaw_prev is not None:
            self._odom_yaw_acc += angle_diff(self._odom_yaw, self._odom_yaw_prev)
        self._odom_yaw_prev = self._odom_yaw
        self._have_odom = True
        self._record()

    def _on_imu(self, msg: Imu):
        self._imu_count += 1
        self._gyro_z = float(msg.angular_velocity.z)
        # Integrate on message timestamps, not wall clock: under sim time the
        # two differ by the realtime factor, which would scale the result.
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._imu_t is not None:
            dt = t - self._imu_t
            if 0.0 < dt < 0.5:
                self._gyro_yaw += self._gyro_z * dt
        self._imu_t = t

    def _record(self):
        if not self._recording:
            return
        if self._t0 is None:
            self._t0 = time.monotonic()
        self._samples.append((
            time.monotonic() - self._t0,
            self._cur_v, self._cur_w,
            self._px, self._py, self._odom_yaw,
            self._gyro_yaw, self._gyro_z,
        ))

    # ---- control ---------------------------------------------------------

    def set_cmd(self, v: float, w: float):
        self._cur_v = v
        self._cur_w = w

    def marks(self) -> Tuple[float, float, float, float]:
        """Both headings are unwrapped accumulators, so they subtract cleanly."""
        return self._px, self._py, self._odom_yaw_acc, self._gyro_yaw

    def start_recording(self):
        self._samples = []
        self._t0 = None
        self._recording = True

    def stop_recording(self) -> List[Tuple]:
        self._recording = False
        return list(self._samples)


def now_s(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def spin_for(node: Node, executor: SingleThreadedExecutor,
             duration_s: float) -> float:
    """Spin for duration_s of ROS time. Returns the elapsed ROS seconds.

    Wall-clock capped so a stalled /clock aborts instead of hanging forever --
    under sim time the ROS clock only advances while Gazebo steps.
    """
    start = now_s(node)
    wall_deadline = time.monotonic() + max(60.0, duration_s * 4.0)
    while now_s(node) - start < duration_s:
        executor.spin_once(timeout_sec=0.005)
        if time.monotonic() > wall_deadline:
            print(f"  [!] ROS clock advanced only {now_s(node) - start:.2f}s of "
                  f"{duration_s:.2f}s before the wall-clock cap -- is the sim "
                  f"stepping, and is use_sim_time set correctly?")
            break
    return now_s(node) - start


def run_test(node: SlipIdent, executor: SingleThreadedExecutor,
             tc: ArcTest) -> dict:
    """Drive one arc, then mirror it to return roughly to the start pose."""
    node.set_cmd(0.0, 0.0)
    spin_for(node, executor, node.settle_s)

    px0, py0, oyaw0, gyaw0 = node.marks()
    node.start_recording()
    node.set_cmd(tc.v, tc.omega)

    hold_actual = spin_for(node, executor, tc.hold_s)

    node.set_cmd(0.0, 0.0)
    spin_for(node, executor, node.settle_s)
    samples = node.stop_recording()

    px1, py1, oyaw1, gyaw1 = node.marks()

    # Both accumulators are unwrapped, so plain subtraction is correct here.
    d_odom = oyaw1 - oyaw0
    d_gyro = gyaw1 - gyaw0
    dist_odom = math.hypot(px1 - px0, py1 - py0)

    # Mirror the arc to come back. Reversing both v and omega retraces the arc
    # rather than driving a mirrored one, so the robot returns to roughly where
    # it started instead of walking away across the room test by test.
    node.set_cmd(-tc.v, -tc.omega)
    spin_for(node, executor, hold_actual)
    node.set_cmd(0.0, 0.0)
    spin_for(node, executor, node.settle_s)

    px2, py2, _, _ = node.marks()

    result = {
        "label": tc.label,
        "v_cmd": tc.v,
        "omega_cmd": tc.omega,
        "hold_s": hold_actual,
        "d_yaw_odom": d_odom,
        "d_yaw_gyro": d_gyro,
        "dist_odom": dist_odom,
        "residual_drift": math.hypot(px2 - px0, py2 - py0),
        "is_spin": abs(tc.v) < 1e-6,
    }

    if abs(d_gyro) < 0.05:
        print(f"  [!] gyro measured only {d_gyro:+.4f} rad -- too small to "
              f"divide by; is the IMU publishing, and did the robot move?")
    else:
        result["chi"] = d_odom / d_gyro
        result["yaw_gain"] = d_gyro / (tc.omega * hold_actual)

    print(f"  yaw: odom {d_odom:+.4f} rad, gyro {d_gyro:+.4f} rad, "
          f"commanded {tc.omega * hold_actual:+.4f} rad")
    if "chi" in result:
        print(f"  chi = odom/gyro = {result['chi']:.4f}   "
              f"(yaw_gain = gyro/cmd = {result['yaw_gain']:.4f})")
    print(f"  odom distance {dist_odom:.3f} m, "
          f"residual drift after return {result['residual_drift']:.3f} m")
    return result, samples


def save_csv(samples, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cmd_v", "cmd_w", "odom_x", "odom_y", "odom_yaw",
                    "gyro_yaw", "gyro_z"])
        for row in samples:
            w.writerow([f"{row[0]:.4f}"] + [f"{x:.5f}" for x in row[1:]])
    print(f"  -> {len(samples)} samples saved to {path}")


def summarize(results: List[dict], cmd_mode: str):
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)

    arcs = [r for r in results if "chi" in r and not r["is_spin"]]
    spins = [r for r in results if "chi" in r and r["is_spin"]]

    if not arcs:
        print("\nNo usable arcs. Check that the IMU and /odom are both live and "
              "that the robot actually moved.")
        return

    chis = [r["chi"] for r in arcs]
    gains = [r["yaw_gain"] for r in arcs]
    mean_chi = sum(chis) / len(chis)
    spread = max(chis) - min(chis)

    print("\nARCS (the operating regime that matters):")
    for r in arcs:
        radius = abs(r["v_cmd"] / r["omega_cmd"]) if r["omega_cmd"] else 0.0
        print(f"  {r['label']:<17} r={radius:4.2f} m   chi={r['chi']:.4f}   "
              f"yaw_gain={r['yaw_gain']:.4f}")
    print(f"\n  mean chi   = {mean_chi:.4f}   (spread {spread:.4f} across radii)")
    print(f"  mean gain  = {sum(gains) / len(gains):.4f}")

    if spins:
        spin_chis = [r["chi"] for r in spins]
        print(f"\nSPINS (worst-conditioned, reported separately): "
              f"chi = {[f'{c:.3f}' for c in spin_chis]}")
        print("  A spin scrubs all four contact patches and is strongly "
              "load-dependent; do not fit slip_chi to it.")

    print("\nINTERPRETATION")
    if spread > 0.25 * max(abs(mean_chi), 1e-6):
        print(f"  chi varies by {spread:.3f} across radii -- it is NOT well "
              f"modelled as a constant here. A single slip_chi will be wrong at "
              f"one end of the range; consider which radii the planner actually "
              f"uses before picking one.")
    else:
        print(f"  chi is roughly constant across radii, so a single slip_chi is "
              f"a fair model.")

    if abs(mean_chi - 1.0) < 0.05:
        print("  chi ~= 1.0. Either this platform genuinely does not scrub, or "
              "the yaw reference is not independent of the wheels -- check that "
              "odom_topic is RAW /odom and not /odometry/filtered.")

    print("\nSUGGESTED CONFIG")
    if cmd_mode == "wheels":
        print(f"  slip_chi = {mean_chi:.4f}   (planner.launch.py; "
              f"PlannerConfig.slip_chi)")
    else:
        print(f"  Measured in twist mode, so this folds chi together with the "
              f"chassis's internal twist->wheel conversion. Use it as "
              f"chassis_gain_omega = {sum(gains) / len(gains):.4f}; re-run with "
              f"cmd_mode:=wheels (sim only) to get slip_chi itself.")

    print(f"\n  wheel_odometry yaw is overstating rotation by {mean_chi:.3f}x. "
          f"While that stands, any consumer of /odom yaw inherits the bias -- "
          f"see the note in config/ekf_params.yaml.")


def preflight(node: SlipIdent, executor: SingleThreadedExecutor) -> bool:
    print("\nConfiguration:")
    print(f"  cmd mode:      {node.cmd_mode}")
    print(f"  odom topic:    {node.get_parameter('odom_topic').value}  "
          f"(must be RAW wheel odometry)")
    print(f"  imu topic:     {node.get_parameter('imu_topic').value}")
    print(f"  output dir:    {node.output_dir}")

    print("\nPre-flight (up to 5 s):")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        spin_for(node, executor, 0.25)
        if (node._pub.get_subscription_count() > 0
                and node._odom_count > 5 and node._imu_count > 5):
            break
    print(f"  cmd subscribers = {node._pub.get_subscription_count()}, "
          f"odom msgs = {node._odom_count}, imu msgs = {node._imu_count}")

    ok = True
    if node._pub.get_subscription_count() == 0:
        print("  [!] nobody is listening to the command topic.")
        ok = False
    if node._odom_count == 0:
        print("  [!] no /odom. Without it there is nothing to compare against.")
        ok = False
    if node._imu_count == 0:
        print("  [!] no IMU. The gyro IS the reference here -- results would be "
              "meaningless. Fix this before running.")
        ok = False
    return ok


def main():
    rclpy.init()
    node = SlipIdent()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        os.makedirs(node.output_dir, exist_ok=True)
        if not preflight(node, executor):
            print("\nAborting: the preflight failures above make the numbers "
                  "meaningless rather than merely noisy.")
            return

        results = []
        for i, tc in enumerate(DEFAULT_TESTS, 1):
            print(f"\n--- Test {i}/{len(DEFAULT_TESTS)}: {tc.label} ---")
            print(f"    v={tc.v:+.2f} m/s, omega={tc.omega:+.2f} rad/s, "
                  f"hold {tc.hold_s}s")
            result, samples = run_test(node, executor, tc)
            save_csv(samples, os.path.join(node.output_dir, f"{tc.label}.csv"))
            results.append(result)

        node.set_cmd(0.0, 0.0)
        spin_for(node, executor, 0.5)
        summarize(results, node.cmd_mode)
    finally:
        try:
            node.set_cmd(0.0, 0.0)
            spin_for(node, executor, 0.3)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
