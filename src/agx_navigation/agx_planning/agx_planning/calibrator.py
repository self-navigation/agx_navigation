#!/usr/bin/env python3
"""Chassis dynamics identification with position-bounded test motions.

Sends step commands of varying magnitude on the linear and angular
channels, records the response from /odom, and prints derived
parameters per test plus an aggregated summary.

Each test runs in three phases:
  1. step phase: command +mag, record response
  2. decay phase: command 0, record settling
  3. return phase: command -mag for the same time as the step phase,
                   bringing the robot back to its starting pose
                   (not analyzed, just for position recovery)

A safety monitor aborts the step phase early if the robot drifts more
than safe_radius_m metres for linear tests or max_yaw_drift_rad radians
for angular tests. The return phase then mirrors the actual (possibly
shortened) step duration so net drift stays bounded.

Per-test CSVs are saved to output_dir.

Usage:
  ros2 run <pkg> chassis_ident.py --ros-args \
      -p use_stamped:=true \
      -p cmd_topic:=/cmd_vel \
      -p odom_topic:=/odom/filtered \
      -p safe_radius_m:=0.4 \
      -p max_yaw_drift_rad:=2.5

Clear safe_radius_m metres around the robot before running. Default
tests are sized to keep travel under 0.4 m and rotation under ~115
degrees per test.
"""
import csv
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry


@dataclass
class TestCase:
    label: str
    channel: str
    magnitude: float
    hold_s: float
    rest_s: float


# Step durations are tuned so that, at the given magnitude, peak travel
# stays well under safe_radius_m (0.4 m default) and peak rotation stays
# under max_yaw_drift_rad (2.5 rad default).
#  - linear large at v=1.0 m/s for 0.4 s would reach 0.4 m IF the chassis
#    tracked instantly; in practice it's less because of the ramp.
#  - angular large at 1.0 rad/s for 2.0 s rotates ~115 degrees if it
#    tracks; that's still inside max_yaw_drift_rad and well clear of
#    cable-twist hazards.
DEFAULT_TESTS = [
    TestCase("ang_pos_small",  "angular", +0.30, 3.0, 3.0),
    TestCase("ang_pos_medium", "angular", +0.60, 2.5, 3.0),
    TestCase("ang_pos_large",  "angular", +1.00, 2.0, 3.0),
    TestCase("ang_neg_medium", "angular", -0.60, 2.5, 3.0),
    TestCase("lin_pos_small",  "linear",  +0.20, 1.5, 2.0),
    TestCase("lin_pos_medium", "linear",  +0.50, 0.8, 2.0),
    TestCase("lin_pos_large",  "linear",  +1.00, 0.4, 2.0),
]


def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def angle_diff(a: float, b: float) -> float:
    """Shortest signed angle b -> a in [-pi, pi]."""
    return ((a - b + math.pi) % (2.0 * math.pi)) - math.pi


class ChassisIdent(Node):

    def __init__(self):
        super().__init__("chassis_ident")
        self.declare_parameter("use_stamped", True)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom/filtered")
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("robot_frame", "base_link")
        self.declare_parameter("output_dir", "/tmp")
        self.declare_parameter("safe_radius_m", 0.4)
        self.declare_parameter("max_yaw_drift_rad", 2.5)
        self.declare_parameter("return_position_tol_m", 0.05)
        self.declare_parameter("return_yaw_tol_rad", 0.10)

        self.use_stamped = bool(self.get_parameter("use_stamped").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.pub_dt = 1.0 / float(self.get_parameter("publish_rate_hz").value)
        self.frame = str(self.get_parameter("robot_frame").value)
        self.output_dir = str(self.get_parameter("output_dir").value)
        self.safe_radius = float(self.get_parameter("safe_radius_m").value)
        self.max_yaw_drift = float(self.get_parameter("max_yaw_drift_rad").value)
        self.return_pos_tol = float(self.get_parameter("return_position_tol_m").value)
        self.return_yaw_tol = float(self.get_parameter("return_yaw_tol_rad").value)

        if self.use_stamped:
            self._pub = self.create_publisher(TwistStamped, self.cmd_topic, 10)
            self._msg_type = "TwistStamped"
        else:
            self._pub = self.create_publisher(Twist, self.cmd_topic, 10)
            self._msg_type = "Twist"

        odom_qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(Odometry, self.odom_topic,
                                 self._on_odom, odom_qos)

        self._cur_v = 0.0
        self._cur_w = 0.0
        self.create_timer(self.pub_dt, self._publish_tick)

        # Latest pose/twist always tracked (used by safety monitor).
        self._px = 0.0
        self._py = 0.0
        self._yaw = 0.0
        self._act_v = 0.0
        self._act_w = 0.0
        self._have_odom = False

        self._samples: List[Tuple[float, float, float, float, float]] = []
        self._t0: Optional[float] = None
        self._recording = False
        self._odom_count = 0

    def _publish_tick(self):
        if self.use_stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame
            msg.twist.linear.x = float(self._cur_v)
            msg.twist.angular.z = float(self._cur_w)
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
        self._yaw = yaw_from_quat(q.x, q.y, q.z, q.w)
        self._act_v = float(msg.twist.twist.linear.x)
        self._act_w = float(msg.twist.twist.angular.z)
        self._have_odom = True

        if not self._recording:
            return
        if self._t0 is None:
            self._t0 = time.monotonic()
        t = time.monotonic() - self._t0
        self._samples.append((
            t,
            float(self._cur_v),
            float(self._cur_w),
            self._act_v,
            self._act_w,
        ))

    @property
    def cmd_sub_count(self) -> int:
        return self._pub.get_subscription_count()

    @property
    def odom_msg_count(self) -> int:
        return self._odom_count

    def pose(self) -> Tuple[float, float, float]:
        return self._px, self._py, self._yaw

    def set_cmd(self, v: float, w: float):
        self._cur_v = v
        self._cur_w = w

    def start_recording(self):
        self._samples = []
        self._t0 = None
        self._recording = True

    def stop_recording(self) -> List[Tuple[float, float, float, float, float]]:
        self._recording = False
        return list(self._samples)


def spin_for(executor: SingleThreadedExecutor, duration_s: float):
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.005)


def hold_with_safety(node: ChassisIdent,
                     executor: SingleThreadedExecutor,
                     tc: TestCase,
                     px0: float, py0: float, yaw0: float) -> Tuple[float, bool]:
    """Run the hold phase, monitoring drift. Returns (actual_duration, aborted).

    Aborts and returns early when:
      - linear channel: hypot(x-x0, y-y0) > 0.7 * safe_radius
      - angular channel: |yaw - yaw0| > 0.9 * max_yaw_drift
    """
    t_start = time.monotonic()
    deadline = t_start + tc.hold_s
    aborted = False
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.005)
        if tc.channel == "linear":
            d = math.hypot(node._px - px0, node._py - py0)
            if d > 0.7 * node.safe_radius:
                aborted = True
                break
        else:
            d = abs(angle_diff(node._yaw, yaw0))
            if d > 0.9 * node.max_yaw_drift:
                aborted = True
                break
    return time.monotonic() - t_start, aborted


def return_to_baseline(node: ChassisIdent,
                       executor: SingleThreadedExecutor,
                       tc: TestCase,
                       hold_actual: float,
                       px0: float, py0: float, yaw0: float):
    """Drive the inverse command for hold_actual seconds to recover the
    initial pose. Stops early if within return tolerance, holds the
    reverse command for the remaining time otherwise. Settles for 1 s.
    """
    inverse_mag = -tc.magnitude
    if tc.channel == "linear":
        node.set_cmd(inverse_mag, 0.0)
    else:
        node.set_cmd(0.0, inverse_mag)

    deadline = time.monotonic() + hold_actual + 1.0
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.005)
        if tc.channel == "linear":
            d = math.hypot(node._px - px0, node._py - py0)
            if d < node.return_pos_tol:
                break
        else:
            d = abs(angle_diff(node._yaw, yaw0))
            if d < node.return_yaw_tol:
                break

    node.set_cmd(0.0, 0.0)
    spin_for(executor, 1.0)


def run_test(node: ChassisIdent,
             executor: SingleThreadedExecutor,
             tc: TestCase) -> Tuple[List, dict]:
    """Run one step test with safety and return-to-baseline."""
    node.set_cmd(0.0, 0.0)
    spin_for(executor, 0.7)
    px0, py0, yaw0 = node.pose()

    node.start_recording()
    if tc.channel == "linear":
        node.set_cmd(tc.magnitude, 0.0)
    else:
        node.set_cmd(0.0, tc.magnitude)

    hold_actual, aborted = hold_with_safety(node, executor, tc,
                                            px0, py0, yaw0)

    node.set_cmd(0.0, 0.0)
    spin_for(executor, tc.rest_s)
    samples = node.stop_recording()

    if aborted:
        print(f"  [!] step phase aborted at t={hold_actual:.2f}s "
              f"(drift exceeded safety limit)")

    # Recovery: drive the inverse for the actual hold duration. Net
    # displacement stays small as long as the chassis is roughly
    # symmetric forward/reverse.
    px_pre, py_pre, yaw_pre = node.pose()
    return_to_baseline(node, executor, tc, hold_actual, px0, py0, yaw0)
    px_post, py_post, yaw_post = node.pose()

    drift = {
        "hold_actual": hold_actual,
        "aborted": aborted,
        "drift_after_step": math.hypot(px_pre - px0, py_pre - py0),
        "yaw_drift_after_step": abs(angle_diff(yaw_pre, yaw0)),
        "drift_after_return": math.hypot(px_post - px0, py_post - py0),
        "yaw_drift_after_return": abs(angle_diff(yaw_post, yaw0)),
    }
    return samples, drift


# ---------- Analysis ------------------------------------------------------

def _linfit(pts: List[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    n = len(pts)
    if n < 3:
        return None
    mt = sum(t for t, _ in pts) / n
    my = sum(y for _, y in pts) / n
    num = sum((t - mt) * (y - my) for t, y in pts)
    den = sum((t - mt) ** 2 for t, _ in pts)
    if den < 1e-12:
        return None
    slope = num / den
    return slope, my - slope * mt


def fit_decay_tau(times: List[float], values: List[float],
                  initial: float) -> Optional[float]:
    """Fit log|y| = log|initial| - t/tau."""
    if abs(initial) < 1e-6:
        return None
    sign = 1.0 if initial > 0 else -1.0
    pts = [(t, math.log(abs(v))) for t, v in zip(times, values)
           if (sign * v) > 0.01 * abs(initial)]
    fit = _linfit(pts)
    if fit is None or fit[0] >= 0:
        return None
    return -1.0 / fit[0]


def fit_ramp_alpha(times: List[float], values: List[float],
                   steady: float) -> Optional[float]:
    if abs(steady) < 1e-6:
        return None
    sign = 1.0 if steady > 0 else -1.0
    lo, hi = 0.10 * abs(steady), 0.90 * abs(steady)
    pts = [(t, v) for t, v in zip(times, values) if lo <= sign * v <= hi]
    fit = _linfit(pts)
    return fit[0] if fit is not None else None


def analyze(samples, tc: TestCase, drift: dict) -> dict:
    result = {"label": tc.label, "channel": tc.channel,
              "magnitude": tc.magnitude, "aborted": drift["aborted"],
              "hold_actual": drift["hold_actual"]}
    if not samples:
        print("  [!] NO SAMPLES recorded.")
        return result

    idx = 4 if tc.channel == "angular" else 3
    unit = "rad/s" if tc.channel == "angular" else "m/s"
    unit_a = "rad/s^2" if tc.channel == "angular" else "m/s^2"
    hold_t = drift["hold_actual"]

    hold = [(s[0], s[idx]) for s in samples if s[0] <= hold_t]
    post = [(s[0] - hold_t, s[idx]) for s in samples if s[0] > hold_t]
    if not hold:
        print("  [!] No hold-phase samples.")
        return result

    # Steady-state estimate: depending on whether we reached steady, use
    # either the settled mean (last 30%) or the peak observed value.
    n_keep = max(3, len(hold) // 3)
    settle = hold[-n_keep:]
    settled_mean = sum(v for _, v in settle) / len(settle)
    settled_std  = (sum((v - settled_mean) ** 2 for _, v in settle)
                    / len(settle)) ** 0.5
    sign = 1.0 if tc.magnitude > 0 else -1.0
    peak = max((sign * v for _, v in hold), default=0.0) * sign

    # If the settled mean is within 5% of the peak, treat it as steady.
    # Otherwise we likely didn't reach steady state.
    reached_steady = (abs(settled_mean) > 0.02
                      and abs(peak - settled_mean) < 0.05 * abs(peak))
    steady = settled_mean if reached_steady else peak
    result["steady"] = steady
    result["steady_std"] = settled_std
    result["reached_steady"] = reached_steady
    if abs(tc.magnitude) > 1e-9:
        result["gain"] = steady / tc.magnitude

    tag = "settled" if reached_steady else "peak"
    print(f"  steady ({tag}) = {steady:+.4f} {unit}  "
          f"(cmd {tc.magnitude:+.3f}, gain {result.get('gain', float('nan')):.3f}, "
          f"sigma {settled_std:.4f})")
    if not reached_steady:
        print(f"  [info] step phase ({hold_t:.2f}s) too short for full "
              f"settling; using peak as steady-state surrogate.")

    if abs(steady) < 0.02:
        print("  [!] chassis did not respond.")
        return result

    sign = 1.0 if steady > 0 else -1.0

    dead = next((t for t, v in hold if abs(v) >= 0.05 * abs(steady)), None)
    if dead is not None:
        result["dead_time"] = dead
        print(f"  dead time           = {dead:.3f} s")

    cross = {}
    for frac in (0.10, 0.50, 0.90):
        tgt = steady * frac
        t = next((tt for tt, v in hold if (sign * v) >= (sign * tgt)), None)
        if t is not None:
            cross[frac] = t
            print(f"  t at {int(frac*100):>2}% steady = {t:.3f} s")

    alpha = fit_ramp_alpha([t for t, _ in hold], [v for _, v in hold], steady)
    if alpha is not None:
        result["alpha_ramp"] = alpha
        print(f"  ramp alpha (10-90%) = {alpha:+.4f} {unit_a}")

    if 0.10 in cross and 0.90 in cross and alpha is not None:
        t10, t90 = cross[0.10], cross[0.90]
        band = [(t, v) for t, v in hold if t10 <= t <= t90]
        if len(band) >= 4:
            mt = sum(t for t, _ in band) / len(band)
            mv = sum(v for _, v in band) / len(band)
            t0_lin = mt - mv / alpha
            res_lin = sum((v - alpha * (t - t0_lin)) ** 2 for t, v in band)

            tau_est = (t90 - t10) / math.log(9.0)
            t0_exp = t10 + tau_est * math.log(1 - 0.10)
            res_exp = 0.0
            for t, v in band:
                tt = t - t0_exp
                if tt > 0:
                    pred = steady * (1.0 - math.exp(-tt / tau_est))
                    res_exp += (v - pred) ** 2
            print(f"  ramp shape: linear-res {res_lin:.5f} vs "
                  f"first-order(tau={tau_est:.3f})-res {res_exp:.5f}")
            if res_lin < 0.6 * res_exp:
                result["ramp_kind"] = "rate-limited"
            elif res_exp < 0.6 * res_lin:
                result["ramp_kind"] = "first-order"
                result["tau_rise"] = tau_est
            else:
                result["ramp_kind"] = "mixed"
            print(f"  -> ramp shape       = {result['ramp_kind']}")

    if post:
        v0 = next((v for t, v in post if abs(v) > 0.05 * abs(steady)), steady)
        ts = [t for t, _ in post if t > 0]
        vs = [v for t, v in post if t > 0]
        tau_d = fit_decay_tau(ts, vs, v0)
        if tau_d is not None and 0.01 < tau_d < 10.0:
            result["tau_decay"] = tau_d
            print(f"  decay tau (off)     = {tau_d:.3f} s")
        t5 = next((t for t, v in post if abs(v) < 0.05 * abs(steady)), None)
        if t5 is not None:
            result["t_settle_5pct"] = t5
            print(f"  decay to 5% steady  = {t5:.3f} s")

    print(f"  position drift: after-step {drift['drift_after_step']:.3f} m, "
          f"after-return {drift['drift_after_return']:.3f} m  |  "
          f"yaw drift: after-step {drift['yaw_drift_after_step']:.3f} rad, "
          f"after-return {drift['yaw_drift_after_return']:.3f} rad")
    return result


def save_csv(samples, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "cmd_v", "cmd_w", "act_v", "act_w"])
        for row in samples:
            w.writerow([f"{row[0]:.4f}"] + [f"{x:.5f}" for x in row[1:]])
    print(f"  -> {len(samples)} samples saved to {path}")


def summarize(results: List[dict]):
    print("\n" + "=" * 64)
    print("SUMMARY")
    print("=" * 64)

    ang = [r for r in results
           if r.get("channel") == "angular" and abs(r.get("steady", 0)) > 0.02]
    lin = [r for r in results
           if r.get("channel") == "linear"  and abs(r.get("steady", 0)) > 0.02]

    def block(name, group, accel_unit):
        if not group:
            print(f"\n{name}: no usable responses.")
            return
        print(f"\n{name} channel:")
        gains  = [r["gain"] for r in group if "gain" in r]
        steady = [abs(r["steady"]) for r in group]
        alphas = [abs(r["alpha_ramp"]) for r in group if "alpha_ramp" in r]
        taus_d = [r["tau_decay"]  for r in group if "tau_decay" in r]
        kinds  = [r.get("ramp_kind", "?") for r in group]
        reached = [r.get("reached_steady", False) for r in group]
        print(f"  gain per test:        {[f'{g:+.3f}' for g in gains]}")
        if gains:
            usable_gains = [g for g, ok in zip(gains, reached) if ok]
            mean = (sum(usable_gains) / len(usable_gains)) if usable_gains else sum(gains)/len(gains)
            tag = "(from settled tests)" if usable_gains else "(includes peak surrogates)"
            print(f"  gain mean:            {mean:+.3f}  {tag}")
        print(f"  |steady| observed:    {[f'{s:.3f}' for s in steady]}")
        print(f"  reached steady:       {reached}")
        print(f"  apparent ceiling:     {max(steady):.3f}")
        if alphas:
            print(f"  ramp alpha per test:  {[f'{a:.3f}' for a in alphas]}")
            print(f"  max ramp alpha:       {max(alphas):.3f} {accel_unit}")
        if taus_d:
            print(f"  decay tau per test:   {[f'{t:.3f}' for t in taus_d]}")
            print(f"  mean decay tau:       {sum(taus_d)/len(taus_d):.3f} s")
        print(f"  ramp shapes:          {kinds}")

    block("ANGULAR", ang, "rad/s^2")
    block("LINEAR",  lin, "m/s^2")

    print("\nSuggested planner config (conservative):")
    if ang:
        ceil = max(abs(r["steady"]) for r in ang)
        a_max = max((abs(r["alpha_ramp"]) for r in ang if "alpha_ramp" in r),
                    default=0.0)
        print(f"  omega_max          = {ceil * 0.95:.3f}")
        if a_max > 0:
            print(f"  alpha_max          = {a_max * 0.8:.3f}")
        taus = [r["tau_decay"] for r in ang if "tau_decay" in r]
        if taus:
            print(f"  chassis_tau_omega  = {sum(taus)/len(taus):.3f}")
    if lin:
        ceil = max(abs(r["steady"]) for r in lin)
        a_max = max((abs(r["alpha_ramp"]) for r in lin if "alpha_ramp" in r),
                    default=0.0)
        print(f"  v_max              = {ceil * 0.95:.3f}")
        if a_max > 0:
            print(f"  a_max              = {a_max * 0.8:.3f}")
        taus = [r["tau_decay"] for r in lin if "tau_decay" in r]
        if taus:
            print(f"  chassis_tau_v      = {sum(taus)/len(taus):.3f}")


def preflight(node: ChassisIdent, executor: SingleThreadedExecutor) -> bool:
    print(f"\nConfiguration:")
    print(f"  publishing:        {node._msg_type} on '{node.cmd_topic}' "
          f"at {1.0/node.pub_dt:.0f} Hz")
    print(f"  subscribing:       Odometry on '{node.odom_topic}' (BEST_EFFORT)")
    print(f"  safe radius:       {node.safe_radius:.2f} m  "
          f"(linear tests abort if drift exceeds 70% of this)")
    print(f"  max yaw drift:     {node.max_yaw_drift:.2f} rad  "
          f"(angular tests abort if exceeded)")
    print(f"  output dir:        {node.output_dir}")

    print(f"\nPre-flight check (up to 5 s):")
    deadline = time.monotonic() + 5.0
    last_sub = -1
    last_odom = -1
    while time.monotonic() < deadline:
        spin_for(executor, 0.25)
        s, o = node.cmd_sub_count, node.odom_msg_count
        if s != last_sub or o != last_odom:
            print(f"  cmd subscribers = {s}, "
                  f"odom messages received = {o}, "
                  f"odom-pose available = {node._have_odom}")
            last_sub, last_odom = s, o
        if s > 0 and o > 5 and node._have_odom:
            break

    ok = True
    if node.cmd_sub_count == 0:
        print(f"\n  [!] NO SUBSCRIBERS on {node.cmd_topic}.")
        print(f"      Check 'ros2 topic info {node.cmd_topic} -v'.")
        print(f"      Toggle use_stamped or remap cmd_topic.")
        ok = False
    if node.odom_msg_count == 0:
        print(f"\n  [!] NO ODOM messages on {node.odom_topic}.")
        print(f"      Check 'ros2 topic info {node.odom_topic} -v'.")
        ok = False
    return ok


def main():
    rclpy.init()
    node = ChassisIdent()
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    try:
        os.makedirs(node.output_dir, exist_ok=True)
        ok = preflight(node, executor)
        if not ok:
            print("\nContinuing anyway; results will be useless until connectivity is fixed.")

        results = []
        for i, tc in enumerate(DEFAULT_TESTS, 1):
            print(f"\n--- Test {i}/{len(DEFAULT_TESTS)}: {tc.label} ---")
            print(f"    {tc.channel} step = {tc.magnitude:+.3f}, "
                  f"hold up to {tc.hold_s}s, rest {tc.rest_s}s")
            samples, drift = run_test(node, executor, tc)
            save_csv(samples, os.path.join(node.output_dir, f"id_{tc.label}.csv"))
            results.append(analyze(samples, tc, drift))

        node.set_cmd(0.0, 0.0)
        spin_for(executor, 0.5)
        summarize(results)
    finally:
        try:
            node.set_cmd(0.0, 0.0)
            spin_for(executor, 0.3)
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
