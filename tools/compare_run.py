#!/usr/bin/env python3
"""One plan x one arm through the FULL ROS stack, scored from Gazebo truth.

This is the Nav2-baseline harness the advisor's ask 1 needs (handover.md,
2026-09-28: the comparison is the only priority). For one cell of the
broad40 x {ours, nav2-dwb, nav2-mppi, nav2-rpp} grid it:

  1. tears the worker's partition down and brings a FRESH stack up
     (tools/fixture_up.sh: baked map, NO SLAM, localization:=amcl, robot
     spawned at the plan's start pose, frontier explorer OFF), retrying on the
     launch races -- a half-up stack is the state this exists to escape;
  2. spawns the along-path slip patches (tools/spawn_patches.py, seed 0 -- the
     same plant the soak numbers came from; the fixture's own near-origin
     patches are OFF so exactly one plant exists per run);
  3. sends the goal the way the arm consumes it: /goal_pose for `ours`
     (vec-pmp offline + TVLQR), the NavigateToPose action for the nav2 arms;
  4. waits a SIM-time deadline (3x plan duration + 60 s -- a wall-clock
     timeout moves with the realtime factor and is wrong by that factor).
     For `ours` that deadline starts at the FIRST WHEEL COMMAND, not at the
     goal: the corrector buffers the whole rollout before driving, planning
     is a CPU-bound solve (1-6 s wall per 0.5 s chunk on the broad plans),
     and a clock started at the goal timed out 8 of 10 probe cells
     (2026-10-01) that never got to drive. Planning is capped separately in
     WALL time (--plan-wall-cap -> outcome planner-timeout), and the whole
     cell -- bring-up, terrain, planning, driving -- by --cell-wall-cap, so a
     wedged planner or stack cannot hold a worker; then
  5. writes ONE JSONL row to --out and moves to the next plan.

SCORING IS FROM GAZEBO GROUND TRUTH, never /odom and never map->base_link:
under amcl that transform is an ESTIMATE, and scoring arms on it would measure
localization error rather than navigation error (CLAUDE.md: "Scoring uses
Gazebo ground truth, never /odom"). Pose comes from /world/<w>/pose/info over
gz-transport, matched by model name -- the same signal run_recorder and the RL
GazeboBridge use. The map frame IS the Gazebo world frame here
(truth_localization.py, FRAMES), so plan coordinates and gz poses compare
directly; spawn_x/y/yaw puts the robot exactly at the plan's start.

METRICS (identical estimator for every arm):
  final_err        hypot(end - goal_xy), end = ground truth at the terminal event.
  outcome          arrived (final_err <= --tolerance, default 0.5 m, the repo's
                   miss-rate convention) | failed (terminal but short) |
                   timeout (drive deadline, or the cell wall cap) |
                   planner-timeout (`ours` only: no wheel command within
                   --plan-wall-cap) | planner-failed (terminal, and no plan
                   was ever published: BVP mesh exhaustion for `ours`,
                   planning failure for nav2) | stack-failed (bring-up or
                   terrain failed).
                   `stack_terminal` records what the STACK said separately
                   (sentinel / nav2 result code), so "arrived but nav2
                   aborted a second late" is still readable.
  path_length      polyline length of the ground-truth track after the goal.
  travel_time      sim seconds, goal sent -> terminal event.
  plan_wall_s      `ours`: wall seconds, goal sent -> first wheel command.
  drive_t0         sim time the drive deadline started (first wheel command
                   for `ours`, the goal for nav2). `rtf` is measured over
                   drive_t0 -> terminal, so it is None for a cell that never
                   drove (it used to be sim/0-wall garbage, ~1e8).
  max_curvature    max discrete Menger curvature of the ground-truth track,
                   arc-length-resampled at 0.1 m: raw pose/info triples spike
                   on sampling noise, not on real turns.
  control_energy   integral of the SQUARED PER-WHEEL ACCELERATIONS,
                   sum_i ∫ (dw_i/dt)^2 dt, computed on the PUBLISHED
                   /wheel_velocity_controller/commands -- both arms command
                   the same topic, so it is the same measurement; commands,
                   not joint states, because it is the control actually
                   applied. Both streams are ZERO-ORDER-HOLD resampled onto a
                   common 0.1 s sim-time grid before differencing (`ctrl_grid_dt`
                   in the row): the vec-pmp corrector publishes at 10 Hz while
                   nav2's twist_to_wheels follows the 20 Hz nav chain, and a
                   raw difference of a step integrates to dW^2 * rate -- a
                   metric that rewards coarse sampling. The grid makes the
                   estimator rate-independent.
  max_cross_track  max distance from the ground-truth track to the plan npz's
                   reference polyline (poses[:, :2]). For `ours` this is a
                   re-plan of the same start/goal pair on the same map; for
                   nav2 it is how differently Nav2 routes the same problem.

BATCH MODE: give --plans FILE and one --arm; plans are walked sequentially
(one fresh stack per plan), and rows already in --skip-from (default: --out)
are skipped, so a killed batch resumes with `the same command`. This is NOT a
one-shot process per cell -- the process stays plain: everything it starts
(tmux window) it also tears down, per plan.

Run ON THE VM, inside the worker's partition:

    tools/with-worker 1 python3 tools/compare_run.py \
        --plans <(sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt) \
        --arm ours --worker 1 --out ~/run_data/<run-dir>/rows.jsonl

or a single cell:

    tools/with-worker 1 python3 tools/compare_run.py \
        --plan $HOME/run_data/2026-08-14_job020_v2-library/traj_data_v2/floor_6_v2_00369.npz --arm nav2-dwb --worker 1 \
        --out /tmp/smoke.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.dirname(HERE)

ARMS = ("ours", "nav2-dwb", "nav2-mppi", "nav2-rpp")
ARM_TO_CONTROLLER = {"nav2-dwb": "dwb", "nav2-mppi": "mppi", "nav2-rpp": "rpp"}


# ---------------------------------------------------------------------------
# metrics helpers (pure numpy)
# ---------------------------------------------------------------------------

def polyline_length(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def nearest_dist_to_polyline(pt, poly: np.ndarray) -> float:
    """Distance from pt to the closest point ON any segment of the polyline."""
    pt = np.asarray(pt, dtype=float)
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    denom = (ab * ab).sum(axis=1)
    denom[denom == 0.0] = 1e-12
    t = np.clip(((pt - a) * ab).sum(axis=1) / denom, 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = proj - pt
    return float(np.hypot(d[:, 0], d[:, 1]).min())


def max_menger_curvature(pts: np.ndarray, resample_m: float = 0.1) -> float:
    """Max discrete Menger curvature along the track, arc-length resampled."""
    if len(pts) < 3:
        return 0.0
    d = np.diff(pts, axis=0)
    s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
    total = s[-1]
    if total <= resample_m:
        return 0.0
    grid = np.arange(0.0, total, resample_m)
    rs = np.stack([np.interp(grid, s, pts[:, 0]),
                   np.interp(grid, s, pts[:, 1])], axis=1)
    a, b, c = rs[:-2], rs[1:-1], rs[2:]
    ab = np.hypot(*(b - a).T)
    bc = np.hypot(*(c - b).T)
    ca = np.hypot(*(a - c).T)
    area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    ok = (ab > 1e-9) & (bc > 1e-9) & (ca > 1e-9)
    if not ok.any():
        return 0.0
    return float((2.0 * area2[ok] / (ab[ok] * bc[ok] * ca[ok])).max())


def control_energy(commands, t0: float, t1: float, grid_dt: float = 0.1) -> float:
    """sum_i ∫ (dw_i/dt)^2 dt, ZOH-resampled on a common grid (see docstring).

    `commands` is [(sim_t, [w0..w3]), ...]; only [t0, t1] counts.
    """
    if len(commands) < 2 or t1 - t0 < grid_dt:
        return 0.0
    ts = np.array([c[0] for c in commands], dtype=float)
    ws = np.array([c[1] for c in commands], dtype=float)
    grid = np.arange(t0, t1, grid_dt)
    idx = np.searchsorted(ts, grid, side="right") - 1
    valid = idx >= 0
    if not valid.any():
        return 0.0
    sampled = np.zeros((len(grid), ws.shape[1]))
    sampled[valid] = ws[idx[valid]]
    dw = np.diff(sampled, axis=0)
    return float(((dw / grid_dt) ** 2 * grid_dt).sum())


# ---------------------------------------------------------------------------
# the ROS-side driver/scorer
# ---------------------------------------------------------------------------

def _sim(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def _yaw_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


class CompareDriver:
    """rclpy node + gz subscription for one run's drive-and-score."""

    def __init__(self, world: str, model: str,
                 goal_tries: int = 3, goal_ack_wait: float = 15.0,
                 plan_wall_cap: float = 600.0):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                               ReliabilityPolicy)
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Path
        from rosgraph_msgs.msg import Clock
        from std_msgs.msg import Float64MultiArray
        from action_msgs.msg import GoalStatusArray

        self.goal_tries = int(goal_tries)
        self.goal_ack_wait = float(goal_ack_wait)
        self.plan_wall_cap = float(plan_wall_cap)
        self._rclpy = rclpy
        rclpy.init()

        # MUST match the stack's own /goal_pose profile (drive_goal.py's
        # warning: a TRANSIENT_LOCAL endpoint silently receives nothing).
        goal_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE)

        self.state = {
            "sim_t": None,
            "pose": None,   # latest ground truth (x, y, yaw)
            "track": [],    # (sim_t, x, y, yaw), downsampled to 20 Hz
            "wheel": [],    # (sim_t, [w0..w3])
            # ours only: the corrector's per-tick CorrectionDiagnostics (#34) and
            # the plan it is actually following (the LIVE solve, which is not
            # the library plan in plan["poses"]).
            "diag": [],     # (sim_t, [CorrectionDiagnostics.FIELDS...])
            "live_plan": None,  # (N, 3) x, y, yaw from the latest Path
            "sentinel": False,
            "n_plan_paths": 0,
            "plan_goal_ids": set(),  # PlanToGoal goals the planner has seen
        }
        st = self.state

        self.node = Node("compare_run_driver")
        self.node.create_subscription(
            Clock, "/clock", lambda m: st.__setitem__("sim_t", _sim(m.clock)), 10)
        # The plan the stack is following: the corrector's ~/plan remap
        # (/optimal_trajectory) for `ours`, nav2 planner_server's /plan for the
        # nav2 arms. Either flowing after the goal means a plan EXISTED -- the
        # planner-failed discrimination.
        for topic in ("/optimal_trajectory", "/plan"):
            self.node.create_subscription(
                Path, topic,
                lambda m: st.__setitem__("n_plan_paths", st["n_plan_paths"] + 1), 5)
        def _on_live_plan(m):
            # Offline mode republishes the CUMULATIVE path per chunk, so the
            # latest is the whole plan; an empty Path is the planner's
            # end/failure clear and must not erase it.
            if not m.poses:
                return
            st["live_plan"] = np.array(
                [(p.pose.position.x, p.pose.position.y,
                  2.0 * math.atan2(p.pose.orientation.z, p.pose.orientation.w))
                 for p in m.poses], dtype=float).reshape(-1, 3)
        self.node.create_subscription(Path, "/optimal_trajectory", _on_live_plan, 5)
        self.node.create_subscription(
            Float64MultiArray, "/wheel_corrector/tvlqr_diagnostics",
            lambda m: st["diag"].append((st["sim_t"], list(m.data)))
            if st["sim_t"] is not None else None, 50)
        self.node.create_subscription(
            Float64MultiArray, "/wheel_velocity_controller/commands",
            lambda m: st["wheel"].append((st["sim_t"], list(m.data)))
            if st["sim_t"] is not None else None, 10)
        # The corrector's terminal sentinel: empty frame_id on /goal_pose =
        # "nobody is pursuing a goal" -- ANY terminal outcome (CLAUDE.md), so
        # arrival is decided from ground truth, never from this alone.
        self.node.create_subscription(
            PoseStamped, "/goal_pose",
            lambda m: st.__setitem__("sentinel", st["sentinel"] or m.header.frame_id == ""),
            goal_qos)
        self._goal_pub = self.node.create_publisher(PoseStamped, "/goal_pose", goal_qos)
        # The planner's action status: a new goal id here means the corrector
        # received /goal_pose and the planner accepted the solve. That is the
        # vec-pmp ACK -- motion is not, because the corrector is silent for
        # the whole planning phase, and republishing into a slow solve
        # restarts it from chunk 0.
        self.node.create_subscription(
            GoalStatusArray, "/pmp_planner/plan_to_goal/_action/status",
            lambda m: st["plan_goal_ids"].update(
                bytes(s.goal_info.goal_id.uuid) for s in m.status_list), 10)

        import gz.transport13 as gz_transport
        from gz.msgs10.pose_v_pb2 import Pose_V

        def _on_truth(msg: Pose_V):
            for p in msg.pose:
                if p.name != model:
                    continue
                yaw = math.atan2(
                    2.0 * (p.orientation.w * p.orientation.z
                           + p.orientation.x * p.orientation.y),
                    1.0 - 2.0 * (p.orientation.y ** 2 + p.orientation.z ** 2))
                st["pose"] = (p.position.x, p.position.y, yaw)
                t = st["sim_t"]
                if t is not None:
                    tr = st["track"]
                    # 20 Hz is plenty: 2.5 cm of travel at the planner's v_max,
                    # far below the 0.1 m curvature resample.
                    if not tr or t - tr[-1][0] >= 0.05:
                        tr.append((t, p.position.x, p.position.y, yaw))
                return

        self._gz = gz_transport.Node()
        self.topic_pose = f"/world/{world}/pose/info"
        if not self._gz.subscribe(Pose_V, self.topic_pose, _on_truth):
            raise RuntimeError(f"could not subscribe to {self.topic_pose}")

    # -- spinning -----------------------------------------------------------
    def spin(self, dur: float) -> None:
        end = time.monotonic() + dur
        while time.monotonic() < end:
            self._rclpy.spin_once(self.node, timeout_sec=0.02)

    def spin_until(self, pred, wall_timeout: float) -> bool:
        end = time.monotonic() + wall_timeout
        while time.monotonic() < end:
            if pred():
                return True
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
        return pred()

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()

    # -- goal dispatch ------------------------------------------------------
    def send_goal(self, arm: str, goal_xy, goal_yaw: float,
                  timeout_s: float, backstop_s: float) -> dict:
        """Send the goal the arm's way; block until terminal or deadline.

        The returned dict carries terminal/travel_time/drive_wall; metrics are
        computed by `run()` from the state the subscriptions kept filling
        WHILE this spun.
        """
        st = self.state
        if not self.spin_until(lambda: st["sim_t"] is not None, 15.0):
            return {"error": "no /clock", "terminal": False,
                    "stack_terminal": "none", "nav2_error_code": None}
        self.spin(0.3)  # a settle beat so the goal isn't raced by the sub match
        goal_t = st["sim_t"]
        if arm == "ours":
            out = self._drive_vec_pmp(goal_xy, goal_yaw, goal_t, timeout_s, backstop_s)
        else:
            out = self._drive_nav2(goal_xy, goal_yaw, goal_t, timeout_s, backstop_s)
        out["goal_t"] = goal_t
        return out

    def _drive_nav2(self, goal_xy, gyaw, goal_t, timeout_s, backstop_s) -> dict:
        from nav2_msgs.action import NavigateToPose
        from rclpy.action import ActionClient

        client = ActionClient(self.node, NavigateToPose, "navigate_to_pose")
        try:
            if not client.wait_for_server(timeout_sec=30.0):
                return {"terminal": False,
                        "error": "navigate_to_pose action server never appeared",
                        "stack_terminal": "none", "nav2_error_code": None}
            goal = NavigateToPose.Goal()
            goal.pose.header.frame_id = "map"
            # Stamped with SIM time: the node's own clock is system time, and a
            # system-time stamp is minutes away from sim time -- a TF lookup at
            # it would extrapolate or fail.
            goal.pose.header.stamp = self._sim_stamp(goal_t)
            goal.pose.pose.position.x = float(goal_xy[0])
            goal.pose.pose.position.y = float(goal_xy[1])
            _, _, qz, qw = _yaw_quat(gyaw)
            goal.pose.pose.orientation.z = qz
            goal.pose.pose.orientation.w = qw

            send_fut = client.send_goal_async(goal)
            if not self.spin_until(lambda: send_fut.done(), 30.0):
                return {"terminal": False, "error": "goal response never arrived",
                        "stack_terminal": "none", "nav2_error_code": None}
            handle = send_fut.result()
            if not handle.accepted:
                return {"terminal": True, "terminal_kind": "rejected",
                        "stack_terminal": "nav2-rejected", "nav2_error_code": None}
            # get_result() BLOCKS waiting for the future -- and the thread
            # that would service that future is this one (spin_once), so it
            # deadlocks (rclpy's own docstring: "do not call this method in a
            # callback"). The async variant is the one that works with an
            # externally-spun executor.
            result_fut = handle.get_result_async()
            out = self._wait_terminal(goal_t, timeout_s, backstop_s, kind="action",
                                      extra_pred=result_fut.done)
            if out.get("terminal") and out.get("terminal_kind") == "action":
                from action_msgs.msg import GoalStatus

                res = result_fut.result()
                out["stack_terminal"] = (
                    "nav2-succeeded" if res.status == GoalStatus.STATUS_SUCCEEDED
                    else f"nav2-status-{res.status}")
                code = getattr(res.result, "error_code", None)
                try:
                    out["nav2_error_code"] = int(code)
                    out["nav2_error_name"] = _nav2_error_name(int(code))
                except (TypeError, ValueError):
                    out["nav2_error_code"] = None
            elif not out.get("terminal"):
                # Deadline: cancel so the stack stops driving into the next
                # run's teardown. A failed cancel must not mask the timeout.
                try:
                    cancel_fut = handle.cancel_goal_async()
                    self.spin_until(lambda: cancel_fut.done(), 5.0)
                    out["cancelled"] = True
                except Exception:  # noqa: BLE001
                    pass
            return out
        finally:
            client.destroy()

    def _wait_terminal(self, goal_t, timeout_s, backstop_s, kind, extra_pred=None) -> dict:
        st = self.state
        wall0 = time.monotonic()
        out: dict = {"terminal": False, "terminal_kind": None,
                     "stack_terminal": "none", "nav2_error_code": None}

        def done() -> bool:
            if kind == "sentinel" and st["sentinel"]:
                return True
            return extra_pred is not None and extra_pred()

        while True:
            if done():
                out["terminal"] = True
                out["terminal_kind"] = kind
                break
            sim_t = st["sim_t"]
            if sim_t is not None and sim_t - goal_t > timeout_s:
                out["timed_out"] = True
                break
            if time.monotonic() - wall0 > backstop_s:
                out["wall_backstop"] = True
                break
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        out["travel_time"] = ((st["sim_t"] - goal_t)
                              if st["sim_t"] is not None else None)
        out["drive_wall"] = time.monotonic() - wall0
        return out

    def _sim_stamp(self, sim_t: float):
        from builtin_interfaces.msg import Time

        sec = int(sim_t)
        return Time(sec=sec, nanosec=int(round((sim_t - sec) * 1e9)))

    # -- the vec-pmp goal path ---------------------------------------------
    def _goal_taken(self, since_t: float, spawn_xy) -> bool:
        """True once the stack demonstrably reacted to the goal.

        "Taken" = the robot moved >=2 cm from where it stood at publish time,
        OR any wheel command was emitted. Wheel commands only flow during
        playback (vec-pmp) or while the nav2 chain drives, so either is a
        reaction to THIS goal, never an idle artifact.
        """
        st = self.state
        moved = any(math.hypot(x - spawn_xy[0], y - spawn_xy[1]) >= 0.02
                    for (t, x, y, _yaw) in st["track"] if t >= since_t)
        return moved or len(st["wheel"]) > 0

    def _drive_vec_pmp(self, goal_xy, gyaw, goal_t, timeout_s, backstop_s) -> dict:
        from geometry_msgs.msg import PoseStamped

        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        _, _, qz, qw = _yaw_quat(gyaw)
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        # /goal_pose is a one-shot publish into a live DDS graph, and the first
        # smoke cell failed exactly there: the publish matched 3 endpoints and
        # was nevertheless never received by the corrector -- the robot sat
        # still for the whole 101 s sim deadline, and the same QoS delivered
        # fine minutes later from a fresh publisher. So: count from the
        # PUBLISHER side (count_publishers = subscriptions matched to my
        # publisher; count_subscribers would count publishers matched to my
        # own subscription, which the corrector's sentinel publisher satisfies
        # trivially), then verify the goal was TAKEN instead of trusting the
        # publish -- republishing ONLY while nothing reacted, because a second
        # New goal mid-playback restarts the corrector from chunk 0 (the
        # corrector re-sends its action on every /goal_pose, and vector_field
        # republishes goals too, so duplicates are normal stack behaviour).
        subs = 0
        end = time.monotonic() + 30.0
        while time.monotonic() < end:
            subs = self.node.count_publishers("/goal_pose")
            if subs >= 2:
                break
            self._rclpy.spin_once(self.node, timeout_sec=0.5)
        self.spin(1.5)  # matched is not ready -- random_goals' settle
        if subs < 2:
            return {"terminal": False, "goal_taken": False,
                    "error": f"only {subs} /goal_pose subscriptions matched (want 2)",
                    "stack_terminal": "none", "nav2_error_code": None}

        spawn_xy = (self.state["pose"][0], self.state["pose"][1])
        ids_before = set(self.state["plan_goal_ids"])

        def taken_now() -> bool:
            return (bool(self.state["plan_goal_ids"] - ids_before)
                    or self._goal_taken(goal_t, spawn_xy))
        taken = False
        tries = 0
        for tries in range(1, self.goal_tries + 1):
            msg.header.stamp = self._sim_stamp(self.state["sim_t"])
            self._goal_pub.publish(msg)
            # The ack window is anchored at NOW, not at goal_t: goal_t goes
            # stale after the first window (sim time kept advancing), and an
            # anchored-at-goal_t window exits instantly, turning tries 2..N
            # into immediate republishes.
            ack_t = self.state["sim_t"]
            self._wait_terminal(ack_t, self.goal_ack_wait, backstop_s,
                                kind="sentinel", extra_pred=taken_now)
            # Re-check taken DIRECTLY instead of trusting the wait's exit
            # reason: the ack window can also end on its own sim-time deadline
            # (timed_out=True) a step before the solve completes and motion
            # starts -- republishing then would restart a playback that was
            # already healthy.
            if taken_now():
                taken = True
                break
            # Genuinely not taken inside the ack window: republish. The real
            # deadline has not started -- the world kept running, nothing was
            # consumed.
        if not taken:
            return {"terminal": False, "goal_taken": False,
                    "error": f"goal never taken after {tries} publish(es) "
                             f"(no planner goal, no motion, no wheel commands)",
                    "stack_terminal": "none", "nav2_error_code": None}
        # Planning phase: silent until the whole rollout is buffered. Wait in
        # WALL time for the first wheel command (or a terminal sentinel =
        # the planner gave up). Sim time is the wrong clock here: the solve
        # is CPU-bound, so a sim deadline would measure the RTF.
        st = self.state
        wall0 = time.monotonic()
        plan_cap = min(self.plan_wall_cap, backstop_s)
        while not st["wheel"] and not st["sentinel"]:
            if time.monotonic() - wall0 > plan_cap:
                break
            self._rclpy.spin_once(self.node, timeout_sec=0.1)
        plan_wall = time.monotonic() - wall0
        if not st["wheel"]:
            out = {"terminal": bool(st["sentinel"]),
                   "terminal_kind": "sentinel" if st["sentinel"] else None,
                   "planner_timeout": not st["sentinel"],
                   "stack_terminal": "none", "nav2_error_code": None,
                   "travel_time": ((st["sim_t"] - goal_t)
                                   if st["sim_t"] is not None else None),
                   "drive_wall": 0.0, "drive_t0": None}
        else:
            # The drive deadline starts at the first wheel command.
            drive_t0 = st["wheel"][0][0]
            out = self._wait_terminal(drive_t0, timeout_s,
                                      max(1.0, backstop_s - plan_wall),
                                      kind="sentinel")
            out["drive_t0"] = drive_t0
        out["plan_wall_s"] = plan_wall
        out["goal_publish_tries"] = tries
        return out


def _nav2_error_name(code: int) -> str:
    from nav2_msgs.action import NavigateToPose

    names = [k for k in dir(NavigateToPose.Result)
             if not k.startswith("_") and isinstance(getattr(NavigateToPose.Result, k), int)
             and getattr(NavigateToPose.Result, k) == code]
    return ",".join(names) or f"code-{code}"


# ---------------------------------------------------------------------------
# orchestration: teardown / bring-up / terrain / one row
# ---------------------------------------------------------------------------

# Scout Mini body (PlannerConfig.fp_half_length / fp_half_width) and the
# perimeter sampling used for the wall-contact score.
FP_HALF_LENGTH, FP_HALF_WIDTH, FP_SPACING = 0.315, 0.2925, 0.02
_SIGNED_DIST = {}


def _signed_wall_dist(floor: int):
    """Signed distance to the baked map's walls [m] (+ free, - inside), cached."""
    if floor not in _SIGNED_DIST:
        import yaml
        from PIL import Image
        from scipy.ndimage import distance_transform_edt
        maps = os.path.join(WORKSPACE, "src", "rudn-ordjo-building", "maps")
        meta = yaml.safe_load(open(os.path.join(maps, f"floor_{floor}.yaml")))
        img = np.asarray(Image.open(os.path.join(maps, meta["image"])))[::-1]  # row 0 = origin y
        wall = img == 0
        r = meta["resolution"]
        d = (distance_transform_edt(~wall) - distance_transform_edt(wall)) * r
        _SIGNED_DIST[floor] = (d, r, meta["origin"][0], meta["origin"][1])
    return _SIGNED_DIST[floor]


def wall_contact(track, floor: int) -> dict:
    """Where the ground-truth BODY met the baked map's walls (#34).

    Scores the rectangle's perimeter, not the centre, against the map the
    stack planned on. Under phantom_walls the robot drives through walls, and
    this is the record of it; with solid walls it shows contact (one-cell,
    5 cm, raster resolution: a grazing touch can read either way).
    """
    if len(track) < 2:
        return dict(wall_min_clear=None, wall_overlap_s=None, wall_first_t=None,
                    wall_episodes=None)
    d, r, ox, oy = _signed_wall_dist(floor)
    T = np.asarray(track, dtype=float)
    nl, nw = int(2 * FP_HALF_LENGTH / FP_SPACING) + 1, int(2 * FP_HALF_WIDTH / FP_SPACING) + 1
    lx, wy = np.linspace(-FP_HALF_LENGTH, FP_HALF_LENGTH, nl), np.linspace(-FP_HALF_WIDTH, FP_HALF_WIDTH, nw)
    body = np.vstack([np.c_[lx, np.full(nl, FP_HALF_WIDTH)], np.c_[lx, np.full(nl, -FP_HALF_WIDTH)],
                      np.c_[np.full(nw, FP_HALF_LENGTH), wy], np.c_[np.full(nw, -FP_HALF_LENGTH), wy]])
    c, s = np.cos(T[:, 3])[:, None], np.sin(T[:, 3])[:, None]
    qx = T[:, 1:2] + body[None, :, 0] * c - body[None, :, 1] * s
    qy = T[:, 2:3] + body[None, :, 0] * s + body[None, :, 1] * c
    col = np.clip(((qx - ox) / r).astype(int), 0, d.shape[1] - 1)
    row = np.clip(((qy - oy) / r).astype(int), 0, d.shape[0] - 1)
    clear = d[row, col].min(axis=1)
    # A body point on a wall PIXEL. The baked walls are often one cell thick,
    # so the signed distance bottoms out at -r and cannot rank depth.
    inside = clear < 0
    dt = np.diff(T[:, 0], append=T[-1, 0])
    return dict(wall_min_clear=round(float(clear.min()), 3),
                wall_overlap_s=round(float(dt[inside].sum()), 2),
                wall_first_t=(round(float(T[inside, 0][0]), 2) if inside.any() else None),
                wall_episodes=int(np.count_nonzero(np.diff(inside.astype(int)) == 1) + inside[0]))


def bring_up_stack(args, arm: str, spawn, log_path: str):
    """Fresh stack via fixture_up.sh. Returns (ok, attempts, detail)."""
    nav_mode = "vec-pmp" if arm == "ours" else "nav2"
    cmd = [
        "bash", os.path.join(HERE, "fixture_up.sh"),
        "--worker", str(args.worker or ""),
        "--nav-mode", nav_mode,
        "--nav2-controller", ARM_TO_CONTROLLER.get(arm, "mppi"),
        "--nav2-profile", args.nav2_profile,
        "--corrector", args.corrector,
        *(["--wheel-bias", args.wheel_bias] if args.wheel_bias else []),
        *(["--phantom-walls"] if args.phantom_walls else []),
        *(["--no-ekf-wheel-yaw"] if args.no_ekf_wheel_yaw else []),
        *(["--lidar-odom"] if args.lidar_odom else []),
        *(["--amcl-params", args.amcl_params] if args.amcl_params else []),
        # The comparison's slip comes from tools/spawn_patches.py (seed 0,
        # along-path). The fixture's own near-origin patches must be OFF, or a
        # wall strike has two candidate causes (CLAUDE.md, SURFACE_PATCHES).
        "--patches", "false",
        "--localization", args.localization,
        "--floor", str(args.floor),
        "--frontier", "false",
        "--spawn", f"{spawn[0]:.6f}", f"{spawn[1]:.6f}", f"{spawn[2]:.6f}",
        "--tries", str(args.tries),
        "--timeout", str(args.stack_timeout),
    ]
    with open(log_path, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=WORKSPACE)
    text = open(log_path).read() if os.path.exists(log_path) else ""
    # Only the real attempt banners (column 0); the "last 20 lines" echo of
    # the fixture log re-prints earlier banners indented.
    attempts = len(re.findall(r"^\[fixture-up\] attempt ", text, flags=re.M))
    ok = rc == 0 and "READY on attempt" in text
    return ok, attempts, text


def _scope_cgroup_dir(worker) -> str | None:
    """/sys/fs/cgroup path of the stack's systemd --user scope (#28), or None.

    fixture_up.sh launches the stack as agx-w<N>.scope (agx-w0 = default
    partition). None when the scope does not exist -- no linger, no user bus,
    or a stack launched by hand -- and the row's cgroup fields are then null.
    """
    unit = f"agx-w{worker or 0}.scope"
    try:
        out = subprocess.run(["systemctl", "--user", "show", "-P", "ControlGroup", unit],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    if not out:
        return None
    path = "/sys/fs/cgroup" + out
    return path if os.path.isdir(path) else None


def cgroup_snapshot(worker) -> dict | None:
    """cpu.stat counters + memory.peak + pids.current of the stack scope."""
    d = _scope_cgroup_dir(worker)
    if d is None:
        return None
    snap = {"t": time.monotonic(), "dir": d}
    try:
        with open(os.path.join(d, "cpu.stat")) as fh:
            for ln in fh:
                k, _, v = ln.partition(" ")
                if k in ("usage_usec", "nr_throttled", "throttled_usec"):
                    snap[k] = int(v)
        for f in ("memory.peak", "memory.current", "pids.current"):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                snap[f] = int(open(p).read().strip())
    except (OSError, ValueError):
        return None
    return snap


def cgroup_fields(a: dict | None, b: dict | None) -> dict:
    """Row fields for the window a->b; all null if either snapshot is missing.

    The window is stack-ready -> end of drive (terrain + planning + driving).
    Bring-up is excluded on purpose: it is launch noise, not the measured cell.
    `cg_cpu_cores` = usage / wall, i.e. mean cores the stack burned. memory.peak
    is the scope's lifetime high-water mark (scope lifetime == one cell).
    """
    keys = ("cg_usage_usec", "cg_nr_throttled", "cg_throttled_usec", "cg_wall_s",
            "cg_cpu_cores", "cg_memory_peak", "cg_pids_start", "cg_pids_end")
    if not a or not b or a.get("dir") != b.get("dir"):
        return {k: None for k in keys}
    wall = b["t"] - a["t"]
    du = b.get("usage_usec", 0) - a.get("usage_usec", 0)
    return {
        "cg_usage_usec": du,
        "cg_nr_throttled": b.get("nr_throttled", 0) - a.get("nr_throttled", 0),
        "cg_throttled_usec": b.get("throttled_usec", 0) - a.get("throttled_usec", 0),
        "cg_wall_s": round(wall, 1),
        "cg_cpu_cores": round(du / 1e6 / wall, 3) if wall > 0 else None,
        "cg_memory_peak": b.get("memory.peak"),
        "cg_pids_start": a.get("pids.current"),
        "cg_pids_end": b.get("pids.current"),
    }


def spawn_patches(args, plan_path: str):
    """tools/spawn_patches.py, in the worker's partition. Returns its JSON."""
    cmd = [os.path.join(HERE, "with-worker"), str(args.worker or ""), "python3",
           os.path.join(HERE, "spawn_patches.py"),
           "--plan", plan_path, "--seed", str(args.seed), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=WORKSPACE)
    for line in reversed((proc.stdout or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    return {"ok": False, "error": (proc.stderr or proc.stdout or "no output")[-400:]}


def load_plan(path: str) -> dict:
    with np.load(path) as f:
        poses = np.asarray(f["poses"], dtype=float)
        start = np.asarray(f["start_xy"], dtype=float) if "start_xy" in f else poses[0, :2]
        goal = np.asarray(f["goal_xy"], dtype=float) if "goal_xy" in f else poses[-1, :2]
        dt = float(f["dt_sample"]) if "dt_sample" in f else 0.1
    goal_yaw = float(poses[-1, 2])
    duration = poses.shape[0] * dt
    return {"poses": poses, "start": start, "goal": goal,
            "goal_yaw": goal_yaw, "dt": dt, "duration": duration}


def run_one(args, plan_path: str) -> dict:
    """One cell, end to end. Always returns a row (never raises)."""
    name = os.path.basename(plan_path)
    plan_name = name[:-4] if name.endswith(".npz") else name
    tag = f"w{args.worker or '0'}_{args.arm}_{plan_name}"
    up_log = os.path.join(args.log_dir, f"up_{tag}.log")
    t_start = time.monotonic()
    plan = load_plan(plan_path)

    row = {
        "plan": plan_name,
        "plan_path": plan_path,
        "arm": args.arm,
        "seed": args.seed,
        "corrector": args.corrector,
        "wheel_bias": args.wheel_bias or None,
        "localization": args.localization,
        "phantom_walls": args.phantom_walls,
        "ekf_wheel_yaw": not args.no_ekf_wheel_yaw,
        "lidar_odom": args.lidar_odom,
        "amcl_params": args.amcl_params,
        "nav2_controller": ARM_TO_CONTROLLER.get(args.arm),
        "nav2_profile": args.nav2_profile if args.arm != "ours" else None,
        "floor": args.floor,
        "world": args.world,
        "start_xy": [float(v) for v in plan["start"]],
        "goal_xy": [float(v) for v in plan["goal"]],
        "goal_yaw": plan["goal_yaw"],
        "plan_duration_s": round(plan["duration"], 2),
        "sim_timeout_s": round(3.0 * plan["duration"] + 60.0, 1),
    }

    # 1. fresh stack --------------------------------------------------------
    # Spawn yaw: the plan's own start heading, poses[0,2] (the npz start_xy key
    # carries no heading). --spawn-yaw-zero pins 0.0 for a flat-ground debug.
    spawn_yaw = 0.0 if args.spawn_yaw_zero else float(plan["poses"][0, 2])
    spawn = (float(plan["start"][0]), float(plan["start"][1]), spawn_yaw)
    up_ok, attempts, _ = bring_up_stack(args, args.arm, spawn, up_log)
    row["stack_up_attempts"] = attempts
    row["stack_up_s"] = round(time.monotonic() - t_start, 1)
    if not up_ok:
        row.update(outcome="stack-failed", stack_up_ok=False,
                   error=f"fixture_up failed after {attempts} attempts (log {up_log})",
                   wall_time=round(time.monotonic() - t_start, 1))
        return row
    # Per-stack cgroup counters (#28): snapshot now, diff at the end of run_one.
    cg_start = cgroup_snapshot(args.worker)
    row["cg_scope"] = cg_start["dir"] if cg_start else None
    row.update(cgroup_fields(None, None))

    # 2. terrain ------------------------------------------------------------
    t_terr = time.monotonic()
    terr = spawn_patches(args, plan_path) if not args.no_terrain else {"ok": True, "patches": []}
    row["patches"] = terr.get("patches", [])
    row["terrain_s"] = round(time.monotonic() - t_terr, 1)
    if not terr.get("ok"):
        row.update(outcome="stack-failed", stack_up_ok=True,
                   error=f"terrain failed: {terr.get('error')}",
                   wall_time=round(time.monotonic() - t_start, 1))
        return row

    # 3. drive + score --------------------------------------------------------
    sim_timeout = 3.0 * plan["duration"] + 60.0
    # The vec-pmp ack window must cover the PIPELINE'S PLANNING LATENCY, not
    # just reaction time: with wait_for_complete (default) the corrector
    # buffers the WHOLE rollout before the first wheel command (CLAUDE.md),
    # and that buffering takes ~(duration/0.5 s segment) x ~0.7 s wall per
    # chunk -- ~19 s of dead silence even for this 13.8 s plan. A fixed
    # 15 s window expired just before motion began, and every "retry"
    # restarted the solve-and-buffer cycle (smoke attempts 4-5). Tie the
    # window to the plan; the retry then only fires on a genuinely lost
    # publish.
    # (The ack is now the planner's action status, which arrives within a
    # second; the planning phase has its own wall cap in _drive_vec_pmp.)
    plan_scaled_ack = args.goal_ack_wait
    drv = None
    try:
        drv = CompareDriver(args.world, args.model,
                            goal_tries=args.goal_tries,
                            goal_ack_wait=plan_scaled_ack,
                            plan_wall_cap=args.plan_wall_cap)
        if not drv.spin_until(lambda: drv.state["pose"] is not None, 30.0):
            row.update(outcome="stack-failed", stack_up_ok=True,
                       error=f"no ground-truth pose on {drv.topic_pose}",
                       wall_time=round(time.monotonic() - t_start, 1))
            return row
        start_gt = drv.state["pose"]
        row["start_gt"] = [round(v, 4) for v in start_gt]
        row["spawn_err"] = round(math.hypot(start_gt[0] - plan["start"][0],
                                            start_gt[1] - plan["start"][1]), 4)

        # Whatever bring-up and terrain left of the cell's wall budget bounds
        # the goal phase (planning + driving) as a whole.
        cell_left = args.cell_wall_cap - (time.monotonic() - t_start)
        res = drv.send_goal(args.arm, plan["goal"], plan["goal_yaw"],
                            timeout_s=sim_timeout,
                            backstop_s=max(1.0, min(args.wall_backstop, cell_left)))
        if res.get("goal_t") is None:
            # The goal never left the tool (no /clock, no action server...):
            # there is nothing to score, and the metrics below would divide by
            # or compare against None.
            row.update(outcome="stack-failed", stack_up_ok=True,
                       error=res.get("error") or "goal not sent",
                       wall_time=round(time.monotonic() - t_start, 1))
            return row
        row["stack_terminal"] = res.get("stack_terminal")
        if res.get("nav2_error_code") is not None:
            row["nav2_error_code"] = res.get("nav2_error_code")
            row["nav2_error_name"] = res.get("nav2_error_name")
        if res.get("error"):
            row["error"] = res["error"]

        st = drv.state
        goal_t = res.get("goal_t")
        drive_t0 = res.get("drive_t0", goal_t)
        end_pose = st["pose"]
        # Geometric track: XY only (the timestamp column must NOT enter the
        # length/curvature integrals -- the first smoke run "travelled" 101 m
        # standing still because t was in the polyline).
        track = [(x, y) for (t, x, y, _yaw) in st["track"] if t >= goal_t]
        travelled = polyline_length(np.array(track)) if len(track) >= 2 else 0.0
        # The full timed track, for tools/plot_compare_run.py (the row keeps
        # only scalars, and a scalar cannot show *how* an arm got stuck).
        track_path = os.path.join(args.log_dir, f"track_{tag}.npz")
        wheel = st["wheel"]
        diag = st["diag"]
        # Compressed: the per-tick diagnostics are the bulk of the file.
        # plan_poses is the LIBRARY plan the goal came from; live_plan is what
        # the stack solved and followed (empty for nav2 arms / no plan).
        np.savez_compressed(
                 track_path, track=np.array(st["track"], dtype=float),
                 wheel_t=np.array([c[0] for c in wheel], dtype=float),
                 wheel_cmd=np.array([c[1] for c in wheel], dtype=float).reshape(-1, 4),
                 diag_t=np.array([c[0] for c in diag], dtype=float),
                 diag=(np.array([c[1] for c in diag], dtype=float) if diag
                       else np.zeros((0, 16))),
                 live_plan=(st["live_plan"] if st["live_plan"] is not None
                            else np.zeros((0, 3))),
                 goal_t=goal_t, plan_poses=plan["poses"],
                 goal_xy=np.asarray(plan["goal"], dtype=float))
        row["track_path"] = track_path
        final_err = (math.hypot(end_pose[0] - plan["goal"][0],
                                end_pose[1] - plan["goal"][1])
                     if end_pose is not None else None)
        travel_time = res.get("travel_time")

        # planner-failed is inferred from TERMINAL + NO MOTION, not from a plan
        # topic: /optimal_trajectory is silent even during solves in this
        # stack, and vector_field republishes /plan (~4 Hz) while idle, so
        # neither discriminates. A stack that reached ANY terminal state
        # (sentinel or nav2 result) without moving a metre of wheel command is
        # one whose planner never delivered a followable plan.
        if final_err is not None and final_err <= args.tolerance:
            outcome = "arrived"
        elif res.get("planner_timeout"):
            outcome = "planner-timeout"
        elif res.get("timed_out") or res.get("wall_backstop"):
            outcome = "timeout"
        elif res.get("terminal") and travelled < 0.05:
            outcome = "planner-failed"
        elif res.get("terminal"):
            outcome = "failed"
        else:
            # No terminal and no deadline flag: the goal never left the tool.
            outcome = "stack-failed"

        row.update(
            outcome=outcome,
            stack_up_ok=True,
            final_err=None if final_err is None else round(final_err, 4),
            end_gt=None if end_pose is None else [round(v, 4) for v in end_pose],
            path_length=round(travelled, 3),
            travel_time=None if travel_time is None else round(travel_time, 2),
            max_curvature=round(max_menger_curvature(np.array(track)), 3) if len(track) >= 3 else 0.0,
            control_energy=round(control_energy(st["wheel"], goal_t,
                                                (goal_t or 0) + (travel_time or 0.0),
                                                grid_dt=args.ctrl_grid_dt), 2),
            n_wheel_cmds=len(st["wheel"]),
            n_plan_topic_msgs=st["n_plan_paths"],
            goal_publish_tries=res.get("goal_publish_tries", 1),
            max_cross_track=(round(max(nearest_dist_to_polyline((x, y), plan["poses"][:, :2])
                                       for (x, y) in track), 4)
                             if track else None),
            **wall_contact([s for s in st["track"] if s[0] >= goal_t], args.floor),
            # Over the drive phase only: sim seconds since drive_t0 per wall
            # second of that same wait.
            rtf=(round((st["sim_t"] - drive_t0) / res["drive_wall"], 3)
                 if drive_t0 is not None and st["sim_t"] is not None
                 and res.get("drive_wall") else None),
            drive_wall_s=round(res.get("drive_wall", 0.0), 1),
            drive_t0=None if drive_t0 is None else round(drive_t0, 2),
            plan_wall_s=(round(res["plan_wall_s"], 1)
                         if res.get("plan_wall_s") is not None else None),
        )
    except Exception as exc:  # noqa: BLE001 -- a crashed run is a row, not a lost one
        row.update(outcome="stack-failed", stack_up_ok=True,
                   error=f"{exc.__class__.__name__}: {exc}",
                   wall_time=round(time.monotonic() - t_start, 1))
    finally:
        if drv is not None:
            drv.close()
        row.update(cgroup_fields(cg_start, cgroup_snapshot(args.worker)))

    row["wall_time"] = round(time.monotonic() - t_start, 1)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_plans(args):
    paths = []
    if args.plan:
        paths = [args.plan]
    elif args.plans:
        with open(args.plans) as fh:
            paths = [ln.strip() for ln in fh if ln.strip()]
    paths = [p.replace("__HOME__", os.path.expanduser("~")) for p in paths]
    if not paths:
        raise SystemExit("compare_run: no plans given")
    return paths


def existing_keys(path: str):
    """(plan, arm, seed) already recorded -- the resume guard."""
    keys = set()
    if not os.path.isfile(path):
        return keys
    with open(path) as fh:
        for ln in fh:
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue  # a torn last line must not un-resume the batch
            keys.add((r.get("plan"), r.get("arm"), r.get("seed")))
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", help="single plan .npz")
    ap.add_argument("--plans", help="file of plan .npz paths, one per line "
                                    "(__HOME__ is substituted)")
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, default=0,
                    help="terrain seed; MUST match the campaign (soak used 0)")
    ap.add_argument("--out", required=True, help="JSONL row sink, appended to")
    ap.add_argument("--skip-from", default=None,
                    help="also skip (plan,arm,seed) present here (default: --out)")
    ap.add_argument("--worker", type=int, default=0, help="worker id; 0 = default partition")
    ap.add_argument("--tolerance", type=float, default=0.5,
                    help="arrival tolerance on ground-truth final_err [m]")
    ap.add_argument("--tries", type=int, default=3, help="stack bring-up attempts per run")
    ap.add_argument("--stack-timeout", type=float, default=120.0,
                    help="per-attempt readiness wait [wall s]")
    ap.add_argument("--wall-backstop", type=float, default=900.0,
                    help="wall-clock backstop on the drive phase [s], for a "
                         "sim whose /clock died")
    ap.add_argument("--plan-wall-cap", type=float, default=600.0,
                    help="`ours`: wall seconds from goal to first wheel "
                         "command before the cell is planner-timeout")
    ap.add_argument("--cell-wall-cap", type=float, default=1500.0,
                    help="wall-clock cap on a whole cell (bring-up + terrain "
                         "+ planning + driving) [s]")
    ap.add_argument("--ctrl-grid-dt", type=float, default=0.1)
    ap.add_argument("--corrector", default="tvlqr", help="vec-pmp arm's corrector")
    ap.add_argument("--wheel-bias", default="",
                    help="sim-only actuator fault for the vec-pmp arm (#27): "
                         "'fl,rl,fr,rr' command scale, e.g. 0.9,0.9,0.9,0.9")
    ap.add_argument("--localization", default="amcl")
    ap.add_argument("--phantom-walls", action="store_true",
                    help="walls visible to the lidar but without collision "
                         "(#34); contact is then scored by wall_* row fields")
    ap.add_argument("--no-ekf-wheel-yaw", action="store_true",
                    help="EKF ignores the chi-biased wheel yaw rate (#34)")
    ap.add_argument("--lidar-odom", action="store_true",
                    help="fuse rf2o laser odometry in the EKF and drop the "
                         "wheels' yaw rate and vy=0 (#34: observe turn skid)")
    ap.add_argument("--amcl-params", default=None,
                    help="params file layered over nav2_params.yaml's amcl (#34)")
    ap.add_argument("--nav2-profile", default="compare_static",
                    help="config/nav2_profile_<name>.yaml for the nav2 arms "
                         "(costmap inflation, collision monitor); '' = stock")
    ap.add_argument("--floor", type=int, default=6, help="baked map / gz floor number")
    ap.add_argument("--world", default="ordjo_world")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--no-terrain", action="store_true",
                    help="skip along-path patches (flat-ground debugging)")
    ap.add_argument("--goal-tries", type=int, default=3,
                    help="republish attempts if the goal is never taken "
                         "(only while nothing has reacted -- see "
                         "_drive_vec_pmp)")
    ap.add_argument("--goal-ack-wait", type=float, default=15.0,
                    help="sim seconds to wait for the goal to be taken (the "
                         "planner's action status shows a new goal) before "
                         "republishing")
    ap.add_argument("--log-dir", default="/tmp/compare_runs",
                    help="per-run stack bring-up logs")
    ap.add_argument("--spawn-yaw-zero", action="store_true",
                    help="spawn with yaw 0 instead of the plan's poses[0,2] "
                         "(flat-ground debugging only -- measured runs must "
                         "spawn as the plan assumed)")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    plans = resolve_plans(args)
    skip = existing_keys(args.out)
    if args.skip_from and os.path.abspath(args.skip_from) != os.path.abspath(args.out):
        skip |= existing_keys(args.skip_from)

    n_made = 0
    for plan_path in plans:
        name = os.path.basename(plan_path)[:-4]
        if (name, args.arm, args.seed) in skip:
            print(f"[compare] skip {name} {args.arm} (row exists)", flush=True)
            continue
        print(f"[compare] run {name} {args.arm} worker={args.worker or 'default'}",
              flush=True)
        try:
            row = run_one(args, plan_path)
        except Exception as exc:  # noqa: BLE001 -- keep the batch alive
            row = {"plan": name, "arm": args.arm, "seed": args.seed,
                   "outcome": "stack-failed", "error": f"{exc.__class__.__name__}: {exc}"}
        # Keep the whole stack log: fixture_up.sh reuses /tmp/fixtureN.log, so
        # the next cell would overwrite the only evidence of why this one failed.
        stack_log = f"/tmp/fixture{args.worker or ''}.log"
        if os.path.isfile(stack_log):
            kept = os.path.join(args.log_dir, f"stack_w{args.worker or '0'}_{args.arm}_{name}.log")
            shutil.copyfile(stack_log, kept)
            row["stack_log"] = kept
        with open(args.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        n_made += 1
        print(f"[compare]   -> {row.get('outcome')} final_err={row.get('final_err')} "
              f"wall={row.get('wall_time')}", flush=True)
    print(f"[compare] done: {n_made} new rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
