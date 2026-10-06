"""SupervisorCore: the stack-independent supervisory layer (#40, #41).

ROS-free and the same for every stack, so every arm gets the same checks.
Thresholds and conventions were agreed with the advisor on 2026-10-06; the
decision record is docs/supervisor-plan.md (Q4, Q5, wall safety).

Progress. The plan is cut into segments by waypoints placed every
``waypoint_frac`` of its ARC LENGTH (not of its sample index: PMP plans contain
turn-in-place stretches that cover time but no distance). A waypoint is
reached when the robot centre comes within ``reach_radius`` of it; reaching is
monotone, so a waypoint is never un-reached. Waypoints are visited in order,
but a later one reached first also marks the earlier ones (a skipped waypoint
must not stall the layer).

Time. The time counter is PER SEGMENT and is reset at each waypoint. There is no
whole-route re-timing. The lag in the current segment is

    lag = (t - t_seg_start) - (t_plan[k*] - t_plan[wp_prev])

where k* is the robot's progress index: the nearest plan sample in a
forward window, never moving backwards. Positive lag means the robot is behind
the plan's schedule. On reaching a waypoint the decision carries
``retime_index`` = the robot's progress index k* at that moment (not the
waypoint's own index: "reached" allows up to reach_radius short of it), so a
time-indexed follower (our playback, GMPC) can jump its clock there and the
new segment's counter starts from zero.

Trigger (local re-plan). Both conditions must hold, as agreed with the advisor:
  (a) lag >= trigger_ratio * prev_lag, where prev_lag is the lag at which the
      previous segment closed; if prev_lag <= 0 (the robot was on or ahead of
      schedule), (a) holds trivially;
  (b) lag converted to distance through the segment's plan speed,
      lag * v_seg, exceeds reach_radius (the robot radius).
The trigger is latched once per segment, so one bad segment asks for one
re-plan, not one per tick.

Walls. ``clearance`` is the signed hull clearance (walls.py). Speed is scaled by
clip((d - stop_dist) / (slow_dist - stop_dist), 0, 1). At or below stop_dist
the robot stops and a re-plan is requested immediately (reason "wall").
Within the slow band a re-plan is also requested, latched once per segment
(reason "near_wall"); "maximum safety" was the advisor's priority.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SupervisorConfig:
    enabled: bool = False
    waypoint_frac: float = 0.05     # waypoint every 5% of route arc length
    reach_radius: float = 0.35      # [m] robot radius; also the trigger's distance floor
    slow_dist: float = 0.5          # [m] hull clearance where slowing starts
    stop_dist: float = 0.15         # [m] hull clearance where the robot stops
    trigger_ratio: float = 2.0      # lag must at least double the previous segment's
    window: int = 400               # forward search window for k* [samples]
    cruise_speed: float = 0.4       # [m/s] time law for plans given without times
    min_seg_speed: float = 0.05     # [m/s] floor for v_seg (turn-in-place segments)


@dataclass
class Decision:
    k_star: int                     # progress index on the plan
    waypoint: int                   # ordinal of the NEXT waypoint (== n_waypoints when done)
    n_waypoints: int
    reached: bool                   # a waypoint was reached on this update
    retime_index: Optional[int]     # plan index to jump a time-indexed clock to
    lag_s: float                    # current segment lag [s]
    prev_lag_s: float               # lag at which the previous segment closed [s]
    lag_m: float                    # lag_s * v_seg [m]
    clearance: float                # hull clearance [m] (inf when unknown)
    speed_scale: float              # multiply the outgoing command by this
    stop: bool                      # publish zero this tick
    trigger: bool                   # request a re-plan this tick
    reason: str = ""                # "lag" | "wall" | "near_wall" | ""

    FIELDS = ("k_star", "waypoint", "n_waypoints", "reached", "lag_s", "prev_lag_s",
              "lag_m", "clearance", "speed_scale", "stop", "trigger")

    def as_array(self) -> List[float]:
        """Flat layout for a Float64MultiArray topic. Append-only."""
        return [float(getattr(self, f)) for f in self.FIELDS]


class SupervisorCore:
    def __init__(self, cfg: Optional[SupervisorConfig] = None):
        self.cfg = cfg or SupervisorConfig()
        self.poses = np.zeros((0, 3))
        self.replans = 0
        self.stops = 0
        self._stopped = False

    # ---- plan ---------------------------------------------------------
    def set_plan(self, poses, times=None, t_now: float = 0.0) -> None:
        """Install a plan: (N,3) poses (x, y, yaw), optional (N,) times [s].

        Without times the plan is an arc-length path (as Nav2 FollowPath
        receives) and ``cruise_speed`` supplies the time law. Resets all
        per-plan state; call it again after every re-plan.
        """
        p = np.asarray(poses, dtype=float).reshape(-1, 3)
        if len(p) == 0:
            raise ValueError("empty plan")
        seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        if times is None:
            self.t_plan = self.s / max(self.cfg.cruise_speed, 1e-6)
        else:
            t = np.asarray(times, dtype=float).reshape(-1)
            if len(t) != len(p):
                raise ValueError("times and poses differ in length")
            self.t_plan = t - t[0]
        self.poses = p
        total = self.s[-1]
        n = max(1, int(round(1.0 / self.cfg.waypoint_frac)))
        if total <= 0:
            idx = [len(p) - 1]
        else:
            targets = total * np.arange(1, n + 1) / n
            idx = [int(min(np.searchsorted(self.s, x), len(p) - 1)) for x in targets]
        # Strictly increasing, last == final sample.
        wp: List[int] = []
        for i in idx:
            if not wp or i > wp[-1]:
                wp.append(i)
        if wp[-1] != len(p) - 1:
            wp.append(len(p) - 1)
        self.wp_index = wp
        self.next_wp = 0
        self.k_star = 0
        self.t_seg_start = float(t_now)
        self.k_seg_start = 0
        self.prev_lag = 0.0
        self.latched = False

    @property
    def n_waypoints(self) -> int:
        return len(self.wp_index)

    @property
    def done(self) -> bool:
        return self.next_wp >= self.n_waypoints

    # ---- update -------------------------------------------------------
    def _progress(self, x: float, y: float) -> int:
        lo = self.k_star
        hi = min(len(self.poses), lo + self.cfg.window + 1)
        d = np.hypot(self.poses[lo:hi, 0] - x, self.poses[lo:hi, 1] - y)
        return lo + int(np.argmin(d))

    def _seg_speed(self) -> float:
        a = self.k_seg_start
        b = self.wp_index[min(self.next_wp, self.n_waypoints - 1)]
        dt = self.t_plan[b] - self.t_plan[a]
        ds = self.s[b] - self.s[a]
        v = ds / dt if dt > 1e-9 else self.cfg.cruise_speed
        return max(v, self.cfg.min_seg_speed)

    def update(self, x: float, y: float, yaw: float, t: float,
               clearance: float = math.inf) -> Decision:
        cfg = self.cfg
        if not len(self.poses):
            raise RuntimeError("update() before set_plan()")
        self.k_star = self._progress(x, y)

        # Waypoints: reached within radius; a later one reached marks earlier ones.
        reached, retime = False, None
        j = self.next_wp
        hit = None
        while j < self.n_waypoints and self.wp_index[j] <= self.k_star + cfg.window:
            w = self.wp_index[j]
            if math.hypot(self.poses[w, 0] - x, self.poses[w, 1] - y) <= cfg.reach_radius:
                hit = j
            if w > self.k_star:
                break
            j += 1
        if hit is not None:
            # The new segment starts at the robot's ACTUAL progress, not at the
            # waypoint's index: "reached" means within reach_radius, i.e. up to
            # 0.35 m short of it, and charging that gap as lag made an
            # on-schedule robot trigger (unit test).
            k = self.k_star
            self.prev_lag = (t - self.t_seg_start) - (self.t_plan[k] - self.t_plan[self.k_seg_start])
            self.next_wp = hit + 1
            self.t_seg_start = float(t)
            self.k_seg_start = k
            self.latched = False
            reached, retime = True, k

        lag = (t - self.t_seg_start) - (self.t_plan[self.k_star] - self.t_plan[self.k_seg_start])
        lag_m = max(lag, 0.0) * self._seg_speed()

        # Walls.
        span = max(cfg.slow_dist - cfg.stop_dist, 1e-9)
        scale = float(np.clip((clearance - cfg.stop_dist) / span, 0.0, 1.0)) if math.isfinite(clearance) else 1.0
        stop = math.isfinite(clearance) and clearance <= cfg.stop_dist

        trigger, reason = False, ""
        if stop:
            # Counted once per stop episode, but the trigger stays up every
            # tick while stopped: the adapter must not resume until a re-plan
            # (or the robot backing off) lifts the clearance.
            trigger, reason = True, "wall"
            if not self._stopped:
                self.stops += 1
                self.replans += 1
            self._stopped = True
        elif not self.latched and not self.done:
            ratio_ok = self.prev_lag <= 0 or lag >= cfg.trigger_ratio * self.prev_lag
            if ratio_ok and lag_m > cfg.reach_radius:
                trigger, reason = True, "lag"
            elif scale < 1.0:
                trigger, reason = True, "near_wall"
            if trigger:
                self.latched = True
                self.replans += 1
        if not stop:
            self._stopped = False

        return Decision(k_star=self.k_star, waypoint=self.next_wp, n_waypoints=self.n_waypoints,
                        reached=reached, retime_index=retime, lag_s=float(lag),
                        prev_lag_s=float(self.prev_lag), lag_m=float(lag_m),
                        clearance=float(clearance), speed_scale=scale, stop=bool(stop),
                        trigger=trigger, reason=reason)
