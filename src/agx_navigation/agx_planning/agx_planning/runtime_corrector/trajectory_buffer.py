"""Playback buffer for the offline-mode PMP planner.

The offline planner streams its rolled-out trajectory as PlanToGoal action
feedback: a sequence of *chunks*, each a run of per-tick samples. The planner
solves much faster than the chassis plays the trajectory back (a multi-second
trajectory is solved in a burst well under a second), so chunks cannot be
published on arrival -- they must be buffered and metered out. This class owns
that buffer and the playback cursor.

It carries no ROS2 dependency: a *chunk* is any object exposing the parallel
sequences the PlanToGoal feedback provides --

    chunk.wheel_left   [rad/s]   per-tick LEFT  wheel-pair setpoint
    chunk.wheel_right  [rad/s]   per-tick RIGHT wheel-pair setpoint
    chunk.pose_x       [m]       planned chassis pose along the nominal
    chunk.pose_y       [m]
    chunk.pose_theta   [rad]

so it is unit-testable with plain stand-in objects.

Call advance() once per PLAYING tick. When it returns None, is_done()
distinguishes "trajectory finished" from "waiting for the next chunk".

Two ways to index the cursor
----------------------------

**Time** (`advance()` with no pose) consumes exactly one sample per tick. The
plan is then a *trajectory*: sample k is what the robot should be doing at
t = k*dt, and playback ends after N ticks no matter where the robot actually is.

That is measurably wrong for a corrector. Every correction the corrector makes
costs forward speed, but the cursor does not know or care, so the samples run
out with the robot short of the goal -- the tighter the tracking, the bigger the
shortfall, because they are the same effect. Measured over 5 seeds on the baked
map, TVLQR cut cross-track rms 43% (0.056 -> 0.032 m) and finished 0.562 m short
against open loop's 0.067 m, missing the arrival threshold on 5 runs out of 5.

**Progress** (`advance(actual_xy)`) still steps one sample per tick, but GATES
that step: the cursor may never run more than `max_lead` samples ahead of where
the robot actually is, measured by projecting the pose onto the plan. A robot
that keeps up sees ordinary time playback; one that falls behind holds the
reference until it catches up, so the plan is played to its end instead of
expiring.

The gate is what makes this work, and replacing the time cursor with the
projection instead does NOT: the plan starts from rest, so sample 0 commands
zero wheel speed. With the cursor slaved to measured progress the robot never
moves, so the projection never advances, so the command stays zero -- a fixed
point. Measured on the fixture, that parked the robot on its spawn point for the
full timeout at `sample 0/199`. The feed-forward has to be allowed to lead the
robot; it just must not be allowed to abandon it.

The projection is only as good as the pose fed to it. Under `localization:=none`
that pose is raw odometry, which cannot observe slip and so tends to over-report
distance travelled; the gate then opens too readily, which is most of why gating
measured as nearly inert. It is worth re-measuring under `localization:=truth`
or `amcl`, where the pose means something.

The forward search is windowed (`max_skip`) and never moves backwards, so a plan
that loops back near itself cannot teleport the cursor across the loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# PMP costates carried per tick, in the order the RL corrector's obs expects:
#   [lam_x, lam_y, lam_theta, lam_wheel_left, lam_wheel_right].
_COSTATE_FIELDS = ("lam_x", "lam_y", "lam_theta", "lam_wheel_left", "lam_wheel_right")


@dataclass
class PlaybackSample:
    """One wheel command produced by the buffer for a single tick.

    `left`/`right` are the planner's publication-ready per-side wheel-pair
    setpoints; the node duplicates each across that side's two physical wheels
    (or splits them, once per-wheel corrections land). `pose` is the planned
    chassis pose at this tick -- the reference the corrector compares the
    measured pose against. `costates` is the per-tick PMP costate vector
    [lam_x, lam_y, lam_theta, lam_wheel_left, lam_wheel_right] when the planner
    feedback carries it (the RL corrector feeds it to the policy), else None.
    """

    left: float
    right: float
    pose: Tuple[float, float, float]  # planned (x, y, theta)
    costates: Optional[Tuple[float, ...]] = None  # PMP costates at this tick


class TrajectoryBuffer:
    """Stores incoming trajectory chunks and serves wheel samples."""

    def __init__(self, default_dt: float, max_skip: int = 20,
                 max_lead: int = 10) -> None:
        # Chunks can arrive out of order, so they are held by index and folded
        # into the flat sample list only as a contiguous run becomes available.
        # Playback indexes the flat list: a projection search has to look across
        # chunk boundaries, which a dict of chunks makes needlessly awkward.
        self._chunks: Dict[int, Any] = {}
        self._samples: List[PlaybackSample] = []
        self._next_chunk: int = 0
        self._cursor: int = 0
        self._at_end: bool = False
        self._max_skip = max(1, int(max_skip))
        # How far the reference may lead the robot, in samples.
        self._max_lead = max(0, int(max_lead))
        self._proj: int = 0
        self._progress_mode: bool = False
        self._active_dt: float = float(default_dt)
        self._default_dt: float = float(default_dt)
        self._active_traj_id: int = -1
        self._result_received: bool = False
        self._result_success: bool = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def active_traj_id(self) -> int:
        return self._active_traj_id

    @property
    def active_dt(self) -> float:
        return self._active_dt

    @property
    def result_received(self) -> bool:
        return self._result_received

    @property
    def result_success(self) -> bool:
        return self._result_success

    @property
    def cur_chunk_idx(self) -> int:
        """Index of the chunk playback is waiting on (for hold diagnostics)."""
        return self._next_chunk

    @property
    def progress(self) -> Tuple[int, int]:
        """(samples consumed, samples known) -- for logging playback progress."""
        return self._cursor, len(self._samples)

    def reset(self, traj_id: int, dt: float) -> None:
        self.clear()
        self._active_traj_id = traj_id
        self._active_dt = float(dt) if dt > 0.0 else self._default_dt

    def clear(self) -> None:
        self._chunks.clear()
        self._samples.clear()
        self._next_chunk = 0
        self._cursor = 0
        self._proj = 0
        self._progress_mode = False
        self._at_end = False
        self._active_traj_id = -1
        self._result_received = False
        self._result_success = False

    # ------------------------------------------------------------------
    # Incoming data
    # ------------------------------------------------------------------

    def add_chunk(self, chunk_index: int, chunk: Any) -> None:
        self._chunks[chunk_index] = chunk
        self._flatten_ready()

    def mark_result(self, success: bool) -> None:
        """Record that the action result for the active trajectory arrived."""
        self._result_received = True
        self._result_success = success

    def _flatten_ready(self) -> None:
        """Append every chunk that is now contiguous with what we already have."""
        while self._next_chunk in self._chunks:
            chunk = self._chunks.pop(self._next_chunk)
            for i in range(len(chunk.wheel_left)):
                self._samples.append(
                    PlaybackSample(
                        left=float(chunk.wheel_left[i]),
                        right=float(chunk.wheel_right[i]),
                        pose=(
                            float(chunk.pose_x[i]),
                            float(chunk.pose_y[i]),
                            float(chunk.pose_theta[i]),
                        ),
                        costates=self._costates_at(chunk, i),
                    )
                )
            self._next_chunk += 1

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        """True once the result is in and the plan has been played out.

        Both indexing modes need the result before finishing: without it, a
        cursor that has merely caught up with the chunks received so far would
        look identical to a finished trajectory.

        In progress mode "played out" means the ROBOT reached the end, not the
        cursor. The gate bounds the reference to `max_lead` samples ahead, so
        finishing when the cursor lands leaves up to that whole lead unexecuted
        -- about 0.5 m at cruise, which is most of the shortfall the gate was
        supposed to remove. Measured: gating alone moved the miss from 0.730 m
        to 0.725 m, because the lag was simply outstanding at termination
        instead of accumulating during the run.
        """
        if not self._result_received or self._chunks:
            return False
        if not self._progress_mode:
            return self._cursor >= len(self._samples)
        return self._proj >= len(self._samples) - 1

    def advance(self, actual_xy: Optional[Tuple[float, float]] = None):
        """Return the sample to command this tick, or None to hold.

        With `actual_xy` omitted the cursor is TIME-indexed: one sample per call.
        With a measured (x, y) it is PROGRESS-indexed: the nearest sample at or
        ahead of the cursor, within `max_skip`. See the module docstring for why
        the two differ and when each is right.

        Returns None when the sample we need has not arrived yet (the caller
        should hold the wheels).
        """
        if not self._samples:
            return None

        if actual_xy is None:
            if self._cursor >= len(self._samples):
                return None
            sample = self._samples[self._cursor]
            self._cursor += 1
        else:
            self._progress_mode = True
            # Clamp at the last sample rather than stopping there: the robot may
            # still be short of it, and the final feed-forward is what carries it
            # in. is_done() ends the run on the ROBOT arriving, and the node's
            # playback timeout bounds the wait if it never does.
            self._cursor = min(self._cursor, len(self._samples) - 1)
            sample = self._samples[self._cursor]
            self._proj = self._project(actual_xy)
            # Step only while the reference is within `max_lead` of the robot.
            # When it is not, re-serve the same sample: the robot is behind, and
            # running the plan on without it is what leaves it short of the goal.
            if self._cursor - self._proj <= self._max_lead:
                self._cursor += 1

        self._at_end = self._cursor >= len(self._samples)
        return sample

    def _project(self, actual_xy: Tuple[float, float]) -> int:
        """Index of the plan sample closest to `actual_xy`, searching forward
        only from the last projection, at most `max_skip` samples ahead.

        This tracks the ROBOT and is separate from the playback cursor, which
        legitimately leads it."""
        ax, ay = actual_xy
        stop = min(len(self._samples), self._proj + self._max_skip + 1)
        best_i = self._proj
        best_d = float("inf")
        for i in range(self._proj, stop):
            px, py, _ = self._samples[i].pose
            d = (px - ax) ** 2 + (py - ay) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    @staticmethod
    def _costates_at(chunk: Any, i: int) -> Optional[Tuple[float, ...]]:
        """Pull the per-tick costate vector from a chunk, or None if the feedback
        carries no (or short) costate arrays -- so a plain stand-in chunk in the
        tests, or a planner build without costates, degrades to None cleanly."""
        out = []
        for field in _COSTATE_FIELDS:
            arr = getattr(chunk, field, None)
            if arr is None or i >= len(arr):
                return None
            out.append(float(arr[i]))
        return tuple(out)

    # ------------------------------------------------------------------
    # Path geometry (debug visualization)
    # ------------------------------------------------------------------

    def build_polyline(self) -> List[Tuple[float, float]]:
        """Ordered (x, y) plan polyline from the current cursor onward."""
        return [(s.pose[0], s.pose[1]) for s in self._samples[self._cursor:]]
