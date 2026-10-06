"""Playback buffer for the offline-mode PMP planner.

The planner streams its rollout as PlanToGoal feedback chunks much faster than
the robot plays them back, so the chunks are buffered and metered out. A chunk
is any object with the parallel sequences wheel_left, wheel_right [rad/s],
pose_x, pose_y [m] and pose_theta [rad], so the class is testable without ROS.
Call advance() once per PLAYING tick; when it returns None, is_done() tells a
finished trajectory from one waiting for its next chunk.

advance() with no pose consumes one sample per tick, so playback is indexed by
time. advance(actual_xy) still steps one sample per tick but never lets the
cursor run more than `max_lead` samples ahead of the robot's projection onto
the plan, so a robot that falls behind is waited for instead of running out of
samples. Indexing by the projection alone would stall at the start, where the
plan commands zero speed. The projection search is windowed (`max_skip`) and
never moves backwards, so a plan that loops back near itself cannot make the
cursor jump.
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
        # Supervisor re-join (#41): samples served BEFORE the plan, after which
        # the cursor resumes at a plan index. See splice().
        self._overlay: List[PlaybackSample] = []
        self._resume: Optional[int] = None

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

    @property
    def cursor(self) -> int:
        """Plan index of the NEXT sample advance() serves (overlay aside)."""
        return self._cursor

    @property
    def overlay_active(self) -> bool:
        return bool(self._overlay)

    @property
    def n_samples(self) -> int:
        return len(self._samples)

    def plan_poses(self) -> List[Tuple[float, float, float]]:
        return [s.pose for s in self._samples]

    def plan_samples(self) -> List[PlaybackSample]:
        return list(self._samples)

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
        self._overlay = []
        self._resume = None

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
        if not self._result_received or self._chunks or self._overlay:
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

        if self._overlay:
            sample = self._overlay.pop(0)
            if not self._overlay:
                self._cursor = self._resume
                self._proj = min(self._resume, len(self._samples) - 1)
                self._resume = None
            self._at_end = False
            return sample

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

    def retime(self, index: int) -> None:
        """Move the time cursor to plan ``index`` (supervisor waypoint, #41).

        The supervisor calls this when the robot reaches a waypoint, with the
        robot's progress index there: the reference waits for a robot that is
        behind (cursor moves back) and catches up with one that is ahead. The
        per-segment clock is the supervisor's; this only places the cursor.
        Ignored while a re-join overlay is playing -- the overlay owns timing
        until it ends at its own resume index.
        """
        if self._overlay or not self._samples:
            return
        self._cursor = int(min(max(index, 0), len(self._samples) - 1))

    def splice(self, samples: List[PlaybackSample], resume_index: int) -> None:
        """Serve ``samples`` next, then resume the plan at ``resume_index``.

        ``samples`` is a re-join (pmp_planner/rejoin_service.RejoinSolver),
        whose last sample is the plan's state at ``resume_index - 1``. The
        overlay replaces the plan from the current cursor, so the corrector
        tracks the re-join's poses as its reference while it plays. A splice
        replaces any overlay still playing.
        """
        if not samples:
            return
        self._overlay = list(samples)
        self._resume = int(min(max(resume_index, 0), len(self._samples)))

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
