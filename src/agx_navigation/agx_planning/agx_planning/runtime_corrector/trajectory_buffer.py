"""Timed playback buffer for the offline-mode PMP planner.

The planner streams its rollout as PlanToGoal feedback chunks much faster than
the robot plays them back, so the chunks are buffered and metered out at the
planned sample rate. A chunk is any object with the parallel sequences
wheel_left, wheel_right [rad/s], pose_x, pose_y [m] and pose_theta [rad], so
the class is testable without ROS. Call advance() once per PLAYING tick; when
it returns None, is_done() tells a finished trajectory from one waiting for its
next chunk.
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
    """Stores incoming trajectory chunks and serves timed wheel samples."""

    def __init__(self, default_dt: float) -> None:
        self._chunks: Dict[int, Any] = {}
        self._cur_chunk_idx: int = 0
        self._cur_sample_idx: int = 0
        self._active_traj_id: int = -1
        self._result_received: bool = False
        self._result_success: bool = False
        self._active_dt: float = default_dt

    # ------------------------------------------------------------------
    # Properties
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
        return self._cur_chunk_idx

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, traj_id: int, dt: float) -> None:
        """Discard the current buffer and start a new trajectory."""
        self._chunks = {}
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._active_traj_id = traj_id
        self._result_received = False
        self._result_success = False
        self._active_dt = dt

    def clear(self) -> None:
        """Full reset to the idle state (active_traj_id = -1)."""
        self._chunks.clear()
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._active_traj_id = -1
        self._result_received = False
        self._result_success = False

    # ------------------------------------------------------------------
    # Incoming data
    # ------------------------------------------------------------------

    def add_chunk(self, chunk_index: int, chunk: Any) -> None:
        self._chunks[chunk_index] = chunk

    def mark_result(self, success: bool) -> None:
        """Record that the action result for the active trajectory arrived."""
        self._result_received = True
        self._result_success = success

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        """True once the result is in and every buffered chunk is consumed."""
        return self._result_received and not self._chunks

    def advance(self) -> Optional[PlaybackSample]:
        """Consume and return the next wheel sample.

        Returns None when the chunk we need has not arrived yet (the caller
        should hold the wheels). Chunk exhaustion -- dropping a drained chunk
        and stepping to the next -- is handled internally.
        """
        cur = self._chunks.get(self._cur_chunk_idx)
        if cur is None:
            return None

        if self._cur_sample_idx < len(cur.wheel_left):
            return self._take(cur, self._cur_sample_idx, advance_to=self._cur_sample_idx + 1)

        # Current chunk exhausted -- drop it and step to the next.
        del self._chunks[self._cur_chunk_idx]
        self._cur_chunk_idx += 1
        self._cur_sample_idx = 0

        nxt = self._chunks.get(self._cur_chunk_idx)
        if nxt is None or len(nxt.wheel_left) == 0:
            return None
        return self._take(nxt, 0, advance_to=1)

    def _take(self, chunk: Any, i: int, advance_to: int) -> PlaybackSample:
        self._cur_sample_idx = advance_to
        return PlaybackSample(
            left=float(chunk.wheel_left[i]),
            right=float(chunk.wheel_right[i]),
            pose=(
                float(chunk.pose_x[i]),
                float(chunk.pose_y[i]),
                float(chunk.pose_theta[i]),
            ),
            costates=self._costates_at(chunk, i),
        )

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
        path: List[Tuple[float, float]] = []
        for chunk_idx in sorted(self._chunks):
            chunk = self._chunks[chunk_idx]
            start = self._cur_sample_idx if chunk_idx == self._cur_chunk_idx else 0
            for i in range(start, len(chunk.pose_x)):
                path.append((float(chunk.pose_x[i]), float(chunk.pose_y[i])))
        return path
