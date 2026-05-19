"""Pure-Python trajectory chunk buffer for the corrector node.

Manages the dict of incoming feedback chunks, playback indices, and
path-polyline construction. No ROS2 imports — chunk objects are treated
as opaque carriers of .linear_x, .angular_z, .pose_x, .pose_y,
.pose_theta sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PlaybackSample:
    """One velocity command produced by the buffer for a single tick."""

    v: float
    omega: float
    samples_consumed: int
    """Value of the samples-consumed counter BEFORE this sample was taken.
    The node uses it to compute the due timestamp:
      due = start_time + Duration(nanoseconds=int(samples_consumed * dt * 1e9))
    """


class TrajectoryBuffer:
    """Stores incoming trajectory chunks and serves timed velocity samples.

    Call advance() on each PLAYING tick. When it returns None, check
    is_done() to distinguish "finished" from "waiting for next chunk".
    """

    def __init__(self, default_dt: float) -> None:
        self._chunks: Dict[int, Any] = {}
        self._cur_chunk_idx: int = 0
        self._cur_sample_idx: int = 0
        self._active_traj_id: int = -1
        self._result_received: bool = False
        self._result_success: bool = False
        self._samples_consumed: int = 0
        self._active_dt: float = default_dt

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_traj_id(self) -> int:
        return self._active_traj_id

    @property
    def result_received(self) -> bool:
        return self._result_received

    @property
    def result_success(self) -> bool:
        return self._result_success

    @property
    def active_dt(self) -> float:
        return self._active_dt

    @property
    def cur_chunk_idx(self) -> int:
        return self._cur_chunk_idx

    @property
    def cur_sample_idx(self) -> int:
        return self._cur_sample_idx

    @property
    def samples_consumed(self) -> int:
        return self._samples_consumed

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
        self._samples_consumed = 0
        self._active_dt = dt

    def clear(self) -> None:
        """Full reset to the idle state (traj_id = -1)."""
        self._chunks.clear()
        self._cur_chunk_idx = 0
        self._cur_sample_idx = 0
        self._active_traj_id = -1
        self._result_received = False
        self._result_success = False
        self._samples_consumed = 0

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
        """True when the result is in and all buffered chunks are consumed."""
        return self._result_received and not self._chunks

    def advance(self) -> Optional[PlaybackSample]:
        """Consume the next velocity sample.

        Returns a PlaybackSample on success, or None if the next chunk has
        not yet been received (caller should hold and call reset_timing()).
        Also returns None after the last sample of a chunk when the following
        chunk is not yet available.

        Chunk exhaustion (advancing cur_chunk_idx) is handled internally;
        the caller does not need to detect it.
        """
        cur = self._chunks.get(self._cur_chunk_idx)

        if cur is None:
            return None

        n = len(cur.linear_x)
        if self._cur_sample_idx < n:
            sample = PlaybackSample(
                v=float(cur.linear_x[self._cur_sample_idx]),
                omega=float(cur.angular_z[self._cur_sample_idx]),
                samples_consumed=self._samples_consumed,
            )
            self._samples_consumed += 1
            self._cur_sample_idx += 1
            return sample

        # Current chunk exhausted — drop it and move to the next.
        del self._chunks[self._cur_chunk_idx]
        self._cur_chunk_idx += 1
        self._cur_sample_idx = 0

        nxt = self._chunks.get(self._cur_chunk_idx)
        if nxt is None or len(nxt.linear_x) == 0:
            return None

        sample = PlaybackSample(
            v=float(nxt.linear_x[0]),
            omega=float(nxt.angular_z[0]),
            samples_consumed=self._samples_consumed,
        )
        self._samples_consumed += 1
        self._cur_sample_idx = 1
        return sample

    def reset_timing(self) -> None:
        """Reset the samples-consumed counter.

        Call this when playback is held (waiting for a chunk) so that
        timestamps are computed relative to the moment playback resumes
        rather than the trajectory's original start.
        """
        self._samples_consumed = 0

    def snap_to(self, chunk_idx: int, sample_idx: int) -> None:
        """Jump playback to a specific position (used after corridor recovery)."""
        self._cur_chunk_idx = chunk_idx
        self._cur_sample_idx = sample_idx
        self._samples_consumed = 0

    def last_endpoint(self) -> Optional[tuple[float, float]]:
        """Return (x, y) of the final pose in the last buffered chunk, or None."""
        if not self._chunks:
            return None
        last_chunk = self._chunks[max(self._chunks)]
        if not last_chunk.pose_x:
            return None
        return float(last_chunk.pose_x[-1]), float(last_chunk.pose_y[-1])

    # ------------------------------------------------------------------
    # Path geometry
    # ------------------------------------------------------------------

    def build_polyline(
        self,
    ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[int, int]]]:
        """Build an ordered path polyline from all buffered chunks.

        Returns:
          path    -- (x, y, theta) poses in trajectory order starting from
                     the current playback position.
          mapping -- parallel list of (chunk_idx, sample_idx) for each point,
                     used to snap playback position after recovery.
        """
        path: List[Tuple[float, float, float]] = []
        mapping: List[Tuple[int, int]] = []
        for chunk_idx in sorted(self._chunks):
            chunk = self._chunks[chunk_idx]
            start = self._cur_sample_idx if chunk_idx == self._cur_chunk_idx else 0
            for sample_idx in range(start, len(chunk.pose_x)):
                path.append(
                    (
                        float(chunk.pose_x[sample_idx]),
                        float(chunk.pose_y[sample_idx]),
                        float(chunk.pose_theta[sample_idx]),
                    )
                )
                mapping.append((chunk_idx, sample_idx))
        return path, mapping
