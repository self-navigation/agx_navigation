"""Recovery strategies for the PMP trajectory corrector.

When the corrector enters CORRECTING state it selects a strategy from a
prioritised list by calling can_handle() on each in order. The first strategy
that returns True generates the (v, omega) command for that tick.

Both can_handle() and compute_twist() receive:
  current   -- the robot's current pose (x, y, theta) in the planning frame.
  candidates -- a list of (x, y, theta) poses drawn from the planned trajectory,
               sorted by a combined position+angle score (nearest first). Each
               strategy may inspect the full list and choose its own target.

The corrector retains the chunk_index / sample_index mapping so that once a
strategy has brought the robot within tolerance the snap-to-resume step can
reference the correct sample. Strategies are not aware of this mapping.

Adding a new strategy:
  1. Subclass RecoveryStrategy and implement can_handle() and compute_twist().
  2. Append an instance (with the shared RecoveryConfig) to the list passed to
     the corrector. The list is evaluated in order -- put more-specific
     strategies first.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class RecoveryConfig:
    # Trigger thresholds: error must exceed these to enter CORRECTING.
    pos_epsilon: float  # [m]
    angle_epsilon: float  # [rad]
    # Exit thresholds: error must fall below these to resume PLAYING.
    # Should be ≤ their trigger counterparts to create a hysteresis band.
    recovery_pos_tolerance: float  # [m]
    recovery_angle_tolerance: float  # [rad]
    # Controller limits and gains.
    v_max: float  # [m/s] maximum forward speed during recovery
    omega_max: float  # [rad/s] maximum yaw rate during recovery
    K_v: float  # forward speed gain  [m/s per m of distance]
    K_bearing: float  # yaw rate gain       [rad/s per rad of bearing error]
    K_theta: float  # yaw rate gain       [rad/s per rad of heading error]


class RecoveryStrategy(ABC):
    def __init__(self, cfg: RecoveryConfig) -> None:
        self._cfg = cfg

    @abstractmethod
    def can_handle(
        self,
        current: tuple[float, float, float],
        candidates: list[tuple[float, float, float]],
    ) -> bool:
        """Return True if this strategy is applicable given the current situation.

        current    -- robot pose (x, y, theta)
        candidates -- trajectory sample poses (x, y, theta), nearest-first
        """

    @abstractmethod
    def compute_twist(
        self,
        current: tuple[float, float, float],
        candidates: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        """Return (v, omega) to reduce the error toward a chosen candidate.

        The strategy is free to pick any candidate from the list as its target.
        """


class RotateInPlaceStrategy(RecoveryStrategy):
    """Correct yaw by rotating in place when position is already within tolerance.

    Picks the candidate with the smallest position error that is still within
    pos_epsilon, then rotates toward its heading. Using v=0 avoids carrying
    the robot further off the spatial track while correcting orientation.
    """

    def can_handle(
        self,
        current: tuple[float, float, float],
        candidates: list[tuple[float, float, float]],
    ) -> bool:
        # Fire when any candidate is close enough spatially that rotating in
        # place won't drift the robot further off-track, but the heading at
        # that candidate is still outside the recovery tolerance.
        rx, ry, rtheta = current
        for tx, ty, ttheta in candidates:
            if math.hypot(rx - tx, ry - ty) <= self._cfg.recovery_pos_tolerance:
                angle_err = abs(math.remainder(rtheta - ttheta, 2 * math.pi))
                return angle_err > self._cfg.recovery_angle_tolerance
        return False

    def compute_twist(
        self,
        current: tuple[float, float, float],
        candidates: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        rx, ry, rtheta = current
        # Among spatially acceptable candidates, pick the one with the smallest
        # position error as the heading target.
        best_tx, best_ty, best_ttheta = candidates[0]
        best_pos = math.hypot(rx - best_tx, ry - best_ty)
        for tx, ty, ttheta in candidates[1:]:
            pos = math.hypot(rx - tx, ry - ty)
            if pos <= self._cfg.pos_epsilon and pos < best_pos:
                best_pos = pos
                best_tx, best_ty, best_ttheta = tx, ty, ttheta
        signed_err = math.remainder(best_ttheta - rtheta, 2 * math.pi)
        omega = _clamp(self._cfg.K_theta * signed_err, self._cfg.omega_max)
        return 0.0, omega


class BearingPursuitStrategy(RecoveryStrategy):
    """Drive toward the spatially nearest candidate when position is off.

    Steers toward the nearest candidate by position (the one most likely to be
    reachable regardless of current heading), commanding forward speed
    proportional to distance and yaw rate proportional to bearing error.
    """

    def can_handle(
        self,
        current: tuple[float, float, float],
        candidates: list[tuple[float, float, float]],
    ) -> bool:
        # Fire whenever the nearest candidate is outside the recovery tolerance,
        # covering the full gap between that tolerance and the trigger epsilon.
        if not candidates:
            return False
        rx, ry, _ = current
        tx, ty, _ = min(candidates, key=lambda c: math.hypot(rx - c[0], ry - c[1]))
        return math.hypot(rx - tx, ry - ty) > self._cfg.recovery_pos_tolerance

    def compute_twist(
        self,
        current: tuple[float, float, float],
        candidates: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        rx, ry, rtheta = current
        # Pick the candidate nearest by position, ignoring angle.
        tx, ty, _ = min(candidates, key=lambda c: math.hypot(rx - c[0], ry - c[1]))
        bearing = math.atan2(ty - ry, tx - rx)
        bearing_err = math.remainder(bearing - rtheta, 2 * math.pi)
        dist = math.hypot(tx - rx, ty - ry)
        v = min(self._cfg.v_max, self._cfg.K_v * dist)
        omega = _clamp(self._cfg.K_bearing * bearing_err, self._cfg.omega_max)
        return v, omega


def default_strategies(cfg: RecoveryConfig) -> list[RecoveryStrategy]:
    """Return the default ordered strategy list.

    RotateInPlace is listed first: it is more specific (position already OK)
    and must be checked before BearingPursuit, which handles all remaining
    cases where the position error exceeds tolerance.
    """
    return [
        RotateInPlaceStrategy(cfg),
        BearingPursuitStrategy(cfg),
    ]
