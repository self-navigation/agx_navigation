"""Recovery strategies for the PMP trajectory corrector.

When the corrector enters CORRECTING state it selects a strategy from a
prioritised list by calling can_handle() on each in order. The first strategy
that returns True generates the (v, omega) command for that tick.

Both can_handle() and compute_twist() receive:
  current -- the robot's current pose (x, y, theta) in the planning frame.
  path    -- the remaining planned trajectory as an ordered list of
             (x, y, theta) poses, starting from the current playback position.
             Strategies may inspect the full polyline and implement their own
             projection / look-ahead logic.

Adding a new strategy:
  1. Subclass RecoveryStrategy and implement can_handle() and compute_twist().
  2. Append an instance (with the shared RecoveryConfig) to the list passed to
     the corrector. The list is evaluated in order -- put more-specific
     strategies first.
"""

from enum import Enum, auto
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from agx_planning.runtime_corrector.geometry import (
    nearest_projection_on_path,
    walk_ahead_on_path,
)


class State(Enum):
    IDLE = auto()
    PLAYING = auto()
    CORRECTING = auto()


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


@dataclass
class RecoveryConfig:
    # Spatial threshold that must be reached to leave CORRECTING and resume
    # PLAYING.  Should be less than the entry corridor_epsilon to create a
    # hysteresis band that prevents rapid state toggling near the boundary.
    recovery_corridor_epsilon: float  # [m]
    # Heading alignment to the path tangent required to leave CORRECTING.
    recovery_angle_tolerance: float  # [rad]
    # How far ahead on the path to aim during look-ahead pursuit.
    look_ahead_distance: float  # [m]
    # Controller limits and gains.
    v_max: float  # [m/s]
    omega_max: float  # [rad/s]
    K_v: float  # forward speed gain   [m/s per m]
    K_bearing: float  # yaw-rate gain  [rad/s per rad of bearing error]
    K_theta: float  # yaw-rate gain    [rad/s per rad of heading error]


class RecoveryStrategy(ABC):
    def __init__(self, cfg: RecoveryConfig) -> None:
        self._cfg = cfg

    @abstractmethod
    def can_handle(
        self,
        current: tuple[float, float, float],
        path: list[tuple[float, float, float]],
    ) -> bool:
        """Return True if this strategy should handle the current situation.

        current -- robot pose (x, y, theta)
        path    -- remaining planned trajectory poses (x, y, theta), in order
        """

    @abstractmethod
    def compute_twist(
        self,
        current: tuple[float, float, float],
        path: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        """Return (v, omega) to reduce the deviation from the path."""


class RotateInPlaceStrategy(RecoveryStrategy):
    """Correct heading by rotating in place when already within the tight corridor.

    Fires when the robot is spatially close to the path but its heading deviates
    from the path tangent at the nearest projection. Rotating with v=0 avoids
    drifting further off the spatial track while correcting orientation.
    """

    def can_handle(
        self,
        current: tuple[float, float, float],
        path: list[tuple[float, float, float]],
    ) -> bool:
        if not path:
            return False
        rx, ry, rtheta = current
        proj_x, proj_y, proj_theta, _, _ = nearest_projection_on_path(rx, ry, path)
        if math.hypot(rx - proj_x, ry - proj_y) > self._cfg.recovery_corridor_epsilon:
            return False
        angle_err = abs(math.remainder(rtheta - proj_theta, 2 * math.pi))
        return angle_err > self._cfg.recovery_angle_tolerance

    def compute_twist(
        self,
        current: tuple[float, float, float],
        path: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        rx, ry, rtheta = current
        _, _, proj_theta, _, _ = nearest_projection_on_path(rx, ry, path)
        signed_err = math.remainder(proj_theta - rtheta, 2 * math.pi)
        omega = _clamp(self._cfg.K_theta * signed_err, self._cfg.omega_max)
        return 0.0, omega


class LookAheadPursuitStrategy(RecoveryStrategy):
    """Drive toward a look-ahead point on the path.

    Projects the robot onto the path polyline, advances along it by
    `look_ahead_distance`, then issues proportional bearing-pursuit commands
    toward that carrot point. Compared to aiming at the nearest sample, the
    look-ahead avoids large in-place turns when the robot is running roughly
    parallel to the path.
    """

    def can_handle(
        self,
        current: tuple[float, float, float],
        path: list[tuple[float, float, float]],
    ) -> bool:
        if not path:
            return False
        rx, ry, _ = current
        proj_x, proj_y, _, _, _ = nearest_projection_on_path(rx, ry, path)
        return (
            math.hypot(rx - proj_x, ry - proj_y) > self._cfg.recovery_corridor_epsilon
        )

    def compute_twist(
        self,
        current: tuple[float, float, float],
        path: list[tuple[float, float, float]],
    ) -> tuple[float, float]:
        rx, ry, rtheta = current
        _, _, _, seg_idx, t = nearest_projection_on_path(rx, ry, path)
        lx, ly, _ = walk_ahead_on_path(path, seg_idx, t, self._cfg.look_ahead_distance)
        bearing = math.atan2(ly - ry, lx - rx)
        bearing_err = math.remainder(bearing - rtheta, 2 * math.pi)
        dist = math.hypot(lx - rx, ly - ry)
        v = min(self._cfg.v_max, self._cfg.K_v * dist)
        omega = _clamp(self._cfg.K_bearing * bearing_err, self._cfg.omega_max)
        return v, omega


def default_strategies(cfg: RecoveryConfig) -> list[RecoveryStrategy]:
    """Return the default ordered strategy list.

    RotateInPlace is first (more specific: position already acceptable).
    LookAheadPursuit handles the general case of spatial deviation.
    """
    return [
        RotateInPlaceStrategy(cfg),
        LookAheadPursuitStrategy(cfg),
    ]
