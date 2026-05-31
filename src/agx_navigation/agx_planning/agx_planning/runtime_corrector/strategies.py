"""Recovery strategies for the PMP trajectory corrector.

When the corrector enters CORRECTING state it selects a strategy from a
prioritised list by calling can_handle() on each in order. The first strategy
that returns True generates the outcome for that tick.

Both can_handle() and compute_outcome() receive a StrategyContext containing:
  current         -- the robot's current pose (x, y, theta) in the planning frame
  path            -- the remaining planned trajectory as an ordered list of
                     (x, y, theta) poses, starting from the current playback position
  result_received -- True if the action result for the active trajectory has arrived

Outcome types:
  TwistOutcome   -- issue (v, omega); continue CORRECTING
  ExitOutcome    -- call _finish_trajectory(); leave CORRECTING
  ResumeOutcome  -- return to PLAYING at snap_index
  WaitOutcome    -- hold position (zero twist); do not exit yet

Adding a new strategy:
  1. Subclass RecoveryStrategy and implement can_handle() and compute_outcome().
  2. Append an instance (with the shared RecoveryConfig) to the list passed to
     the corrector. The list is evaluated in order — put more-specific
     strategies first.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from agx_planning.runtime_corrector.geometry import (
    nearest_projection_on_path,
    walk_ahead_on_path,
)


class State(Enum):
    IDLE = auto()
    PLAYING = auto()
    CORRECTING = auto()


class ExitKind(Enum):
    ENDPOINT = auto()   # robot is within recovery_corridor_epsilon of path end
    OVERSHOT = auto()   # robot passed path end and action result is in
    RECOVERED = auto()  # robot is back within corridor and heading is aligned


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
    # LookAheadPursuit only fires when the bearing to the carrot is within
    # this angle.  Beyond it the robot rotates in place instead of arcing,
    # preventing the spin-in-circles failure when the carrot is to the side.
    max_pursuit_bearing_err: float = math.pi / 2  # [rad]
    # Arc-length distance from the path end within which NearEndpointStrategy
    # takes priority to align heading before LookAheadPursuit drives forward.
    near_endpoint_distance: float = 0.5  # [m]


# ---------------------------------------------------------------------------
# StrategyContext — input to every strategy
# ---------------------------------------------------------------------------

@dataclass
class StrategyContext:
    current: tuple[float, float, float]            # (x, y, theta)
    path: list[tuple[float, float, float]]         # remaining trajectory
    result_received: bool                          # action result has arrived


# ---------------------------------------------------------------------------
# StrategyOutcome — tagged-union return type
# ---------------------------------------------------------------------------

@dataclass
class StrategyOutcome:
    """Base class: fields present on every outcome."""
    name: str                                      # logged as correction_strat
    log_msg: str = ""                              # extra note for log lines / debug markers
    proj: Optional[tuple[float, float]] = None
    carrot: Optional[tuple[float, float]] = None
    perp_dist: float = 0.0


@dataclass
class TwistOutcome(StrategyOutcome):
    """Issue a (v, omega) velocity command; continue CORRECTING."""
    twist: tuple[float, float] = (0.0, 0.0)


@dataclass
class ExitOutcome(StrategyOutcome):
    """Call _finish_trajectory() and leave CORRECTING."""
    exit_kind: ExitKind = ExitKind.ENDPOINT


@dataclass
class ResumeOutcome(StrategyOutcome):
    """Return to PLAYING, resuming playback at snap_index."""
    snap_index: int = 0
    exit_kind: ExitKind = ExitKind.RECOVERED


@dataclass
class WaitOutcome(StrategyOutcome):
    """Hold position (zero twist); do not exit yet."""


# ---------------------------------------------------------------------------
# RecoveryStrategy base class
# ---------------------------------------------------------------------------

class RecoveryStrategy(ABC):
    def __init__(self, cfg: RecoveryConfig) -> None:
        self._cfg = cfg

    @abstractmethod
    def can_handle(self, ctx: StrategyContext) -> bool:
        """Return True if this strategy should handle the current situation."""

    @abstractmethod
    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        """Return the outcome for this tick."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _is_overshot(ctx: StrategyContext) -> bool:
    """Return True when the robot has passed the end of the path.

    Requires the nearest projection to be clamped at the last segment's end
    AND the robot to lie in front of the endpoint along the final heading.
    The second condition prevents false-positives on U-turn paths where the
    robot starts facing away from the goal.
    """
    rx, ry, _ = ctx.current
    fx, fy, ftheta = ctx.path[-1]
    dx, dy = rx - fx, ry - fy
    _, _, _, seg_idx, t = nearest_projection_on_path(rx, ry, ctx.path)
    return (
        seg_idx == len(ctx.path) - 2
        and t > 1.0 - 1e-9
        and dx * math.cos(ftheta) + dy * math.sin(ftheta) > 0
    )


# ---------------------------------------------------------------------------
# Terminal strategies (replace the builtin heuristics)
# ---------------------------------------------------------------------------

class EndpointProximityStrategy(RecoveryStrategy):
    """Exit immediately when the robot is within recovery_corridor_epsilon of
    the path endpoint and the full trajectory has been received.

    The result_received gate prevents a false ENDPOINT exit on the first chunk
    of a turn-in-place trajectory where path[-1] briefly lands on the robot.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if not ctx.path or not ctx.result_received:
            return False
        rx, ry, _ = ctx.current
        fx, fy, _ = ctx.path[-1]
        return math.hypot(rx - fx, ry - fy) < self._cfg.recovery_corridor_epsilon

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        return ExitOutcome(
            name="EndpointProximityStrategy",
            exit_kind=ExitKind.ENDPOINT,
        )


class OvershotWaitStrategy(RecoveryStrategy):
    """Hold position when the robot has overshot but the action result has not
    arrived yet — the buffered path may still grow.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if len(ctx.path) < 2:
            return False
        return _is_overshot(ctx) and not ctx.result_received

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        return WaitOutcome(name="OvershotWaitStrategy")


class OvershotAlignStrategy(RecoveryStrategy):
    """Rotate in place to match the endpoint's final heading after overshooting.

    Fires when the robot has passed the path end, the full trajectory is
    in hand, but the heading still deviates from the endpoint's target theta.
    Once alignment is complete, OvershotExitStrategy takes over.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if len(ctx.path) < 2 or not ctx.result_received:
            return False
        if not _is_overshot(ctx):
            return False
        _, _, rtheta = ctx.current
        _, _, ftheta = ctx.path[-1]
        angle_err = abs(math.remainder(ftheta - rtheta, 2 * math.pi))
        return angle_err >= self._cfg.recovery_angle_tolerance

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        rx, ry, rtheta = ctx.current
        fx, fy, ftheta = ctx.path[-1]
        angle_err = math.remainder(ftheta - rtheta, 2 * math.pi)
        omega = _clamp(self._cfg.K_theta * angle_err, self._cfg.omega_max)
        return TwistOutcome(
            name="OvershotAlignStrategy",
            twist=(0.0, omega),
            proj=(fx, fy),
            perp_dist=math.hypot(rx - fx, ry - fy),
        )


class OvershotExitStrategy(RecoveryStrategy):
    """Exit after overshooting once the heading is aligned to the endpoint.

    This fires after OvershotAlignStrategy has finished its work, or
    immediately if the heading was already correct when the result arrived.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if len(ctx.path) < 2 or not ctx.result_received:
            return False
        if not _is_overshot(ctx):
            return False
        _, _, rtheta = ctx.current
        _, _, ftheta = ctx.path[-1]
        angle_err = abs(math.remainder(ftheta - rtheta, 2 * math.pi))
        return angle_err < self._cfg.recovery_angle_tolerance

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        return ExitOutcome(
            name="OvershotExitStrategy",
            exit_kind=ExitKind.OVERSHOT,
        )


class RecoveredStrategy(RecoveryStrategy):
    """Resume PLAYING when the robot is back within the spatial and heading
    corridor — the recovery is complete.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if not ctx.path:
            return False
        if _is_overshot(ctx):
            return False
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, proj_theta, _, _ = nearest_projection_on_path(rx, ry, ctx.path)
        perp_dist = math.hypot(rx - proj_x, ry - proj_y)
        angle_err = abs(math.remainder(rtheta - proj_theta, 2 * math.pi))
        return (
            perp_dist < self._cfg.recovery_corridor_epsilon
            and angle_err < self._cfg.recovery_angle_tolerance
        )

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, proj_theta, seg_idx, t = nearest_projection_on_path(
            rx, ry, ctx.path
        )
        lx, ly, _ = walk_ahead_on_path(
            ctx.path, seg_idx, t, self._cfg.look_ahead_distance
        )
        snap_idx = seg_idx if t <= 0.5 else min(seg_idx + 1, len(ctx.path) - 1)
        return ResumeOutcome(
            name="RecoveredStrategy",
            snap_index=snap_idx,
            exit_kind=ExitKind.RECOVERED,
            proj=(proj_x, proj_y),
            carrot=(lx, ly),
            perp_dist=math.hypot(rx - proj_x, ry - proj_y),
        )


# ---------------------------------------------------------------------------
# Motion strategies (spatial recovery)
# ---------------------------------------------------------------------------

class RotateInPlaceStrategy(RecoveryStrategy):
    """Correct heading by rotating in place when already within the tight corridor.

    Fires when the robot is spatially close to the path but its heading deviates
    from the path tangent at the nearest projection. Rotating with v=0 avoids
    drifting further off the spatial track while correcting orientation.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if not ctx.path:
            return False
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, proj_theta, _, _ = nearest_projection_on_path(rx, ry, ctx.path)
        if math.hypot(rx - proj_x, ry - proj_y) > self._cfg.recovery_corridor_epsilon:
            return False
        angle_err = abs(math.remainder(rtheta - proj_theta, 2 * math.pi))
        return angle_err > self._cfg.recovery_angle_tolerance

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, proj_theta, seg_idx, t = nearest_projection_on_path(
            rx, ry, ctx.path
        )
        lx, ly, _ = walk_ahead_on_path(
            ctx.path, seg_idx, t, self._cfg.look_ahead_distance
        )
        signed_err = math.remainder(proj_theta - rtheta, 2 * math.pi)
        omega = _clamp(self._cfg.K_theta * signed_err, self._cfg.omega_max)
        return TwistOutcome(
            name="RotateInPlaceStrategy",
            twist=(0.0, omega),
            proj=(proj_x, proj_y),
            carrot=(lx, ly),
            perp_dist=math.hypot(rx - proj_x, ry - proj_y),
        )


class LookAheadPursuitStrategy(RecoveryStrategy):
    """Drive toward a look-ahead point on the path.

    Projects the robot onto the path polyline, advances along it by
    `look_ahead_distance`, then issues proportional bearing-pursuit commands
    toward that carrot point. Compared to aiming at the nearest sample, the
    look-ahead avoids large in-place turns when the robot is running roughly
    parallel to the path.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if not ctx.path:
            return False
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, _, seg_idx, t = nearest_projection_on_path(rx, ry, ctx.path)
        if math.hypot(rx - proj_x, ry - proj_y) <= self._cfg.recovery_corridor_epsilon:
            return False
        lx, ly, _ = walk_ahead_on_path(ctx.path, seg_idx, t, self._cfg.look_ahead_distance)
        bearing_err = abs(
            math.remainder(math.atan2(ly - ry, lx - rx) - rtheta, 2 * math.pi)
        )
        return bearing_err <= self._cfg.max_pursuit_bearing_err

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, _, seg_idx, t = nearest_projection_on_path(rx, ry, ctx.path)
        lx, ly, _ = walk_ahead_on_path(ctx.path, seg_idx, t, self._cfg.look_ahead_distance)
        bearing = math.atan2(ly - ry, lx - rx)
        bearing_err = math.remainder(bearing - rtheta, 2 * math.pi)
        dist = math.hypot(lx - rx, ly - ry)
        v = min(self._cfg.v_max, self._cfg.K_v * dist)
        omega = _clamp(self._cfg.K_bearing * bearing_err, self._cfg.omega_max)
        return TwistOutcome(
            name="LookAheadPursuitStrategy",
            twist=(v, omega),
            proj=(proj_x, proj_y),
            carrot=(lx, ly),
            perp_dist=math.hypot(rx - proj_x, ry - proj_y),
        )


class NearEndpointStrategy(RecoveryStrategy):
    """Rotate in place toward the path endpoint when near the path end.

    When the robot arrives laterally offset near the goal, the look-ahead
    carrot collapses onto the endpoint and may be nearly orthogonal to the
    robot's heading.  LookAheadPursuit is restricted to bearings within
    max_pursuit_bearing_err, so it will not fire in that situation.  This
    strategy bridges the gap: it rotates the robot to face the endpoint
    position directly.  Once the bearing clears max_pursuit_bearing_err,
    LookAheadPursuit takes over and drives forward.
    """

    def can_handle(self, ctx: StrategyContext) -> bool:
        if not ctx.path:
            return False
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, _, seg_idx, t = nearest_projection_on_path(rx, ry, ctx.path)
        if math.hypot(rx - proj_x, ry - proj_y) <= self._cfg.recovery_corridor_epsilon:
            return False  # RotateInPlaceStrategy handles this
        fx, fy = ctx.path[-1][0], ctx.path[-1][1]
        if math.hypot(proj_x - fx, proj_y - fy) >= self._cfg.near_endpoint_distance:
            return False  # not near the end
        # Only fire when the carrot is too far to the side for LookAheadPursuit.
        # This makes the two strategies mutually exclusive: exactly one fires for
        # any given bearing, preventing oscillation.
        lx, ly, _ = walk_ahead_on_path(ctx.path, seg_idx, t, self._cfg.look_ahead_distance)
        bearing_err = abs(
            math.remainder(math.atan2(ly - ry, lx - rx) - rtheta, 2 * math.pi)
        )
        return bearing_err > self._cfg.max_pursuit_bearing_err

    def compute_outcome(self, ctx: StrategyContext) -> StrategyOutcome:
        rx, ry, rtheta = ctx.current
        proj_x, proj_y, _, seg_idx, t = nearest_projection_on_path(rx, ry, ctx.path)
        lx, ly, _ = walk_ahead_on_path(ctx.path, seg_idx, t, self._cfg.look_ahead_distance)
        fx, fy, _ = ctx.path[-1]
        # Rotate toward the endpoint position, not the path tangent.  The
        # tangent at an intermediate projection can point opposite to the goal
        # heading (e.g. approach-south on a south→north terminal segment), which
        # would turn the robot the wrong way.  Facing the endpoint lets
        # LookAheadPursuit take over cleanly once the bearing clears π/2.
        bearing = math.atan2(fy - ry, fx - rx)
        signed_err = math.remainder(bearing - rtheta, 2 * math.pi)
        omega = _clamp(self._cfg.K_theta * signed_err, self._cfg.omega_max)
        return TwistOutcome(
            name="NearEndpointStrategy",
            twist=(0.0, omega),
            proj=(proj_x, proj_y),
            carrot=(lx, ly),
            perp_dist=math.hypot(rx - proj_x, ry - proj_y),
        )


def default_strategies(cfg: RecoveryConfig) -> list[RecoveryStrategy]:
    """Return the default ordered strategy list.

    Terminal strategies run first so motion strategies never fire when
    the robot has already reached or overshot the goal.

    EndpointProximity  -- reached path end (result received, within epsilon)
    OvershotWait       -- past path end, waiting for full trajectory
    OvershotAlign      -- past path end, result in, heading needs fixing
    OvershotExit       -- past path end, result in, heading ok → exit
    Recovered          -- back in corridor (spatial + heading) → resume PLAYING
    RotateInPlace      -- spatially close, heading off → fix heading only
    NearEndpoint       -- near path end, laterally offset, carrot orthogonal
    LookAheadPursuit   -- general spatial deviation, carrot ahead
    """
    return [
        EndpointProximityStrategy(cfg),
        OvershotWaitStrategy(cfg),
        OvershotAlignStrategy(cfg),
        OvershotExitStrategy(cfg),
        RecoveredStrategy(cfg),
        RotateInPlaceStrategy(cfg),
        NearEndpointStrategy(cfg),
        LookAheadPursuitStrategy(cfg),
    ]
