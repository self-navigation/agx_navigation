"""Pure-Python correction/recovery controller for the trajectory corrector.

No ROS2 imports. Operates on plain Python types so it can be unit-tested
outside a ROS2 workspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from agx_planning.runtime_corrector.config import CorrectorConfig
from agx_planning.runtime_corrector.geometry import (
    nearest_projection_on_path,
    walk_ahead_on_path,
)
from agx_planning.runtime_corrector.strategies import RecoveryStrategy


class ExitKind(Enum):
    ENDPOINT = auto()  # robot is within recovery_corridor_epsilon of path end
    OVERSHOT = auto()  # robot passed path end and action result is in
    RECOVERED = auto()  # robot is back within corridor and heading is aligned


@dataclass
class CorrectionResult:
    """Outcome of one correction step; tells the node what to do next."""

    correction_strat: str
    """Name of the recovery strategy that returned this result.
    """

    should_exit: bool
    """True when the node should call _finish_trajectory() and leave CORRECTING.
    Covers: near endpoint, overshoot with action result, and spatial+heading
    recovery."""

    waiting_for_chunks: bool
    """True when the robot overshot the buffered path end but the action result
    has not arrived yet. Node should hold (publish zero twist) without calling
    _finish_trajectory()."""

    snap_index: Optional[int]
    """Path-polyline index to resume playback at when transitioning back to
    PLAYING (recovery exit). None on endpoint/overshoot exits."""

    twist: Optional[Tuple[float, float]]
    """(v, omega) to publish this tick. None when should_exit=True,
    waiting_for_chunks=True, or snap_index is not None."""

    proj: Optional[Tuple[float, float]]
    """(proj_x, proj_y) nearest projection point for debug marker publishing.
    None if not computed (early exits)."""

    carrot: Optional[Tuple[float, float]]
    """(lx, ly) look-ahead carrot point for debug marker publishing.
    None if not computed (early exits)."""

    perp_dist: float
    """Perpendicular distance to path for _publish_metrics(). 0.0 on early
    exits where the projection was never computed."""

    exit_kind: Optional[ExitKind]
    """Set when should_exit=True or snap_index is not None. Used by the node
    to choose the appropriate log message."""


class CorrectionController:
    """Runs one correction step given a pose and path.

    All logic is pure Python / math — no ROS2 types involved.
    """

    def __init__(
        self,
        cfg: CorrectorConfig,
        strategies: List[RecoveryStrategy],
    ) -> None:
        self._cfg = cfg
        self._strategies = strategies

    def step(
        self,
        pose: Tuple[float, float, float],
        path: List[Tuple[float, float, float]],
        result_received: bool,
    ) -> CorrectionResult:
        """Run one correction step.

        pose            -- (x, y, theta) of the robot in the planning frame
        path            -- ordered (x, y, theta) list from _build_path_polyline()
        result_received -- True if the action result for the active trajectory
                           has been received (needed for overshoot exit logic)

        Returns a CorrectionResult describing what the node should do next.
        """
        rx, ry, rtheta = pose

        fx, fy, ftheta = path[-1]
        dx, dy = rx - fx, ry - fy

        # Only treat proximity to path[-1] as "reached the end" once the full
        # trajectory has arrived (result_received).  Without this gate a
        # turn-in-place trajectory's first chunk (v ~ 0, no XY progress) places
        # path[-1] within recovery_corridor_epsilon of the robot, triggering a
        # false ENDPOINT exit before the turn has even started.
        if result_received and math.hypot(dx, dy) < self._cfg.recovery_corridor_epsilon:
            return CorrectionResult(
                correction_strat="builtin:reached recovery corridor epsilon",
                should_exit=True,
                waiting_for_chunks=False,
                snap_index=None,
                twist=None,
                proj=None,
                carrot=None,
                perp_dist=0.0,
                exit_kind=ExitKind.ENDPOINT,
            )

        overshot = dx * math.cos(ftheta) + dy * math.sin(ftheta) > 0
        if overshot:
            if result_received:
                return CorrectionResult(
                    correction_strat="builtin:overshot trajectory end",
                    should_exit=True,
                    waiting_for_chunks=False,
                    snap_index=None,
                    twist=None,
                    proj=None,
                    carrot=None,
                    perp_dist=0.0,
                    exit_kind=ExitKind.OVERSHOT,
                )
            return CorrectionResult(
                correction_strat="builtin:overshot trajectory end",
                should_exit=False,
                waiting_for_chunks=True,
                snap_index=None,
                twist=None,
                proj=None,
                carrot=None,
                perp_dist=0.0,
                exit_kind=None,
            )

        proj_x, proj_y, proj_theta, seg_idx, t = nearest_projection_on_path(
            rx, ry, path
        )
        perp_dist = math.hypot(rx - proj_x, ry - proj_y)
        angle_err = abs(math.remainder(rtheta - proj_theta, 2 * math.pi))

        lx, ly, _ = walk_ahead_on_path(path, seg_idx, t, self._cfg.recovery_look_ahead)

        exiting = (
            perp_dist < self._cfg.recovery_corridor_epsilon
            and angle_err < self._cfg.recovery_angle_tolerance
        )

        if exiting:
            snap_idx = seg_idx if t <= 0.5 else min(seg_idx + 1, len(path) - 1)
            return CorrectionResult(
                correction_strat="builtin:recovered",
                should_exit=False,
                waiting_for_chunks=False,
                snap_index=snap_idx,
                twist=None,
                proj=(proj_x, proj_y),
                carrot=(lx, ly),
                perp_dist=perp_dist,
                exit_kind=ExitKind.RECOVERED,
            )

        for strategy in self._strategies:
            if strategy.can_handle(pose, path):
                v, omega = strategy.compute_twist(pose, path)
                return CorrectionResult(
                    correction_strat=strategy.__class__.__name__,
                    should_exit=False,
                    waiting_for_chunks=False,
                    snap_index=None,
                    twist=(v, omega),
                    proj=(proj_x, proj_y),
                    carrot=(lx, ly),
                    perp_dist=perp_dist,
                    exit_kind=None,
                )

        return CorrectionResult(
            correction_strat="builtin:no strat matches",
            should_exit=False,
            waiting_for_chunks=False,
            snap_index=None,
            twist=None,
            proj=(proj_x, proj_y),
            carrot=(lx, ly),
            perp_dist=perp_dist,
            exit_kind=None,
        )
