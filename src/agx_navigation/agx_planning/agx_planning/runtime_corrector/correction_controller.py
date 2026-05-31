"""Pure-Python correction/recovery controller for the trajectory corrector.

No ROS2 imports. Operates on plain Python types so it can be unit-tested
outside a ROS2 workspace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from agx_planning.runtime_corrector.strategies import (
    ExitKind,
    ExitOutcome,
    RecoveryStrategy,
    ResumeOutcome,
    StrategyContext,
    StrategyOutcome,
    TwistOutcome,
    WaitOutcome,
)
from agx_planning.runtime_corrector.config import CorrectorConfig


@dataclass
class CorrectionResult:
    """Outcome of one correction step; tells the node what to do next."""

    correction_strat: str
    """Name of the recovery strategy that returned this result."""

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


def _outcome_to_result(outcome: StrategyOutcome) -> CorrectionResult:
    """Convert a StrategyOutcome to a CorrectionResult for the node."""
    name = outcome.name
    proj = outcome.proj
    carrot = outcome.carrot
    perp_dist = outcome.perp_dist
    if isinstance(outcome, ExitOutcome):
        return CorrectionResult(
            correction_strat=name,
            should_exit=True,
            waiting_for_chunks=False,
            snap_index=None,
            twist=None,
            proj=proj,
            carrot=carrot,
            perp_dist=perp_dist,
            exit_kind=outcome.exit_kind,
        )
    if isinstance(outcome, ResumeOutcome):
        return CorrectionResult(
            correction_strat=name,
            should_exit=False,
            waiting_for_chunks=False,
            snap_index=outcome.snap_index,
            twist=None,
            proj=proj,
            carrot=carrot,
            perp_dist=perp_dist,
            exit_kind=outcome.exit_kind,
        )
    if isinstance(outcome, WaitOutcome):
        return CorrectionResult(
            correction_strat=name,
            should_exit=False,
            waiting_for_chunks=True,
            snap_index=None,
            twist=None,
            proj=proj,
            carrot=carrot,
            perp_dist=perp_dist,
            exit_kind=None,
        )
    if isinstance(outcome, TwistOutcome):
        return CorrectionResult(
            correction_strat=name,
            should_exit=False,
            waiting_for_chunks=False,
            snap_index=None,
            twist=outcome.twist,
            proj=proj,
            carrot=carrot,
            perp_dist=perp_dist,
            exit_kind=None,
        )
    # Unknown subtype: hold position safely
    return CorrectionResult(
        correction_strat=name,
        should_exit=False,
        waiting_for_chunks=False,
        snap_index=None,
        twist=None,
        proj=proj,
        carrot=carrot,
        perp_dist=perp_dist,
        exit_kind=None,
    )


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
        ctx = StrategyContext(current=pose, path=path, result_received=result_received)
        for strategy in self._strategies:
            if strategy.can_handle(ctx):
                return _outcome_to_result(strategy.compute_outcome(ctx))
        return CorrectionResult(
            correction_strat="no strategy matched",
            should_exit=False,
            waiting_for_chunks=False,
            snap_index=None,
            twist=None,
            proj=None,
            carrot=None,
            perp_dist=0.0,
            exit_kind=None,
        )
