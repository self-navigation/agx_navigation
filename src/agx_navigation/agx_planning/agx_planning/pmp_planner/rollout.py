"""Pure-Python computation layer for the open-loop rollout logic.

No ROS 2 imports -- usable from scripts, notebooks, and tests without
pulling in rclpy.

Public API
----------
  parse_field_array   -- build a VectorFieldGrid from a raw float32 buffer
  goal_reached        -- check if a state is inside the goal-tolerance ball
  compute_diag_values -- extract PMP diagnostic scalars from a costate snapshot
  RolloutChunk        -- one committed BVP segment yielded by rollout_generator
  RolloutResult       -- terminal event; always the last item from rollout_generator
  rollout_generator   -- open-loop rollout as a Python generator
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from math import hypot, pi, tanh
from typing import Callable, Generator, Optional

import numpy as np

_logger = logging.getLogger(__name__)


from agx_planning.vector_field import VectorFieldGrid
from agx_planning.pmp_planner import PMPShootingSolver, PlannerConfig


@dataclass(frozen=True)
class RolloutChunk:
    """One committed BVP segment.

    twists     (N, 2) float64 -- chassis commands [v_cmd, omega_cmd] per sample
    poses      (N, 3) float64 -- planned state [px, py, theta] per sample,
                                 parallel to twists: twists[i] is applied at poses[i]
    costate_t0 (5,)  float64  -- costate row 0 [lx, ly, lth, lv, lom] captured
                                 immediately after the solve; None if unavailable.
                                 Snapshotted here so the consumer does not need to
                                 reach back into the solver after the next solve fires.
    """

    chunk_idx: int
    dt_sample: float
    twists: np.ndarray
    poses: np.ndarray
    costate_t0: Optional[np.ndarray]


@dataclass(frozen=True)
class RolloutResult:
    """Terminal event yielded as the very last item by rollout_generator.

    status  -- "success" | "failed" | "preempted" | "cancelled"
    message -- human-readable details safe to log or surface to a UI
    """

    status: str
    message: str


def parse_field_array(
    data: np.ndarray,
    cfg: PlannerConfig,
) -> Optional[VectorFieldGrid]:
    """Build a VectorFieldGrid from a flat float32 buffer.

    Expected layout:
      [h, w, origin_x, origin_y, resolution, travel_time(H*W), ...]

    Any channels after travel_time (grad_x, grad_y, grad_mag) are ignored --
    the grid recomputes the direction field internally from grad T.
    """
    if data.size < 5:
        return None

    h, w = int(data[0]), int(data[1])
    ox, oy, res = float(data[2]), float(data[3]), float(data[4])
    n = h * w

    if data.size < 5 + n:
        return None

    T = data[5 : 5 + n].reshape(h, w)
    grid = VectorFieldGrid()
    grid.update(T, ox, oy, res, field_eps=cfg.field_eps)
    return grid


def goal_reached(
    state: np.ndarray,
    goal: np.ndarray,
    cfg: PlannerConfig,
) -> bool:
    """True when the pose component of a state is within the goal-tolerance ball.

    state -- (N,) with N >= 3: [px, py, theta, ...]; extra elements are ignored
    goal  -- (3,): [gx, gy, gtheta]

    Accepting N >= 3 lets callers pass either the 3-D pose (online mode uses
    self._xi directly) or the full 5-D state (offline rollout loop).
    """
    d_xy = hypot(state[0] - goal[0], state[1] - goal[1])
    d_th = abs(((goal[2] - state[2] + pi) % (2.0 * pi)) - pi)
    return d_xy < cfg.goal_tolerance_xy and d_th < cfg.goal_tolerance_th


def compute_diag_values(
    costate_t0: np.ndarray,
    cfg: PlannerConfig,
) -> tuple[float, float, float]:
    """Compute (lam_th_0, lam_om_0, alpha_cmd_0) from a costate snapshot at t=0.

    costate_t0 -- (5,): [lx, ly, lth, lv, lom]

    alpha_cmd_0 is the tanh-saturated angular-acceleration command derived from
    the PMP optimality condition  alpha* = -lam_omega / gamma_alpha (sat alpha_max).
    Useful for diagnosing heading-rate planning decisions.

    The caller is responsible for checking whether a costate is available;
    this function does not return Optional so the control-flow at the call site
    stays flat.
    """
    lam_th_0 = float(costate_t0[2])
    lam_om_0 = float(costate_t0[4])
    alpha_cmd_0 = float(
        cfg.alpha_max * tanh(-lam_om_0 / (cfg.gamma_alpha * cfg.alpha_max))
    )
    return lam_th_0, lam_om_0, alpha_cmd_0


def rollout_generator(
    solver: PMPShootingSolver,
    cfg: PlannerConfig,
    x0: np.ndarray,
    goal: np.ndarray,
    stop_fn: Callable[[], Optional[str]] = lambda: None,
) -> Generator[RolloutChunk, None, RolloutResult]:
    """Generate trajectory chunks by repeated BVP solves from x0 to goal.

    Parameters
    ----------
    solver  : PMPShootingSolver
              The caller is responsible for calling reset_warm_start() before
              passing a solver that was previously used for another trajectory.
    cfg     : PlannerConfig
    x0      : (5,) initial state [px, py, theta, v, omega]
    goal    : (3,) target        [gx, gy, gtheta]
    stop_fn : called before each BVP solve; return None to continue, or a
              canonical stop-reason string to abort.
              Expected values: "preempted", "cancelled".
              The string is forwarded verbatim as RolloutResult.status.

    Yields
    ------
    RolloutChunk -- one committed segment; never empty

    Returns
    -------
    RolloutResult -- terminal event carrying status and message; retrieve via
                     GeneratorReturnCatcher (see planner.py)

    Termination paths
    -----------------
    (a) state at an iteration boundary is inside the goal-tolerance ball
    (b) a sample within a segment hits the ball (chunk is truncated to that point)
    (c) stagnation: no meaningful forward progress while already near the goal
    (d) sim-time cap (max_rollout_sim_time backstop)
    (e) stop_fn() returns a stop-reason string
    (f) BVP solve returns None (numerical failure)

    Standalone Python usage
    -----------------------
    >>> gen = GeneratorReturnCatcher(rollout_generator(solver, cfg, x0, goal))
    >>> for chunk in gen:
    ...     record(chunk.twists, chunk.poses)
    >>> result = gen.value   # RolloutResult
    >>> print(result.status, result.message)

    ROS 2 action-server usage
    -------------------------
    >>> def stop_fn():
    ...     if goal_handle.is_cancel_requested:
    ...         return "cancelled"
    ...     return "preempted" if stop_event.is_set() else None
    ...
    >>> gen = GeneratorReturnCatcher(rollout_generator(solver, cfg, x0, goal, stop_fn))
    >>> for chunk in gen:
    ...     publish_feedback(chunk)
    >>> return gen.value.status
    """
    dt_sample = 1.0 / cfg.control_rate
    seg_len_s = min(cfg.dt_segment, cfg.T_horizon)
    n_samples = max(1, int(round(seg_len_s / dt_sample)))

    state = x0.copy()
    sim_t = 0.0
    chunk_idx = 0

    # Stagnation guard: if the chassis stops making forward progress while
    # already near the goal (e.g. BVP emitting "stay here" from a quasi-fixed-
    # point just outside the tolerance ring), terminate rather than burning
    # through max_rollout_sim_time.
    progress_eps = max(0.5 * cfg.goal_tolerance_xy, 5e-3)  # m
    near_goal_thresh = 4.0 * cfg.goal_tolerance_xy
    stagnation_limit = 5
    prev_d_xy = float("inf")
    stagnation_count = 0

    _t_rollout_start = time.perf_counter()

    while sim_t < cfg.max_rollout_sim_time:

        # Check the stop signal BEFORE the solve so a signal arriving mid-rollout
        # skips the next BVP (~30 ms) rather than wasting it.
        stop_reason = stop_fn()
        if stop_reason is not None:
            return RolloutResult(status=stop_reason, message=stop_reason.capitalize())

        # Termination (a): state at the iteration boundary is in the goal ball.
        if goal_reached(state, goal, cfg):
            return RolloutResult(status="success", message="Goal reached")

        _t_solve = time.perf_counter()
        result = solver.sample_committed_segment(state, goal, dt_sample, n_samples)
        _solve_ms = (time.perf_counter() - _t_solve) * 1e3

        if result is None:
            return RolloutResult(
                status="failed",
                message=(
                    f"BVP solve failed at sim_t={sim_t:.2f}s "
                    f"(chunk={chunk_idx}): {solver._last_error}"
                ),
            )

        twists, poses, x_next = result

        # Snapshot the costate immediately after the solve. The next call to
        # sample_committed_segment will overwrite solver._last_costate, so the
        # consumer cannot safely read it after the next iteration starts.
        cs = solver._last_costate
        costate_t0 = cs[0].copy() if cs is not None else None

        # Termination (b): scan for the first sample inside the goal ball.
        # Truncate the chunk so the interpreter applies exactly the commands
        # needed to arrive; twists[i] is applied at poses[i], so the slice
        # [:hit_idx] delivers the chassis to poses[hit_idx] = goal.
        hit_idx = -1
        for i in range(poses.shape[0]):
            d_xy = hypot(poses[i, 0] - goal[0], poses[i, 1] - goal[1])
            d_th = abs(((goal[2] - poses[i, 2] + pi) % (2.0 * pi)) - pi)
            if d_xy < cfg.goal_tolerance_xy and d_th < cfg.goal_tolerance_th:
                hit_idx = i
                break

        if hit_idx == 0:
            # poses[0] == state (BVP pins the initial condition). This can
            # only happen if the (a) check above somehow missed it -- defensive.
            return RolloutResult(status="success", message="Goal reached")

        if hit_idx >= 1:
            _t_yield = time.perf_counter()
            yield RolloutChunk(
                chunk_idx=chunk_idx,
                dt_sample=dt_sample,
                twists=twists[:hit_idx],
                poses=poses[:hit_idx],
                costate_t0=costate_t0,
            )
            _consumer_ms = (time.perf_counter() - _t_yield) * 1e3
            _logger.info(
                "chunk %d (final, %d samples): solve=%.0fms  consumer=%.0fms"
                "  sim_t=%.2fs  wall=%.1fs",
                chunk_idx, hit_idx, _solve_ms, _consumer_ms,
                sim_t, time.perf_counter() - _t_rollout_start,
            )
            return RolloutResult(
                status="success",
                message=(
                    f"Goal reached intra-segment "
                    f"(chunk {chunk_idx}, sample {hit_idx})"
                ),
            )

        _t_yield = time.perf_counter()
        yield RolloutChunk(
            chunk_idx=chunk_idx,
            dt_sample=dt_sample,
            twists=twists,
            poses=poses,
            costate_t0=costate_t0,
        )
        _consumer_ms = (time.perf_counter() - _t_yield) * 1e3

        state = x_next
        sim_t += n_samples * dt_sample
        chunk_idx += 1

        # Termination (c): stagnation check (only meaningful when close to goal).
        new_d_xy = hypot(state[0] - goal[0], state[1] - goal[1])

        _logger.info(
            "chunk %d (%d samples): solve=%.0fms  consumer=%.0fms"
            "  sim_t=%.2fs  d_xy=%.3fm  stag=%d  wall=%.1fs",
            chunk_idx - 1, n_samples, _solve_ms, _consumer_ms,
            sim_t, new_d_xy, stagnation_count,
            time.perf_counter() - _t_rollout_start,
        )
        if new_d_xy < near_goal_thresh and (prev_d_xy - new_d_xy) < progress_eps:
            stagnation_count += 1
            if stagnation_count >= stagnation_limit:
                return RolloutResult(
                    status="failed",
                    message=(
                        f"Stagnated near goal "
                        f"(d_xy={new_d_xy:.3f} m, "
                        f"{stagnation_count} iters without progress)"
                    ),
                )
        else:
            stagnation_count = 0
        prev_d_xy = new_d_xy

    # Termination (d): sim-time cap.
    return RolloutResult(
        status="failed",
        message=f"Exceeded max_rollout_sim_time={cfg.max_rollout_sim_time} s",
    )
