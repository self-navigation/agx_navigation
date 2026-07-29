"""Reward and termination/success predicates. Pure (numpy only).

Reward (per step):
  + w_ontrack * max(0, 1-(d_target/corridor)^2)  dense closeness-to-target bonus
  - w_cross   * e_cross^2          cross-track tracking (dominant)
  - w_heading * e_heading^2        heading tracking
  + w_progress * progress_delta    along-track advance (anti-stall, capped by env)
  - w_effort  * sum(a_i^2)         magnitude of the (clipped, zero-centered) action
  - w_smooth  * sum((a_i - a_prev_i)^2)   action smoothness
  (- term_penalty on failure;  + success_bonus on success)

`progress_delta` (arc-length advanced along the nominal toward the goal this
step) is computed by the env and passed in.
"""

import numpy as np


def is_failure(cfg, err) -> bool:
    """Corridor breach / heading breach / non-finite state."""
    err = np.asarray(err, dtype=float)
    if not np.all(np.isfinite(err)):
        return True
    if abs(err[1]) > cfg.corridor_epsilon:
        return True
    if abs(err[2]) > cfg.max_heading_err:
        return True
    return False


def is_success(cfg, dist_to_goal_xy: float, heading_err_to_goal: float) -> bool:
    """Reached the nominal's endpoint within position and heading tolerance."""
    return (
        dist_to_goal_xy <= cfg.goal_tolerance_xy
        and abs(heading_err_to_goal) <= cfg.goal_tolerance_th
    )


def compute_reward(
    cfg,
    err,
    action,
    prev_action,
    progress_delta: float,
    failed: bool = False,
    succeeded: bool = False,
) -> float:
    """`action`/`prev_action` are the CLIPPED [-1,1] action vectors (0 ==
    identity) -- see coeff.clipped_action -- not the raw rad/s residual."""
    err = np.asarray(err, dtype=float)
    action = np.asarray(action, dtype=float)
    prev_action = (
        np.zeros_like(action) if prev_action is None else np.asarray(prev_action, dtype=float)
    )

    e_along = err[0]
    e_cross = err[1]
    e_heading = err[2]

    r = 0.0
    # Dense POSITIVE on-track reward: 1.0 when sitting on the current target point,
    # decaying quadratically to 0 at the corridor edge (and clamped there). Uses the
    # full distance to the target (along + cross) so keeping PACE with the moving
    # target is rewarded too, not just lateral centring. A corridor breach ends the
    # episode, so this is "reward for every step we stay near the target".
    d_to_target = float(np.hypot(e_along, e_cross))
    on_track = max(0.0, 1.0 - (d_to_target / cfg.corridor_epsilon) ** 2)
    r += cfg.w_ontrack * on_track
    r -= cfg.w_cross * e_cross ** 2
    r -= cfg.w_heading * e_heading ** 2
    r += cfg.w_progress * progress_delta
    r -= cfg.w_effort * float(np.sum(action ** 2))
    r -= cfg.w_smooth * float(np.sum((action - prev_action) ** 2))
    if failed:
        r -= cfg.term_penalty
    if succeeded:
        r += cfg.success_bonus
    return float(r)
