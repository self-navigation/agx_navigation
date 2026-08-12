"""Score a rollout in the advisor's currency: the cost-functional gap, not
cross-track error.

WHY THIS EXISTS
---------------
Every corrector comparison in this project scores `max|e_cross|` -- a tracking
error in metres. The SVCM framework it is meant to be implementing is stated
entirely in terms of

    J[u] <= J*[z] + epsilon

so `epsilon` IS a cost-functional gap, and we have never computed one. A result
reported in metres cannot be compared against a theory stated in `J`, and it is
the theory the thesis defends.

WHICH FUNCTIONAL, AND WHY NOT THE PLANNER'S
-------------------------------------------
The PMP planner's own `L` (see `pmp_planner/node.py`) is the right functional for
the PLANNING problem, but it is defined against the FM2 travel-time field `T(p)`
and the unit vector field `F_unit(p)`. Scoring an executed rollout with it means
reconstructing the field -- skfmm, the map, the goal -- which is heavy, is not
pure, and answers a question we are not asking. The planner's optimality is not
in doubt; the CORRECTOR's is.

So this module scores the *trajectory-following* functional, which is the one the
corrector actually optimizes and which the advisor's own §1.1 gives as the worked
example for exactly this task:

    L = (x - x_ref)' Q (x - x_ref)  +  u' R u

with `x - x_ref` the (along, cross, heading) tracking error already computed by
`tvlqr.tracking_error`, and `u` the CORRECTION (dv, domega) -- not the total
command. Correction rather than total command is the substantive choice: the
nominal command is what the planner already paid for, so charging it again would
score every corrector for the plan's cost instead of its own.

WHAT THE NUMBER MEANS -- READ THIS BEFORE QUOTING IT
----------------------------------------------------
For the tracking subproblem the *ideal* is zero: perfect tracking with no
correction costs nothing. So if the nominal trajectory were exactly achievable,
`J* = 0` and `J` would be `epsilon` itself.

Under slip the nominal is NOT exactly achievable, so `J* > 0` and unknown. That
makes the computed `J` an **upper bound on epsilon**, not epsilon:

    epsilon  =  J[u] - J*  <=  J[u]

This is the honest reading and the useful one -- an upper bound is exactly what
an epsilon-admissibility claim needs, since "we stayed within epsilon" is
established by bounding, not by knowing `J*`. Do not report `J` as epsilon.

Pure: no ROS, no Gazebo, no torch, no skfmm -- same rule as the rest of
`tuning/`. Takes arrays, returns numbers.
"""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class CostWeights:
    """Q and R for the tracking functional.

    Defaults deliberately MIRROR `TVLQRConfig`'s tuned weights, so the score is
    the functional the corrector is optimizing rather than a third opinion. They
    are duplicated rather than imported for the usual reason in this package --
    keeping the pure modules free of the control stack -- so **keep them in sync
    manually**, exactly like `RLCorrectorConfig`'s kinematics constants.
    """

    q_along: float = 1.0
    q_cross: float = 0.2762521839107533
    q_heading: float = 5.0
    r_v: float = 1.0
    r_omega: float = 2.6183452282612643
    # Terminal miss, weighted like the planner's terminal block: a rollout that
    # tracks well and stops short is not epsilon-admissible, and an integral-only
    # score would call it excellent. `final_err` has failed independently of
    # `max_cross` before, which is why both are kept everywhere else too.
    w_terminal: float = 10.0

    def q(self) -> np.ndarray:
        return np.diag([self.q_along, self.q_cross, self.q_heading]).astype(float)

    def r(self) -> np.ndarray:
        return np.diag([self.r_v, self.r_omega]).astype(float)


@dataclass
class EpsilonScore:
    """The bound and its parts. Parts are kept because a single scalar cannot
    say WHY a rollout was expensive, and the two failure modes (wandering vs.
    fighting the plan) are distinguishable only in the split."""

    j_total: float
    j_tracking: float
    j_control: float
    j_terminal: float
    n_steps: int
    dt: float

    def as_dict(self) -> dict:
        return {"j_total": self.j_total, "j_tracking": self.j_tracking,
                "j_control": self.j_control, "j_terminal": self.j_terminal,
                "n_steps": self.n_steps, "dt": self.dt}


def cost_functional(errors: Sequence[Sequence[float]],
                    corrections: Sequence[Sequence[float]],
                    dt: float,
                    final_err: float,
                    weights: CostWeights | None = None) -> EpsilonScore:
    """Integrate the tracking functional over one rollout.

    `errors[k]` is `(e_along, e_cross, e_heading)` at step k, as produced by
    `tvlqr.tracking_error`. `corrections[k]` is `(dv, domega)` -- the corrector's
    OUTPUT, i.e. the applied correction, not the total command.

    Returns an upper bound on epsilon (see the module docstring: `J* > 0` under
    slip and is unknown, so `epsilon <= J`).

    Raises on non-finite input rather than propagating NaN. That is deliberate
    and has a history here: `nan` is expected in recorded cross-track columns
    before the planner publishes, `float("nan")` parses without raising, and one
    NaN turns any sum into NaN -- silently reporting a perfect-looking or
    absurd score. A rollout that cannot be scored must fail loudly.
    """
    w = weights or CostWeights()
    e = np.asarray(errors, dtype=float)
    u = np.asarray(corrections, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3:
        raise ValueError(f"errors must be (n, 3), got {e.shape}")
    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"corrections must be (n, 2), got {u.shape}")
    if len(e) != len(u):
        raise ValueError(f"length mismatch: {len(e)} errors vs {len(u)} corrections")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not np.isfinite(e).all() or not np.isfinite(u).all():
        raise ValueError("non-finite value in errors/corrections -- refusing to "
                         "score; a NaN here would silently poison the sum")
    if not np.isfinite(final_err):
        raise ValueError("non-finite final_err")

    # Sum of quadratic forms, integrated rectangularly. Rectangular rather than
    # trapezoidal on purpose: the samples ARE the control steps, so the cost is
    # genuinely piecewise-constant over each step rather than sampled from a
    # continuous signal.
    j_track = float(np.einsum("ij,jk,ik->", e, w.q(), e) * dt)
    j_ctrl = float(np.einsum("ij,jk,ik->", u, w.r(), u) * dt)
    j_term = float(w.w_terminal * final_err ** 2)
    return EpsilonScore(j_total=j_track + j_ctrl + j_term,
                        j_tracking=j_track, j_control=j_ctrl,
                        j_terminal=j_term, n_steps=len(e), dt=dt)


def compare(scores_a: Sequence[EpsilonScore],
            scores_b: Sequence[EpsilonScore]) -> dict:
    """Mean J and its spread for two arms.

    Returns the means, the sds, and `delta` = mean(b) - mean(a), so a positive
    delta means arm A is cheaper. No significance test: with the mode structure
    documented in CLAUDE.md the samples are frequently BIMODAL, and a t-test on a
    bimodal sample reports a confident interval around a value neither mode takes.
    Report the distributions.
    """
    if not scores_a or not scores_b:
        raise ValueError("both arms need at least one score")
    a = np.array([s.j_total for s in scores_a], dtype=float)
    b = np.array([s.j_total for s in scores_b], dtype=float)
    return {"n_a": len(a), "n_b": len(b),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "sd_a": float(a.std()), "sd_b": float(b.std()),
            "min_a": float(a.min()), "max_a": float(a.max()),
            "min_b": float(b.min()), "max_b": float(b.max()),
            "delta": float(b.mean() - a.mean())}
