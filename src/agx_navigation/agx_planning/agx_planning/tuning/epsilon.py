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
corrector actually optimizes:

    L = (x - x_ref)' Q (x - x_ref)  +  u' R u

with `x - x_ref` the (along, cross, heading) tracking error already computed by
`tvlqr.tracking_error`, and `u` the CORRECTION (dv, domega).

RELATION TO THE SOURCE'S FUNCTIONAL (1.7) -- TWO DELIBERATE DEVIATIONS
----------------------------------------------------------------------
The dissertation's functional is (1.7), §1.3.1, transcribed in
`docs/svcm-source.md`:

    J = 1/2 * int( q_x (x-x_T)^2 + q_y (y-y_T)^2 + q_th (th-th_T)^2
                   + r_v v^2 + r_omega omega^2 ) dt

Our Q/R has exactly that structure and even its variable names, but the form
above differs from it in two places, both on purpose and neither previously
written down (an earlier version of this docstring claimed to mirror the source
and cited a section number that does not contain the functional -- corrected
2026-08-15):

* **Reference, not terminal target.** (1.7) penalizes deviation from the
  terminal state `x_T`; we penalize deviation from the moving reference
  `x_ref(t)`. (1.7) is the PLANNER's functional -- it is what produces the
  trajectory. Scoring a corrector against `x_T` would reward cutting the corner
  off the plan it is supposed to be tracking.
* **Correction, not total control.** (1.7) charges the total `(v, omega)`; we
  charge only `(dv, domega)`. The nominal command is what the planner already
  paid for, so charging it again would score every corrector for the plan's
  cost instead of its own. This is also the decomposition the source itself
  gives on p. 52, `u_adm = u_J + u_bar` -- the existing control plus a
  correction -- so charging `u_bar` alone is scoring the object that
  decomposition names.

Both make this a *tracking* functional rather than the planning one. Say so when
reporting it; do not present a number from here as (1.7).

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


def step_cost(err: Sequence[float],
              correction: Sequence[float],
              dt: float,
              weights: CostWeights | None = None) -> float:
    """The running cost of ONE control step: `(e' Q e + u' R u) * dt`.

    This is the integrand of `cost_functional`, exposed on its own for two
    callers that cannot wait for the rollout to finish:

    * `EpsilonAccumulator`, which sums it online so a rollout yields `J` without
      anyone storing or re-pairing a trace;
    * **a reinforcement-learning reward**, which is `-step_cost(...)`. That
      identity is the point: a policy maximizing the undiscounted return of
      `-step_cost` is minimizing `J` minus its terminal block, so the training
      signal and the reported score become the SAME functional rather than two
      hand-weighted opinions that happen to correlate. `rl_corrector/reward.py`
      predates this and does not have that property -- it carries eight
      independently tuned weights and is not comparable to anything we report.

    Raises on non-finite input, for the reason `cost_functional` does.
    """
    w = weights or CostWeights()
    e = np.asarray(err, dtype=float)
    u = np.asarray(correction, dtype=float)
    if e.shape != (3,):
        raise ValueError(f"err must be (3,), got {e.shape}")
    if u.shape != (2,):
        raise ValueError(f"correction must be (2,), got {u.shape}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if not np.isfinite(e).all() or not np.isfinite(u).all():
        raise ValueError("non-finite value in err/correction -- refusing to score")
    return float((e @ w.q() @ e + u @ w.r() @ u) * dt)


class EpsilonAccumulator:
    """Sum the tracking functional online, one control step at a time.

    WHY THIS EXISTS RATHER THAN SCORING A TRACE AFTERWARDS. `J` needs the
    per-step track, so the first implementation wrote a CSV per rollout and
    paired it up offline. That pairing is a whole class of silent failure: the
    2026-08-14 traced soak produced files holding five rollouts each (tracing was
    armed by file and never disarmed) and a stride that aliased against the cycle
    length, and BOTH bugs scored to plausible numbers rather than raising. A
    rollout that accumulates its own `J` as it runs cannot be paired with the
    wrong plan, cannot be subsampled unevenly, and needs no disk at all.

    Traces remain worth writing for *diagnosis* (`trace_diff` needs them). They
    are no longer the route to a score.

    Usage:

        acc = EpsilonAccumulator(dt)
        for step in rollout:
            acc.push(err, correction)
        score = acc.finalize(final_err)
    """

    def __init__(self, dt: float, weights: CostWeights | None = None) -> None:
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = float(dt)
        self.weights = weights or CostWeights()
        self._track = 0.0
        self._ctrl = 0.0
        self._n = 0

    def push(self, err: Sequence[float], correction: Sequence[float]) -> float:
        """Accumulate one step; returns that step's cost (usable as `-reward`)."""
        e = np.asarray(err, dtype=float)
        u = np.asarray(correction, dtype=float)
        c = step_cost(e, u, self.dt, self.weights)  # validates
        self._track += float(e @ self.weights.q() @ e) * self.dt
        self._ctrl += float(u @ self.weights.r() @ u) * self.dt
        self._n += 1
        return c

    def finalize(self, final_err: float) -> EpsilonScore:
        """Close the rollout with its terminal miss and return the score.

        Refuses an EMPTY rollout. A rollout that produced no control steps did
        not happen, and returning `J = w_terminal * final_err^2` for it would
        report a small, plausible number for a run that never drove -- the exact
        shape of error this module's docstring exists to prevent.
        """
        if self._n == 0:
            raise ValueError("no steps accumulated -- refusing to score an empty "
                             "rollout; a rollout that never drove is a failure, "
                             "not a cheap one")
        if not np.isfinite(final_err):
            raise ValueError("non-finite final_err")
        j_term = float(self.weights.w_terminal * float(final_err) ** 2)
        return EpsilonScore(j_total=self._track + self._ctrl + j_term,
                            j_tracking=self._track, j_control=self._ctrl,
                            j_terminal=j_term, n_steps=self._n, dt=self.dt)


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
