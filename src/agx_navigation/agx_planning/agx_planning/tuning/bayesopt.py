"""Bayesian optimization for a NOISY, expensive objective.

WHY THIS EXISTS (and why Nelder-Mead had to go)
-----------------------------------------------
The tuning objective is stochastic: sd 0.093 m single-sample on the 7-shape eval
set, 0.055 m averaged over 3 repeats. Nelder-Mead has no model of that, and it
fails in two specific ways, both observed in the 2026-08-03 run:

  1. IT CANNOT CONVERGE. A simplex shrinks toward the best vertex; when the
     "best" vertex is decided by noise, it shrinks onto a point, re-samples it
     forever, and never moves. That run stopped moving at eval 49 and then
     re-evaluated ONE gain pair 71 times.
  2. IT REPORTS THE MINIMUM OBSERVED DRAW. Over 71 repeats spanning
     0.9412-1.3052 m it reported 0.9412 -- the lucky tail, not the value. The
     claimed 0.199 m improvement was smaller than the 0.364 m spread it was
     selected from. This is the winner's curse, and it is a property of the
     REPORTING rule, not of the search.

A Gaussian-process surrogate fixes both structurally rather than by averaging
harder. The GP has an explicit noise term, so repeated draws at one point make
it MORE certain instead of jittering the search; and the recommendation is the
minimizer of the POSTERIOR MEAN, which is a smoothed estimate informed by every
nearby observation, not a single lucky sample. Averaging n repeats and taking
the min of those is still winner's curse at 1/sqrt(n) the amplitude; taking the
posterior mean is not.

It also extends to the full Q/R diagonal (q_along, q_heading, r_v) without a
redesign -- GPs are routine up to ~20 dimensions, and the evaluation budget, not
the dimension, is the binding constraint here.

DESIGN NOTES
------------
* Inputs are normalized to the unit cube by `bounds`, so one isotropic
  lengthscale is meaningful and the caller can keep searching in log10 gains.
* Matern 5/2 kernel: the standard BO default. The squared-exponential assumes a
  function smoother than any real physical response, and over-smooths ridges.
* Hyperparameters (lengthscale, signal amplitude, noise) are fitted by grid
  search on the exact marginal likelihood. A grid rather than a gradient method
  because n is small (tens of points), the likelihood is multimodal, and a grid
  cannot silently diverge -- the same reasoning as everything else in this
  directory being boring on purpose.
* The noise term can be PINNED (`noise=`) when the measurement sd is known
  independently, which it is here: pass sd/mean(y) from the repeats table. A
  pinned noise floor stops the GP explaining pure measurement scatter as real
  structure, which is the classic way a surrogate over-fits a noisy objective.
* Expected Improvement over the best posterior MEAN (not the best observed y).
  With a noisy objective the best observed y is an outlier by construction, and
  EI against it under-explores: it thinks it already has something excellent.

Pure module: numpy only. No ROS, no Gazebo, no scipy, no torch -- same rule as
simplex.py / objective.py / cache.py, and for the same reason: the search must be
testable without a simulator, because a search that quietly maximizes looks
exactly like one that minimizes from the outside.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Gaussian process
# ---------------------------------------------------------------------------

def _matern52(xa: np.ndarray, xb: np.ndarray, lengthscale: float) -> np.ndarray:
    """Matern 5/2 correlation matrix between two sets of points (unit amplitude)."""
    d = xa[:, None, :] - xb[None, :, :]
    r = np.sqrt(np.maximum((d * d).sum(axis=-1), 0.0)) / max(lengthscale, 1e-9)
    s5 = math.sqrt(5.0)
    return (1.0 + s5 * r + (5.0 / 3.0) * r * r) * np.exp(-s5 * r)


@dataclass
class GP:
    """Zero-mean GP on standardized y, with a Matern 5/2 kernel and white noise.

    `noise` is the standard deviation of the observation noise IN STANDARDIZED
    UNITS, i.e. relative to the spread of the observed values.
    """

    lengthscale: float
    amplitude: float
    noise: float
    x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _y_mean: float = 0.0
    _y_std: float = 1.0
    _chol: Optional[np.ndarray] = None
    _alpha: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GP":
        self.x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float).ravel()
        self._y_mean = float(y.mean())
        # A constant-y set (every draw identical) has zero spread; standardizing
        # by it would divide by zero. Fall back to 1.0 -- the GP then models a
        # flat function, which is exactly right.
        std = float(y.std())
        self._y_std = std if std > 1e-12 else 1.0
        self.y = (y - self._y_mean) / self._y_std
        k = self.amplitude ** 2 * _matern52(self.x, self.x, self.lengthscale)
        k[np.diag_indices_from(k)] += self.noise ** 2 + 1e-8
        self._chol = np.linalg.cholesky(k)
        self._alpha = np.linalg.solve(
            self._chol.T, np.linalg.solve(self._chol, self.y))
        return self

    def predict(self, xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Posterior (mean, sd) at `xs`, in the ORIGINAL units of y."""
        if self._chol is None:
            raise RuntimeError("GP.predict before fit")
        xs = np.atleast_2d(np.asarray(xs, dtype=float))
        ks = self.amplitude ** 2 * _matern52(self.x, xs, self.lengthscale)
        mean = ks.T @ self._alpha
        v = np.linalg.solve(self._chol, ks)
        var = self.amplitude ** 2 - (v * v).sum(axis=0)
        var = np.maximum(var, 0.0)
        return (mean * self._y_std + self._y_mean, np.sqrt(var) * self._y_std)

    def log_marginal_likelihood(self) -> float:
        if self._chol is None:
            raise RuntimeError("GP.log_marginal_likelihood before fit")
        n = len(self.y)
        return float(-0.5 * self.y @ self._alpha
                     - np.log(np.diag(self._chol)).sum()
                     - 0.5 * n * math.log(2.0 * math.pi))


_LENGTHSCALES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 3.0)
_AMPLITUDES = (0.3, 0.5, 0.75, 1.0, 1.5, 2.5)
_NOISES = (0.01, 0.03, 0.07, 0.15, 0.3, 0.5)


def fit_gp(x, y, noise: Optional[float] = None) -> GP:
    """Fit a GP by grid search on the marginal likelihood.

    `noise` pins the observation-noise sd in units of the spread of y; None fits
    it from the data. Pin it when it is independently known -- an unpinned GP
    with few points routinely explains measurement scatter as structure, which
    produces a confident surrogate of pure noise.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError(f"x has {len(x)} rows but y has {len(y)} entries")
    if len(y) == 0:
        raise ValueError("cannot fit a GP to no observations")
    noises = (float(noise),) if noise is not None else _NOISES
    best, best_ll = None, -math.inf
    for ls in _LENGTHSCALES:
        for amp in _AMPLITUDES:
            for nz in noises:
                gp = GP(lengthscale=ls, amplitude=amp, noise=max(nz, 1e-4))
                try:
                    ll = gp.fit(x, y).log_marginal_likelihood()
                except np.linalg.LinAlgError:
                    continue
                if ll > best_ll:
                    best, best_ll = gp, ll
    if best is None:
        raise RuntimeError("no GP hyperparameter combination could be fitted")
    return best


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------

def _norm_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def expected_improvement(gp: GP, xs: np.ndarray, incumbent: float,
                         xi: float = 0.0) -> np.ndarray:
    """EI for MINIMIZATION. `incumbent` should be the best posterior MEAN.

    Using the best observed y instead would systematically under-explore on a
    noisy objective: the luckiest draw looks better than anything the surrogate
    believes is achievable, so EI collapses toward zero everywhere.
    """
    mean, sd = gp.predict(xs)
    sd = np.maximum(sd, 1e-12)
    z = (incumbent - xi - mean) / sd
    return (incumbent - xi - mean) * _norm_cdf(z) + sd * _norm_pdf(z)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@dataclass
class BOResult:
    x: List[float]              # recommendation: argmin of the POSTERIOR MEAN
    fx: float                   # posterior mean there (NOT a measured sample)
    fx_observed: float          # best single measurement seen, for reference only
    x_observed: List[float]
    n_evals: int
    xs: List[List[float]]
    ys: List[float]
    converged: bool
    message: str


def _to_unit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (x - lo) / np.maximum(hi - lo, 1e-12)


def _from_unit(u: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + u * (hi - lo)


def _sobol_ish(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Latin-hypercube initial design: better coverage than uniform at small n.

    Not Sobol -- a proper low-discrepancy sequence would mean a scipy dependency
    for a gain that vanishes past the first few points.
    """
    out = np.empty((n, dim))
    for d in range(dim):
        out[:, d] = (rng.permutation(n) + rng.random(n)) / n
    return out


def minimize(f: Callable[[Sequence[float]], float],
             bounds: Sequence[Tuple[float, float]],
             max_evals: int = 60,
             n_init: int = 8,
             noise: Optional[float] = None,
             seed: int = 0,
             n_candidates: int = 4000,
             x0: Optional[Sequence[float]] = None,
             callback: Optional[Callable[[int, List[float], float], None]] = None,
             ) -> BOResult:
    """Minimize a noisy `f` over a box, by GP + Expected Improvement.

    Returns the minimizer of the POSTERIOR MEAN over the evaluated points --
    deliberately not the best observed sample. See the module docstring.

    `x0` seeds the initial design with a specific point (the incumbent gains, so
    the search always has the current default measured under the same estimator
    as everything it is compared against).
    """
    bnds = np.asarray(bounds, dtype=float)
    if bnds.ndim != 2 or bnds.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (lo, hi) pairs")
    if np.any(bnds[:, 0] >= bnds[:, 1]):
        raise ValueError(f"every bound needs lo < hi, got {bounds}")
    dim = len(bnds)
    n_init = max(2, min(int(n_init), int(max_evals)))
    rng = np.random.default_rng(seed)

    xs: List[np.ndarray] = []
    ys: List[float] = []

    def observe(u: np.ndarray) -> float:
        x = _from_unit(u, bnds)
        val = float(f(list(x)))
        # A non-finite objective is "this candidate is invalid" (a failed
        # rollout). Feeding inf to the GP would destroy the fit, so map it to a
        # value worse than anything seen -- bad enough never to be recommended,
        # finite enough to keep the surrogate conditioned.
        if not math.isfinite(val):
            val = (max(ys) + 10.0 * (max(1e-6, np.ptp(ys))) if ys else 1e3)
        xs.append(u)
        ys.append(val)
        if callback is not None:
            callback(len(ys), list(x), val)
        return val

    design = _sobol_ish(n_init, dim, rng)
    if x0 is not None:
        design[0] = np.clip(_to_unit(np.asarray(x0, dtype=float), bnds), 0.0, 1.0)
    for u in design:
        if len(ys) >= max_evals:
            break
        observe(u)

    converged, message = False, f"evaluation budget ({max_evals}) exhausted"
    while len(ys) < max_evals:
        xa = np.array(xs)
        gp = fit_gp(xa, np.array(ys), noise=noise)
        # Incumbent = best posterior mean AT AN EVALUATED POINT. Restricting to
        # evaluated points keeps the incumbent from chasing an over-confident
        # dip of the surrogate in a region nothing has been measured.
        mean_at_obs, _ = gp.predict(xa)
        incumbent = float(mean_at_obs.min())

        cand = rng.random((n_candidates, dim))
        # Half the candidates are local perturbations of the incumbent, so the
        # search can refine as well as explore -- pure uniform sampling wastes
        # the budget once the region of interest is small.
        best_u = xa[int(np.argmin(mean_at_obs))]
        local = np.clip(best_u + rng.normal(0.0, 0.08, (n_candidates, dim)),
                        0.0, 1.0)
        cand = np.vstack([cand, local])
        ei = expected_improvement(gp, cand, incumbent)
        if float(ei.max()) <= 1e-12:
            converged = True
            message = (f"expected improvement below 1e-12 after {len(ys)} "
                       f"evaluations -- the surrogate sees nothing left to gain")
            break
        observe(cand[int(np.argmax(ei))])

    xa = np.array(xs)
    gp = fit_gp(xa, np.array(ys), noise=noise)
    mean_at_obs, _ = gp.predict(xa)
    i_post = int(np.argmin(mean_at_obs))
    i_obs = int(np.argmin(ys))
    return BOResult(
        x=list(_from_unit(xa[i_post], bnds)),
        fx=float(mean_at_obs[i_post]),
        fx_observed=float(ys[i_obs]),
        x_observed=list(_from_unit(xa[i_obs], bnds)),
        n_evals=len(ys),
        xs=[list(_from_unit(u, bnds)) for u in xa],
        ys=list(ys),
        converged=converged,
        message=message,
    )
