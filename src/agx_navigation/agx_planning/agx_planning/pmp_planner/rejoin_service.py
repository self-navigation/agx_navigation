"""Re-join solver: plans a return from an off-plan state onto a frozen PMP plan.

ROS-free. Wraps ``PMPShootingSolver.solve_rejoin(..., cost="effort")`` around
one plan and returns the re-join as playback samples the runtime corrector can
splice into its buffer::

    rs = RejoinSolver(plan_samples, field_data, cfg, windows=(3.0, 5.0, 8.0))
    samples = rs.solve(x0, target_index, k_now, accept)
    landed = rs.last["target_index"]   # resume playback at landed + 1

``x0`` is the measured state ``[px, py, theta, w_l, w_r]``. ``target_index`` is
the earliest landing; ``windows`` adds candidates at ``k_now + ceil(T/dt)``.
Each candidate is solved and the cheapest one that converged, stays within
``w_wheel_max + rejoin_wheel_tol`` and passes ``accept`` wins, where cost is
the re-join effort plus the effort of playing the plan to the farthest
candidate. Windows shorter than ``T_w_min`` are stretched and land late.
Returns None if nothing converged feasibly, or the cheapest rejected answer if
``accept`` refused all of them.

``field_data`` supplies the wall distance for the footprint barrier when
``cfg.w_fp > 0``. With ``retry_w_fp > 0`` and ``wall_retry`` set, rejected
answers are re-solved with the barrier (``retry_w_fp``, ``retry_fp_margin``)
until one passes. ``last`` holds the latest solve's diagnostics, including
every window.
"""

from __future__ import annotations

from math import pi
import dataclasses
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from ..runtime_corrector.trajectory_buffer import PlaybackSample
from .config import PlannerConfig
from .shooting_solver import PMPShootingSolver, rejoin_effort_rate

T_W_MIN = 3.0


def wrap(a):
    return (np.asarray(a) + pi) % (2.0 * pi) - pi


def wall_field(swd, wall_sigma: float = 0.05):
    """A ``VectorFieldGrid`` carrying only the wall distance of ``swd`` (a
    ``supervisor.walls.SignedWallDistance``), for cost B's footprint barrier.
    Travel time is zero: cost B never queries it. ``wall_sigma`` is the
    barrier's smoothing (``PlannerConfig.fp_smooth_sigma``)."""
    from ..vector_field import VectorFieldGrid
    g = VectorFieldGrid()
    g.update(np.zeros_like(swd.dist), swd.origin_x, swd.origin_y, swd.resolution,
             wall_dist=swd.dist, wall_sigma=wall_sigma)
    return g


def plan_guess_effort(plan: np.ndarray, k: int, m: int, x0: np.ndarray, dt: float,
                      N: int) -> np.ndarray:
    """Plan segment k..k+m with the initial deviation faded out linearly, zero
    costates (cost B's on-plan optimum). Same as tools/rejoin_phase0._plan_guess
    for cost="effort"."""
    t_seg = np.arange(m + 1) * dt
    t_mesh = np.linspace(0.0, m * dt, N + 1)
    seg = plan[k:k + m + 1].copy()
    seg[:, 2] = np.unwrap(seg[:, 2])
    dev = x0 - seg[0]
    dev[2] = wrap(dev[2])
    s = t_mesh / max(t_mesh[-1], 1e-9)
    st = np.stack([np.interp(t_mesh, t_seg, seg[:, i]) for i in range(5)])
    st += dev[:, None] * (1.0 - s)
    st[2] += x0[2] - seg[0, 2] - dev[2]
    return np.vstack([st, np.zeros((5, t_mesh.size))])


def body_knots(sol, T_w: float, cfg: PlannerConfig, n_knots: int = 6) -> np.ndarray:
    """(v, omega) of a re-join solution at n_knots uniform times over [0, T_w]
    (endpoints included), flattened as [v0, w0, v1, w1, ...]."""
    y = sol(np.linspace(0.0, T_w, n_knots))
    v, om = cfg.wheels_to_body(y[3], y[4])
    return np.column_stack([v, om]).ravel()


class RejoinSolver:
    """See the module docstring for the contract."""

    def __init__(self, plan_samples: Sequence[PlaybackSample], field_data=None,
                 cfg: Optional[PlannerConfig] = None, dt: Optional[float] = None,
                 T_w_min: float = T_W_MIN, max_nodes: Optional[int] = None,
                 retry_w_fp: float = 0.0, retry_fp_margin: Optional[float] = None,
                 windows: Sequence[float] = ()):
        self.cfg = cfg or PlannerConfig(mode="offline")
        self.dt = float(dt if dt is not None else self.cfg.dt)
        self.T_w_min = float(T_w_min)
        self.max_nodes = max_nodes
        self.windows = tuple(float(w) for w in windows)
        self.plan = np.array([[s.pose[0], s.pose[1], s.pose[2], s.left, s.right]
                              for s in plan_samples], dtype=np.float64)
        if len(self.plan) < 2:
            raise ValueError("plan needs at least 2 samples")
        self.goal = self.plan[-1, 0:3].copy()
        # Cumulative cost B of PLAYING the nominal plan (index 0..i), for
        # pricing the stretch a short re-join leaves to playback ("Landing
        # windows"). Accelerations by finite difference of the plan's
        # wheel-speed states, one per interval.
        acc = np.diff(self.plan[:, 3:5], axis=0) / self.dt
        mid = 0.5 * (self.plan[1:, 3:5] + self.plan[:-1, 3:5])
        rate = rejoin_effort_rate(self.cfg, acc[:, 0], acc[:, 1], mid[:, 0], mid[:, 1])
        self._plan_cum = np.concatenate([[0.0], np.cumsum(rate * self.dt)])
        self.solver = PMPShootingSolver(self.cfg, field_data)
        self.retry_solver = None
        if retry_w_fp > 0.0 and field_data is not None and field_data.has_wall_dist:
            # No speed barriers in the retry's cost: with w_fp at ~1000 the BVP
            # is stiff and its convergence guess-fragile, and on the unit-test
            # pillar the (there inactive) barriers alone turned a converging
            # retry into a failing one. The retry is still bound by the
            # feasibility check ("Wheel-speed limits").
            rcfg = dataclasses.replace(self.cfg, w_fp=float(retry_w_fp),
                                       rejoin_w_wheel_barrier=0.0,
                                       rejoin_w_v_barrier=0.0,
                                       rejoin_w_omega_barrier=0.0)
            if retry_fp_margin is not None:
                rcfg.fp_margin = float(retry_fp_margin)
            self.retry_solver = PMPShootingSolver(rcfg, field_data)
        self.last: dict = {}
        self.solve_ms: List[float] = []

    def nearest_index(self, x0: np.ndarray) -> int:
        d = np.hypot(self.plan[:, 0] - x0[0], self.plan[:, 1] - x0[1])
        return int(np.argmin(d))

    def landing_candidates(self, target_index: int, k_now: int) -> List[Tuple[int, int]]:
        """(landing index, window length m in ticks) pairs to try, earliest
        first. The caller's target is always the first; each of
        ``windows`` adds ``k_now + ceil(T/dt)``, never earlier than the
        caller's target and clipped to the plan's last index. ``m`` is the
        schedule ``tgt - k_now`` raised to ``T_w_min`` (the old "stretched"
        rule, which only bites near the plan's end). Duplicates are dropped."""
        last = len(self.plan) - 1
        tgt0 = int(min(max(target_index, 0), last))
        k_now = int(min(max(k_now, 0), tgt0))
        m_min = int(np.ceil(self.T_w_min / self.dt - 1e-9))
        tgts = [tgt0] + [k_now + int(np.ceil(w / self.dt - 1e-9)) for w in self.windows]
        out: List[Tuple[int, int]] = []
        for t in sorted(set(int(min(max(t, tgt0), last)) for t in tgts)):
            out.append((t, max(t - k_now, m_min)))
        return out

    def solve(self, x0: np.ndarray, target_index: int, k_now: Optional[int] = None,
              accept: Optional[Callable[[List[PlaybackSample], int], bool]] = None,
              wall_retry: Optional[bool] = None, budget_s: Optional[float] = None
              ) -> Optional[List[PlaybackSample]]:
        x0 = np.asarray(x0, dtype=np.float64).reshape(5)
        n_plan = len(self.plan)
        tgt0 = int(min(max(target_index, 0), n_plan - 1))
        if k_now is None:
            k_now = self.nearest_index(x0)
        k_now = int(min(max(k_now, 0), tgt0))
        if wall_retry is None:
            wall_retry = accept is not None
        cands = self.landing_candidates(tgt0, k_now)
        t_far = cands[-1][0]
        t_start = time.perf_counter()
        total_ms = 0.0
        rows = []
        for i, (tgt, m) in enumerate(cands):
            if (i > 0 and budget_s is not None and any(c["passed"] for c in rows)
                    and time.perf_counter() - t_start > budget_s):
                break
            r, ms = self._solve_window(self.solver, x0, tgt, m, k_now)
            total_ms += ms
            row = dict(tgt=tgt, m=m, T_w=m * self.dt, stretched=m > tgt - k_now,
                       status=r["status"], message=r["message"], nodes=r["nodes"],
                       solve_ms=ms, J=r.get("J"), J_tail=None, cost=None,
                       max_wheel=r.get("max_wheel"), passed=False, samples=None,
                       sol=r.get("sol"))
            if r["success"]:
                smp = self._to_samples(r["sol"], m)
                if smp is None:
                    row["status"] = "nonfinite"
                else:
                    row["samples"] = smp
                    row["J_tail"] = float(self._plan_cum[t_far] - self._plan_cum[tgt])
                    row["cost"] = row["J"] + row["J_tail"]
                    row["passed"] = accept is None or bool(accept(smp, tgt))
                    if not row["passed"]:
                        row["status"] = "blocked"
            rows.append(row)
        passed = [c for c in rows if c["passed"]]
        best = min(passed, key=lambda c: c["cost"]) if passed else None
        from_retry = False
        if best is None and wall_retry and self.retry_solver is not None:
            # Wall retry, cheapest gate-rejected window first.
            for c in sorted((c for c in rows if c["status"] == "blocked"),
                            key=lambda c: c["cost"]):
                r2 = None
                t_mesh = np.linspace(0.0, c["T_w"], self.cfg.N + 1)
                for g in (c["sol"](t_mesh), "plan", None):
                    r2, ms = self._solve_window(self.retry_solver, x0, c["tgt"], c["m"],
                                                k_now, guesses=(g,))
                    total_ms += ms
                    if r2 is not None and r2["success"]:
                        break
                if r2 is None:
                    continue
                c["status"] = "retry_" + r2["status"]
                if not r2["success"]:
                    continue
                smp = self._to_samples(r2["sol"], c["m"])
                if smp is None:
                    c["status"] = "retry_nonfinite"
                    continue
                c.update(samples=smp, sol=r2["sol"], J=r2.get("J"),
                         max_wheel=r2.get("max_wheel"))
                c["cost"] = c["J"] + c["J_tail"]
                c["passed"] = bool(accept(smp, c["tgt"]))
                best, from_retry = c, True
                if c["passed"]:
                    break
                c["status"] = "retry_blocked"
        if best is None:
            # Nothing passed: hand back the cheapest gate-rejected answer so
            # the caller's own gate counts it as blocked (not as a failure).
            blocked = [c for c in rows if c["samples"] is not None]
            best = min(blocked, key=lambda c: c["cost"]) if blocked else None
        pick = best or rows[0]
        self.last = dict(
            T_w=pick["T_w"], k_now=k_now, target_index=pick["tgt"],
            stretched=pick["stretched"], status=pick["status"] if best else
            self._fail_status(rows), message=pick["message"], nodes=pick["nodes"],
            solve_ms=total_ms, wall_retry=from_retry,
            J=pick["J"], cost=pick["cost"], max_wheel=pick["max_wheel"],
            sol=pick["sol"],
            windows=[{k: c[k] for k in ("tgt", "T_w", "status", "J", "J_tail", "cost",
                                        "max_wheel", "solve_ms", "passed")} for c in rows])
        return best["samples"] if best is not None else None

    @staticmethod
    def _fail_status(rows) -> str:
        """One status for a solve with no usable window: the most specific."""
        st = [c["status"] for c in rows]
        for s in ("infeasible", "nonfinite"):
            if s in st:
                return s
        return st[0]

    def _solve_window(self, solver, x0, tgt: int, m: int, k_now: int,
                      guesses=("plan", None)):
        """solve_rejoin onto plan index ``tgt`` over ``m`` ticks, trying
        ``guesses`` in order ("plan" = the plan segment leading to the target,
        as long as the window and starting earlier if it was stretched; None
        = solve_rejoin's linear blend; an array = that guess) until one
        converges AND passes the feasibility check. Returns (result, total
        solve ms)."""
        T_w = m * self.dt
        x_t = self.plan[tgt]
        k_g = max(tgt - m, 0)
        m_g = tgt - k_g
        r, total = None, 0.0
        for g in guesses:
            if isinstance(g, str):
                if m_g < 1:
                    continue
                # solve_rejoin lays any guess on linspace(0, T_w, ncols), so a
                # segment shorter than the window (target near the plan start)
                # is just stretched in time: shape only.
                g = plan_guess_effort(self.plan, k_g, m_g, x0, self.dt, self.cfg.N)
            r = solver.solve_rejoin(x0, x_t, T_w, self.goal, cost="effort",
                                    y_guess=g, max_nodes=self.max_nodes)
            self.solve_ms.append(r["solve_ms"])
            total += r["solve_ms"]
            if r["success"]:
                break
        return r, total

    def _to_samples(self, sol, m: int) -> Optional[List[PlaybackSample]]:
        """The solution on the plan's tick, None if non-finite. Wheel speeds are
        clipped to wheel_cmd_max, a no-op within rejoin_wheel_tol: an
        over-limit solution never gets here (solve_rejoin: "infeasible")."""
        y = sol(np.arange(m + 1) * self.dt)
        if not np.all(np.isfinite(y)):
            return None
        wmax = self.cfg.wheel_cmd_max
        out = []
        for i in range(m + 1):
            out.append(PlaybackSample(
                left=float(np.clip(y[3, i], -wmax, wmax)),
                right=float(np.clip(y[4, i], -wmax, wmax)),
                pose=(float(y[0, i]), float(y[1, i]), float(wrap(y[2, i]))),
                costates=None))
        return out

    def timing(self) -> dict:
        a = np.asarray(self.solve_ms)
        if a.size == 0:
            return dict(n=0)
        return dict(n=int(a.size), median_ms=float(np.median(a)),
                    p90_ms=float(np.percentile(a, 90)))
