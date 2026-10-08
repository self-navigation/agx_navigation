"""Re-join solver: plans a return from an off-plan state onto a frozen PMP plan.

ROS-free. Wraps ``PMPShootingSolver.solve_rejoin(..., cost="effort")`` around
one plan and returns the re-join as playback samples the runtime corrector can
splice into its buffer::

    rs = RejoinSolver(plan_samples, field_data, cfg)
    samples = rs.solve(x0, target_index)   # Optional[List[PlaybackSample]]

``x0`` is the measured state ``[px, py, theta, w_l, w_r]`` (wheel speeds from
/joint_states). The window is ``T_w = (target_index - k_now) * dt``, raised to
``T_w_min``; a stretched window lands late, and the caller resumes playback at
``target_index + 1`` after the samples. The samples run from x0 to the plan
state at ``target_index`` on the plan's tick, with wheel speeds clipped to
``cfg.wheel_cmd_max``. Any failure returns None. ``last`` holds the latest
solve's diagnostics.

``field_data`` supplies the wall distance for the footprint barrier when
``cfg.w_fp > 0``; without it the re-join ignores walls. With ``retry_w_fp > 0``
the wall-blind answer is solved first and, only if the caller's ``accept`` gate
rejects it, re-solved with the barrier (``retry_w_fp``, ``retry_fp_margin``),
which is slower and stiffer.
"""

from __future__ import annotations

from math import pi
import dataclasses
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..runtime_corrector.trajectory_buffer import PlaybackSample
from .config import PlannerConfig
from .shooting_solver import PMPShootingSolver

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
                 retry_w_fp: float = 0.0, retry_fp_margin: Optional[float] = None):
        self.cfg = cfg or PlannerConfig(mode="offline")
        self.dt = float(dt if dt is not None else self.cfg.dt)
        self.T_w_min = float(T_w_min)
        self.max_nodes = max_nodes
        self.plan = np.array([[s.pose[0], s.pose[1], s.pose[2], s.left, s.right]
                              for s in plan_samples], dtype=np.float64)
        if len(self.plan) < 2:
            raise ValueError("plan needs at least 2 samples")
        self.goal = self.plan[-1, 0:3].copy()
        self.solver = PMPShootingSolver(self.cfg, field_data)
        self.retry_solver = None
        if retry_w_fp > 0.0 and field_data is not None and field_data.has_wall_dist:
            rcfg = dataclasses.replace(self.cfg, w_fp=float(retry_w_fp))
            if retry_fp_margin is not None:
                rcfg.fp_margin = float(retry_fp_margin)
            self.retry_solver = PMPShootingSolver(rcfg, field_data)
        self.last: dict = {}
        self.solve_ms: List[float] = []

    def nearest_index(self, x0: np.ndarray) -> int:
        d = np.hypot(self.plan[:, 0] - x0[0], self.plan[:, 1] - x0[1])
        return int(np.argmin(d))

    def solve(self, x0: np.ndarray, target_index: int, k_now: Optional[int] = None,
              accept: Optional[Callable[[List[PlaybackSample]], bool]] = None
              ) -> Optional[List[PlaybackSample]]:
        x0 = np.asarray(x0, dtype=np.float64).reshape(5)
        n_plan = len(self.plan)
        tgt = int(min(max(target_index, 0), n_plan - 1))
        if k_now is None:
            k_now = self.nearest_index(x0)
        k_now = int(min(max(k_now, 0), tgt))
        m_sched = tgt - k_now
        m = max(m_sched, int(np.ceil(self.T_w_min / self.dt - 1e-9)))
        T_w = m * self.dt
        x_t = self.plan[tgt]
        # Guess: the plan segment leading to the target (as long as the
        # window, starting earlier if the window was stretched).
        k_g = max(tgt - m, 0)
        m_g = tgt - k_g
        self.last = dict(T_w=T_w, k_now=k_now, target_index=tgt, stretched=m > m_sched)
        r = None
        if m_g >= 1:
            g = plan_guess_effort(self.plan, k_g, m_g, x0, self.dt, self.cfg.N)
            # solve_rejoin lays any guess on linspace(0, T_w, ncols), so a
            # segment shorter than the window (target near the plan start)
            # is just stretched in time: shape only.
            r = self.solver.solve_rejoin(x0, x_t, T_w, self.goal, cost="effort",
                                         y_guess=g, max_nodes=self.max_nodes)
            self.solve_ms.append(r["solve_ms"])
        if r is None or not r["success"]:
            r = self.solver.solve_rejoin(x0, x_t, T_w, self.goal, cost="effort",
                                         max_nodes=self.max_nodes)
            self.solve_ms.append(r["solve_ms"])
        self.last.update(status=r["status"], message=r["message"], nodes=r["nodes"],
                         solve_ms=r["solve_ms"], wall_retry=False)
        if not r["success"]:
            return None
        out = self._samples(r["sol"], m)
        if out is None or accept is None or self.retry_solver is None or accept(out):
            return out
        # Rejected by the caller's wall gate: re-solve with the barrier.
        t_mesh = np.linspace(0.0, T_w, self.cfg.N + 1)
        guesses = [r["sol"](t_mesh)]
        if m_g >= 1:
            guesses.append(plan_guess_effort(self.plan, k_g, m_g, x0, self.dt, self.cfg.N))
        guesses.append(None)  # solve_rejoin's linear blend: on a synthetic pillar
        # it was the only guess that converged with the barrier on
        total = r["solve_ms"]
        self.last["wall_retry"] = True
        for g in guesses:
            r2 = self.retry_solver.solve_rejoin(x0, x_t, T_w, self.goal, cost="effort",
                                                y_guess=g, max_nodes=self.max_nodes)
            self.solve_ms.append(r2["solve_ms"])
            total += r2["solve_ms"]
            if r2["success"]:
                break
        self.last.update(status="retry_" + r2["status"], nodes=r2["nodes"], solve_ms=total)
        if not r2["success"]:
            return None
        return self._samples(r2["sol"], m)

    def _samples(self, sol, m: int) -> Optional[List[PlaybackSample]]:
        self.last["sol"] = sol
        y = sol(np.arange(m + 1) * self.dt)
        if not np.all(np.isfinite(y)):
            self.last["status"] = "nonfinite"
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
