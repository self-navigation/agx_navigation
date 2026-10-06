"""Runtime re-join solver: the PMP teacher as a callable (#45, #35 phase 1).

ROS-free. Wraps ``PMPShootingSolver.solve_rejoin(..., cost="effort")`` (the
re-join TPBVP of docs/corrector-design.md, Phase 0 in #11) around ONE frozen
plan, and returns the re-join as playback samples the runtime corrector can
splice into its buffer. S1 wires it in; nothing here touches ROS.

Contract (frozen with S1)::

    rs = RejoinSolver(plan_samples, field_data, cfg)
    samples = rs.solve(x0, target_index)   # Optional[List[PlaybackSample]]

* ``plan_samples`` -- the frozen plan as ``List[PlaybackSample]`` (left/right
  wheel-pair commands + planned pose) at the plan's tick ``dt`` (``cfg.dt``
  unless ``dt`` is given). The plan's commands ARE its predicted wheel-speed
  states (shooting_solver), so plan state i = (pose_i, left_i, right_i).
* ``field_data`` -- the plan's ``VectorFieldGrid`` or None. Cost B ("effort")
  never queries the field, so None is fine; it is kept in the signature so a
  cost-A variant can be swapped in without changing callers. The solver
  (and thus the field) is built once per plan: that is the per-plan cache.
* ``x0`` -- shape (5,) ``[px, py, theta, w_l, w_r]``, the MEASURED state
  (wheel speeds from /joint_states, not the commands).
* ``target_index`` -- the plan index to land on; clipped to the last index.

T_w policy. ``T_w = (target_index - k_now) * dt``, raised to ``T_w_min``
(3.0 s: Phase 0 solved 99% of problems at T_w >= 3 s, and short windows are
where it fails). ``k_now`` is the caller's current playback index; if not
given, the plan index nearest to x0's position is used. When the target is
closer than ``T_w_min`` the window is stretched, NOT the target moved: the
re-join then lands on the plan's state at ``target_index`` LATE by
``T_w - (target_index - k_now) * dt``; the caller must shift its playback clock
by ``len(samples) - 1 - (target_index - k_now)`` ticks (i.e. resume playback at
``target_index + 1`` after emitting the samples, whatever the wall time). The
preferred fix is the caller choosing ``target_index >= k_now + T_w_min/dt``.

Returned samples: ``n = round(T_w/dt) + 1`` samples on the plan's tick, sample 0
at x0 (its pose == x0's pose) and sample n-1 at the plan state at
``target_index`` (to the solver tolerance). left/right are the solution's
wheel-speed STATES (the same convention as the plan), clipped to
``cfg.wheel_cmd_max``. costates are None. Any failure (BVP not converged,
exception, non-finite) returns None -- the caller keeps its current behaviour.

``last`` holds the most recent solve's diagnostic dict (status, solve_ms, T_w,
nodes), and ``solve_ms`` accumulates every solve time for reporting.

Measured (local laptop, 200 Phase-0-distribution problems at T_w>=3 s, see
figures/2026-10-07/s5_rejoin/README.md): solve-time median / p90 reported
there.
"""

from __future__ import annotations

from math import pi
from typing import List, Optional, Sequence

import numpy as np

from ..runtime_corrector.trajectory_buffer import PlaybackSample
from .config import PlannerConfig
from .shooting_solver import PMPShootingSolver

T_W_MIN = 3.0


def wrap(a):
    return (np.asarray(a) + pi) % (2.0 * pi) - pi


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
                 T_w_min: float = T_W_MIN, max_nodes: Optional[int] = None):
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
        self.last: dict = {}
        self.solve_ms: List[float] = []

    def nearest_index(self, x0: np.ndarray) -> int:
        d = np.hypot(self.plan[:, 0] - x0[0], self.plan[:, 1] - x0[1])
        return int(np.argmin(d))

    def solve(self, x0: np.ndarray, target_index: int,
              k_now: Optional[int] = None) -> Optional[List[PlaybackSample]]:
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
                         solve_ms=r["solve_ms"])
        if not r["success"]:
            return None
        self.last["sol"] = r["sol"]
        y = r["sol"](np.arange(m + 1) * self.dt)
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
