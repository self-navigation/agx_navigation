"""Re-join environment for the TD3 and TD3+BC re-join actor, pure numpy.

One episode is one re-join and one step (a contextual bandit, gamma = 0). The
action is a residual on the plan's (v, omega) at n_knots knots over
linspace(0, T_w, n_knots). The knots are integrated into a reference, which a
batched TVLQR tracks on a kinematic plant with yaw deficit
omega_actual = omega_cmd * chi_nominal / chi_plant, chi_plant randomised per
problem. Reward is -(J_track + w_terminal * pin_err**2 + w_contact *
t_contact), with J_track from tuning.epsilon.step_cost. The state is the
pose/wheel error to plan[k], body twist, clearance, chi_hat, T_w and the plan
twist at the knots (see RejoinEnv.state).
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..runtime_corrector.tvlqr import TVLQRConfig
from ..supervisor.walls import SignedWallDistance
from ..tuning.epsilon import CostWeights
from .config import RLCorrectorConfig

N_KNOTS = 21
TRAJ_POINTS = 41   # tools/rejoin_dataset.T_TRAJ
CHI_NOMINAL = RLCorrectorConfig().slip_chi


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class RejoinEnvConfig:
    chi_range: tuple = (1.3, 2.0)
    chi_hat_noise: float = 0.0          # sd of the estimator error fed to s[8]
    w_contact: float = 10.0             # per second of hull-on-wall
    n_knots: int = N_KNOTS              # (TRAJ_POINTS - 1) must divide by n_knots - 1
    n_sub: int = 10                     # plant / reference sub-steps per tick
    weights: CostWeights = field(default_factory=lambda: CostWeights(q_cross=2.5,
                                                                     r_omega=2.6183452282612643))
    tvlqr: TVLQRConfig = field(default_factory=TVLQRConfig)


def load_plans(plan_dir: str) -> Dict[str, np.ndarray]:
    """name -> (N, 5) [x, y, theta, w_l, w_r], the RejoinSolver's plan layout."""
    out = {}
    for f in sorted(glob.glob(os.path.join(plan_dir, "*.npz"))):
        z = np.load(f)
        out[os.path.basename(f)[:-4]] = np.column_stack([z["poses"], z["wheel_cmds"]]).astype(float)
    return out


def load_labels(label_dir: str, keys=("s", "a_pmp", "m", "j_chi0", "plan", "k", "tgt", "dev", "traj")):
    """Concatenate the label shards (tools/rejoin_dataset.py). ``traj`` is the
    big one (N x 41 x 5); drop it from ``keys`` when it is not needed."""
    parts = {k: [] for k in keys}
    for f in sorted(glob.glob(os.path.join(label_dir, "shard_*.npz"))):
        z = np.load(f)
        for k in keys:
            parts[k].append(z[k])
    return {k: np.concatenate(v) for k, v in parts.items()}


# ----------------------------------------------------------------------------
# Batched TVLQR gains (the same recursion as tvlqr.gain_schedule)
# ----------------------------------------------------------------------------


def batched_gains(v_ref: np.ndarray, om_ref: np.ndarray, valid: np.ndarray, dt: float,
                  cfg: TVLQRConfig) -> np.ndarray:
    """(B, T) reference twists -> (B, T, 2, 3) gains. Each row's horizon ends at
    its own last valid tick (``valid`` (B, T) bool, a prefix): the Riccati seed
    Qf is placed there, as ``gain_schedule`` would for that row alone."""
    B, T = v_ref.shape
    Q, R, Qf = cfg.q_matrix(), cfg.r_matrix(), cfg.qf_matrix()
    vmin = cfg.min_speed_for_steering
    vf = np.where(np.abs(v_ref) >= vmin, v_ref, np.where(v_ref < 0, -vmin, vmin))
    S = np.broadcast_to(Qf, (B, 3, 3)).copy()
    K = np.zeros((B, T, 2, 3))
    Bd = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]) * dt
    for t in range(T - 1, -1, -1):
        Ad = np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
        Ad[:, 0, 1] = om_ref[:, t] * dt
        Ad[:, 1, 0] = -om_ref[:, t] * dt
        Ad[:, 1, 2] = vf[:, t] * dt
        BtS = np.einsum("ji,bjk->bik", Bd, S)                     # (B, 2, 3)
        Kt = np.linalg.solve(R + BtS @ Bd, BtS @ Ad)              # (B, 2, 3)
        AmBK = Ad - Bd @ Kt
        Sp = Q + np.einsum("bji,jk,bkl->bil", Kt, R, Kt) + np.einsum("bji,bjk,bkl->bil", AmBK, S, AmBK)
        Sp = 0.5 * (Sp + np.swapaxes(Sp, 1, 2))
        # valid is a prefix, so a row's S stays at Qf through its padding and
        # the recursion starts at its own last tick
        live = valid[:, t]
        K[:, t] = np.where(live[:, None, None], Kt, 0.0)
        S = np.where(live[:, None, None], Sp, S)
    return K


# ----------------------------------------------------------------------------
# The environment
# ----------------------------------------------------------------------------


class RejoinEnv:
    """Batched one-step re-join environment over a fixed pool of problems.

    A problem is (plan, k, m, x0); the label shards supply the pool, so the
    teacher's action is known for every state the env can produce.
    """

    def __init__(self, plans: Dict[str, np.ndarray], labels: dict, walls: SignedWallDistance,
                 cfg: Optional[RejoinEnvConfig] = None, dt: float = 0.1,
                 kin: Optional[RLCorrectorConfig] = None):
        self.cfg = cfg or RejoinEnvConfig()
        self.kin = kin or RLCorrectorConfig()
        self.dt = float(dt)
        self.walls = walls
        self.n_knots = int(self.cfg.n_knots)
        if (TRAJ_POINTS - 1) % (self.n_knots - 1):
            raise ValueError(f"n_knots - 1 must divide {TRAJ_POINTS - 1}")
        self.state_dim = 10 + 2 * self.n_knots
        self.action_dim = 2 * self.n_knots
        names = sorted(plans)
        self.plan_names = names
        lens = np.array([len(plans[n]) for n in names])
        self.plan_arr = np.full((len(names), lens.max(), 5), np.nan)
        for i, n in enumerate(names):
            self.plan_arr[i, :lens[i]] = plans[n]
        self.plan_len = lens
        idx = {n: i for i, n in enumerate(names)}
        self.pid = np.array([idx[str(p)] for p in labels["plan"]])
        self.k = labels["k"].astype(int)
        self.tgt = np.minimum(labels["tgt"].astype(int), lens[self.pid] - 1)
        self.m = self.tgt - self.k
        self.x0 = self.plan_arr[self.pid, self.k] + labels["dev"]
        self.x0[:, 2] = wrap(self.x0[:, 2])
        self.s10 = labels["s"].astype(float).copy()
        self.solved = labels["m"].astype(bool)
        self.j_label = labels["j_chi0"].astype(float)
        self.n = len(self.k)
        self.plan_knots = self._plan_knots(np.arange(self.n))
        self.a_teacher = self._teacher_knots(labels["traj"])

    # -- per-problem features ------------------------------------------------

    def _plan_twist(self, pid, idx):
        p = self.plan_arr[pid, idx]
        return self.kin.wheels_to_body(p[..., 3], p[..., 4])

    def _plan_knots(self, ii):
        """Plan (v, omega) at the knot times, linear in the fractional index."""
        f = self.k[ii, None] + np.linspace(0.0, 1.0, self.n_knots)[None] * self.m[ii, None]
        lo = np.floor(f).astype(int)
        hi = np.minimum(lo + 1, self.plan_len[self.pid[ii], None] - 1)
        w = f - lo
        pid = self.pid[ii, None]
        vl, ol = self._plan_twist(pid, lo)
        vh, oh = self._plan_twist(pid, hi)
        v, om = (1 - w) * vl + w * vh, (1 - w) * ol + w * oh
        return np.stack([v, om], axis=-1).reshape(len(ii), -1)

    def _teacher_knots(self, traj):
        """The PMP solution's (v, omega) at the knots, from the label's 41-point
        trajectory (NaN where the solve failed)."""
        step = (TRAJ_POINTS - 1) // (self.n_knots - 1)
        w = traj[:, ::step, 3:5]
        v, om = self.kin.wheels_to_body(w[..., 0], w[..., 1])
        return np.stack([v, om], axis=-1).reshape(len(traj), -1)

    def state(self, ii, chi_plant, rng=None) -> np.ndarray:
        s = np.empty((len(ii), self.state_dim))
        s[:, :10] = self.s10[ii]
        noise = 0.0
        if rng is not None and self.cfg.chi_hat_noise > 0:
            noise = rng.normal(0.0, self.cfg.chi_hat_noise, len(ii))
        s[:, 8] = chi_plant + noise
        s[:, 10:] = self.plan_knots[ii]
        return s

    def sample_chi(self, n, rng) -> np.ndarray:
        lo, hi = self.cfg.chi_range
        return rng.uniform(lo, hi, n)

    def teacher_residual(self, ii) -> np.ndarray:
        return self.a_teacher[ii] - self.plan_knots[ii]

    def clip_twist(self, v, om):
        """The hardware net: per-wheel cap, mapped back to the twist."""
        cap = self.kin.wheel_cmd_max
        wl, wr = self.kin.body_to_wheels(v, om)
        return self.kin.wheels_to_body(np.clip(wl, -cap, cap), np.clip(wr, -cap, cap))

    # -- the rollout ---------------------------------------------------------

    def rollout(self, ii, action, chi_plant, tvlqr: bool = True, record: bool = False) -> dict:
        """Roll out absolute knot actions (B, 2 n_knots) on problems ``ii`` at
        the given plant chi (B,). Returns the cost parts per problem."""
        ii = np.asarray(ii)
        B = len(ii)
        av, ao = self.clip_twist(np.asarray(action)[:, 0::2], np.asarray(action)[:, 1::2])
        nk = self.n_knots
        m = self.m[ii]
        T = int(m.max())
        dt, ns = self.dt, int(self.cfg.n_sub)
        h = dt / ns
        valid = np.arange(T)[None] < m[:, None]                         # (B, T)
        Tw = m * dt
        rows = np.arange(B)[:, None]

        def knot_twist(t):  # t (1, n) or (B, n) times -> (B, n) v, omega
            u = np.clip(t / Tw[:, None] * (nk - 1), 0, nk - 1 - 1e-9)
            j0 = np.floor(u).astype(int)
            fr = u - j0
            return ((1 - fr) * av[rows, j0] + fr * av[rows, j0 + 1],
                    (1 - fr) * ao[rows, j0] + fr * ao[rows, j0 + 1])

        # reference: the continuous knot twist, finely integrated; its twist at
        # each tick start is what the corrector feeds forward
        v_ref, om_ref = knot_twist(np.arange(T)[None] * dt)
        vf, of = knot_twist((np.arange(T * ns)[None] + 0.5) * h)       # midpoint of each sub-step
        ref = np.empty((B, T + 1, 3))
        p = self.x0[ii, :3].copy()
        ref[:, 0] = p
        for j in range(T * ns):
            th = p[:, 2] + 0.5 * of[:, j] * h
            p[:, 0] += vf[:, j] * np.cos(th) * h
            p[:, 1] += vf[:, j] * np.sin(th) * h
            p[:, 2] += of[:, j] * h
            if (j + 1) % ns == 0:
                ref[:, (j + 1) // ns] = p
        del vf, of
        K = batched_gains(v_ref, om_ref, valid, dt, self.cfg.tvlqr) if tvlqr else None
        tc = self.cfg.tvlqr
        yaw_gain = CHI_NOMINAL / np.asarray(chi_plant, dtype=float)
        pid = self.pid[ii]
        kk = self.k[ii]
        pidx = np.minimum(kk[:, None] + np.arange(T)[None], self.plan_len[pid, None] - 1)
        plan = self.plan_arr[pid[:, None], pidx]                          # (B, T, 5)
        pv, pom = self.kin.wheels_to_body(plan[..., 3], plan[..., 4])
        Wq, Wr = self.cfg.weights.q(), self.cfg.weights.r()
        x = self.x0[ii, :3].copy()
        j_track = np.zeros(B)
        t_contact = np.zeros(B)
        traj = np.empty((B, T + 1, 3)) if record else None
        if record:
            traj[:, 0] = x
        for i in range(T):
            live = valid[:, i]
            vc, oc = v_ref[:, i].copy(), om_ref[:, i].copy()
            if tvlqr:
                d = x[:, :2] - ref[:, i, :2]
                c, s = np.cos(ref[:, i, 2]), np.sin(ref[:, i, 2])
                e = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1],
                              wrap(x[:, 2] - ref[:, i, 2])], axis=1)
                du = -np.einsum("bij,bj->bi", K[:, i], e)
                du[:, 1] = np.where(np.abs(v_ref[:, i]) < tc.min_speed_for_steering, 0.0, du[:, 1])
                vc += np.clip(du[:, 0], -tc.max_dv, tc.max_dv)
                oc += np.clip(du[:, 1], -tc.max_domega, tc.max_domega)
            vc, oc = self.clip_twist(vc, oc)
            # score this tick against the plan
            d = x[:, :2] - plan[:, i, :2]
            c, s = np.cos(plan[:, i, 2]), np.sin(plan[:, i, 2])
            ep = np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1],
                           wrap(x[:, 2] - plan[:, i, 2])], axis=1)
            up = np.stack([vc - pv[:, i], oc - pom[:, i]], axis=1)
            cost = (np.einsum("bi,ij,bj->b", ep, Wq, ep) + np.einsum("bi,ij,bj->b", up, Wr, up)) * dt
            j_track += np.where(live, cost, 0.0)
            clear = self.body_clearance(x)
            t_contact += np.where(live & (clear < 0), dt, 0.0)
            # plant: command held for the tick, yaw deficit chi_nominal / chi_plant
            om_act = np.where(live, oc * yaw_gain, 0.0)
            v_act = np.where(live, vc, 0.0)
            for _ in range(ns):
                th = x[:, 2] + 0.5 * om_act * h
                x[:, 0] += v_act * np.cos(th) * h
                x[:, 1] += v_act * np.sin(th) * h
                x[:, 2] += om_act * h
            if record:
                traj[:, i + 1] = x
        tgt = self.plan_arr[pid, self.tgt[ii]]
        pin = np.hypot(x[:, 0] - tgt[:, 0], x[:, 1] - tgt[:, 1])
        j_term = self.cfg.weights.w_terminal * pin ** 2
        j_contact = self.cfg.w_contact * t_contact
        out = dict(j_track=j_track, j_term=j_term, pin=pin, t_contact=t_contact,
                   j_contact=j_contact, cost=j_track + j_term + j_contact)
        if record:
            out["traj"] = traj
            out["ref"] = ref
        return out

    def body_clearance(self, x: np.ndarray) -> np.ndarray:
        """Vectorised ``SignedWallDistance.body_clearance`` over (B, 3) poses."""
        w = self.walls
        c, s = np.cos(x[:, 2:3]), np.sin(x[:, 2:3])
        bx, by = w.body[None, :, 0], w.body[None, :, 1]
        qx = x[:, 0:1] + bx * c - by * s
        qy = x[:, 1:2] + bx * s + by * c
        col = np.clip(((qx - w.origin_x) / w.resolution).astype(int), 0, w.dist.shape[1] - 1)
        row = np.clip(((qy - w.origin_y) / w.resolution).astype(int), 0, w.dist.shape[0] - 1)
        return w.dist[row, col].min(axis=1)
