#!/usr/bin/env python3
"""PMP teacher labels for the re-join student (#45 / S5, #35 phase 1).

Each SAMPLE is one re-join problem drawn exactly like Phase 0
(tools/rejoin_phase0.make_problems: a broad plan, a random index k, a REAL
TVLQR deviation from the traces times a log-uniform scale, a log-uniform T_w),
except that T_w is drawn from [3, 10] s by default -- the runtime policy's range
(pmp_planner/rejoin_service.T_W_MIN). It is solved by the SAME code the robot
runs, ``RejoinSolver`` (cost B, effort), so the labels are the runtime teacher's
answers, not a near relative's.

Per sample, in each .npz shard:
  s      (N, 10)  e_along, e_cross, e_theta  (x0 - plan[k], plan-heading frame)
                  dw_l, dw_r                 (target wheel speeds - x0's)
                  v, omega                   (x0's body velocities)
                  clearance                  (hull-to-wall, SignedWallDistance)
                  chi_hat                    (slip estimate; see chi_hat())
                  T_w
  a_pmp  (N, 12)  the solution's (v, omega) at 6 knots, linspace(0, T_w, 6)
  traj   (N, T, 5) the solution's state at linspace(0, T_w, T) (NaN if failed)
  m      (N,)     1 = solved, 0 = not (a_pmp/traj/j_chi0 NaN)
  j_chi0 (N,)     the label's tracking cost under NOMINAL chi (the PMP model
                  is exact there, so the solution is the trajectory): sum of
                  tuning.epsilon.step_cost over the plan ticks, error in the
                  plan frame vs plan[k+i], correction = (v, omega) minus the
                  plan's, at the ADOPTED gains (q_cross=2.5, r_omega=2.618).
  plus   plan (names), k, tgt, solve_ms, dev (N,5) for provenance.

    run: tools/rejoin_dataset.py --n 200 --jobs 4 -o <run_data dir>
"""

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = "/home/danya/university/clown-car/agx_navigation"
sys.path[:0] = [os.path.join(REPO, "tools"),
                os.path.join(REPO, "src/agx_navigation/agx_planning"),
                os.path.join(REPO, "src/rudn-ordjo-building")]

import rejoin_phase0 as p0  # noqa: E402

T_KNOTS = 6
T_TRAJ = 41
CHI_NOMINAL = 1.373


def chi_hat(prob):
    """Hook for S4's slip model: the chi estimate the student will be given.
    Nominal for now -- every label is solved under nominal chi anyway."""
    return CHI_NOMINAL


_C = {}


def _ctx(plan_path, tw_floor=3.0):
    from agx_planning.pmp_planner import PlannerConfig
    from agx_planning.pmp_planner.rejoin_service import RejoinSolver
    from agx_planning.runtime_corrector.trajectory_buffer import PlaybackSample
    from agx_planning.supervisor.walls import SignedWallDistance
    if plan_path not in _C:
        z = np.load(plan_path)
        dt = float(z["dt_sample"])
        cfg = PlannerConfig(mode="offline")
        samples = [PlaybackSample(left=float(w[0]), right=float(w[1]),
                                  pose=(float(p[0]), float(p[1]), float(p[2])))
                   for p, w in zip(z["poses"], z["wheel_cmds"])]
        my = str(z["map_yaml"])
        if not os.path.isabs(my):  # worktrees may lack the submodule checkout
            my = next((os.path.join(r, my) for r in (REPO, MAIN)
                       if os.path.exists(os.path.join(r, my))), os.path.join(REPO, my))
        if "walls" not in _C or _C["walls"][0] != my:
            _C["walls"] = (my, SignedWallDistance.from_map_yaml(my))
        _C[plan_path] = (RejoinSolver(samples, None, cfg, dt=dt, T_w_min=tw_floor), cfg, dt)
    return _C[plan_path] + (_C["walls"][1],)


def _label(prob):
    from agx_planning.pmp_planner.rejoin_service import body_knots, wrap
    from agx_planning.tuning.epsilon import CostWeights, step_cost
    adopted = CostWeights(q_cross=2.5, r_omega=2.618)
    rs, cfg, dt, walls = _ctx(prob["plan"], prob.get("tw_floor", 3.0))
    plan = rs.plan
    k, m = prob["k"], prob["m"]
    dev = np.array(prob["dev"])
    x0 = plan[k] + dev
    x0[2] = float(wrap(x0[2]))
    tgt = k + m
    th = plan[k, 2]
    v0, om0 = cfg.wheels_to_body(x0[3], x0[4])
    s = np.array([dev[0] * np.cos(th) + dev[1] * np.sin(th),
                  -dev[0] * np.sin(th) + dev[1] * np.cos(th),
                  wrap(dev[2]), plan[tgt, 3] - x0[3], plan[tgt, 4] - x0[4],
                  v0, om0, walls.body_clearance(x0[0], x0[1], x0[2]),
                  chi_hat(prob), m * dt])
    out = dict(id=prob["id"], plan=os.path.basename(prob["plan"])[:-4], k=k, tgt=tgt,
               dev=dev, s=s, a=np.full(2 * T_KNOTS, np.nan),
               traj=np.full((T_TRAJ, 5), np.nan), m=0, j=np.nan)
    n_before = len(rs.solve_ms)
    samples = rs.solve(x0, tgt, k_now=k)
    out["solve_ms"] = float(sum(rs.solve_ms[n_before:]))  # incl. the fallback retry
    out["status"] = rs.last.get("status")
    if samples is None:
        return out
    sol, T_w = rs.last["sol"], rs.last["T_w"]
    out["a"] = body_knots(sol, T_w, cfg, T_KNOTS)
    out["traj"] = sol(np.linspace(0.0, T_w, T_TRAJ))[0:5].T
    y = sol(np.arange(m) * dt)
    j = 0.0
    for i in range(m):
        ref = plan[k + i]
        d = y[0:2, i] - ref[0:2]
        c, sn = np.cos(ref[2]), np.sin(ref[2])
        e = [d[0] * c + d[1] * sn, -d[0] * sn + d[1] * c, float(wrap(y[2, i] - ref[2]))]
        u = np.subtract(cfg.wheels_to_body(y[3, i], y[4, i]),
                        cfg.wheels_to_body(ref[3], ref[4]))
        j += step_cost(e, u, dt, adopted)
    out.update(m=1, j=j)
    return out


def _write_shard(path, rows):
    np.savez_compressed(
        path,
        s=np.array([r["s"] for r in rows]), a_pmp=np.array([r["a"] for r in rows]),
        traj=np.array([r["traj"] for r in rows]), m=np.array([r["m"] for r in rows]),
        j_chi0=np.array([r["j"] for r in rows]), id=np.array([r["id"] for r in rows]),
        plan=np.array([r["plan"] for r in rows]), k=np.array([r["k"] for r in rows]),
        tgt=np.array([r["tgt"] for r in rows]), dev=np.array([r["dev"] for r in rows]),
        solve_ms=np.array([r["solve_ms"] for r in rows]),
        status=np.array([str(r["status"]) for r in rows]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--traj-dir", default=os.path.join(MAIN, "traj_data_v2"))
    ap.add_argument("--trace-dir", default=os.path.join(MAIN, "broad_gains_traces"))
    ap.add_argument("--soak", default=os.path.join(MAIN, "soak_data/soak_broad_gains.jsonl"))
    ap.add_argument("--gains", default="q1.5000_r2.6180")
    ap.add_argument("--tw-min", type=float, default=3.0)
    ap.add_argument("--tw-max", type=float, default=10.0)
    ap.add_argument("--tw-floor", type=float, default=3.0,
                    help="RejoinSolver.T_w_min (the runtime policy); a sweep below "
                         "3 s lowers it to --tw-min automatically")
    ap.add_argument("--scale-min", type=float, default=0.5)
    ap.add_argument("--scale-max", type=float, default=3.0)
    ap.add_argument("-o", "--out", required=True, help="output directory (run_data/...)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plans = p0.broad_plans(a.traj_dir, a.soak)
    probs = p0.make_problems(plans, a.trace_dir, a.gains, a.n, a.seed,
                             (a.tw_min, a.tw_max), (a.scale_min, a.scale_max))
    for q in probs:
        q["tw_floor"] = min(a.tw_floor, a.tw_min)
    probs.sort(key=lambda q: q["plan"])  # one solver/field per plan per worker
    commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    json.dump(dict(vars(a), commit=commit, n_problems=len(probs)),
              open(os.path.join(a.out, "args.json"), "w"), indent=1)
    print(f"{len(probs)} problems, {a.jobs} jobs -> {a.out}", flush=True)
    t0 = time.time()
    rows, shard = [], 0
    with Pool(a.jobs, maxtasksperchild=500) as pool:
        for i, r in enumerate(pool.imap_unordered(_label, probs, chunksize=4)):
            rows.append(r)
            if len(rows) >= a.shard_size:
                _write_shard(os.path.join(a.out, f"shard_{shard:04d}.npz"), rows)
                shard, rows = shard + 1, []
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(probs)}  {time.time() - t0:.0f}s", flush=True)
    if rows:
        _write_shard(os.path.join(a.out, f"shard_{shard:04d}.npz"), rows)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
