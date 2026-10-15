#!/usr/bin/env python3
"""Offline comparison of re-join speed barriers and landing windows.

Problems are drawn as in tools/rejoin_phase0.make_problems and solved through
``RejoinSolver`` as the runtime corrector calls it: target = k + ceil(3 s /
dt), k_now = k, and the wall gate ``supervisor.walls.rejoin_clear`` against the
baked map with the core's plan-relative stop thresholds. Arms:

  before    -- no speed barriers, no feasibility check, single landing;
  wheel     -- wheel-speed barrier only + feasibility check, single landing;
  barrier   -- wheel/v/omega barriers + feasibility check, single landing;
  wheel_win -- wheel + landing windows (3, 5, 8) s;
  windows   -- barrier + landing windows (3, 5, 8) s.

Per arm: solved (converged and feasible), passed (also through the gate), max
wheel speed and how many exceed w_wheel_max, total solve time.

    .venv/bin/python tools/rejoin_windows_bench.py --n 150 --jobs 4 -o out.jsonl
    .venv/bin/python tools/rejoin_windows_bench.py --report out.jsonl
"""

import argparse
import json
import os
import sys
import warnings
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "tools"),
                os.path.join(REPO, "src/agx_navigation/agx_planning"),
                os.path.join(REPO, "src/rudn-ordjo-building")]

ARMS = {
    "before": dict(bar=0.0, vbar=0.0, tol=float("inf"), windows=()),
    "wheel": dict(bar=50.0, vbar=0.0, tol=0.5, windows=()),
    "barrier": dict(bar=50.0, vbar=50.0, tol=0.5, windows=()),
    "wheel_win": dict(bar=50.0, vbar=0.0, tol=0.5, windows=(3.0, 5.0, 8.0)),
    "windows": dict(bar=50.0, vbar=50.0, tol=0.5, windows=(3.0, 5.0, 8.0)),
}
_CTX = {}


def _ctx(plan_path):
    if plan_path in _CTX:
        return _CTX[plan_path]
    import rejoin_phase0 as p0
    from agx_planning.runtime_corrector.trajectory_buffer import PlaybackSample
    from agx_planning.supervisor.core import SupervisorConfig, SupervisorCore
    from agx_planning.supervisor.walls import SignedWallDistance
    _, _, _, load_occupancy_grid = p0._imports()
    z = np.load(plan_path)
    my = str(z["map_yaml"])
    if not os.path.isabs(my):
        my = os.path.join(REPO, my)
    data, w, h, meta = load_occupancy_grid(my)
    swd = SignedWallDistance.from_occupancy(data, w, h, meta["resolution"],
                                            meta["origin"][0], meta["origin"][1])
    poses = np.asarray(z["poses"])
    dt = float(z["dt_sample"])
    core = SupervisorCore(SupervisorConfig())
    core.set_plan(poses, np.arange(len(poses)) * dt,
                  clearance=[swd.body_clearance(*p) for p in poses])
    samples = [PlaybackSample(left=float(c[0]), right=float(c[1]),
                              pose=(float(p[0]), float(p[1]), float(p[2])))
               for p, c in zip(poses, z["wheel_cmds"])]
    _CTX[plan_path] = (samples, swd, core, dt)
    return _CTX[plan_path]


def _solve(prob):
    warnings.filterwarnings("ignore")
    from agx_planning.pmp_planner import PlannerConfig
    from agx_planning.pmp_planner.rejoin_service import RejoinSolver
    from agx_planning.supervisor.walls import rejoin_clear
    samples, swd, core, dt = _ctx(prob["plan"])
    plan = np.array([[*s.pose, s.left, s.right] for s in samples])
    k = prob["k"]
    x0 = plan[k] + np.array(prob["dev"])
    tgt = min(k + int(np.ceil(3.0 / dt - 1e-9)), len(plan) - 1)
    c0 = swd.body_clearance(*x0[:3])

    def gate(smp, t):
        ks = np.linspace(k, t, len(smp)).round().astype(int)
        return rejoin_clear(smp, swd, c0, [core.wall_thresholds(int(i))[1] for i in ks])

    out = dict(id=prob["id"], plan=os.path.basename(prob["plan"])[:-4], k=k,
               scale=prob["scale"], e_xy=float(np.hypot(*prob["dev"][:2])), c0=float(c0))
    for name, a in ARMS.items():
        cfg = PlannerConfig(mode="offline", rejoin_w_wheel_barrier=a["bar"],
                            rejoin_w_v_barrier=a["vbar"], rejoin_w_omega_barrier=a["vbar"],
                            rejoin_wheel_tol=a["tol"])
        rs = RejoinSolver(samples, None, cfg, dt=dt, max_nodes=3000, windows=a["windows"])
        smp = rs.solve(x0, tgt, k, gate, wall_retry=False)
        L = rs.last
        sol = L.get("sol")
        vmax = None
        if sol is not None:
            y = sol(np.linspace(0.0, L["T_w"], 201))
            vmax = float(np.max(np.abs(cfg.wheels_to_body(y[3], y[4])[0])))
        out[name] = dict(
            solved=any(w["status"] in ("ok", "blocked") for w in L["windows"]),
            passed=smp is not None and gate(smp, L["target_index"]),
            status=L["status"], T_w=L["T_w"], max_wheel=L.get("max_wheel"), max_v=vmax,
            solve_ms=L["solve_ms"],
            windows=[dict(T_w=w["T_w"], status=w["status"], ms=w["solve_ms"],
                          max_wheel=w["max_wheel"], cost=w["cost"], passed=w["passed"])
                     for w in L["windows"]])
    return out


def report(rows):
    print(f"{len(rows)} problems")
    for name in ARMS:
        r = [x[name] for x in rows]
        # over the RETURNED solutions only (a rejected one never reaches the robot)
        mw = np.array([x["max_wheel"] for x in r if x["passed"]])
        ms = np.array([x["solve_ms"] for x in r])
        vm = np.array([x["max_v"] for x in r if x["passed"]])
        print(f"{name:9s} solved {np.mean([x['solved'] for x in r]):6.1%}  "
              f"gate-passed {np.mean([x['passed'] for x in r]):6.1%}  "
              f"max|w| median {np.median(mw):5.2f} max {mw.max():5.2f} "
              f"(>20.05: {int((mw > 20.05).sum())})  "
              f"max|v| median {np.median(vm):4.2f}  "
              f"solve ms median {np.median(ms):6.1f} p90 {np.percentile(ms, 90):7.1f}")
        st = {}
        for x in r:
            st[x["status"]] = st.get(x["status"], 0) + 1
        print(" " * 9 + "status " + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))
    for arm, ref in (("wheel_win", "wheel"), ("windows", "barrier")):
        per = {}
        for x in rows:
            for w in x[arm]["windows"]:
                per.setdefault(round(w["T_w"], 1), []).append(w)
        print(f"{arm} arm, per landing window (T_w rounded):")
        for T in sorted(per):
            ws = per[T]
            print(f"  T_w {T:4.1f} s  n {len(ws):4d}  converged+feasible "
                  f"{np.mean([w['status'] in ('ok', 'blocked') for w in ws]):6.1%}  "
                  f"gate-passed {np.mean([w['passed'] for w in ws]):6.1%}  "
                  f"median {np.median([w['ms'] for w in ws]):6.1f} ms")
        picks = {}
        for x in rows:
            if x[arm]["passed"]:
                T = round(x[arm]["T_w"], 1)
                picks[T] = picks.get(T, 0) + 1
        print(f"  chosen T_w: {dict(sorted(picks.items()))}")
        rescued = sum(1 for x in rows if x[arm]["passed"] and not x[ref]["passed"])
        lost = sum(1 for x in rows if x[ref]["passed"] and not x[arm]["passed"])
        print(f"  vs {ref}: gate-passed gained {rescued}, lost {lost}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-min", type=float, default=0.5)
    ap.add_argument("--scale-max", type=float, default=3.0)
    ap.add_argument("--traj-dir", default=os.path.join(REPO, "traj_data_v2"))
    ap.add_argument("--trace-dir", default=os.path.join(REPO, "broad_gains_traces"))
    ap.add_argument("--soak", default=os.path.join(REPO, "soak_data/soak_broad_gains.jsonl"))
    ap.add_argument("--gains", default="q1.5000_r2.6180")
    ap.add_argument("-o", "--out", default="rejoin_windows_bench.jsonl")
    ap.add_argument("--report", metavar="JSONL")
    a = ap.parse_args()
    if a.report:
        report([json.loads(l) for l in open(a.report)])
        return
    import rejoin_phase0 as p0
    plans = p0.broad_plans(a.traj_dir, a.soak)
    # T_w is fixed by the runtime policy here; make_problems only needs room
    # for an 8 s window after k (so every arm sees the same problems).
    probs = p0.make_problems(plans, a.trace_dir, a.gains, a.n, a.seed, (8.0, 8.0001),
                             (a.scale_min, a.scale_max))
    with Pool(a.jobs) as pool, open(a.out, "w") as f:
        rows = []
        for i, r in enumerate(pool.imap_unordered(_solve, probs, chunksize=2)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(probs)}", flush=True)
    report(rows)


if __name__ == "__main__":
    main()
