#!/usr/bin/env python3
"""Phase 0 of the re-join re-planner: how often does the re-join TPBVP solve?

docs/corrector-design.md, issue #11. The library build failed 36% of fresh
start/goal pairs; if the re-join problem inherits that, PMP cannot be the
teacher and the supervised plan is wrong. This measures it.

One PROBLEM = (plan, k, deviation, T_w):
  - plan: one of the broad plans (traj_data_v2/*.npz);
  - k: a random index into the plan;
  - x0 = the plan's 5-D state at k plus a REAL deviation: the state a TVLQR
    rollout actually had at step k minus the plan's, from the per-step traces
    (trace step i is aligned with plan index i), times a log-uniform scale
    so the sweep covers deviations larger than TVLQR lets happen;
  - T_w: log-uniform over [0.1, 10] s; target = plan state at k + T_w/dt.
Each problem is solved with cost A (field) and cost B (effort).

Plan 5-D state = (pose, wheel_cmds): the plan's commands ARE its predicted
wheel-speed states (shooting_solver: "the published command is the planner's
PREDICTED wheel-speed state"), up to the deadzone/clip.

    run:    tools/rejoin_phase0.py run --n 300 --jobs 8 -o rejoin.jsonl
    report: tools/rejoin_phase0.py report rejoin.jsonl
    rescue: tools/rejoin_phase0.py rescue rejoin.jsonl -o rescue.jsonl
            (retry the failures: bigger mesh, multistart, homotopy)

Needs the workspace (scipy + skfmm + the planner), like
sample_eval_trajectories.py's solve mode.
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _imports():
    sys.path[:0] = [os.path.join(REPO, "src/agx_navigation/agx_planning"),
                    os.path.join(REPO, "src/rudn-ordjo-building")]
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.vector_field import (VectorFieldConfig, VectorFieldGrid,
                                           compute_field, world_to_grid)
    from PIL import Image

    # Inlined rather than imported: map_publisher and generate_trajectories
    # import rclpy at module level, and this tool must run without ROS. Same
    # logic as rudn_ordjo_building.map_publisher.load_occupancy_grid and
    # rl_corrector.generate_trajectories._build_field.
    def load_occupancy_grid(yaml_path):
        f = {}
        for line in open(yaml_path):
            line = line.split("#", 1)[0].strip()
            if ":" in line:
                key, _, val = line.partition(":")
                f[key.strip()] = val.strip()
        nums = lambda t: [float(v) for v in t.strip("[]").split(",")]  # noqa: E731
        meta = dict(image=f["image"], resolution=float(f["resolution"]),
                    origin=nums(f["origin"]), negate=int(f.get("negate", 0)),
                    occupied_thresh=float(f["occupied_thresh"]),
                    free_thresh=float(f["free_thresh"]))
        px = np.flipud(np.array(Image.open(
            os.path.join(os.path.dirname(yaml_path), meta["image"])).convert("L")))
        occ = (255.0 - px.astype(np.float64)) / 255.0
        if meta["negate"]:
            occ = 1.0 - occ
        data = np.full(px.shape, -1, dtype=np.int8)
        data[occ >= meta["occupied_thresh"]] = 100
        data[occ <= meta["free_thresh"]] = 0
        return data.ravel(), px.shape[1], px.shape[0], meta

    def _build_field(data, width, height, resolution, origin, start_xy, goal_xy,
                     occupancy_threshold, planner_cfg):
        arr = np.asarray(data, dtype=np.int8).reshape(height, width)
        gc = world_to_grid(goal_xy[0], goal_xy[1], origin[0], origin[1], resolution, width, height)
        sc = world_to_grid(start_xy[0], start_xy[1], origin[0], origin[1], resolution, width, height)
        if gc is None or sc is None:
            return None
        outcome = compute_field(arr, gc[0], gc[1], resolution, origin[0], origin[1],
                                VectorFieldConfig(), sc[0], sc[1], occupancy_threshold,
                                allow_unknown=False)
        if outcome is None:
            return None
        grid = VectorFieldGrid()
        grid.update(outcome[0].travel_time, origin[0], origin[1], resolution,
                    field_eps=planner_cfg.field_eps)
        return grid

    return PlannerConfig, PMPShootingSolver, _build_field, load_occupancy_grid


def broad_plans(traj_dir, soak):
    """The 40 broad plans = the trajectories in the broad soak."""
    names = sorted({json.loads(l)["trajectory"] for l in open(soak)})
    return [os.path.join(traj_dir, n + ".npz") for n in names]


def trace_deviations(trace_dir, plan_path, gains):
    """(n_traces, n_steps, 5) actual-minus-plan state per step, NaN-padded."""
    stem = os.path.basename(plan_path)[:-4]
    z = np.load(plan_path)
    plan = np.column_stack([z["poses"], z["wheel_cmds"]])
    devs = []
    for f in sorted(glob.glob(os.path.join(trace_dir, f"{stem}_{gains}_*.csv"))):
        rows = [r for r in csv.DictReader(open(f)) if r["phase"] == "step"]
        n = min(len(rows), len(plan))
        if n == 0:  # a rollout that failed its reset guard; no steps
            continue
        # Joint order [FL, RL, FR, RR]: w0 = left pair, w2 = right pair.
        act = np.array([[float(r[c]) for c in ("x", "y", "yaw", "w0", "w2")]
                        for r in rows[:n]])
        d = act - plan[:n]
        d[:, 2] = (d[:, 2] + np.pi) % (2 * np.pi) - np.pi
        pad = np.full((len(plan), 5), np.nan)
        pad[:n] = d
        devs.append(pad)
    return np.array(devs)


def make_problems(plans, trace_dir, gains, n, seed, tw_range, scale_range):
    rng = np.random.default_rng(seed)
    devs = {p: trace_deviations(trace_dir, p, gains) for p in plans}
    plans = [p for p in plans if len(devs[p])]
    probs = []
    while len(probs) < n:
        p = plans[rng.integers(len(plans))]
        z = np.load(p)
        dt = float(z["dt_sample"])
        n_plan = len(z["poses"])
        T_w = float(np.exp(rng.uniform(*np.log(tw_range))))
        m = int(round(T_w / dt))
        if m < 1 or m >= n_plan - 1:
            continue
        k = int(rng.integers(0, n_plan - m))
        tr = int(rng.integers(len(devs[p])))
        d = devs[p][tr, k]
        if not np.all(np.isfinite(d)):
            continue
        scale = float(np.exp(rng.uniform(*np.log(scale_range))))
        probs.append(dict(id=len(probs), plan=p, k=k, m=m, T_w=m * dt, trace=tr,
                          scale=scale, dev=(d * scale).tolist()))
    return probs


_CTX = {}


def _plan_guess(plan, costates, dt, k, m, x0, N, cost):
    """Guess = the nominal plan's own segment k..k+m, with the initial
    deviation faded out linearly so it meets both pins. Costates: the plan's
    (cost A's own, so a near-solution for small deviations); zero for cost B,
    whose on-plan optimum is zero acceleration and hence zero wheel costates."""
    t_seg = np.arange(m + 1) * dt
    t_mesh = np.linspace(0.0, m * dt, N + 1)
    seg = plan[k:k + m + 1].copy()
    seg[:, 2] = np.unwrap(seg[:, 2])
    dev = x0 - seg[0]
    dev[2] = (dev[2] + np.pi) % (2 * np.pi) - np.pi
    s = t_mesh / max(t_mesh[-1], 1e-9)
    st = np.stack([np.interp(t_mesh, t_seg, seg[:, i]) for i in range(5)])
    st += dev[:, None] * (1.0 - s)
    st[2] += x0[2] - seg[0, 2] - dev[2]  # land on x0's heading branch
    if cost == "field":
        cs = np.stack([np.interp(t_mesh, t_seg, costates[k:k + m + 1, i]) for i in range(5)])
    else:
        cs = np.zeros((5, t_mesh.size))
    return np.vstack([st, cs])


def _solve_one(prob):
    PlannerConfig, PMPShootingSolver, _build_field, load_occupancy_grid = _imports()
    p = prob["plan"]
    if p not in _CTX:
        z = np.load(p)
        my = str(z["map_yaml"])
        if not os.path.isabs(my):
            my = os.path.join(REPO, my)
        data, width, height, meta = load_occupancy_grid(my)
        cfg = PlannerConfig(mode="offline")
        goal_xy = z["goal_xy"]
        field = _build_field(data, width, height, meta["resolution"], meta["origin"],
                             z["start_xy"], goal_xy, 65, cfg)
        plan = np.column_stack([z["poses"], z["wheel_cmds"]])
        goal = np.array([goal_xy[0], goal_xy[1], z["poses"][-1, 2]])
        _CTX[p] = (PMPShootingSolver(cfg, field) if field is not None else None, plan, goal,
                   np.asarray(z["costates"]), float(z["dt_sample"]))
    if prob.get("_setup_only"):
        return None
    solver, plan, goal, costates, dt = _CTX[p]
    k, m = prob["k"], prob["m"]
    x0 = plan[k] + np.array(prob["dev"])
    x_t = plan[k + m]
    dev = np.array(prob["dev"])
    th = plan[k, 2]
    out = dict(prob, plan=os.path.basename(p)[:-4],
               e_along=float(dev[0] * np.cos(th) + dev[1] * np.sin(th)),
               e_cross=float(-dev[0] * np.sin(th) + dev[1] * np.cos(th)),
               e_theta=float(dev[2]), e_wheel=float(np.hypot(dev[3], dev[4])),
               # Necessary condition only: each wheel must at least make up
               # its own speed change at the acceleration bound. T_w below
               # this is infeasible regardless of the solver.
               t_min_wheel=float(np.max(np.abs(x_t[3:5] - x0[3:5]))
                                 / solver.cfg.a_wheel_max) if solver else None)
    for cost in ("field", "effort"):
        if solver is None:
            out[cost] = dict(status="nofield")
            continue
        y_guess = (_plan_guess(plan, costates, dt, k, m, x0, solver.cfg.N, cost)
                   if prob.get("guess", "blend") == "plan" else None)
        r = solver.solve_rejoin(x0, x_t, prob["T_w"], goal, cost=cost, y_guess=y_guess)
        res = {k2: r[k2] for k2 in ("status", "message", "nodes", "niter", "solve_ms")}
        if r["success"]:
            yT = r["sol"](prob["T_w"])
            res.update(max_accel=r["max_accel"], max_wheel=r["max_wheel"],
                       pin_err=float(np.max(np.abs(yT[0:5] - x_t))))
        out[cost] = res
    return out


def _bump_guess(base, x0, x_t, amp):
    """base with a sideways detour of amp metres at mid-window (sin profile,
    zero at both pins), perpendicular to the start->target chord. A different
    homotopy class for turn-drive-turn manoeuvres the straight guess misses."""
    g = base.copy()
    s = np.linspace(0.0, 1.0, g.shape[1])
    d = x_t[:2] - x0[:2]
    n = np.array([-d[1], d[0]]) / max(np.hypot(*d), 1e-6)
    g[0:2] += amp * np.sin(np.pi * s)[None, :] * n[:, None]
    return g


def _traj(sol, T_w, n=41):
    y = sol(np.linspace(0.0, T_w, n))
    return np.round(y[0:5].T, 4).tolist()


def _rescue_one(item):
    """Retry one failed (problem, cost) harder, recording what worked:
      1. nodes: the same plan guess, mesh cap raised to --max-nodes;
      2. multistart: blend guess and plan guess + sideways detours of
         +-0.4 / +-1.0 m, each with the raised cap;
      3. homotopy in the deviation: solve dev*alpha for alpha 0 -> 1,
         warm-starting each rung from the previous solution and halving the
         step on failure (on-plan, alpha=0, is trivially solvable). Records
         alpha_max, the largest fraction of the deviation that solved.
    The trajectory of every success is kept (41 samples of the 5-D state) so
    the interesting ones can be drawn."""
    prob, cost, max_nodes = item
    _solve_one_setup(prob)
    solver, plan, goal, costates, dt = _CTX[prob["plan_path"]]
    k, m, T_w = prob["k"], prob["m"], prob["T_w"]
    dev = np.array(prob["dev"])
    x_t = plan[k + m]
    N = solver.cfg.N
    tries = []
    t0 = time.time()

    def attempt(name, x0, guess):
        r = solver.solve_rejoin(x0, x_t, T_w, goal, cost=cost, y_guess=guess,
                                max_nodes=max_nodes)
        tries.append(dict(method=name, status=r["status"], message=r["message"][:60],
                          nodes=r["nodes"], solve_ms=round(r["solve_ms"], 1)))
        return r

    x0 = plan[k] + dev
    pg = _plan_guess(plan, costates, dt, k, m, x0, N, cost)
    win = None
    r = attempt("nodes", x0, pg)
    if r["success"]:
        win = ("nodes", r)
    if win is None:
        cands = [("blend", None)] + [(f"bump{a:+g}", _bump_guess(pg, x0, x_t, a))
                                     for a in (0.4, -0.4, 1.0, -1.0)]
        for name, g in cands:
            r = attempt(name, x0, g)
            if r["success"]:
                win = (name, r)
                break
    alpha_max = None
    if win is None:
        alpha, step, sol = 0.0, 0.25, None
        while step >= 1.0 / 64 and alpha < 1.0:
            a_try = min(1.0, alpha + step)
            xa = plan[k] + dev * a_try
            g = (_plan_guess(plan, costates, dt, k, m, xa, N, cost) if sol is None
                 else sol(np.linspace(0.0, T_w, N + 1)))
            if sol is not None:
                g[0:5, 0] = xa  # move the start pin onto this rung
            r = solver.solve_rejoin(xa, x_t, T_w, goal, cost=cost, y_guess=g,
                                    max_nodes=max_nodes)
            if r["success"]:
                alpha, sol = a_try, r["sol"]
            else:
                step /= 2
        alpha_max = alpha
        tries.append(dict(method="homotopy", alpha_max=alpha))
        if alpha >= 1.0:
            win = ("homotopy", dict(success=True, sol=sol, max_accel=None))
    # 4. more time: the same deviation, the window stretched (target moves
    #    down the plan to k + m'), plan guess, raised cap. tw_solved is the
    #    shortest stretched window that solves -- "how long would it need?".
    #    Only run when the stretch is the point (win is None) or asked for.
    tw_solved, stretch = None, None
    if win is None:
        n_plan = len(plan)
        for f in (1.5, 2.0, 3.0, 4.0, 6.0, 8.0):
            m2 = int(round(m * f))
            if m2 < m + 2 or k + m2 >= n_plan:
                continue
            x_t2 = plan[k + m2]
            r = solver.solve_rejoin(x0, x_t2, m2 * dt, goal, cost=cost,
                                    y_guess=_plan_guess(plan, costates, dt, k, m2, x0, N, cost),
                                    max_nodes=max_nodes)
            tries.append(dict(method=f"stretch{f:g}", status=r["status"], T_w=round(m2 * dt, 2)))
            if r["success"]:
                tw_solved = m2 * dt
                stretch = dict(T_w=tw_solved, traj=_traj(r["sol"], tw_solved),
                               plan_seg=np.round(plan[k:k + m2 + 1, 0:3], 4).tolist())
                break
    out = dict(id=prob["id"], cost=cost, rescued=win is not None,
               method=win[0] if win else None, alpha_max=alpha_max, tries=tries,
               tw_solved=tw_solved, stretch=stretch,
               wall_s=round(time.time() - t0, 2))
    if win:
        out["traj"] = _traj(win[1]["sol"], T_w)
        out["plan_seg"] = np.round(plan[k:k + m + 1, 0:3], 4).tolist()
    return out


def _solve_one_setup(prob):
    p = prob["plan_path"]
    if p not in _CTX:
        _solve_one(dict(prob, plan=p, _setup_only=True))


def cmd_rescue(a):
    rows = [json.loads(l) for l in open(a.jsonl)]
    by_name = {os.path.basename(p)[:-4]: p for p in glob.glob(os.path.join(a.traj_dir, "*.npz"))}
    items = []
    for r in rows:
        r["plan_path"] = by_name[r["plan"]]
        for cost in a.costs.split(","):
            if r[cost]["status"] != "ok":
                items.append((r, cost, a.max_nodes))
    items = items[:a.limit] if a.limit else items
    items.sort(key=lambda it: it[0]["plan"])
    print(f"{len(items)} failed (problem, cost) pairs, {a.jobs} jobs -> {a.out}", flush=True)
    t0 = time.time()
    with open(a.out, "w") as fh, Pool(a.jobs, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(_rescue_one, items, chunksize=1)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def cmd_run(a):
    plans = broad_plans(a.traj_dir, a.soak)
    probs = make_problems(plans, a.trace_dir, a.gains, a.n, a.seed,
                          (a.tw_min, a.tw_max), (a.scale_min, a.scale_max))
    # Group by plan so each worker builds each field once.
    for q in probs:
        q["guess"] = a.guess
    probs.sort(key=lambda q: q["plan"])
    print(f"{len(probs)} problems over {len({q['plan'] for q in probs})} plans, "
          f"{a.jobs} jobs -> {a.out}", flush=True)
    t0 = time.time()
    with open(a.out, "w") as fh, Pool(a.jobs, maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(_solve_one, probs, chunksize=4)):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(probs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


def cmd_report(a):
    rows = [json.loads(l) for l in open(a.jsonl)]
    print(f"{len(rows)} problems\n")
    tw_bins = [0.1, 0.3, 1.0, 3.0, 10.01]
    dev_bins = [0.0, 0.05, 0.15, 0.4, 1e9]
    for cost in ("field", "effort"):
        ok = np.array([r[cost]["status"] == "ok" for r in rows])
        ms = np.array([r[cost]["solve_ms"] for r in rows if "solve_ms" in r[cost]])
        print(f"== cost {cost}: fail {100 * (1 - ok.mean()):.1f}% "
              f"({(~ok).sum()}/{len(rows)}); solve ms median {np.median(ms):.0f} "
              f"p90 {np.percentile(ms, 90):.0f}")
        causes = {}
        for r in rows:
            if r[cost]["status"] != "ok":
                c = r[cost]["status"] + ": " + r[cost].get("message", "")[:60]
                causes[c] = causes.get(c, 0) + 1
        for c, nc in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"   {nc:4d}  {c}")
        tw = np.array([r["T_w"] for r in rows])
        dv = np.array([np.hypot(r["e_along"], r["e_cross"]) for r in rows])
        print("   fail% by T_w:       " + "  ".join(
            f"[{lo:g},{hi:.3g}) {100 * (1 - ok[(tw >= lo) & (tw < hi)].mean()):.0f}% "
            f"n={((tw >= lo) & (tw < hi)).sum()}"
            for lo, hi in zip(tw_bins, tw_bins[1:]) if ((tw >= lo) & (tw < hi)).any()))
        print("   fail% by |e_xy| m:  " + "  ".join(
            f"[{lo:g},{hi:g}) {100 * (1 - ok[(dv >= lo) & (dv < hi)].mean()):.0f}% "
            f"n={((dv >= lo) & (dv < hi)).sum()}"
            for lo, hi in zip(dev_bins, dev_bins[1:]) if ((dv >= lo) & (dv < hi)).any()))
        feas = np.array([r["t_min_wheel"] is not None and r["T_w"] > 1.5 * r["t_min_wheel"]
                         for r in rows])
        if feas.any():
            print(f"   with T_w > 1.5x the wheel-speed lower bound: fail "
                  f"{100 * (1 - ok[feas].mean()):.1f}% (n={feas.sum()}); "
                  f"below it: {100 * (1 - ok[~feas].mean()) if (~feas).any() else float('nan'):.0f}% "
                  f"(n={(~feas).sum()})")
        sat = [r[cost]["max_accel"] for r in rows if r[cost]["status"] == "ok"]
        pe = [r[cost]["pin_err"] for r in rows if r[cost]["status"] == "ok"]
        if sat:
            print(f"   of successes: max|a| >= 95% of bound in "
                  f"{100 * np.mean(np.array(sat) >= 0.95 * 12.5):.0f}%; "
                  f"pin_err max {max(pe):.2g}")
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=300)
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--traj-dir", default=os.path.join(REPO, "traj_data_v2"))
    r.add_argument("--trace-dir", default=os.path.join(REPO, "broad_gains_traces"))
    r.add_argument("--soak", default=os.path.join(REPO, "soak_data/soak_broad_gains.jsonl"))
    r.add_argument("--gains", default="q1.5000_r2.6180",
                   help="trace gains to draw deviations from (no q2.5 traces exist)")
    r.add_argument("--tw-min", type=float, default=0.1)
    r.add_argument("--tw-max", type=float, default=10.0)
    r.add_argument("--scale-min", type=float, default=0.5)
    r.add_argument("--scale-max", type=float, default=3.0)
    r.add_argument("--guess", choices=("blend", "plan"), default="blend",
                   help="blend: linear x0->target, zero costates; plan: the "
                        "nominal segment with the deviation faded out")
    r.add_argument("-o", "--out", default="rejoin_phase0.jsonl")
    r.set_defaults(fn=cmd_run)
    p = sub.add_parser("report")
    p.add_argument("jsonl")
    p.set_defaults(fn=cmd_report)
    q = sub.add_parser("rescue", help="retry a run's failures harder (see _rescue_one)")
    q.add_argument("jsonl")
    q.add_argument("--costs", default="field,effort")
    q.add_argument("--max-nodes", type=int, default=20000)
    q.add_argument("--jobs", type=int, default=8)
    q.add_argument("--limit", type=int, default=0, help="first N pairs only (timing)")
    q.add_argument("--traj-dir", default=os.path.join(REPO, "traj_data_v2"))
    q.add_argument("-o", "--out", default="rejoin_rescue.jsonl")
    q.set_defaults(fn=cmd_rescue)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
