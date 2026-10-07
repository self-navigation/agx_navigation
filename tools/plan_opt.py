#!/usr/bin/env python3
"""Black-box search over planner parameters for wall clearance.

    python tools/plan_opt.py --out DIR [--trials N] [--jobs J] [--smoke]

Minimises, over the broad floor-6 start/goal pairs, the mean share of plan
points whose hull clearance is below min(D_REQ, achievable), where D_REQ = 0.50
m and achievable is the local passage half-width minus the robot half-width, so
narrow doorways are not penalised for being narrow. Optuna constraints require
enough broad pairs and narrow-door cases to solve and keep the mean plan
duration within 1.15x the stack baseline. The planner is deterministic, so each
trial is one evaluation; the first two trials are the stack recipe and the
centred library. Each trial's poses are kept under DIR/trials/<n>/.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(REPO, "tools")]
from replan_footprint import broad_cases, door_cases, load_floor  # noqa: E402
from bench_pmp_rollout import STACK_PLANNER_PARAMS  # noqa: E402
from rejoin_phase0 import _imports  # noqa: E402

D_REQ = 0.50
HALF_W = 0.2925
MIN_BROAD, MIN_DOORS, MAX_DUR = 34, 18, 1.15
BASE_FIELD = dict(speed_profile="exponential", inflation_radius=0.5)
ENQUEUE = [
    dict(centring=False, centring_cap=0.8, speed_decay_rate=2.5, speed_v_min=0.1,
         inflation_radius=0.5, use_fp=False, w_fp=100.0, fp_margin=0.15),        # stack
    dict(centring=True, centring_cap=0.8, speed_decay_rate=1.5, speed_v_min=0.1,
         inflation_radius=0.5, use_fp=False, w_fp=100.0, fp_margin=0.15),        # c_r15
]


def variant(p):
    field = dict(BASE_FIELD, centring=p["centring"], centring_cap=p["centring_cap"],
                 speed_decay_rate=p["speed_decay_rate"], speed_v_min=p["speed_v_min"],
                 inflation_radius=p["inflation_radius"])
    plan = dict(w_fp=p["w_fp"], fp_margin=p["fp_margin"]) if p["use_fp"] else {}
    return dict(field=field, plan=plan)


def run_one(job):
    """One offline rollout; same recipe as #48's regen.py (stack params, allow_unknown)."""
    case, V, out_dir = job
    _imports()
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner import rollout as ro
    from agx_planning.vector_field import VectorFieldConfig, VectorFieldGrid, compute_field

    arr, meta = load_floor(case["floor"])
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    cfg = PlannerConfig(mode="offline")
    for k, v in {**STACK_PLANNER_PARAMS, **V["plan"]}.items():
        setattr(cfg, k, v)
    g = lambda p: (int((p[0] - ox) / r), int((p[1] - oy) / r))  # noqa: E731
    gc, sc = g(case["goal"]), g(case["start"])
    t0 = time.perf_counter()
    P = np.zeros((0, 3))
    try:
        res, _ = compute_field(arr, gc[0], gc[1], r, ox, oy, VectorFieldConfig(**V["field"]),
                               sc[0], sc[1], 65, allow_unknown=True)
        field = VectorFieldGrid()
        field.update(res.travel_time, ox, oy, r, field_eps=cfg.field_eps,
                     wall_dist=res.wall_dist, wall_sigma=cfg.fp_smooth_sigma)
        solver = PMPShootingSolver(cfg, field)
        solver.reset_warm_start()
        x0 = np.array([*case["start"], case["yaw0"], 0.0, 0.0])
        goal = np.array([*case["goal"], case["goal_yaw"]])
        poses = []
        gen = ro.rollout_generator(solver, cfg, x0.copy(), goal)
        try:
            while True:
                poses.append(next(gen).poses)
        except StopIteration as stop:
            status = getattr(stop.value, "status", "?")
        P = np.concatenate(poses) if poses else P
    except Exception as e:  # noqa: BLE001 -- a crashed solve is a failed solve
        status = "exception:" + repr(e)[:120]
    rec = dict(case=case["case"], floor=case["floor"], status=status,
               wall_s=round(time.perf_counter() - t0, 1), n=int(len(P)))
    if len(P):
        rec["duration_s"] = round(len(P) / cfg.control_rate, 2)
        np.savez(os.path.join(out_dir, case["case"] + ".npz"), poses=P)
    return rec


class Scorer:
    """Hull clearance and achievable clearance per floor (ROS-free)."""

    def __init__(self):
        _imports()
        from scipy.ndimage import distance_transform_edt
        from agx_planning.supervisor.walls import SignedWallDistance
        from agx_planning.vector_field.field import local_half_width
        self.swd, self.ach, self.meta = {}, {}, {}
        for fl in (2, 3, 4, 5, 6):
            arr, meta = load_floor(fl)
            edt = distance_transform_edt(~(arr >= 65)) * meta["resolution"]
            self.ach[fl] = local_half_width(edt, meta["resolution"], 3.0) - HALF_W
            self.meta[fl] = meta
            self.swd[fl] = SignedWallDistance.from_map_yaml(
                os.path.join(REPO, f"src/rudn-ordjo-building/maps/floor_{fl}.yaml"))

    def clearance(self, P, fl=6):
        return np.array([self.swd[fl].body_clearance(*p[:3]) for p in P])

    def deficit(self, P, fl=6):
        """(avoidable deficit share, raw share < D_REQ, min clearance)."""
        c = self.clearance(P, fl)
        m = self.meta[fl]
        res, (ox, oy) = m["resolution"], m["origin"][:2]
        a = self.ach[fl]
        rr = np.clip(((P[:, 1] - oy) / res).astype(int), 0, a.shape[0] - 1)
        cc = np.clip(((P[:, 0] - ox) / res).astype(int), 0, a.shape[1] - 1)
        # load_floor returns the image row-flipped to grid convention (row = y).
        need = np.minimum(D_REQ, np.maximum(a[rr, cc], 0.0))
        return float(np.mean(c < need)), float(np.mean(c < D_REQ)), float(c.min())


def evaluate(params, cases, out_dir, pool, scorer, base_dur):
    os.makedirs(out_dir, exist_ok=True)
    V = variant(params)
    recs = list(pool.imap_unordered(run_one, [(c, V, out_dir) for c in cases]))
    broad = [r for r in recs if not r["case"].startswith("door")]
    doors = [r for r in recs if r["case"].startswith("door")]
    ok = lambda r: r["status"] == "success"  # noqa: E731
    defs, raws, mins = [], [], []
    for r in broad:
        if ok(r):
            P = np.load(os.path.join(out_dir, r["case"] + ".npz"))["poses"]
            d, raw, mn = scorer.deficit(P, r["floor"])
            r.update(deficit=d, raw=raw, min_clear=mn)
            defs.append(d), raws.append(raw), mins.append(mn)
    both = [r for r in broad if ok(r) and r["case"] in base_dur]
    dur_ratio = (sum(r["duration_s"] for r in both) / sum(base_dur[r["case"]] for r in both)
                 if both and base_dur else 1.0)
    out = dict(params=params, n_broad=sum(map(ok, broad)), n_doors=sum(map(ok, doors)),
               n_doors_total=len(doors), deficit=float(np.mean(defs)) if defs else 1.0,
               raw=float(np.mean(raws)) if raws else 1.0,
               overlaps=int(sum(m < 0 for m in mins)), dur_ratio=dur_ratio)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(dict(out, recs=recs), fh, indent=1)
    return out, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=max(1, mp.cpu_count() - 4))
    ap.add_argument("--smoke", action="store_true", help="5 broad pairs + 2 doors, 3 trials")
    a = ap.parse_args()
    import optuna

    cases = broad_cases(os.path.join(REPO, "tools/jobs/broad40.txt"))
    doors = door_cases([2, 3, 4, 5, 6], 0.95, np.radians(45))
    if a.smoke:
        cases, doors, a.trials = cases[:5], doors[:2], 3
    allc = cases + doors
    global MIN_BROAD, MIN_DOORS
    if a.smoke:
        MIN_BROAD, MIN_DOORS = 4, 1
    os.makedirs(a.out, exist_ok=True)
    scorer = Scorer()
    base_path = os.path.join(a.out, "base_dur.json")
    base_dur = json.load(open(base_path)) if os.path.exists(base_path) else {}

    study = optuna.create_study(
        study_name="plan_opt", storage=f"sqlite:///{a.out}/study.db", load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=0, n_startup_trials=10, multivariate=True,
            constraints_func=lambda t: t.user_attrs["constraints"]))
    if not study.trials:
        for p in ENQUEUE:
            study.enqueue_trial(p)

    with mp.Pool(a.jobs) as pool:
        def objective(trial):
            p = dict(
                centring=trial.suggest_categorical("centring", [False, True]),
                centring_cap=trial.suggest_float("centring_cap", 0.4, 1.2),
                speed_decay_rate=trial.suggest_float("speed_decay_rate", 0.5, 4.0),
                speed_v_min=trial.suggest_float("speed_v_min", 0.02, 0.4, log=True),
                inflation_radius=trial.suggest_float("inflation_radius", 0.35, 0.7),
                use_fp=trial.suggest_categorical("use_fp", [False, True]),
                w_fp=trial.suggest_float("w_fp", 5.0, 400.0, log=True),
                fp_margin=trial.suggest_float("fp_margin", 0.05, 0.3),
            )
            t0 = time.time()
            out, recs = evaluate(p, allc, os.path.join(a.out, "trials", str(trial.number)),
                                 pool, scorer, base_dur)
            if trial.number == 0 and not base_dur:  # trial 0 = the stack recipe
                base_dur.update({r["case"]: r["duration_s"] for r in recs
                                 if r["status"] == "success" and not r["case"].startswith("door")})
                json.dump(base_dur, open(base_path, "w"))
                out["dur_ratio"] = 1.0
            cons = [MIN_BROAD - out["n_broad"], MIN_DOORS - out["n_doors"],
                    out["dur_ratio"] - MAX_DUR]
            trial.set_user_attr("constraints", cons)
            for k in ("n_broad", "n_doors", "raw", "overlaps", "dur_ratio"):
                trial.set_user_attr(k, out[k])
            print(f"[trial {trial.number}] {time.time() - t0:.0f}s deficit {out['deficit']:.3f} "
                  f"raw<{D_REQ} {out['raw']:.3f} broad {out['n_broad']}/{len(cases)} "
                  f"doors {out['n_doors']}/{len(doors)} dur x{out['dur_ratio']:.2f} "
                  f"overlaps {out['overlaps']} {'OK' if max(cons) <= 0 else 'INFEASIBLE'} {p}",
                  flush=True)
            return out["deficit"]

        done = sum(t.state.is_finished() for t in study.trials)
        study.optimize(objective, n_trials=max(0, a.trials - done))

    feas = [t for t in study.trials if t.value is not None and max(t.user_attrs["constraints"]) <= 0]
    best = min(feas, key=lambda t: t.value) if feas else None
    print("best feasible:", None if best is None else (best.number, best.value, best.params))
    json.dump(dict(best=None if best is None else dict(number=best.number, value=best.value,
                                                       params=best.params, attrs=best.user_attrs)),
              open(os.path.join(a.out, "best.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
