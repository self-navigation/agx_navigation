#!/usr/bin/env python3
"""Score a wall-aware re-join sweep.

Each sub-directory is one configuration's rejoin_dataset output over the same
problems. Reports per configuration: solved, usable (solved and passing the
rejoin_clear wall gate), retry share, solve time percentiles, wall touches
from clear starts, and the cost of replaying each answer through
rl_corrector.rejoin_env at nominal chi and at chi ~ U[1.3, 2.0], over the
problems every configuration solved.

    python3 tools/score_rejoin_wall.py <sweep dir> --plans <traj_data_v2c> --map <floor_6.yaml>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "src/agx_navigation/agx_planning")]


def load(d):
    keys = ("s", "a_pmp", "m", "j_chi0", "plan", "k", "tgt", "dev", "traj", "id",
            "solve_ms", "retry", "gate", "cmin", "status")
    parts = [np.load(f) for f in sorted(glob.glob(os.path.join(d, "shard_*.npz")))]
    z = {k: np.concatenate([p[k] for p in parts]) for k in keys if k in parts[0].files}
    o = np.argsort(z["id"])
    return {k: v[o] for k, v in z.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--plans", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from agx_planning.rl_corrector import rejoin_env as renv
    from agx_planning.supervisor.walls import SignedWallDistance

    cfgs = sorted(d for d in os.listdir(a.sweep)
                  if glob.glob(os.path.join(a.sweep, d, "shard_*.npz")))
    cfgs.sort(key=lambda c: (c != "blind", c))
    Z = {c: load(os.path.join(a.sweep, c)) for c in cfgs}
    ids = Z[cfgs[0]]["id"]
    for c in cfgs:
        assert np.array_equal(Z[c]["id"], ids), f"{c}: different problems"
    common = np.all([Z[c]["m"] > 0 for c in cfgs], axis=0)
    plans = renv.load_plans(a.plans)
    walls = SignedWallDistance.from_map_yaml(a.map)
    rng = np.random.default_rng(a.seed)
    chi_rand = rng.uniform(1.3, 2.0, len(ids))
    clear0 = Z[cfgs[0]]["s"][:, 7] > 0
    rows = []
    for c in cfgs:
        z = Z[c]
        m = z["m"] > 0
        usable = m & (z["gate"] > 0)
        ms = z["solve_ms"]
        r = dict(cfg=c, n=int(len(ids)), solved=float(m.mean()), usable=float(usable.mean()),
                 retry=float(z["retry"].mean()),
                 retry_solved=float(m[z["retry"] > 0].mean()) if z["retry"].any() else float("nan"),
                 ms_p50=float(np.percentile(ms, 50)), ms_p90=float(np.percentile(ms, 90)),
                 ms_p99=float(np.percentile(ms, 99)),
                 touch_clear_start=float(np.mean(z["cmin"][m & clear0] < 0)))
        env = renv.RejoinEnv(plans, {k: z[k] for k in ("s", "traj", "m", "j_chi0", "plan", "k",
                                                       "tgt", "dev")}, walls)
        ii = np.where(common)[0]
        for tag, chi in (("nom", np.full(len(ii), renv.CHI_NOMINAL)), ("rand", chi_rand[ii])):
            out = {k: [] for k in ("cost", "j_track", "j_term", "j_contact", "t_contact")}
            for b in range(0, len(ii), 512):  # chunks: the laptop has ~12 GB
                sl = ii[b:b + 512]
                ro = env.rollout(sl, env.a_teacher[sl], chi[b:b + 512])
                for k in out:
                    out[k].append(ro[k])
            out = {k: np.concatenate(v) for k, v in out.items()}
            r.update({f"{tag}_cost": float(out["cost"].mean()),
                      f"{tag}_track": float(out["j_track"].mean()),
                      f"{tag}_land": float(out["j_term"].mean()),
                      f"{tag}_contact": float(out["j_contact"].mean()),
                      f"{tag}_contact_frac": float(np.mean(out["t_contact"] > 0))})
        r["n_common"] = int(common.sum())
        rows.append(r)
        print(json.dumps(r), flush=True)
    hdr = (f"{'cfg':10s} {'solved':>6s} {'usable':>6s} {'retry':>5s} {'r_ok':>5s} "
           f"{'ms50':>6s} {'ms90':>6s} {'ms99':>6s} {'touch':>5s} | "
           f"{'nom cost':>8s} {'track':>6s} {'land':>5s} {'cont':>5s} {'c%':>5s} | "
           f"{'rnd cost':>8s} {'c%':>5s}")
    lines = [f"{len(ids)} problems; rollouts on the {int(common.sum())} every config solved", hdr]
    for r in rows:
        lines.append(f"{r['cfg']:10s} {r['solved']:6.3f} {r['usable']:6.3f} {r['retry']:5.3f} "
                     f"{r['retry_solved']:5.2f} {r['ms_p50']:6.0f} {r['ms_p90']:6.0f} "
                     f"{r['ms_p99']:6.0f} {r['touch_clear_start']:5.3f} | "
                     f"{r['nom_cost']:8.3f} {r['nom_track']:6.3f} {r['nom_land']:5.3f} "
                     f"{r['nom_contact']:5.3f} {r['nom_contact_frac']:5.3f} | "
                     f"{r['rand_cost']:8.3f} {r['rand_contact_frac']:5.3f}")
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(a.sweep, "score.txt"), "w").write(txt + "\n")
    with open(os.path.join(a.sweep, "score.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
