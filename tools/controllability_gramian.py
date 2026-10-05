#!/usr/bin/env python3
"""Does the corrector's own controllability along a plan predict how badly it fails?

WHY THIS EXISTS (2026-09-23)
----------------------------
The advisor asked for the paper's theorem to be rebuilt so every experiment
follows from it (`../paper/porting-notes.md`, "Theorem rebuild"). The proposed
main hypothesis is UNIFORM CONTROLLABILITY of the tracking-error dynamics along
the plan: over every window of length T_loc the controllability Gramian must be
bounded below. The theorem then says the achievable error bound degrades as that
lower bound shrinks.

That is a TESTABLE prediction, and this script tests it before anyone writes a
proof around it. Written down BEFORE the numbers exist, so the result cannot be
rationalised after the fact:

    PREDICTION. Plans whose Gramian gets close to singular (the robot nearly
    stationary: pivots, reversals, slow turns) have higher J and worse arrival
    under the adopted corrector. Expected sign: Spearman rho between
    `energy_p90` and per-plan mean J is POSITIVE.

    If rho is ~0, the controllability hypothesis is not what separates easy
    plans from hard ones, and the theorem's main branch should not be sold as
    explaining the experiments. That is a useful answer too. Record it.

WHAT IS COMPUTED
----------------
The linearised error dynamics TVLQR uses (`runtime_corrector/tvlqr.py`), in the
reference frame, error e = (e_along, e_cross, e_heading), control (dv, domega):

    A(t) = [[0, w_ref, 0], [-w_ref, 0, v_ref], [0, 0, 0]],   B = [[1,0],[0,0],[0,b]]

where b = chi_plan / chi_plant scales the yaw authority: 1 on the nominal
surface, ~0.09 at chi = 15.6, just below the mu2 knee (tab:friction). For every window start
t0 on the plan, the finite-horizon Gramian

    W(t0) = int_{t0}^{t0+T} Phi(t0,s) B B^T Phi(t0,s)^T ds

and from it `energy_cross` = [W^-1]_{22}: the minimum control energy
(int |du|^2) needed to remove a 1 m cross-track error within the window. Large
means "nearly uncontrollable here"; inf means singular (v_ref ~ 0 throughout).
The lateral direction is steerable only through v_ref * e_heading, so this is
the direction that fails first, and the one the corridor metric measures.

Per plan: median / 90th percentile / max of energy_cross over windows, and the
fraction of windows that are singular. Joined, if --soak is given, with the
per-plan outcomes of ONE arm of ONE campaign (never mix campaigns: CLAUDE.md).

Pure numpy/scipy. Runs anywhere the plans are; needs no ROS and no Gazebo.

USAGE
  tools/controllability_gramian.py --plans $(cat plans.txt) \\
      --soak ~/run_data/2026-08-18_job100_broad-r-at-q25/soak_broad_r_at_q25.jsonl --arm 2.5,2.618 --out gramian.csv
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.linalg import expm
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                "src", "agx_navigation", "agx_planning"))
from agx_planning.rl_corrector.config import RLCorrectorConfig  # noqa: E402
from agx_planning.rl_corrector.nominal import load_recorded  # noqa: E402
from agx_planning.runtime_corrector import tvlqr as tvlqr_mod  # noqa: E402

# chi just below the mu2 knee, from tab:friction (mu_s = 0.45). Used only to
# show how far the Gramian collapses on the second branch; not a measured plant
# state for any particular plan.
CHI_BELOW_KNEE = 15.583
SINGULAR_COND = 1e12


def window_energies(v, w, dt, n_win, yaw_gain):
    """energy_cross for every window start. v, w: reference twist per step."""
    B = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, yaw_gain]])
    BBt = B @ B.T
    steps = [expm(np.array([[0.0, wk, 0.0], [-wk, 0.0, vk], [0.0, 0.0, 0.0]]) * dt)
             for vk, wk in zip(v, w)]
    out = []
    for k in range(len(v) - n_win + 1):
        W = np.zeros((3, 3))
        fwd = np.eye(3)                     # Phi(t_j, t_k)
        for j in range(k, k + n_win):
            back = np.linalg.inv(fwd)       # Phi(t_k, t_j)
            W += back @ BBt @ back.T * dt
            fwd = steps[j] @ fwd
        if np.linalg.cond(W) > SINGULAR_COND:
            out.append(math.inf)
        else:
            out.append(float(np.linalg.inv(W)[1, 1]))
    return np.array(out)


def summarize(e):
    finite = e[np.isfinite(e)]
    return {
        "energy_median": float(np.median(finite)) if finite.size else math.inf,
        "energy_p90": float(np.percentile(finite, 90)) if finite.size else math.inf,
        "energy_max": float(finite.max()) if finite.size else math.inf,
        "singular_frac": float(np.mean(~np.isfinite(e))) if e.size else math.nan,
    }


def outcomes(path, arm):
    """Per-plan mean J / final_err / max_cross / miss rate for ONE arm.

    Failed rollouts invalidate their sample and are never averaged over
    survivors, per the method in CLAUDE.md; they are counted instead.
    """
    rows = defaultdict(list)
    failed = defaultdict(int)
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if arm == "identity":
                if r.get("corrector") != "identity":
                    continue
            else:
                q, rr = arm
                if r.get("q_cross") is None or not (
                        math.isclose(r["q_cross"], q, rel_tol=1e-3)
                        and math.isclose(r["r_omega"], rr, rel_tol=1e-3)):
                    continue
            if "failed" in r:
                failed[r["trajectory"]] += 1
                continue
            rows[r["trajectory"]].append(r)
    out = {}
    for name, rs in rows.items():
        out[name] = {
            "n": len(rs), "n_failed": failed[name],
            "J_mean": float(np.mean([r["j_total"] for r in rs])),
            "final_err_mean": float(np.mean([r["final_err"] for r in rs])),
            "max_cross_mean": float(np.mean([r["max_cross"] for r in rs])),
            "miss_rate": float(np.mean([r["final_err"] > 0.5 for r in rs])),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plans", nargs="+", required=True, help=".npz plan files")
    ap.add_argument("--windows", type=float, nargs="+", default=[1.25, 2.5],
                    help="T_loc values, seconds (default: segment and horizon)")
    ap.add_argument("--soak", help="soak JSONL of ONE campaign to join outcomes from")
    ap.add_argument("--arm", default="2.5,2.618",
                    help="q,r of the arm to join, or 'identity'")
    ap.add_argument("--out", required=True, help="per-plan CSV")
    args = ap.parse_args()

    arm = "identity" if args.arm == "identity" else tuple(float(x) for x in args.arm.split(","))
    kin = RLCorrectorConfig()
    res = outcomes(args.soak, arm) if args.soak else {}

    table = []
    for path in args.plans:
        nom = load_recorded(path)
        name = os.path.basename(path)[:-4]
        twist = [tvlqr_mod.wheels_to_twist(float(a), float(b), kin) for a, b in nom.wheels]
        v = np.array([t[0] for t in twist])
        w = np.array([t[1] for t in twist])
        row = {"trajectory": name, "steps": len(v), "dt": nom.dt,
               "frac_slow": float(np.mean(np.abs(v) < 0.05))}
        for T in args.windows:
            n = max(2, int(round(T / nom.dt)))
            for label, gain in (("nominal", 1.0),
                                ("belowknee", kin.slip_chi / CHI_BELOW_KNEE)):
                for key, val in summarize(window_energies(v, w, nom.dt, n, gain)).items():
                    row[f"{key}_T{T:g}_{label}"] = val
        row.update(res.get(name, {}))
        table.append(row)
        print(f"{name:22s} frac_slow={row['frac_slow']:.2f} "
              f"p90(T={args.windows[0]:g})={row[f'energy_p90_T{args.windows[0]:g}_nominal']:.3g}"
              + (f"  J={row['J_mean']:.2f} final={row['final_err_mean']:.3f}"
                 if "J_mean" in row else ""), flush=True)

    keys = sorted({k for r in table for k in r}, key=lambda k: (k != "trajectory", k))
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(table)
    print(f"\nwrote {args.out}")

    joined = [r for r in table if "J_mean" in r]
    if len(joined) >= 5:
        print(f"\nSpearman rho over {len(joined)} plans (arm {args.arm}); "
              "the prediction is rho > 0 against J:")
        preds = ["frac_slow"] + [k for k in keys if k.startswith(("energy_p90", "singular_frac"))]
        for p in preds:
            x = [r[p] for r in joined]
            for o in ("J_mean", "final_err_mean", "miss_rate", "max_cross_mean"):
                rho, pval = spearmanr(x, [r[o] for r in joined])
                print(f"  {p:34s} vs {o:15s} rho={rho:+.3f} p={pval:.3g}")


if __name__ == "__main__":
    main()
