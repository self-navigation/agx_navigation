#!/usr/bin/env python3
"""Gate G2 (#44): the slip KinematicBridge vs Gazebo, open loop, 10 broad plans.

Replays each plan's wheel commands verbatim (the identity / open-loop arm,
`compare_correctors._identity_wheels`) on KinematicBridge with the SAME seed-0
along-path patches that `tuning.variance_probe.drive` spawned in Gazebo
(`along_path_terrain_sampler(nom.poses)(default_rng(0))`), and compares with the
identity rows of job 120 (`soak_broad_open_loop.jsonl`, 5 repeats per plan).

Gazebo kept no per-step trace for the identity arm (job 120 ran without
--trace-dir), so the comparison uses what each soak row has: `end_pose` and
`final_err`. "Deviation direction" is the sign of the end point's lateral offset
from the goal, in the goal pose's frame (left of the plan's final heading = +).
Gazebo's per plan is the sign of the mean over its repeats.

Pass (supervisor-plan G2): direction agrees on >= 8/10 AND final_err within a
factor 2 (kinematic/Gazebo in [0.5, 2]) on >= 8/10. The no-slip bridge is run
alongside as the reference the slip model must beat.

    python3 tools/validate_slip_model.py --root <main checkout> --out <json>
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "agx_navigation", "agx_planning"))

from agx_planning.rl_corrector import slip_model  # noqa: E402
from agx_planning.rl_corrector.compare_correctors import _identity_wheels  # noqa: E402
from agx_planning.rl_corrector.config import RLCorrectorConfig  # noqa: E402
from agx_planning.rl_corrector.kinematic_bridge import KinematicBridge  # noqa: E402
from agx_planning.rl_corrector.nominal import load_recorded  # noqa: E402
from agx_planning.rl_corrector.terrain import along_path_terrain_sampler  # noqa: E402


def lateral(end_xy, goal):
    gx, gy, gth = goal
    dx, dy = end_xy[0] - gx, end_xy[1] - gy
    return float(-np.sin(gth) * dx + np.cos(gth) * dy)


def replay(cfg, nom, surface_fn):
    br = KinematicBridge(cfg, surface_fn=surface_fn)
    st = br.reset(tuple(nom.poses[0]))
    track, mus = [st.pose], []
    for k in range(len(nom)):
        left, right = float(nom.wheels[k][0]), float(nom.wheels[k][1])
        st = br.step(_identity_wheels(left, right, cfg), nom.dt)
        track.append(st.pose)
        mus.append(br.last_mu if br.last_mu is not None else slip_model.GROUND_MU)
    return np.asarray(track), np.asarray(mus)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/danya/university/clown-car/agx_navigation")
    ap.add_argument("--plans", default=None, help="default: first N of tools/jobs/broad40.txt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--soak", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="results JSON (tracks in a sibling .npz)")
    a = ap.parse_args()

    plans_file = a.plans or os.path.join(a.root, "tools", "jobs", "broad40.txt")
    names = [os.path.basename(l.strip())[:-4] for l in open(plans_file) if l.strip()][: a.n]
    soak = a.soak or os.path.join(a.root, "soak_data", "soak_broad_open_loop.jsonl")
    rows = [json.loads(l) for l in open(soak) if l.strip()]
    rows = [r for r in rows if r.get("corrector") == "identity" and r.get("terrain")
            and r.get("seed") == a.seed and "end_pose" in r]

    cfg = RLCorrectorConfig()
    out, tracks = [], {}
    for name in names:
        nom = load_recorded(os.path.join(a.root, "traj_data_v2", name + ".npz"))
        goal = nom.poses[len(nom)]
        patches = along_path_terrain_sampler(nom.poses)(np.random.default_rng(a.seed))
        sf = slip_model.patches_to_surface_fn(patches)
        tr_s, mus = replay(cfg, nom, sf)
        tr_0, _ = replay(cfg, nom, None)
        gz = [r for r in rows if r["trajectory"] == name]
        if not gz:
            print(f"[g2] {name}: NO Gazebo identity rows", file=sys.stderr)
            continue
        gz_end = np.array([r["end_pose"][:2] for r in gz])
        gz_lat = [lateral(e, goal) for e in gz_end]
        gz_fe = [r["final_err"] for r in gz]
        rec = {
            "plan": name, "n_gz": len(gz), "patches": patches,
            "frac_below_knee": float(np.mean(mus <= slip_model.KNEE_MU)),
            "gz_final_err_mean": float(np.mean(gz_fe)), "gz_final_err_sd": float(np.std(gz_fe)),
            "gz_lat_mean": float(np.mean(gz_lat)),
            "gz_lat_sign_agree": int(np.sum(np.sign(gz_lat) == np.sign(np.mean(gz_lat)))),
            "gz_end": gz_end.tolist(),
        }
        for tag, tr in (("slip", tr_s), ("noslip", tr_0)):
            fe = float(np.hypot(*(tr[-1, :2] - goal[:2])))
            lat = lateral(tr[-1, :2], goal)
            rec[f"{tag}_final_err"] = fe
            rec[f"{tag}_lat"] = lat
            rec[f"{tag}_dir_agree"] = bool(np.sign(lat) == np.sign(rec["gz_lat_mean"]))
            ratio = fe / rec["gz_final_err_mean"]
            rec[f"{tag}_fe_ratio"] = ratio
            rec[f"{tag}_fe_within2"] = bool(0.5 <= ratio <= 2.0)
            tracks[f"{name}__{tag}"] = tr
        tracks[f"{name}__plan"] = nom.poses
        out.append(rec)
        print(f"[g2] {name}: gz fe={rec['gz_final_err_mean']:.3f}±{rec['gz_final_err_sd']:.3f} "
              f"lat={rec['gz_lat_mean']:+.3f} ({rec['gz_lat_sign_agree']}/{len(gz)}) | "
              f"slip fe={rec['slip_final_err']:.3f} lat={rec['slip_lat']:+.3f} "
              f"| noslip fe={rec['noslip_final_err']:.3f} lat={rec['noslip_lat']:+.3f} "
              f"| knee {rec['frac_below_knee']:.0%}")

    summary = {}
    for tag in ("slip", "noslip"):
        d = sum(r[f"{tag}_dir_agree"] for r in out)
        f = sum(r[f"{tag}_fe_within2"] for r in out)
        summary[tag] = {"dir_agree": d, "fe_within2": f, "n": len(out),
                        "pass": d >= 8 and f >= 8}
        print(f"[g2] {tag}: direction {d}/{len(out)}, final_err within x2 {f}/{len(out)} "
              f"-> {'PASS' if summary[tag]['pass'] else 'FAIL'}")
    json.dump({"summary": summary, "plans": out}, open(a.out, "w"), indent=1)
    np.savez(a.out[:-5] + "_tracks.npz", **tracks)


if __name__ == "__main__":
    main()
