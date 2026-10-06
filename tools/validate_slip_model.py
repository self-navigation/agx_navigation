#!/usr/bin/env python3
"""Compare the slip KinematicBridge with Gazebo, open loop, on the broad plans.

Replays each plan's wheel commands on KinematicBridge with the same seed-0
along-path patches used in Gazebo, for three models: noslip, centre and
footprint (see slip_model). Gazebo reference tracks come from per-step soak
traces (--trace-dir). Per plan and model it reports end-point lateral
direction, the final_err ratio and the max cross-track ratio. Passes when
direction agrees on at least 8 of 10 plans and final_err is within a factor 2
on at least 8 of 10.

    python3 tools/validate_slip_model.py --root <main checkout> --out <json>
"""
import argparse
import csv
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

MODELS = ("noslip", "centre", "footprint")
RUN = "run_data/2026-10-07_s4-identity-traces"


def lateral(end_xy, goal):
    gx, gy, gth = goal
    dx, dy = end_xy[0] - gx, end_xy[1] - gy
    return float(-np.sin(gth) * dx + np.cos(gth) * dy)


def max_cross(track, plan):
    n = min(len(track), len(plan))
    t, p = np.asarray(track)[:n], np.asarray(plan)[:n]
    d = t[:, :2] - p[:, :2]
    e = -np.sin(p[:, 2]) * d[:, 0] + np.cos(p[:, 2]) * d[:, 1]
    return float(np.max(np.abs(e)))


def replay(cfg, nom, surface_fn, mode):
    br = KinematicBridge(cfg, surface_fn=surface_fn, contact_mode=mode)
    st = br.reset(tuple(nom.poses[0]))
    track, mus = [st.pose], []
    for k in range(len(nom)):
        left, right = float(nom.wheels[k][0]), float(nom.wheels[k][1])
        st = br.step(_identity_wheels(left, right, cfg), nom.dt)
        track.append(st.pose)
        mus.append(br.last_mu if br.last_mu is not None else slip_model.GROUND_MU)
    return np.asarray(track), np.asarray(mus)


def load_trace(path):
    """Gazebo track (x, y, yaw): the last reset row, then every `step` row."""
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    steps = [r for r in rows if r["phase"] == "step"]
    pre = [r for r in rows if r["phase"] != "step"]
    rows = ([pre[-1]] if pre else []) + steps
    return np.array([[float(r["x"]), float(r["y"]), float(r["yaw"])] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/danya/university/clown-car/agx_navigation")
    ap.add_argument("--plans", default=None, help="default: first N of tools/jobs/broad40.txt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--soak", default=None, help=f"default: <root>/{RUN}/soak_identity.jsonl")
    ap.add_argument("--trace-dir", default=None, help=f"default: <root>/{RUN}/traces")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="results JSON (tracks in a sibling .npz)")
    a = ap.parse_args()

    plans_file = a.plans or os.path.join(a.root, "tools", "jobs", "broad40.txt")
    names = [os.path.basename(l.strip())[:-4] for l in open(plans_file) if l.strip()][: a.n]
    soak = a.soak or os.path.join(a.root, RUN, "soak_identity.jsonl")
    tdir = a.trace_dir or os.path.join(a.root, RUN, "traces")
    rows = [json.loads(l) for l in open(soak) if l.strip()]
    rows = [r for r in rows if r.get("corrector") == "identity" and r.get("terrain")
            and r.get("seed") == a.seed and "end_pose" in r and "failed" not in r]

    cfg = RLCorrectorConfig()
    out, tracks = [], {}
    for name in names:
        nom = load_recorded(os.path.join(a.root, "traj_data_v2", name + ".npz"))
        plan = np.asarray(nom.poses)
        goal = plan[len(nom)]
        patches = along_path_terrain_sampler(nom.poses)(np.random.default_rng(a.seed))
        sf = slip_model.patches_to_surface_fn(patches)
        gz = [r for r in rows if r["trajectory"] == name]
        if not gz:
            print(f"[g2] {name}: NO Gazebo identity rows", file=sys.stderr)
            continue
        gz_tr = [load_trace(os.path.join(tdir, os.path.basename(r["trace"])))
                 for r in gz if r.get("trace")]
        gz_end = np.array([r["end_pose"][:2] for r in gz])
        gz_lat = [lateral(e, goal) for e in gz_end]
        gz_fe = [r["final_err"] for r in gz]
        gz_mc = [max_cross(t, plan) for t in gz_tr]
        rec = {
            "plan": name, "n_gz": len(gz), "n_gz_traces": len(gz_tr), "patches": patches,
            "gz_final_err_mean": float(np.mean(gz_fe)), "gz_final_err_sd": float(np.std(gz_fe)),
            "gz_lat_mean": float(np.mean(gz_lat)),
            "gz_lat_sign_agree": int(np.sum(np.sign(gz_lat) == np.sign(np.mean(gz_lat)))),
            "gz_max_cross_mean": float(np.mean(gz_mc)) if gz_mc else float("nan"),
            "gz_max_cross_sd": float(np.std(gz_mc)) if gz_mc else float("nan"),
            "gz_end": gz_end.tolist(),
        }
        for i, t in enumerate(gz_tr):
            tracks[f"{name}__gz{i}"] = t
        for tag in MODELS:
            tr, mus = replay(cfg, nom, None if tag == "noslip" else sf,
                             "centre" if tag == "noslip" else tag)
            fe = float(np.hypot(*(tr[-1, :2] - goal[:2])))
            lat = lateral(tr[-1, :2], goal)
            mc = max_cross(tr, plan)
            rec[f"{tag}_final_err"] = fe
            rec[f"{tag}_lat"] = lat
            rec[f"{tag}_dir_agree"] = bool(np.sign(lat) == np.sign(rec["gz_lat_mean"]))
            rec[f"{tag}_fe_ratio"] = fe / rec["gz_final_err_mean"]
            rec[f"{tag}_fe_within2"] = bool(0.5 <= rec[f"{tag}_fe_ratio"] <= 2.0)
            rec[f"{tag}_max_cross"] = mc
            rec[f"{tag}_mc_ratio"] = mc / rec["gz_max_cross_mean"]
            rec[f"{tag}_mc_within2"] = bool(0.5 <= rec[f"{tag}_mc_ratio"] <= 2.0)
            rec[f"{tag}_frac_below_knee"] = float(np.mean(mus <= slip_model.KNEE_MU))
            tracks[f"{name}__{tag}"] = tr
        tracks[f"{name}__plan"] = plan
        out.append(rec)
        print(f"[g2] {name}: gz fe={rec['gz_final_err_mean']:.3f}±{rec['gz_final_err_sd']:.3f} "
              f"lat={rec['gz_lat_mean']:+.3f} ({rec['gz_lat_sign_agree']}/{len(gz)}) "
              f"mc={rec['gz_max_cross_mean']:.3f} | " + " | ".join(
                  f"{t} fe={rec[t + '_final_err']:.3f} lat={rec[t + '_lat']:+.3f} "
                  f"mc={rec[t + '_max_cross']:.3f}" for t in MODELS))

    summary = {}
    for tag in MODELS:
        d = sum(r[f"{tag}_dir_agree"] for r in out)
        f = sum(r[f"{tag}_fe_within2"] for r in out)
        m = sum(r[f"{tag}_mc_within2"] for r in out)
        summary[tag] = {"dir_agree": d, "fe_within2": f, "mc_within2": m, "n": len(out),
                        "pass": d >= 8 and f >= 8}
        print(f"[g2] {tag}: direction {d}/{len(out)}, final_err within x2 {f}/{len(out)}, "
              f"max_cross within x2 {m}/{len(out)} "
              f"-> {'PASS' if summary[tag]['pass'] else 'FAIL'}")
    json.dump({"summary": summary, "plans": out}, open(a.out, "w"), indent=1)
    np.savez(a.out[:-5] + "_tracks.npz", **tracks)


if __name__ == "__main__":
    main()
