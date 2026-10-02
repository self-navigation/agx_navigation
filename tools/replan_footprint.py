#!/usr/bin/env python3
"""Replan offline with the footprint barrier at several weights (#34).

No Gazebo, no ROS: the stack's field + planner params (bench_pmp_rollout's
--stack-field), a full offline rollout per (case, w_fp), and a TRUE-footprint
check of every planned pose against the UNsmoothed signed wall distance.

Cases:
  broad   the 40 broad v2 plans (floor 6), start/goal/initial yaw from the npz
  doors   every passage narrower than --door-max on --door-floors, crossed
          from 1.5 m before to 1.5 m after, starting yawed --door-yaw off the
          door axis so the plan has to square up

Writes one JSON line per rollout to --out and the planned poses to
<out>.poses/<case>__wfp<w>.npy:

  .venv/bin/python tools/replan_footprint.py --out /tmp/fp.jsonl --w-fp 0 20 100
  # on the VM, in parallel:  ... --jobs 24
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
from bench_pmp_rollout import STACK_PLANNER_PARAMS  # noqa: E402
from rejoin_phase0 import _imports  # noqa: E402  (sets sys.path too)

MAPS = os.path.join(REPO, "src", "rudn-ordjo-building", "maps")
CHECK_SPACING = 0.02  # [m] outline sampling for the overlap CHECK (finer than the cost's)


def load_floor(floor):
    _, _, _, load_occupancy_grid = _imports()
    data, w, h, meta = load_occupancy_grid(os.path.join(MAPS, f"floor_{floor}.yaml"))
    return np.asarray(data, dtype=np.int8).reshape(h, w), meta


def door_cases(floors, max_width, yaw_off):
    """Start/goal pairs through every narrow passage (saddle of the EDT)."""
    from scipy import ndimage
    out = []
    for fl in floors:
        arr, meta = load_floor(fl)
        r, (ox, oy) = meta["resolution"], meta["origin"][:2]
        free = arr == 0
        edt = ndimage.distance_transform_edt(free) * r
        mxx = ndimage.maximum_filter1d(edt, 3, axis=1)
        mxy = ndimage.maximum_filter1d(edt, 3, axis=0)
        mnx = ndimage.minimum_filter1d(edt, 9, axis=1)
        mny = ndimage.minimum_filter1d(edt, 9, axis=0)
        narrow = free & (edt < max_width / 2) & (edt > 0.2)
        across_x = narrow & (edt >= mxx - 1e-9) & (edt <= mny + 1e-9)  # door axis = y
        across_y = narrow & (edt >= mxy - 1e-9) & (edt <= mnx + 1e-9)  # door axis = x
        for mask, axis in ((across_x, (0.0, 1.0)), (across_y, (1.0, 0.0))):
            lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
            for i in range(1, n + 1):
                rows, cols = np.nonzero(lab == i)
                k = np.argmin(edt[rows, cols])
                cx, cy = ox + (cols[k] + 0.5) * r, oy + (rows[k] + 0.5) * r
                ax = np.array(axis)
                s, g = np.array([cx, cy]) - 1.5 * ax, np.array([cx, cy]) + 1.5 * ax

                def clear(p):
                    c, rr = int((p[0] - ox) / r), int((p[1] - oy) / r)
                    return 0 <= rr < edt.shape[0] and 0 <= c < edt.shape[1] and edt[rr, c] >= 0.45
                if not (clear(s) and clear(g)):
                    continue
                yaw = np.arctan2(ax[1], ax[0])
                out.append(dict(case=f"door_f{fl}_{int(cx * 100)}_{int(cy * 100)}", floor=fl,
                                start=s.tolist(), goal=g.tolist(), yaw0=float(yaw + yaw_off),
                                goal_yaw=float(yaw), width=round(2 * float(edt[rows[k], cols[k]]), 2),
                                door=[cx, cy], axis=list(axis)))
    return out


def broad_cases(list_path):
    out = []
    for ln in open(list_path):
        p = ln.strip().replace("__HOME__", os.path.expanduser("~"))
        if not p:
            continue
        if not os.path.exists(p):
            p = os.path.join(REPO, "traj_data_v2", os.path.basename(p))
        z = np.load(p)
        out.append(dict(case=os.path.basename(p)[:-4], floor=6,
                        start=z["start_xy"].tolist(), goal=z["goal_xy"].tolist(),
                        yaw0=float(z["poses"][0, 2]), goal_yaw=float(z["poses"][-1, 2])))
    return out


def run_one(job):
    case, w_fp, out_dir = job
    PlannerConfig, PMPShootingSolver, _, _ = _imports()
    from agx_planning.pmp_planner import rollout as ro
    from agx_planning.pmp_planner.shooting_solver import _outline
    from agx_planning.vector_field import VectorFieldConfig, VectorFieldGrid, compute_field

    arr, meta = load_floor(case["floor"])
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    cfg = PlannerConfig(mode="offline")
    for k, v in STACK_PLANNER_PARAMS.items():
        setattr(cfg, k, v)
    cfg.w_fp = w_fp
    g = lambda p: (int((p[0] - ox) / r), int((p[1] - oy) / r))  # noqa: E731
    gc, sc = g(case["goal"]), g(case["start"])
    res, _ = compute_field(arr, gc[0], gc[1], r, ox, oy,
                           VectorFieldConfig(speed_profile="exponential", inflation_radius=0.5),
                           sc[0], sc[1], 65, allow_unknown=True)
    field = VectorFieldGrid()
    field.update(res.travel_time, ox, oy, r, field_eps=cfg.field_eps,
                 wall_dist=res.wall_dist, wall_sigma=cfg.fp_smooth_sigma)
    check = VectorFieldGrid()  # exact walls, no blur, for the verdict
    check.update(res.travel_time, ox, oy, r, wall_dist=res.wall_dist, wall_sigma=0.0)

    solver = PMPShootingSolver(cfg, field)
    x0 = np.array([*case["start"], case["yaw0"], 0.0, 0.0])
    goal = np.array([*case["goal"], case["goal_yaw"]])
    t0 = time.perf_counter()
    poses, cmds, lams, status, msg = [], [], [], "?", ""
    gen = ro.rollout_generator(solver, cfg, x0.copy(), goal)
    try:
        while True:
            ch = next(gen)
            poses.append(ch.poses)
            cmds.append(ch.wheel_cmds)
            lams.append(ch.costates)
    except StopIteration as stop:
        status, msg = getattr(stop.value, "status", "?"), getattr(stop.value, "message", "")
    except Exception as e:  # a crash is a failure to count, not to abort the sweep
        status, msg = "exception", repr(e)
    wall = time.perf_counter() - t0
    P = np.concatenate(poses) if poses else np.zeros((0, 3))

    rec = dict(case=case["case"], floor=case["floor"], w_fp=w_fp, status=status,
               message=msg[:200], wall_s=round(wall, 1), n=int(len(P)))
    if len(P):
        out = _outline(cfg.fp_half_length, cfg.fp_half_width, CHECK_SPACING)
        c, s = np.cos(P[:, 2])[:, None], np.sin(P[:, 2])[:, None]
        qx = P[:, 0:1] + out[None, :, 0] * c - out[None, :, 1] * s
        qy = P[:, 1:2] + out[None, :, 0] * s + out[None, :, 1] * c
        d = check.query_dist(qx.ravel(), qy.ravel())[0].reshape(qx.shape).min(axis=1)
        rec.update(overlap_poses=int((d < 0).sum()), min_fp_clear=round(float(d.min()), 3),
                   final_err=round(float(np.hypot(*(P[-1, :2] - goal[:2]))), 3),
                   duration_s=round(len(P) / cfg.control_rate, 2),
                   path_len=round(float(np.hypot(*np.diff(P[:, :2], axis=0).T).sum()), 2))
        if "door" in case:
            near = np.hypot(P[:, 0] - case["door"][0], P[:, 1] - case["door"][1]) < 0.4
            if near.any():
                ax_yaw = np.arctan2(case["axis"][1], case["axis"][0])
                err = np.abs((P[near, 2] - ax_yaw + np.pi / 2) % np.pi - np.pi / 2)
                rec["door_yaw_err_deg"] = round(float(np.degrees(err.max())), 1)
            rec["width"] = case["width"]
        np.save(os.path.join(out_dir, f"{case['case']}__wfp{w_fp:g}.npy"), P)
        # The same plan in traj_data_v2's schema, so the soak bench can drive
        # exactly what the stack would have (the library plans differ, #34).
        np.savez(os.path.join(out_dir, f"{case['case']}__wfp{w_fp:g}.npz"),
                 poses=P, wheel_cmds=np.concatenate(cmds), costates=np.concatenate(lams),
                 dt_sample=1.0 / cfg.control_rate,
                 map_yaml=os.path.join(MAPS, f"floor_{case['floor']}.yaml"),
                 start_xy=np.asarray(case["start"]), goal_xy=goal[:2],
                 shape=f"stack_replan_wfp{w_fp:g}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-fp", type=float, nargs="+", default=[0.0, 20.0, 100.0])
    ap.add_argument("--broad", default=os.path.join(REPO, "tools", "jobs", "broad40.txt"))
    ap.add_argument("--door-floors", type=int, nargs="*", default=[2, 3])
    ap.add_argument("--door-max", type=float, default=0.95)
    ap.add_argument("--door-yaw", type=float, default=np.radians(45))
    ap.add_argument("--only", default="", help="substring filter on case names")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()

    cases = broad_cases(a.broad) if a.broad else []
    cases += door_cases(a.door_floors, a.door_max, a.door_yaw)
    if a.only:
        cases = [c for c in cases if a.only in c["case"]]
    out_dir = a.out + ".poses"
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(c, w, out_dir) for c in cases for w in a.w_fp]
    print(f"{len(cases)} cases x {len(a.w_fp)} weights = {len(jobs)} rollouts", flush=True)
    with open(a.out, "a") as fh, mp.Pool(a.jobs) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_one, jobs), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{i}/{len(jobs)}] {rec['case']} w_fp={rec['w_fp']:g} {rec['status']} "
                  f"overlap={rec.get('overlap_poses')} yaw_err={rec.get('door_yaw_err_deg')} "
                  f"{rec['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
