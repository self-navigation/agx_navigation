#!/usr/bin/env python3
"""Bottleneck alignment (stop-and-aim at doors, #50): the field before and after,
what a simple gradient follower does with it, and what the PMP planner does.

    .venv/bin/python figures/2026-10-08/bottleneck_align/render.py [--plans 00355,00443,00428] [--jobs 3]

Field: the centred library's recipe (#48 c_r15 = tools/plan_opt.py ENQUEUE[1],
stack planner params, allow_unknown=True), with VectorFieldConfig.bottleneck_align
off ("base") and on ("align"). PMP rollouts are cached in
run_data/2026-10-08_nojob_bottleneck-align/poses/<plan>__<variant>.npz.

The "gate follower" is the planner's own heading rule with no optimisation:
a unicycle that turns towards F at up to omega_max and drives at
v_max * gate(F . h), gate = ((1 + x) / 2) ** align_gate_power (the stack's 15).
It shows what the field asks for: where it stops, where it turns in place.
"""
import argparse
import multiprocessing as mp
import os
import sys

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning"),
                os.path.join(R, "src/rudn-ordjo-building")]
OUT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(R, "run_data/2026-10-08_nojob_bottleneck-align")
LIB = os.path.join(R, "run_data/2026-10-07_nojob_centred-library/traj_data_v2c")
FIELD = dict(speed_profile="exponential", inflation_radius=0.5, centring=True, centring_cap=0.8,
             speed_decay_rate=1.5, speed_v_min=0.1)
VARIANTS = {"base": dict(FIELD), "align": dict(FIELD, bottleneck_align=True),
            "sharp": dict(FIELD, bottleneck_align=True, bottleneck_converge_dist=0.07)}


TITLES = {"base": "FM2 field", "align": "aligned, converge 0.15 m", "sharp": "aligned, converge 0.07 m"}


def case(plan):
    z = np.load(os.path.join(LIB, f"floor_6_v2_{plan}.npz"))
    return dict(plan=plan, start=z["start_xy"], goal=z["goal_xy"], yaw0=float(z["poses"][0, 2]),
                goal_yaw=float(z["poses"][-1, 2]), lib=z["poses"])


def field_for(c, variant):
    from replan_footprint import load_floor
    from agx_planning.vector_field import VectorFieldConfig, compute_field
    arr, meta = load_floor(6)
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    g = lambda p: (int((p[0] - ox) / r), int((p[1] - oy) / r))  # noqa: E731
    gc, sc = g(c["goal"]), g(c["start"])
    res, msg = compute_field(arr, gc[0], gc[1], r, ox, oy, VectorFieldConfig(**VARIANTS[variant]),
                             sc[0], sc[1], 65, allow_unknown=True)
    return res, arr, meta


def solve(job):
    """One offline PMP rollout, the plan_opt / #48 recipe."""
    plan, variant = job
    path = os.path.join(DATA, "poses", f"{plan}__{variant}.npz")
    if os.path.exists(path):
        return path
    from bench_pmp_rollout import STACK_PLANNER_PARAMS
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner import rollout as ro
    from agx_planning.vector_field.field import field_result_to_grid
    c = case(plan)
    res, _, _ = field_for(c, variant)
    cfg = PlannerConfig(mode="offline")
    for k, v in STACK_PLANNER_PARAMS.items():
        setattr(cfg, k, v)
    grid = field_result_to_grid(res, field_eps=cfg.field_eps, wall_sigma=cfg.fp_smooth_sigma)
    solver = PMPShootingSolver(cfg, grid)
    solver.reset_warm_start()
    x0 = np.array([*c["start"], c["yaw0"], 0.0, 0.0])
    goal = np.array([*c["goal"], c["goal_yaw"]])
    poses, cmds, status = [], [], "?"
    gen = ro.rollout_generator(solver, cfg, x0.copy(), goal)
    try:
        while True:
            ch = next(gen)
            poses.append(ch.poses)
            if getattr(ch, "wheel_cmds", None) is not None:
                cmds.append(ch.wheel_cmds)
    except StopIteration as stop:
        status = str(getattr(stop.value, "status", "?"))
    except Exception as e:  # noqa: BLE001 -- a crashed solve is a failed solve
        status = "exception:" + repr(e)[:120]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, poses=np.concatenate(poses) if poses else np.zeros((0, 3)),
             wheel_cmds=np.concatenate(cmds) if cmds else np.zeros((0, 4)), status=status,
             start_xy=c["start"], goal_xy=c["goal"], dt_sample=1.0 / cfg.control_rate)
    return path


def gate_follower(grid, x, y, th, goal, v_max=0.448, w_max=1.049, power=15.0, dt=0.05, t_max=40.0):
    out = [(x, y, th, 0.0)]
    for _ in range(int(t_max / dt)):
        _, _, _, fx, fy = grid.query_scalar(x, y)
        m = np.hypot(fx, fy)
        if m < 1e-3 or np.hypot(x - goal[0], y - goal[1]) < 0.15:
            break
        err = np.arctan2(fy, fx) - th
        err = (err + np.pi) % (2 * np.pi) - np.pi
        w = np.clip(3.0 * err, -w_max, w_max)
        v = v_max * ((1 + np.cos(err)) / 2) ** power
        x, y, th = x + v * np.cos(th) * dt, y + v * np.sin(th) * dt, th + w * dt
        out.append((x, y, th, v))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default="00355,00443,00428")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    plans = a.plans.split(",")
    jobs = [(p, v) for p in plans for v in VARIANTS]
    with mp.Pool(a.jobs) as pool:
        for path in pool.imap_unordered(solve, jobs):
            z = np.load(path)
            print(os.path.basename(path), str(z["status"]), len(z["poses"]), flush=True)
    render(plans)


def render(plans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from agx_planning.vector_field.field import field_result_to_grid

    # --- overview: every bottleneck the planner's map has ----------------------
    c0 = case(plans[0])
    res, arr, meta = field_for(c0, "align")
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    EXT = [ox, ox + arr.shape[1] * r, oy, oy + arr.shape[0] * r]
    wall = arr >= 65
    from agx_planning.vector_field.bottleneck import find_bottlenecks
    allb = find_bottlenecks(wall, r, ox, oy)
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.imshow(np.where(wall, 0.0, np.where(arr < 0, 0.75, 1.0)), cmap="gray", origin="lower",
              extent=EXT, vmin=0, vmax=1)
    for b in allb:
        ax.plot([b.x - 0.6 * b.ax, b.x + 0.6 * b.ax], [b.y - 0.6 * b.ay, b.y + 0.6 * b.ay],
                c=plt.cm.plasma((b.width - 0.75) / 0.45), lw=3)
    ax.set_xlim(-24, 31); ax.set_ylim(-12, 27); ax.set_aspect("equal")
    ax.set_title(f"{len(allb)} bottlenecks found from the map alone (EDT saddles, opening < 1.2 m), "
                 "drawn along the passage axis; colour = width 0.75 (dark) .. 1.2 m (yellow). "
                 "Grey = unknown, which the planner treats as free.", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "bottlenecks_floor6.png"), dpi=90); plt.close(fig)

    # --- per plan: the doors its base plan passes, before / after -------------
    stats = []
    for p in plans:
        c = case(p)
        F = {v: field_for(c, v)[0] for v in VARIANTS}
        G = {v: field_result_to_grid(F[v]) for v in VARIANTS}
        P = {v: np.load(os.path.join(DATA, "poses", f"{p}__{v}.npz")) for v in VARIANTS}
        base_path = P["base"]["poses"]
        doors = []
        for b, sgn in (F["align"].bottlenecks or []):
            d = np.hypot(base_path[:, 0] - b.x, base_path[:, 1] - b.y)
            if d.min() < 0.4:
                doors.append((int(np.argmin(d)), b, sgn))
        doors.sort(key=lambda t: t[0])
        doors = doors[:3]
        if not doors:
            print(p, "passes no bottleneck"); continue
        fig, axs = plt.subplots(len(doors), len(VARIANTS), figsize=(6.5 * len(VARIANTS), 6.2 * len(doors)), squeeze=False)
        for i, (k, b, sgn) in enumerate(doors):
            for j, v in enumerate(VARIANTS):
                ax = axs[i, j]
                g = G[v]
                ax.imshow(np.where(wall, 0.0, np.where(arr < 0, 0.75, 1.0)), cmap="gray",
                          origin="lower", extent=EXT, vmin=0, vmax=1)
                L = 2.2
                xs = np.arange(b.x - L, b.x + L, 0.15); ys = np.arange(b.y - L, b.y + L, 0.15)
                XX, YY = np.meshgrid(xs, ys)
                _, _, _, fx, fy = g.query_vec(XX.ravel(), YY.ravel())
                fr = ~wall[((YY.ravel() - oy) / r).astype(int), ((XX.ravel() - ox) / r).astype(int)]
                ax.quiver(XX.ravel()[fr], YY.ravel()[fr], fx[fr], fy[fr], color="0.55", scale=40,
                          width=0.0025, headwidth=3)
                # Gate followers from the base plan's pose 2.5 / 3.5 s before the door,
                # and from points across the corridor at the same distance.
                for back in (25, 35):
                    kk = max(0, k - back)
                    x, y, th = base_path[kk]
                    tr = gate_follower(g, x, y, th, c["goal"], t_max=12.0)
                    sc = ax.scatter(tr[::2, 0], tr[::2, 1], c=tr[::2, 3], cmap="viridis_r",
                                    vmin=0, vmax=0.45, s=5, zorder=3)
                pp = P[v]["poses"]
                ax.plot(pp[:, 0], pp[:, 1], c="tab:red", lw=2, label=f"PMP ({v})", zorder=4)
                if v != "base":
                    pb = P["base"]["poses"]
                    ax.plot(pb[:, 0], pb[:, 1], c="tab:red", lw=1, ls=":", label="PMP (base)", zorder=4)
                ax.plot([b.x - 0.5 * sgn * b.ax, b.x + 0.5 * sgn * b.ax],
                        [b.y - 0.5 * sgn * b.ay, b.y + 0.5 * sgn * b.ay], c="m", lw=1)
                ax.set_xlim(b.x - L, b.x + L); ax.set_ylim(b.y - L, b.y + L); ax.set_aspect("equal")
                ax.set_title(f"{p} door {i + 1} ({b.width:.2f} m): {TITLES[v]}",
                             fontsize=9)
                if i == 0:
                    ax.legend(fontsize=7, loc="lower left")
        fig.colorbar(sc, ax=axs, shrink=0.5, label="gate follower speed [m/s] (dark = stopped)")
        fig.suptitle(f"{p}: grey arrows = field direction F; dots = gate follower (turns towards F, speed = "
                     "v_max * gate); red = PMP plan", fontsize=10)
        fig.savefig(os.path.join(OUT, f"doors_{p}.png"), dpi=80, bbox_inches="tight"); plt.close(fig)
        for v in VARIANTS:
            pp = P[v]["poses"]
            dt = float(P[v]["dt_sample"])
            vv = np.r_[0, np.hypot(*np.diff(pp[:, :2], axis=0).T)] / dt
            ww = np.r_[0, np.diff(np.unwrap(pp[:, 2]))] / dt
            # Lateral acceleration v|w| in the 2.5 s before / 0.5 s after each door,
            # door index taken on THIS variant's path.
            at = []
            for _, b, _ in doors:
                kk = int(np.argmin(np.hypot(pp[:, 0] - b.x, pp[:, 1] - b.y)))
                at.append(np.max((vv * np.abs(ww))[max(0, kk - 25):kk + 5]))
            stats.append(f"{p} {v:5s} status {str(P[v]['status']):12s} duration {len(pp) * dt:6.1f} s  "
                         f"max v {vv.max():.2f}  max v|w| {np.max(vv * np.abs(ww)):.2f}  "
                         f"v|w| at the doors {' '.join(f'{x:.2f}' for x in at)}")
    open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(stats) + "\n")
    print("\n".join(stats))


if __name__ == "__main__":
    main()
