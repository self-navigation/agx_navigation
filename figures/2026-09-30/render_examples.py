"""Re-join Phase 0 (#11): what individual re-joins look like.

Picks problems from rejoin_phase0_plan300.jsonl (see render.py), re-solves
them in-process with the same guess, and draws the path (top) and the left /
right wheel speeds (bottom) against the nominal plan. For a failure the dashed
curve is the solver's LAST ITERATE: it satisfies neither the pins nor the
dynamics, and only shows where the solver gave up.

Needs the workspace (scipy + skfmm), like tools/rejoin_phase0.py.
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("p0", REPO / "tools/rejoin_phase0.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)
PlannerConfig, PMPShootingSolver, _build_field, load_occupancy_grid = p0._imports()

rows = {r["id"]: r for r in map(json.loads, open(HERE / "rejoin_phase0_plan300.jsonl"))}
COST = "effort"


def ok(r):
    return r[COST]["status"] == "ok"


def pick(pred, key):
    c = sorted((r for r in rows.values() if pred(r)), key=key)
    return c[len(c) // 2] if c else None  # the median one: representative, not extreme


exy = lambda r: abs(r["e_cross"])
CASES = [
    ("solved, long window", pick(lambda r: ok(r) and r["T_w"] >= 3 and exy(r) > 0.15, exy)),
    ("solved, 1–3 s", pick(lambda r: ok(r) and 1 <= r["T_w"] < 3 and exy(r) > 0.1, exy)),
    ("failed, 1–3 s", pick(lambda r: not ok(r) and 1 <= r["T_w"] < 3, exy)),
    ("failed, short window", pick(lambda r: not ok(r) and 0.3 <= r["T_w"] < 1, exy)),
]

BODY_L, BODY_W = 0.61, 0.50   # Scout Mini footprint (approx.)
WHEEL_X, WHEEL_Y, WHEEL_L, WHEEL_W = 0.23, 0.21, 0.18, 0.08


def robot(ax, q, color, fill=True, alpha=1.0, ls="-", lw=1.2, z=2):
    """Scout Mini outline at pose q=(x, y, theta): body, 4 wheels, a nose
    triangle for heading."""
    from matplotlib.patches import Polygon
    c, s_ = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s_], [s_, c]])

    def rect(cx, cy, L, W):
        p = np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2 + [cx, cy]
        return p @ R.T + q[:2]
    kw = dict(ec=color, lw=lw, ls=ls, alpha=alpha, zorder=z)
    ax.add_patch(Polygon(rect(0, 0, BODY_L, BODY_W), fc=color if fill else "none", **kw))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Polygon(rect(sx * WHEEL_X, sy * WHEEL_Y, WHEEL_L, WHEEL_W),
                                 fc="0.15" if fill else "none", **kw))
    nose = np.array([[BODY_L / 2 - 0.12, -0.1], [BODY_L / 2, 0], [BODY_L / 2 - 0.12, 0.1]])
    ax.add_patch(Polygon(nose @ R.T + q[:2], fc="white" if fill else color, ec=color,
                         alpha=alpha, zorder=z + .1))


fig, axes = plt.subplots(2, len(CASES), figsize=(5.2 * len(CASES), 9.5),
                         gridspec_kw=dict(height_ratios=[1.6, 1]))
for j, (title, r) in enumerate(CASES):
    z = np.load(REPO / "traj_data_v2" / (r["plan"] + ".npz"))
    my = str(z["map_yaml"])
    my = my if my.startswith("/") else str(REPO / my)
    data, w, h, meta = load_occupancy_grid(my)
    cfg = PlannerConfig(mode="offline")
    field = _build_field(data, w, h, meta["resolution"], meta["origin"],
                         z["start_xy"], z["goal_xy"], 65, cfg)
    solver = PMPShootingSolver(cfg, field)
    plan = np.column_stack([z["poses"], z["wheel_cmds"]])
    dt = float(z["dt_sample"])
    k, m = r["k"], r["m"]
    x0 = plan[k] + np.array(r["dev"])
    x_t = plan[k + m]
    goal = np.array([*z["goal_xy"], z["poses"][-1, 2]])
    guess = p0._plan_guess(plan, np.asarray(z["costates"]), dt, k, m, x0, cfg.N, COST)
    res = solver.solve_rejoin(x0, x_t, r["T_w"], goal, cost=COST, y_guess=guess)
    fn = res.get("sol") or res.get("sol_last")
    t = np.linspace(0, r["T_w"], 200)
    y = fn(t) if fn is not None else None

    ax = axes[0, j]
    lo, hi = max(0, k - 2 * m - 10), min(len(plan), k + 3 * m + 10)
    ax.plot(plan[lo:hi, 0], plan[lo:hi, 1], color="0.75", lw=1, zorder=1)
    ax.plot(plan[k:k + m + 1, 0], plan[k:k + m + 1, 1], color="0.35", lw=1.5, ls=":", zorder=1)
    ts = np.linspace(0, r["T_w"], 5)
    sched = np.stack([np.interp(ts, np.arange(m + 1) * dt, np.unwrap(plan[k:k + m + 1, i]))
                      for i in range(3)], 1)
    for i, (q, tt) in enumerate(zip(sched, ts)):
        robot(ax, q, "0.45", fill=False, ls="--", z=2)
        ax.annotate(f"{tt:.1f}s", q[:2], xytext=(4, -10), textcoords="offset points",
                    fontsize=7, color="0.35")
    if res["success"]:
        ax.plot(y[0], y[1], color="tab:green", lw=2, zorder=3)
        act = fn(ts)[:3].T
        for q, sq in zip(act[1:-1], sched[1:-1]):
            robot(ax, q, "tab:green", alpha=.35, z=3)
            ax.plot([q[0], sq[0]], [q[1], sq[1]], color="tab:green", lw=.7, alpha=.6)
    else:
        ax.text(.5, .04, "NO SOLUTION FOUND", transform=ax.transAxes, ha="center",
                color="tab:red", weight="bold")
    robot(ax, x0, "tab:red", alpha=.8, z=4)
    ax.plot([x0[0], plan[k, 0]], [x0[1], plan[k, 1]], color="tab:red", lw=1)
    robot(ax, x_t, "tab:blue", fill=False, lw=2.2, z=4)
    # Frame the plan window and the robots, never a diverged iterate.
    pts = np.vstack([plan[k:k + m + 1, :2], x0[None, :2]])
    if res["success"]:
        pts = np.vstack([pts, y[:2].T])
    c, span = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 + 0.6
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_aspect("equal")
    ax.grid(alpha=.3)
    ax.set_title(f"{title}: $T_w$={r['T_w']:.1f} s to re-join\n"
                 f"off by {r['e_cross']:+.2f} m sideways, {r['e_along']:+.2f} m ahead, "
                 f"{np.degrees(r['e_theta']):+.0f}°", fontsize=9)

    ax = axes[1, j]
    tp = np.arange(m + 1) * dt
    ax.plot(tp, plan[k:k + m + 1, 3], color="tab:purple", lw=1, ls=":", label="plan left")
    ax.plot(tp, plan[k:k + m + 1, 4], color="tab:brown", lw=1, ls=":", label="plan right")
    if y is not None and res["success"]:
        sty = "-" if res["success"] else "--"
        ax.plot(t, y[3], color="tab:purple", ls=sty, label="re-join left")
        ax.plot(t, y[4], color="tab:brown", ls=sty, label="re-join right")
    ax.set_xlabel("t since deviation [s]")
    ax.set_ylabel("wheel speed [rad/s]")
    ax.grid(alpha=.3)
    msg = "solved" if res["success"] else res["message"][:45]
    ax.set_title(f"{msg}; {res['solve_ms']:.0f} ms", fontsize=8)
    if j == 0:
        ax.legend(fontsize=7)
    print(title, r["id"], res["status"], flush=True)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(fc="tab:red", ec="tab:red", alpha=.8, label="where the robot really is (start)"),
    Patch(fc="none", ec="0.45", ls="--", label="where the plan scheduled it (at 0, ¼, ½, ¾, 1 of $T_w$)"),
    Patch(fc="tab:green", ec="tab:green", alpha=.35, label="re-join solution at the same moments"),
    Patch(fc="none", ec="tab:blue", lw=2.2, label="where it must be at $T_w$ (target)"),
    Line2D([], [], color="0.75", label="rest of the nominal plan"),
], loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(.5, .955))
fig.suptitle("Re-join examples, cost B (min effort), plan-based guess — median case of each kind")
fig.tight_layout(rect=(0, 0, 1, .93))
fig.savefig(HERE / "rejoin_phase0_examples.png", dpi=120)
