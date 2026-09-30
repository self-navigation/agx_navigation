"""Re-join Phase 0 (#11): what individual re-joins look like.

Two figures, both cost B (min effort) with the plan-based guess:

  rejoin_phase0_examples.png   the median problem of four kinds, drawn from
                               rejoin_phase0_plan300.jsonl (see render.py);
  rejoin_phase0_controlled.png one straight stretch of floor_6_v2_00105,
                               changing ONE thing at a time: the window for a
                               fixed 0.3 m sideways offset (top row), then the
                               kind of deviation at a fixed 3 s window (bottom).

Drawing: the robot is drawn in full only at the start (red, where it really
is) and the target (blue outline, where it must be at T_w). In between, the
plan's schedule (grey) and the re-join (green) are lines with a dot every
T_w/4, joined by thin ticks so the schedule gap can be read off; one ghost
footprint at T_w/2 shows the heading. For a failure the solver's last iterate
is not drawn -- it satisfies neither pins nor dynamics and has usually
diverged.

Needs the workspace (scipy + skfmm), like tools/rejoin_phase0.py.
"""
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon

HERE = Path(__file__).parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("p0", REPO / "tools/rejoin_phase0.py")
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)
PlannerConfig, PMPShootingSolver, _build_field, load_occupancy_grid = p0._imports()
COST = "effort"

BODY_L, BODY_W = 0.61, 0.50   # Scout Mini footprint (approx.)
WHEEL_X, WHEEL_Y, WHEEL_L, WHEEL_W = 0.23, 0.21, 0.18, 0.08


def robot(ax, q, color, fill=True, alpha=1.0, ls="-", lw=1.2, z=2):
    """Scout Mini outline at pose q=(x, y, theta): body, 4 wheels, and an
    open chevron at the front for heading (outline only, so it never hides
    what is underneath)."""
    c, s_ = np.cos(q[2]), np.sin(q[2])
    R = np.array([[c, -s_], [s_, c]])

    def rect(cx, cy, L, W):
        p = np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2 + [cx, cy]
        return p @ R.T + q[:2]
    kw = dict(ec=color, lw=lw, ls=ls, zorder=z)
    ax.add_patch(Polygon(rect(0, 0, BODY_L, BODY_W), fc=color if fill else "none",
                         alpha=alpha * (0.35 if fill else 1), **kw))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Polygon(rect(sx * WHEEL_X, sy * WHEEL_Y, WHEEL_L, WHEEL_W),
                                 fc="0.2" if fill else "none", alpha=alpha, **kw))
    nose = np.array([[0.02, -0.14], [BODY_L / 2 - 0.02, 0], [0.02, 0.14]])
    ax.plot(*(nose @ R.T + q[:2]).T, color=color, lw=lw + 0.6, alpha=alpha, zorder=z + .1)


_FIELDS = {}


def setup(plan_name):
    if plan_name not in _FIELDS:
        z = np.load(REPO / "traj_data_v2" / (plan_name + ".npz"))
        my = str(z["map_yaml"])
        my = my if my.startswith("/") else str(REPO / my)
        data, w, h, meta = load_occupancy_grid(my)
        cfg = PlannerConfig(mode="offline")
        field = _build_field(data, w, h, meta["resolution"], meta["origin"],
                             z["start_xy"], z["goal_xy"], 65, cfg)
        plan = np.column_stack([z["poses"], z["wheel_cmds"]])
        goal = np.array([*z["goal_xy"], z["poses"][-1, 2]])
        _FIELDS[plan_name] = (PMPShootingSolver(cfg, field), cfg, plan, goal,
                              np.asarray(z["costates"]), float(z["dt_sample"]))
    return _FIELDS[plan_name]


def draw(ax_xy, ax_w, plan_name, k, m, dev, title):
    solver, cfg, plan, goal, costates, dt = setup(plan_name)
    T_w = m * dt
    x0 = plan[k] + np.asarray(dev, dtype=float)
    x_t = plan[k + m]
    guess = p0._plan_guess(plan, costates, dt, k, m, x0, cfg.N, COST)
    res = solver.solve_rejoin(x0, x_t, T_w, goal, cost=COST, y_guess=guess)
    ok = res["success"]
    t = np.linspace(0, T_w, 200)
    y = res["sol"](t) if ok else None
    ts = np.linspace(0, T_w, 5)
    tp = np.arange(m + 1) * dt
    sched = np.stack([np.interp(ts, tp, np.unwrap(plan[k:k + m + 1, i])) for i in range(3)], 1)

    ax = ax_xy
    lo, hi = max(0, k - m - 15), min(len(plan), k + 2 * m + 15)
    ax.plot(plan[lo:hi, 0], plan[lo:hi, 1], color="0.8", lw=1, zorder=1)
    ax.plot(plan[k:k + m + 1, 0], plan[k:k + m + 1, 1], color="0.4", lw=1.6, ls="--", zorder=2)
    ax.plot(sched[:, 0], sched[:, 1], "o", color="0.4", ms=3.5, zorder=2)
    if ok:
        ax.plot(y[0], y[1], color="tab:green", lw=2.2, zorder=3)
        act = res["sol"](ts)[:3].T
        ax.plot(act[:, 0], act[:, 1], "o", color="tab:green", ms=4, zorder=3)
        for q, sq in zip(act[1:-1], sched[1:-1]):
            ax.plot([q[0], sq[0]], [q[1], sq[1]], color="tab:green", lw=.8, alpha=.7, zorder=2)
        robot(ax, act[2], "tab:green", fill=False, lw=0.9, alpha=.6, z=3)
    else:
        ax.text(.5, .04, "NO SOLUTION", transform=ax.transAxes, ha="center",
                color="tab:red", weight="bold")
    robot(ax, sched[2], "0.4", fill=False, ls=":", lw=0.9, alpha=.6, z=2)
    robot(ax, x_t, "tab:blue", fill=False, lw=2, z=4)
    robot(ax, x0, "tab:red", z=5)
    pts = np.vstack([plan[k:k + m + 1, :2], x0[None, :2]] + ([y[:2].T] if ok else []))
    c, span = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2 + 0.5
    ax.set_xlim(c[0] - span, c[0] + span)
    ax.set_ylim(c[1] - span, c[1] + span)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.grid(alpha=.3)
    ax.set_title(title, fontsize=9)

    if ax_w is not None:
        ax = ax_w
        ax.plot(tp, plan[k:k + m + 1, 3], color="tab:purple", lw=1, ls=":")
        ax.plot(tp, plan[k:k + m + 1, 4], color="tab:brown", lw=1, ls=":")
        if ok:
            ax.plot(t, y[3], color="tab:purple")
            ax.plot(t, y[4], color="tab:brown")
        ax.set_xlabel("t since deviation [s]", fontsize=8)
        ax.set_ylabel("wheel speed [rad/s]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.3)
        ax.set_title(("solved" if ok else res["message"][:45]) + f"; {res['solve_ms']:.0f} ms",
                     fontsize=8)
    return ok


LEGEND = [
    Patch(fc=(0.84, 0.15, 0.16, 0.35), ec="tab:red", label="actual robot at the deviation"),
    Patch(fc="none", ec="tab:blue", lw=2, label="target: plan state at $T_w$"),
    Line2D([], [], color="0.4", ls="--", marker="o", ms=3.5,
           label="where the plan scheduled it (dot every $T_w$/4)"),
    Line2D([], [], color="tab:green", lw=2.2, marker="o", ms=4,
           label="re-join path (same moments; ticks = schedule gap)"),
    Patch(fc="none", ec="0.5", ls=":", label="footprints at $T_w$/2"),
]
WHEEL_LEGEND = [Line2D([], [], color="tab:purple", ls=":", label="plan left"),
                Line2D([], [], color="tab:brown", ls=":", label="plan right"),
                Line2D([], [], color="tab:purple", label="re-join left"),
                Line2D([], [], color="tab:brown", label="re-join right")]


def fig_sampled():
    rows = [json.loads(l) for l in open(HERE / "rejoin_phase0_plan300.jsonl")]
    ok = lambda r: r[COST]["status"] == "ok"
    ex = lambda r: abs(r["e_cross"])

    def pick(pred):
        c = sorted((r for r in rows if pred(r)), key=ex)
        return c[len(c) // 2]  # the median one: representative, not extreme
    cases = [
        ("solved, long window", pick(lambda r: ok(r) and r["T_w"] >= 3 and ex(r) > 0.15)),
        ("solved, 1–3 s", pick(lambda r: ok(r) and 1 <= r["T_w"] < 3 and ex(r) > 0.1)),
        ("failed, 1–3 s", pick(lambda r: not ok(r) and 1 <= r["T_w"] < 3)),
        ("failed, short window", pick(lambda r: not ok(r) and 0.3 <= r["T_w"] < 1)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), gridspec_kw=dict(height_ratios=[1.7, 1]))
    for j, (name, r) in enumerate(cases):
        draw(axes[0, j], axes[1, j], r["plan"], r["k"], r["m"], r["dev"],
             f"{name}: $T_w$={r['T_w']:.1f} s\noff by {r['e_cross']:+.2f} m sideways, "
             f"{r['e_along']:+.2f} m ahead, {np.degrees(r['e_theta']):+.0f}°")
    axes[1, 0].legend(handles=WHEEL_LEGEND, fontsize=7)
    fig.legend(handles=LEGEND, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(.5, .955))
    fig.suptitle("Re-join examples, cost B (min effort), plan-based guess: "
                 "median problem of each kind from the 300")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(HERE / "rejoin_phase0_examples.png", dpi=120)


def fig_controlled():
    """One straight stretch; deviations built in the plan's own frame."""
    name, k = "floor_6_v2_00105", 100
    _, cfg, plan, *_ = setup(name)
    th = plan[k, 2]
    fwd, left = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])

    def dev(side=0.0, ahead=0.0, dth_deg=0.0, wheel_scale=1.0):
        d = np.zeros(5)
        d[:2] = side * left + ahead * fwd
        d[2] = np.radians(dth_deg)
        d[3:5] = plan[k, 3:5] * (wheel_scale - 1)
        return d
    top = [(f"0.3 m to the side, $T_w$={tw:g} s", round(tw / 0.1), dev(side=0.3))
           for tw in (0.5, 1.0, 2.0, 4.0)]
    bot = [
        ("0.5 m BEHIND schedule, 3 s", 30, dev(ahead=-0.5)),
        ("heading 25° off, on the line, 3 s", 30, dev(dth_deg=25)),
        ("wheels 30% slow, on the line, 3 s", 30, dev(wheel_scale=0.7)),
        ("0.3 m side AND facing away 20°, 3 s", 30, dev(side=0.3, dth_deg=20)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10.5))
    for j, (t, m, d) in enumerate(top):
        draw(axes[0, j], None, name, k, m, d, t)
    for j, (t, m, d) in enumerate(bot):
        draw(axes[1, j], None, name, k, m, d, t)
    axes[0, 0].set_ylabel("same offset,\nshrinking window", fontsize=11)
    axes[1, 0].set_ylabel("same 3 s window,\ndifferent deviations", fontsize=11)
    fig.legend(handles=LEGEND, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(.5, .955))
    fig.suptitle(f"Re-join on one straight stretch ({name}, k={k}, "
                 f"cruising ~{cfg.wheels_to_body(*plan[k, 3:5])[0]:.2f} m/s), cost B, plan-based guess")
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(HERE / "rejoin_phase0_controlled.png", dpi=120)


fig_sampled()
fig_controlled()
