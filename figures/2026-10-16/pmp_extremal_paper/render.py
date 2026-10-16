"""Paper figure: one PMP extremal (state and costate against time), Mathematics draft §3.2, 2026-10-16.
Plan f6_medium_0077 from plan set v3, the same plan as figures/2026-10-16/fm2_field_paper.
(a) the plan on the floor map with time marks every 5 s, (b) wheel speeds, (c) position and heading
costates. The three manoeuvres (door 1 turn, door 2 turn, final spin; intervals read off the poses)
are shaded in the same colours on the map and on the time axes.
A panel with a* = a_max tanh(-lambda_w/(gamma a_max)) was dropped: the commands follow it exactly, but
each 0.5 s retained segment restarts the costate, so a* has a sawtooth at every segment boundary that
reads as a numerical defect and adds nothing the text needs.
    .venv/bin/python figures/2026-10-16/pmp_extremal_paper/render.py"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import summarize_compare as sc  # noqa: E402

PLAN = "f6_medium_0077"
EVENTS = [("door 1", 9.0, 14.0, "tab:purple"), ("door 2", 25.5, 33.0, "tab:cyan"), ("spin", 34.3, 37.0, "tab:olive")]
z = np.load(os.path.join(ROOT, "run_data", "2026-10-10_nojob_plan-set-v3", "plans", PLAN + ".npz"),
            allow_pickle=True)
dt = float(z["dt_sample"])
P, W, L = z["poses"], z["wheel_cmds"], z["costates"]
t = np.arange(len(P)) * dt
wall = np.flipud(sc.load_map(6)[0] < 0.1)
extent = [sc.X0, sc.X0 + wall.shape[1] * sc.RES, sc.Y0, sc.Y0 + wall.shape[0] * sc.RES]

plt.rcParams.update({"font.size": 8, "legend.fontsize": 7})
fig = plt.figure(figsize=(7.2, 3.9))
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.55], wspace=0.28, hspace=0.35)
m = fig.add_subplot(gs[:, 0])
ax = [fig.add_subplot(gs[0, 1])]
ax.append(fig.add_subplot(gs[1, 1], sharex=ax[0]))

m.imshow(np.where(wall, 1.0, np.nan), origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1,
         interpolation="nearest")
m.plot(P[:, 0], P[:, 1], "--", color="k", lw=1)
for name, t0, t1, c in EVENTS:
    s = (t >= t0) & (t <= t1)
    m.plot(P[s, 0], P[s, 1], color=c, lw=5, alpha=0.45, solid_capstyle="round")
for tk in range(0, int(t[-1]) + 1, 5):
    i = int(round(tk / dt))
    m.plot(*P[i, :2], "o", color="k", ms=2.5)
    m.annotate(f"{tk}", P[i, :2], xytext=(4, -9), textcoords="offset points", fontsize=7)
m.plot(*P[0, :2], "o", color="tab:green", mec="k", ms=6)
m.plot(*z["goal_xy"], "*", color="tab:red", mec="k", ms=10)
lo, hi = P[:, :2].min(0) - 1.0, P[:, :2].max(0) + 1.0
m.set_xlim(lo[0], hi[0]); m.set_ylim(lo[1], hi[1]); m.set_aspect("equal")
m.set_xlabel("$x$ [m]"); m.set_ylabel("$y$ [m]")

ax[0].plot(t, W[:, 0], color="tab:blue", label="$w_l$")
ax[0].plot(t, W[:, 1], color="tab:orange", label="$w_r$")
ax[0].set_ylabel("wheel speed\n[rad s$^{-1}$]")
ax[1].plot(t, L[:, 0], color="tab:blue", label=r"$\lambda_x$")
ax[1].plot(t, L[:, 1], color="tab:orange", label=r"$\lambda_y$")
ax[1].plot(t, L[:, 2], color="tab:green", label=r"$\lambda_\theta$")
ax[1].set_ylabel("costate"); ax[1].set_xlabel("time [s]")
plt.setp(ax[0].get_xticklabels(), visible=False)
for a in ax:
    for name, t0, t1, c in EVENTS:
        a.axvspan(t0, t1, color=c, alpha=0.18, lw=0)
    a.axhline(0, color="k", lw=0.4)
ax[0].legend(loc="lower center", bbox_to_anchor=(0.53, 0.02), ncol=2, framealpha=.9)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.53, 0.98), ncol=3, framealpha=.9)
for name, t0, t1, c in EVENTS:
    ax[0].annotate(name, ((t0 + t1) / 2, 1.02), xycoords=("data", "axes fraction"), ha="center", fontsize=7,
                   color=c)
for a, tag in zip([m, *ax], "abc"):
    a.set_title(f"({tag})", loc="left", fontweight="bold", pad=10)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"pmp_extremal.{ext}"), dpi=250, bbox_inches="tight")
print(PLAN, "duration", t[-1])
