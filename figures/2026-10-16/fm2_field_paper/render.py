"""Paper figure for the FM2 global planner (Mathematics draft, §2.2), 2026-10-16.
(a) wave speed v_w on the baked floor-6 map, (b) arrival time T with streamlines of F = -grad T,
both with one evaluation plan (plan set v3) drawn over them. The field is recomputed as the
stack configures it (exponential profile, inflation 0.5 m), as in figures/2026-10-02/render_field.py.
    .venv/bin/python figures/2026-10-16/fm2_field_paper/render.py [plan_id]"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src", "agx_navigation", "agx_planning"))
import summarize_compare as sc  # noqa: E402
from agx_planning.vector_field.field import VectorFieldConfig, build_speed_field, compute_field  # noqa: E402

PLAN = sys.argv[1] if len(sys.argv) > 1 else "f6_medium_0077"
PLANS = os.path.join(ROOT, "run_data", "2026-10-10_nojob_plan-set-v3", "plans")
RES, X0, Y0 = sc.RES, sc.X0, sc.Y0
CFG = VectorFieldConfig(speed_profile="exponential", inflation_radius=0.5,
                        smooth_T_before_grad=False, early_exit_enable=False)

img = sc.load_map(6)[0]
occ = np.zeros(img.shape, np.int8)
occ[img < 0.1] = 100
occ[(img > 0.7) & (img < 0.9)] = -1
occ = np.flipud(occ)
wall = occ >= 65
z = np.load(os.path.join(PLANS, PLAN + ".npz"), allow_pickle=True)
pp = z["poses"]
gx, gy = z["goal_xy"]
res, _ = compute_field(occ, int((gx - X0) / RES), int((gy - Y0) / RES), RES, X0, Y0, CFG,
                       allow_unknown=True)
speed = build_speed_field(distance_transform_edt(~wall) * RES, wall, CFG)

lo, hi = pp[:, :2].min(0) - 1.5, pp[:, :2].max(0) + 1.5
extent = [X0, X0 + occ.shape[1] * RES, Y0, Y0 + occ.shape[0] * RES]
xs = X0 + (np.arange(occ.shape[1]) + 0.5) * RES
ys = Y0 + (np.arange(occ.shape[0]) + 0.5) * RES
c0, c1 = int((lo[0] - X0) / RES), int((hi[0] - X0) / RES)
r0, r1 = int((lo[1] - Y0) / RES), int((hi[1] - Y0) / RES)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8})
w = (hi[0] - lo[0]) / (hi[1] - lo[1])
fig, ax = plt.subplots(1, 2, figsize=(7.0, 7.0 / (2 * w) + 0.9), sharey=True)
walls = np.where(wall, 1.0, np.nan)


def decorate(a, tag):
    a.imshow(walls, origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
    a.plot(pp[:, 0], pp[:, 1], "--", color="w", lw=2.6)
    a.plot(pp[:, 0], pp[:, 1], "--", color="k", lw=1.3)
    a.plot(*pp[0, :2], "o", color="tab:green", mec="k", ms=6)
    a.plot(gx, gy, "*", color="tab:red", mec="k", ms=11)
    a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_aspect("equal")
    a.set_xlabel("$x$ [m]"); a.set_title(tag, loc="left", fontweight="bold")


im0 = ax[0].imshow(np.where(wall, np.nan, speed), origin="lower", extent=extent, cmap="viridis",
                   vmin=0, vmax=float(np.nanmax(speed)))
decorate(ax[0], "(a)")
ax[0].set_ylabel("$y$ [m]")
fig.colorbar(im0, ax=ax[0], orientation="horizontal", fraction=0.05, pad=0.13, label="wave speed $v_w$")

T = res.travel_time
im1 = ax[1].imshow(T, origin="lower", extent=extent, cmap="magma_r", vmin=0,
                   vmax=float(np.nanmax(T[r0:r1, c0:c1])))
ax[1].contour(xs[c0:c1], ys[r0:r1], T[r0:r1, c0:c1], levels=12, colors="w", linewidths=0.4, alpha=0.7)
k = 3
gxs, gys = res.grad_x[r0:r1:k, c0:c1:k], res.grad_y[r0:r1:k, c0:c1:k]
ok = np.isfinite(T[r0:r1:k, c0:c1:k]) & ~wall[r0:r1:k, c0:c1:k]
ax[1].streamplot(xs[c0:c1:k], ys[r0:r1:k], np.where(ok, gxs, np.nan), np.where(ok, gys, np.nan),
                 color=(0.2, 0.6, 1.0), density=1.4, linewidth=0.6, arrowsize=0.6)
decorate(ax[1], "(b)")
fig.colorbar(im1, ax=ax[1], orientation="horizontal", fraction=0.05, pad=0.13, label="arrival time $T$")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"fm2_field.{ext}"), dpi=250, bbox_inches="tight")
print(PLAN, "length", float(np.sum(np.hypot(*np.diff(pp[:, :2], axis=0).T))), "speed range",
      float(np.nanmin(speed[~wall])), float(np.nanmax(speed)))
