"""FM2 field around the plans our arm crashed on (#34), 2026-10-02.

    .venv/bin/python figures/2026-10-02/render_field.py

Recomputes the vector field exactly as vec_pmp.launch.py configures it
(exponential profile, inflation_radius 0.5 m, v_min 0.1) on the baked floor-6
map, for the v7 plan's goal. Per plan, one panel:
  background  wave speed v(x) -- what FM2 trades against path length
  arrows      the unit field -grad T
  contours    centre-to-wall distance = 0.29 m (inscribed: the robot's SIDE
              touches when square to the wall) and 0.43 m (circumscribed: a
              CORNER can touch, depending on yaw)
  dashed      the PMP reference plan; blue = ours (ground truth, seed 0)
  rectangles  the true 0.63 x 0.585 m footprint at the plan's own yaw, red
              where the plan's footprint overlaps a wall cell, and the
              robot's actual footprint at its first contact
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src", "agx_navigation", "agx_planning"))
import summarize_compare as sc  # noqa: E402
from agx_planning.vector_field.field import VectorFieldConfig, compute_field  # noqa: E402

BANNER = "PRELIMINARY - NOT FOR PUBLICATION (2026-10-02): diagnosis for #34"
PLANS = ["floor_6_v2_00105", "floor_6_v2_00001", "floor_6_v2_00369", "floor_6_v2_00419"]
HALF_L, HALF_W = 0.315, 0.2925  # circumscribed 0.43 m, inscribed 0.29 m
RES, X0, Y0 = sc.RES, sc.X0, sc.Y0
CFG = VectorFieldConfig(speed_profile="exponential", inflation_radius=0.5,
                        smooth_T_before_grad=False, early_exit_enable=False)


def occupancy():
    img = sc.load_map(6)[0]
    occ = np.zeros(img.shape, np.int8)
    occ[img < 0.1] = 100
    occ[(img > 0.7) & (img < 0.9)] = -1  # 205 = unknown
    return np.flipud(occ)  # OccupancyGrid row 0 is at the origin (bottom)


def corners(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[HALF_L, HALF_W], [HALF_L, -HALF_W], [-HALF_L, -HALF_W], [-HALF_L, HALF_W]])
    return np.c_[x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c]


def hits_wall(occ, poly):
    # sample the rectangle's outline every cell
    pts = np.concatenate([np.linspace(poly[i], poly[(i + 1) % 4], 15) for i in range(4)])
    col = np.clip(((pts[:, 0] - X0) / RES).astype(int), 0, occ.shape[1] - 1)
    row = np.clip(((pts[:, 1] - Y0) / RES).astype(int), 0, occ.shape[0] - 1)
    return bool((occ[row, col] >= 65).any())


def panel(ax, occ, plan):
    z = np.load(os.path.join(ROOT, "run_data", "compare_v7_seed0",
                             f"track_w1_ours_{plan}.npz"))
    pp = np.asarray(z["plan_poses"])
    tr = z["track"]
    tr = tr[tr[:, 0] >= float(z["goal_t"])]
    gx_w, gy_w = z["goal_xy"]
    res, _ = compute_field(occ, int((gx_w - X0) / RES), int((gy_w - Y0) / RES), RES,
                           X0, Y0, CFG, allow_unknown=True)
    from scipy.ndimage import distance_transform_edt
    from agx_planning.vector_field.field import build_speed_field
    wall = occ >= 65
    edt = distance_transform_edt(~wall) * RES
    speed = build_speed_field(edt, wall, CFG)

    pts = np.r_[pp[:, :2], tr[:, 1:3]]
    lo, hi = pts.min(0) - 1.2, pts.max(0) + 1.2
    extent = [X0, X0 + occ.shape[1] * RES, Y0, Y0 + occ.shape[0] * RES]
    sp = np.where(wall, np.nan, speed)
    im = ax.imshow(sp, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1)
    ax.imshow(np.where(wall, 1.0, np.nan), origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=1)
    xs = X0 + (np.arange(occ.shape[1]) + 0.5) * RES
    ys = Y0 + (np.arange(occ.shape[0]) + 0.5) * RES
    ax.contour(xs, ys, edt, levels=[HALF_W, 0.43], colors=["white", "orange"],
               linewidths=0.8, linestyles=["-", "--"])
    c0, c1 = int((lo[0] - X0) / RES), int((hi[0] - X0) / RES)
    r0, r1 = int((lo[1] - Y0) / RES), int((hi[1] - Y0) / RES)
    k = 4
    sl = (slice(r0, r1, k), slice(c0, c1, k))
    X, Y = np.meshgrid(xs[c0:c1:k], ys[r0:r1:k])
    ax.quiver(X, Y, res.grad_x[sl], res.grad_y[sl], color="w", alpha=0.5,
              scale=60, width=0.002)
    ax.plot(pp[:, 0], pp[:, 1], "--", color="k", lw=1.5, label="PMP plan")
    ax.plot(tr[:, 1], tr[:, 2], color="tab:blue", lw=1.5, label="ours (truth)")
    n_bad = 0
    for x, y, yaw in pp[::3]:
        poly = corners(x, y, yaw)
        bad = hits_wall(occ, poly)
        n_bad += bad
        ax.add_patch(Polygon(poly, fill=False, lw=0.6 if not bad else 1.0,
                             ec="red" if bad else (1, 1, 1, 0.35)))
    for t, x, y, yaw in tr:
        poly = corners(x, y, yaw)
        if hits_wall(occ, poly):
            ax.add_patch(Polygon(poly, fill=True, fc=(0, 0.4, 1, 0.4), ec="tab:blue", lw=1.5))
            ax.annotate("first contact", (x, y), (8, 8), textcoords="offset points",
                        color="tab:blue", fontsize=8, weight="bold")
            break
    ax.plot(*pp[0, :2], "go")
    ax.plot(gx_w, gy_w, "r*", ms=12)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal")
    ax.set_title(f"{plan}: {n_bad}/{len(pp[::3])} plan footprints overlap a wall", fontsize=9)
    return im


def main():
    occ = occupancy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 13))
    for ax, plan in zip(axs.flat, PLANS):
        im = panel(ax, occ, plan)
    axs.flat[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=axs, shrink=0.6, label="FM2 wave speed v(x)  (exp profile, R=0.5 m)")
    fig.suptitle("white: centre 0.29 m from wall (side touches)   orange dashed: 0.43 m (a corner can touch)\n"
                 "red boxes: planned footprint overlaps a wall   blue box: actual first contact", fontsize=10)
    tmp = os.path.join(HERE, "field_tmp.png")
    fig.savefig(tmp, dpi=90)
    plt.close(fig)
    sys.path.insert(0, HERE)
    img = plt.imread(tmp)
    h, w = img.shape[:2]
    f2 = plt.figure(figsize=(w / 100, h / 100 + 0.4), dpi=100)
    a = f2.add_axes([0, 0, 1, h / (h + 40)])
    a.imshow(img)
    a.axis("off")
    f2.text(0.5, 1 - 20 / (h + 40), BANNER, ha="center", va="center", color="white",
            fontsize=11, weight="bold", bbox=dict(facecolor="red", edgecolor="none", pad=4))
    f2.savefig(os.path.join(HERE, "field_failures.png"))
    os.remove(tmp)


if __name__ == "__main__":
    main()
