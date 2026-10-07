#!/usr/bin/env python3
"""Draw plans from several sets side by side, coloured by wall clearance.

    .venv/bin/python tools/plot_plan_corridor.py --out fig.png
        --set "old library=LIB_DIR" --set "optimised=DIR/trials/N"
        [--plans 00105 00291 ...] [--n 6]

Each --set is a directory of <plan>.npz with a ``poses`` array (the library
schema and plan_opt.py's trial dirs both qualify). One row per plan, one column
per set: baked map, the plan coloured green where its hull clearance is >=
D_REQ and red where it is below, robot footprints every 1.5 s, and in the title
the share of the plan below D_REQ and the minimum clearance. Without --plans,
picks the plans with the largest share below D_REQ in the first set, among
plans every set contains.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src/agx_navigation/agx_planning"))
sys.path.insert(0, os.path.join(ROOT, "figures/2026-10-07/s1_supervisor"))
from agx_planning.supervisor.walls import SignedWallDistance  # noqa: E402
from render import robot  # noqa: E402

MAP = os.path.join(ROOT, "src/rudn-ordjo-building/maps/floor_6")
D_REQ = 0.50
DT = 0.1


def find(d, plan):
    for name in os.listdir(d):
        if name.endswith(".npz") and plan in name:
            return os.path.join(d, name)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", action="append", required=True, help="LABEL=DIR")
    ap.add_argument("--plans", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    sets = [s.split("=", 1) for s in a.set]
    swd = SignedWallDistance.from_map_yaml(MAP + ".yaml")
    clr = lambda P: np.array([swd.body_clearance(*p[:3]) for p in P])  # noqa: E731

    if a.plans is None:
        d0 = sets[0][1]
        scored = []
        for name in sorted(os.listdir(d0)):
            if (name.endswith(".npz") and "door" not in name
                    and all(os.path.exists(os.path.join(d, name)) for _, d in sets)):
                P = np.load(os.path.join(d0, name))["poses"]
                scored.append((np.mean(clr(P) < D_REQ), name.split("_")[-1][:-4]))
        a.plans = [p for _, p in sorted(scored, reverse=True)[:a.n]]

    meta = yaml.safe_load(open(MAP + ".yaml"))
    img = np.asarray(Image.open(MAP + ".png").convert("L"))
    res, (ox, oy, _) = meta["resolution"], meta["origin"]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]

    nr, nc = len(a.plans), len(sets)
    fig, axs = plt.subplots(nr, nc, figsize=(7.5 * nc, 5.2 * nr), squeeze=False)
    for i, plan in enumerate(a.plans):
        loaded = [(lab, find(d, plan)) for lab, d in sets]
        Ps = [np.load(f)["poses"] for _, f in loaded if f]
        if not Ps:
            continue
        allxy = np.vstack([P[:, :2] for P in Ps])
        lo, hi = allxy.min(0) - 1.0, allxy.max(0) + 1.0
        for j, (lab, f) in enumerate(loaded):
            ax = axs[i, j]
            ax.imshow(img, cmap="gray", extent=ext, origin="upper", vmin=0, vmax=255, zorder=0)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_aspect("equal")
            if f is None:
                ax.set_title(f"{lab}, план {plan}: не решён", fontsize=10, color="tab:red")
                continue
            P = np.load(f)["poses"]
            c = clr(P)
            out = c < D_REQ
            seg = np.stack([P[:-1, :2], P[1:, :2]], axis=1)
            ax.add_collection(LineCollection(seg, colors=np.where(out[:-1], "tab:red", "tab:green"),
                                             lw=2.6, zorder=3))
            for k in range(0, len(P), int(1.5 / DT)):
                robot(ax, P[k], "tab:red" if out[k] else "tab:green", alpha=0.35, z=2)
            robot(ax, P[0], "k", fill=False, lw=1.4, z=4)
            robot(ax, P[-1], "tab:blue", fill=False, lw=1.8, z=4)
            k = int(np.argmin(c))
            ax.plot(*P[k, :2], marker="X", color="k", ms=10, zorder=5)
            L = float(np.hypot(*np.diff(P[:, :2], axis=0).T).sum())
            ax.set_title(f"{lab}, план {plan}: вне коридора {out.mean():.0%} пути, "
                         f"мин. зазор {c.min():.2f} м\nдлина {L:.1f} м, время {len(P) * DT:.1f} с",
                         fontsize=10)
    fig.suptitle(a.title or f"Планы до и после: зелёный — зазор корпуса ≥ {D_REQ} м (внутри ε-коридора), "
                 "красный — меньше; ✖ — минимальный зазор", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(a.out, dpi=90)
    print(a.out, a.plans)


if __name__ == "__main__":
    main()
