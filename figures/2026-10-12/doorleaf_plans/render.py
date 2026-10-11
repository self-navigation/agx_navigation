"""Door-leaf plan set (2026-10-12): each corridor-only base plan without leaves, beside its
leaf variants with the plan re-solved on the leaf map.

Rows = base plans (6 per page); column 0 = floor_6 + the original plan (plan set v3);
columns 1-3 = variants L0-L2: the per-plan leaf map, the leaves (red), the re-solved plan
(blue) and the original plan (cyan dashed) for reference. A missing variant panel = that
variant did not solve. Title: n leaves, min centre-to-leaf distance of the solved plan.
Usage: python render.py [PLANSET_DIR] -> page_<k>.png
"""
import glob, json, os, sys
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-12_nojob_doorleaf-plans")
V3 = os.path.join(R, "run_data/2026-10-10_nojob_plan-set-v3/plans")
BASE = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")
PER_PAGE, NV = 6, 3


def load_img(yaml_path):
    m = yaml.safe_load(open(yaml_path))
    img = np.asarray(Image.open(os.path.join(os.path.dirname(yaml_path), m["image"])).convert("L"))
    r, (ox, oy) = m["resolution"], m["origin"][:2]
    return img, [ox, ox + img.shape[1] * r, oy, oy + img.shape[0] * r]


def leaf_dist(P, leaves):
    best = np.inf
    for L in leaves:
        h, t = np.array(L["hinge"]), np.array(L["tip"]); d = t - h
        s = np.clip(((P - h) @ d) / (d @ d), 0, 1)
        best = min(best, float(np.min(np.hypot(*(P - (h + s[:, None] * d)).T))))
    return best


cands = json.load(open(os.path.join(D, "candidates.json")))["candidates"]
bases = sorted({c["base_plan"] for c in cands})
base_img, base_ext = load_img(BASE)
for page in range((len(bases) + PER_PAGE - 1) // PER_PAGE):
    rows = bases[page * PER_PAGE:(page + 1) * PER_PAGE]
    fig, axs = plt.subplots(len(rows), NV + 1, figsize=(4.2 * (NV + 1), 3.0 * len(rows)), squeeze=False)
    for ax_row, b in zip(axs, rows):
        P0 = np.load(os.path.join(V3, b + ".npz"))["poses"]
        lo, hi = P0[:, :2].min(0) - 1.5, P0[:, :2].max(0) + 1.5
        variants = {c["id"]: c for c in cands if c["base_plan"] == b}
        for ax in ax_row:
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
        ax = ax_row[0]
        ax.imshow(base_img, cmap="gray", extent=base_ext, origin="upper")
        ax.plot(P0[:, 0], P0[:, 1], "b-", lw=1.2)
        ax.plot(*P0[0, :2], "go", ms=4); ax.plot(*P0[-1, :2], "rs", ms=4)
        ax.set_title(f"{b}: no leaves", fontsize=8)
        for v in range(NV):
            ax = ax_row[v + 1]
            cid = f"{b}_L{v}"
            if cid not in variants:
                ax.text(0.5, 0.5, "not sampled", transform=ax.transAxes, ha="center"); continue
            img, ext = load_img(os.path.join(D, "maps", cid + ".yaml"))
            ax.imshow(img, cmap="gray", extent=ext, origin="upper")
            leaves = variants[cid]["leaves"]
            for L in leaves:
                ax.plot(*np.c_[L["hinge"], L["tip"]], "r-", lw=2)
            ax.plot(P0[:, 0], P0[:, 1], "c--", lw=0.8)
            pf = os.path.join(D, "plans", cid + ".npz")
            if os.path.exists(pf):
                P = np.load(pf)["poses"]
                ax.plot(P[:, 0], P[:, 1], "b-", lw=1.2)
                ax.set_title(f"L{v}: {len(leaves)} leaves, min to leaf {leaf_dist(P[:, :2], leaves):.2f} m", fontsize=8)
            else:
                ax.set_title(f"L{v}: {len(leaves)} leaves, NOT SOLVED", fontsize=8, color="red")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"page_{page + 1}.png"), dpi=80)
    plt.close(fig)
    print("page", page + 1, rows)
