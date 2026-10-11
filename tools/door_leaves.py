#!/usr/bin/env python3
"""Add ajar door leaves to corridor plans: detect doors, draw leaves into maps.

Doorways are found from the clearance map as narrow components that connect
the corridor (the spawn's component of cells with clearance >= 0.6 m) to a
room; the jambs are found by ray casting across the doorway. A leaf of
thickness LEAF_T and the gap's length is hinged at a random jamb and opened by
theta into the corridor. The baked map is never edited: each variant gets its
own map copy with the leaf rasterized in, and the plan is re-solved on it.
tools/spawn_leaves.py spawns matching boxes in Gazebo.

    python3 tools/door_leaves.py doors  --floor 6 --png doors.png
    python3 tools/door_leaves.py sample --plans D/set_floor6_corridor.txt --out OUT --variants 3 --seed 0
    python3 tools/build_plan_set.py solve --out OUT --jobs 12
"""
import argparse
import json
import math
import os
import sys

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "tools"))
import build_plan_set as bps  # noqa: E402

DOOR_MIN_CLEAR = 0.30   # doorway centre clearance floor [m]: below this the hull cannot pass
LEAF_T = 0.05           # leaf thickness [m] (one map cell)
NEAR_ROUTE = 1.5        # a door gets a leaf only if its hinge is within this of the plan [m]
THETA = (math.radians(20), math.radians(90))  # opening angle range


def doors(floor):
    """[{centroid, n, j1, j2, width}] in world coordinates, for the doorways off the corridor."""
    from scipy import ndimage
    arr, meta = bps.load_map(floor)
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    edt = bps.free_edt(floor)
    wall = arr != 0
    big = edt >= bps.CORRIDOR_CLEAR
    lab, nlab = ndimage.label(big)
    C = bps.corridor_mask(floor)
    cid = np.bincount(lab[C]).argmax()
    rooms = big & (lab != cid)
    grow = int(math.ceil((bps.CORRIDOR_CLEAR - DOOR_MIN_CLEAR) / r)) + 1
    se = ndimage.generate_binary_structure(2, 1)
    cand = (edt >= DOOR_MIN_CLEAR) & ~ndimage.binary_dilation(big, se, grow)
    dl, nd = ndimage.label(cand)
    Cd = ndimage.binary_dilation(C, se, grow + 2)
    Rd = ndimage.binary_dilation(rooms, se, grow + 2)
    cyx = np.argwhere(C)
    out = []
    for k in range(1, nd + 1):
        m = dl == k
        if not (m & Cd).any() or not (m & Rd).any():
            continue
        py, px = np.argwhere(m).mean(0)
        dC = np.hypot(cyx[:, 0] - py, cyx[:, 1] - px)
        q = cyx[np.argmin(dC)]
        n = np.array([q[1] - px, q[0] - py], float); n /= np.linalg.norm(n)
        t = np.array([-n[1], n[0]])

        def ray(p, d, want_wall, lim=60):
            for i in range(1, lim):
                x, y = p + d * i * 0.5
                c, rr = int(round(x)), int(round(y))
                if not (0 <= rr < arr.shape[0] and 0 <= c < arr.shape[1]):
                    return None
                if wall[rr, c] == want_wall:
                    return np.array([x, y])
            return None
        p = np.array([px, py])
        js = []
        for s in (1, -1):
            hit = ray(p, s * t, True)
            if hit is None:
                break
            # walk along n from the wall hit until the cell beside it (outward along t) is free
            w = hit.copy()
            for i in range(1, 60):
                c2 = w + n * 0.5
                cc, rr = int(round(c2[0])), int(round(c2[1]))
                if not wall[rr, cc]:
                    break
                w = c2
            js.append(w)
        if len(js) != 2:
            continue
        j1, j2 = (np.array([ox + (j[0] + 0.5) * r, oy + (j[1] + 0.5) * r]) for j in js)
        width = float(np.linalg.norm(j1 - j2))
        if not 0.6 <= width <= 1.6:
            continue
        out.append(dict(centroid=[round(ox + (px + 0.5) * r, 3), round(oy + (py + 0.5) * r, 3)],
                        n=[round(float(n[0]), 4), round(float(n[1]), 4)],
                        j1=[round(float(j1[0]), 3), round(float(j1[1]), 3)],
                        j2=[round(float(j2[0]), 3), round(float(j2[1]), 3)], width=round(width, 3)))
    return out


def leaf_segment(door, hinge_side, theta):
    """(hinge xy, tip xy) of a leaf hinged at j1 (side 0) or j2 (side 1), opened theta toward n."""
    h, o = (np.array(door["j1"]), np.array(door["j2"])) if hinge_side == 0 else \
           (np.array(door["j2"]), np.array(door["j1"]))
    L = np.linalg.norm(o - h)
    u = (o - h) / L
    n = np.array(door["n"])
    d = math.cos(theta) * u + math.sin(theta) * n
    return h, h + L * d


def draw_leaf(img, meta, h, tip):
    """Rasterize a LEAF_T-thick segment into a PNG-convention image (row 0 = y_max) as wall (0)."""
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    H = img.shape[0]
    L = np.linalg.norm(tip - h)
    for s in np.linspace(0, 1, int(L / (r / 3)) + 2):
        p = h + s * (tip - h)
        for e in np.linspace(-LEAF_T / 2, LEAF_T / 2, 3):
            q = p + e * np.array([-(tip - h)[1], (tip - h)[0]]) / L
            c, rr = int((q[0] - ox) / r), int((q[1] - oy) / r)
            if 0 <= rr < H and 0 <= c < img.shape[1]:
                img[H - 1 - rr, c] = 0


def _edt(arr, r):
    from scipy import ndimage
    return ndimage.distance_transform_edt(arr == 0) * r


def cmd_sample(a):
    """Per base plan x variant: ajar leaves on the doors along its route, a map copy, a candidate."""
    import yaml
    from PIL import Image
    from scipy import ndimage
    rng = np.random.default_rng(a.seed)
    D = doors(a.floor)
    base_yaml = bps.MAPS[a.floor][0]
    meta = yaml.safe_load(open(base_yaml))
    png0 = np.asarray(Image.open(os.path.join(os.path.dirname(base_yaml), meta["image"])).convert("L")).copy()
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    os.makedirs(os.path.join(a.out, "maps"), exist_ok=True)
    plan_dir = os.path.dirname(os.path.abspath(a.plans))
    ids = [l.split()[0] for l in open(a.plans) if l.strip() and not l.startswith("#")]
    out, skipped = [], []
    for pid in ids:
        path = pid if pid.endswith(".npz") else os.path.join(plan_dir, "plans", pid + ".npz")
        # plan lists write __HOME__ for the VM home; the laptop mirror is the repo's run_data/
        path = next((q for q in (path.replace("__HOME__", os.path.expanduser("~")), path.replace("__HOME__", R))
                     if os.path.exists(q)), path)
        z = np.load(path)
        P = z["poses"]
        pid = os.path.basename(path)[:-4]
        near = [d for d in D if np.min(np.hypot(P[:, 0] - d["centroid"][0], P[:, 1] - d["centroid"][1]))
                < NEAR_ROUTE + d["width"]]
        if not near:
            skipped.append(pid); continue
        start = [float(z["start_xy"][0]), float(z["start_xy"][1]), float(P[0, 2])]
        goal = [float(v) for v in z["goal_pose"]]
        for v in range(a.variants):
            for attempt in range(20):
                leaves = []
                for d in near:
                    if rng.random() < a.p_ajar:
                        th = float(rng.uniform(*THETA)) * (1 - attempt / 20)
                        side = int(rng.integers(2))
                        h, tip = leaf_segment(d, side, th)
                        leaves.append(dict(hinge=[round(float(h[0]), 3), round(float(h[1]), 3)],
                                           tip=[round(float(tip[0]), 3), round(float(tip[1]), 3)],
                                           theta_deg=round(math.degrees(th), 1), door=d["centroid"]))
                if not leaves:
                    continue
                img = png0.copy()
                for L in leaves:
                    draw_leaf(img, meta, np.array(L["hinge"]), np.array(L["tip"]))
                arr = np.flipud(img)
                occ = np.where(arr <= 50, 100, np.where(arr >= 250, 0, -1))
                e = _edt(occ, r)
                cell = lambda p: (int((p[1] - oy) / r), int((p[0] - ox) / r))  # noqa: E731
                if e[cell(start)] < a.end_clear or e[cell(goal)] < a.end_clear:
                    continue
                lab, _ = ndimage.label(e >= a.pass_clear)
                if lab[cell(start)] == 0 or lab[cell(start)] != lab[cell(goal)]:
                    continue
                break
            else:
                skipped.append(f"{pid}/v{v}"); continue
            cid = f"{pid}_L{v}"
            Image.fromarray(img).save(os.path.join(a.out, "maps", cid + ".png"))
            m2 = dict(meta, image=cid + ".png")
            with open(os.path.join(a.out, "maps", cid + ".yaml"), "w") as f:
                f.write(f"# {cid}: {base_yaml} + {len(leaves)} ajar door leaves (tools/door_leaves.py)\n")
                yaml.safe_dump(m2, f, sort_keys=False)
            out.append(dict(id=cid, floor=a.floor, band="doorleaf", end_clear=a.end_clear,
                            start=[round(x, 4) for x in start], goal=[round(x, 4) for x in goal],
                            map_yaml=os.path.abspath(os.path.join(a.out, "maps", cid + ".yaml")),
                            base_plan=pid, leaves=leaves))
    json.dump(dict(maps={a.floor: base_yaml}, doors=D, seed=a.seed, candidates=out, skipped=skipped,
                   params=dict(near_route=NEAR_ROUTE, theta_deg=[math.degrees(t) for t in THETA],
                               p_ajar=a.p_ajar, leaf_t=LEAF_T, pass_clear=a.pass_clear, end_clear=a.end_clear)),
              open(os.path.join(a.out, "candidates.json"), "w"), indent=1)
    print(f"{len(out)} candidates from {len(ids)} plans; skipped {len(skipped)}: {skipped}")


def cmd_doors(a):
    D = doors(a.floor)
    print(f"{len(D)} doors"); json.dump(D, open(a.json, "w"), indent=1) if a.json else None
    if a.png:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        arr, meta = bps.load_map(a.floor)
        r, (ox, oy) = meta["resolution"], meta["origin"][:2]
        ext = [ox, ox + arr.shape[1] * r, oy, oy + arr.shape[0] * r]
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.imshow(np.where(arr == 0, 255, np.where(arr > 0, 0, 205)), cmap="gray", origin="lower", extent=ext)
        ax.imshow(np.ma.masked_where(~bps.corridor_mask(a.floor), np.ones(arr.shape)), cmap="Blues",
                  alpha=0.25, origin="lower", extent=ext, vmin=0, vmax=2)
        for i, d in enumerate(D):
            j1, j2 = np.array(d["j1"]), np.array(d["j2"])
            ax.plot(*np.c_[j1, j2], "r-", lw=2)
            h, tip = leaf_segment(d, 0, math.radians(60))
            ax.plot(*np.c_[h, tip], "m-", lw=1.5)
            ax.text(*d["centroid"], str(i), fontsize=7, color="darkred")
        C = np.argwhere(bps.corridor_mask(a.floor))
        ax.set_xlim(ox + (C[:, 1].min() - 40) * r, ox + (C[:, 1].max() + 40) * r)
        ax.set_ylim(oy + (C[:, 0].min() - 40) * r, oy + (C[:, 0].max() + 40) * r)
        ax.set_title(f"floor {a.floor}: {len(D)} doorways (red: jamb line; magenta: leaf at 60 deg from j1)")
        fig.tight_layout(); fig.savefig(a.png, dpi=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("doors"); p.add_argument("--floor", type=int, default=6)
    p.add_argument("--png"); p.add_argument("--json")
    p = sp.add_parser("sample"); p.add_argument("--floor", type=int, default=6)
    p.add_argument("--plans", required=True, help="plan list (ids resolved under <its dir>/plans) or npz paths")
    p.add_argument("--out", required=True); p.add_argument("--variants", type=int, default=3)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--p-ajar", type=float, default=0.7)
    p.add_argument("--pass-clear", type=float, default=0.35, help="centre clearance the route must keep [m]")
    p.add_argument("--end-clear", type=float, default=0.45, help="endpoint clearance (spins) [m]")
    a = ap.parse_args()
    {"doors": cmd_doors, "sample": cmd_sample}[a.cmd](a)


if __name__ == "__main__":
    main()
