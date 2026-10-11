#!/usr/bin/env python3
"""Job 247 (dev half of plan set v3, floor 6): where do the remaining wall contacts happen, and does the PLAN cause them?

Body-perimeter contact as in tools/compare_run.py wall_contact (copied: that
module imports rclpy). For every run of a cell: whether the truth body touched a
wall pixel, whether the nominal PLAN body (plan_poses) does, and whether the first
contact is in a narrow passage (plan centre clearance < NARROW m within 1 m of it:
doorways). Writes contacts.png + a table on stdout.

    python3 render.py [job_dir]
"""
import glob, json, os, sys
import numpy as np, yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-10_job247_final-dev")
CELLS = [("LTN", "TVLQR, truth"), ("LTNq", "TVLQG, truth"), ("LAN", "TVLQR, amcl"), ("LANq", "TVLQG, amcl")]
HL, HW, SP, NARROW = 0.315, 0.2925, 0.02, 0.55

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))[::-1]
wall = img == 0; res = meta["resolution"]; ox, oy = meta["origin"][:2]
sd = (distance_transform_edt(~wall) - distance_transform_edt(wall)) * res
lx = np.linspace(-HL, HL, int(2 * HL / SP) + 1); wy = np.linspace(-HW, HW, int(2 * HW / SP) + 1)
BODY = np.vstack([np.c_[lx, np.full_like(lx, HW)], np.c_[lx, np.full_like(lx, -HW)],
                  np.c_[np.full_like(wy, HL), wy], np.c_[np.full_like(wy, -HL), wy]])


def look(x, y):
    c = np.clip(((x - ox) / res).astype(int), 0, sd.shape[1] - 1)
    r = np.clip(((y - oy) / res).astype(int), 0, sd.shape[0] - 1)
    return sd[r, c]


def body_clear(P):  # P: (n, 3) x, y, yaw
    c, s = np.cos(P[:, 2:3]), np.sin(P[:, 2:3])
    return look(P[:, 0:1] + BODY[:, 0] * c - BODY[:, 1] * s, P[:, 1:2] + BODY[:, 0] * s + BODY[:, 1] * c).min(1)


out = {}
for cell, _ in CELLS:
    runs = []
    for f in sorted(glob.glob(f"{JOB}/{cell}/s*/track_*.npz")):
        z = np.load(f)
        T, P = z["track"], z["plan_poses"]
        if len(T) < 2 or len(P) < 2:
            continue
        bc = body_clear(T[:, 1:4]); inside = bc < 0
        pc = body_clear(P)
        r = dict(f=f, contact=bool(inside.any()), plan_min=float(pc.min()), xy=None, narrow=None)
        if inside.any():
            i = int(np.argmax(inside)); x, y = T[i, 1], T[i, 2]
            near = np.hypot(P[:, 0] - x, P[:, 1] - y) < 1.0
            cc = look(P[near, 0], P[near, 1]) if near.any() else look(np.array([x]), np.array([y]))
            r.update(xy=(float(x), float(y)), narrow=bool(cc.min() < NARROW),
                     pts=T[inside, 1:3])
        runs.append(r)
    out[cell] = runs
    con = [r for r in runs if r["contact"]]
    print(f"{cell:5s} runs={len(runs)} contact={len(con)} | plan body touches wall: {sum(r['plan_min'] < 0 for r in runs)} runs"
          f" (of contacts {sum(r['plan_min'] < 0 for r in con)}), plan within 5 cm: {sum(r['plan_min'] < 0.05 for r in runs)}"
          f" | contact in narrow passage: {sum(r['narrow'] for r in con)}, open: {sum(not r['narrow'] for r in con)}")

fig, axs = plt.subplots(len(CELLS), 1, figsize=(13, 3.6 * len(CELLS)), sharex=True)
ext = [ox, ox + wall.shape[1] * res, oy, oy + wall.shape[0] * res]
for ax, (cell, label) in zip(axs, CELLS):
    ax.imshow(np.where(wall, 0.2, np.where(img == 205, 0.85, 1.0)), cmap="gray", vmin=0, vmax=1,
              origin="lower", extent=ext, interpolation="nearest")
    for r in out[cell]:
        if r["contact"]:
            ax.plot(r["pts"][:, 0], r["pts"][:, 1], ".", ms=2, color="tab:red" if r["narrow"] else "tab:orange")
    n = len(out[cell]); k = sum(r["contact"] for r in out[cell])
    ax.set_title(f"{label}: body on a wall pixel in {k}/{n} runs (red = doorway/narrow, orange = open)", fontsize=10)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(-12, 11); ax.set_aspect("equal"); ax.set_ylabel("y [m]")
axs[-1].set_xlabel("x [m]")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "contacts.png"), dpi=110)
print("wrote", os.path.join(HERE, "contacts.png"))

# How much body clearance the PLAN leaves where it is tightest, vs whether the run touched.
for cell, _ in CELLS:
    pm = np.array([r["plan_min"] for r in out[cell]]); con = np.array([r["contact"] for r in out[cell]])
    bins = [0, 0.10, 0.15, 0.20, 0.30, 9]
    s = ", ".join(f"[{a:.2f},{b:.2f}): {int(con[(pm >= a) & (pm < b)].sum())}/{int(((pm >= a) & (pm < b)).sum())}"
                  for a, b in zip(bins, bins[1:]))
    print(f"{cell:5s} plan min body clearance -> contact runs/runs: {s}")

# How deep: a 5 cm raster turns a graze into "contact". Deepest body-into-wall per contact run.
print("\ncontact depth (max body penetration into wall pixels, m) per cell:")
for cell, _ in CELLS:
    d = []
    for r in out[cell]:
        if r["contact"]:
            z = np.load(r["f"]); d.append(-float(body_clear(z["track"][:, 1:4]).min()))
    d = np.array(d)
    print(f"{cell:5s} n={len(d)} median {np.median(d):.3f} | <=5cm {int((d <= 0.05).sum())}, 5-10cm {int(((d > 0.05) & (d <= 0.10)).sum())}, >10cm {int((d > 0.10).sum())}")
