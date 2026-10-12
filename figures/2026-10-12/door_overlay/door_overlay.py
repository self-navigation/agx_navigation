#!/usr/bin/env python3
"""Every doorway passage overlaid in one DOOR FRAME, one panel per arm (2026-10-12; asked for by the user: track
figures, aligned so that the excursions stack).

Doors: along each plan's path, the points where the free-space clearance (EDT of the baked floor_6 map) has a local
minimum below DOOR_CLEAR. These are the narrow passages; corridors are wider. Nearby minima (< 1 m) are merged.
Door frame per (plan, door): from the door's OWN geometry, so every door lines up. Jamb 1 = the wall pixel nearest the
plan point at the minimum; jamb 2 = the nearest wall pixel on the other side. Origin = the midpoint of the jambs,
+x = the door normal oriented along the robot's travel (so the robot always enters from the left), y = along the door
line. Black bars: the median door half-width; the grey cloud is every door's real walls. y is mirrored so that the plan TURNS LEFT across the door (heading change from -1.5 m to +1.5 m
>= 0), so the outward side of a door turn is always -y (lateral skid pushes the robot that way).
Drawn per panel: the real walls around every door, transformed into the same frame (grey points: every door's own
jambs, so the door frame appears as a fuzzy aperture); every run's truth track through the window x in [-W, W]
(thin, translucent); runs whose FIRST wall contact falls in the window in red, with the robot footprint at the contact
moment. Contact = wall_overlap_s > 0, first contact from wall_first_t.

usage: door_overlay.py [--out=NAME] [JOB:CELL:LABEL ...]   (default: the four arms below, mu 0.5; NAME.png/.txt)
"""
import glob, json, os, sys
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from PIL import Image
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../.."))
RD = os.path.join(R, "run_data")
DOOR_CLEAR, W, HL, HW = 0.60, 2.5, 0.315, 0.2925   # HL/HW: pmp_planner footprint half length / width (plan_check.py)
OUT = next((a[6:] for a in sys.argv[1:] if a.startswith("--out=")), "door_overlay")
ARMS = [a for a in sys.argv[1:] if not a.startswith("--out=")] or ["2026-10-12_job255_steerable:TR:TVLQR, true pose",
                        "2026-10-12_job255_steerable:T:TVLQG, true pose",
                        "2026-10-12_job255_steerable:Q0:TVLQG, amcl",
                        "2026-10-12_job256_steerable-rtab:RQ:TVLQG, rtab"]

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
if img.ndim == 3: img = img[..., 0]
res, (ox, oy, _) = meta["resolution"], meta["origin"]
H = img.shape[0]
clear = distance_transform_edt(img > 250) * res
wr, wc = np.nonzero(img < 50)
WALL = np.c_[ox + (wc + .5) * res, oy + (H - 1 - wr + .5) * res]


def clearance(xy):
    c = np.clip(((xy[:, 0] - ox) / res).astype(int), 0, img.shape[1] - 1)
    r = np.clip(H - 1 - ((xy[:, 1] - oy) / res).astype(int), 0, H - 1)
    return clear[r, c]


_doors = {}
def doors(plan_path):
    """[(origin xy, heading, mirror)] for one plan, cached."""
    if plan_path in _doors: return _doors[plan_path]
    P = np.load(os.path.join(R, plan_path.replace("/home/programmer/", "")))["poses"]
    xy, th = P[:, :2], np.unwrap(P[:, 2])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    cl = clearance(xy)
    out = []
    for i in range(1, len(cl) - 1):
        if cl[i] < DOOR_CLEAR and cl[i] <= cl[i - 1] and cl[i] <= cl[i + 1]:
            if s[i] < 0.5 or s[-1] - s[i] < 0.5: continue          # start/goal pose, not a passage
            if out and s[i] - out[-1][3] < 1.0: continue
            # the door's own geometry: jamb 1 = nearest wall pixel, jamb 2 = nearest wall pixel on the other side
            p = xy[i]; near = WALL[np.hypot(*(WALL - p).T) < 1.0]
            if len(near) < 2: continue
            d = near - p; k1 = np.argmin(np.hypot(*d.T)); j1 = near[k1]
            opp = d @ (j1 - p) < 0
            if not opp.any(): continue
            j2 = near[opp][np.argmin(np.hypot(*d[opp].T))]
            centre, line = (j1 + j2) / 2, j2 - j1
            if np.linalg.norm(line) > 1.3: continue               # not a door: a corridor pinch
            nrm = np.array([line[1], -line[0]]) / np.linalg.norm(line)
            if nrm @ np.array([np.cos(th[i]), np.sin(th[i])]) < 0: nrm = -nrm   # +x = direction of travel
            h = np.arctan2(nrm[1], nrm[0])
            a, b = np.searchsorted(s, s[i] - 1.5), min(np.searchsorted(s, s[i] + 1.5), len(s) - 1)
            turn = (th[b] - th[a] + np.pi) % (2 * np.pi) - np.pi
            out.append((centre, h, -1.0 if turn < 0 else 1.0, s[i], np.linalg.norm(line)))
    _doors[plan_path] = out
    return out


def to_frame(p, o, h, m):
    d = p - o; c, s_ = np.cos(h), np.sin(h)
    return np.c_[c * d[:, 0] + s_ * d[:, 1], m * (-s_ * d[:, 0] + c * d[:, 1])]


fig, axs = plt.subplots(len(ARMS), 1, figsize=(10.8, 4.6 * len(ARMS) + 1.2), sharex=True)
plt.rcParams.update({"font.size": 18})
summary = []
for ax, arm in zip(np.atleast_1d(axs), ARMS):
    job, cell, label = arm.split(":", 2)
    seen_doors, n_pass, n_hit, widths = set(), 0, 0, []
    for f in sorted(glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if not os.path.isfile(tp): continue
            T = np.load(tp)["track"]
            if len(T) < 5: continue
            hit = (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None
            ic = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1) if hit else None
            D_ = doors(r["plan_path"])
            # a contact belongs to ONE door: the one whose frame puts the contact point closest to the origin
            own = None
            if hit and D_:
                own = int(np.argmin([np.hypot(*to_frame(T[ic:ic + 1, 1:3], o, h, m)[0]) for o, h, m, *_ in D_]))
            for di, (o, h, m, _, width) in enumerate(D_):
                key = (r["plan_path"], round(o[0], 2), round(o[1], 2))
                if key not in seen_doors:
                    seen_doors.add(key)
                    wf = to_frame(WALL[np.hypot(*(WALL - o).T) < W + 0.5], o, h, m)
                    ax.scatter(wf[:, 0], wf[:, 1], s=1.5, color="0.35", alpha=0.10, lw=0, zorder=1)
                    widths.append(width)
                q = to_frame(T[:, 1:3], o, h, m)
                win = (np.abs(q[:, 0]) < W) & (np.abs(q[:, 1]) < 1.5)
                if win.sum() < 3: continue
                n_pass += 1
                here = hit and di == own and win[ic]
                if here:
                    n_hit += 1
                    ax.plot(q[win, 0], q[win, 1], "-", color="tab:red", lw=1.6, alpha=0.9, zorder=3)
                    yaw = m * (T[ic, 3] - h); c, s_ = np.cos(yaw), np.sin(yaw)
                    fp = np.array([[HL, HW], [-HL, HW], [-HL, -HW], [HL, -HW]]) @ np.array([[c, s_], [-s_, c]]) + q[ic]
                    ax.add_patch(Polygon(fp, fill=False, ec="tab:red", lw=1.2, alpha=0.8, zorder=4))
                else:
                    ax.plot(q[win, 0], q[win, 1], "-", color="tab:blue", lw=0.7, alpha=0.18, zorder=2)
    ax.set_title(f"{label}: {n_hit} wall hits in {n_pass} door passages", fontsize=19, fontweight="bold")
    ax.set_xlim(-W, W); ax.set_ylim(-1.3, 1.3); ax.set_facecolor('white'); ax.set_aspect("equal"); ax.set_ylabel("lateral (m)")
    jw = np.median(widths) / 2
    for sg in (1, -1):
        ax.plot([0, 0], [sg * jw, sg * 1.3], color="k", lw=4, solid_capstyle="butt", zorder=5)
    summary.append(f"{label}: {n_hit} hits / {n_pass} passages ({len(seen_doors)} distinct plan-doors)")
np.atleast_1d(axs)[-1].set_xlabel("along the plan through the door (m); robot enters from the left,\n"
                                  "turns mirrored to the left, so the outside of a turn is DOWN")
fig.suptitle("All door passages overlaid in the door's frame\n(mu 0.5 patches, 42 plans x 3 seeds; red = wall hit + robot outline)",
             fontsize=21, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(HERE, OUT + ".png"), dpi=110)
open(os.path.join(HERE, OUT + ".txt"), "w").write("\n".join(summary) + "\n")
print("\n".join(summary))
