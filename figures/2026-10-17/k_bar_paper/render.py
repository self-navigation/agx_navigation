"""Paper version of figures/2026-10-12/k_bar_story (jobs 255/256/257, plan f6_medium_0376, seed 0) as one two-panel figure.
(a) side-slip k used in the linearization against time since the robot starts moving;
(b) ground-truth tracks at the contact door with the footprint at first contact.
Reads run_data/ and the baked floor_6 map.  python3 render.py (repo .venv)."""
import json, os
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from PIL import Image

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(R, "run_data")
PLAN, SEED, K_PLANT = "f6_medium_0376", 0, -0.33
HL, HW = 0.315, 0.2925
SRC = [("ground truth", "black"), ("AMCL", "tab:blue"), ("RTAB-Map", "tab:red")]
RUNS = [  # pose source, prior?, job dir, cell
    (0, False, "2026-10-12_job255_steerable", "T"), (1, False, "2026-10-12_job255_steerable", "Q0"),
    (2, False, "2026-10-12_job256_steerable-rtab", "RQ"), (0, True, "2026-10-12_job257_kprior", "PT"),
    (1, True, "2026-10-12_job257_kprior", "PQ"), (2, True, "2026-10-12_job257_kprior", "PR")]


def load(job, cell):
    for l in open(os.path.join(RD, job, cell, f"rows.s{SEED}.jsonl")):
        r = json.loads(l)
        if PLAN in r["plan"]:
            return r, np.load(os.path.join(RD, job, cell, f"s{SEED}", os.path.basename(r["track_path"])))
    raise KeyError((job, cell))


plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})
ax.axhline(K_PLANT, color="tab:green", lw=4, alpha=.3, zorder=0)
ax.axvspan(9.6, 10.8, color="0.5", alpha=.15, lw=0)
ax.text(10.95, -0.04, "doorway", color="0.35", va="center")
contacts, tracks = [], []
for si, prior, job, cell in RUNS:
    lab, col = SRC[si]
    r, z = load(job, cell)
    D, t, T = z["diag"], z["diag_t"], z["track"]
    t0 = t[np.argmax(D[:, 4] > 0.1)]
    kb, kplan = D[:, 18], D[0, 18]
    hit = (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None
    ls = "--" if prior else "-"
    off = {(0, True): 0.006, (2, True): -0.006}.get((si, prior), 0.0)  # prior arms coincide at -0.33
    ax.plot(t - t0, kb + off, color=col, ls=ls, lw=1.0 if prior else 1.4)
    on = np.abs(kb - kplan) > 1e-9
    if on.any():
        ta = t[np.argmax(on)]
        ax.plot(ta - t0, kplan, "o", ms=5, mfc="white", mec=col, mew=1.2, zorder=5)
    if hit:
        tc = r["wall_first_t"]
        ax.plot(tc - t0, np.interp(tc, t, kb), "X", ms=7, color=col, mec="k", mew=.5, zorder=6)
        i = min(np.searchsorted(T[:, 0], tc), len(T) - 1)
        contacts.append((col, T[i, 1], T[i, 2], T[i, 3]))
    tracks.append((col, ls, prior, T, z))
ax.set_xlim(-1, 25); ax.set_ylim(-0.37, 0.03)
ax.set_xlabel("time since start of motion [s]"); ax.set_ylabel("$k$ in the linearization [s]")
ax.set_title("(a)", loc="left", fontweight="bold")

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
res, (ox0, oy0, _) = meta["resolution"], meta["origin"]
bx.imshow(img, cmap="gray", extent=[ox0, ox0 + img.shape[1] * res, oy0, oy0 + img.shape[0] * res],
          origin="upper", vmin=0, vmax=255)
P = tracks[0][4]["plan_poses"]
bx.plot(P[:, 0], P[:, 1], color="tab:green", lw=1.2, ls=":")
for col, ls, prior, T, _ in tracks:
    bx.plot(T[:, 1], T[:, 2], color=col, ls=ls, lw=1.0 if prior else 1.4)
c, s = np.cos, np.sin
loc = np.array([[HL, HW], [-HL, HW], [-HL, -HW], [HL, -HW]])
for col, x, y, yaw in contacts:
    xy = np.c_[x + c(yaw) * loc[:, 0] - s(yaw) * loc[:, 1], y + s(yaw) * loc[:, 0] + c(yaw) * loc[:, 1]]
    bx.add_patch(Polygon(xy, closed=True, fc=col, alpha=.25, ec=col, lw=1))
cx, cy = np.mean([p[1] for p in contacts]), np.mean([p[2] for p in contacts])
bx.set_xlim(cx - 2.2, cx + 2.2); bx.set_ylim(cy - 2.0, cy + 2.0); bx.set_aspect("equal")
bx.set_xlabel("$x$ [m]"); bx.set_ylabel("$y$ [m]")
bx.set_title("(b)", loc="left", fontweight="bold")

h = [Line2D([], [], color=col, lw=1.4, label=lab) for lab, col in SRC]
h += [Line2D([], [], color="0.55", lw=1.4, label="online estimate of $k$"),
      Line2D([], [], color="0.55", ls="--", lw=1.0, label="scenario prior $k=-0.33$ s"),
      Line2D([], [], color="tab:green", lw=4, alpha=.3, label="emulator value $k=-0.33$ s"),
      Line2D([], [], color="tab:green", ls=":", lw=1.2, label="PMP plan"),
      Line2D([], [], marker="o", ms=5, mfc="white", mec="0.4", ls="none", label="online update starts"),
      Line2D([], [], marker="X", ms=7, color="0.6", mec="k", mew=.5, ls="none", label="first wall contact"),
      plt.Rectangle((0, 0), 1, 1, fc="0.5", alpha=.25, ec="0.5", label="footprint at contact")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.21, 1, 1))
for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"k_bar.{ext}"), dpi=200, bbox_inches="tight")
