"""Scan endpoints (projected at the TRUE spawn pose) over the baked floor_6 map.

Data: run_data/2026-10-10_nojob_scan-map-offset/{debug/,}pts_*.npz from
tools/scan_map_offset.py --dump. Answers: does the lidar see the walls where
the baked map draws them?
"""
import glob, os, sys
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
D = os.path.join(R, "run_data/2026-10-10_nojob_scan-map-offset")
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml"))
img = np.asarray(Image.open(MAP + ".png").convert("L"))[::-1]
res, ox, oy = meta["resolution"], meta["origin"][0], meta["origin"][1]
files = sorted(set(glob.glob(D + "/pts_*.npz")) | set(glob.glob(D + "/debug/pts_*.npz")))
files = list({os.path.basename(f): f for f in files}.values())
fig, axes = plt.subplots(len(files), 1, figsize=(12, 4.5 * len(files)), squeeze=False)
for ax, f in zip(axes[:, 0], files):
    z = np.load(f)
    pts, (x0, y0, th) = z["pts"], z["truth"]
    c, s = np.cos(th), np.sin(th)
    wx, wy = x0 + c * pts[:, 0] - s * pts[:, 1], y0 + s * pts[:, 0] + c * pts[:, 1]
    ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
    ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=255)
    ax.scatter(wx, wy, s=1, c="r", label="scan endpoints @ truth pose")
    ax.plot(x0, y0, "b^", ms=10, label="truth (spawn)")
    if len(z["amcl"]):
        ax.plot(z["amcl"][-1, 0], z["amcl"][-1, 1], "gx", ms=10, label="amcl")
    ax.set_xlim(x0 - 8, x0 + 8); ax.set_ylim(y0 - 3.5, y0 + 3.5); ax.set_aspect("equal")
    ax.set_title(f"{os.path.basename(f)}  ({len(pts)} pts, body-frame r range {np.hypot(*pts.T).min():.2f}-{np.hypot(*pts.T).max():.2f} m)")
    ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "scan_over_map.png"), dpi=110)
