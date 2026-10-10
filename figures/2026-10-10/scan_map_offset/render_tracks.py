"""Panel: scan endpoints at the Gazebo pose vs truth-track occupancy, over the baked map.

If the baked map were exact, scan endpoints would lie on the black wall lines and no
robot centre could sit within ~0.3 m of a wall line. Data:
run_data/2026-10-10_nojob_scan-map-offset/floor6/pts_*.npz (tools/scan_map_offset.py
--dump, truth read from gz pose/info) and the job 244 track npz files (Gazebo truth).
"""
import glob, os
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
meta = yaml.safe_load(open(MAP + ".yaml"))
img = np.asarray(Image.open(MAP + ".png").convert("L"))[::-1]
res, ox, oy = meta["resolution"], meta["origin"][0], meta["origin"][1]
ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
T = np.concatenate([np.load(f)["track"] for f in glob.glob(R + "/run_data/2026-10-09_job244_tvlqg-deviation/L*/s*/track_*.npz")])
fig, ax = plt.subplots(figsize=(13, 6))
ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=255)
ax.plot(T[:, 1], T[:, 2], ",", color="tab:blue", alpha=0.3)
ax.plot([], [], ".", color="tab:blue", label="robot centre, job 244 truth tracks (all arms)")
for f, col in zip(sorted(glob.glob(R + "/run_data/2026-10-10_nojob_scan-map-offset/floor6/pts_*.npz"))[:2], ["r", "m"]):
    z = np.load(f); x0, y0, th = z["truth"]; p = z["pts"]
    c, s = np.cos(th), np.sin(th)
    ax.scatter(x0 + c * p[:, 0] - s * p[:, 1], y0 + s * p[:, 0] + c * p[:, 1], s=1, c=col,
               label=f"scan endpoints, robot at ({x0:.0f}, {y0:.1f}, yaw {th:.2f}) per Gazebo")
    ax.plot(x0, y0, "^", color=col, ms=9)
ax.set_xlim(-20, -2); ax.set_ylim(-6, 2); ax.set_aspect("equal")
ax.set_title("Baked map (black lines) vs what the lidar sees (red/magenta) and where the robot drove (blue)")
ax.legend(loc="upper right", fontsize=8, markerscale=6)
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "scan_vs_tracks.png"), dpi=110)
