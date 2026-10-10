"""Baked floor 2 map with a 1 m / 5 m grid, so locations (robot storage room,
real-robot test spots) can be named in world coordinates."""
import os
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
MAP = os.environ.get("FLOOR2_MAP", os.path.join(R, "src/rudn-ordjo-building/maps/floor_2"))
meta = yaml.safe_load(open(MAP + ".yaml"))
img = np.asarray(Image.open(MAP + ".png").convert("L"))[::-1]
res, ox, oy = meta["resolution"], meta["origin"][0], meta["origin"][1]
ext = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
occ = np.argwhere(img != 205)
y0, x0 = occ.min(0) * res + [oy, ox]; y1, x1 = occ.max(0) * res + [oy, ox]
fig, ax = plt.subplots(figsize=(26, 26 * (y1 - y0 + 2) / (x1 - x0 + 2)))
ax.imshow(img, origin="lower", extent=ext, cmap="gray", vmin=0, vmax=255)
ax.set_xlim(x0 - 1, x1 + 1); ax.set_ylim(y0 - 1, y1 + 1)
ax.set_xticks(np.arange(np.floor(x0 / 5) * 5, x1 + 1, 5)); ax.set_yticks(np.arange(np.floor(y0 / 5) * 5, y1 + 1, 5))
ax.set_xticks(np.arange(np.floor(x0), x1 + 1, 1), minor=True); ax.set_yticks(np.arange(np.floor(y0), y1 + 1, 1), minor=True)
ax.grid(which="major", color="tab:red", lw=0.6, alpha=0.6); ax.grid(which="minor", color="tab:blue", lw=0.2, alpha=0.3)
ax.plot(0, 0, "r^", ms=10); ax.annotate("sim spawn (0,0)", (0, 0), color="r", xytext=(5, 5), textcoords="offset points")
ax.set_title("floor 2 baked map, world coordinates (m); red grid 5 m, blue 1 m")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "floor2_grid.png"), dpi=90)
