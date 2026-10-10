"""All 8 floors baked at full mesh extent (2026-10-10), reachable-from-spawn area
shown white, unknown grey, walls black. Source: a full-range bake of
tools/bake_floor_map.py (bounds per floor from the GLB extents), dir in $FULL_MAPS."""
import os
import numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

D = os.environ.get("FULL_MAPS", "/tmp/full_bake")
fig, axes = plt.subplots(4, 2, figsize=(22, 18))
for fl, ax in zip(range(1, 9), axes.T.ravel()):
    meta = yaml.safe_load(open(f"{D}/floor_{fl}.yaml"))
    img = np.asarray(Image.open(f"{D}/floor_{fl}.png").convert("L"))[::-1]
    res, ox, oy = meta["resolution"], meta["origin"][0], meta["origin"][1]
    ax.imshow(img, origin="lower", extent=[ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res],
              cmap="gray", vmin=0, vmax=255)
    ax.set_xlim(-30, 160); ax.set_ylim(-20, 72); ax.set_aspect("equal")
    ax.plot(0, 0, "r^", ms=6)
    ax.set_title(f"floor {fl}: reachable {np.mean(img == 254) * img.size * res * res:.0f} m²")
    ax.grid(alpha=0.3)
fig.suptitle("All floors, full mesh extent; white = reachable from the spawn (red), grey = unknown/unreachable")
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__), "all_floors.png"), dpi=70)
