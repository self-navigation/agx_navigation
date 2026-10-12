#!/usr/bin/env python3
"""Direction of the last pose jump before contact (exploratory, 2026-10-12).

For each TVLQG contact with a map->odom jump > STEP in the WINDOW before first contact: n = unit vector from the truth
position at contact to the nearest wall pixel of the baked floor_6 map; d = the jump of the map->odom translation
(the estimate shifts by about d). If d.n < 0, the estimate jumped AWAY from the wall: the corrector then believes it
has room and steers the real robot toward the wall. Prediction if jumps cause contacts: d.n < 0 well above half.
Sign test. Compared with a matched null: the last jump before random moments in clean runs at the same doors,
projected on the nearest-wall normal there.
"""
import glob, json, os
import numpy as np, yaml
from PIL import Image
from scipy.stats import binomtest
from scipy.ndimage import distance_transform_edt

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../.."))
RD = os.path.join(R, "run_data"); STEP, WINDOW = 0.03, 1.0
meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
if img.ndim == 3: img = img[..., 0]
res, (ox, oy, _) = meta["resolution"], meta["origin"]
_, (iy, ix) = distance_transform_edt(img > 100, return_indices=True)   # nearest wall (dark) pixel per cell


def wall_normal(x, y):
    c, r = int((x - ox) / res), img.shape[0] - 1 - int((y - oy) / res)
    wx, wy = ox + (ix[r, c] + .5) * res, oy + (img.shape[0] - 1 - iy[r, c] + .5) * res
    v = np.array([wx - x, wy - y]); return v / (np.linalg.norm(v) + 1e-9), np.linalg.norm(v)


for loc, job, cell in (("rtab", "2026-10-12_job256_steerable-rtab", "RQ"), ("amcl", "2026-10-12_job255_steerable", "Q0")):
    away = toward = 0; dots = []
    for f in glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l); tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
            if not (os.path.isfile(tp) and (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None): continue
            z = np.load(tp); MO, T = z["map_odom"], z["track"]; tc = r["wall_first_t"]
            jumps = np.c_[np.diff(MO[:, 1]), np.diff(MO[:, 2])]; ts = MO[1:, 4]
            k = np.flatnonzero((np.hypot(*jumps.T) > STEP) & (ts <= tc) & (ts > tc - WINDOW))
            if not len(k): continue
            d = jumps[k].sum(0)                     # net jump inside the window
            i = min(np.searchsorted(T[:, 0], tc), len(T) - 1)
            n, dist = wall_normal(T[i, 1], T[i, 2])
            dp = float(d @ n); dots.append(dp)
            away += dp < 0; toward += dp > 0
    p = binomtest(away, away + toward).pvalue if away + toward else 1
    print(f"{loc}: contacts with a jump in the last {WINDOW} s: {away + toward}; estimate jumped AWAY from the wall in {away}, "
          f"toward in {toward}, sign test p {p:.2g}; median d.n {np.median(dots) * 100:+.1f} cm")
