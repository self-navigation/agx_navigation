#!/usr/bin/env python3
"""Job 248: one-factor sweep around TVLQG under amcl, dev half of plan set v3 floor 6.

Per run: the MIN body-to-wall clearance over the drive (truth pose, body perimeter
on the 5 cm baked grid, signed: <0 = a body point on a wall pixel; same geometry as
job 245/247's renderer), contact (any wall_overlap_s > 0), arrival. Per cell, paired
over (plan, seed) against SQ (TVLQG as in job 247) and SR (TVLQR): Wilcoxon on the
clearance, sign tests on contact and arrival; also on the no-door plans. Objective
and the p < 0.01 bar were fixed in the job README before the data.

    python3 render.py [job_dir]   -> sweep.png + table on stdout (and table.txt)
"""
import glob, json, os, sys
import numpy as np, yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import wilcoxon, binomtest
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job248_sweep-dev")
SET = os.path.join(R, "run_data/2026-10-10_nojob_plan-set-v3/set.json")
HL, HW, SP = 0.315, 0.2925, 0.02

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))[::-1]
wall = img == 0; res = meta["resolution"]; ox, oy = meta["origin"][:2]
sd = (distance_transform_edt(~wall) - distance_transform_edt(wall)) * res
lx = np.linspace(-HL, HL, int(2 * HL / SP) + 1); wy = np.linspace(-HW, HW, int(2 * HW / SP) + 1)
BODY = np.vstack([np.c_[lx, np.full_like(lx, HW)], np.c_[lx, np.full_like(lx, -HW)],
                  np.c_[np.full_like(wy, HL), wy], np.c_[np.full_like(wy, -HL), wy]])


def look(x, y):
    # BILINEAR in the signed distance between pixel centres: a nearest-pixel
    # lookup quantizes every clearance to a multiple of 5 cm, which made the
    # first pass of this table all ties (medians 0.050 / -0.050, every paired
    # difference 0). Continuous in the robot's position; still relative to the
    # 5 cm raster's wall pixels, so read it as a ranking, not an absolute gap.
    from scipy.ndimage import map_coordinates
    shp = np.shape(x)
    cc = ((np.ravel(x) - ox) / res - 0.5); rr = ((np.ravel(y) - oy) / res - 0.5)
    return map_coordinates(sd, [rr, cc], order=1, mode="nearest").reshape(shp)


def body_min(P):
    c, s = np.cos(P[:, 2:3]), np.sin(P[:, 2:3])
    return float(look(P[:, 0:1] + BODY[:, 0] * c - BODY[:, 1] * s, P[:, 1:2] + BODY[:, 0] * s + BODY[:, 1] * c).min())


S = json.load(open(SET)); plans = {p["id"]: p for p in (S if isinstance(S, list) else S["plans"])}
cells = [l.split()[0] for l in open(os.path.join(R, "tools/jobs/c1/sweep_dev.txt")) if l.strip() and l[0] != "#"]
runs = {}  # (cell, plan, seed) -> dict
for c in cells:
    for f in glob.glob(f"{JOB}/{c}/rows.s*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            pid = os.path.basename(r["plan_path"]).replace(".npz", "")
            tp = os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r.get("track_path") or ""))
            clr = np.nan
            if r["outcome"] not in ("planner-timeout", "stack-failed") and os.path.isfile(tp):
                T = np.load(tp)["track"]
                if len(T) > 2:
                    clr = body_min(T[:, 1:4])
            runs[(c, pid, r["seed"])] = dict(clr=clr, contact=r["wall_overlap_s"] > 0 if np.isfinite(clr) else None,
                                            arr=r["outcome"] == "arrived", ok=np.isfinite(clr),
                                            door=plans[pid]["doors"] > 0)

lines = []
def out(s=""):
    print(s); lines.append(s)

def compare(c, ref, subset):
    keys = [(p, s) for (cc, p, s) in runs if cc == c and subset(p)]
    pr = [(runs[(c, p, s)], runs[(ref, p, s)]) for p, s in keys if (ref, p, s) in runs]
    pr = [(a, b) for a, b in pr if a["ok"] and b["ok"]]
    d = np.array([a["clr"] - b["clr"] for a, b in pr])
    pw = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    fix = sum(b["contact"] and not a["contact"] for a, b in pr); new = sum(a["contact"] and not b["contact"] for a, b in pr)
    pc = binomtest(fix, fix + new).pvalue if fix + new else 1.0
    g = sum(a["arr"] and not b["arr"] for a, b in pr); lo = sum(b["arr"] and not a["arr"] for a, b in pr)
    return len(pr), np.median(d) if len(d) else np.nan, pw, fix, new, pc, g, lo

summary = {}
for name, subset in [("all 41 plans", lambda p: True), ("no-door plans", lambda p: plans[p]["doors"] == 0),
                     ("door plans", lambda p: plans[p]["doors"] > 0)]:
    out(f"\n== {name}")
    out(f"{'cell':7s} {'n':>4s} {'arr':>4s} {'contact':>8s} {'clr med':>8s} {'p10':>6s} | vs SQ: dclr med, p; contact fix/new p; arr +/- "
        f"| vs SR: dclr med, p; contact fix/new p")
    for c in cells:
        R_ = [v for (cc, p, s), v in runs.items() if cc == c and subset(p) and v["ok"]]
        clr = np.array([v["clr"] for v in R_])
        k = sum(v["contact"] for v in R_); a = sum(v["arr"] for v in R_)
        summary.setdefault(name, {})[c] = (clr, k, len(R_))
        s = f"{c:7s} {len(R_):4d} {a:4d} {k:4d} {100*k/len(R_):3.0f}% {np.median(clr):8.3f} {np.percentile(clr,10):6.3f}"
        for ref in ("SQ", "SR"):
            if c == ref:
                s += " |" + " " * 40; continue
            n, dm, pw, fx, nw, pc, g, lo = compare(c, ref, subset)
            s += f" | {dm:+.3f} p={pw:.2g}; {fx}/{nw} p={pc:.2g}" + (f"; arr +{g}/-{lo}" if ref == "SQ" else "")
        out(s)

open(os.path.join(HERE, "table.txt"), "w").write("\n".join(lines) + "\n")

# Clearance is CENSORED at contact: walls are one pixel thick, so the signed
# distance bottoms out at -0.05 and every contact run ties there (~45% of runs).
# It can only rank contact-free runs; the figure is the contact-run rate.
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h

fig, axs = plt.subplots(1, 3, figsize=(17, 5), sharey=False)
for ax, name in zip(axs, ["all 41 plans", "door plans", "no-door plans"]):
    st = [wilson(summary[name][c][1], summary[name][c][2]) for c in cells]
    p = np.array([x[0] for x in st]); lo = np.array([x[1] for x in st]); hi = np.array([x[2] for x in st])
    col = ["tab:red" if c == "SR" else "tab:blue" if c == "SQ" else "tab:green" if c.startswith("SE") else "0.55" for c in cells]
    ax.bar(range(len(cells)), 100 * p, color=col, yerr=[100 * (p - lo), 100 * (hi - p)], capsize=2)
    ax.axhline(100 * p[cells.index("SQ")], color="tab:blue", ls="--", lw=0.8)
    ax.set_xticks(range(len(cells)), cells, rotation=60, fontsize=8)
    ax.set_title(f"{name} (~{summary[name]['SQ'][2]} runs per cell)", fontsize=10)
    ax.set_ylabel("runs with wall contact [%] (Wilson 95%)"); ax.set_ylim(0, 100)
fig.suptitle("Job 248, amcl: no setting moves the contact rate (red TVLQR, blue TVLQG as job 247, green = steer on filter pose, grey = one knob changed)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "sweep.png"), dpi=110)
print("wrote", os.path.join(HERE, "sweep.png"))
