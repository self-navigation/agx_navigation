"""Tyre calibration (jobs 263/264): sideways slide vs gz WheelSlip lateral compliance, TVLQR + truth, plain ground,
v3f6_dev seed 0. Left: fitted k (v_lat = k v omega) per cell, real-robot band shaded. Right: v_lat/omega by speed for
the old tyre and the chosen one (the old tyre's saturation at half the wheelbase is the axle lock).
Reads run_data/2026-10-12_job26{3,4}_*/<cell>/{yaw.json,s0/}.  python3 render.py (repo .venv)."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning")]
import yaw_response as Y
from agx_planning.pmp_planner.config import PlannerConfig
cfg = PlannerConfig(mode="offline")
cells = {"L1u": 1e-6, "L02": .02, "L05": .05, "L10": .1, "L20": .2, "L50": .5, "L100": 1.0, "L200": 2.0}
def d(c): return glob.glob(os.path.join(R, f"run_data/2026-10-12_job26[34]_*/{c}"))[0]
ks = {c: json.load(open(os.path.join(d(c), "yaw.json")))["k"] for c in cells}
k0 = json.load(open(os.path.join(d("L0"), "yaw.json")))["k"]
contacts = {}
for c in ["L0", *cells]:
    rows = [json.loads(l) for f in glob.glob(os.path.join(d(c), "parts/*.jsonl")) for l in open(f)]
    contacts[c] = sum((r.get("wall_overlap_s") or 0) > 0 for r in rows), len(rows)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
x = np.array(list(cells.values())); y = np.array([ks[c] for c in cells])
ax[0].axhspan(-0.16, -0.03, color="tab:green", alpha=.15, label="real Scout Mini (wood -0.03 .. linoleum -0.16)")
ax[0].axhline(k0, color="tab:red", ls="--", label=f"old sim tyre (no WheelSlip): {k0:.2f}")
ax[0].semilogx(x, y, "o-", color="k")
for c, xi, yi in zip(cells, x, y):
    n, N = contacts[c]
    ax[0].annotate(f"{yi:.2f}\nwall {n}/{N}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax[0].plot([1.0], [ks["L100"]], "o", ms=14, mfc="none", mec="tab:green", mew=2, label="chosen: 1.0")
ax[0].set_xlabel("WheelSlip lateral slip compliance"); ax[0].set_ylabel("sideslip k  [s]  (v_lat = k v ω)")
ax[0].set_title(f"Sideways slide vs tyre compliance (TVLQR, truth, plain ground, 41 plans)\nold tyre wall contact {contacts['L0'][0]}/{contacts['L0'][1]}", fontsize=10)
ax[0].legend(fontsize=8, loc="lower right"); ax[0].set_ylim(-0.4, 0.02)
for c, lab, col in [("L0", "old tyre", "tab:red"), ("L100", "chosen tyre (1.0)", "tab:green")]:
    fs = sorted(glob.glob(os.path.join(d(c), "s0/**/track_*.npz"), recursive=True))
    rs = [r for r in (Y.intervals(f, cfg.c_v, cfg.c_w) for f in fs) if r]
    Z = {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}
    m = (np.abs(Z["om_act"]) > 0.3) & (Z["v_lon"] > 0.1)
    bins = np.arange(0.1, 0.85, 0.05); xc, yc = [], []
    for lo in bins:
        s = m & (Z["v_lon"] >= lo) & (Z["v_lon"] < lo + .05)
        if s.sum() > 20: xc.append(lo + .025); yc.append(np.median(Z["v_lat"][s] / Z["om_act"][s]))
    ax[1].plot(xc, yc, "o-", color=col, label=lab)
ax[1].axhline(-0.232, color="gray", ls=":", label="half wheelbase (turn centre on an axle)")
ax[1].set_xlabel("forward speed  [m/s]"); ax[1].set_ylabel("median v_lat / ω  [m]")
ax[1].set_title("Where the turn centre sits (|ω| > 0.3 rad/s)", fontsize=10); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tyre_calibration.png"), dpi=110)
print(ks, contacts)
