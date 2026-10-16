"""Paper version of figures/2026-10-12/tyre_calibration (jobs 263/264) as one two-panel figure.
(a) fitted side-slip k and wall-contact plans against WheelSlip lateral compliance, real-robot band shaded;
(b) median v_lat/omega against forward speed for the old and the chosen tyre.
Reads run_data/2026-10-12_job26{3,4}_*/<cell>/{yaw.json,parts/,s0/}.  python3 render.py (repo .venv)."""
import glob, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning")]
import yaw_response as Y
from agx_planning.pmp_planner.config import PlannerConfig
cfg = PlannerConfig(mode="offline")
OUT = os.path.dirname(os.path.abspath(__file__))
cells = {"L1u": 1e-6, "L02": .02, "L05": .05, "L10": .1, "L20": .2, "L50": .5, "L100": 1.0, "L200": 2.0}
def d(c): return glob.glob(os.path.join(R, f"run_data/2026-10-12_job26[34]_*/{c}"))[0]
ks = {c: json.load(open(os.path.join(d(c), "yaw.json")))["k"] for c in cells}
k0 = json.load(open(os.path.join(d("L0"), "yaw.json")))["k"]
contacts = {}
for c in ["L0", *cells]:
    rows = [json.loads(l) for f in glob.glob(os.path.join(d(c), "parts/*.jsonl")) for l in open(f)]
    contacts[c] = sum((r.get("wall_overlap_s") or 0) > 0 for r in rows), len(rows)

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 7})
fig, ax = plt.subplots(1, 2, figsize=(5.6, 3.1))
x = np.array(list(cells.values())); y = np.array([ks[c] for c in cells])
a = ax[0]
a.axhspan(-0.16, -0.03, color="tab:green", alpha=.15, label="physical robot")
a.axhline(k0, color="tab:red", ls="--", lw=1, label="original tire")
a.semilogx(x, y, "o-", color="k", ms=4, label="side-slip coefficient $k$")
a.plot([1.0], [ks["L100"]], "o", ms=10, mfc="none", mec="tab:green", mew=1.5, label="selected setting")
a.set_xlabel("lateral slip compliance"); a.set_ylabel("$k$ [s]"); a.set_ylim(-0.4, 0.02)
b = a.twinx()
n = np.array([contacts[c][0] for c in cells]); N = contacts["L1u"][1]
b.semilogx(x, n, "s:", color="tab:blue", ms=3.5, mfc="none", lw=1)
b.set_ylabel(f"plans with wall contact (of {N})", color="tab:blue"); b.tick_params(axis="y", colors="tab:blue")
b.set_ylim(0, 40)
h, l = a.get_legend_handles_labels()
h.append(plt.Line2D([], [], color="tab:blue", ls=":", marker="s", mfc="none", ms=3.5)); l.append("wall-contact plans")
a.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
a.set_title("(a)", loc="left", fontweight="bold")
for c, lab, col in [("L0", "original tire", "tab:red"), ("L100", "calibrated tire", "tab:green")]:
    fs = sorted(glob.glob(os.path.join(d(c), "s0/**/track_*.npz"), recursive=True))
    rs = [r for r in (Y.intervals(f, cfg.c_v, cfg.c_w) for f in fs) if r]
    Z = {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}
    m = (np.abs(Z["om_act"]) > 0.3) & (Z["v_lon"] > 0.1)
    xc, yc = [], []
    for lo in np.arange(0.1, 0.85, 0.05):
        s = m & (Z["v_lon"] >= lo) & (Z["v_lon"] < lo + .05)
        if s.sum() > 20: xc.append(lo + .025); yc.append(np.median(Z["v_lat"][s] / Z["om_act"][s]))
    ax[1].plot(xc, yc, "o-", color=col, ms=4, label=lab)
ax[1].axhline(-0.232, color="gray", ls=":", label="half wheelbase")
ax[1].set_xlabel("forward speed [m s$^{-1}$]"); ax[1].set_ylabel(r"median $v_\perp/\omega$ [m]")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False); ax[1].set_title("(b)", loc="left", fontweight="bold")
fig.tight_layout()
for ext in ("pdf", "png"): fig.savefig(os.path.join(OUT, f"tyre_calibration.{ext}"), dpi=200, bbox_inches="tight")
print({c: round(v, 3) for c, v in ks.items()}, "k0", round(k0, 3), contacts)
