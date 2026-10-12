#!/usr/bin/env python3
"""Phone dashboard: portrait PNGs with big fonts, one finding per chart, the title states it.

Chart design from a Fable brainstorm (2026-10-12). Reads jobs 253/255/256/257 from run_data/ and writes
NN_*.png next to this file. Chart 6 (job 257 verdict) leaves P/F bars empty until job 257's rows exist.
Contact = wall_overlap_s > 0 (as everywhere since job 251). Paired sign tests over (plan, seed).
usage: dashboard.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "../../../run_data")
J253, J255, J256, J257 = ("2026-10-11_job253_rtab-test", "2026-10-12_job255_steerable",
                          "2026-10-12_job256_steerable-rtab", "2026-10-12_job257_kprior")
plt.rcParams.update({"font.size": 20, "axes.titlesize": 22, "axes.titleweight": "bold"})
FIG = (10.8, 16)
GREY, BLUE, GREEN, ORANGE = "#9e9e9e", "#1f77b4", "#2ca02c", "#ff7f0e"


def load(job, cell):
    d = {}
    for f in glob.glob(os.path.join(RD, job, cell, "rows.s*.jsonl")) + glob.glob(os.path.join(RD, job, cell, "parts", "*.jsonl")):
        for l in open(f):
            if l.strip():
                r = json.loads(l); d[(os.path.basename(r["plan"]), r["seed"])] = r
    return d


con = lambda r: (r.get("wall_overlap_s") or 0) > 0
arr = lambda r: r.get("outcome") == "arrived"


def sign_p(A, B):
    """A better (fewer contacts) / B better over shared keys, and the two-sided sign-test p."""
    ks = set(A) & set(B)
    a = sum(con(B[k]) and not con(A[k]) for k in ks); b = sum(con(A[k]) and not con(B[k]) for k in ks)
    return (binomtest(a, a + b).pvalue if a + b else 1.0), len(ks)


def ptxt(p):
    return f"p={p:.0e}" if p < 1e-3 else f"p={p:.2f}"


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(HERE, name), dpi=100); plt.close(fig)


# ---- 1. headline
G = [("true pose", J255, "T", J255, "TR"), ("amcl", J255, "Q0", J255, "QR"), ("rtab", J256, "RQ", J256, "RR")]
fig, ax = plt.subplots(figsize=FIG)
for i, (lab, jq, cq, jr, cr) in enumerate(G):
    Q, R_ = load(jq, cq), load(jr, cr)
    nq, nr = sum(map(con, Q.values())), sum(map(con, R_.values()))
    ax.bar(i - 0.2, nr, 0.38, color=GREY, label="TVLQR" if i == 0 else None)
    ax.bar(i + 0.2, nq, 0.38, color=BLUE, label="TVLQG" if i == 0 else None)
    ax.text(i - 0.2, nr + 1, str(nr), ha="center"); ax.text(i + 0.2, nq + 1, str(nq), ha="center")
    p, _ = sign_p(Q, R_)
    ax.text(i, max(nq, nr) + 6, ptxt(p), ha="center", fontsize=18)
ax.set_xticks(range(len(G))); ax.set_xticklabels([g[0] for g in G])
ax.set_ylabel("runs that touched a wall (of 126)"); ax.set_ylim(0, 75); ax.legend(loc="upper left")
ax.set_title("TVLQG hits walls less than TVLQR\nunder every localizer\n(slippery patches mu 0.5, 42 plans x 3 seeds)")
save(fig, "01_headline.png")

# ---- 2. stock patches -> mu 0.5
L = [("TVLQG true pose", "T", J255, "T", BLUE, "-"), ("TVLQG amcl", "Q0", J255, "Q0", BLUE, "-"),
     ("TVLQG rtab", "RQ", J256, "RQ", BLUE, "-"), ("TVLQR true pose", "TR", J255, "TR", GREY, "--"),
     ("TVLQR amcl", "QR", J255, "QR", GREY, "--"), ("TVLQR rtab", "RR", J256, "RR", GREY, "--")]
fig, ax = plt.subplots(figsize=FIG)
for lab, c253, j, c, col, ls in L:
    a, b = load(J253, c253), load(j, c)
    ks = set(a) & set(b)
    y0, y1 = sum(con(a[k]) for k in ks), sum(con(b[k]) for k in ks)
    ax.plot([0, 1], [y0, y1], ls, color=col, lw=4, marker="o", ms=12)
    ax.text(1.04, y1, f"{lab}  {y0}->{y1}", va="center", fontsize=17)
ax.set_xticks([0, 1]); ax.set_xticklabels(["stock patches\n(cannot steer)", "mu 0.5 patches\n(steerable)"])
ax.set_xlim(-0.15, 1.9); ax.set_ylabel("runs that touched a wall (of 126)"); ax.set_ylim(0, 70)
ax.set_title("Steerable patches helped TVLQG,\nnot TVLQR\n(same plans, seeds, patch layouts)")
save(fig, "02_stock_vs_mu05.png")

# ---- 4. localizer error at contact
src = open(os.path.join(HERE, "../job256_steerable_rtab/err_at_contact.txt")).read().splitlines()
vals = {}
for l in src:
    w = l.split()
    # "RQ contact runs: loc err in 1 s before first contact, median 0.073 p90 0.163 (n 37) | clean runs median err 0.031"
    vals[w[0]] = (float(w[w.index("median") + 1]), float(l.rsplit(" ", 1)[1]))
fig, ax = plt.subplots(figsize=FIG)
rows = [("rtab, clean runs", vals["RQ"][1]), ("rtab, 1 s before contact", vals["RQ"][0]),
        ("amcl, clean runs", vals["Q0"][1]), ("amcl, 1 s before contact", vals["Q0"][0])]
ax.barh(range(4), [v for _, v in rows], color=[GREEN, ORANGE, GREEN, ORANGE])
for i, (_, v) in enumerate(rows):
    ax.text(v + 0.005, i, f"{v * 100:.0f} cm", va="center")
ax.axvline(0.15, color="k", ls="--"); ax.text(0.152, 3.45, "door margin\n15 cm / side", fontsize=16)
ax.set_yticks(range(4)); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis()
ax.set_xlabel("median position error of the localizer (m)"); ax.set_xlim(0, 0.32)
ax.set_title("Before a wall touch the localizer\nis off by 7 cm (rtab) / 26 cm (amcl)\nvs a 15 cm door margin")
save(fig, "04_loc_error_at_contact.png")

# ---- 5. k_bar under each localizer
CELLS = {"true pose": (J255, "T"), "amcl": (J255, "Q0"), "rtab": (J256, "RQ")}
COL = {"true pose": "k", "amcl": ORANGE, "rtab": BLUE}
grid = np.arange(0, 40, 0.5)
fig, (a1, a2) = plt.subplots(2, 1, figsize=FIG, gridspec_kw={"height_ratios": [2, 1]})
act = {}
for lab, (job, cell) in CELLS.items():
    curves, ta = [], []
    for f in glob.glob(os.path.join(RD, job, cell, "s*", "track_*.npz")):
        z = np.load(f); D, t = z["diag"], z["diag_t"]
        mv = np.flatnonzero(D[:, 4] > 0.1) if len(D) else []
        if len(mv) == 0: continue
        tt = t - t[mv[0]]
        curves.append(np.interp(grid, tt, D[:, 18], left=np.nan, right=np.nan))
        on = np.abs(D[:, 18]) > 1e-9
        if on.any(): ta.append(tt[np.argmax(on)])
    C = np.asarray(curves)
    a1.plot(grid, np.nanmedian(C, 0), color=COL[lab], lw=4, label=lab)
    a1.fill_between(grid, np.nanpercentile(C, 25, 0), np.nanpercentile(C, 75, 0), color=COL[lab], alpha=0.12)
    act[lab] = np.median(ta)
a1.axhline(-0.33, color="r", ls="--"); a1.text(25, -0.315, "true sim slide -0.33", color="r", fontsize=16)
a1.set_xlabel("seconds since the robot started moving"); a1.set_ylabel("slide value TVLQG uses (k_bar)")
a1.legend(loc="lower left"); a1.set_ylim(-0.4, 0.05)
a2.bar(range(3), [act[l] for l in CELLS], color=[COL[l] for l in CELLS])
for i, l in enumerate(CELLS):
    a2.text(i, act[l] + 0.2, f"{act[l]:.1f} s", ha="center")
a2.set_xticks(range(3)); a2.set_xticklabels(list(CELLS)); a2.set_ylabel("median time to\nswitch on (s)")
a1.set_title("Under a real localizer TVLQG's slip\nestimate is weaker and switches on\n~7 s later (median, band = middle half)")
save(fig, "05_slip_estimate.png")

# ---- 6. job 257 verdict
GR = [("amcl", (J255, "QR"), (J255, "Q0"), (J257, "PQ"), (J257, "FQ")),
      ("rtab", (J256, "RR"), (J256, "RQ"), (J257, "PR"), (J257, "FR")),
      ("true pose", (J255, "TR"), (J255, "T"), (J257, "PT"), (J257, "FT"))]
NAMES = ["TVLQR", "TVLQG", "TVLQG + slip prior", "TVLQR + fixed slip\n(diagnostic)"]
COLS = [GREY, BLUE, GREEN, ORANGE]
fig, ax = plt.subplots(figsize=FIG)
verdict, ps, n257 = [], {}, 0
for i, (lab, *cells) in enumerate(GR):
    data = [load(*c) for c in cells]
    n257 += len(data[2]) + len(data[3])
    for j, d in enumerate(data):
        x = i + (j - 1.5) * 0.2
        if not d:
            ax.text(x, 2, "pending", rotation=90, ha="center", fontsize=14, color="0.5"); continue
        n = sum(map(con, d.values())); a = sum(map(arr, d.values()))
        ax.bar(x, n, 0.19, color=COLS[j])
        ax.text(x, n + 0.8, f"{n}", ha="center", fontsize=15)
        ax.text(x, -6, f"{a}", ha="center", fontsize=12, color="0.3")
    if data[2]:
        p, n = sign_p(data[2], data[1]); ps[lab] = p
        ax.text(i + 0.1, max(sum(map(con, d.values())) for d in data if d) + 6, f"prior vs TVLQG\n{ptxt(p)} (n {n})", ha="center", fontsize=14)
ax.axhline(7, color="k", ls=":"); ax.text(2.45, 8, "TVLQG,\ntrue pose\n(job 255)", fontsize=13, ha="right")
ax.text(-0.5, -6, "arrived:", fontsize=12, color="0.3", ha="right")
ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in GR]); ax.set_ylim(-9, 80); ax.set_xlim(-0.75, 2.5)
from matplotlib.patches import Patch
ax.set_ylabel("runs that touched a wall (of 126)")
ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLS, NAMES)], loc="upper right", fontsize=15)
if n257 == 0:
    title = "Job 257: does a slip prior close the gap?\n(running)"
else:
    done = f"{n257} of 756 runs"
    ok = all(ps.get(k, 1) < 0.01 for k in ("amcl", "rtab"))
    title = (f"Job 257: the slip prior closes part\nof the gap ({done})" if ok
             else f"Job 257: a slip prior does NOT\nclose the gap ({done})")
ax.set_title(title)
save(fig, "06_job257_verdict.png")
print("wrote", sorted(f for f in os.listdir(HERE) if f.endswith(".png")))

# ---- 7. rtab jumps before contact (exploratory; numbers from ../step_and_yaw/*.txt)
SY = os.path.join(HERE, "../step_and_yaw")
t1 = open(os.path.join(SY, "step_and_yaw.txt")).read(); t2 = open(os.path.join(SY, "step_direction.txt")).read()
import re
m = re.search(r"rtab B1 SAME DOORWAYS: step within 1.0 s: contacts (\d+)% vs clean runs passing a contact site (\d+)%", t1)
a = re.search(r"rtab: .*AWAY from the wall in (\d+), toward in (\d+)", t2)
fig, (a1, a2) = plt.subplots(2, 1, figsize=FIG)
a1.bar([0, 1], [int(m.group(2)), int(m.group(1))], color=[GREEN, "tab:red"])
for i, v in enumerate([int(m.group(2)), int(m.group(1))]): a1.text(i, v + 2, f"{v}%", ha="center")
a1.set_xticks([0, 1]); a1.set_xticklabels(["clean pass\nthrough the same doors", "wall hit"]); a1.set_ylim(0, 100)
a1.set_ylabel("% with a rtab pose jump\n> 3 cm in the last 1 s")
a1.set_title("rtab: 86% of wall hits come right\nafter a pose jump (27% of clean passes)")
a2.bar([0, 1], [int(a.group(2)), int(a.group(1))], color=["tab:red", "0.6"])
for i, v in enumerate([int(a.group(2)), int(a.group(1))]): a2.text(i, v + 0.5, str(v), ha="center")
a2.set_xticks([0, 1]); a2.set_xticklabels(["estimate jumped\nTOWARD the wall", "away from\nthe wall"]); a2.set_ylabel("wall hits")
a2.set_title("The jump is a LATE catch-up: the robot\nhad already slid toward the wall\n(exploratory, p 1e-4)")
save(fig, "07_rtab_late_jump.png")
