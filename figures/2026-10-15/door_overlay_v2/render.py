#!/usr/bin/env python3
"""Door passages per tyre/speed regime, as bands of lateral offset from the plan (paper figure, v2 of tyre_three_plants).

Rows: original tyre 0.448 m/s / calibrated tyre 0.448 m/s / calibrated tyre 0.9 m/s, all mu 0.5 patches and
ground-truth pose. Columns: TVLQR, TVLQR + fixed side-slip k, per-plan door-line deviation. Door detection, door
frame and the door-line deviation are those of door_overlay.py / door_margin.py, unchanged.
usage: render.py   (writes door_overlay_v2.png/.pdf and counts.txt here)
"""
import collections, glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../.."))
SRC = os.path.join(R, "figures/2026-10-12/door_overlay/door_overlay.py")
sys.argv = [sys.argv[0]]
g = {"__file__": SRC}; exec(open(SRC).read().split("\nNR = ")[0], g)
doors, to_frame, HL, HW, RD = g["doors"], g["to_frame"], g["HL"], g["HW"], g["RD"]

XG = np.linspace(-1.2, 1.2, 121)
J255, J257, J259 = "2026-10-12_job255_steerable", "2026-10-12_job257_kprior", "2026-10-12_job259_dev-replicate"
J265, J268 = "2026-10-12_job265_tyre-grid", "2026-10-13_job268_door-deviation"
ROWS = [("Original tire, $v_{\\max}$ = 0.45 m/s", -0.33, f"{J255}:TR+{J259}:TR", f"{J257}:FT+{J259}:FT", False),
        ("Calibrated tire, $v_{\\max}$ = 0.45 m/s", -0.10, f"{J265}/test:TR+{J265}/dev:TR", f"{J265}/test:FT+{J265}/dev:FT", False),
        ("Calibrated tire, $v_{\\max}$ = 0.9 m/s", -0.10, f"{J268}/test:TR+{J268}/dev:TR", f"{J268}/test:FT+{J268}/dev:FT", True)]
C_R, C_K, C_HIT = "#d95f02", "#1f5fa8", "#c00000"


def runs(spec):
    for one in spec.split("+"):
        job, cell = one.rsplit(":", 1)
        for f in sorted(glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl")):
            for l in open(f):
                if not l.strip(): continue
                r = json.loads(l)
                tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
                yield r, (np.load(tp)["track"] if os.path.isfile(tp) else None)


_plans = {}
def plan(pp):
    if pp not in _plans: _plans[pp] = np.load(os.path.join(R, pp.replace("/home/programmer/", "")))["poses"]
    return _plans[pp]


def crossing(q):
    i = np.where(np.diff(np.sign(q[:, 0])) > 0)[0]
    return int(i[0]) if len(i) else None


def y_at_door(q):                                     # door_margin.py, verbatim logic
    i = crossing(q)
    if i is None: return np.nan
    a, b = q[i], q[i + 1]
    return float(a[1] + (b[1] - a[1]) * (-a[0]) / (b[0] - a[0]))


def project(pts, A, B, sA):
    """nearest point on the polyline segments A->B: arc length, signed offset (+ = left of travel)"""
    AB = B - A; L2 = np.maximum((AB ** 2).sum(1), 1e-12); L = np.sqrt(L2)   # plans repeat poses at rest
    t = np.clip(((pts[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    F = A[None] + t[..., None] * AB[None]
    d2 = ((pts[:, None, :] - F) ** 2).sum(2); j = d2.argmin(1); n = np.arange(len(pts))
    rel = pts - F[n, j]
    side = np.sign(AB[j, 0] * rel[:, 1] - AB[j, 1] * rel[:, 0])
    return sA[j] + t[n, j] * L[j], side * np.sqrt(d2[n, j]), j


def plan_relative(P, T, o, h, m):
    """track as (offset from plan, heading error) on XG = arc length along the plan from its door-line crossing.
    The plan is straightened, so a turn through the door becomes a straight band; offsets are mirrored (+ = inside of
    the turn). Returns (e, dth, s_of_contact_fn) or None if either path does not cross the door line."""
    qp, q = to_frame(P[:, :2], o, h, m), to_frame(T[:, 1:3], o, h, m)
    ip, it = crossing(qp), crossing(q)
    if ip is None or it is None: return None
    sp = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    f = -qp[ip, 0] / (qp[ip + 1, 0] - qp[ip, 0]); s0 = sp[ip] + f * (sp[ip + 1] - sp[ip])
    w = np.where(np.abs(sp - s0) < XG[-1] + 0.6)[0]; w = np.arange(max(w[0] - 1, 0), min(w[-1] + 2, len(sp)))
    A, B, sA = P[w[:-1], :2], P[w[1:], :2], sp[w[:-1]] - s0
    thp = np.unwrap(P[w[:-1], 2])
    st = np.r_[0, np.cumsum(np.hypot(*np.diff(T[:, 1:3], axis=0).T))]
    k = np.where(np.abs(st - st[it]) < XG[-1] + 0.8)[0]
    s, e, j = project(T[k, 1:3], A, B, sA)
    dth = (T[k, 3] - thp[j] + np.pi) % (2 * np.pi) - np.pi
    o_ = np.argsort(s, kind="stable")
    E = np.interp(XG, s[o_], e[o_], left=np.nan, right=np.nan) * m
    TH = np.interp(XG, s[o_], dth[o_], left=np.nan, right=np.nan) * m
    def at(i):          # (s, e, dth) of track sample i, mirrored
        if not k[0] <= i <= k[-1]: return None
        sc, ec, jc = project(T[i:i + 1, 1:3], A, B, sA)
        return sc[0], m * ec[0], m * (((T[i, 3] - thp[jc[0]]) + np.pi) % (2 * np.pi) - np.pi)
    return E, TH, at


def passages(spec):
    """one record per (run, door) whose track crosses the door line"""
    out, nruns, ncontact = [], 0, 0
    for r, T in runs(spec):
        nruns += 1
        hit = (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None
        ncontact += bool((r.get("wall_overlap_s") or 0) > 0)
        if T is None or len(T) < 5: continue
        P = plan(r["plan_path"]); D = doors(r["plan_path"])
        ic = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1) if hit else None
        own = int(np.argmin([np.hypot(*to_frame(T[ic:ic + 1, 1:3], o, h, m)[0]) for o, h, m, *_ in D])) if hit and D else None
        for di, (o, h, m, _, width) in enumerate(D):
            q, qp = to_frame(T[:, 1:3], o, h, m), to_frame(P[:, :2], o, h, m)
            dev = y_at_door(q) - y_at_door(qp)
            if not np.isfinite(dev): continue
            pr = plan_relative(P, T, o, h, m)
            if pr is None: continue
            E, TH, at = pr
            here = bool(hit and di == own and np.hypot(*to_frame(T[ic:ic + 1, 1:3], o, h, m)[0]) < 1.0)
            y0p = y_at_door(qp)
            out.append(dict(plan=os.path.basename(r["plan"]), door=di, seed=r["seed"], dev=dev, d=E, dth=TH, here=here,
                            ex=at(ic) if here else None, jamb_out=-width / 2 - y0p, jamb_in=width / 2 - y0p))
    return out, nruns, ncontact


def per_plan(ps):
    acc = collections.defaultdict(list)
    for p in ps: acc[p["plan"]].append(p["dev"])
    return {k: np.mean(v) for k, v in acc.items()}


def robot(ax, x, y, th, color, fill, lw=0.9, z=6, alpha=1.0):
    c, s_ = np.cos(th), np.sin(th); Rm = np.array([[c, -s_], [s_, c]])
    def rect(cx, cy, L, W):
        return (np.array([[-L, -W], [L, -W], [L, W], [-L, W]]) / 2 + [cx, cy]) @ Rm.T + [x, y]
    ax.add_patch(Polygon(rect(0, 0, 2 * HL, 2 * HW), fc=color if fill else "none", alpha=alpha * (0.25 if fill else 1),
                         ec=color, lw=lw, zorder=z))
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.add_patch(Polygon(rect(sx * 0.21, sy * 0.21, 0.18, 0.08), fc="0.25" if fill else "none",
                                 ec=color, lw=lw * 0.7, alpha=alpha, zorder=z))
    nose = np.array([[0.03, -0.13], [HL - 0.04, 0], [0.03, 0.13]]) @ Rm.T + [x, y]
    ax.plot(*nose.T, color=color, lw=lw + 0.4, alpha=alpha, zorder=z + 0.1)


plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(6.3, 7.2))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 0.58], wspace=0.08, hspace=0.42,
                      left=0.10, right=0.915, top=0.865, bottom=0.07)
rng = np.random.default_rng(7)
lines, YL = [], 0.78
for ri, (rlab, k, sR, sK, registered) in enumerate(ROWS):
    data = {}
    for col, (spec, colr, name) in enumerate([(sR, C_R, "TVLQR"), (sK, C_K, "TVLQR + side-slip model")]):
        ps, nruns, ncont = passages(spec); data[col] = ps
        ax = fig.add_subplot(gs[ri, col])
        D = np.array([p["d"] for p in ps]); TH = np.array([p["dth"] for p in ps])
        ok = np.isfinite(D).sum(0) >= 0.8 * len(ps)
        q05, q25, q50, q75, q95 = (np.where(ok, np.nanpercentile(D, p, axis=0), np.nan) for p in (5, 25, 50, 75, 95))
        jo, ji = np.median([p["jamb_out"] for p in ps]), np.median([p["jamb_in"] for p in ps])
        for y0, y1 in ((-YL, jo), (ji, YL)):      # jambs at the median distance of each jamb from the planned line
            ax.add_patch(plt.Rectangle((-0.06, y0), 0.12, y1 - y0, fc="0.15", ec="none", zorder=7))
        ax.axhline(0, color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=2)
        ax.fill_between(XG, q05, q95, color=colr, alpha=0.20, lw=0, zorder=3)
        ax.fill_between(XG, q25, q75, color=colr, alpha=0.45, lw=0, zorder=3)
        ax.plot(XG, q50, color=colr, lw=1.6, zorder=5)
        i0 = np.argmin(np.abs(XG))
        robot(ax, 0.0, q50[i0], np.nanmedian(TH[:, i0]), colr, fill=False, lw=0.7, alpha=0.8)
        hits = [p for p in ps if p["here"] and p["ex"] is not None]
        for p in ([hits[i] for i in sorted(rng.choice(len(hits), min(2, len(hits)), replace=False))] if hits else []):
            ax.plot(XG, p["d"], color=C_HIT, lw=0.9, zorder=4)
            xc, yc, tc = p["ex"]; c_, s_ = np.cos(tc), np.sin(tc)
            body = np.array([[-HL, -HW], [HL, -HW], [HL, HW], [-HL, HW]]) @ np.array([[c_, s_], [-s_, c_]]) + [xc, yc]
            ax.add_patch(Polygon(body, fc="none", ec=C_HIT, lw=0.9, ls=(0, (2, 1)), zorder=6))
        nh = sum(p["here"] for p in ps)
        ax.text(0.02, 0.975, f"contacts\n{nh} / {len(ps)}", transform=ax.transAxes, va="top", ha="left",
                fontsize=8.5, zorder=9, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        ax.set_xlim(XG[0], XG[-1]); ax.set_ylim(-YL, YL); ax.set_aspect("equal")
        ax.set_xticks([-1, 0, 1]); ax.set_yticks([-0.5, 0, 0.5])
        if col == 0:
            if ri == 1: ax.set_ylabel("lateral offset from plan (m)")
        else: ax.set_yticklabels([])
        if ri < 2: ax.set_xticklabels([])
        else: ax.set_xlabel("along plan from door line (m)")
        if col == 1: ax.text(0.98, 0.03, f"k = {k:.2f} s".replace("-", "−"), transform=ax.transAxes, ha="right",
                             va="bottom", fontsize=8.5, color=C_K, zorder=9,
                             bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        if ri == 0:
            bb = ax.get_position()
            fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.045, name, ha="center", va="bottom", fontweight="bold", color=colr)
        devs = [p["dev"] for p in ps]
        lines.append(f"{rlab.replace(chr(36), '')} | {name}: door contacts {nh} / {len(ps)} door passages "
                     f"({len({(p['plan'], p['door']) for p in ps})} plan-doors, {len({p['plan'] for p in ps})} plans); "
                     f"runs with any contact {ncont} / {nruns}; door-line deviation median {np.median(devs):+.3f} m "
                     f"[5-95% {np.percentile(devs, 5):+.3f}, {np.percentile(devs, 95):+.3f}]; "
                     f"band at s=0: median {q50[i0]:+.3f} IQR [{q25[i0]:+.3f}, {q75[i0]:+.3f}] 5-95 [{q05[i0]:+.3f}, {q95[i0]:+.3f}]; "
                     f"jambs (median, from plan) {jo:+.2f} / {ji:+.2f} m")
        if col == 0:
            bb = ax.get_position()
            fig.text(0.10, bb.y1 + 0.008, f"({'abc'[ri]}) {rlab}", fontsize=9.5, fontweight="bold", va="bottom", ha="left")
    # column 3: per-plan mean door-line deviation, paired by plan (the unit of tab:highspeed)
    ax = fig.add_subplot(gs[ri, 2])
    A, B = per_plan(data[0]), per_plan(data[1]); ks = sorted(set(A) & set(B))
    a, b = np.array([A[x] for x in ks]) * 100, np.array([B[x] for x in ks]) * 100
    LO, HI = -30, 15
    for u, v in zip(a, b): ax.plot([0, 1], np.clip([u, v], LO + 0.5, HI - 0.5), color="0.65", lw=0.5, zorder=1)
    jit = rng.uniform(-0.08, 0.08, (2, len(ks)))
    ax.scatter(jit[0], np.clip(a, LO + 0.5, HI - 0.5), s=6, color=C_R, lw=0, zorder=2)
    ax.scatter(1 + jit[1], np.clip(b, LO + 0.5, HI - 0.5), s=6, color=C_K, lw=0, zorder=2)
    for xx, v, c in ((0, a, C_R), (1, b, C_K)):
        ax.plot([xx - 0.28, xx + 0.28], [np.median(v)] * 2, color=c, lw=2.4, zorder=3, solid_capstyle="butt")
    ax.axhline(0, color="0.45", lw=0.8, ls=(0, (4, 3)), zorder=0)
    p = wilcoxon(b - a).pvalue
    ax.set_title(f"{len(ks)} plans" + (f", p = {p:.0e}".replace("e-0", "e−") if registered else "")
                 + f"\nshift {np.median(b - a):+.1f} cm".replace("-", "−"), fontsize=8.5, pad=3)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(LO, HI); ax.set_xticks([0, 1]); ax.set_yticks([-30, -20, -10, 0, 10])
    ax.set_xticklabels(["TVLQR", "+ model"] if ri == 2 else [])
    ax.yaxis.tick_right(); ax.yaxis.set_label_position("right")
    if ri == 1: ax.set_ylabel("deviation at door line (cm)")
    if ri == 0:
        bb = ax.get_position()
        fig.text((bb.x0 + bb.x1) / 2, bb.y1 + 0.045, "per-plan mean", ha="center", va="bottom", fontweight="bold")
    clipped = int(((a < LO) | (a > HI)).sum() + ((b < LO) | (b > HI)).sum())
    lines.append(f"   per-plan ({len(ks)} plans): TVLQR median {np.median(a):+.1f} cm, +model median {np.median(b):+.1f} cm, "
                 f"paired shift median {np.median(b - a):+.1f} cm, inward on {(b > a).sum()}/{len(ks)}, Wilcoxon p {p:.2g}"
                 f"{' (registered)' if registered else ' (descriptive)'}; points clipped to the axis: {clipped}")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
fig.legend(handles=[Line2D([], [], color="0.3", lw=1.6, label="median"),
                    Patch(fc="0.3", alpha=0.55, label="25–75%"), Patch(fc="0.3", alpha=0.22, label="5–95%"),
                    Line2D([], [], color=C_HIT, lw=0.9, label="wall contact (example)"),
                    Line2D([], [], color="0.45", lw=0.8, ls=(0, (4, 3)), label="plan")],
           loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.997), frameon=False, handlelength=1.5, columnspacing=0.9,
           handletextpad=0.4)
fig.text(0.5, 0.952, "Plan straightened, travel left to right, turns mirrored: negative offset = outside of the turn.",
         ha="center", fontsize=8.5, color="0.2")
for ext in ("png", "pdf"): fig.savefig(os.path.join(HERE, f"door_overlay_v2.{ext}"), dpi=300)
open(os.path.join(HERE, "counts.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
