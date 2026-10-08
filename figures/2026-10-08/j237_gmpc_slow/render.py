#!/usr/bin/env python3
"""Job 237: does GMPC cut door corners, and does slowing the plan reduce it?

Orig vs door-slowed plans (same path, clock slowed where hull clearance < 0.40 m),
truth, phantom walls, seed 0. KTN = GMPC, LTN = ours-lib. Track = Gazebo GT.

Inward offset: the track's signed lateral offset from the plan at matched plan
arc-length, positive toward the centre of the plan's turn (corner cutting).
    .venv/bin/python figures/2026-10-08/j237_gmpc_slow/render.py
"""
import glob, json, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from PIL import Image
from scipy.stats import binomtest

R = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(R, "src/agx_navigation/agx_planning"))
from agx_planning.supervisor.walls import SignedWallDistance
D = os.path.join(R, "run_data/2026-10-08_job237_door-slow-ab/")
OUT = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(R, "src/rudn-ordjo-building/maps/floor_6")
swd = SignedWallDistance.from_map_yaml(MAP + ".yaml")
meta = yaml.safe_load(open(MAP + ".yaml")); img = np.asarray(Image.open(MAP + ".png").convert("L"))
res, (ox, oy, _) = meta["resolution"], meta["origin"]
EXT = [ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res]
HL, HW = 0.315, 0.2925                       # Scout Mini hull half-length / half-width
TIGHT_CLEAR, TIGHT_KAPPA, SEG_PAD = 0.40, 0.8, 0.3
WIN_BACK, WIN_FWD, MIN_TURN = 1.0, 0.5, np.radians(20)
CFG = {"KTN": "GMPC", "LTN": "ours-lib"}


def load(sub, cfg):
    rows = {}
    for l in open(os.path.join(D, sub, "all_rows.jsonl")):
        r = json.loads(l)
        if f"/{cfg}/" in r.get("track_path", ""):
            z = np.load(os.path.join(D, r["track_path"].split("door-slow-ab/")[1]))
            r["T"], r["P"] = z["track"], z["plan_poses"]
            rows[r["plan"]] = r
    return rows


def plan_geom(P):
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(P[:, :2], axis=0).T))]
    yaw = np.unwrap(P[:, 2])
    ds = np.gradient(d); kappa = np.gradient(yaw) / np.where(ds > 1e-4, ds, np.inf)
    return d, yaw, kappa


def project(T, P, s):
    """Monotone projection of track samples onto the plan -> (arc length, left offset)."""
    k, out_s, out_e = 0, [], []
    tx, ty = np.cos(P[:, 2]), np.sin(P[:, 2])
    for x, y in T[:, 1:3]:
        lo, hi = max(0, k - 5), min(len(P), k + 60)
        j = lo + int(np.argmin(np.hypot(P[lo:hi, 0] - x, P[lo:hi, 1] - y)))
        k = max(k, j)
        out_s.append(s[j]); out_e.append(tx[j] * (y - P[j, 1]) - ty[j] * (x - P[j, 0]))
    return np.array(out_s), np.array(out_e)


def turn_sign(s, yaw, a, b):
    i, j = np.searchsorted(s, a), min(np.searchsorted(s, b), len(s) - 1)
    dy = yaw[j] - yaw[max(i, 0)]
    return 0 if abs(dy) < MIN_TURN else int(np.sign(dy))


def max_inward(T, P, a, b):
    s, yaw, _ = plan_geom(P)
    sg = turn_sign(s, yaw, a, b)
    ts, te = project(T, P, s)
    m = (ts >= a) & (ts <= b)
    if sg == 0 or not m.any():
        return None
    return float((sg * te[m]).max())


def first_contact(T):
    c = np.array([swd.body_clearance(*p) for p in T[:, 1:4]])
    return int(np.argmax(c < 0)) if (c < 0).any() else None


def tight_segments(P):
    """Plan stretches that are both tight (hull clearance < 0.40 m) and turning."""
    s, _, kappa = plan_geom(P)
    c = np.array([swd.body_clearance(*p) for p in P])
    m = (c < TIGHT_CLEAR) & (np.abs(kappa) > TIGHT_KAPPA)
    segs, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j + 1 < len(m) and m[j + 1]:
                j += 1
            segs.append((s[i] - SEG_PAD, s[j] + SEG_PAD)); i = j + 1
        else:
            i += 1
    return segs


def contact_geometry(P, T, i):
    """At the first contact: offset toward the turn, heading error toward the
    turn, and whether the hull point that touches is on the turn's inner side."""
    s, yaw_p, _ = plan_geom(P)
    ts, te = project(T, P, s)
    sg = turn_sign(s, yaw_p, ts[i] - WIN_BACK, ts[i] + WIN_FWD)
    j = min(int(np.searchsorted(s, ts[i])), len(P) - 1)
    x, y, yaw = T[i, 1:4]; c, sn = np.cos(yaw), np.sin(yaw); b = swd.body
    qx, qy = x + b[:, 0] * c - b[:, 1] * sn, y + b[:, 0] * sn + b[:, 1] * c
    col = np.clip(((qx - swd.origin_x) / swd.resolution).astype(int), 0, swd.dist.shape[1] - 1)
    row = np.clip(((qy - swd.origin_y) / swd.resolution).astype(int), 0, swd.dist.shape[0] - 1)
    hy = b[int(np.argmin(swd.dist[row, col])), 1]
    dyaw = (yaw - P[j, 2] + np.pi) % (2 * np.pi) - np.pi
    plan_clear = min(swd.body_clearance(*P[m]) for m in range(max(0, j - 10), min(len(P), j + 10)))
    return dict(sg=sg, off=te[i] * sg, dyaw=np.degrees(dyaw) * sg, inner=bool(np.sign(hy) == sg),
                plan_clear=plan_clear)


def sign_p(a, b):
    a, b = np.asarray(a), np.asarray(b)
    lower, higher = int((b < a).sum()), int((b > a).sum())
    p = binomtest(lower, lower + higher).pvalue if lower + higher else 1.0
    return lower, higher, p


stats, lines = {}, []
data = {c: (load("orig", c), load("slow", c)) for c in CFG}
for cfg, (O, S) in data.items():
    contact, tight = [], []
    for plan in sorted(set(O) & set(S)):
        o, sl = O[plan], S[plan]
        s, _, _ = plan_geom(o["P"])
        i = first_contact(o["T"])
        if i is not None and (o["wall_overlap_s"] or 0) > 0:
            ts, te = project(o["T"], o["P"], s)
            sc = ts[i]
            a, b = sc - WIN_BACK, sc + WIN_FWD
            vo, vs = max_inward(o["T"], o["P"], a, b), max_inward(sl["T"], sl["P"], a, b)
            contact.append(dict(plan=plan, sc=sc, io=vo, is_=vs, slow_contact=(sl["wall_overlap_s"] or 0) > 0,
                                geo=contact_geometry(o["P"], o["T"], i)))
        segs = tight_segments(o["P"])
        po = [max_inward(o["T"], o["P"], a, b) for a, b in segs]
        ps = [max_inward(sl["T"], sl["P"], a, b) for a, b in segs]
        pr = [(x, y) for x, y in zip(po, ps) if x is not None and y is not None]
        if pr:
            tight.append(dict(plan=plan, io=np.mean([x for x, _ in pr]), is_=np.mean([y for _, y in pr]), n=len(pr)))
    stats[cfg] = dict(contact=contact, tight=tight)
    turned = [c for c in contact if c["io"] is not None and c["is_"] is not None]
    inward = sum(c["io"] > 0 for c in turned)
    lo, hi, p = sign_p([c["io"] for c in turned], [c["is_"] for c in turned])
    lines.append(f"{CFG[cfg]} ({cfg}): orig contacts {len(contact)}, at a turn {len(turned)}; "
                 f"track inside the turn at contact {inward}/{len(turned)}; "
                 f"median max inward offset orig {np.median([c['io'] for c in turned]):.3f} m -> slow "
                 f"{np.median([c['is_'] for c in turned]):.3f} m; slow smaller {lo}/{lo + hi}, sign p={p:.3g}")
    g = [c["geo"] for c in contact if c["geo"]["sg"] != 0]
    lines.append(f"{CFG[cfg]} ({cfg}) at first contact (turning: {len(g)}): touching hull point on the INNER side "
                 f"{sum(x['inner'] for x in g)}/{len(g)}; median centre offset toward the turn "
                 f"{np.median([x['off'] for x in g]):+.3f} m; median heading error toward the turn "
                 f"{np.median([x['dyaw'] for x in g]):+.1f} deg; median plan hull clearance there "
                 f"{np.median([x['plan_clear'] for x in g]):.3f} m")
    lo, hi, p = sign_p([t["io"] for t in tight], [t["is_"] for t in tight])
    lines.append(f"{CFG[cfg]} ({cfg}) all tight turns (clear<{TIGHT_CLEAR} m, |kappa|>{TIGHT_KAPPA}/m): "
                 f"{len(tight)} plans, {sum(t['n'] for t in tight)} turns; median per-plan mean inward "
                 f"orig {np.median([t['io'] for t in tight]):.3f} -> slow {np.median([t['is_'] for t in tight]):.3f} m; "
                 f"slow smaller {lo}/{lo + hi}, sign p={p:.3g}")
open(os.path.join(OUT, "stats.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))


# --- figure 1: GMPC contact sites, orig vs slow --------------------------------
def hull(ax, x, y, yaw, color):
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.array([[HL, HW], [HL, -HW], [-HL, -HW], [-HL, HW]])
    ax.add_patch(Polygon(np.c_[x + pts[:, 0] * c - pts[:, 1] * s, y + pts[:, 0] * s + pts[:, 1] * c],
                         closed=True, fill=False, ec=color, lw=1.8))


def contact_sites(cfg, name, fname):
    O, S = data[cfg]
    con = stats[cfg]["contact"]
    fixed = [c for c in con if not c["slow_contact"]]
    kept = [c for c in con if c["slow_contact"]]
    show = fixed + kept[:max(3, 15 - len(fixed))]
    nc = 5; nr = -(-len(show) // nc)
    fig, axs = plt.subplots(nr, nc, figsize=(4 * nc, 4.7 * nr))
    for ax in np.atleast_1d(axs).flat: ax.axis("off")
    for ax, c in zip(np.atleast_1d(axs).flat, show):
        o, sl = O[c["plan"]], S[c["plan"]]
        P = o["P"]; s, _, _ = plan_geom(P)
        i = first_contact(o["T"]); cx, cy = o["T"][i, 1], o["T"][i, 2]
        ts, _ = project(sl["T"], sl["P"], plan_geom(sl["P"])[0])
        j = int(np.argmin(np.abs(ts - c["sc"])))
        ax.imshow(img, cmap="gray", extent=EXT, origin="upper", vmin=0, vmax=255)
        ax.plot(P[:, 0], P[:, 1], "tab:blue", lw=1.5)
        ax.plot(o["T"][:, 1], o["T"][:, 2], color="tab:red", lw=1.2)
        ax.plot(sl["T"][:, 1], sl["T"][:, 2], color="tab:green", lw=1.2)
        hull(ax, *o["T"][i, 1:4], "tab:red"); hull(ax, *sl["T"][j, 1:4], "tab:green")
        ax.set_xlim(cx - 2.5, cx + 2.5); ax.set_ylim(cy - 2.5, cy + 2.5); ax.set_aspect("equal"); ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
        tag = "касание устранено" if not c["slow_contact"] else "касание осталось"
        io = "—" if c["io"] is None else f"{c['io']:+.2f}"
        is_ = "—" if c["is_"] is None else f"{c['is_']:+.2f}"
        ax.set_title(f"{c['plan'][-5:]}: {tag}\nсмещение внутрь поворота {io} → {is_} м", fontsize=9)
    fig.legend(handles=[Line2D([], [], color="tab:blue", lw=2, label="план (PMP)"),
                        Line2D([], [], color="tab:red", lw=2, label=f"{name}, исходный план"),
                        Line2D([], [], color="tab:green", lw=2, label=f"{name}, план замедлен у стен"),
                        Line2D([], [], color="k", marker="s", mfc="none", ls="", ms=10,
                               label="корпус в точке первого касания (той же дуги плана)")],
               loc="upper center", ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"{name} (задание 237): места касаний на исходном плане и тот же участок на замедленном; окно 5x5 м",
                 fontsize=13, y=0.955)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(OUT, fname), dpi=70)


contact_sites("KTN", "GMPC", "gmpc_contact_sites.png")
contact_sites("LTN", "ours-lib", "ourslib_contact_sites.png")

# --- figure 2: inward offset orig vs slow, per arm -----------------------------
fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
for ax, cfg in zip(axs, CFG):
    T_ = stats[cfg]["tight"]; C_ = [c for c in stats[cfg]["contact"] if c["io"] is not None and c["is_"] is not None]
    ax.scatter([t["io"] for t in T_], [t["is_"] for t in T_], s=22, c="tab:gray", label="все тесные повороты (среднее по плану)")
    ax.scatter([c["io"] for c in C_], [c["is_"] for c in C_], s=40, c="tab:red", marker="x", label="место касания на исходном плане")
    lim = [min(-0.05, *(t["io"] for t in T_), *(t["is_"] for t in T_)), 0.5]
    ax.plot(lim, lim, "k--", lw=1); ax.axhline(0, c="k", lw=0.5); ax.axvline(0, c="k", lw=0.5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("макс. смещение внутрь поворота, исходный план [м]")
    ax.set_ylabel("то же, план замедлен у стен [м]")
    ax.set_title(CFG[cfg]); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Срезание углов: ниже диагонали = замедление уменьшило срез (задание 237, truth)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "inward_offset_orig_vs_slow.png"), dpi=80)
print(OUT)
