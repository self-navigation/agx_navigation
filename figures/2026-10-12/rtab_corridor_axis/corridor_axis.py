#!/usr/bin/env python3
"""Is the localizer error before an rtab door-turn wall hit ALONG the pre-turn corridor axis? (exploratory, 2026-10-12)

Hypothesis: rtab's lidar ICP is degenerate along corridors, so before a door turn its error lies along the corridor
(harmless there); after a ~90 deg turn that error is LATERAL at the jamb. Prediction: in the WIN s before an rtab wall
hit, the estimate-minus-truth error, decomposed in the frame of the PRE-TURN plan heading (plan heading 1.5 m before
the door), has a larger along-axis component than clean passages through the same doors.

Method
- error e = (map->odom o odom->base_link) - truth at each odom_filtered stamp: err_vec() from loc_bias_map/bias_map.py.
- doors, door frames and the contact->door assignment: copied from door_overlay/door_overlay.py; doors() here also
  returns the plan heading at s_door - 1.5 m (pre-turn heading) and the turn angle (-1.5 m -> +1.5 m).
- passage = (run, door) whose truth track enters the door window (|x| < 2.5, |y| < 1.5 in the door frame).
  HIT passage: the run's first contact is assigned to this door; anchor t_a = wall_first_t.
  CLEAN passage: no contact at this door in this run; anchor t_a = the moment of minimum truth wall clearance with
  |x| < 1 m in the door frame (the clean analogue of "closest to the wall").
- per passage, over t in [t_a - WIN, t_a]: mean |e . u_pre| (ALONG) and mean |e . n_pre| (ACROSS), u_pre = pre-turn
  heading. Also the signed means (across mirrored so the turn is to +n).
- primary set: TURN doors (|turn| >= TURN_MIN). Tests: Mann-Whitney hit vs clean (passages, not independent: 3 seeds
  x same plan), and a SAME-DOOR stratified permutation test: hit/clean labels shuffled within each plan-door that has
  both, statistic = mean over hits of (along_hit - mean along_clean of its door).
usage: corridor_axis.py
"""
import glob, json, os
import numpy as np, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "../../.."))
RD = os.path.join(R, "run_data")
DOOR_CLEAR, W = 0.60, 2.5
WIN, TURN_MIN, NPERM = 2.0, np.radians(45), 20000
ARMS = {"rtab": [("2026-10-12_job256_steerable-rtab", "RQ"), ("2026-10-12_job257_kprior", "PR")],
        "amcl": [("2026-10-12_job255_steerable", "Q0"), ("2026-10-12_job257_kprior", "PQ")]}
rng = np.random.default_rng(0)

meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))
if img.ndim == 3: img = img[..., 0]
res, (ox, oy, _) = meta["resolution"], meta["origin"]
H = img.shape[0]
clear = distance_transform_edt(img > 250) * res
wr, wc = np.nonzero(img < 50)
WALL = np.c_[ox + (wc + .5) * res, oy + (H - 1 - wr + .5) * res]


def clearance(xy):
    c = np.clip(((xy[:, 0] - ox) / res).astype(int), 0, img.shape[1] - 1)
    r = np.clip(H - 1 - ((xy[:, 1] - oy) / res).astype(int), 0, H - 1)
    return clear[r, c]


_doors = {}
def doors(plan_path):
    """door_overlay.doors() + the pre-turn heading and the signed turn: [(centre, h, mirror, s, width, th_pre, turn)]."""
    if plan_path in _doors: return _doors[plan_path]
    P = np.load(os.path.join(R, plan_path.replace("/home/programmer/", "")))["poses"]
    xy, th = P[:, :2], np.unwrap(P[:, 2])
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    cl = clearance(xy)
    out = []
    for i in range(1, len(cl) - 1):
        if cl[i] < DOOR_CLEAR and cl[i] <= cl[i - 1] and cl[i] <= cl[i + 1]:
            if s[i] < 0.5 or s[-1] - s[i] < 0.5: continue
            if out and s[i] - out[-1][3] < 1.0: continue
            p = xy[i]; near = WALL[np.hypot(*(WALL - p).T) < 1.0]
            if len(near) < 2: continue
            d = near - p; k1 = np.argmin(np.hypot(*d.T)); j1 = near[k1]
            opp = d @ (j1 - p) < 0
            if not opp.any(): continue
            j2 = near[opp][np.argmin(np.hypot(*d[opp].T))]
            centre, line = (j1 + j2) / 2, j2 - j1
            if np.linalg.norm(line) > 1.3: continue
            nrm = np.array([line[1], -line[0]]) / np.linalg.norm(line)
            if nrm @ np.array([np.cos(th[i]), np.sin(th[i])]) < 0: nrm = -nrm
            h = np.arctan2(nrm[1], nrm[0])
            a, b = np.searchsorted(s, s[i] - 1.5), min(np.searchsorted(s, s[i] + 1.5), len(s) - 1)
            turn = (th[b] - th[a] + np.pi) % (2 * np.pi) - np.pi
            out.append((centre, h, -1.0 if turn < 0 else 1.0, s[i], np.linalg.norm(line), th[a], turn))
    _doors[plan_path] = out
    return out


def to_frame(p, o, h, m):
    d = p - o; c, s_ = np.cos(h), np.sin(h)
    return np.c_[c * d[:, 0] + s_ * d[:, 1], m * (-s_ * d[:, 0] + c * d[:, 1])]


def err_vec(z):
    """bias_map.err_vec, also returning the stamps."""
    T, MO, O = z["track"], z["map_odom"], z["odom_filtered"]
    if len(T) < 3 or len(MO) < 3 or len(O) < 3: return None
    st = O[:, 6]
    m = (st >= max(T[0, 0], MO[0, 4])) & (st <= min(T[-1, 0], MO[-1, 4]))
    if m.sum() < 3: return None
    st, ox_, oy_ = st[m], O[m, 1], O[m, 2]
    mx, my = np.interp(st, MO[:, 4], MO[:, 1]), np.interp(st, MO[:, 4], MO[:, 2])
    myaw = np.interp(st, MO[:, 4], np.unwrap(MO[:, 3]))
    x = mx + np.cos(myaw) * ox_ - np.sin(myaw) * oy_
    y = my + np.sin(myaw) * ox_ + np.cos(myaw) * oy_
    tx, ty = np.interp(st, T[:, 0], T[:, 1]), np.interp(st, T[:, 0], T[:, 2])
    return st, tx, ty, x - tx, y - ty


P = {k: [] for k in ARMS}   # loc -> list of passage dicts
for loc, cells in ARMS.items():
    for job, cell in cells:
        for f in sorted(glob.glob(f"{RD}/{job}/{cell}/rows.s*.jsonl")):
            for l in open(f):
                if not l.strip(): continue
                r = json.loads(l)
                tp = f"{RD}/{job}/{cell}/s{r['seed']}/{os.path.basename(r.get('track_path') or 'x')}"
                if not os.path.isfile(tp): continue
                z = np.load(tp); T = z["track"]
                if len(T) < 5: continue
                ev = err_vec(z)
                if ev is None: continue
                st, etx, ety, ex, ey = ev
                hit = (r.get("wall_overlap_s") or 0) > 0 and r.get("wall_first_t") is not None
                ic = min(np.searchsorted(T[:, 0], r["wall_first_t"]), len(T) - 1) if hit else None
                D_ = doors(r["plan_path"])
                own = int(np.argmin([np.hypot(*to_frame(T[ic:ic + 1, 1:3], o, h, m)[0]) for o, h, m, *_ in D_])) if hit and D_ else None
                for di, (o, h, m, s_d, width, th_pre, turn) in enumerate(D_):
                    q = to_frame(T[:, 1:3], o, h, m)
                    win = (np.abs(q[:, 0]) < W) & (np.abs(q[:, 1]) < 1.5)
                    if win.sum() < 3: continue
                    is_hit = hit and di == own and win[ic]
                    if hit and di == own and not is_hit: continue      # contact outside the window: not this door's
                    if hit and r["wall_first_t"] <= T[win, 0].min(): continue   # hit elsewhere before reaching this door
                    if is_hit:
                        ta = r["wall_first_t"]
                    else:
                        cand = np.nonzero(win & (np.abs(q[:, 0]) < 1.0))[0]
                        if not len(cand): continue
                        ta = T[cand[np.argmin(clearance(T[cand, 1:3]))], 0]
                    w = (st >= ta - WIN) & (st <= ta)
                    if w.sum() < 5: continue
                    u = np.array([np.cos(th_pre), np.sin(th_pre)]); n = m * np.array([-u[1], u[0]])
                    al, ac = ex[w] * u[0] + ey[w] * u[1], ex[w] * n[0] + ey[w] * n[1]
                    ia = np.argmin(np.abs(st - ta))
                    tq = to_frame(np.array([[np.interp(ta, T[:, 0], T[:, 1]), np.interp(ta, T[:, 0], T[:, 2])]]), o, h, m)[0]
                    ef = to_frame(np.array([[ex[w].mean(), ey[w].mean()]]) + o, o, h, m)[0]   # mean error, rotated into door frame
                    P[loc].append(dict(cell=cell, plan=r["plan"], door=(r["plan"], round(o[0], 2), round(o[1], 2)),
                                       hit=bool(is_hit), turn=abs(turn), along=np.abs(al).mean(), across=np.abs(ac).mean(),
                                       al_s=al.mean(), ac_s=ac.mean(), al_end=al[-1], ac_end=ac[-1], width=width,
                                       tq=tq, ef=ef, th_pre_f=m * (th_pre - h)))
                del z

lines = []
def out(s=""):
    print(s); lines.append(s)


def strat_perm(ps, key):
    """same-door permutation: statistic = mean over hits of (x_hit - mean x_clean of its door), labels shuffled within door."""
    by = {}
    for p in ps: by.setdefault(p["door"], []).append(p)
    groups = [(np.array([p[key] for p in g]), np.array([p["hit"] for p in g])) for g in by.values()
              if any(p["hit"] for p in g) and not all(p["hit"] for p in g)]
    if not groups: return np.nan, np.nan, 0, 0
    def stat(gs):
        d = [x[h].mean() - x[~h].mean() for x, h in gs]; wts = [h.sum() for x, h in gs]
        return np.average(d, weights=wts)
    obs = stat(groups)
    null = np.array([stat([(x, rng.permutation(h)) for x, h in groups]) for _ in range(NPERM)])
    return obs, (np.sum(null >= obs) + 1) / (NPERM + 1), sum(h.sum() for _, h in groups), len(groups)


out(f"window = {WIN} s before anchor (hit: wall_first_t; clean: min truth clearance within 1 m of the door plane)")
out(f"errors in the PRE-TURN frame (plan heading 1.5 m before the door): ALONG = mean |e.u_pre|, ACROSS = mean |e.n_pre| (m)")
out("passages are not independent (3 seeds x plan); the same-door permutation test is the honest one\n")
summ = {}
for loc in ARMS:
    for setname, sel in (("TURN doors (|turn| >= 45 deg)", lambda p: p["turn"] >= TURN_MIN), ("ALL doors", lambda p: True)):
        ps = [p for p in P[loc] if sel(p)]
        hs, cs = [p for p in ps if p["hit"]], [p for p in ps if not p["hit"]]
        out(f"== {loc} ({'+'.join(c for _, c in ARMS[loc])}), {setname}: {len(hs)} hit passages, {len(cs)} clean passages")
        if len(hs) < 2:
            out("   too few hits\n"); continue
        ps_al = strat_perm(ps, "along")
        for key, nm in (("along", "ALONG |e.u_pre|"), ("across", "ACROSS |e.n_pre|"), ("al_end", "signed along at anchor"), ("ac_end", "signed across at anchor (+ = turn side)")):
            a, b = np.array([p[key] for p in hs]), np.array([p[key] for p in cs])
            obs, pp, nh, nd = strat_perm(ps, key)
            out(f"   {nm:40s} median hit {np.median(a):+.3f}  clean {np.median(b):+.3f}  | MWU p {mannwhitneyu(a, b).pvalue:.2g}"
                f" | same-door: hit - door clean mean {obs:+.3f} m, perm p(one-sided, >) {pp:.2g} ({nh} hits in {nd} doors)")
        fa, fb = [p["along"] / (p["along"] + p["across"]) for p in hs], [p["along"] / (p["along"] + p["across"]) for p in cs]
        out(f"   along share along/(along+across)          median hit {np.median(fa):.2f}  clean {np.median(fb):.2f}  | MWU p {mannwhitneyu(fa, fb).pvalue:.2g}")
        if setname.startswith("TURN"): summ[loc] = (hs, cs, ps_al)
        out()
open(os.path.join(HERE, "corridor_axis.txt"), "w").write("\n".join(lines) + "\n")

# figure: in the door frame, truth position at the anchor and the WIN-mean error vector (x5), hits vs clean
plt.rcParams.update({"font.size": 18})
fig, axs = plt.subplots(2, 1, figsize=(10.8, 15))
K = 5.0
for ax, loc in zip(axs, ARMS):
    hs, cs, (dob, dp, *_) = summ[loc]
    jw = np.median([p["width"] for p in hs + cs]) / 2 if hs + cs else 0.4
    for sg in (1, -1): ax.plot([0, 0], [sg * jw, sg * 1.6], color="k", lw=6, solid_capstyle="butt", zorder=5)
    for ps, col, lw, al, z in ((cs, "tab:blue", 1.0, 0.25, 2), (hs, "tab:red", 2.4, 0.9, 4)):
        for p in ps:
            ax.annotate("", xy=p["tq"] + K * p["ef"], xytext=p["tq"],
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, alpha=al, mutation_scale=14), zorder=z)
            ax.plot(*p["tq"], "o", color=col, ms=4 if col == "tab:blue" else 7, alpha=al, zorder=z)
    ah = np.median([p["along"] for p in hs]) if hs else np.nan; ac = np.median([p["along"] for p in cs]) if cs else np.nan
    ax.set_title(f"{loc} (TVLQG): {len(hs)} hits (red) vs {len(cs)} clean (blue)\n"
                 f"median |along pre-turn axis| {ah * 100:.0f} vs {ac * 100:.0f} cm\n"
                 f"same door: hits {dob * 100:+.1f} cm vs that door's clean, p {dp:.2g}", fontsize=19, fontweight="bold")
    ax.annotate("pre-turn\ncorridor axis", xy=(-1.9, -1.5), xytext=(-1.9, -0.6), fontsize=15, ha="center",
                arrowprops=dict(arrowstyle="<->", lw=2, color="0.3"))
    ax.set_xlim(-2.5, 2.0); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_ylabel("lateral in door (m)")
axs[-1].set_xlabel("through the door (m); robot enters from the left,\nturns mirrored to the left")
fig.suptitle(f"Localization error, mean over the {WIN:.0f} s before a wall hit / the\nclosest approach (arrows x{K:.0f}), "
             "turn doors, mu 0.5. Exploratory", fontsize=19, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "corridor_axis.png"), dpi=100)
