#!/usr/bin/env python3
"""Mechanism checks for the amcl vs truth comparison (run after render.py).
    .venv/bin/python figures/2026-10-10/amcl_vs_truth/extra.py
(i) body-frame lateral velocity of the believed pose vs truth (does amcl follow
    odometry, which has no lateral motion?); (ii) error growth in turns vs
    decay on straights; (iii) error vs time; (iv) per-run error vs total slide;
(v) LAN vs LANq split by whether the surface switch fired.
"""
import json, os, sys
import numpy as np
OUT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, OUT)
from render import rows, analyse, CELLS, sign_p
R_ = rows(); S = json.load(open(os.path.join(OUT, "stats.json")))

def smooth(x, n=5):
    return np.convolve(x, np.ones(n) / n, mode="same")

for cell in ("LTNq", "LANq"):
    slope_lat, slope_fwd, grow_turn, grow_str, tbin, slide_vs_err, tau_dec = [], [], [], [], {}, [], []
    for (plan, c, seed), r in sorted(R_.items()):
        if c != cell: continue
        st, s = analyse(r)
        if s is None: continue
        m = s["drive"]; t = s["t"]
        if m.sum() < 50: continue
        # body-frame velocities, truth heading as the body frame for both
        th = s["tru"][:, 2]; cth, sth = np.cos(th), np.sin(th)
        def bodyv(P):
            vx = np.gradient(smooth(P[:, 0]), t); vy = np.gradient(smooth(P[:, 1]), t)
            return cth * vx + sth * vy, -sth * vx + cth * vy
        tf, tl = bodyv(s["tru"]); bf, bl = bodyv(s["bel"])
        mv = m & (np.hypot(tf, tl) > 0.1)
        # differential: lateral rate of (believed - truth) vs truth lateral speed.
        # -1 = the believed pose does not move sideways when the body does (it
        # follows odometry, which has no lateral channel); 0 = it follows truth.
        if mv.sum() > 30 and np.ptp(tl[mv]) > 0.05:
            dl = bl - tl; df = bf - tf
            slope_lat.append(float((tl[mv] @ dl[mv]) / (tl[mv] @ tl[mv])))
            slope_fwd.append(float((tf[mv] @ df[mv]) / (tf[mv] @ tf[mv])))
        # (ii) growth of |err| per second in turns vs straights
        le = smooth(s["le"]); dle = np.gradient(le, t)
        w = np.abs(np.gradient(np.unwrap(th), t))
        turn = mv & (w > 0.3); straight = mv & (w < 0.1)
        if turn.sum() > 5: grow_turn.append(float(np.mean(dle[turn])))
        if straight.sum() > 5: grow_str.append(float(np.mean(dle[straight])))
        # (iii) error by time bin
        for lo in range(0, 60, 5):
            sel = m & (t >= lo) & (t < lo + 5)
            if sel.any(): tbin.setdefault(lo, []).append(float(np.median(s["le"][sel])))
        # (iv) total slide = integral |v_lat| of truth
        dt = np.gradient(t)
        slide_vs_err.append((float(np.sum(np.abs(tl[mv]) * dt[mv])), float(s["le"][m].max()), float(np.sum(np.abs(tl[mv]) * dt[mv]) / max(np.sum(np.hypot(tf, tl)[mv] * dt[mv]), 1e-6))))
        # decay on straights: segments of >=3 s straight starting with err>0.25
        i = 0; idx = np.flatnonzero(straight)
        if len(idx):
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for seg in runs:
                if len(seg) >= 30 and le[seg[0]] > 0.25:
                    e0, e1 = le[seg[0]], le[seg[-1]]; T = t[seg[-1]] - t[seg[0]]
                    if e1 > 0.02 and e1 < e0: tau_dec.append(T / np.log(e0 / e1))
                    elif e1 >= e0: tau_dec.append(np.inf)
    print(f"{cell}: d(believed-truth)/dt LATERAL vs truth lateral speed: slope med {np.median(slope_lat):.2f} (IQR {np.percentile(slope_lat,25):.2f}..{np.percentile(slope_lat,75):.2f}, n={len(slope_lat)}); "
          f"FORWARD slope med {np.median(slope_fwd):.2f}")
    print(f"   d|err|/dt in turns med {np.median(grow_turn)*1000:+.0f} mm/s, on straights {np.median(grow_str)*1000:+.0f} mm/s "
          f"(runs with growth in turns {sum(g>0 for g in grow_turn)}/{len(grow_turn)}, decay on straights {sum(g<0 for g in grow_str)}/{len(grow_str)})")
    print("   |err| median by 5 s bin: " + " ".join(f"{lo}s:{np.median(v):.2f}(n{len(v)})" for lo, v in sorted(tbin.items())))
    sv = np.array(slide_vs_err)
    print(f"   per-run max err vs total truth slide: rho {np.corrcoef(sv[:,0], sv[:,1])[0,1]:.2f}; slide med {np.median(sv[:,0]):.2f} m per run ({100*np.median(sv[:,2]):.0f}% of path)")
    fin = [x for x in tau_dec if np.isfinite(x)]
    print(f"   straight segments starting >0.25 m err: {len(tau_dec)}; decaying {len(fin)} with time constant med {np.median(fin) if fin else float('nan'):.1f} s; not decaying {len(tau_dec)-len(fin)}")

print("\n(v) LAN vs LANq split by the surface switch")
kk = sorted(set(S["LAN"]) & set(S["LANq"]))
for name, sel in (("switched", [k for k in kk if S["LANq"][k]["k_switch_t"] is not None]),
                  ("not switched (k_bar=0 throughout: same control law as LAN)", [k for k in kk if S["LANq"][k]["k_switch_t"] is None])):
    w = sum(S["LANq"][k]["final_err"] < S["LAN"][k]["final_err"] for k in sel); l = len(sel) - w
    d = [S["LANq"][k]["final_err"] - S["LAN"][k]["final_err"] for k in sel]
    big = sum(abs(x) > 0.3 for x in d)
    print(f"  {name}: n={len(sel)} LANq better {w}/{len(sel)} (p {sign_p(w,l):.2f}); |delta final_err| med {np.median(np.abs(d)):.3f}, >0.3 m on {big}; "
          f"contacts LAN {sum(S['LAN'][k]['contact'] for k in sel)} LANq {sum(S['LANq'][k]['contact'] for k in sel)}; "
          f"arrived LAN {sum(S['LAN'][k]['outcome']=='arrived' for k in sel)} LANq {sum(S['LANq'][k]['outcome']=='arrived' for k in sel)}")
sw = [k for k in kk if S["LANq"][k]["k_switch_t"] is not None]
print("  switched runs: k_bar_end:", sorted(round(S["LANq"][k]["k_bar_end"], 2) for k in sw))
# truth-cell run-to-run noise for reference: LTNq job244 vs job243 LTNq? not local; use LTDq vs LTDf rows (near-identical cells)
rows243 = {}
p243 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(OUT))), "run_data/2026-10-09_job243_tvlqg-fullstack/all_rows.jsonl")
if os.path.exists(p243):
    for line in open(p243):
        r = json.loads(line)
        if r["tvlqr_params"] == "sideslip_v_gate=0.3,tvlqg.enabled=true" and r.get("final_err") is not None:
            rows243[(r["plan"], r["seed"])] = r
    pairs = [(S["LTNq"][k]["final_err"], rows243[(k.split("/")[0], int(k.split("/")[1][1]))]["final_err"]) for k in S["LTNq"] if (k.split("/")[0], int(k.split("/")[1][1])) in rows243]
    d = [a - b for a, b in pairs]
    print(f"  reference: SAME config (LTNq) job244 vs job243, n={len(pairs)}: job244 better {sum(x<0 for x in d)}/{len(d)}, |delta| med {np.median(np.abs(d)):.3f}, >0.3 m on {sum(abs(x)>0.3 for x in d)}")

print("\n(vi) per TURN EPISODE: change of the localization error along the body's lateral axis vs the truth lateral slide")
print("     slope -1 = the believed pose misses the slide entirely (follows odometry); 0 = it tracks the slide")
for cell in ("LTNq", "LANq", "LAN"):
    X, Y, Xs, Ys = [], [], [], []
    for (plan, c, seed), r in sorted(R_.items()):
        if c != cell: continue
        st, s = analyse(r)
        if s is None: continue
        T = s["track"]; tt = T[:, 0] - float(np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(OUT))), *[]), allow_pickle=True)) if False else None
        # unsmoothed truth velocities on the raw 16 Hz track, body-lateral component
        vx = np.gradient(T[:, 1], T[:, 0]); vy = np.gradient(T[:, 2], T[:, 0])
        th = np.unwrap(T[:, 3]); vlat = -np.sin(th) * vx + np.cos(th) * vy
        w = np.gradient(th, T[:, 0])
        t0 = T[0, 0] + (s["t"][0] - 0)  # not used
        # map diag times onto track times: s["t"] is relative to goal_t
        goal_t = T[0, 0] + 0  # placeholder
        tr_rel = T[:, 0] - (T[:, 0][0])  # placeholder
        # episodes of |w| > 0.3 lasting >= 1 s, on the track's own clock
        turning = (np.abs(w) > 0.3) & (np.hypot(vx, vy) > 0.1)
        idx = np.flatnonzero(turning)
        if not len(idx): continue
        d = np.load(os.path.join(*[os.path.dirname(os.path.dirname(os.path.dirname(OUT))), "run_data/2026-10-09_job244_tvlqg-deviation"] + r["track_path"].split("/")[-3:]), allow_pickle=True)
        gt = float(d["goal_t"])
        for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            ta, tb = T[seg[0], 0] - gt, T[seg[-1], 0] - gt
            if tb - ta < 1.0 or ta < 0: continue
            slide = float(np.trapezoid(vlat[seg], T[seg, 0]))
            # loc error (believed - truth) at the diag ticks nearest ta, tb, in the body lateral axis at each time
            ia, ib = np.searchsorted(s["t"], ta), min(np.searchsorted(s["t"], tb), len(s["t"]) - 1)
            if ib <= ia: continue
            e = s["bel"][:, :2] - s["tru"][:, :2]; hth = s["tru"][:, 2]
            lat = -np.sin(hth) * e[:, 0] + np.cos(hth) * e[:, 1]
            X.append(slide); Y.append(float(lat[ib] - lat[ia]))
            # the straight segment that follows: does the error decay?
    X, Y = np.array(X), np.array(Y)
    k = (X @ Y) / (X @ X)
    print(f"  {cell}: {len(X)} turn episodes, slide per episode med {np.median(np.abs(X)):.3f} m (p90 {np.percentile(np.abs(X),90):.2f}); "
          f"d(err_lat) vs slide slope {k:+.2f}, rho {np.corrcoef(X, Y)[0,1]:+.2f}; |d err_lat| med {np.median(np.abs(Y)):.3f}")
