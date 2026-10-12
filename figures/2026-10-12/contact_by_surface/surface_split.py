"""Contact runs split by the surface under the robot (jobs 253, 254; exploratory, post hoc).

Replays each run's terrain (tools/spawn_patches.py: along_path_terrain_sampler(poses)(default_rng(seed)),
checked against the patch centres the row recorded), then asks, per run:
  - which patches the robot's CENTRE crossed (box x = width, y = length, rotated by yaw);
  - for a contact run, the surface under the centre in the 3 s before first contact.
Steering knee = wheel mu2 0.45: a patch with min(mu, mu2) <= 0.45 cannot steer
(linoleum sits ON the knee; directional_* has 0.15 on one axis).
Usage: python surface_split.py JOB_DIR [JOB_DIR ...] -> stdout
"""
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path[:0] = [os.path.join(R, "src/agx_navigation/agx_planning"), os.path.join(R, "src/rudn-ordjo-building")]
from agx_planning.rl_corrector.terrain import along_path_terrain_sampler
from rudn_ordjo_building.surface_patches import PROFILES
KNEE = 0.45
PRE_S = 3.0


def local(p):
    p = p.replace("/home/programmer/", "")
    return p if os.path.exists(p) else os.path.join(R, p)


def inside(xy, pa):
    c, s = np.cos(pa["yaw"]), np.sin(pa["yaw"])
    d = xy - [pa["x"], pa["y"]]
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c
    return (abs(u) <= pa["width"] / 2) & (abs(v) <= pa["length"] / 2)


def sub_knee(prof):
    P = PROFILES[prof]
    return min(P.mu, P.mu2) <= KNEE + 1e-9


def surface_at(xy, patches):
    """Lowest-friction profile under each sample ('ground' if none)."""
    out = np.array(["ground"] * len(xy), dtype=object); best = np.full(len(xy), 9.0)
    for pa in patches:
        m = inside(xy, pa); mu = min(PROFILES[pa["profile"]].mu, PROFILES[pa["profile"]].mu2)
        upd = m & (mu < best); out[upd] = pa["profile"]; best[upd] = mu
    return out


for JOB in sys.argv[1:]:
    print(f"\n===== {os.path.basename(JOB.rstrip('/'))}")
    cells = sorted(d for d in os.listdir(JOB) if os.path.isdir(os.path.join(JOB, d)))
    tab = defaultdict(Counter); mism = 0; lab_of = {}; prof_runs = defaultdict(lambda: [0, 0])
    for c in cells:
        for f in glob.glob(f"{JOB}/{c}/rows.s*.jsonl"):
            for l in open(f):
                r = json.loads(l)
                if not r.get("track_path"):
                    continue
                poses = np.load(local(r["plan_path"]))["poses"]
                pats = along_path_terrain_sampler(poses[:, :3])(np.random.default_rng(r["seed"]))
                rec = [p for p in r.get("patches", []) if p.get("profile") != "door_leaf"]
                if len(rec) != len(pats) or any(a["profile"] != b["profile"] or abs(a["x"] - b["x"]) > 0.01
                                                for a, b in zip(rec, pats)):
                    mism += 1; continue
                T = np.load(os.path.join(JOB, c, f"s{r['seed']}", os.path.basename(r["track_path"])))["track"]
                surf = surface_at(T[:, 1:3], pats)
                crossed = {s for s in set(surf) if s != "ground"}
                knee_crossed = any(sub_knee(s) for s in crossed)
                hit = (r.get("wall_overlap_s") or 0) > 0
                key = "crosses sub-knee patch" if knee_crossed else "no sub-knee patch"
                tab[c][key + (" HIT" if hit else "")] += 1
                lab_of.setdefault((c, r["plan"], r["seed"]), "none")
                for s in crossed:
                    prof_runs[s][0] += 1; prof_runs[s][1] += hit
                if hit:
                    m = (T[:, 0] >= r["wall_first_t"] - PRE_S) & (T[:, 0] <= r["wall_first_t"])
                    pre = [s for s in surf[m] if s != "ground"]
                    lab = (min(pre, key=lambda s: min(PROFILES[s].mu, PROFILES[s].mu2)) if pre else "ground")
                    lab_of[(c, r["plan"], r["seed"])] = "ground" if lab == "ground" else "patch"
                    tab[c]["contact; pre-3s surface: " + ("SUB-KNEE " if lab != "ground" and sub_knee(lab) else "") + lab] += 1
    if mism:
        print(f"(terrain replay mismatched recorded patches on {mism} runs; skipped)")
    for c in cells:
        t = tab[c]
        a, ah = t["crosses sub-knee patch"] + t["crosses sub-knee patch HIT"], t["crosses sub-knee patch HIT"]
        b, bh = t["no sub-knee patch"] + t["no sub-knee patch HIT"], t["no sub-knee patch HIT"]
        pre = {k.split(": ", 1)[1]: v for k, v in t.items() if k.startswith("contact;")}
        print(f"{c:4s} crosses sub-knee: {ah:3d}/{a:3d} hit | none: {bh:3d}/{b:3d} hit | pre-contact surface: "
              + ", ".join(f"{k} {v}" for k, v in sorted(pre.items(), key=lambda kv: -kv[1])))
    print("per profile crossed (all cells): runs, contact runs, rate")
    for s, (n, h) in sorted(prof_runs.items(), key=lambda kv: min(PROFILES[kv[0]].mu, PROFILES[kv[0]].mu2)):
        print(f"  {s:14s} min mu {min(PROFILES[s].mu, PROFILES[s].mu2):.2f}  {n:4d}  {h:4d}  {h/n:.2f}")
    print("paired (A vs B over plan,seed): contact preceded by GROUND / by a PATCH; fixed/new = B hit & A not / A hit & B not; sign test")
    from scipy.stats import binomtest
    for a, b in [("Q0", "QR"), ("RQ", "RR"), ("T", "TR")]:
        if a not in cells or b not in cells:
            continue
        keys = [k[1:] for k in lab_of if k[0] == a and (b,) + k[1:] in lab_of]
        out = []
        for kind in ("ground", "patch"):
            fx = sum(lab_of[(b,) + k] == kind and lab_of[(a,) + k] != kind for k in keys)
            nw = sum(lab_of[(a,) + k] == kind and lab_of[(b,) + k] != kind for k in keys)
            na = sum(lab_of[(a,) + k] == kind for k in keys); nb = sum(lab_of[(b,) + k] == kind for k in keys)
            pv = binomtest(fx, fx + nw).pvalue if fx + nw else 1.0
            out.append(f"{kind}: {na} vs {nb}, {fx}/{nw} p {pv:.2g}")
        print(f"  {a} vs {b} n {len(keys)} | " + " | ".join(out))
