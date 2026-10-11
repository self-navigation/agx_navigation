#!/usr/bin/env python3
"""Where does the pose the controller uses go wrong under amcl, and why?

Over every run of job 248 (all amcl, Aogs), at 10 Hz along each drive:
  believed(t) = map->odom(t) o odom/filtered(t)   (the pose the corrector acts on)
  truth(t)    = Gazebo track
error in the TRUTH body frame: along, lateral, yaw. Split into
  amcl  = /amcl_pose vs truth at its stamp (the filter's own estimate),
  drift = believed - (amcl error of the latest update): odometry/EKF growth between updates.
Binned by turn rate, speed and map cell; amcl's reported sigma vs its actual error.
Static floor-6 scans fit the map to 0-3 cm at truth (run_data/2026-10-10_nojob_scan-map-offset/floor6),
so map geometry is not the suspect.

    python3 render.py [job_dir]  -> loc_error.png, table.txt
"""
import glob, os, sys
import numpy as np, yaml
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "../../.."))
JOB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(R, "run_data/2026-10-11_job248_sweep-dev")
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi


def interp_pose(tq, t, x, y, yaw):
    return np.interp(tq, t, x), np.interp(tq, t, y), wrap(np.interp(tq, t, np.unwrap(yaw)))


def hold(tq, t, cols):  # latest sample at or before tq (a TF is applied as last received)
    i = np.clip(np.searchsorted(t, tq, side="right") - 1, 0, len(t) - 1)
    return [c[i] for c in cols]


def body(dx, dy, dyaw, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy, wrap(dyaw)


S = {k: [] for k in ["al", "la", "ya", "a_al", "a_la", "a_ya", "d_al", "d_la", "d_ya", "w", "v", "x", "y", "since"]}
A = {k: [] for k in ["al", "la", "ya", "sx", "sy", "syaw", "w", "jump_xy", "jump_yaw"]}
nrun = 0
for f in sorted(glob.glob(f"{JOB}/*/s*/track_*.npz")):
    z = np.load(f)
    T, MO, OF, AM = z["track"], z["map_odom"], z["odom_filtered"], z["amcl"]
    if len(T) < 20 or len(MO) < 5 or len(OF) < 5 or len(AM) < 3 or len(z["diag_t"]) < 5:
        continue
    nrun += 1
    t0, t1 = z["diag_t"][0], min(z["diag_t"][-1], T[-1, 0], OF[-1, 0], MO[-1, 0])
    tq = np.arange(t0, t1, 0.1)
    if len(tq) < 10:
        continue
    tx, ty, tyaw = interp_pose(tq, T[:, 0], T[:, 1], T[:, 2], T[:, 3])
    w = np.gradient(np.unwrap(tyaw), tq); v = np.hypot(np.gradient(tx, tq), np.gradient(ty, tq))
    mx, my, myaw = hold(tq, MO[:, 0], [MO[:, 1], MO[:, 2], MO[:, 3]])
    ox_, oy_, oyaw = interp_pose(tq, OF[:, 0], OF[:, 1], OF[:, 2], OF[:, 3])
    bx = mx + np.cos(myaw) * ox_ - np.sin(myaw) * oy_
    by = my + np.sin(myaw) * ox_ + np.cos(myaw) * oy_
    byaw = wrap(myaw + oyaw)
    al, la, ya = body(bx - tx, by - ty, byaw - tyaw, tyaw)
    # amcl's own estimate at its stamp
    ast = AM[:, 7]
    ax_, ay_, ayaw = interp_pose(ast, T[:, 0], T[:, 1], T[:, 2], T[:, 3])
    aal, ala, aya = body(AM[:, 1] - ax_, AM[:, 2] - ay_, AM[:, 3] - ayaw, ayaw)
    keep = (ast >= t0) & (ast <= t1)
    aw = np.interp(ast, tq, w)
    A["al"] += list(aal[keep]); A["la"] += list(ala[keep]); A["ya"] += list(aya[keep]); A["w"] += list(aw[keep])
    A["sx"] += list(np.sqrt(AM[keep, 4])); A["sy"] += list(np.sqrt(AM[keep, 5])); A["syaw"] += list(np.sqrt(AM[keep, 6]))
    # size of the map->odom corrections (what the corrector feels as a pose jump)
    dmo = np.hypot(np.diff(MO[:, 1]), np.diff(MO[:, 2])); dmy = np.abs(wrap(np.diff(MO[:, 3])))
    mk = (MO[1:, 0] >= t0) & (MO[1:, 0] <= t1)
    A["jump_xy"] += list(dmo[mk][dmo[mk] > 0]); A["jump_yaw"] += list(dmy[mk][dmy[mk] > 0])
    # latest amcl update (by reception time) before each tq -> drift since it
    j = np.searchsorted(AM[:, 0], tq, side="right") - 1
    ok = j >= 0; jj = np.clip(j, 0, len(AM) - 1)
    for k, val in [("al", al), ("la", la), ("ya", ya), ("w", w), ("v", v), ("x", tx), ("y", ty)]:
        S[k] += list(val[ok])
    S["a_al"] += list(aal[jj][ok]); S["a_la"] += list(ala[jj][ok]); S["a_ya"] += list(aya[jj][ok])
    S["d_al"] += list((al - aal[jj])[ok]); S["d_la"] += list((la - ala[jj])[ok]); S["d_ya"] += list(wrap(ya - aya[jj])[ok])
    S["since"] += list((tq - AM[jj, 0])[ok])
S = {k: np.array(v) for k, v in S.items()}; A = {k: np.array(v) for k, v in A.items()}

L = []
def out(s=""):
    print(s); L.append(s)
q = lambda a, p: np.percentile(np.abs(a), p)
out(f"runs {nrun}; believed-pose samples {len(S['al'])} (10 Hz); amcl updates {len(A['al'])}")
out("\n|error| percentiles      median    p90    p99   | signed mean")
for name, k in [("believed along", "al"), ("believed lateral", "la"), ("believed yaw [deg]", "ya"),
                ("amcl along (at update)", "a_al"), ("amcl lateral", "a_la"), ("amcl yaw [deg]", "a_ya"),
                ("drift along (since upd)", "d_al"), ("drift lateral", "d_la"), ("drift yaw [deg]", "d_ya")]:
    a = S[k] * (180 / np.pi if "yaw" in name else 1)
    out(f"{name:26s} {q(a,50):7.3f} {q(a,90):6.3f} {q(a,99):6.3f}   | {a.mean():+.3f}")
out(f"\ntime since last amcl update: median {np.median(S['since']):.2f} s, p90 {np.percentile(S['since'],90):.2f} s")
out(f"map->odom corrections: n {len(A['jump_xy'])}, |dxy| median {np.median(A['jump_xy']):.3f} p90 {np.percentile(A['jump_xy'],90):.3f} m; "
    f"|dyaw| median {np.degrees(np.median(A['jump_yaw'])):.2f} p90 {np.degrees(np.percentile(A['jump_yaw'],90)):.2f} deg")
out("\nby |turn rate| (truth)      n     lat med  lat p90 | yaw med  yaw p90 [deg] | drift lat p90  drift yaw p90")
wb = [0, 0.1, 0.3, 0.6, 1.0, 9]
for a, b in zip(wb, wb[1:]):
    m = (np.abs(S["w"]) >= a) & (np.abs(S["w"]) < b)
    out(f"[{a:.1f},{b:.1f}) rad/s     {m.sum():7d}   {q(S['la'][m],50):.3f}   {q(S['la'][m],90):.3f}  | {np.degrees(q(S['ya'][m],50)):5.2f}  {np.degrees(q(S['ya'][m],90)):5.2f}"
        f"        | {q(S['d_la'][m],90):.3f}         {np.degrees(q(S['d_ya'][m],90)):5.2f}")
out("\namcl consistency: |error| / reported sigma (a calibrated filter: median ~0.67, p90 ~1.6)")
for name, e, s in [("x/along-ish", np.hypot(A['al'], A['la']) / np.sqrt(2), np.hypot(A['sx'], A['sy']) / np.sqrt(2)),
                   ("yaw", A['ya'], A['syaw'])]:
    r = np.abs(e) / np.maximum(s, 1e-6)
    out(f"  {name:12s} median {np.median(r):.2f}  p90 {np.percentile(r,90):.2f}  (reported sigma median {np.median(s):.3f})")
open(os.path.join(HERE, "table.txt"), "w").write("\n".join(L) + "\n")

# figure
meta = yaml.safe_load(open(os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml")))
img = np.asarray(Image.open(os.path.join(R, "src/rudn-ordjo-building/maps", meta["image"])))[::-1]
res = meta["resolution"]; ox, oy = meta["origin"][:2]
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(2, 3, 1)
for k, lab in [("la", "believed"), ("a_la", "amcl at last update"), ("d_la", "odom drift since update")]:
    ax.hist(np.clip(S[k], -0.4, 0.4), bins=80, histtype="step", label=lab, density=True)
ax.set_xlabel("lateral error [m] (+ = left of truth)"); ax.legend(fontsize=8); ax.set_title("lateral error, all samples")
ax = fig.add_subplot(2, 3, 2)
for k, lab in [("ya", "believed"), ("a_ya", "amcl at last update"), ("d_ya", "odom drift since update")]:
    ax.hist(np.clip(np.degrees(S[k]), -15, 15), bins=80, histtype="step", label=lab, density=True)
ax.set_xlabel("yaw error [deg]"); ax.legend(fontsize=8); ax.set_title("yaw error, all samples")
ax = fig.add_subplot(2, 3, 3)
wc = 0.5 * (np.array(wb[:-1]) + np.array(wb[1:])); wc[-1] = 1.3
for k, lab, c in [("la", "lateral believed", "C0"), ("d_la", "lateral drift", "C2")]:
    ax.plot(wc, [q(S[k][(np.abs(S['w']) >= a) & (np.abs(S['w']) < b)], 90) for a, b in zip(wb, wb[1:])], "o-", color=c, label=lab + " p90 [m]")
ax2 = ax.twinx()
for k, lab, c in [("ya", "yaw believed", "C1"), ("d_ya", "yaw drift", "C3")]:
    ax2.plot(wc, [np.degrees(q(S[k][(np.abs(S['w']) >= a) & (np.abs(S['w']) < b)], 90)) for a, b in zip(wb, wb[1:])], "s--", color=c, label=lab + " p90 [deg]")
ax.set_xlabel("|turn rate| [rad/s]"); ax.set_ylabel("m"); ax2.set_ylabel("deg")
ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8); ax.set_title("error p90 vs turn rate")
ax = fig.add_subplot(2, 1, 2)
ax.imshow(np.where(img == 0, 0.2, np.where(img == 205, 0.85, 1.0)), cmap="gray", vmin=0, vmax=1, origin="lower",
          extent=[ox, ox + img.shape[1] * res, oy, oy + img.shape[0] * res], interpolation="nearest")
# mean WORLD-frame position error per 1 m cell
ex = S["al"] * np.cos(0) * 0  # placeholder to keep shapes; world error computed below
gx = np.floor(S["x"]).astype(int); gy = np.floor(S["y"]).astype(int)
# world-frame error = rotate body error back by truth yaw is not stored; recompute from al/la is lossy, so use |lateral| colour
cells = {}
for i in range(len(gx)):
    cells.setdefault((gx[i], gy[i]), []).append(i)
X, Y, C, N = [], [], [], []
for (cx, cy), idx in cells.items():
    if len(idx) < 30: continue
    X.append(cx + 0.5); Y.append(cy + 0.5); C.append(np.percentile(np.abs(S["la"][idx]), 90)); N.append(len(idx))
sc = ax.scatter(X, Y, c=C, s=18, marker="s", cmap="viridis_r", vmin=0, vmax=0.25)
plt.colorbar(sc, ax=ax, label="lateral |error| p90 [m] per 1 m cell (believed pose)")
ax.set_xlim(-30, 30); ax.set_ylim(-12, 11); ax.set_aspect("equal"); ax.set_title("where the believed pose is laterally wrong")
fig.suptitle(f"Job 248 (amcl Aogs), {nrun} runs: believed pose = map->odom o EKF vs Gazebo truth", fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "loc_error.png"), dpi=100)
print("wrote", os.path.join(HERE, "loc_error.png"))
