#!/usr/bin/env python3
"""Build an evaluation plan set by random sampling, solving, typing and selection.

Stages, each reading the previous one's files under --out:

  sample   random start/goal pairs per floor and length band, in the robot's
           connected component, with spin clearance and random headings
  solve    the offline PMP planner on the centred FM2 field, cropped to the
           pair (needs the workspace venv)
  select   type each solved plan, drop trivial ones and pick per band by
           greedy max-min distance over standardised descriptors
  gallery  contact sheets of the selected and the rejected plans

A plan's type comes from segmenting its (v, omega) series with Foote's
checkerboard novelty into SPIN/REV/STR/L/R segments. Trivial means no turning
segment, under 45 deg of end spin and under 90 deg of heading change while
moving.

    python3 tools/build_plan_set.py sample --out D --floors 6,2 --per-band 250
    python3 tools/build_plan_set.py solve  --out D --jobs 36
    python3 tools/build_plan_set.py select --out D --per-band 20
    python3 tools/build_plan_set.py gallery --out D
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys

import numpy as np

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [os.path.join(R, "tools"), os.path.join(R, "src/agx_navigation/agx_planning"),
                os.path.join(R, "src/rudn-ordjo-building")]

# Floor -> (map yaml, component seed). Floor 6 is the INSTALLED map, the one the
# campaign's Gazebo runs score against (frozen). Floor 2 is the full-extent bake
# (run_data/2026-10-10_nojob_full-floor-bake, not installed yet): its two halves
# are disconnected, so plans stay in the half with the robot's room (x ~47, y ~2).
MAPS = {
    6: (os.path.join(R, "src/rudn-ordjo-building/maps/floor_6.yaml"), (0.0, 0.0)),
    2: (next((p for p in (os.path.join(R, "run_data/2026-10-10_nojob_full-floor-bake/floor_2.yaml"),
                          os.path.expanduser("~/run_data/2026-10-10_nojob_full-floor-bake/floor_2.yaml"))
              if os.path.exists(p)), "floor_2.yaml: full bake not found"), (32.0, -1.0)),
}
BANDS = {"short": (2.0, 6.0), "medium": (6.0, 15.0), "long": (15.0, 50.0),  # start-goal distance [m]
         "corridor": (3.0, 40.0)}  # both endpoints in the corridor (doors are actuated in reality)
CORRIDOR_CLEAR = 0.6  # doors are ~1.0-1.1 m (centre clearance 0.50-0.55): >= 0.6 cuts the rooms off
END_CLEAR = 0.50   # endpoint centre clearance [m]: room to spin the 0.43 m-radius hull
FIELD = dict(speed_profile="exponential", inflation_radius=0.5, centring=True, centring_cap=0.8,
             speed_decay_rate=1.5, speed_v_min=0.1)
V_MAX, W_MAX = 0.448, 1.049
HL, HW = 0.315, 0.2925
END_SPIN_M = 0.5
CORNER_RAD = math.radians(45)  # a 'straight' segment turning more than this is a corner
LEN_BANDS = {"short": (0, 8.0), "medium": (8.0, 20.0), "long": (20.0, 1e9)}  # solved path length [m]


# ------------------------------------------------------------------ maps

_MAP = {}


def load_map(floor):
    """(occ grid int8 with 0 free / 100 wall / -1 unknown, row 0 = y_min; meta)."""
    if floor not in _MAP:
        # map_publisher.load_occupancy_grid's convention, inlined: that module imports rclpy.
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(MAPS[floor][0]))
        px = np.flipud(np.asarray(Image.open(os.path.join(os.path.dirname(MAPS[floor][0]), meta["image"]))
                                  .convert("L"))).astype(float)
        occ = (255.0 - px) / 255.0
        if meta.get("negate"):
            occ = 1.0 - occ
        g = np.full(px.shape, -1, dtype=np.int8)
        g[occ >= meta["occupied_thresh"]] = 100
        g[occ <= meta["free_thresh"]] = 0
        _MAP[floor] = (g, meta)
    return _MAP[floor]


def free_edt(floor):
    from scipy import ndimage
    arr, meta = load_map(floor)
    return ndimage.distance_transform_edt(arr == 0) * meta["resolution"]


def corridor_mask(floor):
    """The corridor: the seed's component at CORRIDOR_CLEAR, where doorways disconnect the rooms."""
    return component(floor, CORRIDOR_CLEAR)


def component(floor, clearance):
    """Cells with centre clearance >= clearance, in the seed's connected component."""
    from scipy import ndimage
    arr, meta = load_map(floor)
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    ok = free_edt(floor) >= clearance
    lab, _ = ndimage.label(ok)
    sx, sy = MAPS[floor][1]
    seed = lab[int((sy - oy) / r), int((sx - ox) / r)]
    if seed == 0:  # seed itself too tight: take the component nearest to it
        idx = np.argwhere(lab > 0)
        d = np.hypot(idx[:, 0] - (sy - oy) / r, idx[:, 1] - (sx - ox) / r)
        seed = lab[tuple(idx[np.argmin(d)])]
    return lab == seed


# ------------------------------------------------------------------ sample

def cmd_sample(a):
    rng = np.random.default_rng(a.seed)
    out = []
    for floor in [int(f) for f in a.floors.split(",")]:
        arr, meta = load_map(floor)
        r, (ox, oy) = meta["resolution"], meta["origin"][:2]
        all_cells = np.argwhere(component(floor, END_CLEAR))
        corr_cells = np.argwhere(corridor_mask(floor))
        for band, (lo, hi) in BANDS.items():
            cells = corr_cells if band == "corridor" else all_cells
            xy = np.c_[ox + (cells[:, 1] + 0.5) * r, oy + (cells[:, 0] + 0.5) * r]
            got, tries = 0, 0
            while got < a.per_band and tries < 200 * a.per_band:
                tries += 1
                s = xy[rng.integers(len(xy))]
                # goal uniform on the annulus lo..hi around s, snapped to a component cell
                ang, rad = rng.uniform(-math.pi, math.pi), math.sqrt(rng.uniform(lo * lo, hi * hi))
                g = s + rad * np.array([math.cos(ang), math.sin(ang)])
                c, rr = int((g[0] - ox) / r), int((g[1] - oy) / r)
                if not (0 <= rr < arr.shape[0] and 0 <= c < arr.shape[1]):
                    continue
                k = np.flatnonzero((cells[:, 0] == rr) & (cells[:, 1] == c))
                if not len(k):
                    continue
                out.append(dict(id=f"f{floor}_{band}_{got:04d}", floor=floor, band=band,
                                start=[round(float(s[0]), 3), round(float(s[1]), 3),
                                       round(float(rng.uniform(-math.pi, math.pi)), 4)],
                                goal=[round(float(g[0]), 3), round(float(g[1]), 3),
                                      round(float(rng.uniform(-math.pi, math.pi)), 4)]))
                got += 1
            print(f"floor {floor} {band}: {got} pairs ({tries} draws)")
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(maps={f: MAPS[f][0] for f in MAPS}, bands=BANDS, end_clear=END_CLEAR,
                   seed=a.seed, candidates=out), open(os.path.join(a.out, "candidates.json"), "w"), indent=1)
    print(f"{len(out)} candidates -> {a.out}/candidates.json")


# ------------------------------------------------------------------ solve

def _solve(job):
    c, out_dir, margin = job
    out = os.path.join(out_dir, "plans", c["id"] + ".npz")
    if os.path.exists(out):
        return dict(id=c["id"], status="cached")
    import time
    t0 = time.time()
    from bench_pmp_rollout import STACK_PLANNER_PARAMS
    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner import rollout as ro
    from agx_planning.vector_field import VectorFieldConfig, compute_field
    from agx_planning.vector_field.field import field_result_to_grid
    arr, meta = load_map(c["floor"])
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]
    # Crop to the pair's box + margin: the full floor-2 bake is 6.4 M cells.
    xs, ys = (c["start"][0], c["goal"][0]), (c["start"][1], c["goal"][1])
    c0 = max(0, int((min(xs) - margin - ox) / r)); c1 = min(arr.shape[1], int((max(xs) + margin - ox) / r))
    r0 = max(0, int((min(ys) - margin - oy) / r)); r1 = min(arr.shape[0], int((max(ys) + margin - oy) / r))
    sub, cox, coy = np.ascontiguousarray(arr[r0:r1, c0:c1]), ox + c0 * r, oy + r0 * r
    g = lambda p: (int((p[0] - cox) / r), int((p[1] - coy) / r))  # noqa: E731
    gc, sc = g(c["goal"]), g(c["start"])
    rec = dict(id=c["id"], floor=c["floor"], band=c["band"])
    try:
        res = compute_field(sub, gc[0], gc[1], r, cox, coy, VectorFieldConfig(**FIELD),
                            sc[0], sc[1], 65, allow_unknown=True)
        if res is None or res[0] is None:
            return dict(rec, status="no-field", t=round(time.time() - t0, 1))
        cfg = PlannerConfig(mode="offline")
        for k, v in STACK_PLANNER_PARAMS.items():
            setattr(cfg, k, v)
        solver = PMPShootingSolver(cfg, field_result_to_grid(res[0], field_eps=cfg.field_eps,
                                                             wall_sigma=cfg.fp_smooth_sigma))
        solver.reset_warm_start()
        x0 = np.array([*c["start"], 0.0, 0.0])
        goal = np.array(c["goal"])
        poses, cmds, cos, status = [], [], [], "?"
        gen = ro.rollout_generator(solver, cfg, x0.copy(), goal)
        try:
            while True:
                ch = next(gen)
                poses.append(ch.poses); cmds.append(ch.wheel_cmds); cos.append(ch.costates)
        except StopIteration as stop:
            status = str(getattr(stop.value, "status", "?"))
    except Exception as e:  # noqa: BLE001 -- a crashed solve is a failed solve
        status = "exception:" + repr(e)[:160]
    rec.update(status=status, t=round(time.time() - t0, 1))
    if status != "success" or not poses:
        return rec
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, poses=np.concatenate(poses), wheel_cmds=np.concatenate(cmds),
             costates=np.concatenate(cos), dt_sample=1.0 / cfg.control_rate,
             start_xy=np.array(c["start"][:2]), goal_xy=np.array(c["goal"][:2]),
             goal_pose=np.array(c["goal"]), floor=c["floor"], band=c["band"],
             map_yaml=MAPS[c["floor"]][0], shape="", screen_blocked=False, screen_detour=0.0,
             screen_score=0.0)
    return rec


def cmd_solve(a):
    cands = json.load(open(os.path.join(a.out, "candidates.json")))["candidates"]
    if a.only:
        cands = [c for c in cands if any(c["id"].startswith(p) for p in a.only.split(","))]
    # Interleave floors/bands so a partial run is still a balanced sample.
    cands.sort(key=lambda c: int(c["id"].rsplit("_", 1)[1]))
    if a.limit:
        cands = cands[:a.limit]
    log = open(os.path.join(a.out, "solve_log.jsonl"), "a")
    n_ok = 0
    with mp.Pool(a.jobs, maxtasksperchild=20) as pool:
        for i, rec in enumerate(pool.imap_unordered(_solve, [(c, a.out, a.crop_margin) for c in cands])):
            log.write(json.dumps(rec) + "\n"); log.flush()
            n_ok += rec["status"] in ("success", "cached")
            if (i + 1) % 25 == 0 or i + 1 == len(cands):
                print(f"{i + 1}/{len(cands)} done, {n_ok} solved", flush=True)


# ------------------------------------------------------------------ describe

def foote_boundaries(F, half=10, sigma=0.35, min_gap=8, thresh=0.15):
    """Foote checkerboard novelty peaks of a (n, d) feature series -> boundary indices."""
    n = len(F)
    if n < 2 * half + 2:
        return []
    D2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(-1)
    S = np.exp(-D2 / (2 * sigma ** 2))
    u = np.arange(-half, half) + 0.5
    taper = np.exp(-(u[:, None] ** 2 + u[None, :] ** 2) / (2 * (0.5 * half) ** 2))
    K = np.sign(u)[:, None] * np.sign(u)[None, :] * taper
    K /= np.abs(K).sum()
    nov = np.zeros(n)
    for i in range(half, n - half):
        nov[i] = (K * S[i - half:i + half, i - half:i + half]).sum()
    nov = np.maximum(nov, 0)  # similar same-side blocks, dissimilar cross blocks -> positive
    peaks = [i for i in range(1, n - 1) if nov[i] > thresh and nov[i] >= nov[i - 1] and nov[i] >= nov[i + 1]]
    out = []
    for p in sorted(peaks, key=lambda i: -nov[i]):
        if all(abs(p - q) >= min_gap for q in out):
            out.append(p)
    return sorted(out)


def seg_label(v, w):
    mv, mw = float(np.mean(v)), float(np.mean(w))
    if abs(mv) < 0.08 and abs(mw) > 0.2:
        return "SPIN"
    if mv < -0.05:
        return "REV"
    if abs(mw) < 0.12 or abs(mw / max(abs(mv), 1e-3)) < 0.25:  # curvature < 0.25 1/m
        return "STR"
    return "L" if mw > 0 else "R"


def describe(path, floor_edt=None):
    z = np.load(path)
    P, dt = z["poses"], float(z["dt_sample"])
    floor = int(z["floor"])
    xy, th = P[:, :2], np.unwrap(P[:, 2])
    d = np.diff(xy, axis=0)
    ds = np.hypot(d[:, 0], d[:, 1])
    fwd = np.sign(d[:, 0] * np.cos(th[:-1]) + d[:, 1] * np.sin(th[:-1]))
    v = np.r_[fwd * ds / dt, 0.0]
    w = np.r_[np.diff(th) / dt, 0.0]
    k = 5  # 0.5 s moving average: the per-sample signal is chunk-stitched
    sm = lambda x: np.convolve(x, np.ones(k) / k, mode="same")  # noqa: E731
    v, w = sm(v), sm(w)
    F = np.c_[v / V_MAX, w / W_MAX]
    b = [0] + foote_boundaries(F) + [len(F)]
    segs = []
    for s0, s1 in zip(b, b[1:]):
        lab = seg_label(v[s0:s1], w[s0:s1])
        turn = float(np.sum(w[s0:s1]) * dt)
        # A long straight with one smooth corner has a low MEAN curvature, so
        # seg_label calls it STR; the corner shows in the integrated heading.
        if lab == "STR" and abs(turn) > CORNER_RAD:
            lab = "L" if turn > 0 else "R"
        if segs and segs[-1]["lab"] == lab:
            segs[-1].update(s1=s1, turn=segs[-1]["turn"] + turn)
        else:
            segs.append(dict(lab=lab, s0=s0, s1=s1, turn=turn))
    # merge sub-0.8 s non-spin segments into the longer neighbour
    i = 0
    while len(segs) > 1 and i < len(segs):
        sg = segs[i]
        if (sg["s1"] - sg["s0"]) * dt < 0.8 and sg["lab"] != "SPIN":
            j = i - 1 if i > 0 and (i == len(segs) - 1 or segs[i - 1]["s1"] - segs[i - 1]["s0"]
                                     >= segs[i + 1]["s1"] - segs[i + 1]["s0"]) else i + 1
            o = segs[j]
            o.update(s0=min(o["s0"], sg["s0"]), s1=max(o["s1"], sg["s1"]), turn=o["turn"] + sg["turn"])
            segs.pop(i)
            # re-merge equal neighbours
            m = 1
            while m < len(segs):
                if segs[m]["lab"] == segs[m - 1]["lab"]:
                    segs[m - 1].update(s1=segs[m]["s1"], turn=segs[m - 1]["turn"] + segs[m]["turn"]); segs.pop(m)
                else:
                    m += 1
            i = 0
            continue
        i += 1
    seq = [s["lab"] for s in segs]
    # A spin is an END spin if under END_SPIN_M of travel separates it from the
    # endpoint (the planner creeps a few cm after the final spin).
    arc = np.r_[0.0, np.cumsum(ds)]
    tot = arc[-1]
    spins = [s for s in segs if s["lab"] == "SPIN"]
    at0 = [s for s in spins if arc[min(s["s0"], len(arc) - 1)] < END_SPIN_M]
    at1 = [s for s in spins if tot - arc[min(s["s1"], len(arc) - 1)] < END_SPIN_M and s not in at0]
    spin0 = sum(abs(s["turn"]) for s in at0)
    spin1 = sum(abs(s["turn"]) for s in at1)
    core = [s for s in segs if s["lab"] in ("L", "R") and abs(s["turn"]) > math.radians(20)]
    mid_spins = [s for s in spins if s not in at0 and s not in at1 and abs(s["turn"]) > math.radians(20)]
    typ = "".join(s["lab"] for s in core) or "STR"
    if mid_spins:
        typ += "+pivot"
    if any(s["lab"] == "REV" for s in segs):
        typ += "+rev"
    length = float(ds.sum())
    abs_turn = float(np.sum(np.abs(np.diff(th))))
    net_turn = float(th[-1] - th[0])
    # doors: free width across the heading < 1.1 m, counted as separate passages
    edt = floor_edt if floor_edt is not None else free_edt(floor)
    arr, meta = load_map(floor)
    r, (ox, oy) = meta["resolution"], meta["origin"][:2]

    def look(x, y):
        c = np.clip(((x - ox) / r).astype(int), 0, edt.shape[1] - 1)
        rr = np.clip(((y - oy) / r).astype(int), 0, edt.shape[0] - 1)
        return edt[rr, c], arr[rr, c]

    widths = []
    steps = np.arange(0.0, 1.5, r / 2)
    for i in range(0, len(P), 3):
        nx, ny = -math.sin(P[i, 2]), math.cos(P[i, 2])
        wd = 0.0
        for sgn in (1, -1):
            _, o = look(P[i, 0] + sgn * steps * nx, P[i, 1] + sgn * steps * ny)
            hit = np.flatnonzero(o != 0)
            wd += steps[hit[0]] if len(hit) else steps[-1]
        widths.append(wd)
    narrow = np.array(widths) < 1.1
    doors = int(np.count_nonzero(np.diff(narrow.astype(int)) == 1) + (narrow[0] if len(narrow) else 0))
    c, s = np.cos(P[:, 2:3]), np.sin(P[:, 2:3])
    lx, wy = np.linspace(-HL, HL, 17), np.linspace(-HW, HW, 15)
    body = np.vstack([np.c_[lx, np.full_like(lx, HW)], np.c_[lx, np.full_like(lx, -HW)],
                      np.c_[np.full_like(wy, HL), wy], np.c_[np.full_like(wy, -HL), wy]])
    e, _ = look(P[:, 0:1] + body[:, 0] * c - body[:, 1] * s, P[:, 1:2] + body[:, 0] * s + body[:, 1] * c)
    band = str(z["band"])
    if band != "corridor":  # sampled by start-goal distance; reported by what was solved
        band = next(b for b, (lo, hi) in LEN_BANDS.items() if lo <= length < hi)
    return dict(id=os.path.basename(path)[:-4], path=path, floor=floor, band=band, sampled_band=str(z["band"]),
                length=round(length, 2), duration=round(len(P) * dt, 1),
                abs_turn=round(abs_turn, 3), net_turn=round(net_turn, 3),
                n_turns=len(core), sign_changes=int(sum(1 for a_, b_ in zip(core, core[1:]) if a_["lab"] != b_["lab"])),
                spin_start=round(spin0, 3), spin_end=round(spin1, 3), doors=doors,
                rev_frac=round(float(np.mean(v < -0.05)), 3), min_hull_clear=round(float(e.min()), 3),
                peak_omega=round(float(np.max(np.abs(w))), 3), seq="-".join(seq), type=typ,
                segs=[[s_["lab"], s_["s0"], s_["s1"], round(s_["turn"], 3)] for s_ in segs],
                trivial=bool(not core and not mid_spins and spin0 + spin1 < math.radians(45)
                             and abs_turn - spin0 - spin1 < math.radians(90)))  # gentle S-bends are not trivial


FEATS = ["log_length", "abs_turn", "abs_net_turn", "n_turns", "sign_changes", "spin_start",
         "spin_end", "doors", "rev_frac", "min_hull_clear", "peak_omega"]


def fvec(d):
    return [math.log(d["length"]), d["abs_turn"], abs(d["net_turn"]), d["n_turns"], d["sign_changes"],
            d["spin_start"], d["spin_end"], d["doors"], d["rev_frac"], d["min_hull_clear"], d["peak_omega"]]


def maxmin_select(D, k):
    """Seed one plan per type (the most typical of it), then farthest-point."""
    if not D:
        return []
    X = np.array([fvec(d) for d in D], float)
    X = (X - X.mean(0)) / np.where(X.std(0) > 1e-9, X.std(0), 1.0)
    chosen = []
    for t in sorted({d["type"] for d in D}, key=lambda t: -sum(d["type"] == t for d in D)):
        idx = [i for i, d in enumerate(D) if d["type"] == t]
        cen = X[idx].mean(0)
        chosen.append(idx[int(np.argmin(np.linalg.norm(X[idx] - cen, axis=1)))])
        if len(chosen) >= k:
            break
    dist = np.min(np.linalg.norm(X[:, None, :] - X[chosen][None], axis=2), axis=1)
    while len(chosen) < min(k, len(D)):
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, np.linalg.norm(X - X[i], axis=1))
    nov = [round(float(min([np.linalg.norm(X[i] - X[j]) for j in chosen if j != i], default=0.0)), 3)
           for i in chosen]
    return [dict(D[i], novelty=n) for i, n in zip(chosen, nov)]


def cmd_select(a):
    import glob
    edts = {}
    D = []
    for f in sorted(glob.glob(os.path.join(a.out, "plans", "*.npz"))):
        fl = int(np.load(f)["floor"])
        if fl not in edts:
            edts[fl] = free_edt(fl)
        try:
            D.append(describe(f, edts[fl]))
        except Exception as e:  # noqa: BLE001
            print("describe failed", f, e)
    with open(os.path.join(a.out, "descriptors.jsonl"), "w") as fh:
        for d in D:
            fh.write(json.dumps(d) + "\n")
    sel = []
    for fl in sorted({d["floor"] for d in D}):
        for band in BANDS:
            pool = [d for d in D if d["floor"] == fl and d["band"] == band and not d["trivial"]]
            ntriv = sum(d["floor"] == fl and d["band"] == band and d["trivial"] for d in D)
            pick = maxmin_select(pool, a.per_band)
            sel += pick
            types = {}
            for d in pool:
                types[d["type"]] = types.get(d["type"], 0) + 1
            print(f"floor {fl} {band}: {len(pool)} non-trivial ({ntriv} trivial) -> {len(pick)} picked; "
                  f"types {dict(sorted(types.items(), key=lambda kv: -kv[1])[:10])}")
    json.dump(sel, open(os.path.join(a.out, "set.json"), "w"), indent=1)
    for fl in sorted({d["floor"] for d in sel}):
        with open(os.path.join(a.out, f"set_floor{fl}.txt"), "w") as fh:
            for d in sel:
                if d["floor"] == fl:
                    fh.write(os.path.abspath(d["path"]).replace(os.path.expanduser("~"), "__HOME__") + "\n")
        with open(os.path.join(a.out, f"set_floor{fl}_corridor.txt"), "w") as fh:
            for d in sel:
                if d["floor"] == fl and d["doors"] == 0:
                    fh.write(os.path.abspath(d["path"]).replace(os.path.expanduser("~"), "__HOME__") + "\n")
    print(f"selected {len(sel)}; corridor-only (no doors) {sum(d['doors'] == 0 for d in sel)}")


# ------------------------------------------------------------------ gallery

def cmd_gallery(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = json.load(open(os.path.join(a.out, "set.json")))
    D = [json.loads(l) for l in open(os.path.join(a.out, "descriptors.jsonl"))]
    pages = [(f"floor{fl}_{band}", [d for d in sel if d["floor"] == fl and d["band"] == band])
             for fl in sorted({d["floor"] for d in sel}) for band in BANDS]
    triv = [d for d in D if d["trivial"]]
    if triv:
        pages.append(("trivial_rejected", triv[:30]))
    col = dict(SPIN="tab:purple", REV="tab:brown", STR="0.35", L="tab:blue", R="tab:red")
    for name, ds in pages:
        if not ds:
            continue
        n = len(ds); nc = 5; nr = math.ceil(n / nc)
        fig, axs = plt.subplots(nr, nc, figsize=(3.4 * nc, 3.3 * nr), squeeze=False)
        for ax in axs.flat:
            ax.axis("off")
        for ax, d in zip(axs.flat, sorted(ds, key=lambda d: (d["type"], d["length"]))):
            z = np.load(d["path"])
            P = z["poses"]
            arr, meta = load_map(d["floor"])
            r, (ox, oy) = meta["resolution"], meta["origin"][:2]
            pad = 1.5
            x0, x1 = P[:, 0].min() - pad, P[:, 0].max() + pad
            y0, y1 = P[:, 1].min() - pad, P[:, 1].max() + pad
            span = max(x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1, y0, y1 = cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2
            c0, c1 = max(0, int((x0 - ox) / r)), min(arr.shape[1], int((x1 - ox) / r))
            r0, r1 = max(0, int((y0 - oy) / r)), min(arr.shape[0], int((y1 - oy) / r))
            sub = arr[r0:r1, c0:c1]
            ax.imshow(np.where(sub == 100, 0.15, np.where(sub == 0, 1.0, 0.8)), cmap="gray", vmin=0, vmax=1,
                      origin="lower", extent=[ox + c0 * r, ox + c1 * r, oy + r0 * r, oy + r1 * r],
                      interpolation="nearest")
            for lab, s0, s1, _ in d["segs"]:
                ax.plot(P[s0:s1 + 1, 0], P[s0:s1 + 1, 1], "-", color=col[lab], lw=1.6)
            for lab, s0, s1, turn in d["segs"]:
                if lab == "SPIN" and abs(turn) > math.radians(20):
                    m_ = min((s0 + s1) // 2, len(P) - 1)
                    ax.add_patch(plt.Circle((P[m_, 0], P[m_, 1]), 0.25 + 0.15 * abs(turn), fill=False,
                                            color="tab:purple", lw=1.2))
                    ax.annotate(f"{math.degrees(turn):+.0f}°", (P[m_, 0], P[m_, 1]), fontsize=6,
                                color="tab:purple", xytext=(4, 4), textcoords="offset points")
            for i, m in ((0, "o"), (-1, "s")):
                ax.plot(P[i, 0], P[i, 1], m, color="k", ms=4)
                ax.arrow(P[i, 0], P[i, 1], 0.6 * math.cos(P[i, 2]), 0.6 * math.sin(P[i, 2]),
                         head_width=0.18, color="k", lw=0.8)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_title(f"{d['id']}  {d['type']}\n{d['length']:.1f} m {d['duration']:.0f} s, "
                         f"turn {math.degrees(d['abs_turn']):.0f}°, spins {math.degrees(d['spin_start']):.0f}/"
                         f"{math.degrees(d['spin_end']):.0f}°, doors {d['doors']}, clr {d['min_hull_clear']:.2f}",
                         fontsize=7)
        fig.suptitle(f"{name}: {n} plans (blue L, red R, grey straight, purple spin, brown reverse; "
                     f"o start, square goal, arrows = heading)", fontsize=9)
        fig.tight_layout()
        f = os.path.join(a.out, f"gallery_{name}.png")
        fig.savefig(f, dpi=90); plt.close(fig)
        print("wrote", f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("sample"); s.add_argument("--out", required=True); s.add_argument("--floors", default="6,2")
    s.add_argument("--per-band", type=int, default=250); s.add_argument("--seed", type=int, default=0)
    s = sp.add_parser("solve"); s.add_argument("--out", required=True); s.add_argument("--jobs", type=int, default=8)
    s.add_argument("--limit", type=int, default=0); s.add_argument("--only", default="")
    s.add_argument("--crop-margin", type=float, default=12.0)
    s = sp.add_parser("select"); s.add_argument("--out", required=True); s.add_argument("--per-band", type=int, default=20)
    s = sp.add_parser("gallery"); s.add_argument("--out", required=True)
    a = ap.parse_args()
    dict(sample=cmd_sample, solve=cmd_solve, select=cmd_select, gallery=cmd_gallery)[a.cmd](a)


if __name__ == "__main__":
    main()
