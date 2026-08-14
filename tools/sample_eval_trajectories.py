#!/usr/bin/env python3
"""Generate an evaluation set by CONSTRUCTION instead of by luck.

THE PROBLEM
-----------
The 100-plan library came from uniform random start/goal pairs with a distance
filter, and it shows: the gallery's first page holds every interesting shape and
pages 3-5 are straight lines. A distance filter cannot help -- a long straight
corridor passes it perfectly. So the evaluation set is capped by what the
library happens to contain, and every per-shape claim this project has made
rests on exactly ONE plan of that shape. The U-turn notch is 5906 rollouts of
`floor_6_00031`; we cannot tell a property of U-turns from a property of that
U-turn.

THE APPROACH
------------
Screening is cheap and the PMP solve is expensive, so invert the ratio: sample
many pairs, PREDICT each route with a grid search costing milliseconds, score
the prediction for how much of a problem it poses, and only then pay for PMP on
the survivors.

The screen applies the two constraints that actually generate work for a planner
(the reason a plan has any shape at all):

  1. the straight line start->goal must be BLOCKED, so the plan has to route
     around something rather than drive at the goal;
  2. the start and goal HEADINGS force in-place rotation -- the realistic case,
     since a robot is parked facing whatever it was facing and must dock facing
     whatever the goal requires. Neither is likely to be the path's direction.

WHY DIJKSTRA AND NOT FM2
------------------------
The runtime planner uses Fast Marching Square, so FM2 would be the exact proxy.
But it needs `skfmm`, and keeping this runnable without the ROS/sim stack is
worth more than exactness for a SCREEN -- nothing here decides a result, it only
decides what is worth solving. To keep the proxy fair rather than merely cheap,
the search reproduces FM2's defining behaviour: the step cost rises as clearance
falls, so routes bow away from walls instead of hugging them, which is what
makes an FM2 path look different from a shortest path.

`--validate` measures that fairness against the 100 plans we already have. RUN
2026-08-14, and it changed the design:

    length          corr(PMP, predicted) = +0.99
    straightness                          +0.96
    total_abs_turn                        +0.30      <-- not usable
    sign_changes                          +0.34      <-- not usable

The cheap route knows WHERE the plan goes and not HOW it turns, and smoothing
the lattice staircase does not rescue it (+0.32 at best; label agreement rises
15% -> 52% only because both distributions shift toward STRAIGHT, which is
agreement without predictive power).

So the screen ranks ONLY on what survives that check -- blocked line of sight,
detour, pivot demand -- and shape is labelled afterwards from the SOLVED plan,
where the descriptors are real. Screening on predicted tortuosity would have
ranked candidates on a signal ~uncorrelated with the planner, invisibly. This is
why the proxy gets validated before it gets trusted.

OFFLINE TOOL: numpy + scipy + PIL + yaml, no ROS, no Gazebo, no skfmm. The PMP
solve (`--solve`) is the one part that needs the workspace, and it is a separate
step deliberately, so screening can be iterated on a laptop.

    # is the proxy fair? (needs traj_data/)
    tools/sample_eval_trajectories.py --validate --plans traj_data

    # screen pairs into a candidate list (laptop)
    tools/sample_eval_trajectories.py --map src/rudn-ordjo-building/maps/floor_6.yaml \\
        --count 2000 --keep 400 --out candidates.json

    # solve them and label the real plans (needs the workspace, so: the VM)
    tools/sample_eval_trajectories.py --solve candidates.json --out-dir traj_data_v2
"""

import argparse
import glob
import heapq
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "src", "agx_navigation", "agx_planning"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                "src", "rudn-ordjo-building"))

from agx_planning.tuning.shape import (  # noqa: E402
    Candidate, descriptors, label, line_blocked, pivot_demand, screen_score,
)


# --------------------------------------------------------------------- map


def load_map(map_yaml):
    """Load a baked floor map as (occupied, free_inflated, origin, resolution).

    `occupied` is the raw obstacle mask used for line-of-sight; `free_inflated`
    is that dilated by the robot clearance and is what routes may use. They are
    deliberately different: a line of sight must be blocked by a REAL wall, not
    by the inflation halo around one, or every long pair would read as blocked.
    """
    import yaml
    from PIL import Image
    from scipy import ndimage

    # Read the map directly rather than via rudn_ordjo_building.map_publisher,
    # which imports rclpy -- this tool is deliberately runnable without the ROS
    # stack. The baked PNG is plain greyscale by design (254 free / 0 wall /
    # 205 unknown, see CLAUDE.md), so there is nothing subtle to reproduce.
    with open(map_yaml) as fh:
        meta = yaml.safe_load(fh)
    resolution = float(meta["resolution"])
    origin = (float(meta["origin"][0]), float(meta["origin"][1]))
    img_path = meta["image"]
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(map_yaml)), img_path)

    # PNG row 0 is y_MAX; the OccupancyGrid convention is row 0 = y_min, which
    # is what every world<->cell conversion here assumes. Flip once, here.
    px = np.flipud(np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8))

    free_thresh = float(meta.get("free_thresh", 0.196))
    if int(meta.get("negate", 0)):
        px = 255 - px
    occupancy = (255 - px.astype(np.float64)) / 255.0     # 0 free .. 1 occupied
    # Anything not confidently free counts as blocked -- that includes unknown
    # (205), which the baker writes everywhere outside the building envelope. A
    # route out there is not a route.
    occupied = occupancy > free_thresh

    r = int(math.ceil(0.45 / resolution))          # random_goals' clearance
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    free = ~ndimage.binary_dilation(occupied, structure=disk)
    return occupied, free, origin, resolution


def largest_component(free):
    from scipy import ndimage
    labels, n = ndimage.label(free)
    if n == 0:
        raise RuntimeError("no free cells after clearance inflation")
    sizes = ndimage.sum(free, labels, index=range(1, n + 1))
    return labels == (1 + int(np.argmax(sizes)))


# ------------------------------------------------------------------- route


def clearance_cost(free, resolution, wall_weight=3.0, reach=1.2):
    """Per-cell step multiplier that rises near walls.

    This is what makes the proxy resemble FM2 rather than a shortest path: FM2's
    speed profile is a function of the distance transform, so its routes bow
    away from walls. A plain grid search would cut every corner tight and
    predict a different shape from the one the planner produces.
    """
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(free) * resolution
    near = np.clip(1.0 - dist / reach, 0.0, 1.0)
    return 1.0 + wall_weight * near * near


def route(free, cost, start_cell, goal_cell):
    """Least-cost 8-connected route, as a list of (row, col). None if unreachable.

    Plain Dijkstra: the maps are small and this runs once per candidate, so the
    A* heuristic is not worth the risk of getting it wrong for an inadmissible
    speedup on a screen.
    """
    h, w = free.shape
    sr, sc = start_cell
    gr, gc = goal_cell
    if not (free[sr, sc] and free[gr, gc]):
        return None

    INF = math.inf
    dist = np.full((h, w), INF)
    prev = np.full((h, w, 2), -1, dtype=np.int32)
    dist[sr, sc] = 0.0
    pq = [(0.0, sr, sc)]
    steps = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
             (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]

    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        if (r, c) == (gr, gc):
            break
        for dr, dc, base in steps:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
                continue
            # No diagonal squeeze between two blocked orthogonal neighbours --
            # the robot has width, and a route through a corner gap is not one.
            if dr and dc and not (free[r + dr, c] and free[r, c + dc]):
                continue
            nd = d + base * 0.5 * (cost[r, c] + cost[nr, nc])
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                prev[nr, nc] = (r, c)
                heapq.heappush(pq, (nd, nr, nc))

    if not math.isfinite(dist[gr, gc]):
        return None
    path = [(gr, gc)]
    while path[-1] != (sr, sc):
        r, c = path[-1]
        pr, pc = prev[r, c]
        if pr < 0:
            return None
        path.append((int(pr), int(pc)))
    return path[::-1]


def cells_to_world(cells, origin, resolution):
    return [(origin[0] + (c + 0.5) * resolution, origin[1] + (r + 0.5) * resolution)
            for r, c in cells]


# Smoothing half-width in metres. A route on an 8-connected lattice is a
# STAIRCASE: its heading flips between 8 discrete directions every few cells, so
# the raw route scores total_abs_turn 3.24 against the PMP plans' 1.46 and
# correlates with them at only +0.30 -- it measures the lattice, not the shape.
# Smoothing over ~0.6 m (12 cells at 0.05 m) is well below the radius of any
# real corner and well above the staircase period, which is exactly the
# separation that makes this work rather than a fitted constant.
SMOOTH_M = 0.6


def smooth_route(pts, resolution, span_m=SMOOTH_M, passes=2):
    """Moving-average the lattice staircase out of a route.

    Endpoints are pinned: they are the sampled start and goal, and moving them
    would screen a different pair than the one that gets solved.
    """
    if len(pts) < 5:
        return list(pts)
    k = max(1, int(round(span_m / resolution / 2)))
    arr = np.asarray(pts, dtype=float)
    for _ in range(passes):
        padded = np.pad(arr, ((k, k), (0, 0)), mode="edge")
        kernel = np.ones(2 * k + 1) / (2 * k + 1)
        sm = np.stack([np.convolve(padded[:, 0], kernel, mode="valid"),
                       np.convolve(padded[:, 1], kernel, mode="valid")], axis=1)
        sm[0], sm[-1] = arr[0], arr[-1]
        arr = sm
    return [(float(x), float(y)) for x, y in arr]


def world_to_cell(p, origin, resolution):
    return (int((p[1] - origin[1]) / resolution), int((p[0] - origin[0]) / resolution))


# ---------------------------------------------------------------- sampling


def screen(map_yaml, count, rng, min_range=6.0, max_range=25.0, verbose=True):
    """Sample and score candidate pairs. Returns a list of Candidate."""
    occupied, free, origin, resolution = load_map(map_yaml)
    comp = largest_component(free)
    cost = clearance_cost(free, resolution)
    rows, cols = np.nonzero(comp)
    n_cells = len(rows)

    occ_list = occupied.tolist()
    out = []
    tried = 0
    while len(out) < count and tried < count * 60:
        tried += 1
        i, j = rng.integers(0, n_cells, size=2)
        start = (origin[0] + (cols[i] + 0.5) * resolution,
                 origin[1] + (rows[i] + 0.5) * resolution)
        goal = (origin[0] + (cols[j] + 0.5) * resolution,
                origin[1] + (rows[j] + 0.5) * resolution)
        straight_d = math.hypot(goal[0] - start[0], goal[1] - start[1])
        if not (min_range <= straight_d <= max_range):
            continue

        blocked = line_blocked(start, goal, occ_list, origin, resolution)
        cells = route(comp, cost, (rows[i], cols[i]), (rows[j], cols[j]))
        if cells is None or len(cells) < 8:
            continue
        pts = cells_to_world(cells, origin, resolution)
        d = descriptors(pts)
        if d is None or d.length <= 0:
            continue

        # Headings: sample freely, exactly as the planner's own generator does.
        # We do NOT force a pivot -- we SCORE it, so the set keeps a spread of
        # rotation demands rather than making every plan start with a spin.
        start_theta = float(rng.uniform(-math.pi, math.pi))
        goal_theta = float(rng.uniform(-math.pi, math.pi))
        sp, gp = pivot_demand(start_theta, goal_theta, pts)

        out.append(Candidate(
            start=start, goal=goal, start_theta=start_theta, goal_theta=goal_theta,
            blocked=blocked, detour=d.length / straight_d if straight_d > 0 else 1.0,
            desc=d, start_pivot=sp, goal_pivot=gp, shape=label(d),
        ))
        if verbose and len(out) % 200 == 0:
            print(f"  screened {len(out)}/{count} (tried {tried})", flush=True)
    return out


# -------------------------------------------------------------- validation


def validate(plans_glob, verbose=True):
    """Is the Dijkstra route a fair proxy for what PMP actually produces?

    For each recorded plan, rebuild the prediction from its own start/goal and
    compare shapes. Reports label agreement and descriptor correlation. A screen
    that disagrees with the planner selects for the wrong thing, invisibly.
    """
    files = sorted(glob.glob(plans_glob))
    if not files:
        raise SystemExit(f"no plans matched {plans_glob}")

    by_map = {}
    rows = []
    for f in files:
        with np.load(f) as z:
            if "start_xy" not in z or "map_yaml" not in z:
                continue
            poses = z["poses"]
            start = tuple(float(v) for v in z["start_xy"])
            goal = tuple(float(v) for v in z["goal_xy"])
            map_yaml = str(z["map_yaml"])
        if not os.path.exists(map_yaml):
            map_yaml = os.path.join("src/rudn-ordjo-building/maps",
                                    os.path.basename(map_yaml))
        if map_yaml not in by_map:
            occupied, free, origin, resolution = load_map(map_yaml)
            by_map[map_yaml] = (occupied, largest_component(free), origin, resolution,
                                clearance_cost(free, resolution))
        occupied, comp, origin, resolution, cost = by_map[map_yaml]

        pmp = descriptors(list(zip(poses[:, 0], poses[:, 1])))
        sc = world_to_cell(start, origin, resolution)
        gc = world_to_cell(goal, origin, resolution)
        h, w = comp.shape
        if not (0 <= sc[0] < h and 0 <= sc[1] < w and 0 <= gc[0] < h and 0 <= gc[1] < w):
            continue
        cells = route(comp, cost, sc, gc)
        if cells is None or pmp is None:
            continue
        pred = descriptors(cells_to_world(cells, origin, resolution))
        if pred is None:
            continue
        rows.append((os.path.basename(f), label(pmp), label(pred), pmp, pred))
        if verbose and len(rows) % 20 == 0:
            print(f"  validated {len(rows)}/{len(files)}", flush=True)

    agree = sum(1 for r in rows if r[1] == r[2])
    print(f"\n{len(rows)} plans compared; label agreement "
          f"{agree}/{len(rows)} = {100 * agree / max(len(rows), 1):.0f}%")

    for attr in ("length", "total_abs_turn", "straightness", "sign_changes"):
        a = np.array([getattr(r[3], attr) for r in rows], dtype=float)
        b = np.array([getattr(r[4], attr) for r in rows], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        r = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 and a[ok].std() > 0 else float("nan")
        print(f"  {attr:>16}: corr(PMP, predicted) = {r:+.3f}   "
              f"mean {a[ok].mean():.2f} vs {b[ok].mean():.2f}")

    print("\nconfusion (rows = PMP truth, cols = predicted):")
    labels = sorted({r[1] for r in rows} | {r[2] for r in rows})
    print(f"{'':>10}" + "".join(f"{c:>9}" for c in labels))
    for t in labels:
        counts = [sum(1 for r in rows if r[1] == t and r[2] == c) for c in labels]
        print(f"{t:>10}" + "".join(f"{v:>9}" for v in counts))
    return rows


# ------------------------------------------------------------------- main


def solve(candidates_json, out_dir, occupancy_threshold=65, limit=0, verbose=True):
    """PMP-solve screened candidates and label the SOLVED plans.

    The shape label comes from here, not from the screen: a cheap route predicts
    turning at corr ~+0.30 (see `screen_score`), so the only trustworthy shape is
    the one the real planner produces. The screen's job was just to stop us
    paying for solves of pairs with no problem in them.

    Needs the workspace (scipy + skfmm + the planner), so this is the one step
    that does not run on a laptop -- hence its own mode.
    """
    import time

    from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
    from agx_planning.pmp_planner.rollout import rollout_generator
    from agx_planning.utils import GeneratorReturnCatcher
    from agx_planning.rl_corrector.generate_trajectories import _build_field
    from rudn_ordjo_building.map_publisher import load_occupancy_grid

    with open(candidates_json) as fh:
        cands = json.load(fh)
    if limit:
        cands = cands[:limit]
    os.makedirs(out_dir, exist_ok=True)

    maps = {}
    cfg = PlannerConfig(mode="offline")
    written, failed = 0, 0
    for idx, c in enumerate(cands):
        my = c["map_yaml"]
        if my not in maps:
            maps[my] = load_occupancy_grid(my)
        data, width, height, meta = maps[my]
        resolution, origin = meta["resolution"], meta["origin"]

        start_xy = np.array(c["start"], dtype=float)
        goal_xy = np.array(c["goal"], dtype=float)
        field = _build_field(data, width, height, resolution, origin, start_xy,
                             goal_xy, occupancy_threshold, cfg)
        if field is None:
            failed += 1
            continue

        solver = PMPShootingSolver(cfg, field)
        solver.reset_warm_start()
        x0 = np.array([start_xy[0], start_xy[1], c["start_theta"], 0.0, 0.0])
        goal = np.array([goal_xy[0], goal_xy[1], c["goal_theta"]])

        gen = GeneratorReturnCatcher(rollout_generator(solver, cfg, x0, goal))
        wheel_cmds, poses, costates, dt_sample = [], [], [], None
        t0 = time.perf_counter()
        for chunk in gen:
            wheel_cmds.append(chunk.wheel_cmds)
            poses.append(chunk.poses)
            costates.append(chunk.costates)
            dt_sample = chunk.dt_sample
        result = gen.value
        if result.status != "success" or not wheel_cmds:
            failed += 1
            if verbose:
                print(f"  [{idx}] {result.status}: {result.message}", flush=True)
            continue

        all_poses = np.concatenate(poses, axis=0)
        d = descriptors(list(zip(all_poses[:, 0], all_poses[:, 1])))
        if d is None:
            failed += 1
            continue
        shape = label(d)
        stem = f"{os.path.basename(my).replace('.yaml', '')}_v2_{idx:05d}"
        np.savez(
            os.path.join(out_dir, f"{stem}.npz"),
            poses=all_poses,
            wheel_cmds=np.concatenate(wheel_cmds, axis=0),
            costates=np.concatenate(costates, axis=0),
            dt_sample=dt_sample, map_yaml=my,
            start_xy=start_xy, goal_xy=goal_xy,
            # Provenance: which screen produced this, and what it predicted.
            # Lets a later session ask whether the screen was worth it without
            # re-deriving anything.
            shape=shape, screen_blocked=c["blocked"], screen_detour=c["detour"],
            screen_score=c["score"],
        )
        written += 1
        if verbose:
            print(f"  [{idx}] {stem}  {shape:>9}  turn {d.total_abs_turn:5.2f}  "
                  f"sgn {d.sign_changes}  len {d.length:5.1f}  "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    print(f"\nsolved {written}, failed {failed}, into {out_dir}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--validate", action="store_true",
                    help="check the proxy against recorded PMP plans and exit")
    ap.add_argument("--plans", default="traj_data",
                    help="directory or glob of recorded .npz plans (--validate)")
    ap.add_argument("--map", dest="maps", action="append", default=[],
                    help="baked floor map .yaml (repeatable)")
    ap.add_argument("--count", type=int, default=2000,
                    help="candidate pairs to screen per map")
    ap.add_argument("--per-shape", type=int, default=5,
                    help="how many candidates to keep per shape bucket")
    ap.add_argument("--min-range", type=float, default=6.0)
    ap.add_argument("--max-range", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occupancy-threshold", type=int, default=65)
    ap.add_argument("--out", help="write the screened selection here as JSON")
    ap.add_argument("--keep", type=int, default=400,
                    help="how many screened candidates to keep for solving")
    ap.add_argument("--require-blocked", action="store_true", default=True,
                    help="keep only pairs whose straight line is obstructed")
    ap.add_argument("--allow-clear", dest="require_blocked", action="store_false")
    ap.add_argument("--solve", metavar="CANDIDATES_JSON",
                    help="PMP-solve a screened set (needs the workspace)")
    ap.add_argument("--out-dir", default="traj_data_v2",
                    help="where --solve writes .npz plans")
    ap.add_argument("--limit", type=int, default=0, help="cap --solve at N")
    args = ap.parse_args()

    if args.solve:
        solve(args.solve, args.out_dir, args.occupancy_threshold, args.limit)
        return

    if args.validate:
        g = args.plans
        if os.path.isdir(g):
            g = os.path.join(g, "*.npz")
        validate(g)
        return

    if not args.maps:
        ap.error("--map is required unless --validate")

    rng = np.random.default_rng(args.seed)
    cands = []
    c_map = {}
    for m in args.maps:
        print(f"screening {m} ...", flush=True)
        got = screen(m, args.count, rng, args.min_range, args.max_range)
        for c in got:
            c_map[id(c)] = m
        cands.extend(got)

    print(f"\n{len(cands)} candidates screened")
    from collections import Counter
    print(f"  blocked line of sight: {sum(c.blocked for c in cands)}/{len(cands)}")

    # The screen ranks on what a cheap route predicts reliably -- blocked line
    # of sight, detour, pivot demand -- and NOT on predicted turning, which
    # correlates with the planner at only ~+0.30 (see shape.screen_score).
    # Shape is decided later, from the solved plan.
    pool = [c for c in cands if c.blocked] if args.require_blocked else list(cands)
    print(f"  kept after the blocked-line-of-sight filter: {len(pool)}")
    pool.sort(key=screen_score, reverse=True)
    picked = pool[:args.keep]
    if picked:
        print(f"  selected {len(picked)}: screen score "
              f"{screen_score(picked[-1]):.2f} .. {screen_score(picked[0]):.2f}, "
              f"detour {min(c.detour for c in picked):.2f} .. "
              f"{max(c.detour for c in picked):.2f}")
        print("  predicted shapes (INDICATIVE ONLY, the solve decides):",
              dict(Counter(c.shape for c in picked)))

    if args.out:
        with open(args.out, "w") as fh:
            json.dump([dict(
                map_yaml=c_map[id(c)], score=screen_score(c),
                start=list(c.start), goal=list(c.goal),
                start_theta=c.start_theta, goal_theta=c.goal_theta,
                blocked=bool(c.blocked), detour=c.detour,
                predicted_shape=c.shape, predicted_length=c.desc.length,
                start_pivot=c.start_pivot, goal_pivot=c.goal_pivot,
            ) for c in picked], fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
