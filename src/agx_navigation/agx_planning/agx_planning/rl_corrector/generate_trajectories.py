"""Batch-generate real PMP-solved trajectories for RL corrector training (Tier-B).

Headless and ROS-free: it drives PMPShootingSolver/rollout_generator directly
against a baked floor OccupancyGrid (the same maps/floor_N.png + .yaml the
offline planner sees via rudn_ordjo_building.map_publisher), with no Gazebo, no
rclpy, no live map/goal topics. This is the practical way to generate a LARGE
trajectory library: each solve costs a BVP (tens of ms), not a sim episode.

Each solved rollout is concatenated into one poses/wheel_cmds/costates/dt_sample
bundle and written to <out_dir>/<map_stem>_<index>.npz, in exactly the shape
nominal.load_recorded() expects (RolloutChunk's field names, see rollout.py).

The training world (rl_corrector.world) has no building geometry -- the plan
avoidance problem is solved here, offline, against the real map; only the
corrector's problem (hold this trajectory under slip) needs a sim at all, and
that sim doesn't need to know what map the trajectory was planned against.

Usage:
    python3 -m agx_planning.rl_corrector.generate_trajectories \\
        --map-yaml /path/to/floor_3.yaml --count 200 --out-dir ~/pmp_trajectories

Run from a venv with scipy/scikit-fmm installed (same offline-tool convention as
bake_floor_map.py) -- these deps are deliberately not in any package's
install_requires.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from agx_planning.pmp_planner import PlannerConfig, PMPShootingSolver
from agx_planning.pmp_planner.rollout import rollout_generator
from agx_planning.utils import GeneratorReturnCatcher
from agx_planning.vector_field import VectorFieldConfig, VectorFieldGrid, compute_field, world_to_grid

from rudn_ordjo_building.map_publisher import load_occupancy_grid
from agx_bringup.random_goals import reachable_mask


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-yaml", required=True, action="append", dest="map_yamls",
                    help="baked floor map .yaml (repeatable, e.g. one per floor)")
    ap.add_argument("--count", type=int, default=100,
                    help="trajectories to generate PER map")
    ap.add_argument("--out-dir", required=True, help="output directory for .npz files")
    ap.add_argument("--clearance", type=float, default=0.45,
                    help="wall clearance for start/goal sampling [m] (matches "
                         "random_goals' default)")
    ap.add_argument("--min-range", type=float, default=2.0, help="min start-goal distance [m]")
    ap.add_argument("--max-range", type=float, default=15.0, help="max start-goal distance [m]")
    ap.add_argument("--occupancy-threshold", type=int, default=65)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def _sample_pairs(data, width, height, resolution, origin, clearance, min_range,
                   max_range, count, rng):
    """Sample `count` (start_xy, goal_xy) pairs from the map's largest reachable
    component, both endpoints drawn from it so every pair is actually reachable
    (mirrors random_goals' seed-component logic, but with no fixed seed pose --
    take the largest connected free component instead)."""
    from scipy import ndimage

    grid = np.asarray(data, dtype=np.int8).reshape(height, width)
    blocked = grid != 0
    radius_cells = int(np.ceil(clearance / resolution))
    if radius_cells > 0:
        r = radius_cells
        yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
        disk = (xx * xx + yy * yy) <= r * r
        blocked = ndimage.binary_dilation(blocked, structure=disk)
    free = ~blocked

    labels, n_labels = ndimage.label(free)
    if n_labels == 0:
        raise RuntimeError("no free cells on this map after clearance inflation")
    sizes = ndimage.sum(free, labels, index=range(1, n_labels + 1))
    largest = 1 + int(np.argmax(sizes))
    mask = labels == largest

    rows, cols = np.nonzero(mask)
    xs = origin[0] + (cols + 0.5) * resolution
    ys = origin[1] + (rows + 0.5) * resolution
    candidates = np.stack([xs, ys], axis=1)

    pairs = []
    attempts = 0
    max_attempts = count * 50
    while len(pairs) < count and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(candidates), size=2)
        p0, p1 = candidates[i], candidates[j]
        d = float(np.hypot(*(p1 - p0)))
        if min_range <= d <= max_range:
            pairs.append((p0, p1))
    if len(pairs) < count:
        raise RuntimeError(
            f"only found {len(pairs)}/{count} start-goal pairs in "
            f"[{min_range}, {max_range}] m; widen the range or lower --count"
        )
    return pairs


def _build_field(data, width, height, resolution, origin, start_xy, goal_xy,
                  occupancy_threshold, planner_cfg):
    map_array = np.asarray(data, dtype=np.int8).reshape(height, width)
    goal_cell = world_to_grid(goal_xy[0], goal_xy[1], origin[0], origin[1],
                              resolution, width, height)
    start_cell = world_to_grid(start_xy[0], start_xy[1], origin[0], origin[1],
                               resolution, width, height)
    if goal_cell is None or start_cell is None:
        return None
    outcome = compute_field(
        map_array, goal_cell[0], goal_cell[1], resolution, origin[0], origin[1],
        VectorFieldConfig(), start_cell[0], start_cell[1], occupancy_threshold,
        allow_unknown=False,
    )
    if outcome is None:
        return None
    result, _msg = outcome
    grid = VectorFieldGrid()
    grid.update(result.travel_time, origin[0], origin[1], resolution,
                field_eps=planner_cfg.field_eps)
    return grid


def generate_for_map(map_yaml: str, count: int, out_dir: str, clearance: float,
                     min_range: float, max_range: float, occupancy_threshold: int,
                     rng: np.random.Generator) -> int:
    data, width, height, meta = load_occupancy_grid(map_yaml)
    resolution = meta["resolution"]
    origin = meta["origin"]
    map_stem = Path(map_yaml).stem

    pairs = _sample_pairs(data, width, height, resolution, origin, clearance,
                          min_range, max_range, count, rng)

    planner_cfg = PlannerConfig(mode="offline")
    written = 0
    for idx, (start_xy, goal_xy) in enumerate(pairs):
        theta0 = float(rng.uniform(-np.pi, np.pi))
        goal_theta = float(rng.uniform(-np.pi, np.pi))

        field = _build_field(data, width, height, resolution, origin, start_xy,
                             goal_xy, occupancy_threshold, planner_cfg)
        if field is None:
            print(f"[{map_stem}] pair {idx}: field build failed, skipping")
            continue

        solver = PMPShootingSolver(planner_cfg, field)
        solver.reset_warm_start()
        x0 = np.array([start_xy[0], start_xy[1], theta0, 0.0, 0.0])
        goal = np.array([goal_xy[0], goal_xy[1], goal_theta])

        gen = GeneratorReturnCatcher(rollout_generator(solver, planner_cfg, x0, goal))
        wheel_cmds, poses, costates = [], [], []
        dt_sample = None
        t0 = time.perf_counter()
        for chunk in gen:
            wheel_cmds.append(chunk.wheel_cmds)
            poses.append(chunk.poses)
            costates.append(chunk.costates)
            dt_sample = chunk.dt_sample
        result = gen.value

        if result.status != "success" or not wheel_cmds:
            print(f"[{map_stem}] pair {idx}: {result.status} -- {result.message}")
            continue

        out_path = os.path.join(out_dir, f"{map_stem}_{idx:05d}.npz")
        np.savez(
            out_path,
            poses=np.concatenate(poses, axis=0),
            wheel_cmds=np.concatenate(wheel_cmds, axis=0),
            costates=np.concatenate(costates, axis=0),
            dt_sample=dt_sample,
            map_yaml=map_yaml,
            start_xy=start_xy,
            goal_xy=goal_xy,
        )
        written += 1
        print(f"[{map_stem}] pair {idx}/{count}: wrote {out_path} "
              f"({sum(w.shape[0] for w in wheel_cmds)} samples, "
              f"{time.perf_counter() - t0:.1f}s)")
    return written


def main() -> None:
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    total = 0
    for map_yaml in args.map_yamls:
        total += generate_for_map(
            map_yaml, args.count, args.out_dir, args.clearance, args.min_range,
            args.max_range, args.occupancy_threshold, rng,
        )
    print(f"done: {total} trajectories written to {args.out_dir}")


if __name__ == "__main__":
    main()
