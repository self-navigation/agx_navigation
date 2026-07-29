"""Identity-policy baseline over the RECORDED (Tier-B) trajectory library.

Bisects "is the 95% corridor-breach rate an RL-training problem, or is the task
(these trajectories + this corridor width + this terrain) already close to
unwinnable before any correction is learned?" Drives a zero-action (identity)
policy -- no torch/SB3 needed -- over a sample of --recorded-dir trajectories and
reports the corridor/heading/success breakdown, mirroring the outcome-rate
metrics train.py logs to tensorboard.

Run three ways to localize the failure:
    # 1. Kinematic, no slip -- should be ~0% corridor failures. If not, the bug
    #    is in the recorded data or the tracking-error/corridor geometry, not RL.
    python3 -m agx_planning.rl_corrector.validate_recorded \\
        --recorded-dir ~/pmp_trajectories --bridge kinematic --episodes 50

    # 2. Real Gazebo physics, flat ground -- tests the kinematic model (which
    #    bakes in slip_chi) against real DART physics with no domain randomization.
    python3 -m agx_planning.rl_corrector.validate_recorded \\
        --recorded-dir ~/pmp_trajectories --bridge gazebo --episodes 50

    # 3. Real Gazebo physics + slip terrain -- this is the number that matters:
    #    compare it to the trained policy's corridor_rate. If IDENTITY already
    #    fails this often, the corrector isn't failing to learn -- the task as
    #    currently configured is close to unwinnable from scratch.
    python3 -m agx_planning.rl_corrector.validate_recorded \\
        --recorded-dir ~/pmp_trajectories --bridge gazebo --terrain --episodes 50
"""

import argparse
import time

import numpy as np

from .config import RLCorrectorConfig
from .env import WheelCorrectorEnv, make_recorded_sampler
from .nominal import load_recorded_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recorded-dir", required=True)
    ap.add_argument("--bridge", choices=("kinematic", "gazebo"), default="kinematic")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--terrain", action="store_true",
                    help="drop slip patches on the path (gazebo bridge only)")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--deterministic", action="store_true", default=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False)
    paths = load_recorded_dir(args.recorded_dir)
    rng_pick = np.random.default_rng(args.seed)
    if len(paths) > args.episodes:
        paths = list(rng_pick.choice(paths, size=args.episodes, replace=False))
    sampler = make_recorded_sampler(paths)

    if args.bridge == "kinematic":
        from .kinematic_bridge import KinematicBridge
        bridge = KinematicBridge(cfg)
    else:
        from .gazebo_bridge import GazeboBridge
        bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                              deterministic=args.deterministic)

    terrain_sampler = None
    if args.terrain:
        if args.bridge != "gazebo":
            raise SystemExit("--terrain only applies to --bridge gazebo")
        from .terrain import along_path_terrain_sampler
        # Bound to whatever env.nominal is AT CALL TIME (set by reset() just
        # before this runs, see env.py:192/205), so it follows the current
        # episode's recorded trajectory rather than one fixed nominal.
        def terrain_sampler(rng):
            return along_path_terrain_sampler(env.nominal.poses)(rng)

    env = WheelCorrectorEnv(
        cfg, bridge, nominal_sampler=sampler,
        terrain_sampler=terrain_sampler, start_offset=(0.0, 0.0, 0.0), seed=args.seed,
    )

    outcomes = {"success": 0, "corridor": 0, "heading": 0, "contact": 0,
               "timeout_or_ran_out": 0}
    max_cross_all = []
    ep_lens = []

    try:
        t0 = time.monotonic()
        for ep in range(len(paths)):
            env.reset(seed=args.seed + ep)
            max_cross = 0.0
            steps = 0
            done = False
            info = {}
            while not done:
                _o, _r, terminated, truncated, info = env.step(
                    np.zeros(cfg.action_dim, dtype=np.float32))
                max_cross = max(max_cross, abs(info["e_cross"]))
                steps += 1
                done = terminated or truncated
            ep_lens.append(steps)
            max_cross_all.append(max_cross)
            if info.get("succeeded"):
                outcomes["success"] += 1
            elif abs(info["e_cross"]) > cfg.corridor_epsilon:
                outcomes["corridor"] += 1
            elif abs(info["e_heading"]) > cfg.max_heading_err:
                outcomes["heading"] += 1
            elif info.get("failed"):
                outcomes["contact"] += 1
            else:
                outcomes["timeout_or_ran_out"] += 1
            print(f"  ep {ep:3d}: steps={steps:4d} max|e_cross|={max_cross:.3f} "
                  f"succeeded={info.get('succeeded')} failed={info.get('failed')}",
                  flush=True)
        wall = time.monotonic() - t0
    finally:
        env.close()

    n = len(paths)
    print(f"\n[validate_recorded] bridge={args.bridge} terrain={args.terrain} "
          f"n={n} wall={wall:.1f}s")
    for k, v in outcomes.items():
        print(f"  {k:20s}: {v:3d}  ({100.0 * v / n:.1f}%)")
    print(f"  mean ep_len         : {np.mean(ep_lens):.1f} steps")
    print(f"  mean max|e_cross|   : {np.mean(max_cross_all):.3f} m "
          f"(corridor_epsilon={cfg.corridor_epsilon} m)")


if __name__ == "__main__":
    main()
