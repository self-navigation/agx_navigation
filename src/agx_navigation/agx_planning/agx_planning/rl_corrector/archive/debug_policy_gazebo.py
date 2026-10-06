"""One-off: roll out a saved SB3 policy (deterministic) on GazeboBridge over a
handful of recorded trajectories, dumping per-step action/error, to check
whether a policy trained on KinematicBridge is already unstable on real physics
before any Gazebo-phase gradient updates (a bridge distribution-shift gap,
distinct from whether training itself is diverging).

python3 -m agx_planning.rl_corrector.debug_policy_gazebo \\
    --policy ~/rl_corrector_p0.zip --recorded-dir ~/pmp_trajectories \\
    --episodes 5 --world rl_corrector --model scout_mini
"""

import argparse

import numpy as np

from .config import RLCorrectorConfig
from .env import WheelCorrectorEnv, make_recorded_sampler
from .nominal import load_recorded_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--recorded-dir", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--max-steps", type=int, default=80)
    args = ap.parse_args()

    from stable_baselines3 import SAC

    cfg = RLCorrectorConfig(use_costates=False)
    model = SAC.load(args.policy)

    from .gazebo_bridge import GazeboBridge
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)

    paths = load_recorded_dir(args.recorded_dir)
    rng = np.random.default_rng(0)
    paths = list(rng.choice(paths, size=min(args.episodes, len(paths)), replace=False))
    sampler = make_recorded_sampler(paths)

    terrain_sampler = None
    if args.terrain:
        from .terrain import along_path_terrain_sampler
        def terrain_sampler(rng):
            return along_path_terrain_sampler(env.nominal.poses)(rng)

    env = WheelCorrectorEnv(cfg, bridge, nominal_sampler=sampler,
                            terrain_sampler=terrain_sampler,
                            start_offset=(0.0, 0.0, 0.0), seed=0)

    try:
        for ep in range(len(paths)):
            obs, _ = env.reset(seed=ep)
            print(f"\n--- episode {ep} ({env.nominal.label}) ---")
            print(f"{'k':>3} {'a0':>7} {'a1':>7} {'v':>7} {'w':>7} "
                  f"{'e_along':>8} {'e_cross':>8} {'e_head':>8}")
            done = False
            k = 0
            info = {}
            while not done and k < args.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
                print(f"{k:3d} {action[0]:7.3f} {action[1]:7.3f} "
                      f"{info['v']:7.3f} {info['omega']:7.3f} "
                      f"{info.get('e_along', float('nan')):8.3f} "
                      f"{info['e_cross']:8.3f} {info['e_heading']:8.3f}")
                done = terminated or truncated
                k += 1
            print(f"  ended: succeeded={info.get('succeeded')} "
                  f"failed={info.get('failed')} outcome={info.get('outcome')}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
