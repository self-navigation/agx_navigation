"""One-off: dump per-step commanded vs measured wheel speed for a single recorded
trajectory under the identity policy + GazeboBridge, to see whether corridor
breaches come from gradual physics divergence or an actuator-lag/step change.

python3 -m agx_planning.rl_corrector.debug_one_trajectory \\
    --traj /home/programmer/pmp_trajectories/floor_1_00000.npz \\
    --world rl_corrector --model scout_mini
"""

import argparse

import numpy as np

from .config import RLCorrectorConfig
from .env import WheelCorrectorEnv
from .gazebo_bridge import GazeboBridge
from .nominal import load_recorded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False)
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)

    nom = load_recorded(args.traj)

    def sampler(rng):
        return nom

    env = WheelCorrectorEnv(cfg, bridge, nominal_sampler=sampler,
                            start_offset=(0.0, 0.0, 0.0), seed=0)

    try:
        env.reset(seed=0)
        print(f"traj n={len(nom)} dt={nom.dt}")
        print(f"{'k':>3} {'cmd_l':>7} {'cmd_r':>7} {'v_cmd':>7} {'w_cmd':>7} "
              f"{'v_meas':>7} {'w_meas':>7} {'e_along':>8} {'e_cross':>8} {'e_head':>8}")
        done = False
        k = 0
        while not done and k < 80:
            cmd_idx = min(env.k, len(nom) - 1)
            cmd_l, cmd_r = nom.wheels[cmd_idx]
            v_cmd, w_cmd = cfg.wheels_to_body(cmd_l, cmd_r)
            _o, _r, terminated, truncated, info = env.step(
                np.zeros(cfg.action_dim, dtype=np.float32))
            print(f"{k:3d} {cmd_l:7.2f} {cmd_r:7.2f} {v_cmd:7.3f} {w_cmd:7.3f} "
                  f"{info['v']:7.3f} {info['omega']:7.3f} "
                  f"{info.get('e_along', float('nan')):8.3f} "
                  f"{info['e_cross']:8.3f} {info['e_heading']:8.3f}")
            done = terminated or truncated
            k += 1
        print(f"terminated: succeeded={info['succeeded']} failed={info['failed']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
