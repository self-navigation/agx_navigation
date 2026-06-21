"""Check GazeboBridge against a live sim with the identity action.

Drives a straight nominal through the env with zero actions and reports the
max cross-track error, which should be small on flat ground and drift with
--terrain. Does not need torch.

    python3 -m agx_planning.rl_corrector.validate_gazebo --world ordjo_world --model scout_mini [--terrain]
"""

import argparse

import numpy as np

from .config import RLCorrectorConfig
from .env import WheelCorrectorEnv
from .gazebo_bridge import GazeboBridge
from .nominal import generate_primitive


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--world", default="ordjo_world")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--deterministic", action="store_true",
                    help="pause + multi_step the world (reproducible) vs wall-clock")
    ap.add_argument("--terrain", action="store_true",
                    help="drop slip patches on the path (expect drift)")
    ap.add_argument("--v", type=float, default=0.3, help="nominal forward speed")
    ap.add_argument("--duration", type=float, default=4.0, help="nominal seconds")
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False)
    bridge = GazeboBridge(
        cfg, world_name=args.world, model_name=args.model,
        deterministic=args.deterministic,
    )

    def straight(rng):
        return generate_primitive(cfg, "straight", v=args.v, omega=0.0,
                                  duration=args.duration)

    terrain_sampler = None
    if args.terrain:
        from .terrain import along_path_terrain_sampler
        nom = straight(None)
        terrain_sampler = along_path_terrain_sampler(nom.poses)

    env = WheelCorrectorEnv(
        cfg, bridge, nominal_sampler=straight,
        terrain_sampler=terrain_sampler, start_offset=(0.0, 0.0, 0.0), seed=0,
    )

    try:
        env.reset(seed=0)
        max_cross = 0.0
        steps = 0
        done = False
        while not done:
            _o, _r, terminated, truncated, info = env.step(np.zeros(cfg.action_dim,
                                                                    dtype=np.float32))
            max_cross = max(max_cross, abs(info["e_cross"]))
            steps += 1
            done = terminated or truncated
        print(f"[validate_gazebo] steps={steps} max|e_cross|={max_cross:.4f} m "
              f"succeeded={info['succeeded']} failed={info['failed']}")
        if args.terrain:
            print("  (terrain on: a large drift here is EXPECTED for identity.)")
        else:
            print("  (flat ground: expect max|e_cross| well under the corridor "
                  f"epsilon {cfg.corridor_epsilon} m.)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
