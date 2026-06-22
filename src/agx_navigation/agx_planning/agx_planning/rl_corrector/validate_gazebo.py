"""Manual Phase-2/3 gate for the GazeboBridge against a LIVE sim.

This is the Gazebo counterpart of test_rl_env.py::test_identity_tracks_noslip_nominal:
it drives a straight nominal through the real env with an identity action (zeros)
and reports the max cross-track error. On high-friction flat ground it should stay
small; with slip terrain (--terrain) the identity policy should visibly drift --
the signal the trained policy must remove.

Run (needs gz_sim + sim_control up, e.g. the project's main launch):
    python3 -m agx_planning.rl_corrector.validate_gazebo \
        --world ordjo_world --model scout_mini [--deterministic] [--terrain]

It does NOT need torch or stable-baselines3 (identity action only).
"""

import argparse
import time

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
        t0 = time.monotonic()
        while not done:
            _o, _r, terminated, truncated, info = env.step(np.zeros(cfg.action_dim,
                                                                    dtype=np.float32))
            max_cross = max(max_cross, abs(info["e_cross"]))
            steps += 1
            done = terminated or truncated
        wall = time.monotonic() - t0
        print(f"[validate_gazebo] steps={steps} max|e_cross|={max_cross:.4f} m "
              f"succeeded={info['succeeded']} failed={info['failed']}")
        # Throughput proof: each step advances control_dt of sim time, so
        # real-time is exactly 1/control_dt steps/s. Deterministic headless
        # stepping should beat that (real-time factor > 1); ~0.1x means the
        # per-step world-control ack is still stalling the loop.
        if wall > 0.0:
            sps = steps / wall
            rtf = (steps * cfg.control_dt) / wall
            print(f"  timing: {sps:.1f} env-steps/s, {rtf:.2f}x real-time "
                  f"({steps * cfg.control_dt:.1f}s sim in {wall:.1f}s wall; "
                  f"real-time = {1.0 / cfg.control_dt:.0f} steps/s)")
        if args.terrain:
            print("  (terrain on: a large drift here is EXPECTED for identity.)")
        else:
            print("  (flat ground: expect max|e_cross| well under the corridor "
                  f"epsilon {cfg.corridor_epsilon} m.)")
    finally:
        env.close()


if __name__ == "__main__":
    main()
