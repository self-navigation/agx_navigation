"""Train the residual wheel corrector with SAC against the Gazebo env.

This is the only deliberately heavy entrypoint in the package: it imports
stable-baselines3 / torch (the pure logic and the env do not). It wires the
GazeboBridge into WheelCorrectorEnv and runs SAC, saving the policy as an SB3
`.zip` that policy.py / the deployed _correct() load back.

Run (needs the minimal sim up; deterministic stepping gives rtf>1 headless):
    ros2 launch agx_bringup rl_corrector_sim.launch.py headless:=true
    python3 -m agx_planning.rl_corrector.train \
        --timesteps 200000 --terrain --out ~/rl_corrector_policy

or via the console script after a colcon build:
    ros2 run agx_planning rl_corrector_train --timesteps 200000 --terrain

The observation/action math is shared verbatim with deployment (obs.py / coeff.py),
so the saved policy sees the same features on-robot that SAC trained on. Reward is
computed on Gazebo GROUND-TRUTH pose (see gazebo_bridge.py), never on /odom, so the
policy is rewarded for actually staying on the path rather than for phantom
wheel-odometry progress over slip patches.
"""

import argparse
import os

from .config import RLCorrectorConfig
from .env import WheelCorrectorEnv
from .gazebo_bridge import GazeboBridge


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=200_000,
                    help="total environment steps to train for")
    ap.add_argument("--out", default=os.path.expanduser("~/rl_corrector_policy"),
                    help="output path for the saved policy (.zip appended by SB3)")
    # Sim / bridge.
    ap.add_argument("--world", default="ordjo_world")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--wall-clock", action="store_true",
                    help="step the sim in real time instead of deterministic "
                         "multi-stepping (slower; default is deterministic)")
    ap.add_argument("--terrain", action=argparse.BooleanOptionalAction, default=True,
                    help="randomize slip patches on the nominal path each episode")
    # MDP / observation.
    ap.add_argument("--action-dim", type=int, choices=(2, 4), default=4)
    ap.add_argument("--costates", action=argparse.BooleanOptionalAction, default=False,
                    help="include PMP costates in the obs (Tier-B recorded "
                         "nominals only; parametric training has none -> default off)")
    # SAC.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="torch device (cpu/cuda/auto)")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--buffer-size", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=1_000)
    ap.add_argument("--checkpoint-freq", type=int, default=20_000,
                    help="env steps between checkpoint saves (0 disables)")
    ap.add_argument("--tensorboard", default=None,
                    help="tensorboard log dir (default: none)")
    return ap.parse_args()


def build_env(args) -> WheelCorrectorEnv:
    """Construct the training env from CLI args (no SB3 import needed)."""
    cfg = RLCorrectorConfig(action_dim=args.action_dim, use_costates=args.costates)
    bridge = GazeboBridge(
        cfg, world_name=args.world, model_name=args.model,
        deterministic=not args.wall_clock,
    )
    env = WheelCorrectorEnv(cfg, bridge, seed=args.seed)

    if args.terrain:
        # Patches must land on the CURRENT episode's (randomized) nominal, so the
        # sampler reads env.nominal -- which reset() populates before it calls the
        # terrain sampler. Bind lazily here rather than to a fixed path.
        from .terrain import along_path_terrain_sampler

        def terrain_sampler(rng):
            return along_path_terrain_sampler(env.nominal.poses)(rng)

        env.terrain_sampler = terrain_sampler

    return env


def main() -> None:
    args = _parse_args()

    # Heavy deps imported only here, with a clear message if the ML stack is
    # missing (the package install_requires it, but a bare checkout may not).
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "training needs stable-baselines3 + torch:\n"
            "    pip install stable-baselines3[extra] torch\n"
            f"(import failed: {e})"
        )

    env = Monitor(build_env(args))

    callbacks = []
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.join(out_dir, "checkpoints"),
            name_prefix=os.path.basename(args.out),
        ))

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        learning_starts=args.learning_starts,
        seed=args.seed,
        device=args.device,
        tensorboard_log=args.tensorboard,
        verbose=1,
    )

    try:
        model.learn(total_timesteps=args.timesteps,
                    callback=callbacks or None,
                    progress_bar=True)
        model.save(args.out)
        print(f"[train] saved policy to {args.out}.zip")
    finally:
        # Tear the bridge down (gz subscriptions + rclpy node) even on Ctrl-C, so
        # the sim is left clean for the next run.
        env.close()


if __name__ == "__main__":
    main()
