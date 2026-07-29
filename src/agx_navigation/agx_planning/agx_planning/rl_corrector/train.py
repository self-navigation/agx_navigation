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
import copy
import os
import sys
from collections import deque

# Keep TensorFlow out of this process. stable-baselines3's logger does
# `from torch.utils.tensorboard import SummaryWriter` at import time, which pulls
# in tensorboard and (if installed) tensorflow -- even when --tensorboard is not
# used. TF's bundled protobuf/abseil C++ runtime clashes with gz-transport's
# protobuf runtime (gazebo_bridge imports gz.transport/gz.msgs), and loading both
# into one process segfaults the trainer right after SAC setup. Poisoning the
# module makes `import tensorflow` raise cleanly; tensorboard and SB3 both fall
# back gracefully, so .tfevents logging via torch's writer still works. Done
# before any SB3/gz import so TF never gets a chance to load.
sys.modules.setdefault("tensorflow", None)

import numpy as np  # safe: numpy does not pull in TF, unlike the SB3/torch stack

from .config import RLCorrectorConfig
from .env import (WheelCorrectorEnv, make_blended_sampler, make_nominal_sampler,
                   make_recorded_sampler)
from . import nominal as nominal_mod
# NOTE: GazeboBridge is imported lazily inside build_env (it pulls gz.transport),
# so a --bridge kinematic run is fully Gazebo-free and needs no sim process.


def make_kinematic_slip_sampler(slip_max: float = 0.3):
    """Randomized per-wheel multiplicative slip for the KinematicBridge.

    Each wheel keeps a fraction (1 - u) of its commanded speed, u ~ U[0, slip_max],
    drawn independently per wheel each episode. Asymmetric draws make the robot
    veer, so the corrector must learn to counter slip -- a fast, Gazebo-free proxy
    for terrain. Returns the {"slip": [...]} dict KinematicBridge.reset() consumes."""
    def sample(rng):
        slip = 1.0 - rng.uniform(0.0, slip_max, size=4)
        return {"slip": slip.tolist()}
    return sample


class _Tee:
    """Fan stdout out to several streams at once, so the full console (per-episode
    lines + SB3's rollout tables) can be mirrored to a log file while still showing
    live. The tqdm progress bar writes to stderr, so it is NOT duplicated into the
    file (no carriage-return spam)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self._streams:
            s.flush()

    def isatty(self):  # tqdm/SB3 probe this; defer to the real terminal
        return getattr(self._streams[0], "isatty", lambda: False)()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=200_000,
                    help="total environment steps to train for")
    ap.add_argument("--out", default=os.path.expanduser("~/rl_corrector_policy"),
                    help="output path for the saved policy (.zip appended by SB3)")
    ap.add_argument("--load", default=None,
                    help="continue training from an existing policy .zip (curriculum: "
                         "e.g. warm up on flat ground, then continue on terrain). The "
                         "obs/action layout must match -- keep --action-dim/--costates "
                         "the same across phases.")
    # Sim / bridge.
    ap.add_argument("--bridge", choices=("gazebo", "kinematic"), default="gazebo",
                    help="training backend. 'gazebo' = the real physics sim (needs "
                         "the sim up). 'kinematic' = the fast Gazebo-free analytic "
                         "bridge (thousands of steps/s): use it to PRETRAIN a baseline "
                         "policy, then --load it into a 'gazebo' run to fine-tune on "
                         "real physics. Same obs/action contract, so the transfer is "
                         "byte-compatible.")
    ap.add_argument("--kin-slip-max", type=float, default=0.3,
                    help="(kinematic + --terrain) max per-wheel slip fraction: each "
                         "wheel keeps 1-U[0,this] of its command each episode. "
                         "Asymmetric draws veer the robot so the policy learns to "
                         "counter slip. Ignored for --bridge gazebo.")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--wall-clock", action="store_true",
                    help="step the sim in real time instead of deterministic "
                         "multi-stepping (slower; default is deterministic)")
    ap.add_argument("--realtime", action="store_true",
                    help="keep the world's real-time cap (rtf=1) instead of lifting "
                         "it for max throughput. Use to watch training at 1x; default "
                         "unthrottles (rtf high, update-rate unlimited).")
    ap.add_argument("--terrain", action=argparse.BooleanOptionalAction, default=True,
                    help="randomize slip patches on the nominal path each episode")
    # Nominal sampler range -- exposed so you can drive a CURRICULUM (start easy,
    # then widen) by launching successive runs, chaining with --load.
    ap.add_argument("--nominal-kinds", nargs="+",
                    choices=("straight", "arc", "scurve"), default=None,
                    metavar="KIND",
                    help="primitive types sampled each episode (default: the full "
                         "mix, arcs 2x). Repeat to weight, e.g. "
                         "--nominal-kinds straight straight arc. Easy phase: "
                         "--nominal-kinds straight arc")
    ap.add_argument("--v-min", type=float, default=0.15, help="min body speed [m/s]")
    ap.add_argument("--v-max", type=float, default=0.45, help="max body speed [m/s]")
    ap.add_argument("--omega-max", type=float, default=1.0,
                    help="max |omega| [rad/s] for arcs/scurves (sampled [0.2, "
                         "omega_max], random sign). Lower => gentler turns; the "
                         "high-|omega| turns are where the heading corridor breaches.")
    ap.add_argument("--duration-min", type=float, default=2.0, help="min primitive [s]")
    ap.add_argument("--duration-max", type=float, default=5.0, help="max primitive [s]")
    # MDP / observation.
    ap.add_argument("--action-dim", type=int, choices=(2, 4), default=2,
                    help="2 = per-side residual (hardware-realizable, default); "
                         "4 = independent per-wheel residual (sim-only).")
    ap.add_argument("--imu", action=argparse.BooleanOptionalAction, default=True,
                    help="include the IMU (gyro_z + body accel) in the obs: a "
                         "slip-observing, on-robot signal. On by default; must match "
                         "deployment (the deployed corrector reads /imu/data too).")
    ap.add_argument("--costates", action=argparse.BooleanOptionalAction, default=False,
                    help="include PMP costates in the obs (Tier-B recorded "
                         "nominals only; parametric training has none -> default off)")
    ap.add_argument("--recorded-dir", default=None,
                    help="directory of .npz trajectories from generate_trajectories.py "
                         "(Tier-B, real PMP-solved rollouts). Default: none, synthetic "
                         "primitives only.")
    ap.add_argument("--recorded-frac", type=float, default=1.0,
                    help="fraction of episodes drawn from --recorded-dir when set (the "
                         "rest are synthetic primitives). 1.0 = recorded only. A mixed "
                         "curriculum with --costates has zero-costate synthetic episodes "
                         "(see env._costate_at) -- a real distribution-shift risk, so "
                         "prefer 1.0 once Tier-B is introduced rather than a permanent "
                         "blend.")
    ap.add_argument("--corridor-epsilon", type=float, default=None,
                    help="override RLCorrectorConfig.corridor_epsilon [m] (default "
                         "0.5). Widen this for early curriculum phases so episodes "
                         "survive long enough to get gradient signal -- see "
                         "rl-corrector-turn-induced-corridor-breach memory. Reward "
                         "stays dense regardless (w_ontrack/w_cross/w_progress are "
                         "per-step), so widening only changes the termination bound.")
    ap.add_argument("--w-effort", type=float, default=None,
                    help="override RLCorrectorConfig.w_effort (default 0.1): penalty "
                         "on the clipped action's magnitude, i.e. non-zero additive "
                         "control.")
    ap.add_argument("--w-smooth", type=float, default=None,
                    help="override RLCorrectorConfig.w_smooth (default 0.1): penalty "
                         "on action change between steps.")
    # SAC.
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="torch device (cpu/cuda/auto). NB throughput is gated by CPU "
                         "contention, not the bridge: with a Gazebo sim (esp. the GUI) "
                         "or other heavy desktop apps running, the machine is "
                         "oversubscribed and cuda (~12 fps) beats cpu (<1 fps) by "
                         "offloading the matmuls. For a true-speed offline kinematic "
                         "run, FREE THE CPU FIRST (stop the sim -- kinematic needs none).")
    ap.add_argument("--torch-threads", type=int, default=1,
                    help="cap torch/OpenMP intra-op threads (default 1). The SAC "
                         "policy/critic are tiny [256,256] MLPs: on CPU, torch's "
                         "default of one thread per core OVERSUBSCRIBES and thrashes "
                         "(esp. under desktop contention), so single-threaded is both "
                         "faster and far steadier here. Set 0 to leave torch's default "
                         "(only worth it on an otherwise-idle many-core box).")
    ap.add_argument("--debug-steps", type=int, default=0,
                    help="print the first N steps of every episode (measured twist, "
                         "tracking error, error-rate, reward). For hunting the Gazebo "
                         "reset-transition spike; 0 disables.")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--buffer-size", type=int, default=300_000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-starts", type=int, default=1_000)
    ap.add_argument("--ent-coef", default="auto_0.1",
                    help="SAC entropy temperature. 'auto_0.1' auto-tunes but starts "
                         "the temperature at 0.1 instead of SB3's default 1.0 -- the "
                         "default's wide-open early exploration randomly scaled the "
                         "wheels by +-50%% and breached the corridor before the critic "
                         "could learn (SAC_4 diverged this way). Pass a float to pin it.")
    ap.add_argument("--checkpoint-freq", type=int, default=5_000,
                    help="env steps between checkpoint saves (0 disables). Each "
                         "checkpoint is a resumable policy .zip under "
                         "checkpoints/; pass it to --load to continue from there.")
    ap.add_argument("--eval-freq", type=int, default=2_500,
                    help="(kinematic only) env steps between deterministic evals on a "
                         "separate held-out KinematicBridge. Saves the BEST-so-far "
                         "policy to <out>_best/best_model.zip, so the run keeps the "
                         "peak even when SAC's later updates wobble (the SAC_2/3 limit "
                         "cycle that degrades goal-rate after ~step 7k). Also logs a "
                         "clean eval/mean_reward curve to TensorBoard. 0 disables. "
                         "Skipped for --bridge gazebo: one live sim can't host a second "
                         "eval env without fighting the training episode.")
    ap.add_argument("--eval-episodes", type=int, default=20,
                    help="episodes averaged per --eval-freq evaluation.")
    ap.add_argument("--tensorboard", default=None,
                    help="tensorboard log dir (default: none)")
    ap.add_argument("--log-file", default=None,
                    help="mirror all console output (per-episode lines + SB3 "
                         "tables) to this file too. Default: train_<timestamp>.log "
                         "next to --out. Pass an empty string to disable.")
    return ap.parse_args()


def build_env(args) -> WheelCorrectorEnv:
    """Construct the training env from CLI args (no SB3 import needed)."""
    cfg_overrides = {}
    if args.corridor_epsilon is not None:
        cfg_overrides["corridor_epsilon"] = args.corridor_epsilon
    if args.w_effort is not None:
        cfg_overrides["w_effort"] = args.w_effort
    if args.w_smooth is not None:
        cfg_overrides["w_smooth"] = args.w_smooth
    cfg = RLCorrectorConfig(action_dim=args.action_dim, use_imu=args.imu,
                            use_costates=args.costates, **cfg_overrides)
    if args.bridge == "kinematic":
        from .kinematic_bridge import KinematicBridge
        bridge = KinematicBridge(cfg)
    else:
        from .gazebo_bridge import GazeboBridge
        bridge = GazeboBridge(
            cfg, world_name=args.world, model_name=args.model,
            deterministic=not args.wall_clock,
            unthrottle=not args.realtime,
        )
    primitive_sampler = make_nominal_sampler(
        cfg,
        kinds=args.nominal_kinds,
        v_range=(args.v_min, args.v_max),
        omega_max=args.omega_max,
        duration_range=(args.duration_min, args.duration_max),
    )
    if args.recorded_dir:
        recorded_sampler = make_recorded_sampler(nominal_mod.load_recorded_dir(args.recorded_dir))
        if args.recorded_frac >= 1.0:
            sampler = recorded_sampler
        else:
            sampler = make_blended_sampler(
                [recorded_sampler, primitive_sampler],
                [args.recorded_frac, 1.0 - args.recorded_frac],
            )
    else:
        sampler = primitive_sampler
    env = WheelCorrectorEnv(cfg, bridge, nominal_sampler=sampler, seed=args.seed,
                            debug_steps=args.debug_steps)

    if args.terrain:
        if args.bridge == "kinematic":
            # Fast analytic slip: a per-wheel multiplier sampled per episode (no
            # geometry needed -- the bridge applies it directly to the commands).
            env.terrain_sampler = make_kinematic_slip_sampler(args.kin_slip_max)
        else:
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

    # Cap CPU thread fan-out BEFORE torch is imported (SB3 pulls it in below).
    # torch reads OMP_NUM_THREADS at import; setting it here keeps the matmul
    # backend from spawning one thread per core for our tiny MLPs. See
    # --torch-threads: single-threaded is faster + steadier on a contended box.
    if args.torch_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))

    # Heavy deps imported only here, with a clear message if the ML stack is
    # missing (the package install_requires it, but a bare checkout may not).
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import (
            BaseCallback, CheckpointCallback, EvalCallback)
        from stable_baselines3.common.monitor import Monitor
        import torch
    except ImportError as e:  # pragma: no cover - environment guard
        raise SystemExit(
            "training needs stable-baselines3 + torch:\n"
            "    pip install stable-baselines3[extra] torch\n"
            f"(import failed: {e})"
        )

    # Belt-and-suspenders alongside the OMP/MKL env above: torch's own intra-op
    # pool. Clamps the [256,256] SAC matmuls to args.torch_threads cores.
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        print(f"[train] torch intra-op threads capped at {args.torch_threads}")

    # SB3's Monitor only surfaces ep_rew_mean/ep_len_mean, which can't tell a
    # goal-reaching episode from a corridor breach from a run that just truncated
    # at the nominal's end -- they all just move the mean reward around. The env
    # already returns succeeded/failed/e_cross/outcome in the step info (see
    # env.py), so this callback both prints a one-line summary of every finished
    # episode (what nominal it tracked + how it ended) and logs outcome metrics.
    #
    # Two flavours of metric, deliberately:
    #   * Cumulative counters (rollout/episodes, success_total, failure_total) are
    #     monotonic and logged from the first episode -- never sample-size
    #     misleading.
    #   * Windowed rates (rollout/success_rate, failure_rate over the last
    #     `window` episodes) are NOT emitted until that many episodes have
    #     actually completed. Emitting a rate over a handful of episodes is what
    #     made SAC_4's success_rate read as a spurious 0.06 "peak" from a single
    #     early fluke before decaying to 0 (a small-window artifact, not a
    #     regression). Gating removes that illusion. success_rate + failure_rate
    #     < 1 means the remainder truncated short of the goal tolerance.
    #
    # BaseCallback is defined here (not at module scope) to keep the SB3/torch
    # import lazy, preserving the TF-poisoning order enforced above.
    # Every distinct outcome_kind the env can emit (see env._classify_outcome).
    # Listed explicitly so each gets a TB series from step 0 -- a kind that never
    # happens reads as a flat 0 rather than silently missing.
    OUTCOME_KINDS = ("goal", "corridor", "heading", "collision",
                     "ran_out", "timeout", "nonfinite")
    FAILURE_KINDS = ("corridor", "heading", "collision", "nonfinite")

    class EpisodeOutcomeCallback(BaseCallback):
        def __init__(self, window: int = 100, print_episodes: bool = True):
            super().__init__()
            self._window = window
            self._print_episodes = print_episodes
            self._term_cross = deque(maxlen=window)
            self._kinds = deque(maxlen=window)          # recent outcome_kind strings
            self._episodes = 0
            self._kind_total = {k: 0 for k in OUTCOME_KINDS}

        def _emit(self, line: str) -> None:
            # tqdm.write keeps the line from clobbering the progress bar.
            try:
                from tqdm import tqdm
                tqdm.write(line)
            except Exception:  # pragma: no cover - tqdm always present with SB3
                print(line, flush=True)

        def _on_step(self) -> bool:
            for done, info in zip(self.locals["dones"], self.locals["infos"]):
                if not done:
                    continue
                self._episodes += 1
                kind = info.get("outcome_kind") or "timeout"
                self._kind_total[kind] = self._kind_total.get(kind, 0) + 1
                self._kinds.append(kind)
                if "e_cross" in info:
                    self._term_cross.append(abs(float(info["e_cross"])))
                if self._print_episodes:
                    label = info.get("nominal_label") or "?"
                    outcome = info.get("outcome") or "ended"
                    self._emit(f"[ep {self._episodes:5d} | {self.num_timesteps:7d} steps] "
                               f"{label}  ->  {outcome}")

            # Cumulative counters: unambiguous from episode #1.
            self.logger.record("rollout/episodes", self._episodes)
            self.logger.record("rollout/success_total", self._kind_total["goal"])
            self.logger.record("rollout/failure_total",
                               sum(self._kind_total[k] for k in FAILURE_KINDS))
            # Per-outcome-kind cumulative counts: shows WHICH failure dominates
            # (corridor breach vs heading breach vs ...) instead of one lumped flag.
            for k in OUTCOME_KINDS:
                self.logger.record(f"outcomes/{k}_total", self._kind_total[k])
            # Windowed fractions: only once the window is actually full, so a
            # handful of early episodes can't read as a spurious rate (SAC_4).
            if len(self._kinds) >= self._window:
                self.logger.record("rollout/success_rate",
                                   self._kinds.count("goal") / len(self._kinds))
                self.logger.record("rollout/failure_rate",
                                   sum(self._kinds.count(k) for k in FAILURE_KINDS)
                                   / len(self._kinds))
                for k in OUTCOME_KINDS:
                    self.logger.record(f"outcomes/{k}_rate",
                                       self._kinds.count(k) / len(self._kinds))
            if self._term_cross:
                self.logger.record("rollout/terminal_abs_e_cross",
                                   float(np.mean(self._term_cross)))
            return True

    env = Monitor(build_env(args))

    callbacks = [EpisodeOutcomeCallback()]
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    if args.checkpoint_freq > 0:
        callbacks.append(CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=os.path.join(out_dir, "checkpoints"),
            name_prefix=os.path.basename(args.out),
        ))

    # Deterministic eval + save-best. SAC here reaches a good policy early (~step
    # 7k) then walks away from it as the actor/critic limit-cycle (SAC_2/3), so the
    # FINAL model is worse than the peak. A periodic deterministic eval on a held-out
    # env keeps the best policy regardless of later wobble. Kinematic only: it builds
    # a SECOND bridge (pure numpy, no shared world), seeded apart so the eval paths
    # are a fixed held-out set. A Gazebo run has one live sim, so a 2nd eval env would
    # fight the in-flight training episode -- skip there (matches the deploy note).
    if args.eval_freq > 0:
        if args.bridge == "kinematic":
            eval_args = copy.copy(args)
            eval_args.seed = args.seed + 1000
            eval_env = Monitor(build_env(eval_args))
            best_dir = os.path.join(out_dir, os.path.basename(args.out) + "_best")
            callbacks.append(EvalCallback(
                eval_env,
                best_model_save_path=best_dir,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                render=False,
                verbose=1,
            ))
            print(f"[train] deterministic eval every {args.eval_freq} steps "
                  f"({args.eval_episodes} eps); best policy -> "
                  f"{os.path.join(best_dir, 'best_model.zip')}")
        else:
            print("[train] --eval-freq ignored for --bridge gazebo "
                  "(single live sim can't host a separate eval env)")

    if args.load:
        # Curriculum continuation: restore the policy/critic weights and attach
        # the new env. SB3 reloads the saved hyperparameters, so the per-phase
        # CLI hyperparams (--learning-rate, --buffer-size, ...) do NOT apply to a
        # loaded run. The replay buffer is intentionally NOT restored -- the
        # previous phase's transitions came from different dynamics (e.g. flat
        # ground), so the terrain phase starts collecting fresh experience.
        model = SAC.load(
            args.load, env=env, device=args.device,
            tensorboard_log=args.tensorboard,
        )
        print(f"[train] continuing from {args.load}")
    else:
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            ent_coef=args.ent_coef,
            seed=args.seed,
            device=args.device,
            tensorboard_log=args.tensorboard,
            verbose=1,
        )

    # Mirror the console to a log file. Set before learn() so SB3's logger (which
    # grabs sys.stdout when it configures inside learn()) writes through the tee.
    log_path = args.log_file
    if log_path is None:
        from datetime import datetime
        log_path = os.path.join(out_dir, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
    logf = open(log_path, "w", buffering=1) if log_path else None
    if logf is not None:
        sys.stdout = _Tee(sys.__stdout__, logf)
        print(f"[train] mirroring console output to {log_path}")

    try:
        # Continued runs keep counting timesteps from the loaded total (so the
        # checkpoints/tensorboard read as one curriculum); fresh runs start at 0.
        model.learn(total_timesteps=args.timesteps,
                    callback=callbacks or None,
                    reset_num_timesteps=args.load is None,
                    progress_bar=True)
        model.save(args.out)
        print(f"[train] saved policy to {args.out}.zip")
    finally:
        # Tear the bridge down (gz subscriptions + rclpy node) even on Ctrl-C, so
        # the sim is left clean for the next run.
        env.close()
        sys.stdout = sys.__stdout__
        if logf is not None:
            logf.close()


if __name__ == "__main__":
    main()
