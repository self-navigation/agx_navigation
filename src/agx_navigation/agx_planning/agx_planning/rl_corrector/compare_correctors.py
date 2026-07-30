"""Three-way corrector comparison over recorded (Tier-B) PMP trajectories.

Answers the question `validate_recorded.py` cannot: not "is the identity baseline
already failing?" but "does any corrector actually BEAT the identity baseline, on
trajectories of materially different shape?" Replays the SAME frozen nominal under
each corrector in turn and records the true path each one drove, so the three can
be drawn on top of each other.

    identity  execute the planner's wheel commands verbatim -- the null
              hypothesis, and the only leg that reads no pose at all.
    tvlqr     neighboring-optimal feedback (runtime_corrector/tvlqr.py), using
              its OWN authority limits (max_dv / max_domega).
    rl        the trained SAC residual policy, using ITS own authority limits
              (wheel_residual_max + action_rate_limit).

WHY THIS DRIVES THE BRIDGE DIRECTLY AND NOT WheelCorrectorEnv
-------------------------------------------------------------
The env's action IS the RL residual channel: an additive per-side wheel offset
capped by `wheel_residual_max` and slew-limited by `action_rate_limit`. Pushing
TVLQR through it would silently re-quantize a (dv, domega) correction into the
RL corrector's authority envelope -- handicapping TVLQR with a limit it does not
have when deployed, and making any comparison a statement about the channel
rather than about the two control laws. Each corrector here computes its wheel
command the same way its own deployment path does; only the bridge, the nominal,
and the terrain are shared.

The RL leg still builds its observation with the shared `build_observation`, in
exactly the argument order `env._make_obs` uses, so the policy sees the same
layout it trained on. A policy whose `observation_dim` disagrees with the current
config is rejected up front rather than fed a mislaid vector.

GROUND TRUTH, AS EVERYWHERE ELSE IN THIS PACKAGE
------------------------------------------------
Cross-track error is measured from the bridge's pose (Gazebo ground truth under
GazeboBridge), never odometry -- wheel odometry cannot observe slip, which is the
entire phenomenon being corrected.

OUTPUT
  <out-dir>/<trajectory>__<corrector>.csv   per-step: k, plan x/y/theta, true
                                            x/y/theta, e_along/e_cross/e_heading
  <out-dir>/summary.csv                     one row per (trajectory, corrector)

USAGE
  # pick trajectories of different shape first (see tools/classify_plans.py)
  python3 -m agx_planning.rl_corrector.compare_correctors \\
      --trajectories ~/pmp_trajectories_v2/floor_1_00049.npz \\
                     ~/pmp_trajectories_v2/floor_6_00042.npz \\
      --correctors identity tvlqr rl --policy ~/rl_corrector_p0.zip \\
      --bridge gazebo --terrain --out-dir /tmp/compare
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

from ..runtime_corrector import tvlqr as tvlqr_mod
from .coeff import apply_residual, clipped_action
from .config import RLCorrectorConfig
from .nominal import load_recorded
from .obs import build_observation, observation_dim


def _identity_wheels(left: float, right: float, cfg) -> list:
    m = cfg.wheel_cmd_max
    return [float(np.clip(w, -m, m)) for w in (left, left, right, right)]


def _tvlqr_wheels(left, right, planned_pose, actual_pose, cfg, tvcfg, cache, k):
    """TVLQR's own deployment path: nominal twist -> feedback -> wheels."""
    v_ref, omega_ref = tvlqr_mod.wheels_to_twist(left, right, cfg)
    err = tvlqr_mod.tracking_error(planned_pose, actual_pose)
    K = cache.get(v_ref, omega_ref)
    v_cmd, omega_cmd, diag = tvlqr_mod.correct(K, err, v_ref, omega_ref, tvcfg, k)
    wl, wr = tvlqr_mod.twist_to_wheels(v_cmd, omega_cmd, cfg)
    m = cfg.wheel_cmd_max
    wheels = [float(np.clip(w, -m, m)) for w in (wl, wl, wr, wr)]
    return wheels, diag


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectories", nargs="+", required=True,
                    help=".npz trajectory files to replay (pick different SHAPES)")
    ap.add_argument("--correctors", nargs="+", default=["identity", "tvlqr", "rl"],
                    choices=("identity", "tvlqr", "rl"))
    ap.add_argument("--policy", default="", help="SB3 .zip, required for --correctors rl")
    ap.add_argument("--bridge", choices=("kinematic", "gazebo"), default="gazebo")
    ap.add_argument("--terrain", action="store_true",
                    help="drop slip patches along the path (gazebo bridge only)")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="/tmp/compare")
    # The corridor exists to cut short a hopeless TRAINING episode; here it would
    # truncate exactly the divergent runs we want to see drawn in full.
    ap.add_argument("--stop-on-breach", action="store_true",
                    help="terminate an episode on corridor/heading breach "
                         "(default: run every episode to the full nominal length)")
    args = ap.parse_args()

    if "rl" in args.correctors and not args.policy:
        raise SystemExit("--correctors rl requires --policy")
    if args.terrain and args.bridge != "gazebo":
        raise SystemExit("--terrain only applies to --bridge gazebo")

    cfg_kwargs = {"use_costates": False}
    if not args.stop_on_breach:
        cfg_kwargs["corridor_epsilon"] = 1e9
        cfg_kwargs["max_heading_err"] = 1e9
    cfg = RLCorrectorConfig(**cfg_kwargs)

    policy = None
    if "rl" in args.correctors:
        from .policy import load_policy
        policy = load_policy(args.policy)
        if policy is None:
            raise SystemExit(f"failed to load policy from {args.policy}")
        # A width mismatch means the policy trained with different use_* toggles.
        # Catch it here rather than letting a mislaid vector look like bad control.
        # Spaces live on the wrapped SB3 model -- Policy deliberately exposes only
        # predict(), to keep torch/SB3 types out of the deployment seam.
        model = policy._model
        want = observation_dim(cfg)
        got = int(np.prod(model.observation_space.shape))
        if want != got:
            raise SystemExit(
                f"policy observation width {got} != config's {want} -- this policy "
                f"was trained with different use_* toggles / coeff_k. Deploying it "
                f"against this config would feed it a mislaid observation.")
        if int(np.prod(model.action_space.shape)) != cfg.action_dim:
            raise SystemExit("policy action_dim disagrees with config")

    tvcfg = tvlqr_mod.TVLQRConfig(enabled=True)

    if args.bridge == "kinematic":
        from .kinematic_bridge import KinematicBridge
        bridge = KinematicBridge(cfg)
    else:
        from .gazebo_bridge import GazeboBridge
        bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                              deterministic=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_rows = []

    try:
        for traj_path in args.trajectories:
            nom = load_recorded(traj_path)
            name = Path(traj_path).stem
            cache = tvlqr_mod.GainCache(tvcfg, nom.dt)

            terrain = None
            if args.terrain:
                from .terrain import along_path_terrain_sampler
                # Same terrain for every corrector on this trajectory -- the whole
                # point is that only the control law differs between the legs.
                terrain = along_path_terrain_sampler(nom.poses)(
                    np.random.default_rng(args.seed))

            for corrector in args.correctors:
                t0 = time.monotonic()
                st = bridge.reset(tuple(nom.poses[0]), terrain)
                prev_err = None
                prev_action = np.zeros(cfg.action_dim, dtype=float)
                rows = []
                max_cross = 0.0
                cross_sq = 0.0
                n_steps = len(nom)

                for k in range(n_steps):
                    planned_pose = nom.poses[k]
                    left, right = float(nom.wheels[k][0]), float(nom.wheels[k][1])

                    if corrector == "identity":
                        wheels = _identity_wheels(left, right, cfg)
                    elif corrector == "tvlqr":
                        wheels, _diag = _tvlqr_wheels(
                            left, right, planned_pose, st.pose, cfg, tvcfg, cache, k)
                    else:
                        obs, err_obs = build_observation(
                            cfg, planned_pose, st.pose, prev_err,
                            cmd_left=left, cmd_right=right,
                            v_meas=st.v, omega_meas=st.omega,
                            prev_action=prev_action,
                            imu=st.imu if cfg.use_imu else None,
                            wheel_speeds=st.wheel_speeds if cfg.use_wheel_speeds else None,
                            costates=None,
                        )
                        # Policy.predict is already deterministic and returns the
                        # action itself, not SB3's (action, state) tuple.
                        action = policy.predict(obs)
                        wheels = apply_residual(action, left, right, cfg, prev_action)
                        prev_action = clipped_action(action, cfg, prev_action)
                        # The observation's rate term differentiates ITS own error
                        # sequence, which is measured BEFORE the step -- not the
                        # post-step error logged below. Feeding the latter back
                        # would shift the rate by one step relative to training.
                        prev_err = err_obs

                    st = bridge.step(wheels, nom.dt)
                    err = tvlqr_mod.tracking_error(planned_pose, st.pose)
                    max_cross = max(max_cross, abs(err[1]))
                    cross_sq += err[1] ** 2
                    rows.append((k, planned_pose[0], planned_pose[1], planned_pose[2],
                                 st.pose[0], st.pose[1], st.pose[2],
                                 err[0], err[1], err[2]))

                out = os.path.join(args.out_dir, f"{name}__{corrector}.csv")
                with open(out, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["k", "plan_x", "plan_y", "plan_theta",
                                "true_x", "true_y", "true_theta",
                                "e_along", "e_cross", "e_heading"])
                    w.writerows(rows)

                rms_cross = (cross_sq / n_steps) ** 0.5 if n_steps else 0.0
                # Distance from where the robot ended to where the plan ends --
                # the "did it arrive" number, independent of cross-track.
                final_err = float(np.hypot(st.pose[0] - nom.poses[n_steps][0],
                                           st.pose[1] - nom.poses[n_steps][1]))
                summary_rows.append(dict(
                    trajectory=name, corrector=corrector, steps=n_steps,
                    max_cross=max_cross, rms_cross=rms_cross, final_err=final_err,
                    wall=time.monotonic() - t0))
                print(f"  {name:22s} {corrector:9s}: steps={n_steps:4d} "
                      f"max|e_cross|={max_cross:7.3f} rms|e_cross|={rms_cross:7.3f} "
                      f"final_err={final_err:7.3f}  ({time.monotonic()-t0:.1f}s)",
                      flush=True)
    finally:
        bridge.close()

    summary = os.path.join(args.out_dir, "summary.csv")
    with open(summary, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\n[compare_correctors] bridge={args.bridge} terrain={args.terrain} "
          f"-> {args.out_dir}")
    print(f"\n{'trajectory':22s} " + " ".join(f"{c:>22s}" for c in args.correctors))
    for traj_path in args.trajectories:
        name = Path(traj_path).stem
        cells = []
        for c in args.correctors:
            r = next((r for r in summary_rows
                      if r["trajectory"] == name and r["corrector"] == c), None)
            cells.append(f"{r['max_cross']:8.3f}/{r['final_err']:<8.3f}" if r else " " * 22)
        print(f"{name:22s} " + " ".join(f"{c:>22s}" for c in cells))
    print("  (cells are max|e_cross| / final_err, metres)")


if __name__ == "__main__":
    main()
