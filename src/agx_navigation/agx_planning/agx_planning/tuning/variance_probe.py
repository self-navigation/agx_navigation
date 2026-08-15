"""Measure where the offline-mode run-to-run variance comes from.

THE QUESTION
------------
Identical inputs -- same trajectory, same gains, same seed, deterministic
stepping -- have scored TVLQR at 0.224 m and at 1.549 m on floor_6_00042. That
spread is larger than most effects being measured, so it currently invalidates
any comparison of two correctors or two gain settings.

Two hypotheses, and they call for opposite fixes:

  WITHIN-process   state accumulates as one process drives repeatedly -- sim
                   clock drift in GazeboBridge._wait_clock_advance, terrain
                   models piling up, solver warm starts. Then a run's score
                   depends on how many runs preceded it, and every multi-rollout
                   tool (the tuner, compare_correctors) is measuring a moving
                   target.
  ACROSS-process   each fresh process starts the world in a slightly different
                   state (spawn settling, residual velocities, service ordering).
                   Then single rollouts are just noisy and results need repeats,
                   but a long-lived process is internally consistent -- which is
                   what the tuner assumed.

The experiment is the cheapest thing that separates them: drive the SAME
trajectory N times inside one process, and N times in N processes, then compare
the two spreads. Nothing here is aggregated across the two modes; they are
deliberately reported as separate distributions.

    # N rollouts, one process
    python3 -m agx_planning.tuning.variance_probe --mode within --repeats 10 \\
        --trajectory ~/pmp_trajectories_v2/floor_6_00042.npz --out ~/var_within.jsonl

    # one rollout; run this command N times for the across-process arm
    python3 -m agx_planning.tuning.variance_probe --mode across --repeats 1 ...
"""

import argparse
import json
import os
import time

import numpy as np

from ..rl_corrector.compare_correctors import _tvlqr_wheels
from ..rl_corrector.config import RLCorrectorConfig
from ..rl_corrector.nominal import load_recorded
from ..runtime_corrector import tvlqr as tvlqr_mod
from . import epsilon as epsilon_mod


def drive(bridge, cfg, tvcfg, nom, seed, use_terrain=True):
    """One rollout. Returns the metrics compare_correctors and the tuner use.

    `final_err` and `max_cross` are kept separately because they have failed
    independently before: a run can track well and stop short (identity on the
    straight) or wander badly and still end near the goal.
    """
    from ..rl_corrector.terrain import along_path_terrain_sampler

    cache = tvlqr_mod.GainCache(tvcfg, nom.dt)
    # Patches are the suspected AMPLIFIER of the tiny reset spread: an 8 mm
    # difference in start position decides whether a wheel catches a patch edge,
    # which is a discontinuous change in friction under that wheel. Turning them
    # off asks whether the run-to-run spread is chaos seeded by the patches or a
    # defect in the bridge.
    terrain = (along_path_terrain_sampler(nom.poses)(np.random.default_rng(seed))
               if use_terrain else None)
    lost0 = bridge.lost_steps
    st = bridge.reset(tuple(nom.poses[0]), terrain)
    # State the robot ACTUALLY starts from, which reset is supposed to make
    # identical every time. Recorded alongside the score so the two can be
    # correlated directly instead of argued about.
    start_state = {"start_pose": [float(v) for v in st.pose[:3]],
                   "start_v": float(st.v), "start_omega": float(st.omega),
                   "reset_ticks": int(bridge.reset_ticks)}
    max_cross = 0.0
    cross_sq = 0.0
    n_steps = len(nom)
    # J is accumulated ONLINE, so a rollout reports the SVCM cost functional
    # directly instead of via a trace file scored afterwards. The offline route
    # (`tools/score_epsilon.py`) still exists for already-recorded runs, but it
    # is a pairing problem -- match a track to the wrong plan and you get a
    # large, plausible, meaningless number -- and 2026-08-14 lost a whole soak's
    # traces to exactly that. The pairing convention here is deliberately the
    # SAME as the offline scorer's (post-step pose against poses[k], correction
    # as applied-minus-nominal at k) so numbers from the two are comparable.
    #
    # ONE KNOWN DIFFERENCE, not yet quantified: `diag.dv/domega` is the
    # correction TVLQR applied in twist space, already saturated by its own
    # dv_max/domega_max but taken BEFORE `_tvlqr_wheels` clips the wheel
    # commands at `wheel_cmd_max`. The offline scorer reads the clipped wheel
    # commands back and differences those, so on a rollout that clips a wheel
    # the two disagree, with the offline one charging less. Neither is wrong --
    # they answer "what did the corrector ask for" vs "what reached the plant" --
    # but do not mix them in one table without checking `saturated_*` first.
    acc = epsilon_mod.EpsilonAccumulator(nom.dt)
    for k in range(n_steps):
        planned = nom.poses[k]
        left, right = float(nom.wheels[k][0]), float(nom.wheels[k][1])
        wheels, diag = _tvlqr_wheels(left, right, planned, st.pose,
                                     cfg, tvcfg, cache, k)
        st = bridge.step(wheels, nom.dt)
        err = tvlqr_mod.tracking_error(planned, st.pose)
        max_cross = max(max_cross, abs(err[1]))
        cross_sq += err[1] ** 2
        acc.push(err, (diag.dv, diag.domega))
    # `poses` holds n_steps+1 entries; index n_steps is the goal, matching
    # compare_correctors exactly. Using [-1] would silently measure something
    # else if the recorder ever pads.
    goal = nom.poses[n_steps]
    final_err = float(np.hypot(st.pose[0] - goal[0], st.pose[1] - goal[1]))
    score = acc.finalize(final_err)
    out = {"max_cross": float(max_cross), "final_err": final_err,
           "rms_cross": float((cross_sq / n_steps) ** 0.5) if n_steps else 0.0,
           # The SVCM currency, and its split: a single J cannot say WHETHER a
           # rollout was expensive because it wandered or because it fought the
           # plan, and those are the two failure modes.
           "j_total": score.j_total, "j_tracking": score.j_tracking,
           "j_control": score.j_control, "j_terminal": score.j_terminal,
           "end_pose": [float(v) for v in st.pose[:3]],
           # Steps whose physics never happened before the wall-clock deadline
           # expired. Non-zero means this rollout is not the trajectory the
           # commands describe, and comparing it to another rollout is invalid.
           "lost_steps": int(bridge.lost_steps - lost0), "n_steps": n_steps}
    out.update(start_state)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--mode", choices=["within", "across"], required=True,
                    help="labels the records; 'across' is meant to be run once "
                         "per process by an outer shell loop")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--q-cross", type=float, default=10.0)
    ap.add_argument("--r-omega", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0, help="terrain seed, held fixed")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--no-terrain", action="store_true",
                    help="drive on bare ground -- isolates the bridge's own "
                         "reproducibility from chaos seeded by patch edges")
    ap.add_argument("--reset-world", action="store_true",
                    help="full gz world reset every episode -- the only thing "
                         "that zeroes JOINT velocities, which the settle loops "
                         "leave at ~1e-9 rad/s. Tests whether that residual is "
                         "the last seed of the run-to-run spread")
    ap.add_argument("--out", required=True, help="JSONL, appended to")
    ap.add_argument("--trace-dir",
                    help="write a per-rollout state trace CSV here "
                         "(<trace-dir>/<pid>_<index>.csv); compare two with "
                         "python3 -m agx_planning.tuning.trace_diff")
    args = ap.parse_args()

    name = os.path.basename(args.trajectory)[:-4]
    # use_wheel_speeds subscribes /joint_states. TVLQR does not read them, so
    # this changes no control decision -- it is on so the trace can show whether
    # a commanded wheel speed actually took effect on the step it was issued for
    # (the ROS-publish / gz-step race). Harmless here; it would NOT be harmless
    # for an RL policy, where it changes the observation layout.
    cfg = RLCorrectorConfig(use_costates=False, use_wheel_speeds=True,
                            corridor_epsilon=1e9,
                            max_heading_err=1e9)
    tvcfg = tvlqr_mod.TVLQRConfig(enabled=True, q_cross=args.q_cross,
                                  r_omega=args.r_omega)
    nom = load_recorded(args.trajectory)

    from ..rl_corrector.gazebo_bridge import GazeboBridge
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True, reset_world=args.reset_world)
    pid = os.getpid()
    try:
        with open(args.out, "a") as fh:
            for i in range(args.repeats):
                t0 = time.monotonic()
                if args.trace_dir:
                    os.makedirs(args.trace_dir, exist_ok=True)
                    trace_path = os.path.join(args.trace_dir,
                                              f"{name}_{pid}_{i:02d}.csv")
                    bridge.enable_trace(trace_path)
                try:
                    rec = drive(bridge, cfg, tvcfg, nom, args.seed,
                                use_terrain=not args.no_terrain)
                except Exception as exc:                      # noqa: BLE001
                    rec = {"failed": f"{exc.__class__.__name__}: {exc}"}
                # `index` is the whole point of the within-process arm: if the
                # spread is drift rather than noise, the score correlates with it.
                rec.update(mode=args.mode, index=i, pid=pid, trajectory=name,
                           terrain=not args.no_terrain,
                           reset_world=args.reset_world,
                           q_cross=args.q_cross, r_omega=args.r_omega,
                           seed=args.seed, wall=time.monotonic() - t0)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                print(f"[var] {args.mode} {i:2d} pid={pid} "
                      f"max_cross={rec.get('max_cross', float('nan')):.4f} "
                      f"final={rec.get('final_err', float('nan')):.4f} "
                      f"lost={rec.get('lost_steps', -1)} "
                      f"rticks={rec.get('reset_ticks', -1)} "
                      f"v0={rec.get('start_v', float('nan')):+.4f} "
                      f"({rec['wall']:.0f}s)", flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
