"""Tune TVLQR's `q_cross` / `r_omega` against real Gazebo rollouts.

WHY THIS IS THE CHEAP WIN
-------------------------
On a clean world (2026-08-01) TVLQR is excellent on straights and corners and
*worse than open loop* on the S-curve, where it visibly oscillates about the
path. `q_cross=10` / `r_omega=0.25` have never been tuned -- they are the values
they were written with. This needs no training and no policy.

WHAT ONE EVALUATION IS
----------------------
A fixed list of recorded plans, each driven start-to-goal under TVLQR with the
candidate gains, on the same seeded terrain. The score is the mean of
max|e_cross| over that list (see objective.py, including why a failed rollout
invalidates the whole evaluation instead of shrinking the denominator).
Nothing is sampled and no rollout is ever truncated: `--max-evals` bounds how
many GAIN PAIRS are tried, never how far the robot drives.

OPTIMIZER
---------
Default is `--optimizer bayes` (bayesopt.py): a GP surrogate with expected
improvement, which models the measurement noise explicitly and reports the
minimizer of the POSTERIOR MEAN. Nelder-Mead (`--optimizer simplex`) is kept
only to reproduce pre-2026-08-04 runs -- it cannot converge on a noisy objective
and reports the minimum observed draw, which is how two earlier runs reported
improvements smaller than the spread they were selected from.

SEARCH SPACE
------------
Both gains are searched in log10, because they are positive scale factors
spanning orders of magnitude: a step means a ratio, so the search behaves the
same at q=0.1 and q=100, and no move can propose a negative gain. Bounds are
the physically sensible range, clipped rather than penalised.

Those bounds are an ASSUMPTION, and `--q-bounds` / `--r-bounds` override them.
The 2026-08-07 run put 23 of 100 evaluations into the bottom decade with its best
point at q=0.276, i.e. it pushed against the q=0.1 floor -- which is a reason to
doubt the floor, not to trust the answer. The flags matter for single-point
probes too: `x0` is CLIPPED into the box, so probing q=0.01 without them measures
q=0.1 and reports it under the label you asked for.

RESUMABLE
---------
Every evaluation is appended to a JSONL cache before the next one starts, and a
resumed run replays that cache for free to reconstruct the search (cache.py).
An interrupted run therefore costs nothing to continue -- which matters when one
evaluation is ~75 s and the VM has been stopped mid-run before.

    python3 -m agx_planning.tuning.tune_tvlqr \\
        --trajectories ~/pmp_trajectories_v2/floor_1_00049.npz ... \\
        --max-evals 0 --cache ~/tvlqr_tune.jsonl   # 0 = to convergence
"""

import argparse
import json
import math
import os
import signal
import time

import numpy as np

from ..rl_corrector.compare_correctors import _tvlqr_wheels
from ..rl_corrector.config import RLCorrectorConfig
from ..rl_corrector.nominal import load_recorded
from ..runtime_corrector import tvlqr as tvlqr_mod
from .bayesopt import minimize as bayes_minimize
from .cache import EvalCache
from .objective import aggregate, reduce_repeats
from .simplex import minimize

# Log10 bounds. q_cross below 0.1 is no cross-track feedback at all; above 1000
# the gains saturate the wheel limits every step. r_omega spans the same span
# around its default 0.25.
BOUNDS_LOG = [(-1.0, 3.0), (-2.0, 2.0)]

# Bump this WHENEVER THE PLANT CHANGES -- world friction, robot model, physics
# step, anything that alters what a rollout measures. It goes into the cache key,
# so a stale cache is refused instead of being replayed as if it described the
# current world. The friction weights are keyed separately (they are readable at
# runtime); this covers everything that is not, notably the world file's own
# ground <mu>, which the tuner has no way to inspect from a world NAME.
#
#   2026-08-04-linoleum-ground : world ground mu 1.0 -> 0.45 (dry linoleum),
#       matching the `linoleum` patch profile. Everything before this was
#       measured on a concrete-grippy floor.
#   2026-08-07-wheel-mu2-045 : REVERTED that ground change (0.45 was BELOW the
#       wheel's lateral mu2 of 0.7, which caps both friction channels equally and
#       destroys the anisotropy a skid-steer steers with -- the robot achieved 6%
#       of commanded yaw rate). Ground is back at 1.0 and the WHEEL's mu2 is
#       0.7 -> 0.45 instead, which is where slipperiness belongs. Measured
#       baseline on this plant: identity 2.127, TVLQR 1.127 m mean max|e_cross|,
#       objective sd 0.082 single-sample / 0.047 at --repeats 3.
PLANT_VERSION = "2026-08-07-wheel-mu2-045"


class EvalTimeout(Exception):
    pass


class _Deadline:
    """SIGALRM watchdog around one evaluation.

    With `--max-evals 0` (run to convergence) there is no budget left to bound
    the run, so a rollout that hangs -- a wedged bridge, a sim that stopped
    stepping -- would stall the search forever with no output. A completed
    evaluation is ~75 s, so anything near the cap is broken, not slow.
    """

    def __init__(self, seconds):
        self.seconds = int(seconds)

    def __enter__(self):
        if self.seconds > 0:
            signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if self.seconds > 0:
            signal.alarm(0)
        return False

    def _fire(self, signum, frame):
        raise EvalTimeout(f"evaluation exceeded {self.seconds}s")


def run_trajectories(bridge, cfg, traj_paths, q_cross, r_omega, seed, log=print):
    """Drive every trajectory under one candidate. Returns {name: max|e_cross|}.

    A trajectory missing from the returned dict is a FAILED rollout, and
    objective.aggregate turns that into an invalid evaluation rather than a
    cheaper-looking mean.
    """
    from ..rl_corrector.terrain import along_path_terrain_sampler

    tvcfg = tvlqr_mod.TVLQRConfig(enabled=True, q_cross=q_cross, r_omega=r_omega)
    out = {}
    for path in traj_paths:
        name = os.path.basename(path)[:-4]
        try:
            nom = load_recorded(path)
            cache = tvlqr_mod.GainCache(tvcfg, nom.dt)
            # Same seed every evaluation: the terrain is part of the problem, not
            # part of the noise. Re-deriving it per trajectory (not per call)
            # keeps it identical to what compare_correctors measures.
            terrain = along_path_terrain_sampler(nom.poses)(
                np.random.default_rng(seed))
            st = bridge.reset(tuple(nom.poses[0]), terrain)
            max_cross = 0.0
            for k in range(len(nom)):
                planned = nom.poses[k]
                left, right = float(nom.wheels[k][0]), float(nom.wheels[k][1])
                wheels, _ = _tvlqr_wheels(left, right, planned, st.pose,
                                          cfg, tvcfg, cache, k)
                st = bridge.step(wheels, nom.dt)
                err = tvlqr_mod.tracking_error(planned, st.pose)
                max_cross = max(max_cross, abs(err[1]))
            out[name] = float(max_cross)
        except Exception as exc:                      # noqa: BLE001
            # Deliberately broad: any failure to complete a rollout must reach
            # aggregate() as a MISSING entry. Swallowing it into a partial mean
            # is the bug objective.py exists to prevent.
            log(f"    !! {name} failed: {exc.__class__.__name__}: {exc}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+",
                    help="explicit .npz paths; omit to use --trajectory-config")
    ap.add_argument("--trajectory-config",
                    help="config/eval_trajectories.yaml -- reads `selected` and "
                         "resolves it against `trajectory_dir`. PREFER THIS: the "
                         "Justfile recipe used to carry its own hard-coded copy "
                         "of the list, which silently went stale the moment the "
                         "eval set changed, so a run could be tuned against a "
                         "different set than every document claimed.")
    ap.add_argument("--cache", default=os.path.expanduser("~/tvlqr_tune.jsonl"))
    ap.add_argument("--max-evals", type=int, default=0,
                    help="candidate GAIN PAIRS to try; 0 = run until the simplex "
                         "converges (never truncates a rollout either way)")
    ap.add_argument("--eval-timeout", type=float, default=900.0,
                    help="seconds before one evaluation is declared hung "
                         "(a healthy one is ~75 s); 0 disables")
    ap.add_argument("--max-consecutive-failures", type=int, default=5,
                    help="abort after this many failed evaluations in a row -- a "
                         "dead sim fails in milliseconds and would otherwise spin")
    ap.add_argument("--q-cross", type=float, default=10.0, help="starting q_cross")
    ap.add_argument("--r-omega", type=float, default=0.25, help="starting r_omega")
    ap.add_argument("--step", type=float, default=0.35,
                    help="initial simplex size, in log10 units (0.35 ~ x2.2); "
                         "--optimizer simplex only")
    ap.add_argument("--optimizer", default="bayes", choices=["bayes", "simplex"],
                    help="bayes: GP surrogate + expected improvement, reports "
                         "the POSTERIOR-MEAN optimum. simplex: Nelder-Mead, "
                         "which cannot converge on a noisy objective and reports "
                         "the minimum observed draw -- kept only to reproduce "
                         "pre-2026-08-04 runs. See bayesopt.py.")
    ap.add_argument("--noise", type=float, default=0.055,
                    help="known measurement sd of the objective, in metres, "
                         "passed to the GP as a pinned noise floor. Default is "
                         "the measured sd at --repeats 3; scale by sqrt(3/n) if "
                         "you change --repeats. 0 lets the GP fit it, which "
                         "over-fits scatter as structure at small n.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="rollouts of the whole trajectory set per candidate. "
                         "The objective is noisy (sd 0.093 single-sample); "
                         "averaging n cuts that by sqrt(n). 3 -> 0.055, which "
                         "resolves the effect a real gain improvement produces. "
                         "1 reproduces the old single-sample behaviour and is "
                         "NOT enough to rank candidates on this eval set.")
    ap.add_argument("--reduce", default="mean", choices=["mean", "median", "max"],
                    help="how to combine repeats, per trajectory. mean is the "
                         "measured best for search (see objective.py)")
    ap.add_argument("--seed", type=int, default=0, help="terrain seed")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--out", default=os.path.expanduser("~/tvlqr_tuned.json"))
    ap.add_argument("--q-bounds", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="override the q_cross search box, in LINEAR gain units "
                         f"(default {10.0 ** BOUNDS_LOG[0][0]:g} "
                         f"{10.0 ** BOUNDS_LOG[0][1]:g}). The default floor of "
                         "0.1 is an assumption, not a measurement, and the "
                         "2026-08-07 run piled 23 of 100 evaluations against it "
                         "-- so probing below it needs this flag. Note x0 is "
                         "CLIPPED into the box, so a single-point probe outside "
                         "the default bounds silently measures the wrong gains "
                         "without it.")
    ap.add_argument("--r-bounds", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None, help="same, for r_omega")
    args = ap.parse_args()

    trajectories = args.trajectories
    if args.trajectory_config:
        if trajectories:
            ap.error("pass --trajectories or --trajectory-config, not both")
        import yaml
        with open(args.trajectory_config) as fh:
            tc = yaml.safe_load(fh)
        selected = tc.get("selected") or []
        if not selected:
            ap.error(f"{args.trajectory_config} lists no `selected` trajectories")
        tdir = tc["trajectory_dir"]
        trajectories = [os.path.join(tdir, f"{n}.npz") for n in selected]
    if not trajectories:
        ap.error("need --trajectories or --trajectory-config")

    missing = [p for p in trajectories if not os.path.exists(p)]
    if missing:
        # Fail here rather than 40 minutes in: a missing trajectory makes every
        # evaluation inf (all-or-nothing aggregate), so the search would burn the
        # night reflecting away from a vertex that was never measurable.
        raise SystemExit("trajectories not found:\n  " + "\n  ".join(missing))

    names = [os.path.basename(p)[:-4] for p in trajectories]
    print(f"[tune] {len(names)} trajectories: {', '.join(names)}")

    cfg = RLCorrectorConfig(use_costates=False, corridor_epsilon=1e9,
                            max_heading_err=1e9)
    from ..rl_corrector.gazebo_bridge import GazeboBridge
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)

    # The cache key pins everything that would make an old cache meaningless.
    # `repeats`/`reduce` are in it because they change what a cached number MEANS
    # (a single noisy draw vs. a mean of 3), and `patch_weights` because the
    # friction distribution defines the PLANT -- replaying evaluations measured
    # on the old ice-heavy terrain would silently mix two different problems.
    from rudn_ordjo_building.surface_patches import DEFAULT_PATCH_WEIGHTS
    key = {"trajectories": sorted(names), "seed": args.seed,
           "metric": "mean_max_cross", "plant": PLANT_VERSION,
           "repeats": args.repeats, "reduce": args.reduce,
           "patch_weights": {k: round(v, 4)
                             for k, v in sorted(DEFAULT_PATCH_WEIGHTS.items())}}
    store = EvalCache(args.cache, key=key)
    recovered = store.load()
    store.write_header()
    if recovered:
        print(f"[tune] resuming: {recovered} evaluations replayed from {args.cache}")

    t_start = time.monotonic()
    state = {"n": 0, "consec_fail": 0}
    # Per-trajectory errors from the evaluation currently in flight. `wrap` calls
    # detail() straight after the objective returns, so this hands the breakdown
    # over without threading a return type through the cache API. Recorded so the
    # landscape can be plotted (and re-aggregated per shape) after the fact,
    # WITHOUT re-driving Gazebo -- an hour of rollouts is far too expensive to
    # keep only the scalar the search happened to minimize.
    inflight = {}

    def objective(x_log):
        q_cross = float(10.0 ** x_log[0])
        r_omega = float(10.0 ** x_log[1])
        state["n"] += 1
        print(f"[tune] eval {state['n']:3d}  q_cross={q_cross:9.3f} "
              f"r_omega={r_omega:8.4f}", flush=True)
        t_eval = time.monotonic()
        # REPEATS. The objective is stochastic (sd 0.093 on this eval set at
        # fixed gains), so a single sample ranks candidates by luck as often as
        # by merit -- that is what collapsed the 2026-08-03 run, which re-drew
        # ONE gain pair 71 times over a 0.364 m spread and reported the minimum.
        # Averaging n repeats cuts the sd by sqrt(n); see objective.py for the
        # measured table. Each repeat re-drives every trajectory with the same
        # seed and terrain: the spread is per-rollout physics, not sampling.
        reps = []
        try:
            with _Deadline(args.eval_timeout * args.repeats):
                for _ in range(args.repeats):
                    reps.append(run_trajectories(bridge, cfg, trajectories,
                                                 q_cross, r_omega, args.seed))
        except EvalTimeout as exc:
            print(f"        !! {exc}", flush=True)
            reps.append({})
        per = reduce_repeats(reps, names, how=args.reduce)
        score = aggregate(per, names)

        # A dead sim fails every rollout in milliseconds. Unbounded, the search
        # would "evaluate" thousands of points in seconds and converge on noise;
        # this is exactly what happened when the tuner was killed mid-evaluation
        # and its rclpy context went invalid. Stop and say so.
        if math.isfinite(score):
            state["consec_fail"] = 0
        else:
            state["consec_fail"] += 1
            if state["consec_fail"] >= args.max_consecutive_failures:
                raise SystemExit(
                    f"[tune] aborting: {state['consec_fail']} consecutive failed "
                    f"evaluations -- the sim is almost certainly dead. "
                    f"Check `pgrep -af 'gz[ -]sim'`; the cache keeps every good "
                    f"evaluation, so re-running resumes for free.")
        inflight.clear()
        inflight.update(per_traj=per, eval_index=state["n"],
                        wall=time.monotonic() - t_eval,
                        elapsed=time.monotonic() - t_start,
                        repeats=args.repeats, reduce=args.reduce,
                        # Every individual rollout, not just the reduction, so
                        # the noise can be re-analysed (and a different
                        # estimator tried) without re-driving anything.
                        per_traj_repeats=reps,
                        failed=[n for n in names if n not in per])
        print(f"        -> mean max|e_cross| = {score:.4f} m   "
              f"{ {k: round(v, 3) for k, v in per.items()} }"
              f"   [{time.monotonic() - t_start:.0f}s elapsed]", flush=True)
        return score

    def detail(x_log):
        rec = {"q_cross": float(10.0 ** x_log[0]),
               "r_omega": float(10.0 ** x_log[1])}
        rec.update(inflight)
        return rec

    wrapped = store.wrap(objective, detail=detail,
                         on_hit=lambda x, fx: print(
                             f"[tune] cached  q_cross={10.0**x[0]:9.3f} "
                             f"r_omega={10.0**x[1]:8.4f} -> {fx:.4f} m", flush=True))

    bounds_log = list(BOUNDS_LOG)
    for i, override in enumerate((args.q_bounds, args.r_bounds)):
        if override is not None:
            lo, hi = override
            if not (0.0 < lo < hi):
                raise SystemExit(f"[tune] bad bounds {lo} {hi}: need 0 < lo < hi")
            bounds_log[i] = (math.log10(lo), math.log10(hi))
    if bounds_log != list(BOUNDS_LOG):
        print(f"[tune] search box overridden: "
              f"q_cross [{10.0 ** bounds_log[0][0]:g}, {10.0 ** bounds_log[0][1]:g}] "
              f"r_omega [{10.0 ** bounds_log[1][0]:g}, {10.0 ** bounds_log[1][1]:g}]",
              flush=True)

    x0 = [math.log10(args.q_cross), math.log10(args.r_omega)]
    try:
        if args.optimizer == "bayes":
            if args.max_evals <= 0:
                # Unlike a simplex, a GP has no convergence criterion that means
                # "done" on a noisy objective -- EI stays positive as long as
                # anything is uncertain. A budget is the honest stopping rule.
                raise SystemExit("[tune] --optimizer bayes needs --max-evals > 0 "
                                 "(a GP has no natural convergence test on a "
                                 "noisy objective; pick a budget)")
            res = bayes_minimize(
                wrapped, bounds=bounds_log, max_evals=args.max_evals,
                x0=x0, noise=(args.noise if args.noise > 0 else None),
                seed=args.seed)
        else:
            res = minimize(wrapped, x0=x0, step=[args.step, args.step],
                           bounds=bounds_log,
                           max_evals=(args.max_evals if args.max_evals > 0
                                      else None),
                           xtol=1e-2, ftol=1e-3)
    finally:
        bridge.close()

    best = {"q_cross": float(10.0 ** res.x[0]), "r_omega": float(10.0 ** res.x[1]),
            "mean_max_cross": res.fx, "n_evals": res.n_evals,
            "optimizer": args.optimizer, "plant": PLANT_VERSION,
            "repeats": args.repeats, "reduce": args.reduce,
            "trajectories": names, "seed": args.seed,
            "start": {"q_cross": args.q_cross, "r_omega": args.r_omega}}
    if args.optimizer == "bayes":
        # Record BOTH, and label them. `mean_max_cross` is the posterior mean --
        # what we believe the gains are worth. `observed` is the luckiest single
        # measurement, which is what the old runs reported and is biased low.
        # Quoting the observed value is how 0.183 m and 0.9412 m got into two
        # write-ups that neither survived.
        best["value_is"] = "posterior_mean"
        best["observed"] = {
            "q_cross": float(10.0 ** res.x_observed[0]),
            "r_omega": float(10.0 ** res.x_observed[1]),
            "mean_max_cross": res.fx_observed,
            "note": "best single measurement; biased low, do not quote",
        }
        best["converged"] = res.converged
        best["message"] = res.message
    with open(args.out, "w") as fh:
        json.dump(best, fh, indent=2)

    print(f"\n[tune] best q_cross={best['q_cross']:.4f} "
          f"r_omega={best['r_omega']:.5f}  -> {res.fx:.4f} m "
          f"after {res.n_evals} evaluations")
    print(f"[tune] wrote {args.out};  full history in {args.cache}")


if __name__ == "__main__":
    main()
