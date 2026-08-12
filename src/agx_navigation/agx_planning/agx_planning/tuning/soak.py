"""Accumulate raw rollout distributions, indefinitely, on idle machine time.

WHY THIS IS NOT A SEARCH, AND MUST NOT BECOME ONE
-------------------------------------------------
The tempting thing to leave running overnight is an optimizer. Do not. An
optimizer needs a well-posed objective, and 2026-08-12 established that ours is
not: the 7-shape mean is substantially selecting which mode two BISTABLE shapes
(the U-turn `floor_6_00031` and the S `floor_6_00018`) fall into, each landing at
~1.2 m or ~2.7 m and nothing between. Running BO or a GA harder against that
metric buys a more confident number for a quantity we already know is the wrong
one. (A GA is additionally the wrong algorithm here: 2-D, continuous, 105 s per
sample, and no noise model at all -- it breeds from the lucky draw, which is the
winner's curse that invalidated two tuning runs already.)

What idle time IS good for is the question no focused session can afford:
**how often does each mode happen, and does that frequency depend on the
gains?** That is a frequency estimate, and frequencies need tens to hundreds of
samples per condition. We currently have n=5 at two gain points. At ~5 s a
rollout an overnight run is thousands, which settles it outright.

THE PROPERTY THAT MAKES THIS SAFE TO LEAVE ON
---------------------------------------------
It writes **raw per-rollout, per-trajectory** results and NEVER aggregates. So
it survives us changing our minds about the objective: if the eval metric is
later re-weighted or normalized per shape, every row already collected is
recomputed for free. Nothing here encodes today's opinion about what the score
should be.

It expires only if the PLANT changes, which `PLANT_VERSION` records in every row
-- so a post-plant-change soak cannot be silently pooled with a pre-change one.

ONE SIM AT A TIME
-----------------
This holds the only Gazebo instance, so focused work cannot start while it runs
(`just check-sim` refuses, correctly). It is therefore built to die instantly and
lose nothing: every rollout is written, flushed and fsync'd before the next one
starts, and SIGTERM/SIGINT finish the current rollout and exit 0. `just kill-sim`
before focused work is the existing habit and remains correct.

WHY IT RESTARTS ITSELF
----------------------
`--max-rollouts` exits cleanly after N so a shell wrapper can restart the
process. `variance_probe` found no within-process drift (Spearman -0.14) but that
was over 10 rollouts, not 10 000; periodically starting fresh bounds any
accumulation we have not thought of, and labels every row with its `pid` and
`process_index` so the question can be re-asked from the data instead of assumed.

    # forever, restarting every 200 rollouts
    while true; do
      python3 -m agx_planning.tuning.soak --trajectory-config config/eval_trajectories.yaml \\
          --gains 0.276,2.618 --gains 10,0.25 --max-rollouts 200 --out ~/soak.jsonl || break
    done
"""

import argparse
import itertools
import json
import os
import signal
import time

from ..rl_corrector.config import RLCorrectorConfig
from ..rl_corrector.nominal import load_recorded
from ..runtime_corrector import tvlqr as tvlqr_mod
from .tune_tvlqr import PLANT_VERSION
from .variance_probe import drive

# Set by the signal handler; checked between rollouts. A rollout is ~5 s, so
# finishing the current one costs nothing and keeps every row complete -- a
# half-written rollout is worse than one fewer sample.
_STOP = False


def _request_stop(signum, frame):        # noqa: ARG001
    global _STOP
    _STOP = True
    print(f"[soak] signal {signum}: finishing this rollout, then stopping",
          flush=True)


def parse_gains(items):
    """`--gains q,r` -> [(q, r)]. Defaults to the two 2026-08-12 validated points.

    Rejects malformed input loudly: a soak is left unattended for hours, so a
    typo that silently fell back to a default would produce a large, confident,
    mislabelled dataset -- the most expensive kind of mistake here.
    """
    if not items:
        return [(0.2762521839107533, 2.6183452282612643), (10.0, 0.25)]
    out = []
    for item in items:
        try:
            q_str, r_str = item.split(",")
            q, r = float(q_str), float(r_str)
        except ValueError:
            raise SystemExit(f"[soak] bad --gains {item!r}: want Q,R e.g. 0.276,2.618")
        if q <= 0 or r <= 0:
            raise SystemExit(f"[soak] bad --gains {item!r}: both must be > 0")
        out.append((q, r))
    return out


def resolve_trajectories(args):
    if args.trajectories:
        return list(args.trajectories)
    if not args.trajectory_config:
        raise SystemExit("[soak] need --trajectories or --trajectory-config")
    import yaml
    with open(args.trajectory_config) as fh:
        cfg = yaml.safe_load(fh)
    root = os.path.expanduser(cfg["trajectory_dir"])
    return [os.path.join(root, f"{n}.npz") for n in cfg["selected"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+")
    ap.add_argument("--trajectory-config")
    ap.add_argument("--gains", action="append", metavar="Q,R",
                    help="gain pair to cycle; repeatable. Default: the two "
                         "points validated 2026-08-12 (tuned and default).")
    ap.add_argument("--out", required=True, help="JSONL, appended to")
    ap.add_argument("--max-rollouts", type=int, default=0,
                    help="exit cleanly after N rollouts so a wrapper can restart "
                         "the process; 0 = until signalled")
    ap.add_argument("--seed", type=int, default=0, help="terrain seed, held fixed")
    ap.add_argument("--no-terrain", action="store_true")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    gains = parse_gains(args.gains)
    traj_paths = resolve_trajectories(args)
    # Load once: these are the same objects every cycle, and re-reading them per
    # rollout would put file I/O inside the measured loop.
    noms = [(os.path.basename(p)[:-4], load_recorded(p)) for p in traj_paths]

    # Matches variance_probe exactly, including use_wheel_speeds (TVLQR does not
    # read them, so no control decision changes) and the disabled corridor/heading
    # aborts -- a soak must record what a bad rollout DOES, not cut it short.
    cfg = RLCorrectorConfig(use_costates=False, use_wheel_speeds=True,
                            corridor_epsilon=1e9, max_heading_err=1e9)

    from ..rl_corrector.gazebo_bridge import GazeboBridge
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)
    pid = os.getpid()
    n_done = 0
    t_start = time.monotonic()
    print(f"[soak] pid={pid} plant={PLANT_VERSION} "
          f"{len(noms)} trajectories x {len(gains)} gain pairs, "
          f"max_rollouts={args.max_rollouts or 'unbounded'} -> {args.out}",
          flush=True)
    try:
        with open(args.out, "a") as fh:
            # Cycle gains in the OUTER loop and trajectories inner, so an
            # interrupted soak still has balanced coverage of the gain points
            # rather than a complete picture of one and nothing of the other.
            for cycle in itertools.count():
                if _STOP or (args.max_rollouts and n_done >= args.max_rollouts):
                    break
                for q_cross, r_omega in gains:
                    tvcfg = tvlqr_mod.TVLQRConfig(enabled=True, q_cross=q_cross,
                                                  r_omega=r_omega)
                    for name, nom in noms:
                        if _STOP or (args.max_rollouts and n_done >= args.max_rollouts):
                            break
                        t0 = time.monotonic()
                        try:
                            rec = drive(bridge, cfg, tvcfg, nom, args.seed,
                                        use_terrain=not args.no_terrain)
                        except Exception as exc:              # noqa: BLE001
                            # Recorded, never silently skipped: a soak that
                            # quietly drops failures reports a survivorship-
                            # filtered distribution, which is exactly the
                            # mistake objective.py exists to prevent.
                            rec = {"failed": f"{exc.__class__.__name__}: {exc}"}
                        rec.update(trajectory=name, q_cross=q_cross,
                                   r_omega=r_omega, seed=args.seed,
                                   terrain=not args.no_terrain,
                                   plant=PLANT_VERSION, cycle=cycle,
                                   pid=pid, process_index=n_done,
                                   t=time.time(), wall=time.monotonic() - t0)
                        fh.write(json.dumps(rec) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
                        n_done += 1
                        print(f"[soak] {n_done:5d} c{cycle:04d} {name:<14s} "
                              f"q={q_cross:<7.4f} r={r_omega:<7.4f} "
                              f"max_cross={rec.get('max_cross', float('nan')):.4f} "
                              f"final={rec.get('final_err', float('nan')):.4f} "
                              f"lost={rec.get('lost_steps', -1)} "
                              f"({rec['wall']:.1f}s)", flush=True)
    finally:
        bridge.close()
    dt = time.monotonic() - t_start
    print(f"[soak] stopped after {n_done} rollouts in {dt / 60:.1f} min "
          f"({dt / max(n_done, 1):.1f}s each)", flush=True)


if __name__ == "__main__":
    main()
