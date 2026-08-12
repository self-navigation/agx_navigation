"""Run TVLQR rollouts indefinitely and append raw per-rollout results.

Writes one row per rollout and trajectory, never aggregates, and tags each
row with PLANT_VERSION, pid and process_index. Each row is flushed and
fsync'd before the next rollout, and SIGTERM/SIGINT finish the current one and
exit 0. --max-rollouts exits after N so a wrapper loop can restart it.

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
