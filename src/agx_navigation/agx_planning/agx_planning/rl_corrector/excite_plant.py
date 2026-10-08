"""Drive open-loop excitation sequences through a bridge and save the response.

The same seeded sequences go to every backend, so Gazebo and surrogate files
with the same name are comparable sample by sample.

    python3 -m agx_planning.rl_corrector.excite_plant --bridge gazebo \\
        --seeds 0-19 --plans <lib>/*.npz --out <run>/gazebo
    python3 -m agx_planning.rl_corrector.excite_plant --bridge kinematic \\
        --seeds 0-19 --out /tmp/kin

Writes one <name>.npz per sequence plus index.jsonl. The Gazebo backend needs
make rl-sim running in the same WORKER partition.
"""

import argparse
import json
import os
import time

import numpy as np

from . import excitation
from .config import RLCorrectorConfig


def parse_seeds(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bridge", choices=("gazebo", "kinematic", "dynamic"), default="gazebo")
    ap.add_argument("--seeds", default="0-9", help="random sequences, e.g. 0-19 or 1,4,7")
    ap.add_argument("--plans", nargs="*", default=[], help="library .npz files to replay open loop")
    ap.add_argument("--duration", type=float, default=60.0, help="random sequence length [s]")
    ap.add_argument("--dt", type=float, default=0.02, help="control step [s] (2 physics ticks)")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--params", default="", help="dynamic bridge: JSON file of fitted params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_wheel_speeds=True, use_imu=True)
    if args.bridge == "gazebo":
        from .gazebo_bridge import GazeboBridge
        bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                              deterministic=True)
    elif args.bridge == "kinematic":
        from .kinematic_bridge import KinematicBridge
        bridge = KinematicBridge(cfg)
    else:
        from .dynamic_bridge import DynamicBridge, DynamicParams
        params = DynamicParams.load(args.params) if args.params else DynamicParams()
        bridge = DynamicBridge(cfg, params)

    seqs = [excitation.random_segments(s, args.duration, args.dt) for s in parse_seeds(args.seeds)]
    seqs += [excitation.plan_commands(p, args.dt) for p in args.plans]
    os.makedirs(args.out, exist_ok=True)
    try:
        with open(os.path.join(args.out, "index.jsonl"), "a") as idx:
            for seq in seqs:
                t0 = time.monotonic()
                rec = excitation.run_sequence(bridge, seq)
                np.savez_compressed(os.path.join(args.out, seq.name + ".npz"), **rec)
                row = dict(name=seq.name, bridge=args.bridge, dt=seq.dt, n=len(seq.wheels),
                           wall_s=round(time.monotonic() - t0, 2),
                           final_pose=[round(float(x), 4) for x in rec["pose"][-1]])
                idx.write(json.dumps(row) + "\n"); idx.flush()
                print(json.dumps(row), flush=True)
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
