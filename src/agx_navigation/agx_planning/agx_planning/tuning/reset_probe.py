"""Does `GazeboBridge.reset()` actually restore the same initial state?

WHY THIS IS THE RIGHT QUESTION
------------------------------
The bridge steps Gazebo deterministically (paused world + multi_step), so the
physics is reproducible given identical inputs. That guarantees only:

    same initial state + same commands  ->  same trajectory

It says nothing about whether reset() re-establishes the same initial state. If
it does not, deterministic physics faithfully amplifies whatever differs into a
different outcome -- so the run-to-run spread is not "noise" at all, but a hidden
input we fail to control. The 2026-08-02 variance probe pointed straight here:
four of ten identical rollouts agreed to 3 decimals (0.223 m) while the rest
landed on distinct values up to 6.9 m, and there was no trend with rollout index.
Discrete, reproducible modes = a small number of distinct starting states.

WHAT THIS PROBE DOES DIFFERENTLY
--------------------------------
The variance probe drove the robot and looked at the score. This one looks at
the STATE reset hands back, and -- crucially -- varies what the robot was doing
BEFORE the reset. A reset that only works from rest is exactly the bug we are
hunting: episodes end at speed, so every reset in a real run is a reset from
motion.

    idle     nothing before reset            -- the easy case
    forward  driven straight, still rolling  -- residual linear velocity
    spin     driven in place, still yawing   -- residual angular velocity
    reverse  driven backwards                -- sign asymmetry in any damping

If reset is clean, all four scenarios return the SAME state within epsilon, and
repeats within a scenario are identical. Any spread is the bug, and the pattern
says which: spread only after `forward`/`spin` means residual velocity survives
the reset; spread even after `idle` means the settle itself is nondeterministic
(the wall-clock-paced confirm loop in _set_pose).

    python3 -m agx_planning.tuning.reset_probe --repeats 5 \\
        --trajectory ~/pmp_trajectories_v2/floor_6_00042.npz --out ~/reset_probe.jsonl
"""

import argparse
import json
import math
import os

import numpy as np

from ..rl_corrector.config import RLCorrectorConfig
from ..rl_corrector.nominal import load_recorded

# What counts as "the same state". These are deliberately loose -- we are not
# checking precision, we are checking whether the spread is ~0 or ~the size of
# the effects we measure. A reset leaving 0.05 m/s of residual speed is already
# enough to move the robot 1 cm before the first command lands.
EPS = {"x": 0.01, "y": 0.01, "theta": 0.02, "v": 0.02, "omega": 0.02}

# Wheel commands for the pre-reset disturbance, in rad/s.
SCENARIOS = {
    "idle":    [0.0, 0.0, 0.0, 0.0],
    "forward": [6.0, 6.0, 6.0, 6.0],
    "spin":    [-6.0, -6.0, 6.0, 6.0],
    "reverse": [-6.0, -6.0, -6.0, -6.0],
}


def disturb(bridge, cfg, wheels, steps):
    """Drive the robot so the reset has something real to undo."""
    for _ in range(steps):
        bridge.step(wheels, cfg.control_dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True,
                    help="only its start pose and terrain are used; the plan is "
                         "never followed")
    ap.add_argument("--repeats", type=int, default=5,
                    help="resets per scenario")
    ap.add_argument("--disturb-steps", type=int, default=20,
                    help="control steps of pre-reset motion (20 = 2 s at 10 Hz)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--settle-extra", type=int, default=0,
                    help="extra zero-command steps after reset before reading; "
                         "if a nonzero value collapses the spread, the reset is "
                         "merely under-settled rather than wrong")
    ap.add_argument("--no-terrain", action="store_true",
                    help="skip patches, to separate reset behaviour from terrain")
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False, corridor_epsilon=1e9,
                            max_heading_err=1e9)
    nom = load_recorded(args.trajectory)
    start = tuple(nom.poses[0])

    from ..rl_corrector.gazebo_bridge import GazeboBridge
    from ..rl_corrector.terrain import along_path_terrain_sampler

    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)
    rows = []
    try:
        with open(args.out, "w") as fh:
            for scenario, wheels in SCENARIOS.items():
                for i in range(args.repeats):
                    terrain = None if args.no_terrain else \
                        along_path_terrain_sampler(nom.poses)(
                            np.random.default_rng(args.seed))
                    # Put the robot in the scenario's state, THEN reset. The
                    # first iteration's disturbance also runs, so no reset in
                    # this probe is ever "the first one after startup".
                    bridge.reset(start, terrain)
                    disturb(bridge, cfg, wheels, args.disturb_steps)

                    lost0 = bridge.lost_steps
                    st = bridge.reset(start, terrain)
                    for _ in range(args.settle_extra):
                        st = bridge.step([0.0, 0.0, 0.0, 0.0], cfg.control_dt)

                    rec = {
                        "scenario": scenario, "index": i,
                        "x": float(st.pose[0]), "y": float(st.pose[1]),
                        "theta": float(st.pose[2]),
                        "z": (float(bridge._pose_z)
                              if bridge._pose_z is not None else None),
                        "v": float(st.v), "omega": float(st.omega),
                        "wheel_speeds": [float(w) for w in (st.wheel_speeds or [])],
                        "reset_ticks": int(bridge.reset_ticks),
                        "lost_steps": int(bridge.lost_steps - lost0),
                        "settle_extra": args.settle_extra,
                    }
                    rows.append(rec)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                    print(f"[reset] {scenario:8s} {i} "
                          f"pos=({rec['x']:.5f},{rec['y']:.5f}) "
                          f"th={rec['theta']:+.5f} "
                          f"z={rec['z'] if rec['z'] is None else round(rec['z'], 5)} "
                          f"v={rec['v']:+.5f} omega={rec['omega']:+.5f} "
                          f"rticks={rec['reset_ticks']} lost={rec['lost_steps']}",
                          flush=True)
    finally:
        bridge.close()

    report(rows, start)


def report(rows, start):
    fields = ["x", "y", "theta", "v", "omega"]
    print("\n=== per-scenario spread (max - min) ===")
    for scenario in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == scenario]
        if not sub:
            continue
        parts = []
        for f in fields:
            vals = [r[f] for r in sub]
            parts.append(f"{f} {max(vals) - min(vals):.5f}")
        zs = [r["z"] for r in sub if r["z"] is not None]
        ztxt = f"   z {min(zs):.4f}..{max(zs):.4f}" if zs else ""
        print(f"  {scenario:8s} " + "  ".join(parts) + ztxt)

    print("\n=== ACROSS all scenarios (this is the number that matters) ===")
    verdict_ok = True
    for f in fields:
        vals = [r[f] for r in rows]
        spread = max(vals) - min(vals)
        ok = spread <= EPS[f]
        verdict_ok &= ok
        print(f"  {f:6s} spread {spread:.5f}   eps {EPS[f]:.3f}   "
              f"{'ok' if ok else 'FAIL'}")

    # Distance from where reset was ASKED to put the robot -- a systematic
    # offset is a different bug from a spread, and both matter.
    dxy = [math.hypot(r["x"] - start[0], r["y"] - start[1]) for r in rows]
    print(f"\n  offset from requested start pose: "
          f"min {min(dxy):.4f}  max {max(dxy):.4f} m")

    ticks = [r["reset_ticks"] for r in rows]
    lost = sum(r["lost_steps"] for r in rows)
    print(f"  reset_ticks: {min(ticks)}..{max(ticks)}"
          f"{'   <- VARIES: wall-clock-paced settle' if min(ticks) != max(ticks) else ''}")
    print(f"  lost steps during disturbance: {lost}")

    print("\nVERDICT: " + ("reset is clean within epsilon" if verdict_ok else
                           "RESET IS NOT DETERMINISTIC -- this is the variance source"))


if __name__ == "__main__":
    main()
