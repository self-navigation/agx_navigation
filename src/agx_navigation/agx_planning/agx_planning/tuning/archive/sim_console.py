"""Interactive single-step console for the Gazebo bridge.

WHY THIS EXISTS
---------------
Rollouts disagree even when the robot's reset is bit-exact and every requested
patch is verified present. The remaining difference is something that happens
between a service request and the world actually reflecting it, and that is
invisible from a batch script: by the time a rollout finishes, whatever raced has
already been absorbed into the trajectory.

So: drive the world one tick at a time, by hand, and watch WHEN each change
lands. Every command prints the entity list and the robot pose, so "how many
steps until the patch appears" is read off directly instead of inferred.

Both a human (over Moonlight, with a GUI attached via `just gui`) and an agent
(over ssh, screenshotting with `DISPLAY=:0 import -window root`) can drive it.

    python3 -m agx_planning.tuning.sim_console \\
        --trajectory ~/pmp_trajectories_v2/floor_6_00042.npz

COMMANDS (bare Enter repeats the previous one)
    <Enter> / s [n]   step n physics ticks (default 1)
    c [n]             step n CONTROL steps (n * control_dt of sim time)
    p                 request this episode's patches (does NOT step)
    x                 request removal of tracked patches (does NOT step)
    t                 teleport the robot to the trajectory start (no settle)
    R                 full bridge reset() -- the thing under suspicion
    w <l> <r>         set wheel command (rad/s), applied on following steps
    0                 zero the wheel command
    e                 print entities + pose now, without stepping
    d                 diff entities against the previous printout
    q                 quit

The point of `p`/`x` NOT stepping is that gz-sim commits entity changes on a
world step; separating "request" from "step" is what makes the commit latency
observable.
"""

import argparse
import os

import numpy as np

from ..rl_corrector.config import RLCorrectorConfig
from ..rl_corrector.nominal import load_recorded


def fmt_entities(names, interesting=("rl_patch", "rl_ground", "scout")):
    """Entity names worth showing -- the world also contains walls, sun, etc."""
    hits = sorted(n for n in names
                  if any(n.startswith(pfx) or pfx in n for pfx in interesting))
    return ", ".join(hits) if hits else "(none)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False, corridor_epsilon=1e9,
                            max_heading_err=1e9)
    nom = load_recorded(args.trajectory)
    start = tuple(nom.poses[0])

    from ..rl_corrector.gazebo_bridge import GazeboBridge
    from ..rl_corrector.terrain import along_path_terrain_sampler

    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)
    terrain = along_path_terrain_sampler(nom.poses)(
        np.random.default_rng(args.seed))
    wanted = [p.get("name") or f"rl_patch_{i}" for i, p in enumerate(terrain)]

    print(f"\ntrajectory {os.path.basename(args.trajectory)}   start {start}")
    print(f"this episode's patches: {wanted}")
    for p in terrain:
        print(f"    {p.get('name')}: profile={p['profile']} "
              f"at ({p['x']:.2f},{p['y']:.2f}) "
              f"{p['width']:.2f}x{p['length']:.2f} yaw={p['yaw']:+.2f}")
    print("\ntype 'q' to quit, Enter to step one tick, '?' for commands\n")

    wheels = [0.0, 0.0, 0.0, 0.0]
    prev_names = set()
    ticks = 0
    last = "s 1"

    def show(tag=""):
        nonlocal prev_names
        bridge._exec.spin_once(timeout_sec=0.02)
        names = set(bridge._entity_names)
        pose = bridge._pose_xyth
        z = bridge._pose_z
        added = sorted(names - prev_names)
        gone = sorted(prev_names - names)
        prev_names = names
        pose_txt = ("?" if pose is None else
                    f"({pose[0]:+.4f},{pose[1]:+.4f},{pose[2]:+.4f})"
                    f" z={z:.4f}" if z is not None else "")
        print(f"  [tick {ticks:5d}] {tag}")
        print(f"     pose {pose_txt}")
        print(f"     patches present: {fmt_entities(names)}")
        missing = [n for n in wanted if n not in names]
        if missing:
            print(f"     MISSING: {missing}")
        if added:
            print(f"     +appeared: {fmt_entities(added)}")
        if gone:
            print(f"     -vanished: {fmt_entities(gone)}")

    show("initial")
    try:
        while True:
            try:
                raw = input("sim> ").strip()
            except EOFError:
                break
            if raw == "":
                raw = last
            last = raw
            parts = raw.split()
            cmd = parts[0]

            if cmd == "q":
                break
            elif cmd == "?":
                print(__doc__.split("COMMANDS")[1])
            elif cmd in ("s", "step") or cmd.isdigit():
                n = int(parts[1]) if len(parts) > 1 else (
                    int(cmd) if cmd.isdigit() else 1)
                for _ in range(n):
                    bridge._node.publish_wheels(wheels)
                    bridge._world_control(multi_step=1,
                                          ack_ms=bridge._step_ack_ms)
                    ticks += 1
                show(f"stepped {n} tick(s)")
            elif cmd == "c":
                n = int(parts[1]) if len(parts) > 1 else 1
                for _ in range(n):
                    bridge._node.publish_wheels(wheels)
                    bridge._advance(cfg.control_dt)
                    ticks += int(round(cfg.control_dt / bridge.physics_step))
                show(f"stepped {n} control step(s)")
            elif cmd == "p":
                # Request only -- deliberately no step, so the commit latency is
                # visible on the following `s`.
                from ..rl_corrector.terrain import patch_sdf
                from gz.msgs10.entity_factory_pb2 import EntityFactory
                from gz.msgs10.boolean_pb2 import Boolean
                for i, patch in enumerate(terrain):
                    name = patch.get("name") or f"rl_patch_{i}"
                    req = EntityFactory()
                    req.sdf = patch_sdf(patch, name)
                    req.name = name
                    req.pose.position.x = float(patch["x"])
                    req.pose.position.y = float(patch["y"])
                    req.pose.position.z = float(patch.get("z", 0.001))
                    ok = bridge._gz.request(bridge._svc_create, req,
                                            EntityFactory, Boolean,
                                            bridge._ack_ms)
                    if name not in bridge._terrain_models:
                        bridge._terrain_models.append(name)
                    print(f"     create {name}: ack={ok}")
                show("requested spawn (NOT stepped)")
            elif cmd == "x":
                from gz.msgs10.entity_pb2 import Entity
                from gz.msgs10.boolean_pb2 import Boolean
                for name in list(bridge._terrain_models) or wanted:
                    req = Entity()
                    req.name = name
                    req.type = Entity.MODEL
                    ok = bridge._gz.request(bridge._svc_remove, req, Entity,
                                            Boolean, bridge._ack_ms)
                    print(f"     remove {name}: ack={ok}")
                bridge._terrain_models = []
                show("requested removal (NOT stepped)")
            elif cmd == "t":
                ok = bridge._set_pose(start[0], start[1], bridge.reset_z,
                                      start[2])
                show(f"teleport confirmed={ok}")
            elif cmd == "R":
                bridge.reset(start, terrain)
                show(f"full reset(); offset={bridge.reset_offset:.5f} "
                     f"missing={bridge.terrain_missing}")
            elif cmd == "w":
                l, r = float(parts[1]), float(parts[2])
                wheels = [l, l, r, r]
                print(f"     wheel command = {wheels}")
            elif cmd == "0":
                wheels = [0.0, 0.0, 0.0, 0.0]
                bridge._node.publish_wheels(wheels)
                print("     wheel command zeroed")
            elif cmd == "e":
                show("(no step)")
            elif cmd == "d":
                bridge._exec.spin_once(timeout_sec=0.02)
                print(f"     all entities: {sorted(bridge._entity_names)}")
            else:
                print("     ? unknown -- '?' for commands")
    finally:
        bridge.close()
        print("\nbridge closed (world left runnable)")


if __name__ == "__main__":
    main()
