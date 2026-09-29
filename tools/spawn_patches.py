#!/usr/bin/env python3
"""Spawn the soak harness's along-path slip patches into a RUNNING fixture sim.

WHY. The full-stack comparison (tools/compare_run.py) drives the ROS stack --
vector_field -> pmp_planner -> corrector, or Nav2 -- rather than the RL env's
GazeboBridge, but it must measure on the SAME plant the soak numbers came from:
one episode of `along_path_terrain_sampler(nom.poses)` at seed 0, the patches
centred on randomly chosen vertices of the plan's own reference path. This
tool is the one-shot adapter: it samples exactly what `variance_probe.drive`
samples (same sampler, same seed) and creates the entities in the live world,
reusing the bridge's terrain code so the SDFs are byte-identical.

NEVER reset the world to change terrain (reset_world deletes runtime-spawned
entities, INCLUDING scout_mini -- CLAUDE.md, "reset_world DESTROYS THE ROBOT").
The two rules encoded here are the ones the bridge learned the hard way
(2026-08-02, and the CLAUDE.md stubs):

  * Stale patches are REMOVED first and the removal is waited out before any
    create: `create` on a name that still exists fails, and removals commit on
    the next physics tick -- which happens on its own here (the world is
    running, unlike the bridge's paused deterministic mode), but only if we
    wait for it.
  * Every create is issued ONCE and never re-issued on a missing name. A
    "missing" patch is in flight; re-creating genuinely fails. pose/info is
    the only honest answer to "is it there" -- the create ack is not consulted
    at all.

Usage (on the VM, inside the stack's partition):

    tools/with-worker 1 python3 tools/spawn_patches.py \
        --plan ~/traj_data_v2/floor_6_v2_00369.npz --seed 0

Exit 0 = every patch confirmed present in pose/info. 1 = failure (the caller
must not start a measured run: a rollout that begins before its ground exists
is not comparable with anything -- this exact silence caused the 0.22-vs-6.9 m
run-to-run spread that _apply_terrain was written to kill).
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import gz.transport13 as gz_transport
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.boolean_pb2 import Boolean

# Same sampler, same seed convention as tuning/variance_probe.drive.
from agx_planning.rl_corrector.terrain import along_path_terrain_sampler

# rl_ground covers the ground_friction_sampler case too; 8 matches the bridge's
# _STALE_PATCH_LIMIT so nothing a previous process could have left behind
# survives the sweep.
STALE_NAMES = ["rl_ground"] + [f"rl_patch_{i}" for i in range(8)]

ACK_MS = 800  # same best-effort contract as the bridge: not consulted, just sent


class PatchSpawner:
    def __init__(self, world: str):
        self.world = world
        self._svc_create = f"/world/{world}/create"
        self._svc_remove = f"/world/{world}/remove"
        self._topic_pose = f"/world/{world}/pose/info"
        self._gz = gz_transport.Node()
        self._entities: set[str] = set()
        self._seen_pose = False
        if not self._gz.subscribe(Pose_V, self._topic_pose, self._on_pose):
            raise RuntimeError(f"could not subscribe to {self._topic_pose} "
                               "-- is the sim up in this partition?")

    def _on_pose(self, msg: Pose_V) -> None:
        self._seen_pose = True
        self._entities = {p.name for p in msg.pose}

    def _spin_until(self, pred, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # pose/info is republished every physics tick, so a plain wall-clock
            # spin observes entity changes without any stepping from us.
            time.sleep(0.02)
            if pred():
                return True
        return pred()

    def present(self, names) -> bool:
        return not (set(names) - self._entities)

    def wait_pose_stream(self, timeout_s: float = 15.0) -> bool:
        return self._spin_until(lambda: self._seen_pose, timeout_s)

    def remove_and_wait(self, names, timeout_s: float = 10.0) -> bool:
        """Remove `names` that exist and wait until none of them remains.

        A remove of a missing model is a harmless negative ack (same contract
        as the bridge); the WAIT is the load-bearing half, because a create of
        a same-named entity issued before the removal commits would fail.
        """
        stale = [n for n in names if n in self._entities]
        for name in names:
            req = Entity()
            req.name = name
            req.type = Entity.MODEL
            self._gz.request(self._svc_remove, req, Entity, Boolean, ACK_MS)
        if not stale:
            return True
        return self._spin_until(lambda: self.present(stale) is False, timeout_s)

    def create_and_wait(self, patch: dict, name: str, timeout_s: float = 20.0) -> bool:
        """Create ONE patch, once, and wait for it to appear. Never re-issue."""
        from agx_planning.rl_corrector.terrain import patch_sdf

        req = EntityFactory()
        req.sdf = patch_sdf(patch, name)
        req.name = name
        req.pose.position.x = float(patch.get("x", 0.0))
        req.pose.position.y = float(patch.get("y", 0.0))
        req.pose.position.z = float(patch.get("z", 0.001))
        self._gz.request(self._svc_create, req, EntityFactory, Boolean, ACK_MS)
        return self._spin_until(lambda: name in self._entities, timeout_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", required=True, help="plan .npz (poses key, (N,3))")
    ap.add_argument("--seed", type=int, default=0,
                    help="terrain seed; MUST match the campaign (soak used 0)")
    ap.add_argument("--world", default="ordjo_world")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="wall seconds to wait for the pose stream / each patch")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    with np.load(args.plan) as f:
        poses = np.asarray(f["poses"], dtype=float)

    sp = PatchSpawner(args.world)
    out: dict = {"plan": args.plan, "seed": args.seed, "world": args.world,
                 "patches": [], "ok": False}

    try:
        if not sp.wait_pose_stream(args.timeout):
            out["error"] = (f"no pose/info on /world/{args.world}/pose/info "
                            f"within {args.timeout:g}s -- sim not up in this partition")
            return _emit(out, args.json, 1)

        if not sp.remove_and_wait(STALE_NAMES):
            out["error"] = "stale rl_* patches never disappeared; terrain state is inherited"
            return _emit(out, args.json, 1)

        patches = along_path_terrain_sampler(poses[:, :3])(np.random.default_rng(args.seed))
        for idx, patch in enumerate(patches):
            name = patch.get("name") or f"rl_patch_{idx}"
            if not sp.create_and_wait(patch, name):
                # Do NOT re-issue (see module docstring) -- record and fail.
                out["error"] = f"patch {name} never appeared in pose/info"
                out["patches"].append({"name": name, "profile": patch.get("profile"),
                                       "ok": False})
                return _emit(out, args.json, 1)
            out["patches"].append({
                "name": name, "profile": patch.get("profile"),
                "x": round(float(patch["x"]), 3), "y": round(float(patch["y"]), 3),
                "ok": True,
            })
        out["ok"] = True
        return _emit(out, args.json, 0)
    finally:
        # The patches must OUTLIVE this tool (the stack drives over them); only
        # the gz node is torn down.
        pass


def _emit(out: dict, as_json: bool, rc: int) -> int:
    if as_json:
        print(json.dumps(out))
    else:
        status = "OK" if out["ok"] else f"FAIL: {out.get('error')}"
        print(f"[spawn_patches] {status}")
        for p in out["patches"]:
            print(f"    {p['name']:<12} {p.get('profile', '?'):<14} "
                  f"({p.get('x', '?')}, {p.get('y', '?')}) {'ok' if p['ok'] else 'MISSING'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
