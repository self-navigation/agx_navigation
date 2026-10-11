#!/usr/bin/env python3
"""Spawn the along-path slip patches into a running fixture sim.

Samples the same terrain as variance_probe.drive (along_path_terrain_sampler
on the plan's poses at the given seed) and creates the entities with the
bridge's terrain code. Stale patches are removed and the removal waited out
first; each create is issued once and presence is confirmed through
pose/info, never by re-issuing. Never reset the world to change terrain: it
deletes the robot too. Exits 0 when every patch is present, 1 otherwise.

    tools/with-worker 1 python3 tools/spawn_patches.py --plan ~/traj_data_v2/floor_6_v2_00369.npz --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
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
STALE_NAMES = ["rl_ground"] + [f"rl_patch_{i}" for i in range(8)] + [f"door_leaf_{i}" for i in range(16)]

# Door leaves (tools/door_leaves.py): a plan npz with a `leaves` key carries ajar
# leaves that are also drawn into its own map copy. Each is a static box along
# hinge->tip, LEAF_T thick and LEAF_H tall, VISUAL always (gpu_lidar renders
# visuals) and collision only without --phantom, matching the walls.
LEAF_T, LEAF_H = 0.05, 2.0


def leaf_sdf(name: str, length: float, collide: bool) -> str:
    geom = f"<geometry><box><size>{length:.3f} {LEAF_T} {LEAF_H}</size></box></geometry>"
    col = f"<collision name='c'>{geom}</collision>" if collide else ""
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<link name='leaf'>{col}<visual name='v'>{geom}<material><ambient>0.55 0.4 0.25 1</ambient>"
            f"<diffuse>0.55 0.4 0.25 1</diffuse></material></visual></link></model></sdf>")

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

    def create_leaf_and_wait(self, leaf: dict, name: str, collide: bool, timeout_s: float = 20.0) -> bool:
        """Create ONE door leaf, once, and wait for it to appear. Never re-issue."""
        h, t = np.asarray(leaf["hinge"], float), np.asarray(leaf["tip"], float)
        c, yaw = (h + t) / 2, math.atan2(t[1] - h[1], t[0] - h[0])
        req = EntityFactory()
        req.sdf = leaf_sdf(name, float(np.linalg.norm(t - h)), collide)
        req.name = name
        req.pose.position.x, req.pose.position.y, req.pose.position.z = float(c[0]), float(c[1]), LEAF_H / 2
        req.pose.orientation.z, req.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
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
    ap.add_argument("--phantom", action="store_true", help="door leaves without collision (phantom walls)")
    ap.add_argument("--no-patches", action="store_true", help="door leaves only, no slip patches")
    args = ap.parse_args()

    with np.load(args.plan) as f:
        poses = np.asarray(f["poses"], dtype=float)
        leaves = json.loads(str(f["leaves"])) if "leaves" in f else []

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

        patches = ([] if args.no_patches else
                   along_path_terrain_sampler(poses[:, :3])(np.random.default_rng(args.seed)))
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
        for idx, leaf in enumerate(leaves):
            name = f"door_leaf_{idx}"
            ok = sp.create_leaf_and_wait(leaf, name, collide=not args.phantom)
            out["patches"].append({"name": name, "profile": "door_leaf", "x": leaf["hinge"][0],
                                   "y": leaf["hinge"][1], "ok": ok})
            if not ok:
                out["error"] = f"leaf {name} never appeared in pose/info"
                return _emit(out, args.json, 1)
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
