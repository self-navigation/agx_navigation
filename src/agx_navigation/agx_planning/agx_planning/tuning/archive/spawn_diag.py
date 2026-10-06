"""Why does /world/<w>/create reject the terrain patches?

The sim console showed `create` returning ack=False and the patch never
appearing, no matter how many ticks are stepped -- so this is a REJECTED request,
not a commit-latency race. This isolates the variables one at a time:

  * paused vs running world   (does create need the world to be advancing?)
  * short vs long ack timeout (is 800 ms simply too short?)
  * the real patch SDF vs a minimal box (is the generated SDF malformed?)

Prints the ack and whether the entity subsequently shows up in pose/info.
"""

import argparse
import time

import numpy as np

from ..rl_corrector.config import RLCorrectorConfig
from ..rl_corrector.nominal import load_recorded

MINIMAL_SDF = """<?xml version="1.0" ?>
<sdf version="1.8">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <collision name="c">
        <geometry><box><size>1 1 0.01</size></box></geometry>
      </collision>
      <visual name="v">
        <geometry><box><size>1 1 0.01</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>"""


def try_create(bridge, name, sdf, ack_ms, x=0.0, y=0.0, z=0.05):
    from gz.msgs10.entity_factory_pb2 import EntityFactory
    from gz.msgs10.boolean_pb2 import Boolean
    req = EntityFactory()
    req.sdf = sdf
    req.name = name
    req.pose.position.x = float(x)
    req.pose.position.y = float(y)
    req.pose.position.z = float(z)
    t0 = time.monotonic()
    ack = bridge._gz.request(bridge._svc_create, req, EntityFactory, Boolean, ack_ms)
    dt = time.monotonic() - t0
    # Step a few ticks and spin, so a slow-but-successful create still shows up.
    for _ in range(5):
        bridge._world_control(multi_step=1, ack_ms=bridge._step_ack_ms)
    for _ in range(20):
        bridge._exec.spin_once(timeout_sec=0.02)
    present = name in bridge._entity_names
    print(f"    ack={ack} in {dt*1000:.0f} ms   present={present}")
    return present


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
    from ..rl_corrector.gazebo_bridge import GazeboBridge
    from ..rl_corrector.terrain import along_path_terrain_sampler, patch_sdf

    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)
    terrain = along_path_terrain_sampler(nom.poses)(np.random.default_rng(args.seed))
    patch = terrain[0]
    real_sdf = patch_sdf(patch, "diag_real")

    print(f"\ncreate service: {bridge._svc_create}")
    print(f"ack_ms={bridge._ack_ms}\n")
    print("--- generated patch SDF (first 400 chars) ---")
    print(real_sdf[:400])
    print("---\n")

    print("1. minimal box, paused world, default ack:")
    try_create(bridge, "diag_box1", MINIMAL_SDF.format(name="diag_box1"),
               bridge._ack_ms, x=patch["x"], y=patch["y"])

    print("2. minimal box, paused world, 5 s ack:")
    try_create(bridge, "diag_box2", MINIMAL_SDF.format(name="diag_box2"),
               5000, x=patch["x"], y=patch["y"])

    print("3. real patch SDF, paused world, 5 s ack:")
    try_create(bridge, "diag_real", real_sdf, 5000,
               x=patch["x"], y=patch["y"], z=patch.get("z", 0.001))

    print("4. real patch SDF, RUNNING world, 5 s ack:")
    bridge._world_control(pause=False)
    time.sleep(0.5)
    try_create(bridge, "diag_real_run", patch_sdf(patch, "diag_real_run"), 5000,
               x=patch["x"], y=patch["y"], z=patch.get("z", 0.001))
    bridge._world_control(pause=True)

    print("\nentities now:",
          sorted(n for n in bridge._entity_names if "diag" in n or "rl_" in n))

    # Clean up whatever landed, so the world is left as we found it.
    from gz.msgs10.entity_pb2 import Entity
    from gz.msgs10.boolean_pb2 import Boolean
    for name in ("diag_box1", "diag_box2", "diag_real", "diag_real_run"):
        req = Entity()
        req.name = name
        req.type = Entity.MODEL
        bridge._gz.request(bridge._svc_remove, req, Entity, Boolean, bridge._ack_ms)
    for _ in range(5):
        bridge._world_control(multi_step=1, ack_ms=bridge._step_ack_ms)
    bridge.close()


if __name__ == "__main__":
    main()
