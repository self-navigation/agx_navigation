"""One-off: run the identity policy on ONE recorded trajectory under GazeboBridge
with corridor/heading termination disabled, log both the planned nominal path and
the actual ground-truth pose at every step, and save a comparison plot. Answers
"is a large e_cross a real physical drift, or a bug in the cross-track/arclength
projection for a path that curves back near itself?"

python3 -m agx_planning.rl_corrector.debug_plot_identity \\
    --traj ~/pmp_trajectories_v2/floor_6_00013.npz --out /tmp/identity_track.png
"""

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RLCorrectorConfig
from .env import WheelCorrectorEnv
from .nominal import load_recorded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False, corridor_epsilon=1e9, max_heading_err=1e9)
    nom = load_recorded(args.traj)

    from .gazebo_bridge import GazeboBridge
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)

    def sampler(rng):
        return nom

    env = WheelCorrectorEnv(cfg, bridge, nominal_sampler=sampler,
                            start_offset=(0.0, 0.0, 0.0), seed=0)

    actual_xy = []
    e_cross_log = []
    try:
        env.reset(seed=0)
        actual_xy.append((nom.poses[0, 0], nom.poses[0, 1]))
        done = False
        while not done:
            _o, _r, terminated, truncated, info = env.step(
                np.zeros(cfg.action_dim, dtype=np.float32))
            actual_xy.append((env.bridge._pose_xyth[0], env.bridge._pose_xyth[1]))
            e_cross_log.append(info["e_cross"])
            done = terminated or truncated
    finally:
        env.close()

    actual_xy = np.array(actual_xy, dtype=float)
    planned_xy = nom.poses[:, :2]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    ax1.plot(planned_xy[:, 0], planned_xy[:, 1], "b-", lw=2, label="planned (nominal)")
    ax1.plot(actual_xy[:, 0], actual_xy[:, 1], "r-", lw=1.5, label="actual (ground truth)")
    ax1.plot(*planned_xy[0], "go", ms=10, label="start")
    ax1.plot(*planned_xy[-1], "kx", ms=12, mew=2, label="planned goal")
    ax1.set_aspect("equal")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"path: {args.traj.split('/')[-1]}")

    ax2.plot(e_cross_log)
    ax2.axhline(0.5, color="orange", ls="--", label="old corridor_epsilon=0.5m")
    ax2.set_xlabel("step")
    ax2.set_ylabel("e_cross [m]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("reported cross-track error over time")

    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")
    print(f"planned path length: {len(planned_xy)} points, "
          f"bbox x=[{planned_xy[:,0].min():.2f},{planned_xy[:,0].max():.2f}] "
          f"y=[{planned_xy[:,1].min():.2f},{planned_xy[:,1].max():.2f}]")
    print(f"actual path bbox: x=[{np.nanmin(actual_xy[:,0]):.2f},{np.nanmax(actual_xy[:,0]):.2f}] "
          f"y=[{np.nanmin(actual_xy[:,1]):.2f},{np.nanmax(actual_xy[:,1]):.2f}]")


if __name__ == "__main__":
    main()
