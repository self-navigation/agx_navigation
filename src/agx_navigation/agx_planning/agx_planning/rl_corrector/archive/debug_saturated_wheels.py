"""One-off: send a FIXED, large per-side wheel differential straight to
GazeboBridge (bypassing env/policy entirely) and compare measured v/omega
against what RLCorrectorConfig.wheels_to_body predicts, at increasing
differential magnitude. Isolates whether large residual authority causes a
genuine kinematic-model-vs-real-physics mismatch, independent of RL/reward.

python3 -m agx_planning.rl_corrector.debug_saturated_wheels \\
    --world rl_corrector --model scout_mini
"""

import argparse

from .config import RLCorrectorConfig
from .gazebo_bridge import GazeboBridge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="rl_corrector")
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    cfg = RLCorrectorConfig(use_costates=False)
    bridge = GazeboBridge(cfg, world_name=args.world, model_name=args.model,
                          deterministic=True)

    # Baseline nominal magnitude (~mean |w| from the recorded trajectories) plus
    # increasing symmetric differentials, up to the full +/-wheel_residual_max
    # on each side (worst case: left gets +max, right gets -max).
    base = 8.0
    diffs = [0.0, 1.0, 2.0, 4.0, cfg.wheel_residual_max, 2 * cfg.wheel_residual_max]

    try:
        for d in diffs:
            wl, wr = base + d, base - d
            v_pred, w_pred = cfg.wheels_to_body(wl, wr)
            bridge.reset((0.0, 0.0, 0.0))
            st = None
            for _ in range(args.steps):
                st = bridge.step([wl, wl, wr, wr], cfg.control_dt)
            print(f"diff={d:5.2f}  wl={wl:6.2f} wr={wr:6.2f}  "
                  f"predicted: v={v_pred:+.3f} w={w_pred:+.3f}  "
                  f"measured: v={st.v:+.3f} w={st.omega:+.3f}  "
                  f"(w ratio meas/pred = {st.omega / w_pred if abs(w_pred) > 1e-6 else float('nan'):.2f})")
    finally:
        bridge.close()


if __name__ == "__main__":
    main()
