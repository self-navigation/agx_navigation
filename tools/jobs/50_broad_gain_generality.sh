#!/bin/bash
# Does the adopted gain pair generalise, or is it 7 plans' answer?
#
# WHY. Every gain claim we hold rests on `config/eval_trajectories.yaml`'s seven
# hand-picked plans, and 2026-08-14's library sweep already showed that set is
# ENRICHED FOR HARD PLANS (11 of 51 library plans are hard; most of our seven
# are). 2026-08-15 sharpened the worry: the U-turn `q_cross` basin that shaped
# much of the tuning story turned out to belong to ONE PLAN -- a near-duplicate
# route showed no basin at all (figures/2026-08-15/).
#
# So this re-runs the gain question on 40 plans from the CONSTRUCTED library,
# chosen by tools/select_broad_eval.py rather than by us, spread over four shape
# labels and 8.7-36.0 m of path length. None of the seven appear. Six gain pairs
# map the q plateau and its cliff, and test r=1.0 against the adopted 2.618 on
# ground that is not `floor_6_00031`.
#
# EVERY rollout is traced, so the whole run is scoreable in J as well as metres
# -- the two disagreed on 24 of 51 plans in the library sweep, and J is the
# quantity SVCM is actually stated in.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

OUT=$HOME/soak_broad_gains.jsonl
TRACES=$HOME/broad_gains_traces
PLANS=$HOME/broad_eval_plans.txt

cat >"$PLANS" <<'PLANEOF'
__HOME__/traj_data_v2/floor_6_v2_00369.npz
__HOME__/traj_data_v2/floor_6_v2_00482.npz
__HOME__/traj_data_v2/floor_6_v2_00147.npz
__HOME__/traj_data_v2/floor_6_v2_00105.npz
__HOME__/traj_data_v2/floor_6_v2_00208.npz
__HOME__/traj_data_v2/floor_6_v2_00486.npz
__HOME__/traj_data_v2/floor_6_v2_00096.npz
__HOME__/traj_data_v2/floor_6_v2_00249.npz
__HOME__/traj_data_v2/floor_6_v2_00302.npz
__HOME__/traj_data_v2/floor_6_v2_00443.npz
__HOME__/traj_data_v2/floor_6_v2_00440.npz
__HOME__/traj_data_v2/floor_6_v2_00108.npz
__HOME__/traj_data_v2/floor_6_v2_00350.npz
__HOME__/traj_data_v2/floor_6_v2_00081.npz
__HOME__/traj_data_v2/floor_6_v2_00428.npz
__HOME__/traj_data_v2/floor_6_v2_00102.npz
__HOME__/traj_data_v2/floor_6_v2_00320.npz
__HOME__/traj_data_v2/floor_6_v2_00419.npz
__HOME__/traj_data_v2/floor_6_v2_00402.npz
__HOME__/traj_data_v2/floor_6_v2_00239.npz
__HOME__/traj_data_v2/floor_6_v2_00342.npz
__HOME__/traj_data_v2/floor_6_v2_00291.npz
__HOME__/traj_data_v2/floor_6_v2_00420.npz
__HOME__/traj_data_v2/floor_6_v2_00201.npz
__HOME__/traj_data_v2/floor_6_v2_00152.npz
__HOME__/traj_data_v2/floor_6_v2_00047.npz
__HOME__/traj_data_v2/floor_6_v2_00184.npz
__HOME__/traj_data_v2/floor_6_v2_00074.npz
__HOME__/traj_data_v2/floor_6_v2_00355.npz
__HOME__/traj_data_v2/floor_6_v2_00001.npz
__HOME__/traj_data_v2/floor_6_v2_00115.npz
__HOME__/traj_data_v2/floor_6_v2_00052.npz
__HOME__/traj_data_v2/floor_6_v2_00150.npz
__HOME__/traj_data_v2/floor_6_v2_00457.npz
__HOME__/traj_data_v2/floor_6_v2_00034.npz
__HOME__/traj_data_v2/floor_6_v2_00219.npz
__HOME__/traj_data_v2/floor_6_v2_00366.npz
__HOME__/traj_data_v2/floor_6_v2_00403.npz
__HOME__/traj_data_v2/floor_6_v2_00388.npz
__HOME__/traj_data_v2/floor_6_v2_00061.npz
PLANEOF
sed -i "s|__HOME__|$HOME|" "$PLANS"

echo "[broad] starting at $(date -Is); $(wc -l <"$PLANS") plans; out=$OUT"

# 6 gain pairs x 40 plans = 240 rollouts per cycle; 3 cycles = 720, n=3 per cell.
# ~14 s a rollout on these (they are longer than the seven) => ~2.8 h. FINITE,
# because a queued job that does not end blocks everything behind it.
for batch in $(seq 1 3); do
    echo "[broad] cycle $batch/3 at $(date -Is)"
    python3 -m agx_planning.tuning.soak \
        --trajectories $(cat "$PLANS") \
        --gains 0.276,2.618 \
        --gains 0.276,1.0 \
        --gains 0.6,2.618 \
        --gains 1.5,2.618 \
        --gains 4.0,2.618 \
        --gains 10,0.25 \
        --trace-dir "$TRACES" --trace-every 1 \
        --max-rollouts 240 --out "$OUT" || break
    echo "[broad] cycle finished at $(date -Is); rows so far: $(wc -l <"$OUT")"
done

echo "[broad] done at $(date -Is); total rows $(wc -l <"$OUT")"
