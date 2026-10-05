#!/bin/bash
# Full-stack comparison, ours vs Nav2 (dwb/mppi/rpp), on the 40 broad plans.
#
# WHY. The advisor's ask 1 (handover.md 2026-09-28: the ONLY priority) is the
# comparison against a published nav stack. Every number the paper has for the
# corrector comes from the RL-env soak harness, which drives the frozen plan
# directly -- it never exercises vector_field, the action plumbing, or goal
# handling, and it has no external baseline at all. This job runs each broad
# plan's start/goal pair through the FULL ROS stack once per arm: `ours`
# (vec-pmp offline + TVLQR) and Nav2 (SmacPlanner2D + controller in
# {dwb, mppi, rpp}), same baked map, same amcl localization, same spawn pose
# (the plan's start), same along-path slip terrain (seed 0 -- the soak plant),
# scored from Gazebo ground truth by tools/compare_run.py.
#
# ONE FRESH STACK PER CELL, by construction: compare_run.py calls
# tools/fixture_up.sh per plan, which tears the partition down first. This is
# not an optimization choice -- odometry is never reset and a second run on a
# live stack plans from a stale odom belief (CLAUDE.md, "freshly started
# fixture").
#
# HOW TO READ IT
#   - outcome / final_err / miss rate (final_err > 0.5 m) are the arrival
#     currencies; compare arms per-plan by paired sign test over the 40
#     (CLAUDE.md, "Reading a gain result"). `stack_terminal` is kept BESIDE
#     outcome because the nav2 result code and "arrived" can disagree by a
#     late abort -- read outcome (ground truth), not the stack's own verdict.
#   - control_energy is the integral of squared per-wheel accelerations from
#     the PUBLISHED /wheel_velocity_controller/commands, ZOH-resampled to a
#     common 0.1 s grid so the 10 Hz vec-pmp stream and the ~20 Hz nav2 chain
#     are the same estimator. It is NOT `J`: this harness has no epsilon
#     accumulator (nav2 has no correction to charge), so do not mix these
#     rows into a J table.
#   - max_cross_track is to the npz reference polyline; for nav2 arms the
#     planner routes differently, so this bounds "how differently", not
#     tracking error per se.
#   - amplification caveat: these are single runs per cell, no repeats. The
#     soak's mean-of-5 standard is unaffordable at full-stack cost; state
#     single-run status wherever a number from this job is quoted.
#
# COST. One cell = bring-up (~1-2 min, launch + amcl + lidar boot) + terrain
# (~5 s) + drive (up to 3x plan duration + 60 s of sim time). Smoke test
# (2026-09-29, first broad40 plan x 4 arms, one worker): measured wall times
# in ~/run_data/2026-09-28_job140_smoke/compare_smoke.jsonl and the job report. 160 cells split over 4 workers
# (one arm each) at the smoke-test worst case is an overnight run; use the
# measured numbers, not this paragraph, when queueing.
#
# RESUMABLE: each worker appends to its own JSONL and skips (plan, arm, seed)
# rows already there, so re-running the same command continues a killed batch.
# Nothing else may write to w<N>.jsonl while its worker is live.
#
# Run by tools/jobq.sh (ROS already sourced, cwd = repo), or detached:
#   tools/agx-run --detach 'bash tools/jobs/140_compare_broad40.sh'
# Override the split with ARMS/WORKERS (zipped, so counts must match):
#   ARMS="ours nav2-dwb" WORKERS="1 2" bash tools/jobs/140_compare_broad40.sh
set -uo pipefail

# compare_run.py needs rclpy in its OWN process (the drive phase is in-process
# rclpy; fixture_up.sh sources ROS for itself, but that does not help us). A VM
# login shell sources nothing (no ROS in ~/.profile), so a detached
# `agx-run --detach` launch dies at the first goal with
# `ModuleNotFoundError: No module named 'rclpy'` -- smoke, 2026-09-29, two
# wasted nav2-rpp cells. Source here, exactly as jobq.sh does; guarded, so the
# jobq path (already sourced) is unchanged.
if [ -z "${ROS_VERSION:-}" ]; then
    set +u
    source /opt/ros/jazzy/setup.bash
    source "$(pwd)/install/setup.bash"
    set -u
fi

OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-09-29_INVALID_job140-compare-v1/compare_broad40}
ARMS_=${ARMS:-"ours nav2-mppi nav2-rpp"}
WORKERS_=${WORKERS:-"1 2 3"}
SEED=${SEED:-0}

read -r -a ARMS <<<"$ARMS_"
read -r -a WORKERS <<<"$WORKERS_"
if [ "${#ARMS[@]}" -ne "${#WORKERS[@]}" ]; then
    echo "[cmp] ARMS and WORKERS must have the same length" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
PLANS=$(mktemp /tmp/broad_cmp_plans.XXXXXX)
sed "s|__HOME__|$HOME|" tools/jobs/broad40.txt | head -n "${N_PLANS:-1000}" >"$PLANS"

# Whatever happens, do not leave sims running on the workers: one gz sim per
# partition is the whole isolation mechanism, and an orphan here poisons the
# next job's partition (CLAUDE.md).
cleanup() {
    local w part
    for w in "${WORKERS[@]}"; do
        part=${w:+agx$w}
        part=${part:-default}
        bash tools/kill_stack.sh "$PWD" kill "$part" >/dev/null 2>&1
    done
}
trap cleanup EXIT

echo "[cmp] starting at $(date -Is); ${#ARMS[@]} arm(s) x $(grep -c . "$PLANS") plans; out=$OUT_DIR"

PIDS=()
for i in "${!ARMS[@]}"; do
    arm=${ARMS[$i]}
    w=${WORKERS[$i]}
    echo "[cmp] worker $w -> arm $arm, log $OUT_DIR/w$w.log"
    tools/with-worker "$w" python3 tools/compare_run.py \
        --plans "$PLANS" --arm "$arm" --worker "$w" --seed "$SEED" \
        --out "$OUT_DIR/rows.w$w.jsonl" --log-dir "$OUT_DIR" \
        >"$OUT_DIR/w$w.log" 2>&1 &
    PIDS+=($!)
done

rc=0
for p in "${PIDS[@]}"; do
    wait "$p" || rc=1
done

# Convenience merge for the summariser; per-worker files remain the source of
# truth for resuming (the merge is not consulted by --skip-existing).
cat "$OUT_DIR"/rows.w*.jsonl >"$OUT_DIR"/all_rows.jsonl 2>/dev/null

echo "[cmp] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT_DIR"/all_rows.jsonl 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit "$rc"
