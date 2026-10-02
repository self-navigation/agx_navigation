# Does TVLQR drift off the STACK's plans on bare ground too? (#34, 2026-10-02)
#
# In compare v7 our arm left clear plans by >0.2 m in 65/66 runs, on schedule,
# with a correct pose and no saturation. But v7 drove the plans the stack
# solved LIVE (STACK_PLANNER_PARAMS), which differ from the traj_data_v2
# library plans the gains were tuned and validated on (00105: 280 vs 186
# samples). This soaks both, same start/goal pairs, adopted gains, interleaved
# in one process (the control is carried, per CLAUDE.md), with traces.
#
#   live plans drift here too   -> the stack's plans are hard to track (tuning/plan dynamics)
#   only in the building        -> walls / amcl / ROS timing after all
#
# Live plans come from tools/replan_footprint.py --w-fp 0 (34 that solve);
# upload them to ~/live40/ and the paired list to ~/live_vs_lib.txt first.
set -uo pipefail

OUT=$HOME/soak_live_vs_lib.jsonl
PLANS=$(mktemp /tmp/live_vs_lib.XXXXXX)
sed "s|__HOME__|$HOME|" "$HOME/live_vs_lib.txt" >"$PLANS"

echo "[lvl] starting at $(date -Is); $(grep -c . "$PLANS") plans; out=$OUT"

tools/parallel_soak.sh \
    --out "$OUT" --plans "$PLANS" --trace-dir "$HOME/soak_live_vs_lib_traces" \
    --repeats 5 --workers 1 \
    -- 2.5,2.618
rc=$?

echo "[lvl] finished at $(date -Is) rc=$rc; rows $(grep -c . "$OUT" 2>/dev/null || echo 0)"
rm -f "$PLANS"
exit 0
