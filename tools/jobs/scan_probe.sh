#!/usr/bin/env bash
# amcl corridor-offset probe (2026-10-10): at each pose, bring the fixture up
# under localization:=amcl (no patches, no goal), then tools/scan_map_offset.py
# finds the rigid shift aligning the lidar scan (at the TRUE pose) with the
# baked walls and reads amcl's pose. Run ON THE VM. Out: $OUT_DIR/pose_*.json.
set -u
W=${WORKER:-13}
OUT_DIR=${OUT_DIR:-$HOME/run_data/2026-10-10_nojob_scan-map-offset}
mkdir -p "$OUT_DIR"
POSES=${POSES:-"-12,-1.7,0 -6,-1.7,0 2,-1.7,0 8,-1.7,0 14,-1.7,0 20,-1.7,0 26,-1.7,0 -6,-1.7,3.1416 14,-1.7,3.1416"}
for p in $POSES; do
    IFS=, read -r x y th <<<"$p"
    echo "[probe] $(date -Is) pose $x $y $th"
    bash tools/fixture_up.sh --worker "$W" --localization amcl --patches false --corrector tvlqr \
        --spawn "$x" "$y" "$th" >"$OUT_DIR/up_${x}_${y}_${th}.log" 2>&1
    echo "[probe] up rc=$?"
    ( set +u; source /opt/ros/jazzy/setup.bash; source install/setup.bash
      export GZ_PARTITION=agx$W ROS_DOMAIN_ID=$((40 + W))
      timeout 300 python3 tools/scan_map_offset.py --truth "$x" "$y" "$th" --n 20 \
          --out "$OUT_DIR/pose_${x}_${y}_${th}.json" ) >>"$OUT_DIR/probe.log" 2>&1
    echo "[probe] probe rc=$?"
    systemctl --user stop "agx-w$W.scope" >/dev/null 2>&1 || true
done
echo "[probe] finished at $(date -Is)"
