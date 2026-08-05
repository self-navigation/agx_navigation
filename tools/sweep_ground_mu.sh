#!/usr/bin/env bash
# Measure slip_chi as a function of the ground plane's friction coefficient.
#
# WHY THIS EXISTS (2026-08-05)
# ----------------------------
# Dropping the world ground from mu=1.0 to mu=0.45 made the PMP plans
# unfollowable: max|e_cross| on a corner went from ~1.2 m to 27 m, for BOTH
# identity and TVLQR, with no terrain patches involved. slip_ident then measured
# chi = 18.69 at mu=0.45 (yaw gain 0.054, i.e. 5% of commanded yaw rate, and 7 of
# 8 arcs unmeasurable) against chi = 1.3727 at mu=1.0.
#
# The hypothesis: each wheel carries mu1=200.0 (rolling) / mu2=0.7 (lateral) --
# a 285:1 anisotropy which IS the skid-steer's steering mechanism, since it yaws
# by gripping longitudinally while scrubbing sideways. Gazebo combines two
# contacting surfaces by taking the SMALLER coefficient, so an ISOTROPIC ground
# below 0.7 caps both directions equally and collapses the ratio to 1:1 --
# deleting the mechanism rather than merely making the floor slippery.
#
# THE FALSIFIABLE PREDICTION this sweep tests: lateral is capped by the wheel's
# own 0.7 for any ground mu >= 0.7, so chi should sit flat near 1.37 down to 0.7
# and rise sharply below it. A KNEE AT 0.7. If instead chi degrades smoothly from
# 1.0, the min-combination model is wrong and the usable range is different.
#
# WHY IT MATTERS BEYOND THIS BUG: patches are ground entities, so a patch can
# only ever be isotropic. Every profile in surface_patches.py sits below 0.7
# (linoleum 0.45, wet_tile 0.30, slippery 0.20, icy 0.05), which means every slip
# patch ever driven over has been collapsing the anisotropy to 1:1. The curve
# this script produces bounds what a patch can physically express, and therefore
# whether surface realism has to move into the WHEEL's mu1/mu2 pair instead.
#
# Uses the REAL-TIME world: slip_ident integrates the gyro on message timestamps
# and cannot keep up with the uncapped world's ~3 kHz IMU (it drops ~800 of every
# 801 samples and correctly refuses to report a chi).
#
#   ./tools/sweep_ground_mu.sh [out.csv]
#
# ~2 min per point (sim restart + 8 arcs at 1x realtime). Leaves the world file
# at the LAST value swept -- reset it deliberately when done.

set -uo pipefail
cd "$(dirname "$0")/.."

WORLD=src/rudn-ordjo-building/worlds/rl_corrector_rt.world
OUT="${1:-sweep_data/ground_mu_chi.csv}"
MUS="${MUS:-1.0 0.9 0.8 0.7 0.6 0.5 0.45}"

mkdir -p "$(dirname "$OUT")" sweep_data/logs
echo "ground_mu,mean_chi,mean_yaw_gain,usable_arcs,spread" > "$OUT"

for mu in $MUS; do
    echo "=== ground mu = $mu ==============================================="

    # The ground plane is the only <mu>/<mu2> pair in this file.
    sed -i -E "s|<mu>[0-9.]+</mu>|<mu>${mu}</mu>|; s|<mu2>[0-9.]+</mu2>|<mu2>${mu}</mu2>|" "$WORLD"

    just kill-sim >/dev/null 2>&1
    # remote-sim syncs and builds; the world is installed by a glob in setup.py,
    # and `make build` is stamp-gated on src/, so touching the file forces it.
    just remote-sim rl_corrector_rt.world >/dev/null 2>&1

    # Wait for the sim rather than sleeping a guessed interval.
    ready=$(ssh -F ssh_config agx 'source /opt/ros/jazzy/setup.bash && source ~/agx_navigation/install/setup.bash && \
        for i in $(seq 1 30); do timeout 6 ros2 topic list 2>/dev/null | grep -q "imu/data" && { echo ready; break; }; sleep 2; done')
    if [ "$ready" != "ready" ]; then
        echo "  [!] sim never came up at mu=$mu -- skipping"
        echo "$mu,,,0," >> "$OUT"
        continue
    fi

    log=$(tools/agx-run --detach 'cd ~/agx_navigation && source /opt/ros/jazzy/setup.bash && source install/setup.bash && \
        ros2 run agx_planning slip_ident --ros-args -p use_sim_time:=true \
        -p cmd_mode:=wheels -p imu_topic:=/imu/data -p odom_topic:=/odom' \
        | grep -oE '/tmp/agx-run-[0-9-]+\.log' | head -1)

    # Poll the LOG, not the process table: `pgrep -f slip_ident` matches the
    # polling command's own command line and never terminates.
    ssh -F ssh_config agx "for i in \$(seq 1 90); do \
        grep -q 'SUMMARY\|Aborting' $log 2>/dev/null && break; sleep 4; done"

    ssh -F ssh_config agx "cat $log" > "sweep_data/logs/mu_${mu}.log" 2>/dev/null

    chi=$(grep -oE 'mean chi   = [0-9.]+' "sweep_data/logs/mu_${mu}.log" | grep -oE '[0-9.]+$')
    gain=$(grep -oE 'mean gain  = [0-9.]+' "sweep_data/logs/mu_${mu}.log" | grep -oE '[0-9.]+$')
    spread=$(grep -oE 'spread [0-9.]+ across radii' "sweep_data/logs/mu_${mu}.log" | grep -oE '[0-9.]+')
    arcs=$(grep -cE '^  (arc|spin)_[a-z_]+ +r=' "sweep_data/logs/mu_${mu}.log")

    echo "$mu,${chi:-},${gain:-},${arcs:-0},${spread:-}" >> "$OUT"
    echo "  chi=${chi:-FAILED}  yaw_gain=${gain:-}  usable_arcs=${arcs:-0}"
done

echo
echo "=== done -> $OUT ==="
cat "$OUT"
echo
echo "NOTE: $WORLD is left at mu=$(echo $MUS | awk '{print $NF}') -- set it deliberately."
