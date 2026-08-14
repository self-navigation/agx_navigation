#!/bin/bash
# Is the q_cross basin a property of U-TURNS, or of floor_6_00031?
#
# This is the reason the v2 library was built. Every per-shape claim we have
# rests on one plan of that shape, and the U-turn's is the sharpest claim of the
# lot: 5906 rollouts (2026-08-13) put a basin at q in [0.276, 0.4] with
# near-vertical walls -- ~100% bad at 0.2 and at 0.5, ~1% bad inside. All of it
# is floor_6_00031. Nothing so far distinguishes "U-turns have a basin" from
# "this U-turn has a basin", and the sub-ladder's walls make that a live risk
# rather than a pedantic one.
#
# So: the same q rungs, on other U-turns drawn from the v2 library, WITH
# floor_6_00031 carried in the same run as a control. Carrying the control is
# the point -- it means a null result is readable as "the others differ from
# 00031" rather than "something changed on the box".
#
# Reading it:
#   basin reproduces on the new U-turns   -> it is a property of the SHAPE, and
#                                            the notch story stands as written;
#   basin appears only on 00031           -> the notch is one plan's, q=0.276 was
#                                            adopted partly on a coincidence, and
#                                            CLAUDE.md's U-turn sections need
#                                            rewriting.
# Either outcome is worth the night; the second is worth more.
#
# Depends on 20_generate_v2_library.sh having produced ~/traj_data_v2. If it did
# not, this exits non-zero and the runner moves on to the next job rather than
# stopping -- an empty library is a reason to skip, never to idle the box.
#
# Run by tools/jobq.sh, which has already sourced ROS and cd'd to the repo.
set -uo pipefail

LIB=$HOME/traj_data_v2
# The plan dir is the eval config's, not a guess: it is ~/pmp_trajectories_v2
# on the VM, which is easy to confuse with the new ~/traj_data_v2 library.
CONTROL=$(python3 -c "import yaml,os;c=yaml.safe_load(open('$PWD/config/eval_trajectories.yaml'));print(os.path.join(os.path.expanduser(c['trajectory_dir']),'floor_6_00031.npz'))")
OUT=$HOME/soak_uturn_generality.jsonl
TRACES=$HOME/uturn_generality_traces
PICK=4

if [[ ! -d $LIB ]]; then
    echo "[ugen] no $LIB -- job 20 did not produce a library; skipping"; exit 1
fi

# Pick by screen_score among plans the SOLVER labelled UTURN. The label comes
# from the solved plan, never from the screen's cheap route, which predicts
# turning at only +0.30 (2026-08-14) -- see tools/sample_eval_trajectories.py.
mapfile -t PLANS < <(python3 - "$LIB" "$PICK" <<'PY'
import glob, os, sys
import numpy as np
lib, pick = sys.argv[1], int(sys.argv[2])
found = []
for f in sorted(glob.glob(os.path.join(lib, "*.npz"))):
    try:
        d = np.load(f, allow_pickle=False)
        if str(d["shape"]) == "UTURN":
            found.append((float(d["screen_score"]), f))
    except Exception:                      # a truncated npz is not fatal here
        continue
found.sort(reverse=True)
for _, f in found[:pick]:
    print(f)
print(f"[ugen] {len(found)} UTURN plans in library", file=sys.stderr)
PY
)

if [[ ${#PLANS[@]} -lt 2 ]]; then
    echo "[ugen] only ${#PLANS[@]} UTURN plans in $LIB -- not enough to generalise; skipping"
    exit 1
fi
[[ -f $CONTROL ]] && PLANS+=("$CONTROL")

echo "[ugen] starting at $(date -Is) with ${#PLANS[@]} plans:"
printf '  %s\n' "${PLANS[@]}"

# 4 q rungs spanning the claimed basin and both walls, r held at the adopted
# 2.618. ~5 plans x 4 rungs = 20 cells; 6 batches of 200 = 1200 rollouts, n=60
# per cell, which resolves a mode FREQUENCY -- the quantity that actually
# differs between the rungs. ~2 h. FINITE, like every queued job.
for batch in $(seq 1 6); do
    echo "[ugen] batch $batch/6 at $(date -Is)"
    python3 -m agx_planning.tuning.soak \
        --trajectories "${PLANS[@]}" \
        --gains 0.2,2.618 \
        --gains 0.276,2.618 \
        --gains 0.4,2.618 \
        --gains 0.5,2.618 \
        --trace-dir "$TRACES" --trace-every 4 \
        --max-rollouts 200 --out "$OUT" || break
    echo "[ugen] batch finished at $(date -Is); rows so far: $(wc -l <"$OUT")"
done

echo "[ugen] done at $(date -Is); total rows $(wc -l <"$OUT")"
