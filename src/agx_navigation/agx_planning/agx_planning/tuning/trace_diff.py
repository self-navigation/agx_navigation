"""Find the FIRST step at which two supposedly-identical rollouts diverge.

WHY
---
Every reproducibility result so far has been an end-of-rollout scalar: two runs
scored 1.71 and 2.08, so something differed. That tells us nothing about WHAT or
WHEN -- by the final step, a micrometre of difference 200 steps earlier has been
amplified through contact, slip and feedback into metres. The interesting number
is the first row where the two worlds stop agreeing, and which column moves
first:

  * `terrain` differs at reset  -> the patches are not being placed identically
  * `cmd*` differs before the state does -> the CONTROLLER diverged, not physics
    (i.e. our own code is the non-determinism, not Gazebo)
  * state differs while cmd is still bit-identical -> genuine physics divergence
  * `world_steps`/`lost_steps` differ -> a step was dropped; not physics at all

Pure: reads CSVs, no ROS, no Gazebo. Unit-tested.

    python3 -m agx_planning.tuning.trace_diff a.csv b.csv [--eps 1e-9]
"""

import argparse
import csv
import math
from typing import Dict, List, Optional, Sequence, Tuple

# Columns compared as numbers; everything else (phase, terrain) as exact strings.
_TEXT_COLUMNS = ("phase", "terrain")

# Counters that accumulate over the LIFE OF THE PROCESS AND THE WORLD, not over
# the rollout. The second rollout in a process necessarily starts at a larger
# sim_time and world_steps than the first, and the world is deliberately not
# restarted between them -- so comparing them raw reports a 250-step "difference"
# on every run that is nothing but "this one happened later". Compared as deltas
# from each rollout's own first compared row, which is the quantity that carries
# meaning: did the two runs simulate the same AMOUNT of time.
_CUMULATIVE_COLUMNS = ("sim_time", "world_steps", "lost_steps",
                       "stale_pose_steps", "pose_stamp")

# The columns that describe where the robot actually is. Divergence in these is
# the phenomenon under study; imu_* and z are sensitive side-channels that move
# first, and cmd*/terrain answer "whose fault".
_POSE_COLUMNS = ("x", "y", "yaw")


def _relativize(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Rebase the cumulative counters onto each trace's own first row."""
    if not rows:
        return rows
    base = {c: _as_float(rows[0].get(c, "nan")) for c in _CUMULATIVE_COLUMNS}
    out = []
    for r in rows:
        r = dict(r)
        for c in _CUMULATIVE_COLUMNS:
            if c in r:
                r[c] = "%.9g" % (_as_float(r[c]) - base[c])
        out.append(r)
    return out


def load_trace(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _as_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _differs(col: str, a: str, b: str, eps: float) -> Optional[float]:
    """Return the magnitude of the difference, or None if they agree.

    NaN == NaN counts as agreement: NaN here means "this channel was not
    populated" (no IMU configured, no wheel speeds), which is a property of the
    configuration and identical in both runs -- not a divergence. A NaN on one
    side only IS a divergence, and is reported as inf.
    """
    if col in _TEXT_COLUMNS:
        return None if a == b else float("inf")
    fa, fb = _as_float(a), _as_float(b)
    if math.isnan(fa) and math.isnan(fb):
        return None
    if math.isnan(fa) or math.isnan(fb):
        return float("inf")
    d = abs(fa - fb)
    return None if d <= eps else d


def align_steps(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Just the rollout rows.

    Reset rows are deliberately NOT aligned positionally: the refine loop runs a
    variable number of times by design, so two clean runs legitimately have
    different reset row counts. Comparing those positionally would report a
    divergence that is only a different amount of polishing. Reset state is
    compared separately, at its last row (`reset_state`).
    """
    return [r for r in rows if r.get("phase") == "step"]


def reset_state(rows: Sequence[Dict[str, str]]) -> Optional[Dict[str, str]]:
    """The final pre-rollout snapshot: the true initial condition of the run."""
    resets = [r for r in rows if str(r.get("phase", "")).startswith("reset")]
    return resets[-1] if resets else None


def first_divergence(
    a: Sequence[Dict[str, str]], b: Sequence[Dict[str, str]], eps: float = 1e-9
) -> Optional[Tuple[int, Dict[str, float]]]:
    """(step index, {column: magnitude}) of the first differing rollout step."""
    for i, (ra, rb) in enumerate(zip(a, b)):
        diffs = {}
        for col in ra:
            if col == "row":
                continue  # a row counter, not state; offset by the reset length
            d = _differs(col, ra[col], rb.get(col, ""), eps)
            if d is not None:
                diffs[col] = d
        if diffs:
            return i, diffs
    return None


def compare(path_a: str, path_b: str, eps: float = 1e-9) -> Dict:
    ra, rb = load_trace(path_a), load_trace(path_b)
    sa, sb = _relativize(align_steps(ra)), _relativize(align_steps(rb))
    init_a, init_b = reset_state(ra), reset_state(rb)

    init_diffs: Dict[str, float] = {}
    if init_a and init_b:
        for col in init_a:
            # Cumulative counters are excluded outright here: at the reset row
            # they encode only "how much world has happened before this run".
            if col in ("row", "phase") or col in _CUMULATIVE_COLUMNS:
                continue
            d = _differs(col, init_a[col], init_b.get(col, ""), eps)
            if d is not None:
                init_diffs[col] = d

    return {
        "n_steps_a": len(sa),
        "n_steps_b": len(sb),
        "initial_state_diffs": init_diffs,
        "first_divergence": first_divergence(sa, sb, eps),
        "growth": growth_profile(sa, sb),
        "onsets": column_onsets(sa, sb, eps),
    }


def column_onsets(a: Sequence[Dict[str, str]], b: Sequence[Dict[str, str]],
                  eps: float = 1e-9) -> Dict[str, Optional[int]]:
    """First step each column differs at, per column.

    `first_divergence` says only which column moved first overall, and a noisy
    side-channel (the IMU) masks the ordering of everything else. The ORDER
    across columns is the diagnosis: wheel speeds moving before the pose means
    the commanded speed reached the plant differently -- our ROS publish racing
    the gz step -- while the pose moving first under identical wheel speeds
    means the solver itself.
    """
    onsets: Dict[str, Optional[int]] = {}
    for i, (ra, rb) in enumerate(zip(a, b)):
        for col in ra:
            if col == "row" or col in onsets:
                continue
            if _differs(col, ra[col], rb.get(col, ""), eps) is not None:
                onsets[col] = i
    for col in (a[0] if a else {}):
        onsets.setdefault(col, None)
    onsets.pop("row", None)
    return onsets


# Thresholds the pose separation is timestamped against, in metres/radians.
_GROWTH_LEVELS = (1e-9, 1e-6, 1e-3, 1e-2, 1e-1, 1.0)


def growth_profile(a: Sequence[Dict[str, str]],
                   b: Sequence[Dict[str, str]]) -> Dict:
    """When does the pose separation cross each order of magnitude?

    This is the question the first-divergence step cannot answer. Two rollouts
    that differ by 1e-16 at step 0 and 3 m at step 200 have a divergence RATE;
    if the separation grows exponentially from the floating-point floor, the
    system is chaotic and no amount of tightening the reset will help -- repeats
    are the only answer. If instead it sits at ~0 for 80 steps and then jumps,
    something discrete happened there (a patch edge, a contact mode switch) and
    that step is worth looking at directly.
    """
    crossings: Dict[float, Optional[int]] = {lv: None for lv in _GROWTH_LEVELS}
    seps: List[float] = []
    for i, (ra, rb) in enumerate(zip(a, b)):
        d = math.hypot(_as_float(ra.get("x", "nan")) - _as_float(rb.get("x", "nan")),
                       _as_float(ra.get("y", "nan")) - _as_float(rb.get("y", "nan")))
        seps.append(d)
        for lv in _GROWTH_LEVELS:
            if crossings[lv] is None and d > lv:
                crossings[lv] = i
    return {"crossings": crossings, "separation": seps,
            "final_separation": seps[-1] if seps else float("nan")}


def _fmt(diffs: Dict[str, float]) -> str:
    return ", ".join(
        "%s=%s" % (k, "MISMATCH" if math.isinf(v) else "%.3g" % v)
        for k, v in sorted(diffs.items(), key=lambda kv: -kv[1])
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_a")
    ap.add_argument("trace_b")
    ap.add_argument("--eps", type=float, default=1e-9)
    args = ap.parse_args()

    res = compare(args.trace_a, args.trace_b, args.eps)
    print("rollout steps: %d vs %d" % (res["n_steps_a"], res["n_steps_b"]))

    if res["initial_state_diffs"]:
        print("\nINITIAL STATE DIFFERS (post-reset, pre-rollout):")
        print("  " + _fmt(res["initial_state_diffs"]))
        print("  -> the runs did not start from the same world; divergence")
        print("     downstream of this is expected and proves nothing.")
    else:
        print("\ninitial state: IDENTICAL to %g" % args.eps)

    fd = res["first_divergence"]
    if fd is None:
        print("\nrollouts are IDENTICAL to %g over %d compared steps."
              % (args.eps, min(res["n_steps_a"], res["n_steps_b"])))
    else:
        i, diffs = fd
        print("\nfirst divergence at rollout step %d:" % i)
        print("  " + _fmt(diffs))
        cmd_moved = any(k.startswith("cmd") for k in diffs)
        state_moved = any(k in diffs for k in
                          ("x", "y", "z", "yaw", "v", "omega",
                           "qx", "qy", "qz", "qw", "w0", "w1", "w2", "w3"))
        if any(k in diffs for k in ("world_steps", "lost_steps")):
            print("  -> a STEP WAS DROPPED. Not a physics difference: one run")
            print("     simulated less time than the other.")
        elif diffs.get("terrain"):
            print("  -> the TERRAIN differs. Different plant, not divergence.")
        elif cmd_moved and not state_moved:
            print("  -> the COMMAND diverged first, with the state still")
            print("     identical: the non-determinism is in our controller.")
        elif state_moved and not cmd_moved:
            print("  -> the STATE diverged under an identical command:")
            print("     genuine physics/solver non-determinism.")

    print("\nfirst differing step, per column (earliest first):")
    ordered = sorted(res["onsets"].items(),
                     key=lambda kv: (kv[1] is None, kv[1]))
    for col, at in ordered:
        print("  %-18s %s" % (col, "never" if at is None else "step %d" % at))

    g = res["growth"]
    print("\nxy separation growth:")
    for lv in _GROWTH_LEVELS:
        at = g["crossings"][lv]
        print("  > %-8g  %s" % (lv, "step %d" % at if at is not None
                                else "never"))
    print("  final: %.4f m" % g["final_separation"])


if __name__ == "__main__":
    main()
