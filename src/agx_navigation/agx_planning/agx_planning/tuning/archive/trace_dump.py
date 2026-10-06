"""Print selected rows of a rollout trace, for eyeballing what a run did.

trace_diff answers "where do two runs differ"; this answers "what did this run
actually do", which is the question when a run is bad in ISOLATION rather than
merely different from another.

    python3 -m agx_planning.tuning.trace_dump run.csv [--rows 0,1,2,10,100,-1]
"""

import argparse

from . import trace_diff as td

_FIELDS = ("x", "y", "yaw", "v", "omega", "cmd0", "cmd2",
           "sim_time", "pose_stamp", "stale_pose_steps", "lost_steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--rows", default="0,1,2,3,5,10,30,100,-1")
    args = ap.parse_args()

    rows = td.align_steps(td.load_trace(args.trace))
    print("%s: %d rollout steps" % (args.trace, len(rows)))
    for spec in args.rows.split(","):
        i = int(spec)
        if not rows or i >= len(rows):
            continue
        r = rows[i]
        parts = []
        for f in _FIELDS:
            v = r.get(f)
            if v is None:
                continue
            try:
                parts.append("%s=%+.4f" % (f, float(v)))
            except ValueError:
                parts.append("%s=%s" % (f, v))
        print("  %5d %s" % (i, " ".join(parts)))


if __name__ == "__main__":
    main()
