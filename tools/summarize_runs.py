#!/usr/bin/env python3
"""Aggregate `run_recorder` summaries into a paired identity-vs-corrector table.

Runs are paired by seed: `<corrector>_s<seed>_summary.txt`. Pairing matters
because `random_goals` draws the same goal for a given seed, so the only
variable between the two rows of a pair is the corrector itself.

Scores come from the recorder, which reads Gazebo ground truth -- never /odom.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from pathlib import Path

FIELDS = {
    "final_error": r"final error:\s*([-\d.naif]+)",
    "rms": r"cross-track rms:\s*([-\d.naif]+)",
    "max": r"cross-track max:\s*([-\d.naif]+)",
    "duration": r"duration:\s*([-\d.naif]+)",
}


def parse(path: Path) -> dict[str, float]:
    text = path.read_text()
    out = {}
    for name, pattern in FIELDS.items():
        m = re.search(pattern, text)
        # float("nan") parses happily, so check finiteness rather than trusting
        # that a successful parse means a usable number.
        out[name] = float(m.group(1)) if m else math.nan
    return out


def agg(values: list[float]) -> str:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return "     n/a"
    mean = statistics.mean(finite)
    sd = statistics.stdev(finite) if len(finite) > 1 else 0.0
    return f"{mean:.3f}±{sd:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", default="run_data", type=Path)
    ap.add_argument("--correctors", nargs="+", default=["identity", "tvlqr"])
    args = ap.parse_args()

    runs: dict[str, dict[str, dict[str, float]]] = {c: {} for c in args.correctors}
    for path in sorted(args.run_dir.glob("*_summary.txt")):
        m = re.match(r"(\w+?)_s(\d+)_summary\.txt", path.name)
        if not m or m.group(1) not in runs:
            continue
        runs[m.group(1)][m.group(2)] = parse(path)

    seeds = sorted(set().union(*(set(r) for r in runs.values())) if runs else [])
    if not seeds:
        raise SystemExit(f"no seeded runs found in {args.run_dir}")

    head = f"{'seed':>6}" + "".join(f"{c:>28}" for c in args.correctors)
    print(head)
    print(f"{'':>6}" + "".join(f"{'rms / max / final [m]':>28}" for _ in args.correctors))
    print("-" * len(head))
    for seed in seeds:
        row = f"{seed:>6}"
        for c in args.correctors:
            r = runs[c].get(seed)
            row += (
                f"{r['rms']:>10.3f}{r['max']:>9.3f}{r['final_error']:>9.3f}"
                if r
                else f"{'--':>28}"
            )
        print(row)
    print("-" * len(head))
    for metric in ("rms", "max", "final_error"):
        line = f"{metric:>6}"
        for c in args.correctors:
            line += f"{agg([r[metric] for r in runs[c].values()]):>28}"
        print(line)


if __name__ == "__main__":
    main()
