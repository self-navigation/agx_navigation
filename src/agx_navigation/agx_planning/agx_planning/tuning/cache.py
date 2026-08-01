"""Persist every objective evaluation, so a tuning run is resumable.

WHY THIS EXISTS
---------------
One evaluation is ~75 s of Gazebo (three trajectories driven to completion), so
a 60-evaluation search is over an hour, and the VM running it has been stopped
mid-run before. Losing that to a dropped ssh session or a reboot is the same
mistake the RL trainer already solved with frequent checkpoints.

HOW RESUME WORKS
----------------
Nelder-Mead is deterministic given its objective, so the search does not need
its simplex serialised: re-running it against a *memoized* objective replays the
identical sequence of moves for free, and the first parameter set the cache has
never seen is exactly where the previous run stopped. Resume therefore costs
zero Gazebo time and cannot drift from a fresh run.

The consequence to be aware of: a replayed evaluation returns the value measured
at the time, not a fresh measurement. That is deliberate -- the search must see
one consistent objective, and the Gazebo objective is not perfectly repeatable
(see the offline-mode variance note in CLAUDE.md). It also means a cache file
from a DIFFERENT plant (different trajectories, terrain seed, or corrector code)
is poison; `key` records what the run was, and `load` refuses a mismatch rather
than silently resuming onto a different problem.

Pure module: json + math only. No ROS, no Gazebo.
"""

import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple


def quantize(x: Sequence[float], places: int = 9) -> str:
    """Cache key for a parameter vector.

    Rounded before hashing because the simplex recomputes centroids in floating
    point: the *same* logical point can differ in the last bits between a run
    and its replay, and an exact-match cache would then miss and spend 75 s
    re-measuring a point it already had.
    """
    return json.dumps([round(float(v), places) for v in x])


class EvalCache:
    """Append-only JSONL record of (x, fx) plus arbitrary per-evaluation detail."""

    def __init__(self, path: str, key: Optional[Dict] = None):
        self.path = path
        self.key = key or {}
        self.entries: Dict[str, float] = {}
        self.records: List[Dict] = []

    # --- persistence -----------------------------------------------------
    def load(self, strict: bool = True) -> int:
        """Read an existing cache. Returns how many evaluations were recovered."""
        if not os.path.isfile(self.path):
            return 0
        recovered = 0
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a torn final line. Losing one
                    # evaluation is fine; refusing to resume over it is not.
                    continue
                if rec.get("_meta"):
                    if strict and self.key and rec["_meta"] != self.key:
                        raise ValueError(
                            f"cache {self.path} was written for a different setup:\n"
                            f"  cached: {rec['_meta']}\n  current: {self.key}\n"
                            "Delete it or pass a different --cache to start fresh.")
                    continue
                if "x" not in rec or "fx" not in rec:
                    continue
                self.records.append(rec)
                fx = float(rec["fx"])
                # Skip failures on the way back in too, for cache files written
                # before failures stopped being memoized -- and because a
                # transient sim failure must never become permanent.
                if rec.get("_failed") or not math.isfinite(fx):
                    continue
                self.entries[quantize(rec["x"])] = fx
                recovered += 1
        return recovered

    def _append(self, rec: Dict) -> None:
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
            # Flushed and fsynced per line: the whole point is surviving a kill,
            # and a buffered write loses exactly the evaluations that cost most.
            fh.flush()
            os.fsync(fh.fileno())

    def write_header(self) -> None:
        if not os.path.isfile(self.path) or os.path.getsize(self.path) == 0:
            self._append({"_meta": self.key})

    # --- use -------------------------------------------------------------
    def wrap(self, f: Callable[[Sequence[float]], float],
             detail: Optional[Callable[[Sequence[float]], Dict]] = None,
             on_hit: Optional[Callable[[Sequence[float], float], None]] = None,
             ) -> Callable[[Sequence[float]], float]:
        """Memoize `f`, recording every fresh evaluation to disk immediately."""

        def wrapped(x):
            k = quantize(x)
            if k in self.entries:
                if on_hit is not None:
                    on_hit(x, self.entries[k])
                return self.entries[k]
            fx = float(f(x))
            rec = {"x": [float(v) for v in x], "fx": fx}
            if detail is not None:
                rec.update(detail(x))
            # FAILURES ARE NEVER CACHED. An `inf` here means "this evaluation did
            # not complete", which is usually a statement about the sim, not
            # about the gains: killing the tuner mid-evaluation once invalidated
            # the bridge's rclpy context, after which every rollout failed in
            # 2 ms and 56 bogus `inf`s were written in three seconds. Memoized,
            # those would replay as real measurements on every future resume and
            # steer the search permanently. Recorded for diagnosis, not returned
            # from the cache.
            if math.isfinite(fx):
                self.entries[k] = fx
            else:
                rec["_failed"] = True
            self.records.append(rec)
            self._append(rec)
            return fx

        return wrapped

    def best(self) -> Optional[Tuple[List[float], float]]:
        if not self.records:
            return None
        rec = min(self.records, key=lambda r: r["fx"])
        return rec["x"], rec["fx"]
