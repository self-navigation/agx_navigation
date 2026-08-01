"""Persist every objective evaluation so a tuning run can be resumed.

Nelder-Mead is deterministic given its objective, so re-running the search
against the memoized objective replays the same moves at no cost and resumes
at the first unseen point. Replayed evaluations return the stored value, not
a fresh measurement. key records the run configuration and load refuses a
cache from a different one. Pure module: json and math only.
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
