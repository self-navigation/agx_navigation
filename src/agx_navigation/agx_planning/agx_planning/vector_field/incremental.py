"""Replay a complete FMM field from a checked native checkpoint."""

import operator

import numpy as np
import skfmm


class IncrementalEikonal:
    """Keep one tracked march and a bounded set of checkpoints."""

    def __init__(self, checkpoint_interval=10000, max_checkpoints=4,
                 cache_bytes=256 * 1024**2):
        self.interval = operator.index(checkpoint_interval)
        self.max_checkpoints = operator.index(max_checkpoints)
        self.cache_bytes = operator.index(cache_bytes)
        if self.interval <= 0 or self.max_checkpoints < 0 or self.cache_bytes < 0:
            raise ValueError("checkpoint interval must be positive; count and bytes must be nonnegative")
        self.clear()

    def clear(self):
        self._state = None
        self._key = None
        self._checkpoints = []
        self.last_stats = {}

    def _full(self, obstacles, speed, goal_col, goal_row, resolution, target, margin, reason):
        from .field import solve_eikonal

        self.clear()
        result = solve_eikonal(obstacles, speed, goal_col, goal_row, resolution,
                               early_exit_target=target, early_exit_margin_cells=margin)
        self.last_stats = dict(status="full_recompute", reason=reason, changed_cells=None,
                               reused_batches=0, first_affected_batch=None,
                               total_batches=None, computed_batches=None,
                               checkpoint_count=0, cache_array_bytes=0)
        return result

    def _finish(self):
        stored_bytes = 0
        while len(self._checkpoints) < self.max_checkpoints:
            stats = self._state.stats
            size = stats["checkpoint_array_bytes"]
            if stats["working_array_bytes"] + stored_bytes + size > self.cache_bytes:
                break
            self._checkpoints.append((self._state.checkpoint(max_bytes=size), size))
            stored_bytes += size
            if stats["complete"] or len(self._checkpoints) == self.max_checkpoints:
                break
            self._state.advance(self.interval)
        self._state.advance()

    def solve(self, obstacles, speed, goal_col, goal_row, resolution,
              early_exit_target=None, early_exit_margin_cells=0):
        try:
            return self._solve(obstacles, speed, goal_col, goal_row, resolution,
                               early_exit_target, early_exit_margin_cells)
        except Exception:
            self.clear()
            raise

    def _solve(self, obstacles, speed, goal_col, goal_row, resolution, target, margin):
        if target is not None:
            return self._full(obstacles, speed, goal_col, goal_row, resolution,
                              target, margin, "early_exit")
        # A tracked state needs at least 64 bytes per cell, plus 20 for a checkpoint.
        if not self.max_checkpoints or obstacles.size * 84 > self.cache_bytes:
            return self._full(obstacles, speed, goal_col, goal_row, resolution,
                              target, margin, "cache_budget")

        key = (obstacles.shape, goal_col, goal_row, resolution)
        info = dict(status="full_required", reason="new_field", changed_cells=0,
                    reused_batches=0, first_affected_batch=-1)
        if self._state is not None and key == self._key:
            info = self._state.replay(speed, obstacles, [c for c, _ in self._checkpoints])
        if info["status"] == "unchanged":
            pass
        else:
            self._checkpoints.clear()
            if info["status"] == "full_required":
                self._state = None
                phi = np.ones(obstacles.shape, dtype=np.float64)
                phi[goal_row, goal_col] = -1.0
                self._state = skfmm.TravelTimeState(
                    np.ma.array(phi, mask=obstacles), speed, dx=resolution,
                    track_dependencies=True,
                )
                info["status"] = "full_recompute"
                self._key = key
                initial = self._state.stats
                if initial["working_array_bytes"] + initial["checkpoint_array_bytes"] > self.cache_bytes:
                    return self._full(obstacles, speed, goal_col, goal_row, resolution,
                                      target, margin, "cache_budget")
            self._finish()

        stats = self._state.stats
        self.last_stats = {
            **info,
            "total_batches": stats["batches"],
            "computed_batches": (0 if info["status"] == "unchanged"
                                 else stats["batches"] - info["reused_batches"]),
            "checkpoint_count": len(self._checkpoints),
            "cache_array_bytes": (stats["working_array_bytes"]
                                  + sum(size for _, size in self._checkpoints)),
        }
        return np.ma.filled(self._state.result(), np.nan)
