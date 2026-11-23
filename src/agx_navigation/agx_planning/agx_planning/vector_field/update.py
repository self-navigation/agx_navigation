"""Reuse field work when the obstacle mask and settings allow it."""

from copy import deepcopy
from dataclasses import astuple

import numpy as np

from . import field as vf


class FieldUpdater:
    """Keep one map and one field in memory.

    Changed obstacles or settings trigger full recomputation. A new goal or
    early-exit target reuses map preprocessing. Identical requests reuse the
    field. Returned results are copies, so callers can change them safely.
    Use one instance per serial stream of requests.
    """

    def __init__(self):
        self.clear()

    def clear(self):
        self._map_key = None
        self._map_data = None
        self._field_key = None
        self._outcome = None
        self.last_update = "empty"

    def update(
        self,
        map_array: np.ndarray,
        goal_col: int,
        goal_row: int,
        resolution: float,
        origin_x: float,
        origin_y: float,
        cfg: vf.VectorFieldConfig,
        robot_col: int | None = None,
        robot_row: int | None = None,
        occupancy_threshold: int = 65,
        allow_unknown: bool = False,
    ) -> tuple[vf.VectorFieldResult, str] | None:
        """Match compute_field(), including None for invalid request cells.

        last_update is full_recompute, map_reuse, field_reuse, or invalid.
        An invalid request or exception clears the cache.
        """
        try:
            obstacles = vf._obstacle_mask(map_array, occupancy_threshold, allow_unknown)
            if not vf._valid_cells(obstacles, goal_col, goal_row, cfg, robot_col, robot_row):
                self.clear()
                self.last_update = "invalid"
                return None

            map_key = (resolution, origin_x, origin_y, occupancy_threshold,
                       allow_unknown, astuple(cfg))
            same_map = (map_key == self._map_key and self._map_data is not None
                        and np.array_equal(obstacles, self._map_data.obstacle_mask))
            target = (robot_col, robot_row) if cfg.early_exit_enable else None
            field_key = (goal_col, goal_row, target)
            if same_map and field_key == self._field_key:
                self.last_update = "field_reuse"
                return deepcopy(self._outcome)

            prepared = (self._map_data if same_map
                        else vf._prepare_map(obstacles, resolution, cfg))
            outcome = vf._compute_prepared_field(
                prepared, goal_col, goal_row, resolution, origin_x, origin_y,
                cfg, robot_col, robot_row,
            )
            self._map_key, self._map_data = map_key, prepared
            self._field_key, self._outcome = field_key, outcome
            self.last_update = "map_reuse" if same_map else "full_recompute"
            return deepcopy(outcome)
        except Exception:
            self.clear()
            self.last_update = "invalid"
            raise
