"""Deterministic whole-object revelation for model-level experiments.

The reveal model and evaluator may access the true map. A planner receives only
the current revealed MapSnapshot returned by scan or reveal_ids. Visibility of
one occupied cell centre reveals the whole object; this is not a scan simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from agx_planning.maps import MapSnapshot
from agx_planning.maps.io import apply_rectangles


@dataclass(frozen=True)
class HiddenObject:
    object_id: str
    rectangles: tuple[tuple[int, int, int, int, int], ...]

    def __post_init__(self):
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("object ID must be a nonempty string")
        rectangles = tuple(tuple(rectangle) for rectangle in self.rectangles)
        if not rectangles:
            raise ValueError("hidden object must contain at least one rectangle")
        for rectangle in rectangles:
            if len(rectangle) != 5 or any(type(value) is not int for value in rectangle):
                raise ValueError("object rectangles require five integers")
            c0, r0, c1, r1, value = rectangle
            if min(c0, r0) < 0 or c1 <= c0 or r1 <= r0 or not 0 <= value <= 100:
                raise ValueError("invalid hidden-object rectangle")
        object.__setattr__(self, "rectangles", rectangles)


class ObjectRevealer:
    """Reveal obstacle additions by deterministic range, FOV, and grid LOS.

    Hidden objects may overlap known walls but not each other. Known wall cells
    cannot reveal an object by themselves. Unknown cells follow the map's
    occupancy policy when testing occlusion. Controlled reveal_ids supports
    matched, predeclared observation schedules without a visibility claim.
    """

    def __init__(self, base, objects, max_range, fov=2.0 * math.pi):
        if not isinstance(base, MapSnapshot):
            raise ValueError("base must be a MapSnapshot")
        max_range, fov = float(max_range), float(fov)
        if not math.isfinite(max_range) or max_range <= 0:
            raise ValueError("reveal range must be finite and positive")
        if not math.isfinite(fov) or not 0 < fov <= 2.0 * math.pi:
            raise ValueError("reveal FOV must be in (0, 2*pi]")
        objects = tuple(objects)
        if any(not isinstance(obj, HiddenObject) for obj in objects):
            raise ValueError("objects must contain HiddenObject records")
        ids = [obj.object_id for obj in objects]
        if len(set(ids)) != len(ids):
            raise ValueError("hidden-object IDs must be unique")
        assigned = np.zeros(base.geometry.shape, dtype=bool)
        true_occupancy = base.occupancy.copy()
        candidates = {}
        for obj in objects:
            footprint = np.zeros(base.geometry.shape, dtype=bool)
            for c0, r0, c1, r1, value in obj.rectangles:
                if c1 > base.geometry.width or r1 > base.geometry.height:
                    raise ValueError("hidden-object rectangle is outside the map")
                if value < base.policy.threshold:
                    raise ValueError("hidden objects must add occupied cells")
                footprint[r0:r1, c0:c1] = True
            if np.any(footprint & assigned):
                raise ValueError("hidden objects must not overlap each other")
            assigned |= footprint
            # Do not reveal a hidden leaf from an already known wall alone.
            visible_cells = footprint & (base.occupancy < base.policy.threshold)
            rows, cols = np.nonzero(visible_cells)
            if not rows.size:
                raise ValueError("hidden object must add a previously unknown obstacle cell")
            candidates[obj.object_id] = tuple(zip(cols.tolist(), rows.tolist()))
            true_occupancy = apply_rectangles(true_occupancy, obj.rectangles)
        self._objects = objects
        self._by_id = {obj.object_id: obj for obj in objects}
        self._candidates = candidates
        self._current = base
        # A separate session prevents a true-map certificate from being used
        # as a revealed-map certificate even when revisions happen to match.
        self._true = MapSnapshot(base.session + ":truth", base.revision, base.geometry,
                                 true_occupancy, base.policy)
        self._revealed = set()
        self.max_range = max_range
        self.fov = fov

    @property
    def current(self):
        return self._current

    @property
    def true_map(self):
        """Evaluator input; do not pass this snapshot to a hidden-map planner."""
        return self._true

    @property
    def revealed_ids(self):
        return tuple(obj.object_id for obj in self._objects if obj.object_id in self._revealed)

    def reveal_ids(self, ids):
        if isinstance(ids, str):
            raise ValueError("reveal IDs must be a sequence, not one string")
        ids = tuple(ids)
        if any(not isinstance(name, str) or name not in self._by_id for name in ids):
            raise ValueError("unknown hidden-object ID")
        requested = set(ids)
        newly_revealed = tuple(obj.object_id for obj in self._objects
                               if obj.object_id in requested and obj.object_id not in self._revealed)
        if newly_revealed:
            occupancy = self._current.occupancy.copy()
            for name in newly_revealed:
                occupancy = apply_rectangles(occupancy, self._by_id[name].rectangles)
            self._current = self._current.updated(occupancy)
            self._revealed.update(newly_revealed)
        return self._current, newly_revealed

    def _visible(self, x, y, target_col, target_row):
        """Conservative supercover traversal of a segment to a cell centre."""
        geometry = self._true.geometry
        col, row = geometry.world_to_cell(x, y)
        mask = self._true.obstacles
        if mask[row, col] and (col, row) != (target_col, target_row):
            return False
        x0 = (x - geometry.origin[0]) / geometry.resolution
        y0 = (y - geometry.origin[1]) / geometry.resolution
        dx, dy = target_col + 0.5 - x0, target_row + 0.5 - y0
        step_col = 1 if dx > 0 else -1 if dx < 0 else 0
        step_row = 1 if dy > 0 else -1 if dy < 0 else 0
        delta_col = abs(1.0 / dx) if dx else math.inf
        delta_row = abs(1.0 / dy) if dy else math.inf
        next_col = ((col + 1 if dx > 0 else col) - x0) / dx if dx else math.inf
        next_row = ((row + 1 if dy > 0 else row) - y0) / dy if dy else math.inf

        def blocked(c, r):
            if (c, r) == (target_col, target_row):
                return False
            return not (0 <= c < geometry.width and 0 <= r < geometry.height) or bool(mask[r, c])

        # Both endpoints are in the grid, so each step reaches a new row or
        # column and this finite bound is larger than every possible traversal.
        for _ in range(geometry.width + geometry.height + 1):
            if (col, row) == (target_col, target_row):
                return True
            if math.isclose(next_col, next_row, rel_tol=0.0, abs_tol=1e-12):
                if blocked(col + step_col, row) or blocked(col, row + step_row):
                    return False
                col += step_col
                row += step_row
                next_col += delta_col
                next_row += delta_row
            elif next_col < next_row:
                col += step_col
                next_col += delta_col
            else:
                row += step_row
                next_row += delta_row
            if blocked(col, row):
                return False
        raise RuntimeError("line-of-sight traversal did not reach its target")

    def scan(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.ndim != 1 or pose.size < 3 or not np.isfinite(pose[:3]).all():
            raise ValueError("scan pose must contain finite x, y, and heading")
        x, y, theta = pose[:3]
        self._current.geometry.world_to_cell(x, y)
        visible = []
        for obj in self._objects:
            if obj.object_id in self._revealed:
                continue
            for col, row in self._candidates[obj.object_id]:
                cx, cy = self._current.geometry.cell_center(col, row)
                if math.hypot(cx - x, cy - y) > self.max_range:
                    continue
                angle = (math.atan2(cy - y, cx - x) - theta + math.pi) % (2.0 * math.pi) - math.pi
                if abs(angle) > self.fov / 2.0:
                    continue
                if self._visible(x, y, col, row):
                    visible.append(obj.object_id)
                    break
        return self.reveal_ids(visible)
