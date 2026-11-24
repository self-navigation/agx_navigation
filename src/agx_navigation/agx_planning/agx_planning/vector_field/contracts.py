"""Identified, immutable field inputs for one numerical rollout.

Legacy packed arrays have no map revision or source metadata. They can be used
through the legacy grid adapter, but cannot produce an identified FieldKey.
"""

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json

import numpy as np

from agx_planning.maps.grid import GridGeometry
from .preparation import prepare_travel_time


def _digest(value):
    if is_dataclass(value):
        value = asdict(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _owned_readonly(value, dtype, shape, name):
    value = np.asarray(value, dtype=dtype)
    if value.shape != shape:
        raise ValueError(f"{name} must match field geometry")
    # Immutable bytes backing also prevents callers from re-enabling writes.
    return np.frombuffer(np.ascontiguousarray(value).tobytes(), dtype=value.dtype).reshape(shape)


@dataclass(frozen=True)
class FieldKey:
    map_session: str
    map_revision: int
    geometry: GridGeometry
    goal_cell: tuple[int, int]
    recipe_id: str
    numerical_version: str
    occupancy_policy_id: str
    domain: str = "complete"
    source_definition: str = "single-negative-cell-v1"

    def __post_init__(self):
        for name in ("map_session", "recipe_id", "numerical_version", "occupancy_policy_id",
                     "domain", "source_definition"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        if isinstance(self.map_revision, bool) or not isinstance(self.map_revision, int) or self.map_revision < 0:
            raise ValueError("map_revision must be a nonnegative integer")
        if not isinstance(self.geometry, GridGeometry):
            raise TypeError("geometry must be GridGeometry")
        cell = tuple(self.goal_cell)
        if len(cell) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in cell):
            raise ValueError("goal_cell must contain integer column and row")
        col, row = map(int, cell)
        if not (0 <= row < self.geometry.shape[0] and 0 <= col < self.geometry.shape[1]):
            raise ValueError("goal_cell is outside field geometry")
        object.__setattr__(self, "goal_cell", (col, row))

    @property
    def digest(self):
        return _digest(self.to_dict())

    def to_dict(self):
        return {"map_session": self.map_session, "map_revision": self.map_revision,
                "geometry": self.geometry.to_dict(), "goal_cell": list(self.goal_cell),
                "recipe_id": self.recipe_id, "numerical_version": self.numerical_version,
                "occupancy_policy_id": self.occupancy_policy_id, "domain": self.domain,
                "source_definition": self.source_definition}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["geometry"] = GridGeometry.from_dict(value["geometry"])
        return cls(**value)

    @classmethod
    def for_map(cls, map_snapshot, goal_cell, recipe, numerical_version, domain="complete"):
        return cls(map_session=map_snapshot.session, map_revision=map_snapshot.revision,
                   geometry=map_snapshot.geometry, goal_cell=tuple(goal_cell),
                   recipe_id=_digest(recipe), numerical_version=numerical_version,
                   occupancy_policy_id=map_snapshot.policy.identity, domain=domain)


@dataclass(frozen=True, eq=False)
class FieldSnapshot:
    key: FieldKey
    travel_time: np.ndarray
    obstacle_mask: np.ndarray
    reached_mask: np.ndarray
    speed: np.ndarray | None = None
    wall_distance: np.ndarray | None = None
    uncomputed_mask: np.ndarray | None = None
    direction: tuple[np.ndarray, np.ndarray] | None = None

    def __post_init__(self):
        if not isinstance(self.key, FieldKey):
            raise TypeError("an identified FieldKey is required")
        shape = self.key.geometry.shape
        tt = _owned_readonly(self.travel_time, np.float64, shape, "travel_time")
        obstacles = _owned_readonly(self.obstacle_mask, bool, shape, "obstacle_mask")
        reached = _owned_readonly(self.reached_mask, bool, shape, "reached_mask")
        if np.isinf(tt).any() or not np.array_equal(reached, np.isfinite(tt)):
            raise ValueError("reached_mask must exactly identify finite travel times; use NaN elsewhere")
        if np.any(obstacles & reached) or np.any(tt[reached] < 0):
            raise ValueError("obstacle/reached masks or travel times are invalid")
        if self.uncomputed_mask is None:
            if self.key.domain != "complete":
                raise ValueError("partial fields require an explicit uncomputed mask")
            uncomputed = np.zeros(shape, dtype=bool)
        else:
            uncomputed = self.uncomputed_mask
        uncomputed = _owned_readonly(uncomputed, bool, shape, "uncomputed_mask")
        if np.any(uncomputed & (reached | obstacles)):
            raise ValueError("uncomputed cells must be free and unreached")
        if self.key.domain == "complete" and np.any(uncomputed):
            raise ValueError("complete fields cannot contain uncomputed cells")
        col, row = self.key.goal_cell
        if not reached[row, col]:
            raise ValueError("field source is blocked or unreached")
        object.__setattr__(self, "travel_time", tt)
        object.__setattr__(self, "obstacle_mask", obstacles)
        object.__setattr__(self, "reached_mask", reached)
        object.__setattr__(self, "uncomputed_mask", uncomputed)
        if self.speed is not None:
            speed = _owned_readonly(self.speed, np.float64, shape, "speed")
            if not np.isfinite(speed).all() or np.any(speed[reached] <= 0):
                raise ValueError("speed must be finite and positive on reached cells")
            object.__setattr__(self, "speed", speed)
        if self.wall_distance is not None:
            wall = _owned_readonly(self.wall_distance, np.float64, shape, "wall_distance")
            if not np.isfinite(wall).all():
                raise ValueError("wall_distance must be finite")
            object.__setattr__(self, "wall_distance", wall)
        if self.direction is not None:
            if len(self.direction) != 2:
                raise ValueError("direction requires x and y arrays")
            direction = tuple(_owned_readonly(value, np.float64, shape, "direction")
                              for value in self.direction)
            if not all(np.isfinite(value).all() for value in direction):
                raise ValueError("direction must be finite")
            object.__setattr__(self, "direction", direction)

    def __reduce__(self):
        # Revalidate ownership and read-only buffers after process transport.
        return (type(self), (self.key, self.travel_time, self.obstacle_mask,
                            self.reached_mask, self.speed, self.wall_distance,
                            self.uncomputed_mask, self.direction))

    @classmethod
    def from_result(cls, key, result, obstacle_mask, speed=None, uncomputed_mask=None):
        geometry = key.geometry
        if (result.travel_time.shape != geometry.shape
                or result.resolution != geometry.resolution
                or (result.origin_x, result.origin_y, 0.0) != geometry.origin):
            raise ValueError("computed result does not match the field key geometry")
        direction = None
        if result.dir_x is not None or result.dir_y is not None:
            if result.dir_x is None or result.dir_y is None:
                raise ValueError("computed direction is incomplete")
            direction = (result.dir_x, result.dir_y)
        return cls(key, result.travel_time, obstacle_mask, np.isfinite(result.travel_time),
                   speed, result.wall_dist, uncomputed_mask, direction)

    def build_grid(self, cfg):
        """Build the identified float64 planner path with deployed sentinel rules."""
        from .grid import VectorFieldGrid
        geometry = self.key.geometry
        grid = VectorFieldGrid()
        grid.update(prepare_travel_time(self.travel_time), geometry.origin[0],
                    geometry.origin[1], geometry.resolution, field_eps=cfg.field_eps,
                    wall_dist=self.wall_distance, wall_sigma=cfg.fp_smooth_sigma,
                    direction=self.direction, field_key=self.key)
        return grid.frozen_copy()
