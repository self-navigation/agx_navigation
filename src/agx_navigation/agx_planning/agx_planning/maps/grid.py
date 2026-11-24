"""Checked map revisions and one cell-coordinate convention.

The origin is the lower-left corner of cell (0, 0). Cell indices are (col, row),
array indices are [row, col], and sampled positions are cell centres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import numbers

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _owned_readonly(array, dtype):
    """Use immutable bytes so callers cannot turn WRITEABLE back on."""
    array = np.ascontiguousarray(array, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _occupancy(array, shape=None):
    value = np.asarray(array)
    if value.ndim != 2 or value.dtype.kind not in "iu":
        raise ValueError("occupancy must be a two-dimensional integer array")
    if shape is not None and value.shape != tuple(shape):
        raise ValueError("occupancy shape differs from geometry")
    if value.size == 0 or np.any(value < -1) or np.any(value > 100):
        raise ValueError("occupancy values must be in [-1, 100]")
    return _owned_readonly(value, np.int8)


def _digest(metadata, array=None):
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, allow_nan=False).encode())
    if array is not None:
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class GridGeometry:
    shape: tuple[int, int]
    resolution: float
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame: str = "map"
    convention_version: int = 1

    def __post_init__(self):
        if len(self.shape) != 2:
            raise ValueError("grid shape must have two dimensions")
        shape = tuple(_integer(n, "grid dimension", 1) for n in self.shape)
        resolution = float(self.resolution)
        origin = tuple(float(v) for v in self.origin)
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        if len(origin) != 3 or not all(math.isfinite(v) for v in origin):
            raise ValueError("origin must contain three finite values")
        if origin[2] != 0.0:
            raise ValueError("Rotated map origins are unsupported")
        upper = (origin[0] + shape[1] * resolution, origin[1] + shape[0] * resolution)
        if not all(math.isfinite(v) and v > o for v, o in zip(upper, origin[:2])):
            raise ValueError("map extent must be finite and representable")
        if not isinstance(self.frame, str) or not self.frame:
            raise ValueError("frame must be a nonempty string")
        if _integer(self.convention_version, "convention_version", 1) != 1:
            raise ValueError("unsupported grid convention version")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "convention_version", 1)

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    def world_to_cell(self, x, y):
        x, y = float(x), float(y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("world coordinates must be finite")
        if not (self.origin[0] <= x < self.origin[0] + self.width * self.resolution
                and self.origin[1] <= y < self.origin[1] + self.height * self.resolution):
            raise ValueError(f"Position outside map: ({x}, {y})")
        col = math.floor((x - self.origin[0]) / self.resolution)
        row = math.floor((y - self.origin[1]) / self.resolution)
        if not (0 <= col < self.width and 0 <= row < self.height):
            raise ValueError(f"Position outside map: ({x}, {y})")
        return col, row

    def cell_center(self, col, row):
        col = _integer(col, "column")
        row = _integer(row, "row")
        if col >= self.width or row >= self.height:
            raise ValueError("cell outside map")
        return (self.origin[0] + (col + 0.5) * self.resolution,
                self.origin[1] + (row + 0.5) * self.resolution)

    def to_dict(self, include_metadata=True):
        result = {"shape": list(self.shape), "resolution": self.resolution,
                  "origin": list(self.origin)}
        if include_metadata:
            result.update(frame=self.frame, convention_version=self.convention_version)
        return result

    as_dict = to_dict

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True)
class OccupancyPolicy:
    threshold: int = 65
    allow_unknown: bool = True

    def __post_init__(self):
        threshold = _integer(self.threshold, "threshold", 1)
        if threshold > 100:
            raise ValueError("threshold must be in [1, 100]")
        if not isinstance(self.allow_unknown, bool):
            raise ValueError("allow_unknown must be a boolean")
        object.__setattr__(self, "threshold", threshold)

    def mask(self, occupancy):
        occupancy = _occupancy(occupancy)
        result = occupancy >= self.threshold
        if not self.allow_unknown:
            result |= occupancy < 0
        return _owned_readonly(result, bool)

    def to_dict(self):
        return {"threshold": self.threshold, "allow_unknown": self.allow_unknown}

    @property
    def identity(self):
        return _digest(self.to_dict())


@dataclass(frozen=True)
class MapSnapshot:
    session: str
    revision: int
    geometry: GridGeometry
    occupancy: np.ndarray = field(repr=False, compare=False)
    policy: OccupancyPolicy = field(default_factory=OccupancyPolicy)
    identity: str = field(init=False)
    obstacles: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("map session must be a nonempty string")
        revision = _integer(self.revision, "map revision")
        if not isinstance(self.geometry, GridGeometry) or not isinstance(self.policy, OccupancyPolicy):
            raise ValueError("map requires GridGeometry and OccupancyPolicy")
        occupancy = _occupancy(self.occupancy, self.geometry.shape)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "occupancy", occupancy)
        object.__setattr__(self, "obstacles", self.policy.mask(occupancy))
        object.__setattr__(self, "identity", _digest(self.to_dict(), occupancy))

    def to_dict(self):
        return {"session": self.session, "revision": self.revision,
                "geometry": self.geometry.to_dict(), "policy": self.policy.to_dict()}

    def updated(self, occupancy, revision=None):
        revision = self.revision + 1 if revision is None else _integer(revision, "map revision")
        if revision <= self.revision:
            raise ValueError("new map revision must be greater than the current revision")
        return MapSnapshot(self.session, revision, self.geometry, occupancy, self.policy)


@dataclass(frozen=True)
class MapDelta:
    session: str
    parent_revision: int
    revision: int
    geometry: GridGeometry
    indices: np.ndarray = field(repr=False, compare=False)
    values: np.ndarray = field(repr=False, compare=False)
    parent_identity: str | None = None

    def __post_init__(self):
        if not isinstance(self.session, str) or not self.session:
            raise ValueError("map session must be a nonempty string")
        parent = _integer(self.parent_revision, "parent revision")
        revision = _integer(self.revision, "map revision")
        if revision <= parent:
            raise ValueError("delta revision must be greater than parent revision")
        if not isinstance(self.geometry, GridGeometry):
            raise ValueError("delta requires GridGeometry")
        indices, values = np.asarray(self.indices), np.asarray(self.values)
        if indices.ndim != 1 or values.ndim != 1 or indices.size != values.size:
            raise ValueError("delta indices and values must be parallel one-dimensional arrays")
        if not indices.size:
            indices, values = indices.astype(np.int64), values.astype(np.int8)
        if indices.dtype.kind not in "iu" or values.dtype.kind not in "iu":
            raise ValueError("delta indices and values must be integers")
        if (np.any(indices < 0) or np.any(indices >= math.prod(self.geometry.shape))
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("delta indices must be unique cells inside the grid")
        if np.any(values < -1) or np.any(values > 100):
            raise ValueError("delta values must be in [-1, 100]")
        if self.parent_identity is not None and not isinstance(self.parent_identity, str):
            raise ValueError("parent identity must be a string or None")
        object.__setattr__(self, "parent_revision", parent)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "indices", _owned_readonly(indices, np.int64))
        object.__setattr__(self, "values", _owned_readonly(values, np.int8))

    def apply(self, parent):
        if not isinstance(parent, MapSnapshot):
            raise ValueError("delta parent must be a MapSnapshot")
        if (parent.session != self.session or parent.revision != self.parent_revision
                or parent.geometry != self.geometry):
            raise ValueError("delta parent session, revision, or geometry does not match")
        if self.parent_identity is not None and parent.identity != self.parent_identity:
            raise ValueError("delta parent content identity does not match")
        occupancy = parent.occupancy.copy()
        occupancy.flat[self.indices] = self.values
        return parent.updated(occupancy, self.revision)

    @classmethod
    def between(cls, before, after):
        if (before.session != after.session or before.geometry != after.geometry
                or before.policy != after.policy):
            raise ValueError("delta cannot change map session, geometry, or occupancy policy")
        indices = np.flatnonzero(before.occupancy != after.occupancy)
        return cls(before.session, before.revision, after.revision, before.geometry,
                   indices, after.occupancy.ravel()[indices], before.identity)
