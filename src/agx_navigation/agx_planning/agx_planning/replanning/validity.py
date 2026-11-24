"""Filled-footprint and conservative swept validity checks without ROS.

Map cells are closed squares for collision purposes. Contact is rejected.
Reported clearance is a conservative lower bound, not an exact distance field.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Footprint:
    half_length: float = 0.315
    half_width: float = 0.2925
    margin: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.half_length, self.half_width, self.margin)):
            raise ValueError("footprint dimensions and margin must be finite")
        if min(self.half_length, self.half_width) <= 0 or self.margin < 0:
            raise ValueError("footprint dimensions must be positive and margin nonnegative")
        if not math.isfinite(math.hypot(self.half_length + self.margin, self.half_width + self.margin)):
            raise ValueError("expanded footprint must have finite extent")

    @property
    def radius(self):
        return math.hypot(self.half_length + self.margin, self.half_width + self.margin)


@dataclass(frozen=True)
class ValidityResult:
    valid: bool
    reason: str
    clearance_lower_bound: float
    first_unsafe_interval: int | None
    valid_until: float
    map_revision: str


@dataclass(frozen=True, init=False, eq=False)
class GridCollisionChecker:
    """Own a fixed obstacle map and check rectangles against occupied cell areas.

    ``occupied`` is a bool grid whose row zero starts at origin y. Occupancy
    threshold and unknown-cell policy must already have been applied by the
    map snapshot. The caller supplies that snapshot's complete identity.
    """

    occupied: np.ndarray
    resolution: float
    origin: tuple[float, float]
    map_revision: str
    footprint: Footprint
    clearance_cap: float

    def __init__(self, occupied, resolution, origin=(0.0, 0.0), *,
                 map_revision, footprint=Footprint(), clearance_cap=None):
        occupied = np.asarray(occupied)
        if occupied.dtype != np.bool_ or occupied.ndim != 2 or min(occupied.shape) < 1:
            raise ValueError("occupied must be a nonempty two-dimensional bool grid")
        origin = np.asarray(origin, dtype=float)
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("origin must contain finite x and y")
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError("resolution must be finite and positive")
        cap = resolution if clearance_cap is None else float(clearance_cap)
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("clearance_cap must be finite and positive")
        if not isinstance(map_revision, str) or not map_revision:
            raise ValueError("map_revision must be a nonempty snapshot identity string")
        if not np.isfinite(origin + np.asarray(occupied.shape[::-1]) * resolution).all():
            raise ValueError("map extent must be finite")
        immutable = np.frombuffer(occupied.tobytes(), dtype=bool).reshape(occupied.shape)
        object.__setattr__(self, "occupied", immutable)
        object.__setattr__(self, "resolution", float(resolution))
        object.__setattr__(self, "origin", tuple(float(v) for v in origin))
        object.__setattr__(self, "map_revision", map_revision)
        object.__setattr__(self, "footprint", footprint)
        object.__setattr__(self, "clearance_cap", cap)

    def _pose(self, pose, extra):
        x, y, theta = pose
        c, s = math.cos(theta), math.sin(theta)
        a = self.footprint.half_length + self.footprint.margin + extra
        b = self.footprint.half_width + self.footprint.margin + extra
        extent_x, extent_y = abs(c) * a + abs(s) * b, abs(s) * a + abs(c) * b
        if not all(math.isfinite(v) for v in (a, b, extent_x, extent_y)):
            raise ValueError("inflated footprint extent must be finite")
        ox, oy = self.origin
        height, width = self.occupied.shape
        res = self.resolution
        tolerance = 16 * np.finfo(float).eps * (1 + abs(x) + abs(y) + a + b + res + abs(ox) + abs(oy))
        boundary = min(x - extent_x - ox, ox + width * res - x - extent_x,
                       y - extent_y - oy, oy + height * res - y - extent_y) - tolerance
        if boundary <= 0:
            return False, "out_of_bounds", float(boundary)

        cap = self.clearance_cap
        c0 = max(0, math.floor((x - extent_x - cap - ox) / res))
        c1 = min(width, math.floor((x + extent_x + cap - ox) / res) + 1)
        r0 = max(0, math.floor((y - extent_y - cap - oy) / res))
        r1 = min(height, math.floor((y + extent_y + cap - oy) / res) + 1)
        rows, cols = np.nonzero(self.occupied[r0:r1, c0:c1])
        clearance = min(cap, boundary)
        if len(rows):
            dx = ox + (cols + c0 + 0.5) * res - x
            dy = oy + (rows + r0 + 0.5) * res - y
            cell_half = 0.5 * res
            projected_cell = cell_half * (abs(c) + abs(s))
            # Separating axes of an oriented rectangle and an axis-aligned cell.
            gaps = np.maximum.reduce((
                np.abs(dx) - extent_x - cell_half,
                np.abs(dy) - extent_y - cell_half,
                np.abs(c * dx + s * dy) - a - projected_cell,
                np.abs(-s * dx + c * dy) - b - projected_cell,
            )) - tolerance
            clearance = min(clearance, float(gaps.min()))
            if np.any(gaps <= 0):
                return False, "occupied", float(clearance)
        return True, "clear", float(clearance)

    def check_pose(self, pose, *, extra_margin=0.0):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("pose must contain finite x, y, and unwrapped yaw")
        if not math.isfinite(extra_margin) or extra_margin < 0:
            raise ValueError("extra_margin must be finite and nonnegative")
        valid, reason, clearance = self._pose(pose, extra_margin)
        return ValidityResult(valid, reason, clearance, None if valid else 0,
                              0.0, self.map_revision)

    def check_path(self, poses, times=None, *, interval_motion_bounds=None,
                   pose_error_bounds=None, max_corner_step=None):
        """Certify a path under an explicit motion model.

        Without motion bounds, position and unwrapped yaw interpolate linearly.
        Each subdivision is checked at its midpoint with an envelope covering
        all footprint motion in that subdivision. This is not a certificate
        for an arbitrary curve through the same endpoints.

        For a curved executed path, supply one bound per interval on the total
        footprint-point motion: integral(|v| + footprint.radius*|omega|) dt.
        The two endpoint footprints, each inflated by half that bound, cover
        the interval. No assumed intermediate pose is introduced in this mode.
        Optional per-pose spatial error bounds inflate the endpoint envelopes.
        In linear mode, the larger endpoint error must also bound error inside
        that interval. In bounded-motion mode, errors cover the endpoint poses.
        The returned valid_until is a conservative prefix boundary.
        """
        poses = np.asarray(poses, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 3 or len(poses) < 1 or not np.isfinite(poses).all():
            raise ValueError("poses must be a finite nonempty N by 3 array")
        times = np.arange(len(poses), dtype=float) if times is None else np.asarray(times, dtype=float)
        if times.shape != (len(poses),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite and strictly increasing")
        errors = np.zeros(len(poses)) if pose_error_bounds is None else np.asarray(pose_error_bounds, dtype=float)
        if errors.shape != (len(poses),) or not np.isfinite(errors).all() or np.any(errors < 0):
            raise ValueError("pose_error_bounds must be finite nonnegative values at each pose")
        step = self.resolution / 2 if max_corner_step is None else float(max_corner_step)
        if not math.isfinite(step) or step <= 0:
            raise ValueError("max_corner_step must be finite and positive")
        bounds = None if interval_motion_bounds is None else np.asarray(interval_motion_bounds, dtype=float)
        if bounds is not None and (bounds.shape != (len(poses) - 1,) or
                                   not np.isfinite(bounds).all() or np.any(bounds < 0)):
            raise ValueError("interval_motion_bounds must cover each interval with finite nonnegative values")
        first = self.check_pose(poses[0], extra_margin=errors[0])
        if not first.valid:
            return ValidityResult(False, first.reason, first.clearance_lower_bound,
                                  0, float(times[0]), self.map_revision)
        minimum = first.clearance_lower_bound
        radius = self.footprint.radius
        for i, (a, b) in enumerate(zip(poses[:-1], poses[1:])):
            displacement = float(np.linalg.norm(b[:2] - a[:2]) + radius * abs(b[2] - a[2]))
            if not math.isfinite(displacement):
                raise ValueError("path interval displacement must be finite")
            error = float(max(errors[i], errors[i + 1]))
            if bounds is not None:
                if bounds[i] + errors[i] + errors[i + 1] + 1e-12 < displacement:
                    raise ValueError("motion bound is smaller than endpoint displacement")
                for endpoint in (a, b):
                    valid, reason, clearance = self._pose(endpoint, error + bounds[i] / 2)
                    minimum = min(minimum, clearance)
                    if not valid:
                        return ValidityResult(False, reason, minimum, i, float(times[i]), self.map_revision)
            else:
                count = max(1, math.ceil(displacement / step))
                # Avoid unbounded work for malformed or extreme intervals.
                if count > 1_000_000:
                    raise ValueError("path interval requires too many subdivisions")
                envelope = displacement / (2 * count) + error
                for j in range(count):
                    pose = a + ((j + 0.5) / count) * (b - a)
                    valid, reason, clearance = self._pose(pose, envelope)
                    minimum = min(minimum, clearance)
                    if not valid:
                        prefix = times[i] + (j / count) * (times[i + 1] - times[i])
                        return ValidityResult(False, reason, minimum, i, float(prefix), self.map_revision)
        return ValidityResult(True, "clear", minimum, None, float(times[-1]), self.map_revision)
