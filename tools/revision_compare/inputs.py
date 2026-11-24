"""Version-independent input identities and cell lookup for the common protocol."""

import hashlib
import json
import math

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_digest(array):
    value = np.asarray(array)
    result = hashlib.sha256(str((value.shape, value.dtype.str)).encode())
    result.update(np.ascontiguousarray(value).tobytes())
    return result.hexdigest()


def cell(pose, geometry):
    """Floor cell lookup, shared by both revisions and frozen with the inputs."""
    x, y = map(float, pose[:2])
    height, width = geometry["shape"]
    resolution = float(geometry["resolution"])
    origin = geometry["origin"]
    ox, oy = map(float, origin[:2])
    if (not all(math.isfinite(v) for v in (x, y, ox, oy, resolution))
            or resolution <= 0 or len(origin) != 3 or origin[2] != 0):
        raise ValueError("invalid or rotated map geometry")
    if not (ox <= x < ox + width * resolution and oy <= y < oy + height * resolution):
        raise ValueError("position is outside the map")
    col, row = math.floor((x - ox) / resolution), math.floor((y - oy) / resolution)
    if not (0 <= col < width and 0 <= row < height):
        raise ValueError("position cannot be represented as a map cell")
    return col, row


def validate_input(metadata, occupancy):
    """Reject changed, duplicate, or malformed prepared inputs before timing."""
    if occupancy.ndim != 2 or occupancy.dtype != np.int8:
        raise ValueError("prepared occupancy must be a two-dimensional int8 array")
    if list(occupancy.shape) != list(metadata["geometry"]["shape"]):
        raise ValueError("prepared occupancy shape differs from geometry")
    if np.any(occupancy < -1) or np.any(occupancy > 100):
        raise ValueError("prepared occupancy values are outside [-1, 100]")
    if array_digest(occupancy) != metadata["map_sha256"]:
        raise ValueError("prepared occupancy hash differs from manifest")
    identity = {key: value for key, value in metadata.items() if key != "input_sha256"}
    if digest(identity) != metadata["input_sha256"]:
        raise ValueError("prepared input metadata hash differs from manifest")
    for key, count in (("start", 5), ("goal", 3)):
        value = np.asarray(metadata[key], float)
        if value.shape != (count,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain {count} finite values")
        cell(value, metadata["geometry"])
    for key in ("case_id", "snapshot_id"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise ValueError(f"{key} must be a nonempty string")
