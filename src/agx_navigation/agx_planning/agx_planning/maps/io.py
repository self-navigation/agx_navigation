"""Pure trinary PNG/YAML loading and synthetic map construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .grid import GridGeometry, MapSnapshot, OccupancyPolicy


def apply_rectangles(occupancy, rectangles):
    """Copy and apply half-open [col0, row0, col1, row1, value] rectangles."""
    result = np.array(occupancy, copy=True)
    if result.ndim != 2 or result.dtype.kind not in "iu":
        raise ValueError("occupancy must be a two-dimensional integer array")
    if np.any(result < -1) or np.any(result > 100):
        raise ValueError("Occupancy values must be in [-1, 100]")
    # int8 is the canonical occupancy representation, including unknown=-1.
    result = result.astype(np.int8, copy=False)
    height, width = result.shape
    for rectangle in rectangles:
        if len(rectangle) != 5 or any(type(v) is not int for v in rectangle):
            raise ValueError("Rectangles require five integers")
        c0, r0, c1, r1, value = rectangle
        if not (0 <= c0 < c1 <= width and 0 <= r0 < r1 <= height):
            raise ValueError(f"Rectangle outside map: {rectangle}")
        if not -1 <= value <= 100:
            raise ValueError("Occupancy values must be in [-1, 100]")
        result[r0:r1, c0:c1] = value
    return result


def load_map(spec, directory):
    """Return occupancy, legacy geometry metadata, and source hashes.

    Supports {"yaml": relative_path} and the existing benchmark {"grid": ...}
    format. Pixel conversion, threshold inclusivity, and row flipping preserve
    the recorded benchmark convention.
    """
    directory = Path(directory)
    if set(spec) == {"yaml"}:
        path = (directory / spec["yaml"]).resolve()
        meta = yaml.safe_load(path.read_text())
        if meta.get("mode", "trinary") != "trinary":
            raise ValueError("Only trinary map YAML is supported")
        lo, hi = float(meta["free_thresh"]), float(meta["occupied_thresh"])
        if not 0 <= lo < hi <= 1 or meta.get("negate", 0) not in (0, 1):
            raise ValueError("Invalid map thresholds or negate")
        image_path = (path.parent / meta["image"]).resolve()
        with Image.open(image_path) as image:
            pixels = np.flipud(np.asarray(image.convert("L"))).astype(float)
        probability = (255.0 - pixels) / 255.0
        if meta.get("negate", 0):
            probability = 1.0 - probability
        occupancy = np.full(pixels.shape, -1, dtype=np.int8)
        occupancy[probability >= hi] = 100
        occupancy[probability <= lo] = 0
        source = {"yaml": str(path), "yaml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "image": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()}
    elif set(spec) == {"grid"}:
        meta = spec["grid"]
        shape = meta["shape"]
        if len(shape) != 2 or any(type(n) is not int or n < 3 for n in shape):
            raise ValueError("Grid shape must contain two integers >= 3")
        occupancy = apply_rectangles(np.zeros(shape, dtype=np.int8), meta.get("rectangles", []))
        source = {"synthetic": spec}
    else:
        raise ValueError("Map requires exactly one of yaml or grid")
    geometry = GridGeometry(occupancy.shape, meta["resolution"], meta["origin"])
    return occupancy, geometry.to_dict(include_metadata=False), source


def load_snapshot(path, *, session, revision=0, policy=None):
    """Load one map YAML into an identified immutable snapshot."""
    path = Path(path).resolve()
    occupancy, geometry, _ = load_map({"yaml": path.name}, path.parent)
    return MapSnapshot(session, revision, GridGeometry.from_dict(geometry), occupancy,
                       OccupancyPolicy() if policy is None else policy)
