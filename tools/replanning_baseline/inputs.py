"""Benchmark inputs and provenance; no ROS dependencies."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
from PIL import Image
import yaml

from agx_planning.pmp_planner import PlannerConfig
from agx_planning.vector_field import VectorFieldConfig


REPO = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def array_digest(array):
    h = hashlib.sha256()
    h.update(str((array.shape, array.dtype.str)).encode())
    h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_profile(path):
    profile = yaml.safe_load(Path(path).read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported profile schema_version")
    planner = PlannerConfig(**profile["planner"])
    field = VectorFieldConfig(**profile["field"])
    if field.bottleneck_align:
        raise ValueError("The packed field path does not transport bottleneck directions")
    if profile["field_path"] != "packed_float32":
        raise ValueError("Only the production packed_float32 field path is supported")
    if not isinstance(profile["allow_unknown"], bool):
        raise ValueError("allow_unknown must be a boolean")
    threshold = profile["occupancy_threshold"]
    if not isinstance(threshold, int) or not 1 <= threshold <= 100:
        raise ValueError("occupancy_threshold must be an integer in [1, 100]")
    if min(planner.control_rate, planner.dt_segment, planner.T_horizon,
           planner.max_rollout_sim_time) <= 0:
        raise ValueError("Planner timing parameters must be positive")
    period = 1.0 / planner.control_rate
    ticks = min(planner.dt_segment, planner.T_horizon) / period
    if ticks < 1 or not np.isclose(ticks, round(ticks)):
        raise ValueError("Committed duration must be an integer number of sample periods")
    if profile["pin_tolerance"] <= 0:
        raise ValueError("pin_tolerance must be positive")
    return {**profile, "planner": asdict(planner), "field": asdict(field)}


def apply_rectangles(occupancy, rectangles):
    """Apply half-open [col0, row0, col1, row1, value] rectangles."""
    result = occupancy.copy()
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
    resolution = float(meta["resolution"])
    origin = np.asarray(meta["origin"], dtype=float)
    if not np.isfinite(resolution) or resolution <= 0 or origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid map geometry")
    if origin[2] != 0:
        raise ValueError("Rotated map origins are unsupported")
    geometry = {"shape": list(occupancy.shape), "resolution": resolution, "origin": origin.tolist()}
    return occupancy, geometry, source


def cell(pose, geometry):
    xy = (np.asarray(pose[:2]) - geometry["origin"][:2]) / geometry["resolution"]
    col, row = np.floor(xy).astype(int)
    height, width = geometry["shape"]
    if not (0 <= col < width and 0 <= row < height):
        raise ValueError(f"Pose outside map: {pose}")
    return int(col), int(row)


def vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} requires {size} finite values")
    return result.tolist()


def load_suite(path, selected=()):
    path = Path(path).resolve()
    suite = yaml.safe_load(path.read_text())
    if suite.get("schema_version") != 1 or not suite.get("cases"):
        raise ValueError("Expected schema_version 1 and nonempty cases")
    profile = load_profile(path.parent / suite["profile"])
    base, geometry, source = load_map(suite["map"], path.parent)
    ids = [case["id"] for case in suite["cases"]]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Case IDs must be nonempty and unique")
    if set(selected) - set(ids):
        raise ValueError(f"Unknown case IDs: {sorted(set(selected) - set(ids))}")
    snapshots = []
    for case in suite["cases"]:
        if selected and case["id"] not in selected:
            continue
        occupancy = base.copy()
        start = vector(case["start"], 5, "start")
        goal = vector(case["goal"], 3, "goal")
        seen = set()
        for update in [{"id": "base", "rectangles": []}, *case.get("updates", [])]:
            name = update["id"]
            if not isinstance(name, str) or not name or name in seen:
                raise ValueError("Snapshot IDs must be nonempty and unique within a case")
            seen.add(name)
            new = apply_rectangles(occupancy, update.get("rectangles", []))
            start = vector(update.get("start", start), 5, "start")
            goal = vector(update.get("goal", goal), 3, "goal")
            cell(start, geometry)
            cell(goal, geometry)
            metadata = {"case_id": case["id"], "snapshot_id": name,
                        "parent_route_id": case.get("parent_route_id", case["id"]),
                        "start": start, "goal": goal, "geometry": geometry,
                        "map_sha256": array_digest(new),
                        "changed_cells": int(np.count_nonzero(new != occupancy))}
            metadata["input_sha256"] = digest(metadata)
            snapshots.append((metadata, new))
            occupancy = new
    return profile, snapshots, {"path": str(path), "content": suite, "map_source": source}


def provenance():
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()

    import skfmm
    from skfmm import cfmm
    packages = {}
    for name in ("numpy", "scipy", "scikit-fmm", "Pillow", "PyYAML", "pytest"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    binary = Path(cfmm.__file__)
    return {"git_head": git("rev-parse", "HEAD"),
            "git_status": git("status", "--porcelain", "--untracked-files=normal"),
            "tracked_diff_sha256": hashlib.sha256(git("diff", "HEAD", "--binary").encode()).hexdigest(),
            "submodules": git("submodule", "status", "--recursive").splitlines(),
            "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "processor": platform.processor(), "packages": packages,
            "skfmm_path": skfmm.__file__, "skfmm_binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}
