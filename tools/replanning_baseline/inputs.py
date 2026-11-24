"""Benchmark inputs and provenance; no ROS dependencies."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import yaml

from agx_planning.maps import GridGeometry
from agx_planning.maps.io import apply_rectangles, load_map
from agx_planning.profiles import load_profile as load_core_profile


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
    return load_core_profile(path, allowed_field_paths=("packed_float32",))


def cell(pose, geometry):
    return GridGeometry.from_dict(geometry).world_to_cell(*pose[:2])


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
