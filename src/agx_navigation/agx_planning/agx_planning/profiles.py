"""Shared numerical profiles for ROS-free planning and experiment tools.

The returned mapping contains every planner and field dataclass setting. Save
that mapping and its digest in each run. Map bounds and source revisions belong
to the run manifest because they depend on the selected map and checkout.

The two field paths are explicit experiment choices. They do not imply that
float32 and float64 fields are numerically identical. Existing ROS launch files
still own their launch defaults; loading a profile does not change them.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import yaml

from agx_planning.maps import OccupancyPolicy
from agx_planning.pmp_planner.config import PlannerConfig
from agx_planning.vector_field.field import VectorFieldConfig


FIELD_PATHS = ("packed_float32", "raw_float64")


def _config(cls, values, name):
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must be a mapping")
    try:
        config = cls(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid {name} settings: {exc}") from exc
    defaults = cls()
    for field in fields(config):
        value = getattr(config, field.name)
        default = getattr(defaults, field.name)
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int):
            valid = isinstance(value, Integral) and not isinstance(value, bool)
        elif isinstance(default, float):
            valid = (isinstance(value, Real) and not isinstance(value, bool)
                     and math.isfinite(value))
        else:
            valid = isinstance(value, type(default))
        if not valid:
            raise ValueError(f"Invalid {name}.{field.name}: {value!r}")
    return config


def planner_config(profile):
    """Build a fresh production PlannerConfig from an expanded profile."""
    return _config(PlannerConfig, profile["planner"], "planner")


def field_config(profile):
    """Build a fresh production VectorFieldConfig from an expanded profile."""
    return _config(VectorFieldConfig, profile["field"], "field")


def expand_profile(profile, *, allowed_field_paths=FIELD_PATHS):
    """Validate a profile and fill dataclass defaults without sharing state.

    Benchmark adapters can restrict ``allowed_field_paths`` to the transport
    they measure. The core accepts both named paths. Bottleneck shaping remains
    disabled in this initial campaign, including the direct float64 path.
    """
    if not isinstance(profile, Mapping):
        raise ValueError("Profile must be a mapping")
    profile = deepcopy(dict(profile))
    if type(profile.get("schema_version")) is not int or profile["schema_version"] != 1:
        raise ValueError("Unsupported profile schema_version")
    required = {"planner", "field", "field_path", "occupancy_threshold",
                "allow_unknown", "pin_tolerance"}
    missing = required - profile.keys()
    if missing:
        raise ValueError(f"Missing profile settings: {sorted(missing)}")
    planner = planner_config(profile)
    field = field_config(profile)
    if profile["field_path"] not in FIELD_PATHS:
        raise ValueError(f"Unsupported field_path: {profile['field_path']!r}")
    if profile["field_path"] not in allowed_field_paths:
        raise ValueError(f"field_path must be one of {tuple(allowed_field_paths)!r}")
    if field.bottleneck_align:
        raise ValueError("Bottleneck shaping is disabled in the initial profile campaign")
    OccupancyPolicy(profile["occupancy_threshold"], profile["allow_unknown"])
    if planner.mode != "offline":
        raise ValueError("Experiment profiles require offline planner mode")
    timing = (planner.control_rate, planner.dt_segment, planner.T_horizon,
              planner.max_rollout_sim_time)
    if min(timing) <= 0:
        raise ValueError("Planner timing parameters must be positive")
    period = 1.0 / planner.control_rate
    ticks = min(planner.dt_segment, planner.T_horizon) / period
    if not math.isfinite(ticks) or ticks < 1 or not np.isclose(ticks, round(ticks)):
        raise ValueError("Committed duration must be an integer number of sample periods")
    if planner.N < 2 or planner.bvp_max_nodes < planner.N:
        raise ValueError("Planner mesh requires N >= 2 and bvp_max_nodes >= N")
    positive = ("wheel_radius", "track", "slip_chi", "a_wheel_max", "gamma_wheel",
                "w_wheel_max", "wheel_cmd_max", "bvp_tol", "fp_sample_spacing",
                "fp_half_length", "fp_half_width")
    if any(getattr(planner, name) <= 0 for name in positive):
        raise ValueError("Planner geometry, bounds, and solver scales must be positive")
    if field.speed_profile not in ("linear", "exponential"):
        raise ValueError("Unsupported field speed_profile")
    if not 0 < field.speed_v_min <= field.speed_v_max:
        raise ValueError("Field speed requires 0 < speed_v_min <= speed_v_max")
    if field.inflation_radius <= 0 or field.centring_cap <= 0:
        raise ValueError("Field clearance scales must be positive")
    if min(field.centring_smooth, field.smooth_T_sigma, field.early_exit_margin,
           field.speed_decay_rate) < 0:
        raise ValueError("Field smoothing, margin, and decay must be nonnegative")
    tolerance = profile["pin_tolerance"]
    if (isinstance(tolerance, bool) or not isinstance(tolerance, Real)
            or not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("pin_tolerance must be finite and positive")
    expanded = {**profile, "planner": asdict(planner), "field": asdict(field)}
    # Fail here, before an experiment starts, if extra metadata is not portable.
    try:
        json.dumps(expanded, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Profile must contain finite JSON values") from exc
    return expanded


def load_profile(path, *, allowed_field_paths=FIELD_PATHS):
    """Load and expand a versioned YAML profile without importing ROS."""
    return expand_profile(yaml.safe_load(Path(path).read_text()),
                          allowed_field_paths=allowed_field_paths)


def profile_digest(profile):
    """Hash the complete expanded profile, independent of mapping key order."""
    expanded = expand_profile(profile)
    payload = json.dumps(expanded, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
