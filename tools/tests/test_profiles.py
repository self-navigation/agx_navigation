"""The benchmark and model runner share one expanded numerical profile."""

from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "src/agx_navigation/agx_planning"
sys.path[:0] = [str(REPO / "tools"), str(CORE)]
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.profiles import (expand_profile, field_config, load_profile,
                                      planner_config, profile_digest)
    from replanning_baseline.inputs import digest, load_profile as load_benchmark_profile
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


PROFILE_PATH = REPO / "config/replanning/baseline_profile.yaml"


def test_frozen_profile_keeps_benchmark_values_and_digest():
    source = yaml.safe_load(PROFILE_PATH.read_text())
    core = load_profile(PROFILE_PATH)
    benchmark = load_benchmark_profile(PROFILE_PATH)
    assert core == benchmark == source
    assert profile_digest(core) == digest(benchmark)
    assert asdict(planner_config(core)) == core["planner"]
    assert asdict(field_config(core)) == core["field"]


def test_expansion_and_factories_do_not_share_mutable_settings():
    source = load_profile(PROFILE_PATH)
    source["notes"] = {"tags": ["frozen"]}
    del source["planner"]["w_h"]
    expanded = expand_profile(source)
    assert "w_h" not in source["planner"]
    assert expanded["planner"]["w_h"] == 5.0
    source["notes"]["tags"].append("edited")
    assert expanded["notes"]["tags"] == ["frozen"]
    first = planner_config(expanded)
    second = planner_config(expanded)
    first.w_h = 100.0
    assert second.w_h == expanded["planner"]["w_h"] == 5.0
    first_field = field_config(expanded)
    first_field.centring = False
    assert field_config(expanded).centring is True


def test_raw_float64_is_explicit_and_benchmark_stays_packed(tmp_path):
    profile = load_profile(PROFILE_PATH)
    packed_digest = profile_digest(profile)
    profile["field_path"] = "raw_float64"
    path = tmp_path / "raw.yaml"
    path.write_text(yaml.safe_dump(profile))
    assert load_profile(path)["field_path"] == "raw_float64"
    assert profile_digest(profile) != packed_digest
    with pytest.raises(ValueError, match="packed_float32"):
        load_benchmark_profile(path)


@pytest.mark.parametrize("section,key,value", [
    (None, "schema_version", True),
    (None, "pin_tolerance", float("nan")),
    (None, "pin_tolerance", -1.0),
    (None, "allow_unknown", "true"),
    (None, "occupancy_threshold", 101),
    (None, "field_path", "automatic"),
    ("planner", "control_rate", float("inf")),
    ("planner", "dt_segment", 0.15),
    ("planner", "N", 40.0),
    ("planner", "bvp_max_nodes", 2),
    ("planner", "wheel_radius", 0.0),
    ("planner", "mode", "online"),
    ("planner", "not_a_setting", 1),
    ("field", "bottleneck_align", True),
    ("field", "speed_v_min", 0.0),
    ("field", "speed_v_max", 0.05),
    ("field", "speed_profile", "other"),
    ("field", "early_exit_margin", -1.0),
])
def test_invalid_profile_fails_before_starting_run(section, key, value):
    profile = load_profile(PROFILE_PATH)
    target = profile if section is None else profile[section]
    target[key] = value
    with pytest.raises(ValueError):
        expand_profile(profile)


def test_digest_includes_defaults_and_all_numerical_inputs():
    profile = load_profile(PROFILE_PATH)
    original = profile_digest(profile)
    reordered = {key: profile[key] for key in reversed(list(profile))}
    assert profile_digest(reordered) == original
    for section, key in [("planner", "gamma_wheel"), ("planner", "fp_margin"),
                         ("field", "smooth_T_sigma"), ("field", "centring_cap")]:
        changed = deepcopy(profile)
        changed[section][key] *= 2
        assert profile_digest(changed) != original
    # Explicit defaults and omitted defaults identify the same settings.
    del profile["planner"]["wheel_radius"]
    assert profile_digest(profile) == original


def test_profile_import_needs_no_ros_in_clean_process():
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; from agx_planning.profiles import load_profile; "
         f"load_profile({str(PROFILE_PATH)!r}); "
         "assert not any(name == 'rclpy' or name.startswith('rclpy.') "
         "for name in sys.modules)"],
        env={**os.environ, "PYTHONPATH": str(CORE)},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
