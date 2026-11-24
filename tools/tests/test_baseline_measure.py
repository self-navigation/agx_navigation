import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

# The unit suite installs a ROS mock without an import spec.
_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.pmp_planner import PlannerConfig, parse_field_array
    from agx_planning.pmp_planner import shooting_solver
    from agx_planning.vector_field import field as vf
    from replanning_baseline.inputs import cell, load_suite
    from replanning_baseline.measure import (
        measure_field,
        measure_rollout,
        measure_snapshot,
    )
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.fixture
def inputs():
    profile, snapshots, _ = load_suite(
        REPO / "config/replanning/synthetic.yaml", ["straight"]
    )
    return profile, snapshots


def test_field_instrumentation_preserves_production_output(inputs):
    profile, snapshots = inputs
    metadata, occupancy = snapshots[0]
    observed, stages = measure_field(occupancy, metadata, profile)
    geometry = metadata["geometry"]
    reference, _ = vf.compute_field(
        occupancy,
        *cell(metadata["goal"], geometry),
        geometry["resolution"],
        *geometry["origin"][:2],
        vf.VectorFieldConfig(**profile["field"]),
        *cell(metadata["start"], geometry),
        profile["occupancy_threshold"],
        profile["allow_unknown"]
    )
    np.testing.assert_array_equal(observed.travel_time, reference.travel_time)
    np.testing.assert_array_equal(observed.grad_x, reference.grad_x)
    assert stages["compute_field_s"] >= sum(
        stages[k] for k in ("edt_s", "speed_s", "fmm_s", "gradient_s")
    )


def test_disconnected_start_skips_planner(inputs):
    profile, snapshots = inputs
    metadata, occupancy = snapshots[2]
    record, _ = measure_snapshot(occupancy, metadata, profile)
    assert record["status"] == "start_unreached"
    assert record["solves"] == []
    assert record["travel_time_at_start"] is None


@pytest.mark.parametrize("suite, case", [
    ("synthetic.yaml", "straight"),
    ("floor6_smoke.yaml", "floor6_straight"),
])
def test_completed_rollout_includes_terminal_raw_state(suite, case):
    profile, snapshots, _ = load_suite(REPO / "config/replanning" / suite, [case])
    metadata, occupancy = snapshots[0]
    record, arrays = measure_snapshot(occupancy, metadata, profile)
    assert record["status"] == "completed"
    n = len(arrays["wheel_cmds"])
    assert arrays["states"].shape == (n + 1, 5)
    np.testing.assert_allclose(arrays["states"][:-1, :3], arrays["poses"], atol=1e-6)
    np.testing.assert_array_equal(arrays["states"][0], metadata["start"])
    assert record["goal_position_error_m"] < profile["planner"]["goal_tolerance_xy"]
    assert record["goal_heading_error_rad"] < profile["planner"]["goal_tolerance_th"]
    assert np.max(np.abs(arrays["states"][-1, 3:])) <= 0.5
    assert record["nominal_duration_s"] == pytest.approx(
        n / profile["planner"]["control_rate"]
    )
    assert len(record["chunk_ready_s"]) == record["chunks"]
    assert (
        record["total_wall_s"] >= record["planner_wall_s"] + record["compute_field_s"]
    )


def test_failed_solve_cannot_replay_cached_segment(inputs, monkeypatch):
    profile, snapshots = inputs
    metadata, occupancy = snapshots[0]
    result, _ = measure_field(occupancy, metadata, profile)
    grid = parse_field_array(
        vf.pack_field_array(result), PlannerConfig(**profile["planner"])
    )
    original = shooting_solver.solve_bvp
    calls = []

    def fail_after_first(*args, **kwargs):
        if calls:
            raise RuntimeError("injected BVP failure")
        solution = original(*args, **kwargs)
        assert solution.success
        calls.append(solution)
        return solution

    monkeypatch.setattr(shooting_solver, "solve_bvp", fail_after_first)
    record, arrays = measure_rollout(grid, metadata, profile)
    assert record["status"] == "bvp_failed"
    assert record["chunks"] == 1
    assert len(record["solves"]) == 2
    assert record["solves"][1]["success"] is False
    assert len(arrays["states"]) == len(arrays["wheel_cmds"]) + 1
    assert record["planner_wall_s"] >= sum(s["wall_s"] for s in record["solves"])


def test_clean_process_cli_and_output_contract(tmp_path):
    output = tmp_path / "run"
    script = """
import sys
from bench_replanning import main
sys.argv = ['bench_replanning', '--config', sys.argv[1], '--out', sys.argv[2],
            '--case', 'already_at_goal', '--warmups', '1', '--repeat', '2', '--save-fields']
assert main() == 0
assert 'rclpy' not in sys.modules
assert 'torch' not in sys.modules
"""
    import os

    env = {**os.environ, "PYTHONPATH": str(REPO / "tools")}
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(REPO / "config/replanning/synthetic.yaml"),
            str(output),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    records = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    assert len(records) == 3 and sum(r["warmup"] for r in records) == 1
    assert manifest["profile"]["planner"]["N"] == 40
    assert manifest["environment"]["thread_environment"]["OPENBLAS_NUM_THREADS"] == "1"
    assert (output / "COMPLETE").exists()
    assert records[-1]["status"] == "completed"
    assert records[-1]["solves"] == []
    with np.load(output / records[-1]["artifact"], allow_pickle=False) as arrays:
        assert arrays["states"].shape == (1, 5)
        assert arrays["wheel_cmds"].shape == (0, 2)
        assert arrays["travel_time"].shape == (61, 101)
