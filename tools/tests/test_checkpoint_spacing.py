"""Check bounded checkpoint spacing and exact replay through the public controller."""

from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import skfmm

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "src/agx_navigation/agx_planning")]

_rclpy_mock = sys.modules.pop("rclpy", None)
try:
    from agx_planning.vector_field import FieldUpdater
    from agx_planning.vector_field import field as vf
    from agx_planning.vector_field.incremental import IncrementalEikonal
finally:
    if _rclpy_mock is not None:
        sys.modules["rclpy"] = _rclpy_mock


@pytest.fixture
def inputs():
    mask = np.zeros((31, 91), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    speed = np.random.default_rng(823).uniform(0.8, 1.2, mask.shape)
    return mask, speed, 8, 15, 0.1


def fresh_state(inputs):
    mask, speed, col, row, resolution = inputs
    phi = np.ones(mask.shape)
    phi[row, col] = -1.0
    return skfmm.TravelTimeState(np.ma.array(phi, mask=mask), speed,
                                 dx=resolution, track_dependencies=True)


@pytest.mark.parametrize("growth,offsets", [
    (1, [0, 10, 20, 30]), (2, [0, 10, 20, 40]), (4, [0, 10, 40, 160]),
])
@pytest.mark.parametrize("count", range(5))
def test_spacing_count_limits_and_exact_snapshots(inputs, growth, offsets, count):
    controller = IncrementalEikonal(10, count, 4 * 1024**2, checkpoint_growth=growth)
    actual = controller.solve(*inputs)
    expected = vf.solve_eikonal(*inputs)
    np.testing.assert_array_equal(actual, expected)
    stats = controller.last_stats
    assert stats["checkpoint_batches"] == offsets[:count]
    assert stats["checkpoint_copies"] == stats["checkpoint_count"] == count
    assert stats["cache_array_bytes"] <= controller.cache_bytes
    assert bool(stats["checkpoint_copy_bytes"]) == bool(count)
    if count:
        assert stats["checkpoint_copy_bytes"] == sum(size for _, size, _ in controller._checkpoints)
        for saved, _, batch in controller._checkpoints:
            controller._state.restore(saved)
            assert controller._state.stats["batches"] == batch
            controller._state.advance()
            np.testing.assert_array_equal(np.ma.filled(controller._state.result(), np.nan), expected)


def test_default_growth_matches_fixed_interval(inputs):
    default = IncrementalEikonal(checkpoint_interval=10)
    explicit = IncrementalEikonal(checkpoint_interval=10, checkpoint_growth=1)
    np.testing.assert_array_equal(default.solve(*inputs), explicit.solve(*inputs))
    assert default.last_stats == explicit.last_stats
    assert default.last_stats["checkpoint_batches"] == [0, 10, 20, 30]


@pytest.mark.parametrize("budget_kind,count", [("zero", 0), ("too_small", 0), ("one", 1), ("two", 2)])
def test_byte_budget_limits_actual_snapshot_copies(inputs, budget_kind, count, monkeypatch):
    state = fresh_state(inputs)
    working = state.stats["working_array_bytes"]
    first = state.stats["checkpoint_array_bytes"]
    state.advance(10)
    second = state.stats["checkpoint_array_bytes"]
    budgets = {"zero": 0, "too_small": working + first - 1,
               "one": working + first, "two": working + first + second}
    copied = []
    checkpoint = skfmm.TravelTimeState.checkpoint

    def capture(self, *args, **kwargs):
        copied.append(self.stats["checkpoint_array_bytes"])
        assert self.stats["working_array_bytes"] + sum(copied) <= budgets[budget_kind]
        return checkpoint(self, *args, **kwargs)

    monkeypatch.setattr(skfmm.TravelTimeState, "checkpoint", capture)
    controller = IncrementalEikonal(10, 4, budgets[budget_kind], checkpoint_growth=4)
    np.testing.assert_array_equal(controller.solve(*inputs), vf.solve_eikonal(*inputs))
    stats = controller.last_stats
    assert stats["checkpoint_copies"] == stats["checkpoint_count"] == len(copied) == count
    assert stats["checkpoint_copy_bytes"] == sum(copied)
    assert stats["cache_array_bytes"] <= budgets[budget_kind]
    if not count:
        assert stats["checkpoint_batches"] == []
        assert stats["reason"] == "cache_budget"


@pytest.mark.parametrize("growth", [1, 4])
@pytest.mark.parametrize("extra", [0, 1])
def test_completion_at_or_before_next_checkpoint_with_tied_groups(growth, extra):
    inputs = (np.zeros((21, 21), dtype=bool), np.ones((21, 21)), 10, 10, 0.1)
    state = fresh_state(inputs)
    state.advance()
    batches = state.stats["batches"]
    assert batches < inputs[0].size // 2
    controller = IncrementalEikonal(batches + extra, 4, checkpoint_growth=growth)
    np.testing.assert_array_equal(controller.solve(*inputs), vf.solve_eikonal(*inputs))
    assert controller.last_stats["checkpoint_batches"] == [0, batches]
    assert controller.last_stats["checkpoint_copies"] == 2


def test_large_integer_growth_finishes_without_counter_overflow(inputs):
    controller = IncrementalEikonal(10, 4, checkpoint_growth=2**100)
    np.testing.assert_array_equal(controller.solve(*inputs), vf.solve_eikonal(*inputs))
    stats = controller.last_stats
    assert stats["checkpoint_batches"] == [0, 10, stats["total_batches"]]
    assert stats["checkpoint_copies"] == 3


@pytest.mark.parametrize("growth,error", [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("4", TypeError)])
def test_growth_requires_a_positive_integer(growth, error):
    with pytest.raises(error):
        IncrementalEikonal(checkpoint_growth=growth)
    with pytest.raises(error):
        FieldUpdater(incremental=True, checkpoint_growth=growth)


def test_repeated_replay_uses_offsets_from_the_reused_prefix(inputs):
    mask, speed, col, row, resolution = inputs
    controller = IncrementalEikonal(10, 4, checkpoint_growth=np.int64(4))
    controller.solve(*inputs)
    previous = 0
    for point, value in [((12, 70), 0.45), ((20, 80), 0.55), ((12, 70), speed[12, 70])]:
        speed = speed.copy()
        speed[point] = value
        request = mask, speed, col, row, resolution
        np.testing.assert_array_equal(controller.solve(*request), vf.solve_eikonal(*request))
        stats = controller.last_stats
        assert stats["status"] == "replay"
        reused = stats["reused_batches"]
        assert reused > previous
        assert reused < stats["first_affected_batch"]
        assert stats["checkpoint_batches"] == [reused + n for n in (0, 10, 40, 160)]
        assert stats["checkpoint_copies"] == 4
        previous = reused


def test_unchanged_inputs_keep_checkpoints_and_report_no_copy_work(inputs):
    controller = IncrementalEikonal(10, 4, checkpoint_growth=4)
    expected = controller.solve(*inputs)
    batches = controller.last_stats["checkpoint_batches"].copy()
    controller.last_stats["checkpoint_batches"][:] = [-1]
    np.testing.assert_array_equal(controller.solve(*inputs), expected)
    assert controller.last_stats["status"] == "unchanged"
    assert controller.last_stats["checkpoint_batches"] == batches
    assert controller.last_stats["checkpoint_copies"] == 0
    assert controller.last_stats["checkpoint_copy_bytes"] == 0


def test_early_exit_fallback_has_no_checkpoint_work(inputs):
    controller = IncrementalEikonal(10, 4, checkpoint_growth=4)
    actual = controller.solve(*inputs, early_exit_target=(15, 20))
    np.testing.assert_array_equal(actual, vf.solve_eikonal(*inputs, early_exit_target=(15, 20)))
    assert controller.last_stats["reason"] == "early_exit"
    assert controller.last_stats["checkpoint_batches"] == []
    assert controller.last_stats["checkpoint_copies"] == controller.last_stats["checkpoint_copy_bytes"] == 0


def test_field_updater_passes_growth_and_keeps_every_field_channel_exact(inputs):
    occupancy = np.where(inputs[0], 100, 0).astype(np.int8)
    cfg = vf.VectorFieldConfig(early_exit_enable=False, inflation_radius=0)
    updater = FieldUpdater(incremental=True, checkpoint_interval=10, checkpoint_growth=4)
    for value in (0, 100, 0):
        occupancy[12:15, 70] = value
        request = (occupancy, 8, 15, 0.1, 0.0, 0.0, cfg)
        actual, expected = updater.update(*request), vf.compute_field(*request)
        assert actual[1] == expected[1]
        for item in fields(vf.VectorFieldResult):
            np.testing.assert_equal(getattr(actual[0], item.name), getattr(expected[0], item.name))
        batch = updater.last_fmm["reused_batches"]
        assert updater.last_fmm["checkpoint_batches"] == [batch + n for n in (0, 10, 40, 160)]


@pytest.mark.parametrize("growth,expected", [(None, [0, 2, 4, 6]), (4, [0, 2, 8, 32])])
def test_cli_records_growth_and_checkpoint_work(tmp_path, growth, expected):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [21, 41]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles: [[0, 0, 41, 1, 100]]
cases:
  - id: spacing
    start: [0.5, 1, 0, 0, 0]
    goal: [1.5, 1, 0]
    updates: [{{id: repeated}}]
""")
    output = tmp_path / "run"
    command = [sys.executable, str(REPO / "tools/bench_replanning.py"),
               "--config", str(suite), "--out", str(output), "--field-only",
               "--field-backend", "incremental", "--checkpoint-interval", "2", "--verify-fields"]
    if growth is not None:
        command += ["--checkpoint-growth", str(growth)]
    subprocess.run(command, env={**os.environ, "PYTHONPATH": str(REPO / "tools")},
                   check=True, capture_output=True, text=True)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["options"]["checkpoint_growth"] == (growth or 1)
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert all(record["verification"]["status"] == "match" for record in records)
    stats = records[0]["fmm_update"]
    assert stats["checkpoint_batches"] == expected
    assert stats["checkpoint_copies"] == 4 and stats["checkpoint_copy_bytes"] > 0
    assert "fmm_update" not in records[1]


def test_cli_rejects_nonpositive_growth(tmp_path):
    result = subprocess.run([sys.executable, str(REPO / "tools/bench_replanning.py"),
                             "--config", "unused", "--out", str(tmp_path / "unused"),
                             "--checkpoint-growth", "0"], capture_output=True, text=True)
    assert result.returncode == 2
    assert "checkpoint growth must be at least one" in result.stderr
