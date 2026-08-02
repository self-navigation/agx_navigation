"""Tests for the rollout trace differ.

These are aimed at one thing: a differ that reports "identical" when the traces
are not is worse than no differ at all -- it would retire the reproducibility
question with a false answer. So every test that asserts agreement is paired
with one that asserts the disagreement is caught.
"""

import csv

import pytest

from agx_planning.tuning import trace_diff


def _row(phase="step", **over):
    base = {
        "row": "0", "phase": phase, "sim_time": "1.0", "world_steps": "10",
        "lost_steps": "0", "cmd0": "1", "cmd1": "1", "cmd2": "1", "cmd3": "1",
        "x": "0", "y": "0", "z": "0.18", "yaw": "0",
        "qx": "0", "qy": "0", "qz": "0", "qw": "1",
        "v": "0", "omega": "0", "w0": "0", "w1": "0", "w2": "0", "w3": "0",
        "imu_gz": "nan", "imu_ax": "nan", "imu_ay": "nan",
        "terrain": "rl_patch_0:1,2,0",
    }
    base.update({k: str(v) for k, v in over.items()})
    return base


def _write(tmp_path, name, rows):
    path = tmp_path / name
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_identical_traces_report_no_divergence(tmp_path):
    rows = [_row("reset_settle"), _row(), _row(), _row()]
    a = _write(tmp_path, "a.csv", rows)
    b = _write(tmp_path, "b.csv", rows)
    res = trace_diff.compare(a, b)
    assert res["first_divergence"] is None
    assert res["initial_state_diffs"] == {}


def test_finds_the_first_differing_step_not_a_later_one(tmp_path):
    a_rows = [_row("reset_settle"), _row(x=0), _row(x=1), _row(x=2)]
    b_rows = [_row("reset_settle"), _row(x=0), _row(x=1.5), _row(x=99)]
    res = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                             _write(tmp_path, "b.csv", b_rows))
    idx, diffs = res["first_divergence"]
    assert idx == 1                      # step index, reset rows excluded
    assert diffs["x"] == pytest.approx(0.5)


def test_eps_suppresses_differences_below_it(tmp_path):
    a = _write(tmp_path, "a.csv", [_row(x=1.0)])
    b = _write(tmp_path, "b.csv", [_row(x=1.0 + 1e-12)])
    assert trace_diff.compare(a, b, eps=1e-9)["first_divergence"] is None
    assert trace_diff.compare(a, b, eps=1e-15)["first_divergence"] is not None


def test_terrain_mismatch_is_reported(tmp_path):
    a = _write(tmp_path, "a.csv", [_row(terrain="rl_patch_0:1,2,0")])
    b = _write(tmp_path, "b.csv", [_row(terrain="rl_patch_0:1,2.5,0")])
    _, diffs = trace_diff.compare(a, b)["first_divergence"]
    assert "terrain" in diffs


def test_nan_on_both_sides_is_agreement_but_one_sided_nan_is_not(tmp_path):
    """Unconfigured channels are NaN in both runs by construction; that is not
    a divergence. A NaN appearing in only one run is a real difference."""
    a = _write(tmp_path, "a.csv", [_row(imu_gz="nan")])
    b = _write(tmp_path, "b.csv", [_row(imu_gz="nan")])
    assert trace_diff.compare(a, b)["first_divergence"] is None

    c = _write(tmp_path, "c.csv", [_row(imu_gz=0.5)])
    _, diffs = trace_diff.compare(a, c)["first_divergence"]
    assert diffs["imu_gz"] == float("inf")


def test_variable_reset_length_does_not_count_as_divergence(tmp_path):
    """The refine loop legitimately runs a different number of times. Comparing
    reset rows positionally would flag that as a divergence when it is not."""
    a_rows = [_row("reset_settle"), _row("reset_settle"), _row(x=1)]
    b_rows = [_row("reset_settle"), _row(x=1)]
    res = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                             _write(tmp_path, "b.csv", b_rows))
    assert res["first_divergence"] is None
    assert res["initial_state_diffs"] == {}


def test_differing_initial_state_is_surfaced_separately(tmp_path):
    """A run that starts from a different pose has not 'diverged' -- it was
    never the same experiment. That must not be reported as step-0 divergence."""
    a_rows = [_row("reset_settle", x=0.0), _row(x=1)]
    b_rows = [_row("reset_settle", x=0.05), _row(x=1)]
    res = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                             _write(tmp_path, "b.csv", b_rows))
    assert res["initial_state_diffs"]["x"] == pytest.approx(0.05)
    assert res["first_divergence"] is None


def test_row_counter_is_ignored(tmp_path):
    """`row` is a file offset, not state; a longer reset shifts it harmlessly."""
    a = _write(tmp_path, "a.csv", [_row(row=3)])
    b = _write(tmp_path, "b.csv", [_row(row=9)])
    assert trace_diff.compare(a, b)["first_divergence"] is None


def test_command_divergence_is_distinguishable_from_state_divergence(tmp_path):
    a = _write(tmp_path, "a.csv", [_row(cmd0=1.0)])
    b = _write(tmp_path, "b.csv", [_row(cmd0=1.2)])
    _, diffs = trace_diff.compare(a, b)["first_divergence"]
    assert set(diffs) == {"cmd0"}        # state columns all still agree


def test_dropped_step_shows_in_the_counters(tmp_path):
    """lost_steps is cumulative over the process, so what identifies a drop is
    the counter ADVANCING within one rollout, not its absolute value -- run b
    entering the rollout with a higher lifetime count is not a defect."""
    a = _write(tmp_path, "a.csv", [_row(lost_steps=0), _row(lost_steps=0)])
    b = _write(tmp_path, "b.csv", [_row(lost_steps=4), _row(lost_steps=5)])
    idx, diffs = trace_diff.compare(a, b)["first_divergence"]
    assert idx == 1 and "lost_steps" in diffs


def test_cumulative_counters_are_compared_as_deltas(tmp_path):
    """The world is not restarted between rollouts, so a later run starts at a
    larger sim_time and world_steps. Comparing those raw reported a spurious
    250-step divergence on every single pair -- see the first real diff run."""
    a_rows = [_row(sim_time=10.0, world_steps=1000),
              _row(sim_time=10.1, world_steps=1010)]
    b_rows = [_row(sim_time=99.0, world_steps=9000),
              _row(sim_time=99.1, world_steps=9010)]
    res = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                             _write(tmp_path, "b.csv", b_rows))
    assert res["first_divergence"] is None


def test_a_genuinely_dropped_step_still_shows_after_rebasing(tmp_path):
    """Rebasing must not blind the differ to the thing it was built to catch:
    one run simulating less time than the other."""
    a_rows = [_row(sim_time=10.0), _row(sim_time=10.1), _row(sim_time=10.2)]
    b_rows = [_row(sim_time=99.0), _row(sim_time=99.1), _row(sim_time=99.15)]
    res = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                             _write(tmp_path, "b.csv", b_rows))
    idx, diffs = res["first_divergence"]
    assert idx == 2 and "sim_time" in diffs


def test_growth_profile_timestamps_each_order_of_magnitude(tmp_path):
    a_rows = [_row(x=0), _row(x=0), _row(x=0), _row(x=0)]
    b_rows = [_row(x=0), _row(x=1e-7), _row(x=1e-4), _row(x=0.5)]
    g = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                           _write(tmp_path, "b.csv", b_rows))["growth"]
    assert g["crossings"][1e-9] == 1
    assert g["crossings"][1e-6] == 2
    assert g["crossings"][1e-1] == 3
    assert g["crossings"][1.0] is None
    assert g["final_separation"] == pytest.approx(0.5)


def test_column_onsets_order_separates_command_path_from_solver(tmp_path):
    """The ordering is the diagnosis: a wheel speed that moves before the pose
    means the command reached the plant differently, not that physics differ."""
    a_rows = [_row(w0=1, x=0), _row(w0=1, x=0), _row(w0=1, x=0)]
    b_rows = [_row(w0=1, x=0), _row(w0=2, x=0), _row(w0=2, x=5)]
    on = trace_diff.compare(_write(tmp_path, "a.csv", a_rows),
                            _write(tmp_path, "b.csv", b_rows))["onsets"]
    assert on["w0"] == 1
    assert on["x"] == 2
    assert on["y"] is None
