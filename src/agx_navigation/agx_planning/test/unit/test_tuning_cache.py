"""Tests for the resumable-evaluation cache.

The claim being tested is specific: an interrupted tuning run, resumed, reaches
the same place a single uninterrupted run would -- WITHOUT re-driving Gazebo for
anything it already measured. Both halves matter. If resume re-measures, an
interrupted run costs double; if resume diverges, the search silently explores a
different problem than the one on disk.

Pure module: json + math, no ROS, no Gazebo.
"""

import json

import pytest

from agx_planning.tuning.cache import EvalCache, quantize
from agx_planning.tuning.simplex import minimize


def rosenbrock(x):
    return float((1 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)


def test_quantize_is_stable_under_floating_point_noise():
    a = 0.1 + 0.2
    assert quantize([a, 1.0]) == quantize([0.3, 1.0])


def test_records_every_evaluation_to_disk(tmp_path):
    path = str(tmp_path / "c.jsonl")
    c = EvalCache(path, key={"trajs": ["a"]})
    c.write_header()
    f = c.wrap(rosenbrock)
    for x in ([0.0, 0.0], [1.0, 1.0], [0.5, 0.25]):
        f(x)

    lines = [json.loads(l) for l in open(path) if l.strip()]
    assert lines[0]["_meta"] == {"trajs": ["a"]}
    assert len(lines) == 4                       # meta + 3 evaluations
    assert lines[2]["fx"] == pytest.approx(0.0)  # rosenbrock optimum


def test_repeated_point_is_served_from_cache(tmp_path):
    calls = []

    def f(x):
        calls.append(tuple(x))
        return rosenbrock(x)

    c = EvalCache(str(tmp_path / "c.jsonl"))
    g = c.wrap(f)
    g([0.3, 0.4])
    g([0.3, 0.4])
    g([0.3, 0.4])
    assert len(calls) == 1, "cached point was re-measured"


def test_resume_matches_an_uninterrupted_run_and_costs_nothing(tmp_path):
    """The whole contract, in one test."""
    path = str(tmp_path / "c.jsonl")
    x0, step = [-1.2, 1.0], [0.4, 0.4]

    reference = minimize(rosenbrock, x0=x0, step=step, max_evals=60)

    # First leg: stopped early, as if the VM went down.
    c1 = EvalCache(path)
    c1.write_header()
    minimize(c1.wrap(rosenbrock), x0=x0, step=step, max_evals=25)
    partial = len(c1.entries)
    assert partial == 25

    # Second leg: fresh process, same cache file.
    fresh_calls = []

    def counted(x):
        fresh_calls.append(tuple(x))
        return rosenbrock(x)

    c2 = EvalCache(path)
    assert c2.load() == partial
    resumed = minimize(c2.wrap(counted), x0=x0, step=step, max_evals=60)

    # Same answer as never having been interrupted...
    assert resumed.x == pytest.approx(reference.x)
    assert resumed.fx == pytest.approx(reference.fx)
    # ...and the replayed prefix cost no fresh evaluations at all.
    assert len(fresh_calls) == 60 - partial


def test_torn_final_line_does_not_block_resume(tmp_path):
    """A process killed mid-write leaves a partial JSON line."""
    path = str(tmp_path / "c.jsonl")
    c = EvalCache(path)
    c.write_header()
    f = c.wrap(rosenbrock)
    f([0.1, 0.2])
    f([0.3, 0.4])
    with open(path, "a") as fh:
        fh.write('{"x": [0.5, 0.6], "fx": 1.2')   # no newline, no closing brace

    c2 = EvalCache(path)
    assert c2.load() == 2


def test_cache_from_a_different_setup_is_refused(tmp_path):
    path = str(tmp_path / "c.jsonl")
    EvalCache(path, key={"trajs": ["a", "b"], "seed": 0}).write_header()

    other = EvalCache(path, key={"trajs": ["c"], "seed": 0})
    with pytest.raises(ValueError, match="different setup"):
        other.load()

    # ...but the same setup resumes fine.
    same = EvalCache(path, key={"trajs": ["a", "b"], "seed": 0})
    assert same.load() == 0


def test_best_returns_the_minimum(tmp_path):
    c = EvalCache(str(tmp_path / "c.jsonl"))
    f = c.wrap(rosenbrock)
    for x in ([0.0, 0.0], [1.0, 1.0], [2.0, 2.0]):
        f(x)
    x, fx = c.best()
    assert x == [1.0, 1.0] and fx == pytest.approx(0.0)


def test_detail_is_captured_per_evaluation_not_shared(tmp_path):
    """Each record must carry ITS OWN breakdown.

    tune_tvlqr hands the breakdown over through a mutable dict that the
    objective refills each call, so a stale or shared reference would write the
    same numbers onto every row and quietly flatten the landscape plot.
    """
    path = str(tmp_path / "c.jsonl")
    c = EvalCache(path)
    c.write_header()
    inflight = {}

    def f(x):
        inflight.clear()
        inflight["per_traj"] = {"t1": float(x[0]), "t2": float(x[1])}
        return float(x[0] + x[1])

    g = c.wrap(f, detail=lambda x: dict(inflight))
    g([1.0, 2.0])
    g([3.0, 4.0])

    rows = [json.loads(l) for l in open(path) if l.strip()][1:]
    assert rows[0]["per_traj"] == {"t1": 1.0, "t2": 2.0}
    assert rows[1]["per_traj"] == {"t1": 3.0, "t2": 4.0}


def test_detail_fields_are_persisted(tmp_path):
    """Per-trajectory breakdown must survive, so a finished run can be analysed
    without re-driving anything."""
    path = str(tmp_path / "c.jsonl")
    c = EvalCache(path)
    c.write_header()
    f = c.wrap(rosenbrock, detail=lambda x: {"per_traj": {"t1": 0.5, "t2": 1.5}})
    f([0.2, 0.3])
    rec = [json.loads(l) for l in open(path) if l.strip()][-1]
    assert rec["per_traj"] == {"t1": 0.5, "t2": 1.5}
