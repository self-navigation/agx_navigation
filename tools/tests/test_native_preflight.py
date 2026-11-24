"""Reject installed native behavior that predates complete-state reuse."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import skfmm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from replanning_baseline.native_check import check_incremental_native


def test_installed_native_reuses_the_complete_state_without_another_advance(monkeypatch):
    original = skfmm.TravelTimeState
    calls = []

    class ObservedState(original):
        def advance(self, *args, **kwargs):
            calls.append("advance")
            return super().advance(*args, **kwargs)

        def checkpoint(self, *args, **kwargs):
            pytest.fail("the preflight must not create a checkpoint")

        def replay(self, speed, mask, checkpoints):
            calls.append("replay")
            assert checkpoints == []
            return super().replay(speed, mask, checkpoints)

    monkeypatch.setattr(skfmm, "TravelTimeState", ObservedState)
    report = check_incremental_native()
    assert report["complete_state_reuse"] == "passed"
    assert report["replay"]["reason"] == "complete_state_reused"
    assert calls == ["advance", "replay"]


@pytest.mark.parametrize("failure", ["old_replay", "incomplete", "batch_change", "field_change"])
def test_rejected_behavior_has_an_actionable_same_python_rebuild_command(monkeypatch, failure):
    class WrongState:
        def __init__(self, *args, **kwargs):
            self.replayed = False

        def advance(self):
            return True

        @property
        def stats(self):
            return {"complete": not (self.replayed and failure == "incomplete"),
                    "batches": 10 + int(self.replayed and failure == "batch_change")}

        def result(self):
            return np.array([float(self.replayed and failure == "field_change")])

        def replay(self, *args):
            self.replayed = True
            if failure == "old_replay":
                return {"status": "full_required", "reason": "no_safe_checkpoint"}
            return {"status": "replay", "reason": "complete_state_reused"}

    monkeypatch.setattr(skfmm, "TravelTimeState", WrongState)
    with pytest.raises(ValueError, match="Incremental native preflight failed") as caught:
        check_incremental_native()
    message = str(caught.value)
    assert sys.executable in message
    assert "--no-build-isolation --no-deps --force-reinstall ./depend/scikit-fmm" in message


def test_missing_state_api_requires_rebuild(monkeypatch):
    monkeypatch.delattr(skfmm, "TravelTimeState")
    with pytest.raises(ValueError, match="Rebuild the installed scikit-fmm extension"):
        check_incremental_native()


def test_cli_rejects_old_native_before_creating_output(tmp_path):
    output = tmp_path / "rejected"
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import skfmm
original = skfmm.TravelTimeState.replay
def old_replay(self, *args):
    return {'status': 'full_required', 'reason': 'no_safe_checkpoint'}
skfmm.TravelTimeState.replay = old_replay
from bench_replanning import main
sys.argv = ['bench_replanning', '--config', 'unused.yaml', '--out', sys.argv[2],
            '--field-only', '--field-backend', 'incremental']
raise SystemExit(main())
"""
    completed = subprocess.run([sys.executable, "-c", script, str(REPO / "tools"), str(output)],
                               cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "expected complete_state_reused" in completed.stderr
    assert "unused.yaml" not in completed.stderr
    assert not output.exists()


@pytest.mark.parametrize("backend", ["full", "cached", "incremental"])
def test_cli_records_success_only_for_incremental_and_keeps_legacy_backends(tmp_path, backend):
    suite = tmp_path / "suite.yaml"
    suite.write_text(f"""schema_version: 1
profile: {REPO / 'config/replanning/baseline_profile.yaml'}
map:
  grid:
    shape: [21, 31]
    resolution: 0.1
    origin: [0, 0, 0]
    rectangles: [[0, 0, 31, 1, 100]]
cases:
  - id: preflight
    start: [0.5, 1.0, 0, 0, 0]
    goal: [1.5, 1.0, 0]
""")
    output = tmp_path / backend
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import skfmm
if sys.argv[4] != 'incremental':
    del skfmm.TravelTimeState
from bench_replanning import main
sys.argv = ['bench_replanning', '--config', sys.argv[2], '--out', sys.argv[3],
            '--field-only', '--field-backend', sys.argv[4]]
raise SystemExit(main())
"""
    completed = subprocess.run([sys.executable, "-c", script, str(REPO / "tools"),
                                str(suite), str(output), backend],
                               cwd=REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    checks = manifest["environment"].get("native_checks")
    if backend == "incremental":
        assert checks["complete_state_reuse"] == "passed"
        assert checks["replay"]["reason"] == "complete_state_reused"
        from summarize_replanning import summarize
        assert summarize(output)["environment"]["native_checks"] == checks
    else:
        assert checks is None
    assert (output / "COMPLETE").exists()
