"""The common checks keep outcomes and do not borrow another revision's code."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools/revision_compare/checks.py"
SPEC = importlib.util.spec_from_file_location("revision_compare_checks_under_test", SCRIPT)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_current_kernels_pass_the_shared_behavior_protocol(tmp_path):
    output = tmp_path / "recorded"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(REPO),
                             "--out", str(output)], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((output / "checks.json").read_text())
    assert report["protocol"] == "common-behavior-v1"
    assert report["counts"] == {"pass": 9, "fail": 0, "error": 0, "unsupported": 0}
    assert len(report["harness_sha256"]) == len(report["native_binary_sha256"]) == 64
    assert {item["kind"] for item in report["checks"]} == {
        "unit_geometry", "unit_precision_controlled", "unit_mock_bvp", "compiled_native"}
    for name, path in report["module_paths"].items():
        if name.startswith("agx_planning"):
            assert Path(path).is_relative_to(REPO / "src/agx_navigation/agx_planning")
    assert all("expected" in item and "actual" in item for item in report["checks"])
    assert report["capabilities"]["structured_solve_outcome"]


def test_check_failures_are_recorded_without_preventing_later_checks():
    failure = checks._record("old_behavior", "unit_mock_bvp",
                             lambda: (False, {"committed": True}, {"committed": False}))
    passing = checks._record("next_case", "unit_geometry",
                             lambda: (True, {"cell": None}, {"cell": None}))
    assert failure["status"] == "fail"
    assert failure["actual"] != failure["expected"]
    assert passing["status"] == "pass"


@pytest.mark.parametrize("exception,status", [
    (checks.UnsupportedCheck("API absent"), "unsupported"),
    (ValueError("invalid result"), "error"),
])
def test_unsupported_api_and_broken_check_are_distinct(exception, status):
    def callback():
        raise exception
    record = checks._record("case", "compiled_native", callback)
    assert record["status"] == status
    assert record["check_wall_s"] >= 0


def test_foreign_planning_import_is_rejected_before_any_check(monkeypatch, tmp_path):
    foreign = SimpleNamespace(__file__=str(tmp_path / "another_checkout/agx_planning/field.py"))
    monkeypatch.setitem(sys.modules, "agx_planning.foreign_revision", foreign)
    with pytest.raises(ValueError, match="another revision is already imported"):
        checks._load_modules(REPO)


def test_setup_failure_keeps_a_machine_readable_record(tmp_path):
    output = tmp_path / "bad-run"
    result = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(tmp_path),
                             "--out", str(output)], text=True, capture_output=True)
    assert result.returncode == 2
    record = json.loads((output / "checks.json").read_text())
    assert record["status"] == "setup_error"
    assert "no planning package" in record["error"]


def test_existing_check_output_is_not_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "checks.json"
    sentinel.write_text(' {"retained": true}\n')
    result = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(REPO),
                             "--out", str(output)], text=True, capture_output=True)
    assert result.returncode != 0
    assert sentinel.read_text() == ' {"retained": true}\n'
