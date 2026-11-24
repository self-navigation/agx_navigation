"""Focused stopping reruns preserve the old data and fail on false success."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("stopping_check", REPO / "tools/check_planner_stopping.py")
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def collection(tmp_path):
    root = tmp_path / "previous"
    for suite, pairs in check.SUITES.items():
        directory = root / "inputs" / suite
        write_json(directory / "manifest.json", {"snapshots": [
            {"case_id": case, "snapshot_id": snapshot} for case, snapshot in pairs]})
        (directory / "inputs.npz").write_bytes(b"saved inputs, read only")
    return root


def fake_workers(monkeypatch, *, change=None, returncode=0):
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        if returncode:
            return SimpleNamespace(returncode=returncode)
        out = Path(command[command.index("--out") + 1])
        out.mkdir()
        pairs = check.SUITES[out.name]
        rows = [{"case_id": case, "snapshot_id": snapshot, "status": "completed",
                 "phase": "measure", "repetition": 0, "warmup_index": None,
                 "experiment": "planner", "backend": "full",
                 "evaluation": {"common_numerical_acceptance": True}}
                for case, snapshot in pairs]
        if change:
            change(rows)
        (out / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(out / "COMPLETE", {"expected_rows": len(pairs), "recorded_rows": len(rows),
                                      "verification_failures": 0})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(check.subprocess, "run", run)
    return commands


def args(collection, tmp_path):
    return SimpleNamespace(collection=str(collection), out=str(tmp_path / "new"), python=sys.executable)


def test_only_six_saved_planner_inputs_run_with_current_worker(collection, tmp_path, monkeypatch):
    before = {p.relative_to(collection): p.read_bytes() for p in collection.rglob('*') if p.is_file()}
    commands = fake_workers(monkeypatch)
    options = args(collection, tmp_path)
    assert check.run(options) == 0
    summary = json.loads((Path(options.out) / "summary.json").read_text())
    assert summary["status"] == "passed"
    assert summary["expected_rows"] == len(summary["results"]) == 6
    assert sum(row["required"] for row in summary["results"]) == 3
    assert len(commands) == 2
    for command, kwargs in commands:
        assert command[1] == str(REPO / "tools/revision_compare/worker.py")
        assert command[command.index("--repo") + 1] == str(REPO)
        assert command[command.index("--phase") + 1] == "measure"
        assert command[command.index("--warmups") + 1] == "0"
        assert command[command.index("--backend") + 1] == "full"
        assert kwargs["cwd"] == REPO
        assert all(kwargs["env"][name] == "1" for name in check.THREADS)
    assert before == {p.relative_to(collection): p.read_bytes() for p in collection.rglob('*') if p.is_file()}


@pytest.mark.parametrize("case", ["straight", "floor6_straight", "already_at_goal"])
def test_completed_label_cannot_pass_failed_independent_acceptance(collection, tmp_path, monkeypatch, case):
    def change(rows):
        for row in rows:
            if row["case_id"] == case and row["snapshot_id"] == "base":
                row["evaluation"]["common_numerical_acceptance"] = False
    fake_workers(monkeypatch, change=change)
    options = args(collection, tmp_path)
    assert check.run(options) == 1
    summary = json.loads((Path(options.out) / "summary.json").read_text())
    assert summary["status"] == "acceptance_failed"
    assert summary["failed_required_cases"] == [case + "/base"]
    assert len(summary["results"]) == 6


def test_other_outcomes_are_retained_without_requiring_failure(collection, tmp_path, monkeypatch):
    def change(rows):
        for row in rows:
            if row["snapshot_id"] in ("leaf", "closed"):
                row["status"] = "bvp_failed" if row["snapshot_id"] == "leaf" else "start_unreached"
                row["evaluation"]["common_numerical_acceptance"] = False
    fake_workers(monkeypatch, change=change)
    options = args(collection, tmp_path)
    assert check.run(options) == 0
    summary = json.loads((Path(options.out) / "summary.json").read_text())
    outcomes = {row["snapshot_id"]: row["status"] for row in summary["results"]}
    assert outcomes["leaf"] == "bvp_failed"
    assert outcomes["closed"] == "start_unreached"
    assert outcomes["reopened"] == "completed"


@pytest.mark.parametrize("problem", ["missing", "duplicate", "warmup"])
def test_incomplete_or_wrong_records_are_collection_errors(collection, tmp_path, monkeypatch, problem):
    def change(rows):
        if problem == "missing":
            rows.pop()
        elif problem == "duplicate":
            rows[-1] = rows[0]
        else:
            rows[0]["phase"] = "warmup"
    fake_workers(monkeypatch, change=change)
    options = args(collection, tmp_path)
    assert check.run(options) == 2
    assert json.loads((Path(options.out) / "summary.json").read_text())["status"] == "collection_error"


def test_worker_setup_failure_is_distinct_from_acceptance_failure(collection, tmp_path, monkeypatch):
    fake_workers(monkeypatch, returncode=2)
    options = args(collection, tmp_path)
    assert check.run(options) == 2
    summary = json.loads((Path(options.out) / "summary.json").read_text())
    assert summary["status"] == "collection_error"
    assert summary["commands"][0]["returncode"] == 2
    assert "worker failed" in summary["error"]


def test_missing_after_venv_is_reported_before_starting_workers(collection, tmp_path, monkeypatch):
    commands = fake_workers(monkeypatch)
    options = args(collection, tmp_path)
    options.python = None
    assert check.run(options) == 2
    summary = json.loads((Path(options.out) / "summary.json").read_text())
    assert "After venv is unavailable" in summary["error"]
    assert commands == []


def test_existing_output_and_previous_collection_cannot_be_overwritten(collection, tmp_path):
    options = args(collection, tmp_path)
    output = Path(options.out)
    output.mkdir()
    sentinel = output / "summary.json"
    sentinel.write_text("old result")
    with pytest.raises(FileExistsError):
        check.run(options)
    assert sentinel.read_text() == "old result"
    options.out = str(collection / "new")
    with pytest.raises(ValueError, match="outside"):
        check.run(options)
    assert not (collection / "new").exists()


def test_worker_environment_cannot_import_foreign_python_paths(monkeypatch):
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        monkeypatch.setenv(name, "foreign")
    environment = check.worker_environment()
    assert not any(name in environment for name in (
        "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV"))
    assert environment["PYTHONDONTWRITEBYTECODE"] == environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONHASHSEED"] == "0"
