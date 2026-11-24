"""The collector preserves paired attempts, source isolation and child cleanup."""

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import zipfile

import pytest


REPO = Path(__file__).resolve().parents[2]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_script("collector_under_test", REPO / "tools/compare_revisions.py")
audit = load_script("collector_audit_under_test", REPO / "tools/revision_compare/audit.py")


def suites():
    return [{"suite": "fields", "experiment": "field", "directory": "fields",
             "profile_sha256": "profile", "snapshots": [
                 {"case_id": "route", "snapshot_id": "base", "map_sha256": "base-map", "input_sha256": "base-input"},
                 {"case_id": "route", "snapshot_id": "edit", "map_sha256": "edited-map", "input_sha256": "edited-input"}]},
            {"suite": "plans", "experiment": "planner", "directory": "plans",
             "profile_sha256": "profile", "snapshots": [
                 {"case_id": "route", "snapshot_id": "base", "map_sha256": "base-map", "input_sha256": "base-input"},
                 {"case_id": "route", "snapshot_id": "edit", "map_sha256": "edited-map", "input_sha256": "edited-input"},
                 {"case_id": "route", "snapshot_id": "again", "map_sha256": "edited-map", "input_sha256": "edited-input"}]}]


def test_planned_rows_have_exact_counts_unique_attempts_and_matched_pairs():
    blocks, rows = collector.build_plan(suites(), {"before": "old", "after": "new"},
                                        repeat=2, warmups=1)
    assert len(blocks) == 15
    assert len(rows) == 60
    assert len({block["block_id"] for block in blocks}) == len(blocks)
    identities = {(row["block_id"], row["phase"], row["warmup_index"], row["snapshot_index"])
                  for row in rows}
    assert len(identities) == len(rows)
    paired = defaultdict(list)
    for row in rows:
        paired[row["pair_id"]].append(row)
    for group in paired.values():
        expected = {"before_full", "after_full"}
        if group[0]["experiment"] == "field":
            expected.add("after_incremental")
        assert {row["arm"] for row in group} == expected
        assert len(group) == len(expected)
        assert len({row["map_sha256"] for row in group}) == 1
        assert len({row["profile_sha256"] for row in group}) == 1
        assert len({row["phase"] for row in group}) == 1
    assert sum(row["phase"] == "verify" for row in rows) == 12
    assert sum(row["phase"] == "warmup" for row in rows) == 24
    assert sum(row["phase"] == "measure" for row in rows) == 24


def test_two_arm_measurements_do_not_always_run_the_baseline_first():
    blocks, _ = collector.build_plan(suites()[1:], {"before": "old", "after": "new"},
                                     repeat=2, warmups=0)
    first = {}
    for block in blocks:
        if block["phase"] == "measure":
            first.setdefault(block["repetition"], block["arm"])
    assert Counter(first.values()) == {"before_full": 1, "after_full": 1}


def test_complete_counterbalance_cycle_covers_every_position_equally():
    blocks, _ = collector.build_plan(suites(), {"before": "old", "after": "new"},
                                     repeat=6, warmups=0)
    orders = defaultdict(list)
    for block in blocks:
        if block["phase"] == "measure":
            orders[block["suite"], block["repetition"]].append(block["arm"])
    for suite, arms in (("fields", {"before_full", "after_full", "after_incremental"}),
                        ("plans", {"before_full", "after_full"})):
        for position in range(len(arms)):
            counts = Counter(orders[suite, repetition][position] for repetition in range(6))
            assert set(counts) == arms
            assert len(set(counts.values())) == 1


def test_no_incremental_option_keeps_complete_full_pairs():
    blocks, rows = collector.build_plan(suites(), {"before": "old", "after": "new"},
                                        repeat=1, warmups=0, incremental=False)
    assert {block["backend"] for block in blocks} == {"full"}
    assert len(blocks) == 8
    assert len(rows) == 20
    assert set(Counter(row["pair_id"] for row in rows).values()) == {2}


def test_environment_discards_foreign_python_pip_and_pytest_configuration(monkeypatch, tmp_path):
    removed = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV",
               "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    for name in removed:
        monkeypatch.setenv(name, "foreign-value")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "64")
    python = tmp_path / "work/envs/after/bin/python"
    checkout = tmp_path / "work/checkouts/after"
    env = collector.clean_environment(tmp_path, 2, python=python, repo=checkout)
    for name in removed:
        if name not in ("PYTHONPATH", "VIRTUAL_ENV"):
            assert name not in env
    assert env["VIRTUAL_ENV"] == str(python.parent.parent)
    assert env["PATH"].split(os.pathsep)[0] == str(python.parent)
    assert all(Path(path).is_relative_to(checkout) for path in env["PYTHONPATH"].split(os.pathsep))
    assert all(env[name] == "2" for name in collector.THREADS)
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == env["PYTHONNOUSERSITE"] == "1"
    assert env["PYTHONHASHSEED"] == "0"


def test_collect_cannot_overwrite_an_existing_directory(monkeypatch, tmp_path):
    existing = tmp_path / "data"
    existing.mkdir()
    sentinel = existing / "protocol.json"
    sentinel.write_text("retained\n")
    monkeypatch.setattr(collector, "git", lambda *_args: str(REPO))
    monkeypatch.setattr(collector, "resolve_revision", lambda _repo, value: value)
    args = SimpleNamespace(before="old", after="new", out=str(existing))
    with pytest.raises(FileExistsError):
        collector.collect(args)
    assert sentinel.read_text() == "retained\n"
    assert list(existing.iterdir()) == [sentinel]


def _running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/stat")
    return not status.exists() or status.read_text().split()[2] != "Z"


@pytest.mark.skipif(os.name != "posix", reason="collector supports Linux/WSL process groups")
def test_timeout_kills_a_child_even_after_its_parent_exits_on_term(tmp_path):
    pid_file = tmp_path / "child.pid"
    child = ("import signal,sys,time; from pathlib import Path; "
             "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
             "Path(sys.argv[1]).write_text(str(__import__('os').getpid())); time.sleep(60)")
    parent = ("import subprocess,sys,time; "
              "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(60)")
    commands = collector.Commands(tmp_path)
    child_pid = None
    try:
        record = commands.run("timeout", [sys.executable, "-c", parent, child, str(pid_file)],
                              cwd=tmp_path, env=os.environ.copy(), timeout=.6)
        assert record["returncode"] == 124 and record["timed_out"]
        assert pid_file.exists(), "child did not initialize before the timeout"
        child_pid = int(pid_file.read_text())
        for _ in range(20):
            if not _running(child_pid):
                break
            time.sleep(.025)
        assert not _running(child_pid), "timed-out child survived after the parent exited"
        recorded = json.loads((tmp_path / "commands.json").read_text())
        assert len(recorded) == 1 and recorded[0]["returncode"] == 124
        assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 1
    finally:
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None and _running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_archive_keeps_raw_data_and_hashes_but_excludes_build_work(tmp_path):
    (tmp_path / "work/checkouts/after").mkdir(parents=True)
    (tmp_path / "work/envs/after").mkdir(parents=True)
    (tmp_path / "work/wheels").mkdir(parents=True)
    for name in ("work/checkouts/after/source.py", "work/envs/after/native.so", "work/wheels/cache.whl"):
        (tmp_path / name).write_bytes(b"reproducible work")
    (tmp_path / "runs").mkdir()
    (tmp_path / "protocol.json").write_text('{"schema_version":1}\n')
    (tmp_path / "runs/results.jsonl").write_text('{"status":"failed"}\n')
    collector.archive_collection(tmp_path)
    collector.archive_collection(tmp_path)
    with zipfile.ZipFile(tmp_path / "raw-data.zip") as archive:
        assert set(archive.namelist()) == {"protocol.json", "runs/results.jsonl", "SHA256SUMS"}
        assert archive.testzip() is None
        for line in archive.read("SHA256SUMS").decode().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_missing_rows_after_timeout_are_retained_as_unmeasured_attempts(tmp_path):
    blocks, rows = collector.build_plan(suites()[:1], {"before": "old", "after": "new"},
                                        repeat=1, warmups=0, incremental=False)
    block = {**blocks[0], "returncode": 124, "timed_out": True}
    planned = [row for row in rows if row["block_id"] == block["block_id"]]
    collector.save_plan(tmp_path, [block], planned)
    result = audit.audit_collection(tmp_path)
    assert result["expected"] == result["missing"] == 2
    assert result["recorded"] == 0
    assert not result["data_complete"]
    recorded = [json.loads(line) for line in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(recorded) == 2
    assert all(row["status"] == "timeout" and not row["measured"] for row in recorded)
    assert {row["pair_id"] for row in recorded} == {row["pair_id"] for row in planned}


@pytest.mark.parametrize("failed", [0, 3])
def test_native_test_command_accepts_the_historical_integer_result(monkeypatch, tmp_path, failed):
    class Recorder:
        calls = []

        def run(self, name, command, **_options):
            self.calls.append((name, command))
            return {"returncode": 0, "command": list(map(str, command))}

    recorder = Recorder()
    collector.run_tests(tmp_path, tmp_path / "harness", {"before": (tmp_path, Path(sys.executable))},
                        recorder, SimpleNamespace(threads=1, timeout=10))
    native = next(command for name, command in recorder.calls if name.endswith("native-doctests"))
    monkeypatch.setitem(sys.modules, "skfmm", SimpleNamespace(test=lambda: failed))
    with pytest.raises(SystemExit) as terminal:
        exec(native[-1], {})
    assert bool(terminal.value.code) == bool(failed)


def test_preparation_freezes_identical_inputs_for_both_experiments(tmp_path):
    import numpy as np
    output = tmp_path / "frozen"
    script = REPO / "tools/revision_compare/prepare.py"
    scenario = "config/replanning/synthetic.yaml"
    result = subprocess.run([sys.executable, str(script), "--repo", str(REPO), "--out", str(output),
                             "--field-config", scenario, "--planner-config", scenario],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    index = json.loads((output / "index.json").read_text())
    assert len(index) == 2
    assert [entry["experiment"] for entry in index] == ["field", "planner"]
    assert index[0]["profile_sha256"] == index[1]["profile_sha256"]
    assert index[0]["snapshots"] == index[1]["snapshots"]
    all_arrays = []
    for entry in index:
        directory = output / entry["directory"]
        assert hashlib.sha256((directory / "inputs.npz").read_bytes()).hexdigest() == entry["input_file_sha256"]
        assert (directory / "scenario.yaml").read_bytes() == (REPO / scenario).read_bytes()
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["profile"]["field_path"] == "packed_float32"
        assert not manifest["profile"]["field"]["early_exit_enable"]
        with np.load(directory / "inputs.npz", allow_pickle=False) as arrays:
            assert len(arrays.files) == len(entry["snapshots"])
            all_arrays.append({name: arrays[name] for name in arrays.files})
    assert all_arrays[0].keys() == all_arrays[1].keys()
    for name in all_arrays[0]:
        np.testing.assert_array_equal(all_arrays[0][name], all_arrays[1][name])


def test_preparation_rejects_config_paths_outside_the_selected_head(tmp_path):
    # Some CI environments put pytest's tmp_path inside the checkout.
    with tempfile.TemporaryDirectory(prefix="compare-outside-", dir=REPO.parent) as directory:
        external = Path(directory) / "outside.yaml"
        external.write_text("schema_version: 1\n")
        result = subprocess.run([sys.executable, str(REPO / "tools/revision_compare/prepare.py"),
                                 "--repo", str(REPO), "--out", str(tmp_path / "frozen"),
                                 "--field-config", str(external)], text=True, capture_output=True)
    assert result.returncode != 0
    assert "config must be a committed path inside the selected head" in result.stderr
    assert not (tmp_path / "frozen/index.json").exists()
