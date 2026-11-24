#!/usr/bin/env python3
"""Collect matched before/after kernel data and correctness evidence without ROS.

Only the Python standard library is needed to start. Separate clean worktrees,
virtual environments, and freshly built native wheels are created for both
revisions. No command switches or edits the caller's checkout.
"""

import argparse
from contextlib import suppress
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

BASELINE = "e395ad967f414b21bf24962397ce4b486f8aec0e"
MODULES = ("depend/scikit-fmm", "src/rudn-ordjo-building")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def resolve_revision(repo, value):
    # End option parsing before a user-supplied ref and pin it once.
    return git(repo, "rev-parse", "--verify", "--end-of-options", value + "^{commit}")


def clean_environment(root, threads, python=None, repo=None):
    env = dict(os.environ)
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV",
                 "PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        env.pop(name, None)
    env.update({name: str(threads) for name in THREADS})
    env.update(PYTHONNOUSERSITE="1", PYTHONHASHSEED="0", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONPYCACHEPREFIX=str(root / "work/pycache"), TMPDIR=str(root / "work/tmp"))
    if python is not None:
        env["PATH"] = str(Path(python).parent) + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = str(Path(python).parent.parent)
    if repo is not None:
        env["PYTHONPATH"] = os.pathsep.join(str(repo / path) for path in
            ("tools", "src/agx_navigation/agx_planning", "src/rudn-ordjo-building"))
    return env


class Commands:
    def __init__(self, root):
        self.root = Path(root)
        self.records = []

    def run(self, name, command, *, cwd, env, timeout, required=False):
        """Save failures too, and terminate the whole child group on timeout."""
        log = self.root / "logs" / f"{len(self.records):04d}-{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        record = {"name": name, "command": list(map(str, command)), "cwd": str(cwd),
                  "started_utc": utc(), "log": str(log.relative_to(self.root)),
                  "timeout_s": timeout, "timed_out": False}
        print(f"[{record['started_utc']}] {name}", flush=True)
        begin = time.monotonic()
        interrupted = False
        with log.open("wb") as stream:
            process = subprocess.Popen(record["command"], cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                returncode = process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                interrupted = isinstance(error, KeyboardInterrupt)
                record["timed_out"] = not interrupted
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                # The parent may exit before a child that ignores SIGTERM.
                # Always finish this process group, then reap the parent.
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                returncode = 130 if interrupted else 124
        record.update(returncode=returncode, finished_utc=utc(), wall_s=time.monotonic() - begin)
        self.records.append(record)
        write_json(self.root / "commands.json", self.records)
        with (self.root / "observations.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if interrupted:
            raise KeyboardInterrupt
        if required and returncode:
            raise RuntimeError(f"{name} failed (exit {returncode}); see {log}")
        return record


def build_plan(suites, revisions, repeat, warmups, incremental=True):
    """Counterbalance whole serial blocks, preserving each update sequence."""
    blocks, rows = [], []
    for phase in ("verify", "measure"):
        for repetition in range(1 if phase == "verify" else repeat):
            for suite in suites:
                arms = [("before_full", "before", "full"), ("after_full", "after", "full")]
                if suite["experiment"] == "field" and incremental:
                    arms.append(("after_incremental", "after", "incremental"))
                # Rotate the first arm; reverse on alternating repetitions.
                offset = repetition % len(arms)
                arms = arms[offset:] + arms[:offset]
                if (repetition // len(arms)) % 2:
                    arms = arms[::-1]
                for arm, revision, backend in arms:
                    block_id = f"block-{len(blocks):05d}"
                    block = {"block_id": block_id, "arm": arm, "revision": revisions[revision],
                             "revision_label": revision, "backend": backend,
                             "experiment": suite["experiment"], "suite": suite["suite"],
                             "input_directory": suite["directory"], "phase": phase,
                             "repetition": repetition, "out": f"runs/{block_id}",
                             "warmups": warmups if phase == "measure" else 0}
                    blocks.append(block)
                    phases = [("warmup", i) for i in range(block["warmups"])] + [(phase, None)]
                    for row_phase, warmup_index in phases:
                        for index, snapshot in enumerate(suite["snapshots"]):
                            pair = [suite["suite"], snapshot["case_id"], snapshot["snapshot_id"],
                                    row_phase, repetition, warmup_index]
                            rows.append({key: block[key] for key in
                                ("block_id", "arm", "revision", "experiment", "suite", "repetition")}
                                | {"phase": row_phase, "warmup_index": warmup_index,
                                   "snapshot_index": index, "case_id": snapshot["case_id"],
                                   "snapshot_id": snapshot["snapshot_id"],
                                   "map_sha256": snapshot["map_sha256"],
                                   "input_sha256": snapshot["input_sha256"],
                                   "profile_sha256": suite["profile_sha256"],
                                   "pair_id": hashlib.sha256(json.dumps(pair).encode()).hexdigest()})
    return blocks, rows


def save_plan(root, blocks, rows):
    write_json(root / "blocks.json", blocks)
    write_json(root / "planned_runs.json", rows)
    with (root / "planned_runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["block_id"])
        writer.writeheader()
        writer.writerows(rows)


def archive_collection(root):
    """Keep large reproducible worktrees, environments and caches out of data."""
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and "work" not in path.relative_to(root).parts
                   and path.name not in ("SHA256SUMS", "raw-data.zip", "raw-data.zip.tmp"))
    sums = "".join(hashlib.sha256(path.read_bytes()).hexdigest() + "  "
                   + str(path.relative_to(root)) + "\n" for path in files)
    (root / "SHA256SUMS").write_text(sums)
    temporary = root / "raw-data.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files + [root / "SHA256SUMS"]:
            archive.write(path, path.relative_to(root))
    temporary.replace(root / "raw-data.zip")


def setup_revision(root, source, revision, label, python, wheels, commands, args):
    checkout = root / "work/checkouts" / label
    venv = root / "work/envs" / label
    env = clean_environment(root, args.threads)
    commands.run(f"{label}-checkout", ["git", "worktree", "add", "--detach", checkout, revision],
                 cwd=source, env=env, timeout=args.setup_timeout, required=True)
    for module in MODULES:
        command = ["git", "-C", checkout, "submodule", "update", "--init"]
        reference = source / module
        if reference.exists() and (reference / ".git").exists():
            command += ["--reference", reference]
        commands.run(f"{label}-submodule-{Path(module).name}", command + ["--", module],
                     cwd=source, env=env, timeout=args.setup_timeout, required=True)
    commands.run(f"{label}-venv", [python, "-m", "venv", venv], cwd=root,
                 env=env, timeout=args.setup_timeout, required=True)
    executable = venv / "bin/python"
    env = clean_environment(root, args.threads, executable)
    commands.run(f"{label}-packages", [executable, "-m", "pip", "install", "--no-index",
                 "--find-links", wheels, "-r", root / "protocol/requirements.txt"],
                 cwd=root, env=env, timeout=args.setup_timeout, required=True)
    native_wheels = root / "work/native-wheels" / label
    native_wheels.mkdir(parents=True)
    commands.run(f"{label}-build-native", [executable, "-m", "pip", "wheel", "--no-deps",
                 "--no-build-isolation", "--no-cache-dir", "--wheel-dir", native_wheels,
                 checkout / "depend/scikit-fmm"], cwd=root, env=env,
                 timeout=args.setup_timeout, required=True)
    built = list(native_wheels.glob("*.whl"))
    if len(built) != 1:
        raise RuntimeError(f"expected one freshly built native wheel for {label}")
    commands.run(f"{label}-install-native", [executable, "-m", "pip", "install", "--no-index",
                 "--no-deps", "--force-reinstall", built[0]], cwd=root, env=env,
                 timeout=args.setup_timeout, required=True)
    return checkout, executable


def environment_probe(checkout, python, root, label, commands, args):
    target = root / "environments" / f"{label}.json"
    target.parent.mkdir(exist_ok=True)
    code = """import hashlib, importlib.metadata, json, os, pathlib, platform, sys
import numpy, scipy, skfmm
from skfmm import cfmm
binary=pathlib.Path(cfmm.__file__).resolve()
assert binary.is_relative_to(pathlib.Path(sys.prefix).resolve()), str(binary)
packages={d.metadata['Name'].lower().replace('_','-'): d.version for d in importlib.metadata.distributions()}
value={'python':sys.version,'executable':sys.executable,'prefix':sys.prefix,'platform':platform.platform(),
'cpu_count':os.cpu_count(),'packages':packages,'skfmm_path':skfmm.__file__,'native_path':str(binary),
'native_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
pathlib.Path(sys.argv[1]).write_text(json.dumps(value,indent=2)+'\\n')
"""
    commands.run(f"{label}-environment", [python, "-c", code, target], cwd=root,
                 env=clean_environment(root, args.threads, python), timeout=args.timeout, required=True)
    data = json.loads(target.read_text())
    data.update(git_head=git(checkout, "rev-parse", "HEAD"),
                git_status=git(checkout, "status", "--porcelain", "--untracked-files=all"),
                submodules=git(checkout, "submodule", "status").splitlines(),
                native_revision=git(checkout / MODULES[0], "rev-parse", "HEAD"),
                thread_environment={name: str(args.threads) for name in THREADS})
    if data["git_status"]:
        raise RuntimeError(f"{label} checkout is not clean after build: {data['git_status']}")
    write_json(target, data)
    return data


def run_tests(root, harness, workspaces, commands, args):
    records = []
    (root / "work/test-tmp").mkdir(parents=True, exist_ok=True)
    for label, (checkout, python) in workspaces.items():
        env = clean_environment(root, args.threads, python, checkout)
        commands_to_run = [("common-checks", [python, harness / "checks.py", "--repo", checkout,
                                                "--out", root / "tests" / label / "common"])]
        native_code = ("import json,skfmm,sys; r=skfmm.test(); "
                       "failed=int(r) if isinstance(r,int) else r.failed; "
                       "print(json.dumps({'failed':failed,'attempted':getattr(r,'attempted',None)})); "
                       "sys.exit(bool(failed))")
        commands_to_run.append(("native-doctests", [python, "-c", native_code]))
        for name, relative in (("unit", "src/agx_navigation/agx_planning/test/unit"),
                               ("tools", "tools/tests")):
            if not (checkout / relative).exists():
                records.append({"revision": label, "test": name, "status": "unsupported",
                                "reason": "test directory is absent in this revision"})
                continue
            junit = root / "tests" / label / f"{name}.xml"
            commands_to_run.append((name, [python, "-m", "pytest", "-q", relative,
                "--junitxml", junit, "--basetemp", root / "work/test-tmp" / f"{label}-{name}"]))
        for name, command in commands_to_run:
            record = commands.run(f"{label}-{name}", command, cwd=checkout, env=env, timeout=args.timeout)
            records.append({"revision": label, "test": name, **record})
            write_json(root / "tests/index.json", records)
    return records


def collect(args):
    if os.name != "posix":
        raise ValueError("use Linux or WSL with a C++ compiler and Python development headers")
    source = Path(git(Path(__file__).resolve().parents[1], "rev-parse", "--show-toplevel"))
    revisions = {"before": resolve_revision(source, args.before), "after": resolve_revision(source, args.after)}
    if revisions["before"] == revisions["after"]:
        raise ValueError("before and after must identify different commits")
    root = Path(args.out).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "work/tmp").mkdir(parents=True)
    (root / "protocol").mkdir()
    commands = Commands(root)
    harness = root / "harness/revision_compare"
    shutil.copytree(Path(__file__).with_name("revision_compare"), harness,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copyfile(__file__, root / "harness/compare_revisions.py")
    requirement_text = git(source, "show", revisions["after"] + ":requirements-baseline.txt") + "\n"
    (root / "protocol/requirements.txt").write_text(requirement_text)
    protocol = {"schema_version": 1, "created_utc": utc(), "revisions": revisions,
                "options": vars(args), "mode": "smoke" if args.smoke else "collection",
                "scope": "guarded_kernel_computation_without_ROS",
                "precision": "float64_raw_field_packed_float32_planner",
                "domain": "complete", "geometry": "revision_native_planner_interpretation",
                "incremental": {"checkpoint_interval": 10000, "checkpoint_growth": 1,
                                "max_checkpoints": 4, "cache_mib": 256},
                "success_rules": "preserve solver status; evaluate finite pins and final pose/heading/speed separately",
                "build_environment": {name: os.environ.get(name) for name in
                                      ("CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS")},
                "stage": "setup"}
    write_json(root / "protocol.json", protocol)
    blocks = []
    workspaces = {}
    exitcode = 2
    try:
        python = str(Path(args.python).expanduser().absolute()) if os.path.sep in args.python else shutil.which(args.python)
        if not python:
            raise ValueError("Python executable was not found")
        wheels = root / "work/wheels"
        wheels.mkdir()
        download = [python, "-m", "pip", "download", "--only-binary=:all:", "--dest", wheels,
                    "-r", root / "protocol/requirements.txt"]
        if args.wheelhouse:
            download += ["--find-links", str(Path(args.wheelhouse).resolve())]
        if args.offline:
            download += ["--no-index"]
        commands.run("download-common-packages", download, cwd=root,
                     env=clean_environment(root, args.threads), timeout=args.setup_timeout, required=True)
        write_json(root / "protocol/wheels.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(wheels.glob("*.whl"))})
        environments = {}
        for label, revision in revisions.items():
            workspaces[label] = setup_revision(root, source, revision, label, python, wheels, commands, args)
            checkout, executable = workspaces[label]
            environments[label] = environment_probe(checkout, executable, root, label, commands, args)
        dependencies = [{k: v for k, v in data["packages"].items() if k != "scikit-fmm"}
                        for data in environments.values()]
        if dependencies[0] != dependencies[1]:
            raise RuntimeError("the two environments have different numerical or supporting package versions")
        after, after_python = workspaces["after"]
        prepare = [after_python, harness / "prepare.py", "--repo", after, "--out", root / "inputs"]
        field_configs = args.field_config or (["config/replanning/synthetic.yaml"] if args.smoke else
            ["config/replanning/floor6_field_matrix.yaml", "config/replanning/interior_field_updates.yaml",
             "config/replanning/unread_field_updates.yaml"])
        planner_configs = args.planner_config or (["config/replanning/synthetic.yaml"] if args.smoke else
            ["config/replanning/synthetic.yaml", "config/replanning/floor6_smoke.yaml"])
        for option, paths in (("--field-config", field_configs), ("--planner-config", planner_configs)):
            for path in paths:
                prepare += [option, path]
        commands.run("freeze-common-inputs", prepare, cwd=root,
                     env=clean_environment(root, args.threads, after_python), timeout=args.timeout, required=True)
        suites = json.loads((root / "inputs/index.json").read_text())
        blocks, rows = build_plan(suites, revisions, args.repeat, args.warmups, not args.no_incremental)
        save_plan(root, blocks, rows)
        protocol.update(stage="planned", expected_rows=len(rows), expected_blocks=len(blocks))
        write_json(root / "protocol.json", protocol)
        tests = [] if args.skip_tests else run_tests(root, harness, workspaces, commands, args)
        for block in blocks:
            checkout, executable = workspaces[block["revision_label"]]
            command = [executable, harness / "worker.py", "--repo", checkout,
                       "--inputs", root / "inputs" / block["input_directory"],
                       "--experiment", block["experiment"], "--backend", block["backend"],
                       "--out", root / block["out"], "--repetition", str(block["repetition"]),
                       "--phase", block["phase"], "--warmups", str(block["warmups"])]
            record = commands.run(block["block_id"] + "-" + block["arm"], command,
                                  cwd=checkout, env=clean_environment(root, args.threads, executable), timeout=args.timeout)
            block.update({key: record[key] for key in ("returncode", "timed_out", "log", "wall_s")})
            write_json(root / "blocks.json", blocks)
        # New end-to-end behavior is evidence for after only, never a fabricated
        # baseline implementation. It is excluded from the paired timing table.
        features = []
        for label, (checkout, executable) in workspaces.items():
            if not (checkout / "tools/replan_bench.py").exists():
                features.append({"revision": label, "feature": "R1_execution", "status": "unsupported",
                                 "reason": "revision has no reveal/brake/replan runner"})
                continue
            feature = commands.run(label + "-R1-smoke", [executable, checkout / "tools/replan_bench.py",
                "--config", checkout / "config/replanning/execution_smoke.yaml", "--out", root / "features" / (label + "-R1"),
                "--threads", str(args.threads)], cwd=checkout,
                env=clean_environment(root, args.threads, executable), timeout=args.timeout)
            features.append({"revision": label, "feature": "R1_execution", **feature})
        write_json(root / "features/index.json", features)
        audit = commands.run("audit-collection", [after_python, harness / "audit.py", "--root", root],
                             cwd=root, env=clean_environment(root, args.threads, after_python), timeout=args.timeout)
        clean = {label: git(checkout, "status", "--porcelain", "--untracked-files=all")
                 for label, (checkout, _) in workspaces.items()}
        write_json(root / "environments/final-source-status.json", clean)
        complete = all(block.get("returncode") == 0 and (root / block["out"] / "COMPLETE").exists()
                       for block in blocks) and not any(clean.values()) and audit["returncode"] == 0
        test_failures = sum(record.get("returncode", 0) != 0 for record in tests + features)
        protocol.update(stage="complete" if complete else "incomplete", failed_test_commands=test_failures,
                        completed_utc=utc(), tests_skipped=args.skip_tests)
        exitcode = (1 if test_failures else 0) if complete else 2
        if complete:
            (root / "COMPLETE").write_text("All planned attempts are recorded. Inspect outcomes and test failures.\n")
    except KeyboardInterrupt:
        protocol.update(stage="interrupted", completed_utc=utc())
        exitcode = 130
    except Exception as error:
        protocol.update(stage="setup_or_collection_error", error=f"{type(error).__name__}: {error}", completed_utc=utc())
        print(protocol["error"], file=sys.stderr)
    finally:
        if protocol["stage"] not in ("complete", "incomplete") and blocks and "after" in workspaces:
            # Preserve the planned denominator after interruption or a launch
            # error too. Never repeat a numerical attempt behind the user's back.
            try:
                _, executable = workspaces["after"]
                commands.run("audit-partial-collection", [executable, harness / "audit.py", "--root", root],
                             cwd=root, env=clean_environment(root, args.threads, executable),
                             timeout=min(args.timeout, 60))
            except (OSError, KeyboardInterrupt) as error:
                protocol["partial_audit_error"] = type(error).__name__
        write_json(root / "protocol.json", protocol)
        archive_collection(root)
    failures = protocol.get("failed_test_commands", 0)
    detail = f"; {failures} test commands failed (see tests/index.json and features/index.json)" if failures else ""
    print(f"Collection {protocol['stage']}{detail}; raw data: {root / 'raw-data.zip'}", flush=True)
    return exitcode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default=BASELINE)
    parser.add_argument("--after", default="HEAD")
    parser.add_argument("--out", required=True, help="new collection directory")
    parser.add_argument("--python", default=sys.executable, help="Python used to create both isolated environments")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1, help="untimed sequence passes in each measurement process")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900, help="seconds per test or data block")
    parser.add_argument("--setup-timeout", type=float, default=1800)
    parser.add_argument("--smoke", action="store_true", help="small synthetic inputs; still uses requested repetition counts")
    parser.add_argument("--skip-tests", action="store_true", help="explicitly omit correctness/regression tests")
    parser.add_argument("--no-incremental", action="store_true")
    parser.add_argument("--field-config", action="append", help="committed config path in after revision; repeatable")
    parser.add_argument("--planner-config", action="append", help="committed config path in after revision; repeatable")
    parser.add_argument("--wheelhouse", help="optional directory of dependency wheels")
    parser.add_argument("--offline", action="store_true", help="require dependencies from --wheelhouse")
    args = parser.parse_args()
    if (min(args.repeat, args.threads) < 1 or args.warmups < 0
            or not all(math.isfinite(value) and value > 0 for value in (args.timeout, args.setup_timeout))):
        parser.error("repeat, threads and timeouts must be positive; warmups must be nonnegative")
    try:
        return collect(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
